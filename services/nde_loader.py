"""Lightweight NDE loader focused on in-memory workflows.

This module retains only the logic required by the current application:
    - reading `.nde` files (public or domain flavor)
    - orienting/normalizing the resulting volume for models and views
    - optionally generating endview frames straight in memory (no disk I/O)

All export-to-disk helpers or advanced metadata utilities from the previous
iteration have been removed to keep the service compact and memory-focused.
"""

from __future__ import annotations

import json
import logging
from typing import Dict, List, Optional, Tuple

import cv2
import h5py
import numpy as np

from models.nde_model import NDEModel
from services.nde_debug_logger import nde_debug_logger


class NdeLoaderService:
    """Loads NDE volumes and prepares in-memory slices for the UI."""

    def __init__(self) -> None:
        self.logger = logging.getLogger(__name__)
        self._current_nde_data: Optional[Dict] = None
        self._current_nde_path: Optional[str] = None
        self._cached_images: Optional[List[np.ndarray]] = None
        self._cached_image_names: Optional[List[str]] = None
        self._cached_raw_slices: Optional[List[np.ndarray]] = None

    # ------------------------------------------------------------------
    # Core loading pipeline
    # ------------------------------------------------------------------

    def load_nde_model(self, nde_file: str, group_idx: int = 1) -> NDEModel:
        """Return an `NDEModel` populated with an oriented + normalized volume."""
        nde_data = self.load_nde_data(nde_file, group_idx)
        data_array = nde_data["data_array"]
        structure = nde_data.get("structure", "public")

        orientation_cfg = self.detect_optimal_orientation(data_array, structure)
        oriented_volume = self.orient_volume(data_array, orientation_cfg)
        normalized_volume = self.normalize_volume(
            oriented_volume,
            nde_data.get("min_value"),
            nde_data.get("max_value"),
        )

        metadata = {
            "structure": structure,
            "orientation": orientation_cfg,
            "path": nde_file,
            "group_idx": group_idx,
            "dimensions": nde_data.get("dimensions"),
            "positions": nde_data.get("positions"),
            "group_info": nde_data.get("group_info"),
            "min_value": nde_data.get("min_value"),
            "max_value": nde_data.get("max_value"),
            "normalization": {
                "method": "minmax",
                "source_range": (
                    nde_data.get("min_value"),
                    nde_data.get("max_value"),
                ),
                "clipped": True,
            },
        }

        model = NDEModel()
        model.set_volume(oriented_volume, metadata, normalized_volume=normalized_volume)
        model.set_current_slice(0)

        return model

    def load_nde_as_memory_images(
        self, nde_file: str, group_idx: int = 1
    ) -> Tuple[List[np.ndarray], List[str]]:
        """Generate grayscale endviews directly in memory (BGR uint8)."""
        self._cached_images = []
        self._cached_image_names = []
        self._cached_raw_slices = []

        nde_data = self.load_nde_data(nde_file, group_idx)
        data_array = nde_data["data_array"]
        structure = nde_data.get("structure", "public")

        orientation_cfg = self.detect_optimal_orientation(data_array, structure)
        orientation = orientation_cfg["slice_orientation"]
        transpose = orientation_cfg["transpose"]
        num_images = orientation_cfg["num_images"]

        self.logger.info("Génération de %s endviews en mémoire...", num_images)
        nde_debug_logger.log_section("GENERATION EN MEMOIRE")
        nde_debug_logger.log_variable("num_images", num_images)
        nde_debug_logger.log_variable("orientation", orientation)
        nde_debug_logger.log_variable("transpose", transpose)

        for idx in range(num_images):
            img_data = self.extract_slice(data_array, idx, orientation)
            if transpose:
                img_data = img_data.T

            if nde_data["max_value"] == nde_data["min_value"]:
                img_data_normalized = np.zeros_like(img_data, dtype=float)
            else:
                img_data_normalized = (img_data - nde_data["min_value"]) / (
                    nde_data["max_value"] - nde_data["min_value"]
                )
                img_data_normalized = np.clip(img_data_normalized, 0.0, 1.0)

            img_uint8 = (img_data_normalized * 255).astype(np.uint8)
            if not transpose:
                img_uint8 = np.rot90(img_uint8, k=-1)
                img_data_normalized = np.rot90(img_data_normalized, k=-1)

            img_bgr = cv2.cvtColor(np.ascontiguousarray(img_uint8), cv2.COLOR_GRAY2BGR)

            self._cached_raw_slices.append(
                np.ascontiguousarray(img_data_normalized.astype(np.float32))
            )
            self._cached_images.append(img_bgr)
            self._cached_image_names.append(f"endview_{idx * 1500:012d}.png")

            if idx % 100 == 0 and idx > 0:
                self.logger.info("Généré %s/%s images en mémoire...", idx, num_images)

        self.logger.info(
            "Génération terminée: %s images en mémoire", len(self._cached_images)
        )
        return self._cached_images, self._cached_image_names

    def load_nde_data(self, nde_file: str, group_idx: int = 1) -> Dict:
        """Read raw NDE data (Domain or Public structure)."""
        try:
            nde_debug_logger.log_section("LOAD_NDE_DATA")
            nde_debug_logger.log_variable("nde_file", nde_file)
            nde_debug_logger.log_variable("group_idx", group_idx)

            structure = self.detect_nde_structure(nde_file)
            nde_debug_logger.log_variable("structure", structure, indent=1)
            self.logger.info("Structure détectée: %s", structure)

            with h5py.File(nde_file, "r") as handle:
                if structure == "domain":
                    result = self._load_domain_structure(handle, group_idx)
                else:
                    result = self._load_public_structure(handle, group_idx)

            nde_debug_logger.log("Données chargées:", indent=1)
            nde_debug_logger.log_variable("data_array", result["data_array"], indent=2)
            nde_debug_logger.log_variable("structure", result.get("structure"), indent=2)

            self._current_nde_data = result
            self._current_nde_path = nde_file
            return result

        except Exception as exc:  # pragma: no cover - logging convenience
            self.logger.error("Erreur lors du chargement NDE: %s", exc)
            nde_debug_logger.log(f"ERREUR: {exc}", indent=1)
            raise

    # ------------------------------------------------------------------
    # File structure helpers
    # ------------------------------------------------------------------

    def detect_nde_structure(self, nde_file: str) -> str:
        """Return 'domain' or 'public' based on the file's root keys."""
        with h5py.File(nde_file, "r") as handle:
            root_keys = set(handle.keys())
        if "Domain" in root_keys:
            return "domain"
        if "Public" in root_keys:
            return "public"
        raise ValueError(f"Structure NDE non reconnue: {root_keys}")

    def _load_domain_structure(self, handle: h5py.File, group_idx: int) -> Dict:
        json_str = handle["Domain/Setup"][()]
        json_decoded = json.loads(json_str)

        data_path = f"Domain/DataGroups/{group_idx-1}/Datasets/0/Amplitude"
        if data_path not in handle:
            raise ValueError(f"Chemin de données non trouvé: {data_path}")

        data_array = handle[data_path][:]
        data_array = self._correct_negative_values(data_array)
        group_info = json_decoded["groups"][group_idx - 1]

        min_value = np.min(data_array)
        max_value = np.max(data_array)

        dimensions = self._extract_domain_dimensions(group_info, data_array)
        positions = self._build_positions(dimensions)

        return {
            "data_array": data_array,
            "min_value": min_value,
            "max_value": max_value,
            "dimensions": dimensions,
            "positions": positions,
            "group_info": group_info,
            "structure": "domain",
        }

    def _load_public_structure(self, handle: h5py.File, group_idx: int) -> Dict:
        json_str = handle["Public/Setup"][()]
        json_decoded = json.loads(json_str)

        datasets_path = f"Public/Groups/{group_idx-1}/Datasets"
        if datasets_path not in handle:
            raise ValueError(f"Chemin de données non trouvé: {datasets_path}")

        data_array = None
        for key in handle[datasets_path].keys():
            if "AScanAmplitude" in key:
                data_array = handle[f"{datasets_path}/{key}"][:]
                break
        if data_array is None:
            raise ValueError("Aucun dataset AScanAmplitude trouvé")

        group_info = json_decoded["groups"][group_idx - 1]
        dimensions = group_info["datasets"][0]["dimensions"]
        positions = self._build_positions(dimensions)

        return {
            "data_array": data_array,
            "min_value": np.min(data_array),
            "max_value": np.max(data_array),
            "dimensions": dimensions,
            "positions": positions,
            "group_info": group_info,
            "structure": "public",
        }

    def _extract_domain_dimensions(
        self, group_info: Dict, data_array: np.ndarray
    ) -> List[Dict]:
        try:
            return group_info["data"]["ascan"]["dataset"]["amplitude"]["dimensions"]
        except KeyError:
            pass
        try:
            return group_info["dataset"]["amplitude"]["dimensions"]
        except KeyError:
            pass
        try:
            return group_info["ascan"]["dataset"]["amplitude"]["dimensions"]
        except KeyError:
            self.logger.warning(
                "Structure Domain non standard. Clés disponibles: %s",
                list(group_info.keys()),
            )
            return [
                {"offset": 0, "resolution": 1, "quantity": data_array.shape[axis]}
                for axis in range(3)
            ]

    def _build_positions(self, dimensions: List[Dict]) -> Dict[str, np.ndarray]:
        axis_names = ["lengthwise", "crosswise", "ultrasound"]
        positions: Dict[str, np.ndarray] = {}
        for axis_name, info in zip(axis_names, dimensions):
            offset = info.get("offset", 0)
            resolution = info.get("resolution", 1)
            quantity = info.get("quantity", 0)
            positions[axis_name] = np.array(
                [offset + i * resolution for i in range(quantity)]
            )
        return positions

    @staticmethod
    def _correct_negative_values(data: np.ndarray) -> np.ndarray:
        min_val = np.min(data)
        if min_val >= 0:
            return data
        corrected = np.clip(data, 0, None)
        return corrected

    # ------------------------------------------------------------------
    # Orientation helpers
    # ------------------------------------------------------------------

    def detect_optimal_orientation(
        self, data_array: np.ndarray, structure: str = "public"
    ) -> Dict:
        nde_debug_logger.log_section("DETECT_OPTIMAL_ORIENTATION")
        nde_debug_logger.log_variable("structure", structure)

        lengthwise_qty, crosswise_qty, ultrasound_qty = data_array.shape
        nde_debug_logger.log(
            f"Dimensions: lengthwise={lengthwise_qty}, crosswise={crosswise_qty}, ultrasound={ultrasound_qty}",
            indent=1,
        )

        if structure == "public":
            sample = data_array[0, :, :]
            aspect = sample.shape[1] / sample.shape[0] if sample.shape[0] else 1.0
            transpose = aspect < 1.0
            cfg = {
                "slice_orientation": "lengthwise",
                "transpose": transpose,
                "num_images": lengthwise_qty,
                "shape": sample.shape,
                "aspect": aspect,
            }
            nde_debug_logger.log_variable("orientation_config", cfg, indent=1)
            return cfg

        orientations = []
        slices = {
            "lengthwise": (data_array[0, :, :], lengthwise_qty),
            "crosswise": (data_array[:, 0, :], crosswise_qty),
            "ultrasound": (data_array[:, :, 0], ultrasound_qty),
        }
        for name, (sample, qty) in slices.items():
            aspect = sample.shape[1] / sample.shape[0] if sample.shape[0] else 1.0
            orientations.append({
                "name": name,
                "shape": sample.shape,
                "aspect": aspect,
                "num_images": qty,
            })

        best_orientation = max(
            orientations,
            key=lambda o: (
                (20 if o["num_images"] >= 1000 else 15 if o["num_images"] >= 500 else 10 if o["num_images"] >= 100 else 5)
                + (2 if 0.1 <= o["aspect"] <= 50.0 else 1 if 0.05 <= o["aspect"] <= 100.0 else 0)
            ),
        )

        transpose = best_orientation["aspect"] < 1.0
        cfg = {
            "slice_orientation": best_orientation["name"],
            "transpose": transpose,
            "num_images": best_orientation["num_images"],
            "shape": best_orientation["shape"],
            "aspect": best_orientation["aspect"],
        }
        nde_debug_logger.log_variable("orientation_config", cfg, indent=1)
        return cfg

    @staticmethod
    def orient_volume(data_array: np.ndarray, orientation_cfg: Dict) -> np.ndarray:
        orientation = orientation_cfg.get("slice_orientation", "lengthwise")
        if orientation == "crosswise":
            oriented = np.moveaxis(data_array, 1, 0)
        elif orientation == "ultrasound":
            oriented = np.moveaxis(data_array, 2, 0)
        else:
            oriented = data_array

        if orientation_cfg.get("transpose"):
            oriented = oriented.transpose(0, 2, 1)
        return oriented

    @staticmethod
    def normalize_volume(
        volume: np.ndarray, min_value: Optional[float], max_value: Optional[float]
    ) -> np.ndarray:
        if min_value is None or max_value is None:
            return volume.astype(np.float32, copy=False)
        if max_value == min_value:
            return np.zeros_like(volume, dtype=np.float32)
        normalized = (volume - min_value) / (max_value - min_value)
        normalized = np.clip(normalized, 0.0, 1.0)
        return normalized.astype(np.float32, copy=False)

    @staticmethod
    def extract_slice(data_array: np.ndarray, idx: int, orientation: str) -> np.ndarray:
        if orientation == "crosswise":
            return data_array[:, idx, :]
        if orientation == "ultrasound":
            return data_array[:, :, idx]
        return data_array[idx, :, :]

    # ------------------------------------------------------------------
    # Cache accessors
    # ------------------------------------------------------------------

    def get_cached_image(self, index: int) -> Optional[np.ndarray]:
        if self._cached_images is None:
            return None
        if index < 0 or index >= len(self._cached_images):
            return None
        return self._cached_images[index]

    def get_cached_image_name(self, index: int) -> Optional[str]:
        if self._cached_image_names is None:
            return None
        if index < 0 or index >= len(self._cached_image_names):
            return None
        return self._cached_image_names[index]

    def get_cached_raw_slice(self, index: int) -> Optional[np.ndarray]:
        if self._cached_raw_slices is None:
            return None
        if index < 0 or index >= len(self._cached_raw_slices):
            return None
        return self._cached_raw_slices[index].copy()

    def clear_memory_cache(self) -> None:
        self._cached_images = None
        self._cached_image_names = None
        self._cached_raw_slices = None
        self.logger.info("Cache mémoire libéré")
