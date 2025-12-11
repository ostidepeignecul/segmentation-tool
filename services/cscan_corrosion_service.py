"""C-Scan corrosion projection and analysis service."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from config.constants import MASK_COLORS_BGRA
from models.annotation_model import AnnotationModel
from models.nde_model import NdeModel
from services.ascan_service import export_ascan_values_to_json
from services.cscan_service import CScanService
from services.distance_measurement import DistanceMeasurementService


@dataclass
class CorrosionAnalysisResult:
    """Structure de retour du service corrosion."""

    distance_results: Dict
    distance_map: np.ndarray
    distance_value_range: Tuple[float, float]
    interpolated_distance_map: np.ndarray
    interpolated_value_range: Tuple[float, float]
    overlay_volume: np.ndarray
    overlay_label_ids: Tuple[int, int]
    overlay_palette: Dict[int, Tuple[int, int, int, int]]
    overlay_npz_path: Optional[str]


class CScanCorrosionService(CScanService):
    """Compute corrosion projections and orchestrate corrosion analysis."""

    _CLASS_TO_PLOT_VALUE = {
        1: 5,
        2: 6,
        3: 7,
        4: 8,
    }

    def __init__(self) -> None:
        super().__init__()
        self._distance_service = DistanceMeasurementService()
        self._logger = logging.getLogger(__name__)

    # --- Projection -----------------------------------------------------------------
    def compute_corrosion_projection(
        self,
        distance_map: np.ndarray,
        *,
        value_range: Optional[Tuple[float, float]] = None,
    ) -> Tuple[np.ndarray, Tuple[float, float]]:
        if distance_map is None:
            raise ValueError("Distance map is required for corrosion projection.")

        data = np.asarray(distance_map, dtype=np.float32)
        if data.ndim != 2:
            raise ValueError(f"Expected a 2D distance map (Z, X), got shape {data.shape}.")

        projection = np.array(data, dtype=np.float32, copy=True)
        finite_values = projection[np.isfinite(projection)]
        if value_range is None:
            if finite_values.size == 0:
                value_range = (0.0, 0.0)
            else:
                value_range = (float(finite_values.min()), float(finite_values.max()))

        projection = np.nan_to_num(projection, nan=0.0, posinf=0.0, neginf=0.0)
        return projection, value_range

    # --- Analyse corrosion end-to-end ----------------------------------------------
    def run_analysis(
        self,
        *,
        global_masks: List[np.ndarray],
        volume_data: np.ndarray,
        nde_data: Dict,
        nde_filename: str,
        orientation: str,
        transpose: bool,
        rotation_applied: bool,
        resolution_crosswise_mm: float,
        resolution_ultrasound_mm: float,
        output_directory: str,
        class_A_id: int,
        class_B_id: int,
        use_mm: bool = False,
    ) -> CorrosionAnalysisResult:
        if not global_masks:
            raise ValueError("Aucun masque global disponible pour l'analyse corrosion")

        self._logger.info("=== ANALYSE CORROSION : démarrage ===")

        label_filter = {int(class_A_id), int(class_B_id)}
        ascan_data = export_ascan_values_to_json(
            global_masks_array=global_masks,
            volume_data=volume_data,
            orientation=orientation,
            transpose=transpose,
            rotation_applied=rotation_applied,
            nde_filename=nde_filename,
            allowed_labels=label_filter,
        )

        distance_results = self._distance_service.measure_distance_all_endviews(
            ascan_data=ascan_data,
            class_A=class_A_id,
            class_B=class_B_id,
            resolution_crosswise=resolution_crosswise_mm,
            resolution_ultrasound=resolution_ultrasound_mm,
        )

        if distance_results.get("status") == "error":
            raise RuntimeError(distance_results.get("error", "Erreur lors du calcul des distances"))

        height, width = global_masks[0].shape
        distance_map = self._distance_service.build_distance_map(
            distance_results=distance_results,
            num_endviews=len(global_masks),
            width=width,
            use_mm=use_mm,
        )

        valid_values = distance_map[np.isfinite(distance_map)]
        if valid_values.size > 0:
            value_range = (float(valid_values.min()), float(valid_values.max()))
        else:
            value_range = (0.0, 0.0)

        lines_overlay = self._build_front_back_overlay(
            distance_results=distance_results,
            num_endviews=len(global_masks),
            image_shape=global_masks[0].shape,
            class_A_id=class_A_id,
            class_B_id=class_B_id,
        )

        color_A = self._CLASS_TO_PLOT_VALUE.get(class_A_id, 5)
        color_B = self._CLASS_TO_PLOT_VALUE.get(class_B_id, 6)

        interpolated_distance_map = self._build_interpolated_distance_map(
            overlay=lines_overlay,
            class_A_value=color_A,
            class_B_value=color_B,
            use_mm=use_mm,
            resolution_ultrasound_mm=resolution_ultrasound_mm,
        )

        interpolated_valid = interpolated_distance_map[np.isfinite(interpolated_distance_map)]
        interpolated_value_range = (
            (float(interpolated_valid.min()), float(interpolated_valid.max()))
            if interpolated_valid.size > 0
            else (0.0, 0.0)
        )

        overlay_palette = {
            color_A: tuple(int(c) for c in MASK_COLORS_BGRA.get(color_A, (255, 0, 255, 160))),
            color_B: tuple(int(c) for c in MASK_COLORS_BGRA.get(color_B, (255, 0, 255, 160))),
        }

        self._logger.info("=== ANALYSE CORROSION : terminée ===")

        return CorrosionAnalysisResult(
            distance_results=distance_results,
            distance_map=distance_map,
            distance_value_range=value_range,
            interpolated_distance_map=interpolated_distance_map,
            interpolated_value_range=interpolated_value_range,
            overlay_volume=lines_overlay,
            overlay_label_ids=(color_A, color_B),
            overlay_palette=overlay_palette,
            overlay_npz_path=None,
        )

    # --- Helpers --------------------------------------------------------------------
    def _build_front_back_overlay(
        self,
        *,
        distance_results: Dict,
        num_endviews: int,
        image_shape: Tuple[int, int],
        class_A_id: int,
        class_B_id: int,
        line_thickness: int = 2,
    ) -> np.ndarray:
        height, width = image_shape
        lines_volume = np.zeros((num_endviews, height, width), dtype=np.uint8)
        color_A = self._CLASS_TO_PLOT_VALUE.get(class_A_id, 5)
        color_B = self._CLASS_TO_PLOT_VALUE.get(class_B_id, 6)

        endviews_data = distance_results.get("endviews", {})
        for endview_id_str, endview_result in endviews_data.items():
            try:
                endview_id = int(endview_id_str)
            except (TypeError, ValueError):
                continue

            if endview_id < 0 or endview_id >= num_endviews:
                continue

            if endview_result.get("status") != "success":
                continue

            measurements = [
                m for m in endview_result.get("measurements", []) if m.get("status") == "success"
            ]
            if not measurements:
                continue

            mask = np.zeros((height, width), dtype=np.uint8)
            points_A: List[Tuple[float, float]] = []
            points_B: List[Tuple[float, float]] = []

            for measurement in measurements:
                point_A = measurement.get("point_A", {})
                point_B = measurement.get("point_B", {})
                points_A.append((point_A.get("x", 0), point_A.get("y", 0)))
                points_B.append((point_B.get("x", 0), point_B.get("y", 0)))

            points_A_sorted = sorted(points_A, key=lambda p: p[0])
            points_B_sorted = sorted(points_B, key=lambda p: p[0])

            for i in range(len(points_A_sorted) - 1):
                pt1 = (int(round(points_A_sorted[i][0])), int(round(points_A_sorted[i][1])))
                pt2 = (int(round(points_A_sorted[i + 1][0])), int(round(points_A_sorted[i + 1][1])))
                cv2.line(mask, pt1, pt2, color=color_A, thickness=line_thickness)

            for i in range(len(points_B_sorted) - 1):
                pt1 = (int(round(points_B_sorted[i][0])), int(round(points_B_sorted[i][1])))
                pt2 = (int(round(points_B_sorted[i + 1][0])), int(round(points_B_sorted[i + 1][1])))
                cv2.line(mask, pt1, pt2, color=color_B, thickness=line_thickness)

            lines_volume[endview_id] = mask

        if lines_volume.size == 0:
            return lines_volume

        return lines_volume

    def _save_overlay(self, output_directory: str, nde_filename: str, overlay: np.ndarray) -> Optional[str]:
        if overlay.size == 0:
            return None

        os.makedirs(output_directory, exist_ok=True)
        base_name = os.path.splitext(os.path.basename(nde_filename))[0]
        npz_path = os.path.join(output_directory, f"{base_name}_corrosion_overlay.npz")
        np.savez_compressed(npz_path, arr_0=overlay)
        self._logger.info("Overlay corrosion sauvegardé: %s", npz_path)
        return npz_path

    def _build_interpolated_distance_map(
        self,
        *,
        overlay: np.ndarray,
        class_A_value: int,
        class_B_value: int,
        use_mm: bool,
        resolution_ultrasound_mm: float,
    ) -> np.ndarray:
        """
        Recalcule une carte de distances (Z, X) à partir des lignes interpolées BW/FW.

        Pour chaque slice (Z) et position X, on recherche la moyenne des positions Y
        pour les valeurs class_A_value et class_B_value dans l'overlay, puis on
        calcule la distance verticale. Les positions manquantes sont laissées à NaN.
        """
        if overlay.size == 0:
            return np.empty((0, 0), dtype=np.float32)

        volume = np.asarray(overlay)
        if volume.ndim != 3:
            raise ValueError(f"Overlay attendu 3D (Z,H,W), reçu shape {volume.shape}")

        num_slices, height, width = volume.shape
        interpolated = np.full((num_slices, width), np.nan, dtype=np.float32)

        for z in range(num_slices):
            slice_mask = volume[z]
            for x in range(width):
                ys_A = np.where(slice_mask[:, x] == class_A_value)[0]
                ys_B = np.where(slice_mask[:, x] == class_B_value)[0]
                if ys_A.size == 0 or ys_B.size == 0:
                    continue
                yA = float(ys_A.mean())
                yB = float(ys_B.mean())
                dist = abs(yA - yB)
                if use_mm:
                    dist *= float(resolution_ultrasound_mm)
                interpolated[z, x] = float(dist)

        return interpolated


@dataclass
class CorrosionWorkflowResult:
    """Résultat du workflow d'analyse corrosion."""

    ok: bool
    message: str = ""
    projection: Optional[np.ndarray] = None
    value_range: Optional[Tuple[float, float]] = None
    raw_distance_map: Optional[np.ndarray] = None
    interpolated_distance_map: Optional[np.ndarray] = None
    interpolated_projection: Optional[np.ndarray] = None
    interpolated_value_range: Optional[Tuple[float, float]] = None
    overlay_volume: Optional[np.ndarray] = None
    overlay_label_ids: Optional[Tuple[int, int]] = None
    overlay_palette: Optional[Dict[int, Tuple[int, int, int, int]]] = None


class CorrosionWorkflowService:
    """Orchestre le workflow complet d'analyse corrosion."""

    def __init__(
        self,
        cscan_corrosion_service: CScanCorrosionService,
    ) -> None:
        self.cscan_corrosion_service = cscan_corrosion_service

    def run(
        self,
        nde_model: NdeModel,
        annotation_model: AnnotationModel,
        volume: np.ndarray,
    ) -> CorrosionWorkflowResult:
        """
        - Valide la présence du volume NDE et des masques
        - Vérifie le nombre de labels visibles (exactement 2: frontwall/backwall)
        - Prépare les masques globaux et les résolutions (mm_per_pixel, sample_spacing, etc.)
        - Déduit un chemin d'output si nécessaire (à partir des metadata/filename)
        - Lance CScanCorrosionService.run_analysis() et compute_corrosion_projection()
        - Retourne une projection corrosion + plage de valeurs
        """
        # Validation volume et mask_volume
        mask_volume = annotation_model.mask_volume
        if volume is None or mask_volume is None:
            return CorrosionWorkflowResult(
                ok=False,
                message="Corrosion analysis aborted: volume or masks missing.",
            )

        # Validation des shapes
        if mask_volume.shape[0] != volume.shape[0]:
            return CorrosionWorkflowResult(
                ok=False,
                message=f"Corrosion analysis aborted: mask depth {mask_volume.shape[0]} != volume depth {volume.shape[0]}",
            )

        # Extraction des labels visibles (uniquement frontwall/backwall)
        visible_labels = [
            lbl for lbl, vis in (annotation_model.label_visibility or {}).items() if vis and int(lbl) > 0
        ]
        if len(visible_labels) != 2:
            return CorrosionWorkflowResult(
                ok=False,
                message=f"Corrosion analysis requires exactly 2 visible labels; found {len(visible_labels)}.",
            )

        class_A_id, class_B_id = sorted(int(x) for x in visible_labels[:2])

        # Construction du masque global par label
        global_masks = [mask_volume[idx] for idx in range(mask_volume.shape[0])]

        # Récupération des résolutions dans NdeModel.metadata
        resolution_cross, resolution_ultra = self._extract_resolutions(nde_model)

        # Choix du output_directory
        nde_filename = "unknown"
        if nde_model is not None:
            nde_filename = str(nde_model.metadata.get("path", "unknown"))
        output_directory = os.path.dirname(nde_filename) if nde_filename not in ("", "unknown") else "."

        try:
            # Appel à CScanCorrosionService.run_analysis()
            result = self.cscan_corrosion_service.run_analysis(
                global_masks=global_masks,
                volume_data=volume,
                nde_data=nde_model.metadata if nde_model else {},
                nde_filename=nde_filename,
                orientation="lengthwise",
                transpose=False,
                rotation_applied=False,
                resolution_crosswise_mm=resolution_cross,
                resolution_ultrasound_mm=resolution_ultra,
                output_directory=output_directory,
                class_A_id=class_A_id,
                class_B_id=class_B_id,
                use_mm=False,
            )

            # Appel à compute_corrosion_projection()
            projection, value_range = self.cscan_corrosion_service.compute_corrosion_projection(
                result.distance_map,
                value_range=result.distance_value_range,
            )

            interpolated_projection: Optional[np.ndarray] = None
            interpolated_value_range: Optional[Tuple[float, float]] = None

            if result.interpolated_distance_map.size > 0:
                interpolated_projection, interpolated_value_range = self.cscan_corrosion_service.compute_corrosion_projection(
                    result.interpolated_distance_map,
                    value_range=result.interpolated_value_range,
                )

            return CorrosionWorkflowResult(
                ok=True,
                message="Analyse corrosion terminée",
                projection=projection,
                value_range=value_range,
                raw_distance_map=result.distance_map,
                interpolated_distance_map=result.interpolated_distance_map,
                interpolated_projection=interpolated_projection,
                interpolated_value_range=interpolated_value_range,
                overlay_volume=result.overlay_volume,
                overlay_label_ids=result.overlay_label_ids,
                overlay_palette=result.overlay_palette,
            )

        except Exception as exc:
            return CorrosionWorkflowResult(
                ok=False,
                message=f"Corrosion analysis failed: {exc}",
            )

    def _extract_resolutions(self, nde_model: Optional[NdeModel]) -> Tuple[float, float]:
        """Get crosswise and ultrasound resolutions from metadata, defaults to 1.0."""
        if nde_model is None:
            return 1.0, 1.0
        dimensions = nde_model.metadata.get("dimensions", [])
        try:
            cross = float(dimensions[1].get("resolution", 1.0)) if len(dimensions) >= 3 else 1.0
            ultra = float(dimensions[2].get("resolution", 1.0)) if len(dimensions) >= 3 else 1.0
        except Exception:
            cross, ultra = 1.0, 1.0
        return cross, ultra
