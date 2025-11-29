"""C-Scan corrosion projection and analysis service."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from services.ascan_service import export_ascan_values_to_json
from services.cscan_service import CScanService
from services.distance_measurement import DistanceMeasurementService


@dataclass
class CorrosionAnalysisResult:
    """Structure de retour du service corrosion."""

    distance_results: Dict
    distance_map: np.ndarray
    distance_value_range: Tuple[float, float]
    overlay_volume: np.ndarray
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

        overlay_npz_path = self._save_overlay(output_directory, nde_filename, lines_overlay)

        self._logger.info("=== ANALYSE CORROSION : terminée ===")

        return CorrosionAnalysisResult(
            distance_results=distance_results,
            distance_map=distance_map,
            distance_value_range=value_range,
            overlay_volume=lines_overlay,
            overlay_npz_path=overlay_npz_path,
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

        volume_flipped = np.array([np.fliplr(slice_img) for slice_img in lines_volume])
        volume_transposed = volume_flipped.transpose((0, 2, 1))
        return volume_transposed

    def _save_overlay(self, output_directory: str, nde_filename: str, overlay: np.ndarray) -> Optional[str]:
        if overlay.size == 0:
            return None

        os.makedirs(output_directory, exist_ok=True)
        base_name = os.path.splitext(os.path.basename(nde_filename))[0]
        npz_path = os.path.join(output_directory, f"{base_name}_corrosion_overlay.npz")
        np.savez_compressed(npz_path, arr_0=overlay)
        self._logger.info("Overlay corrosion sauvegardé: %s", npz_path)
        return npz_path
