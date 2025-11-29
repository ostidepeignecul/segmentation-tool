#!/usr/bin/env python3
"""Service haut-niveau pour lancer l'analyse corrosion en une seule étape."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from services.ascan_extractor import export_ascan_values_to_json
from services.distance_measurement import DistanceMeasurementService


@dataclass
class CorrosionAnalysisResult:
    """Structure de retour du service corrosion."""

    distance_results: Dict
    overlay_volume: np.ndarray
    overlay_npz_path: Optional[str]


class CorrosionAnalysisService:
    """Orchestre extraction A-scan, calcul de distances et overlay corrosion."""

    _CLASS_TO_PLOT_VALUE = {
        'frontwall': 5,
        'backwall': 6,
        'flaw': 7,
        'indication': 8,
    }

    def __init__(self) -> None:
        self._logger = logging.getLogger(__name__)
        self._distance_service = DistanceMeasurementService()

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
        class_A: str = 'frontwall',
        class_B: str = 'backwall'
    ) -> CorrosionAnalysisResult:
        """Lance l'analyse corrosion complète et retourne les artefacts générés."""

        if not global_masks:
            raise ValueError("Aucun masque global disponible pour l'analyse corrosion")

        self._logger.info("=== ANALYSE CORROSION : démarrage ===")

        # Étape 1 : extraction A-scan + distances
        ascan_data = export_ascan_values_to_json(
            global_masks_array=global_masks,
            volume_data=volume_data,
            orientation=orientation,
            transpose=transpose,
            rotation_applied=rotation_applied,
            nde_filename=nde_filename,
        )

        distance_results = self._distance_service.measure_distance_all_endviews(
            ascan_data=ascan_data,
            class_A=class_A,
            class_B=class_B,
            resolution_crosswise=resolution_crosswise_mm,
            resolution_ultrasound=resolution_ultrasound_mm,
        )

        if distance_results.get('status') == 'error':
            raise RuntimeError(distance_results.get('error', 'Erreur lors du calcul des distances'))

        # Étape 2 : overlay corrosion (lignes front/back uniquement)
        lines_overlay = self._build_front_back_overlay(
            distance_results=distance_results,
            num_endviews=len(global_masks),
            image_shape=global_masks[0].shape,
            class_A=class_A,
            class_B=class_B,
        )

        overlay_npz_path = self._save_overlay(output_directory, nde_filename, lines_overlay)

        self._logger.info("=== ANALYSE CORROSION : terminée ===")

        return CorrosionAnalysisResult(
            distance_results=distance_results,
            overlay_volume=lines_overlay,
            overlay_npz_path=overlay_npz_path,
        )

    def _build_front_back_overlay(
        self,
        *,
        distance_results: Dict,
        num_endviews: int,
        image_shape: Tuple[int, int],
        class_A: str,
        class_B: str,
        line_thickness: int = 2
    ) -> np.ndarray:
        """Construit un volume avec uniquement les lignes interpolées front/back.

        Args:
            line_thickness: Contrôle l'épaisseur des lignes tracées pour une meilleure lisibilité.
        """

        height, width = image_shape
        lines_volume = np.zeros((num_endviews, height, width), dtype=np.uint8)
        color_A = self._CLASS_TO_PLOT_VALUE.get(class_A, 5)
        color_B = self._CLASS_TO_PLOT_VALUE.get(class_B, 6)

        endviews_data = distance_results.get('endviews', {})
        for endview_id_str, endview_result in endviews_data.items():
            try:
                endview_id = int(endview_id_str)
            except (TypeError, ValueError):
                continue

            if endview_id < 0 or endview_id >= num_endviews:
                continue

            if endview_result.get('status') != 'success':
                continue

            measurements = [
                m for m in endview_result.get('measurements', []) if m.get('status') == 'success'
            ]
            if not measurements:
                continue

            mask = np.zeros((height, width), dtype=np.uint8)
            points_A: List[Tuple[float, float]] = []
            points_B: List[Tuple[float, float]] = []

            for measurement in measurements:
                point_A = measurement.get('point_A', {})
                point_B = measurement.get('point_B', {})
                points_A.append((point_A.get('x', 0), point_A.get('y', 0)))
                points_B.append((point_B.get('x', 0), point_B.get('y', 0)))

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

        # Appliquer les mêmes transformations que pour l'overlay NPZ classique
        if lines_volume.size == 0:
            return lines_volume

        volume_flipped = np.array([np.fliplr(slice_img) for slice_img in lines_volume])
        volume_transposed = volume_flipped.transpose((0, 2, 1))
        return volume_transposed


    def _save_overlay(self, output_directory: str, nde_filename: str, overlay: np.ndarray) -> Optional[str]:
        """Sauvegarde l'overlay corrosion dans un NPZ dédié."""

        if overlay.size == 0:
            return None

        os.makedirs(output_directory, exist_ok=True)
        base_name = os.path.splitext(os.path.basename(nde_filename))[0]
        npz_path = os.path.join(output_directory, f"{base_name}_corrosion_overlay.npz")
        np.savez_compressed(npz_path, arr_0=overlay)
        self._logger.info("Overlay corrosion sauvegardé: %s", npz_path)
        return npz_path
