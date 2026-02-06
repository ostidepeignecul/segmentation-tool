"""C-Scan corrosion projection and analysis service."""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from config.constants import MASK_COLORS_BGRA
from models.annotation_model import AnnotationModel
from models.nde_model import NdeModel
from services.cscan_service import CScanService
from services.distance_measurement import DistanceMeasurementService


@dataclass
class CorrosionAnalysisResult:
    """Structure de retour du service corrosion."""

    distance_results: Dict
    distance_map: np.ndarray
    peak_index_map_a: np.ndarray
    peak_index_map_b: np.ndarray
    distance_value_range: Tuple[float, float]
    interpolated_distance_map: np.ndarray
    interpolated_value_range: Tuple[float, float]
    overlay_volume: np.ndarray
    overlay_label_ids: Tuple[int, int]
    overlay_palette: Dict[int, Tuple[int, int, int, int]]
    overlay_npz_path: Optional[str]
    piece_volume_raw: Optional[np.ndarray] = None
    piece_volume_interpolated: Optional[np.ndarray] = None
    piece_anchor: Optional[Tuple[float, float, float]] = None


class CScanCorrosionService(CScanService):
    """Compute corrosion projections and orchestrate corrosion analysis."""

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
        label_palette: Optional[Dict[int, Tuple[int, int, int, int]]] = None,
        use_mm: bool = False,
    ) -> CorrosionAnalysisResult:
        if not global_masks:
            raise ValueError("Aucun masque global disponible pour l'analyse corrosion")

        self._logger.info("=== ANALYSE CORROSION : démarrage ===")
        self._log_progress(0.0, "Démarrage")
        t0 = time.perf_counter()

        # Calcul vectorisé de la carte de distances (Z,X) directement depuis volume + masques
        mask_stack = np.stack(global_masks, axis=0)
        distance_map, peak_index_map_a, peak_index_map_b = self._distance_service.measure_distance_and_peaks_vectorized(
            volume=volume_data,
            masks=mask_stack,
            class_A=class_A_id,
            class_B=class_B_id,
            use_mm=use_mm,
            resolution_ultrasound=resolution_ultrasound_mm,
        )
        self._logger.info("[Corrosion] Carte distance calculée en %.2f s", time.perf_counter() - t0)
        self._log_progress(0.5, "Carte distance")
        t_overlay = time.perf_counter()
        distance_results: Dict = {}

        color_A = int(class_A_id)
        color_B = int(class_B_id)
        lines_overlay = self._build_overlay_from_peak_maps(
            peak_map_a=peak_index_map_a,
            peak_map_b=peak_index_map_b,
            image_shape=mask_stack.shape[1:],
            class_A_id=class_A_id,
            class_B_id=class_B_id,
            line_thickness=2,
        )
        self._logger.info("[Corrosion] Overlay lignes construit en %.2f s", time.perf_counter() - t_overlay)
        self._log_progress(0.75, "Overlay lignes")
        t_interp = time.perf_counter()

        valid_values = distance_map[np.isfinite(distance_map)]
        if valid_values.size > 0:
            value_range = (float(valid_values.min()), float(valid_values.max()))
        else:
            value_range = (0.0, 0.0)

        interpolated_distance_map = self._build_interpolated_distance_map(
            overlay=lines_overlay,
            class_A_value=color_A,
            class_B_value=color_B,
            use_mm=use_mm,
            resolution_ultrasound_mm=resolution_ultrasound_mm,
        )
        self._logger.info("[Corrosion] Carte interpolée calculée en %.2f s", time.perf_counter() - t_interp)
        self._log_progress(0.9, "Interpolation")

        interpolated_valid = interpolated_distance_map[np.isfinite(interpolated_distance_map)]
        interpolated_value_range = (
            (float(interpolated_valid.min()), float(interpolated_valid.max()))
            if interpolated_valid.size > 0
            else (0.0, 0.0)
        )

        palette_source = label_palette or {}
        color_a_bgra = palette_source.get(color_A)
        if color_a_bgra is None:
            color_a_bgra = MASK_COLORS_BGRA.get(color_A, (255, 0, 255, 160))
        color_b_bgra = palette_source.get(color_B)
        if color_b_bgra is None:
            color_b_bgra = MASK_COLORS_BGRA.get(color_B, (255, 0, 255, 160))
        overlay_palette = {
            color_A: tuple(int(c) for c in color_a_bgra),
            color_B: tuple(int(c) for c in color_b_bgra),
        }

        piece_volume_raw = self._build_solid_volume(
            mask_stack=mask_stack,
            class_A_id=class_A_id,
            class_B_id=class_B_id,
        )
        piece_volume_interpolated = self._build_solid_volume(
            mask_stack=lines_overlay,
            class_A_id=color_A,
            class_B_id=color_B,
        )
        piece_anchor = self._compute_piece_anchor(piece_volume_raw, piece_volume_interpolated)

        self._logger.info("=== ANALYSE CORROSION : terminée ===")
        self._log_progress(1.0, "Terminé")

        return CorrosionAnalysisResult(
            distance_results={},
            distance_map=distance_map,
            peak_index_map_a=peak_index_map_a,
            peak_index_map_b=peak_index_map_b,
            distance_value_range=value_range,
            interpolated_distance_map=interpolated_distance_map,
            interpolated_value_range=interpolated_value_range,
            overlay_volume=lines_overlay,
            overlay_label_ids=(color_A, color_B),
            overlay_palette=overlay_palette,
            overlay_npz_path=None,
            piece_volume_raw=piece_volume_raw,
            piece_volume_interpolated=piece_volume_interpolated,
            piece_anchor=piece_anchor,
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
        color_A = int(class_A_id)
        color_B = int(class_B_id)

        endviews_data = distance_results.get("endviews", {}) if distance_results else {}
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

        Pour chaque slice (Z), calcule en une passe la moyenne des positions Y par X
        pour les valeurs class_A_value et class_B_value via des bincounts, puis
        calcule la distance verticale. Les positions manquantes sont laissées à NaN.
        """
        if overlay.size == 0:
            return np.empty((0, 0), dtype=np.float32)

        volume = np.asarray(overlay)
        if volume.ndim != 3:
            raise ValueError(f"Overlay attendu 3D (Z,H,W), reçu shape {volume.shape}")

        num_slices, height, width = volume.shape
        interpolated = np.full((num_slices, width), np.nan, dtype=np.float32)

        scale = float(resolution_ultrasound_mm) if use_mm else 1.0

        for z in range(num_slices):
            slice_mask = volume[z]
            yA, xA = np.nonzero(slice_mask == class_A_value)
            yB, xB = np.nonzero(slice_mask == class_B_value)
            if yA.size == 0 or yB.size == 0:
                continue

            sumA = np.bincount(xA, weights=yA, minlength=width)
            cntA = np.bincount(xA, minlength=width)
            sumB = np.bincount(xB, weights=yB, minlength=width)
            cntB = np.bincount(xB, minlength=width)

            valid = (cntA > 0) & (cntB > 0)
            if not np.any(valid):
                continue

            meanA = np.zeros(width, dtype=np.float32)
            meanB = np.zeros(width, dtype=np.float32)
            meanA[valid] = sumA[valid] / cntA[valid]
            meanB[valid] = sumB[valid] / cntB[valid]

            dist = np.abs(meanA - meanB) * scale
            interpolated[z, valid] = dist[valid]

        return interpolated

    def _build_overlay_from_peak_maps(
        self,
        *,
        peak_map_a: np.ndarray,
        peak_map_b: np.ndarray,
        image_shape: Tuple[int, int],
        class_A_id: int,
        class_B_id: int,
        line_thickness: int = 1,
    ) -> np.ndarray:
        """Construit un overlay de lignes BW/FW Ã  partir des indices de pics (Y par X)."""
        if peak_map_a.ndim != 2 or peak_map_b.ndim != 2:
            raise ValueError(
                f"Peak maps attendus 2D (Z,W), reÃ§us {peak_map_a.shape} et {peak_map_b.shape}"
            )

        height, width = image_shape
        num_slices = min(peak_map_a.shape[0], peak_map_b.shape[0])
        lines_volume = np.zeros((num_slices, height, width), dtype=np.uint8)
        color_A = int(class_A_id)
        color_B = int(class_B_id)

        width_map = min(width, peak_map_a.shape[1], peak_map_b.shape[1])
        if width_map <= 0 or num_slices <= 0:
            return lines_volume

        for z in range(num_slices):
            slice_a = peak_map_a[z]
            slice_b = peak_map_b[z]

            valid_a = np.where((slice_a[:width_map] >= 0) & (slice_a[:width_map] < height))[0]
            pts_a = [(int(x), int(slice_a[x])) for x in valid_a]
            if len(pts_a) >= 2:
                for i in range(len(pts_a) - 1):
                    cv2.line(lines_volume[z], pts_a[i], pts_a[i + 1], color=color_A, thickness=line_thickness)
            elif len(pts_a) == 1:
                lines_volume[z, pts_a[0][1], pts_a[0][0]] = color_A

            valid_b = np.where((slice_b[:width_map] >= 0) & (slice_b[:width_map] < height))[0]
            pts_b = [(int(x), int(slice_b[x])) for x in valid_b]
            if len(pts_b) >= 2:
                for i in range(len(pts_b) - 1):
                    cv2.line(lines_volume[z], pts_b[i], pts_b[i + 1], color=color_B, thickness=line_thickness)
            elif len(pts_b) == 1:
                lines_volume[z, pts_b[0][1], pts_b[0][0]] = color_B

        return lines_volume

    def _build_overlay_from_masks(
        self,
        *,
        mask_stack: np.ndarray,
        class_A_id: int,
        class_B_id: int,
        line_thickness: int = 1,
    ) -> np.ndarray:
        """Construit un overlay de lignes BW/FW (moyenne Y par X, ligne fine)."""
        if mask_stack.ndim != 3:
            raise ValueError(f"Overlay attendu 3D (Z,H,W), reçu shape {mask_stack.shape}")

        num_slices, height, width = mask_stack.shape
        lines_volume = np.zeros((num_slices, height, width), dtype=np.uint8)
        color_A = int(class_A_id)
        color_B = int(class_B_id)

        for z in range(num_slices):
            slice_mask = mask_stack[z]
            yA, xA = np.nonzero(slice_mask == class_A_id)
            yB, xB = np.nonzero(slice_mask == class_B_id)
            if yA.size == 0 or yB.size == 0:
                continue

            # Moyenne Y par X pour chaque classe
            sumA = np.bincount(xA, weights=yA, minlength=width)
            cntA = np.bincount(xA, minlength=width)
            sumB = np.bincount(xB, weights=yB, minlength=width)
            cntB = np.bincount(xB, minlength=width)
            validA = cntA > 0
            validB = cntB > 0

            ysA = np.zeros(width, dtype=np.float32)
            ysB = np.zeros(width, dtype=np.float32)
            ysA[validA] = sumA[validA] / cntA[validA]
            ysB[validB] = sumB[validB] / cntB[validB]

            x_coords_A = np.nonzero(validA)[0].tolist()
            x_coords_B = np.nonzero(validB)[0].tolist()
            pts_A = [(int(x), int(round(ysA[x]))) for x in x_coords_A]
            pts_B = [(int(x), int(round(ysB[x]))) for x in x_coords_B]

            if len(pts_A) >= 2:
                for i in range(len(pts_A) - 1):
                    cv2.line(lines_volume[z], pts_A[i], pts_A[i + 1], color=color_A, thickness=line_thickness)
            elif len(pts_A) == 1:
                lines_volume[z, pts_A[0][1], pts_A[0][0]] = color_A

            if len(pts_B) >= 2:
                for i in range(len(pts_B) - 1):
                    cv2.line(lines_volume[z], pts_B[i], pts_B[i + 1], color=color_B, thickness=line_thickness)
            elif len(pts_B) == 1:
                lines_volume[z, pts_B[0][1], pts_B[0][0]] = color_B

        return lines_volume

    def _log_progress(self, fraction: float, label: str) -> None:
        """Affiche une barre de progression ASCII dans les logs."""
        frac = max(0.0, min(1.0, float(fraction)))
        filled = int(frac * 20)
        bar = "#" * filled + "-" * (20 - filled)
        self._logger.info("[Corrosion][%3d%%] [%s] %s", int(frac * 100), bar, label)

    def _build_solid_volume(
        self,
        *,
        mask_stack: np.ndarray,
        class_A_id: int,
        class_B_id: int,
    ) -> np.ndarray:
        """
        Construit un volume solide en remplissant l'espace entre frontwall et backwall.

        Pour chaque slice Z et chaque colonne X où A et B sont présents, on remplit
        verticalement entre les extrêmes des deux classes. Le résultat est un volume
        float32 de même shape que ``mask_stack`` (Z, H, W) avec des valeurs 0/1.
        """
        if mask_stack.ndim != 3:
            raise ValueError(f"Volume attendu 3D (Z,H,W), reçu shape {mask_stack.shape}")

        depth, height, width = mask_stack.shape
        solid = np.zeros((depth, height, width), dtype=np.float32)

        def _min_max_by_x(y_coords: np.ndarray, x_coords: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
            if y_coords.size == 0:
                return np.array([], dtype=np.int32), np.array([], dtype=np.int32), np.array([], dtype=np.int32)
            order = np.argsort(x_coords)
            xs = x_coords[order]
            ys = y_coords[order]
            unique_x, start_idx, counts = np.unique(xs, return_index=True, return_counts=True)
            end_idx = start_idx + counts - 1
            min_y = ys[start_idx]
            max_y = ys[end_idx]
            return unique_x.astype(np.int32), min_y.astype(np.int32), max_y.astype(np.int32)

        for z in range(depth):
            slice_mask = mask_stack[z]
            yA, xA = np.nonzero(slice_mask == class_A_id)
            yB, xB = np.nonzero(slice_mask == class_B_id)
            if yA.size == 0 or yB.size == 0:
                continue

            xA_unique, minA, maxA = _min_max_by_x(yA, xA)
            xB_unique, minB, maxB = _min_max_by_x(yB, xB)
            if xA_unique.size == 0 or xB_unique.size == 0:
                continue

            common_x = np.intersect1d(xA_unique, xB_unique, assume_unique=True)
            if common_x.size == 0:
                continue

            minA_full = np.full(width, np.inf, dtype=np.float32)
            maxA_full = np.full(width, -np.inf, dtype=np.float32)
            minB_full = np.full(width, np.inf, dtype=np.float32)
            maxB_full = np.full(width, -np.inf, dtype=np.float32)
            minA_full[xA_unique] = minA
            maxA_full[xA_unique] = maxA
            minB_full[xB_unique] = minB
            maxB_full[xB_unique] = maxB

            for x in common_x:
                start_y = min(minA_full[x], minB_full[x])
                end_y = max(maxA_full[x], maxB_full[x])
                if not np.isfinite(start_y) or not np.isfinite(end_y):
                    continue
                y0 = int(max(0, min(height - 1, start_y)))
                y1 = int(max(0, min(height - 1, end_y)))
                if y1 < y0:
                    y0, y1 = y1, y0
                solid[z, y0 : y1 + 1, int(x)] = 1.0

        return solid

    @staticmethod
    def _compute_center_of_mass(volume: Optional[np.ndarray]) -> Optional[Tuple[float, float, float]]:
        """Return (x, y, z) center of mass for a binary volume (values > 0)."""
        if volume is None or volume.size == 0:
            return None
        mask = np.asarray(volume) > 0.0
        if not np.any(mask):
            return None
        z_idx, y_idx, x_idx = np.nonzero(mask)
        if z_idx.size == 0:
            return None
        x_mean = float(np.mean(x_idx, dtype=np.float64))
        y_mean = float(np.mean(y_idx, dtype=np.float64))
        z_mean = float(np.mean(z_idx, dtype=np.float64))
        return (x_mean, y_mean, z_mean)

    def _compute_piece_anchor(
        self,
        primary: Optional[np.ndarray],
        fallback: Optional[np.ndarray],
    ) -> Optional[Tuple[float, float, float]]:
        """Compute anchor from the primary solid volume, fallback to interpolated."""
        anchor = self._compute_center_of_mass(primary)
        if anchor is None:
            anchor = self._compute_center_of_mass(fallback)
        return anchor


@dataclass
class CorrosionWorkflowResult:
    """Résultat du workflow d'analyse corrosion."""

    ok: bool
    message: str = ""
    projection: Optional[np.ndarray] = None
    value_range: Optional[Tuple[float, float]] = None
    raw_distance_map: Optional[np.ndarray] = None
    peak_index_map_a: Optional[np.ndarray] = None
    peak_index_map_b: Optional[np.ndarray] = None
    interpolated_distance_map: Optional[np.ndarray] = None
    interpolated_projection: Optional[np.ndarray] = None
    interpolated_value_range: Optional[Tuple[float, float]] = None
    overlay_volume: Optional[np.ndarray] = None
    overlay_label_ids: Optional[Tuple[int, int]] = None
    overlay_palette: Optional[Dict[int, Tuple[int, int, int, int]]] = None
    piece_volume_raw: Optional[np.ndarray] = None
    piece_volume_interpolated: Optional[np.ndarray] = None
    piece_anchor: Optional[Tuple[float, float, float]] = None


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
        *,
        label_a: Optional[int] = None,
        label_b: Optional[int] = None,
    ) -> CorrosionWorkflowResult:
        """
        - Valide la présence du volume NDE et des masques
        - Vérifie les labels sélectionnés (2 labels distincts)
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

        # Validation des labels sélectionnés
        try:
            class_A_id = int(label_a) if label_a is not None else None
        except Exception:
            class_A_id = None
        try:
            class_B_id = int(label_b) if label_b is not None else None
        except Exception:
            class_B_id = None
        if class_A_id is None or class_B_id is None:
            return CorrosionWorkflowResult(
                ok=False,
                message="Veuillez choisir deux labels pour l'analyse corrosion.",
            )
        if class_A_id <= 0 or class_B_id <= 0:
            return CorrosionWorkflowResult(
                ok=False,
                message="Les labels de corrosion doivent être > 0.",
            )
        if class_A_id == class_B_id:
            return CorrosionWorkflowResult(
                ok=False,
                message="Veuillez choisir deux labels distincts pour l'analyse corrosion.",
            )
        available_labels = {
            int(lbl) for lbl in (annotation_model.label_palette or {}).keys() if int(lbl) > 0
        }
        if not available_labels or class_A_id not in available_labels or class_B_id not in available_labels:
            return CorrosionWorkflowResult(
                ok=False,
                message="Labels sélectionnés absents de la palette.",
            )
        if not np.any(mask_volume == class_A_id):
            return CorrosionWorkflowResult(
                ok=False,
                message=f"Aucun voxel pour le label {class_A_id} (corrosion).",
            )
        if not np.any(mask_volume == class_B_id):
            return CorrosionWorkflowResult(
                ok=False,
                message=f"Aucun voxel pour le label {class_B_id} (corrosion).",
            )

        # Construction du masque global par label
        global_masks = [mask_volume[idx] for idx in range(mask_volume.shape[0])]

        # Récupération des résolutions dans NdeModel.metadata
        resolution_cross, resolution_ultra = self._extract_resolutions(nde_model)

        # Choix du output_directory
        nde_filename = "unknown"
        if nde_model is not None:
            nde_filename = str(nde_model.metadata.get("path", "unknown"))
        output_directory = os.path.dirname(nde_filename) if nde_filename not in ("", "unknown") else "."
        palette_source = annotation_model.get_label_palette()

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
                label_palette=palette_source,
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
                peak_index_map_a=result.peak_index_map_a,
                peak_index_map_b=result.peak_index_map_b,
                interpolated_distance_map=result.interpolated_distance_map,
                interpolated_projection=interpolated_projection,
                interpolated_value_range=interpolated_value_range,
                overlay_volume=result.overlay_volume,
                overlay_label_ids=result.overlay_label_ids,
                overlay_palette=result.overlay_palette,
                piece_volume_raw=result.piece_volume_raw,
                piece_volume_interpolated=result.piece_volume_interpolated,
                piece_anchor=result.piece_anchor,
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
