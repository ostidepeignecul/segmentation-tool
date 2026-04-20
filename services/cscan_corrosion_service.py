"""C-Scan corrosion projection and analysis service."""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.interpolate import (
    Akima1DInterpolator,
    CloughTocher2DInterpolator,
    LinearNDInterpolator,
    PchipInterpolator,
    RBFInterpolator,
    griddata,
)
from scipy.spatial import QhullError

from config.constants import MASK_COLORS_BGRA, normalize_interpolation_algo
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
    raw_peak_index_map_a: np.ndarray
    raw_peak_index_map_b: np.ndarray
    ascan_support_map: np.ndarray
    distance_value_range: Tuple[float, float]
    interpolated_distance_map: np.ndarray
    interpolated_value_range: Tuple[float, float]
    overlay_volume: np.ndarray
    overlay_label_ids: Tuple[int, int]
    overlay_palette: Dict[int, Tuple[int, int, int, int]]
    overlay_npz_path: Optional[str]
    piece_volume_raw: Optional[np.ndarray] = None
    piece_volume_interpolated: Optional[np.ndarray] = None
    piece_volume_legacy_raw: Optional[np.ndarray] = None
    piece_volume_legacy_interpolated: Optional[np.ndarray] = None
    piece_anchor: Optional[Tuple[float, float, float]] = None


class CScanCorrosionService(CScanService):
    """Compute corrosion projections and orchestrate corrosion analysis."""

    # Gap size in pixels allowed only for zones without A-scan support.
    # `0` keeps support-present holes interpolated, but never bridges absent A-scan zones.
    MAX_INTERPOLATION_GAP_PX = 0
    # Robust upper bound for corrosion heatmaps to avoid one aberrant peak flattening the whole map.
    HEATMAP_UPPER_PERCENTILE = 99.0
    # Keep thin-plate interpolation bounded on dense maps.
    RBF_MAX_POINTS = 2000
    RBF_NEIGHBORS = 128
    # Diagnostic smoothing after nearest-neighbor fill.
    GAUSSIAN_FILL_SIGMA = 1.0

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
        if value_range is None:
            value_range = self.compute_display_value_range(projection)

        return projection, value_range

    @classmethod
    def compute_display_value_range(
        cls,
        distance_map: np.ndarray,
    ) -> Tuple[float, float]:
        """Return a robust display range for corrosion heatmaps without mutating the data."""
        data = np.asarray(distance_map, dtype=np.float32)
        finite_values = data[np.isfinite(data)]
        if finite_values.size == 0:
            return (0.0, 0.0)

        vmin = float(finite_values.min())
        vmax = float(finite_values.max())
        if finite_values.size < 3 or vmax <= vmin:
            return (vmin, vmax)

        try:
            clipped_vmax = float(np.percentile(finite_values, cls.HEATMAP_UPPER_PERCENTILE))
        except Exception:
            clipped_vmax = vmax

        if not np.isfinite(clipped_vmax):
            return (vmin, vmax)
        if clipped_vmax > vmin:
            return (vmin, min(vmax, clipped_vmax))

        below_raw_max = finite_values[finite_values < vmax]
        if below_raw_max.size > 0:
            fallback_vmax = float(below_raw_max.max())
            if fallback_vmax > vmin:
                return (vmin, fallback_vmax)

        return (vmin, float(np.nextafter(np.float32(vmin), np.float32(np.inf))))

    # --- Analyse corrosion end-to-end ----------------------------------------------
    def run_analysis(
        self,
        *,
        global_masks: List[np.ndarray],
        volume_data: np.ndarray,
        support_volume_data: Optional[np.ndarray],
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
        peak_selection_mode_a: str = "max_peak",
        peak_selection_mode_b: Optional[str] = None,
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
        ascan_support_map = self.build_ascan_support_map(
            support_volume_data if support_volume_data is not None else volume_data
        )
        distance_map, peak_index_map_a, peak_index_map_b = self._distance_service.measure_distance_and_peaks_vectorized(
            volume=volume_data,
            masks=mask_stack,
            class_A=class_A_id,
            class_B=class_B_id,
            peak_selection_mode_a=peak_selection_mode_a,
            peak_selection_mode_b=peak_selection_mode_b,
            support_map=ascan_support_map,
            use_mm=use_mm,
            resolution_ultrasound=resolution_ultrasound_mm,
        )
        self._logger.info("[Corrosion] Carte distance calculee en %.2f s", time.perf_counter() - t0)
        self._log_progress(0.5, "Carte distance")
        t_overlay = time.perf_counter()

        # Conserver les peak maps bruts avant toute opération
        raw_peak_index_map_a = peak_index_map_a.copy()
        raw_peak_index_map_b = peak_index_map_b.copy()

        distance_results: Dict = {}

        color_A = int(class_A_id)
        color_B = int(class_B_id)
        lines_overlay = self.build_overlay_from_peak_maps(
            peak_map_a=peak_index_map_a,
            peak_map_b=peak_index_map_b,
            image_shape=mask_stack.shape[1:],
            class_A_id=class_A_id,
            class_B_id=class_B_id,
            line_thickness=1,
        )
        self._logger.info("[Corrosion] Overlay lignes construit en %.2f s", time.perf_counter() - t_overlay)
        self._log_progress(0.75, "Overlay lignes")

        value_range = self.compute_display_value_range(distance_map)

        # Pas d'interpolation au lancement — les données brutes sont affichées en premier.
        # L'interpolation sera déclenchée manuellement via le bouton Calculer.
        interpolated_distance_map = np.empty((0, 0), dtype=np.float32)
        interpolated_value_range = (0.0, 0.0)

        self._log_progress(0.9, "Palette")

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

        piece_volume_legacy_raw = self._build_solid_volume(
            mask_stack=mask_stack,
            class_A_id=class_A_id,
            class_B_id=class_B_id,
        )
        piece_volume_legacy_interpolated = None
        piece_volume_raw = self._build_prismatic_piece_from_distance_map(distance_map)
        piece_volume_interpolated = None
        piece_anchor = self._compute_piece_anchor(
            piece_volume_interpolated,
            piece_volume_raw,
            piece_volume_legacy_interpolated,
            piece_volume_legacy_raw,
        )

        self._logger.info("=== ANALYSE CORROSION : terminée ===")
        self._log_progress(1.0, "Terminé")

        return CorrosionAnalysisResult(
            distance_results={},
            distance_map=distance_map,
            peak_index_map_a=peak_index_map_a,
            peak_index_map_b=peak_index_map_b,
            raw_peak_index_map_a=raw_peak_index_map_a,
            raw_peak_index_map_b=raw_peak_index_map_b,
            ascan_support_map=ascan_support_map,
            distance_value_range=value_range,
            interpolated_distance_map=interpolated_distance_map,
            interpolated_value_range=interpolated_value_range,
            overlay_volume=lines_overlay,
            overlay_label_ids=(color_A, color_B),
            overlay_palette=overlay_palette,
            overlay_npz_path=None,
            piece_volume_raw=piece_volume_raw,
            piece_volume_interpolated=piece_volume_interpolated,
            piece_volume_legacy_raw=piece_volume_legacy_raw,
            piece_volume_legacy_interpolated=piece_volume_legacy_interpolated,
            piece_anchor=piece_anchor,
        )

    def apply_interpolation(
        self,
        *,
        raw_result: CorrosionAnalysisResult,
        algo: str,
        mask_height: int,
        use_mm: bool = False,
        resolution_ultrasound_mm: float = 1.0,
    ) -> CorrosionAnalysisResult:
        """Apply an interpolation algorithm on the raw peak maps and return a new result.

        ``algo`` can be:
        - ``"1d_dual_axis"`` — existing dual-axis linear interpolation.
        - ``"1d_pchip_dual_axis"`` — shape-preserving cubic interpolation on both axes.
        - ``"1d_makima_dual_axis"`` — MAKIMA interpolation on both axes.
        - ``"2d_linear_nd"`` — 2D simplicial linear interpolation over valid (Z, X) points.
        - ``"2d_clough_tocher"`` — 2D Clough-Tocher interpolation over valid (Z, X) points.
        - ``"2d_rbf_thin_plate"`` — 2D thin-plate spline interpolation on valid (Z, X) points.
        - ``"2d_gaussian_fill"`` — nearest-neighbor fill followed by Gaussian smoothing.
        """
        peak_a = self.interpolate_peak_map_with_algo(
            raw_result.raw_peak_index_map_a,
            algo=algo,
            height=mask_height,
            support_map=raw_result.ascan_support_map,
        )
        peak_b = self.interpolate_peak_map_with_algo(
            raw_result.raw_peak_index_map_b,
            algo=algo,
            height=mask_height,
            support_map=raw_result.ascan_support_map,
        )

        distance_map = self._build_distance_map_from_peak_maps(
            peak_map_a=peak_a,
            peak_map_b=peak_b,
            use_mm=use_mm,
            resolution_ultrasound_mm=resolution_ultrasound_mm,
        )
        value_range = self.compute_display_value_range(distance_map)

        class_A_id, class_B_id = raw_result.overlay_label_ids
        overlay = self.build_overlay_from_peak_maps(
            peak_map_a=peak_a,
            peak_map_b=peak_b,
            image_shape=(mask_height, peak_a.shape[1]) if peak_a.ndim == 2 else (mask_height, 1),
            class_A_id=class_A_id,
            class_B_id=class_B_id,
            line_thickness=1,
        )

        interpolated_distance_map = self.build_interpolated_distance_map(
            overlay=overlay,
            class_A_value=int(class_A_id),
            class_B_value=int(class_B_id),
            use_mm=use_mm,
            resolution_ultrasound_mm=resolution_ultrasound_mm,
        )
        interpolated_value_range = self.compute_display_value_range(interpolated_distance_map)

        piece_volume_raw = self._build_prismatic_piece_from_distance_map(distance_map)
        piece_volume_interpolated = self._build_prismatic_piece_from_distance_map(interpolated_distance_map)
        piece_volume_legacy_interpolated = self._build_solid_volume(
            mask_stack=overlay,
            class_A_id=class_A_id,
            class_B_id=class_B_id,
        )
        piece_anchor = self._compute_piece_anchor(
            piece_volume_interpolated,
            piece_volume_raw,
            piece_volume_legacy_interpolated,
            raw_result.piece_volume_legacy_raw,
        )

        return CorrosionAnalysisResult(
            distance_results={},
            distance_map=distance_map,
            peak_index_map_a=peak_a,
            peak_index_map_b=peak_b,
            raw_peak_index_map_a=raw_result.raw_peak_index_map_a,
            raw_peak_index_map_b=raw_result.raw_peak_index_map_b,
            ascan_support_map=raw_result.ascan_support_map,
            distance_value_range=value_range,
            interpolated_distance_map=interpolated_distance_map,
            interpolated_value_range=interpolated_value_range,
            overlay_volume=overlay,
            overlay_label_ids=raw_result.overlay_label_ids,
            overlay_palette=raw_result.overlay_palette,
            overlay_npz_path=None,
            piece_volume_raw=piece_volume_raw,
            piece_volume_interpolated=piece_volume_interpolated,
            piece_volume_legacy_raw=raw_result.piece_volume_legacy_raw,
            piece_volume_legacy_interpolated=piece_volume_legacy_interpolated,
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

    def save_cscan_projection(
        self,
        *,
        output_directory: str,
        nde_filename: str,
        projection: np.ndarray,
        value_range: Optional[Tuple[float, float]] = None,
    ) -> Tuple[str, str]:
        """Save the displayed corrosion C-scan as NPZ and PNG."""
        if not output_directory:
            raise ValueError("Aucun dossier de sortie fourni pour l'export du C-scan corrosion.")

        data = np.asarray(projection, dtype=np.float32)
        if data.ndim != 2:
            raise ValueError(f"C-scan corrosion attendu 2D (Z,X), recu shape {data.shape}")
        if data.size == 0:
            raise ValueError("Le C-scan corrosion est vide.")

        if value_range is None:
            value_range = self.compute_display_value_range(data)

        os.makedirs(output_directory, exist_ok=True)
        base_name = os.path.splitext(os.path.basename(str(nde_filename)))[0] or "unknown"
        npz_path = os.path.join(output_directory, f"{base_name}_cscan.npz")
        png_path = os.path.join(output_directory, f"{base_name}_cscan.png")
        np.savez_compressed(
            npz_path,
            projection=data,
            value_range=np.asarray(value_range, dtype=np.float32),
        )
        self._logger.info("C-scan corrosion sauvegardÃ©: %s", npz_path)
        rgb = self._render_cscan_projection_rgb(data, value_range)
        saved_png = cv2.imwrite(png_path, cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        if not saved_png:
            raise ValueError(f"Impossible de sauvegarder l'image PNG: {png_path}")
        self._logger.info("C-scan corrosion image sauvegardée: %s", png_path)
        return npz_path, png_path

    @classmethod
    def _render_cscan_projection_rgb(
        cls,
        data: np.ndarray,
        value_range: Tuple[float, float],
    ) -> np.ndarray:
        """Render a corrosion projection to RGB using the corrosion C-scan palette."""
        vmin, vmax = value_range
        valid = np.isfinite(data)
        rgb = np.zeros((*data.shape, 3), dtype=np.uint8)
        if vmax <= vmin:
            return rgb

        normalized = np.zeros_like(data, dtype=np.float32)
        normalized[valid] = (data[valid] - vmin) / (vmax - vmin)
        normalized = np.clip(normalized, 0.0, 1.0)
        indices = np.zeros(data.shape, dtype=np.int32)
        indices[valid] = (normalized[valid] * 255.0).astype(np.int32)
        lut = cls._build_cscan_export_lut()
        rgb[valid] = lut[indices[valid]]
        return rgb

    @staticmethod
    def _build_cscan_export_lut() -> np.ndarray:
        """Build the corrosion LUT used for exported PNGs."""
        stops = [
            (0.0, (255, 0, 0)),
            (0.33, (255, 128, 0)),
            (0.66, (255, 255, 0)),
            (1.0, (0, 128, 255)),
        ]
        lut = np.zeros((256, 3), dtype=np.uint8)
        for idx in range(len(stops) - 1):
            start_pos, start_col = stops[idx]
            end_pos, end_col = stops[idx + 1]
            start_idx = int(round(start_pos * 255))
            end_idx = int(round(end_pos * 255))
            span = max(1, end_idx - start_idx)
            for channel in range(3):
                lut[start_idx:end_idx + 1, channel] = np.linspace(
                    start_col[channel],
                    end_col[channel],
                    span + 1,
                    dtype=np.uint8,
                )
        lut[-1, :] = stops[-1][1]
        return lut

    @staticmethod
    def build_ascan_support_map(volume: np.ndarray) -> np.ndarray:
        """Return a (Z,X) support map where a column has at least one finite non-zero A-scan sample."""
        data = np.asarray(volume, dtype=np.float32)
        if data.ndim != 3:
            raise ValueError(f"Volume support attendu 3D (Z,H,W), recu shape {data.shape}")
        return np.any(np.isfinite(data) & (data != 0.0), axis=1)

    @classmethod
    def build_fillable_support_mask(cls, support_map: Optional[np.ndarray]) -> Optional[np.ndarray]:
        """Allow interpolation where support exists, plus short unsupported X gaps up to MAX_INTERPOLATION_GAP_PX."""
        if support_map is None:
            return None

        support = np.asarray(support_map, dtype=bool)
        if support.ndim != 2:
            raise ValueError(f"Support map attendu 2D (Z,W), recu shape {support.shape}")

        fillable = np.array(support, dtype=bool, copy=True)
        max_gap = cls._get_max_interpolation_gap_px()
        if max_gap <= 0 or fillable.size == 0:
            return fillable

        for z in range(fillable.shape[0]):
            row = fillable[z]
            x = 0
            width = row.shape[0]
            while x < width:
                if row[x]:
                    x += 1
                    continue
                start = x
                while x < width and not row[x]:
                    x += 1
                if (x - start) <= max_gap:
                    row[start:x] = True
        return fillable

    @staticmethod
    def _clip_peak_indices_in_place(data: np.ndarray, *, height: Optional[int]) -> None:
        valid = data >= 0
        if not np.any(valid):
            return

        data[valid] = np.maximum(data[valid], 0)
        if height is not None:
            data[valid] = np.clip(data[valid], 0, int(height) - 1)

    def interpolate_peak_map_with_algo(
        self,
        peak_map: np.ndarray,
        *,
        algo: str,
        height: Optional[int] = None,
        support_map: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        normalized = normalize_interpolation_algo(algo)
        data = np.asarray(peak_map, dtype=np.int32)

        if normalized == "1d_dual_axis":
            return self.interpolate_peak_map_1d_dual_axis(
                data,
                height=height,
                support_map=support_map,
                method="linear",
            )

        if normalized == "1d_pchip_dual_axis":
            return self.interpolate_peak_map_1d_dual_axis(
                data,
                height=height,
                support_map=support_map,
                method="pchip",
            )

        if normalized == "1d_makima_dual_axis":
            return self.interpolate_peak_map_1d_dual_axis(
                data,
                height=height,
                support_map=support_map,
                method="makima",
            )

        if normalized == "2d_linear_nd":
            return self.interpolate_peak_map_2d_nd(
                data,
                height=height,
                fillable_mask=self.build_fillable_support_mask(support_map),
                method="linear_nd",
            )

        if normalized == "2d_clough_tocher":
            return self.interpolate_peak_map_2d_nd(
                data,
                height=height,
                fillable_mask=self.build_fillable_support_mask(support_map),
                method="clough_tocher",
            )

        if normalized == "2d_rbf_thin_plate":
            return self.interpolate_peak_map_2d_rbf(
                data,
                height=height,
                fillable_mask=self.build_fillable_support_mask(support_map),
            )

        if normalized == "2d_gaussian_fill":
            return self.interpolate_peak_map_2d_gaussian_fill(
                data,
                height=height,
                fillable_mask=self.build_fillable_support_mask(support_map),
            )

        raise ValueError(f"Algorithme d'interpolation inconnu : {algo!r}")

    @classmethod
    def _subsample_interpolation_points(
        cls,
        points: np.ndarray,
        values: np.ndarray,
        *,
        max_points: Optional[int] = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        limit = cls.RBF_MAX_POINTS if max_points is None else int(max_points)
        if limit <= 0 or points.shape[0] <= limit:
            return points, values

        indices = np.linspace(0, points.shape[0] - 1, num=limit, dtype=np.int64)
        unique_indices = np.unique(indices)
        return points[unique_indices], values[unique_indices]

    def _prepare_2d_interpolation_data(
        self,
        peak_map: np.ndarray,
        *,
        height: Optional[int] = None,
        fillable_mask: Optional[np.ndarray] = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        data = np.asarray(peak_map, dtype=np.int32)
        if data.ndim != 2:
            raise ValueError(f"Peak map attendu 2D (Z,W), recu shape {data.shape}")

        interpolated = np.array(data, dtype=np.int32, copy=True)
        if interpolated.size == 0:
            return interpolated, np.empty((0, 2), dtype=np.int64), np.empty((0, 2), dtype=np.int64), np.empty((0,), dtype=np.float64), np.empty((0, 2), dtype=np.float64)

        fillable: Optional[np.ndarray] = None
        if fillable_mask is not None:
            fillable = np.asarray(fillable_mask, dtype=bool)
            if fillable.shape != interpolated.shape:
                raise ValueError(
                    f"Fillable mask attendu en shape {interpolated.shape}, recu {fillable.shape}"
                )

        missing = interpolated < 0
        if fillable is not None:
            missing &= fillable
        candidate_coords = np.argwhere(missing)

        valid = interpolated >= 0
        points = np.argwhere(valid)
        values = interpolated[valid].astype(np.float64)
        points_f = points.astype(np.float64)
        self._clip_peak_indices_in_place(interpolated, height=height)
        return interpolated, candidate_coords, points, values, points_f

    @staticmethod
    def _evaluate_1d_interpolation(
        sample_idx: np.ndarray,
        valid_idx: np.ndarray,
        valid_values: np.ndarray,
        *,
        method: str,
    ) -> np.ndarray:
        sample = np.asarray(sample_idx, dtype=np.float64)
        x = np.asarray(valid_idx, dtype=np.float64)
        y = np.asarray(valid_values, dtype=np.float64)

        if method == "linear" or x.size <= 2:
            return np.interp(sample, x, y)

        if method == "pchip":
            interpolator = PchipInterpolator(x, y, extrapolate=False)
            return np.asarray(interpolator(sample), dtype=np.float64)

        if method == "makima":
            interpolator = Akima1DInterpolator(x, y, method="makima", extrapolate=False)
            return np.asarray(interpolator(sample), dtype=np.float64)

        raise ValueError(f"Methode 1D inconnue : {method!r}")

    def interpolate_peak_map_1d(
        self,
        peak_map: np.ndarray,
        *,
        height: Optional[int] = None,
        fillable_mask: Optional[np.ndarray] = None,
        method: str = "linear",
    ) -> np.ndarray:
        """Fill missing Y gaps (-1) along X only where fillable_mask allows interpolation."""
        data = np.asarray(peak_map, dtype=np.int32)
        if data.ndim != 2:
            raise ValueError(f"Peak map attendu 2D (Z,W), recu shape {data.shape}")

        interpolated = np.array(data, dtype=np.int32, copy=True)
        if interpolated.size == 0:
            return interpolated

        fillable: Optional[np.ndarray] = None
        if fillable_mask is not None:
            fillable = np.asarray(fillable_mask, dtype=bool)
            if fillable.shape != interpolated.shape:
                raise ValueError(
                    f"Fillable mask attendu en shape {interpolated.shape}, recu {fillable.shape}"
                )

        for z in range(interpolated.shape[0]):
            row = interpolated[z]
            valid = row >= 0
            valid_idx = np.flatnonzero(valid)
            if valid_idx.size < 2:
                self._clip_peak_indices_in_place(row, height=height)
                continue

            first = int(valid_idx[0])
            last = int(valid_idx[-1])
            if last <= first:
                continue

            segment_values = self._evaluate_1d_interpolation(
                np.arange(first, last + 1, dtype=np.float64),
                valid_idx,
                row[valid],
                method=method,
            )
            finite_segment = np.isfinite(segment_values)
            segment = np.zeros(segment_values.shape, dtype=np.int32)
            if np.any(finite_segment):
                rounded = np.rint(segment_values[finite_segment]).astype(np.int32)
                rounded = np.maximum(rounded, 0)
                if height is not None:
                    rounded = np.clip(rounded, 0, int(height) - 1)
                segment[finite_segment] = rounded

            segment_row = row[first : last + 1]
            missing = segment_row < 0
            if fillable is not None:
                missing &= fillable[z, first : last + 1]
            missing &= finite_segment
            if np.any(missing):
                segment_row[missing] = segment[missing]
                row[first : last + 1] = segment_row

            self._clip_peak_indices_in_place(row, height=height)

        return interpolated

    def interpolate_peak_map_1d_dual_axis(
        self,
        peak_map: np.ndarray,
        *,
        height: Optional[int] = None,
        support_map: Optional[np.ndarray] = None,
        method: str = "linear",
    ) -> np.ndarray:
        """Apply support-aware interpolation on the primary axis, then on the secondary axis."""
        fillable_mask = self.build_fillable_support_mask(support_map)
        primary = self.interpolate_peak_map_1d(
            peak_map,
            height=height,
            fillable_mask=fillable_mask,
            method=method,
        )
        if primary.ndim != 2 or primary.size == 0:
            return primary

        secondary = self.interpolate_peak_map_1d(
            np.ascontiguousarray(primary.T),
            height=height,
            fillable_mask=None if fillable_mask is None else np.ascontiguousarray(fillable_mask.T),
            method=method,
        )
        return np.ascontiguousarray(secondary.T)

    def interpolate_peak_map_2d_nd(
        self,
        peak_map: np.ndarray,
        *,
        height: Optional[int] = None,
        fillable_mask: Optional[np.ndarray] = None,
        method: str,
    ) -> np.ndarray:
        """Fill missing Y values using a 2D interpolator over valid (Z, X) samples."""
        interpolated, candidate_coords, points, values, points_f = self._prepare_2d_interpolation_data(
            peak_map,
            height=height,
            fillable_mask=fillable_mask,
        )
        if interpolated.size == 0 or candidate_coords.size == 0:
            return interpolated
        algo_label = "2d_linear_nd" if method == "linear_nd" else "2d_clough_tocher"
        if points.shape[0] < 3:
            raise ValueError(
                f"{algo_label} requiert au moins 3 points valides pour combler les trous."
            )
        if np.unique(points[:, 0]).size < 2 or np.unique(points[:, 1]).size < 2:
            raise ValueError(
                f"{algo_label} requiert des points valides repartis sur les axes Z et X."
            )

        candidates_f = candidate_coords.astype(np.float64)
        try:
            if method == "linear_nd":
                interpolator = LinearNDInterpolator(points_f, values, fill_value=np.nan)
            elif method == "clough_tocher":
                interpolator = CloughTocher2DInterpolator(points_f, values, fill_value=np.nan)
            else:
                raise ValueError(f"Methode 2D inconnue : {method!r}")
            interpolated_values = np.asarray(interpolator(candidates_f), dtype=np.float64).reshape(-1)
        except QhullError as exc:
            raise ValueError(
                f"{algo_label} impossible sur cette geometrie de points: {exc}"
            ) from exc

        finite = np.isfinite(interpolated_values)
        if not np.any(finite):
            self._clip_peak_indices_in_place(interpolated, height=height)
            return interpolated

        rounded = np.rint(interpolated_values[finite]).astype(np.int32)
        rounded = np.maximum(rounded, 0)
        if height is not None:
            rounded = np.clip(rounded, 0, int(height) - 1)

        targets = candidate_coords[finite]
        interpolated[targets[:, 0], targets[:, 1]] = rounded
        self._clip_peak_indices_in_place(interpolated, height=height)
        return interpolated

    def interpolate_peak_map_2d_rbf(
        self,
        peak_map: np.ndarray,
        *,
        height: Optional[int] = None,
        fillable_mask: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Fill missing Y values using a thin-plate spline RBF interpolator."""
        (
            interpolated,
            candidate_coords,
            points,
            values,
            points_f,
        ) = self._prepare_2d_interpolation_data(
            peak_map,
            height=height,
            fillable_mask=fillable_mask,
        )
        if interpolated.size == 0 or candidate_coords.size == 0:
            return interpolated

        if points.shape[0] < 3:
            raise ValueError(
                "2d_rbf_thin_plate requiert au moins 3 points valides pour combler les trous."
            )
        if np.unique(points[:, 0]).size < 2 or np.unique(points[:, 1]).size < 2:
            raise ValueError(
                "2d_rbf_thin_plate requiert des points valides repartis sur les axes Z et X."
            )

        sample_points_f, sample_values = self._subsample_interpolation_points(points_f, values)
        neighbors = None
        if int(self.RBF_NEIGHBORS) > 0 and sample_points_f.shape[0] > int(self.RBF_NEIGHBORS):
            neighbors = int(self.RBF_NEIGHBORS)

        candidates_f = candidate_coords.astype(np.float64)
        try:
            interpolator = RBFInterpolator(
                sample_points_f,
                sample_values,
                neighbors=neighbors,
                smoothing=0.0,
                kernel="thin_plate_spline",
                degree=1,
            )
            interpolated_values = np.asarray(interpolator(candidates_f), dtype=np.float64).reshape(-1)
        except Exception as exc:
            raise ValueError(f"2d_rbf_thin_plate impossible sur cette geometrie: {exc}") from exc

        finite = np.isfinite(interpolated_values)
        if not np.any(finite):
            return interpolated

        rounded = np.rint(interpolated_values[finite]).astype(np.int32)
        rounded = np.maximum(rounded, 0)
        if height is not None:
            rounded = np.clip(rounded, 0, int(height) - 1)

        targets = candidate_coords[finite]
        interpolated[targets[:, 0], targets[:, 1]] = rounded
        self._clip_peak_indices_in_place(interpolated, height=height)
        return interpolated

    def interpolate_peak_map_2d_gaussian_fill(
        self,
        peak_map: np.ndarray,
        *,
        height: Optional[int] = None,
        fillable_mask: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Fill missing Y values with nearest-neighbor interpolation, then smooth them."""
        (
            interpolated,
            candidate_coords,
            _points,
            values,
            points_f,
        ) = self._prepare_2d_interpolation_data(
            peak_map,
            height=height,
            fillable_mask=fillable_mask,
        )
        if interpolated.size == 0 or candidate_coords.size == 0:
            return interpolated

        if points_f.shape[0] == 0:
            raise ValueError("2d_gaussian_fill requiert au moins 1 point valide.")

        all_missing_coords = np.argwhere(interpolated < 0)
        if all_missing_coords.size == 0:
            return interpolated

        nearest_values = np.asarray(
            griddata(
                points_f,
                values,
                all_missing_coords.astype(np.float64),
                method="nearest",
            ),
            dtype=np.float64,
        ).reshape(-1)
        finite_nearest = np.isfinite(nearest_values)
        if not np.any(finite_nearest):
            return interpolated

        working = interpolated.astype(np.float64, copy=True)
        nearest_targets = all_missing_coords[finite_nearest]
        working[nearest_targets[:, 0], nearest_targets[:, 1]] = nearest_values[finite_nearest]
        smoothed = gaussian_filter(
            working,
            sigma=float(self.GAUSSIAN_FILL_SIGMA),
            mode="nearest",
        )

        smoothed_values = np.asarray(
            smoothed[candidate_coords[:, 0], candidate_coords[:, 1]],
            dtype=np.float64,
        )
        finite = np.isfinite(smoothed_values)
        if not np.any(finite):
            return interpolated

        rounded = np.rint(smoothed_values[finite]).astype(np.int32)
        rounded = np.maximum(rounded, 0)
        if height is not None:
            rounded = np.clip(rounded, 0, int(height) - 1)

        targets = candidate_coords[finite]
        interpolated[targets[:, 0], targets[:, 1]] = rounded
        self._clip_peak_indices_in_place(interpolated, height=height)
        return interpolated

    def build_distance_map_from_peak_maps(
        self,
        *,
        peak_map_a: np.ndarray,
        peak_map_b: np.ndarray,
        use_mm: bool,
        resolution_ultrasound_mm: float,
    ) -> np.ndarray:
        """Public wrapper to rebuild a (Z,X) distance map directly from BW/FW peak maps."""
        return self._build_distance_map_from_peak_maps(
            peak_map_a=peak_map_a,
            peak_map_b=peak_map_b,
            use_mm=use_mm,
            resolution_ultrasound_mm=resolution_ultrasound_mm,
        )

    @staticmethod
    def _build_distance_map_from_peak_maps(
        *,
        peak_map_a: np.ndarray,
        peak_map_b: np.ndarray,
        use_mm: bool,
        resolution_ultrasound_mm: float,
    ) -> np.ndarray:
        """Rebuild a (Z,X) distance map from two peak maps."""
        data_a = np.asarray(peak_map_a, dtype=np.int32)
        data_b = np.asarray(peak_map_b, dtype=np.int32)
        if data_a.ndim != 2 or data_b.ndim != 2:
            raise ValueError(
                f"Peak maps attendus 2D (Z,W), recus {data_a.shape} et {data_b.shape}"
            )
        if data_a.shape != data_b.shape:
            raise ValueError(
                f"Peak maps incompatibles: {data_a.shape} != {data_b.shape}"
            )

        scale = float(resolution_ultrasound_mm) if bool(use_mm) else 1.0
        distance_map = np.full(data_a.shape, np.nan, dtype=np.float32)
        valid = (data_a >= 0) & (data_b >= 0)
        if np.any(valid):
            diff = np.abs(data_a.astype(np.float32) - data_b.astype(np.float32)) * scale
            distance_map[valid] = diff[valid]
        return distance_map

    def build_overlay_from_peak_maps(
        self,
        *,
        peak_map_a: np.ndarray,
        peak_map_b: np.ndarray,
        image_shape: Tuple[int, int],
        class_A_id: int,
        class_B_id: int,
        line_thickness: int = 1,
    ) -> np.ndarray:
        """Public wrapper to rebuild BW/FW overlay lines from peak maps."""
        return self._build_overlay_from_peak_maps(
            peak_map_a=peak_map_a,
            peak_map_b=peak_map_b,
            image_shape=image_shape,
            class_A_id=class_A_id,
            class_B_id=class_B_id,
            line_thickness=line_thickness,
        )

    def build_interpolated_distance_map(
        self,
        *,
        overlay: np.ndarray,
        class_A_value: int,
        class_B_value: int,
        use_mm: bool,
        resolution_ultrasound_mm: float,
    ) -> np.ndarray:
        """Public wrapper to recompute corrosion distances from BW/FW line overlay."""
        return self._build_interpolated_distance_map(
            overlay=overlay,
            class_A_value=class_A_value,
            class_B_value=class_B_value,
            use_mm=use_mm,
            resolution_ultrasound_mm=resolution_ultrasound_mm,
        )

    def build_piece_volume_from_distance_map(
        self,
        distance_map: np.ndarray,
    ) -> np.ndarray:
        """Public wrapper for the corrosion prism 3D geometry."""
        return self._build_prismatic_piece_from_distance_map(distance_map)

    def build_legacy_piece_volume(
        self,
        *,
        mask_stack: np.ndarray,
        class_A_id: int,
        class_B_id: int,
    ) -> np.ndarray:
        """Public wrapper for the legacy BW/FW 3D geometry."""
        return self._build_solid_volume(
            mask_stack=mask_stack,
            class_A_id=class_A_id,
            class_B_id=class_B_id,
        )

    def compute_piece_anchor(
        self,
        *candidates: Optional[np.ndarray],
    ) -> Optional[Tuple[float, float, float]]:
        """Public wrapper for the preferred camera anchor computation."""
        return self._compute_piece_anchor(*candidates)

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
        """Construit un overlay de lignes BW/FW �f  partir des indices de pics (Y par X)."""
        if peak_map_a.ndim != 2 or peak_map_b.ndim != 2:
            raise ValueError(
                f"Peak maps attendus 2D (Z,W), re�f§us {peak_map_a.shape} et {peak_map_b.shape}"
            )

        height, width = image_shape
        num_slices = min(peak_map_a.shape[0], peak_map_b.shape[0])
        lines_volume = np.zeros((num_slices, height, width), dtype=np.uint8)
        color_A = int(class_A_id)
        color_B = int(class_B_id)
        max_gap = self._get_max_interpolation_gap_px()

        width_map = min(width, peak_map_a.shape[1], peak_map_b.shape[1])
        if width_map <= 0 or num_slices <= 0:
            return lines_volume

        for z in range(num_slices):
            slice_a = peak_map_a[z]
            slice_b = peak_map_b[z]

            valid_a = np.where((slice_a[:width_map] >= 0) & (slice_a[:width_map] < height))[0]
            pts_a = [(int(x), int(slice_a[x])) for x in valid_a]
            if len(pts_a) >= 2:
                for pt_a, pt_b in self._iter_gap_limited_segments(pts_a, max_gap=max_gap):
                    cv2.line(lines_volume[z], pt_a, pt_b, color=color_A, thickness=line_thickness)
            elif len(pts_a) == 1:
                lines_volume[z, pts_a[0][1], pts_a[0][0]] = color_A

            valid_b = np.where((slice_b[:width_map] >= 0) & (slice_b[:width_map] < height))[0]
            pts_b = [(int(x), int(slice_b[x])) for x in valid_b]
            if len(pts_b) >= 2:
                for pt_a, pt_b in self._iter_gap_limited_segments(pts_b, max_gap=max_gap):
                    cv2.line(lines_volume[z], pt_a, pt_b, color=color_B, thickness=line_thickness)
            elif len(pts_b) == 1:
                lines_volume[z, pts_b[0][1], pts_b[0][0]] = color_B

        return lines_volume

    @classmethod
    def _get_max_interpolation_gap_px(cls) -> int:
        try:
            return max(0, int(cls.MAX_INTERPOLATION_GAP_PX))
        except Exception:
            return 0

    @staticmethod
    def _iter_gap_limited_segments(
        points: List[Tuple[int, int]],
        *,
        max_gap: int,
    ):
        for left, right in zip(points[:-1], points[1:]):
            gap = int(right[0]) - int(left[0]) - 1
            if gap <= max_gap:
                yield left, right

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

    def _build_prismatic_piece_from_distance_map(
        self,
        distance_map: np.ndarray,
    ) -> np.ndarray:
        """Build a rectangular prism volume from a (Z, X) distance map only."""
        data = np.asarray(distance_map, dtype=np.float32)
        if data.ndim != 2:
            raise ValueError(f"Distance map attendu 2D (Z,W), recu shape {data.shape}")

        depth, width = data.shape
        finite = data[np.isfinite(data)]
        finite = finite[finite > 0.0]
        if finite.size == 0:
            return np.zeros((depth, 1, width), dtype=np.float32)

        height = max(1, int(np.ceil(float(np.max(finite)))))
        clipped = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)
        clipped = np.maximum(clipped, 0.0)
        column_heights = np.ceil(clipped).astype(np.int32)
        column_heights = np.clip(column_heights, 0, height)

        # For each (z, x), fill y in [0, column_heights[z, x]).
        y_indices = np.arange(height, dtype=np.int32)[None, :, None]
        solid = (y_indices < column_heights[:, None, :]).astype(np.float32)
        return solid

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
        *candidates: Optional[np.ndarray],
    ) -> Optional[Tuple[float, float, float]]:
        """Compute anchor from first non-empty candidate volume."""
        for volume in candidates:
            anchor = self._compute_center_of_mass(volume)
            if anchor is not None:
                return anchor
        return None


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
    raw_peak_index_map_a: Optional[np.ndarray] = None
    raw_peak_index_map_b: Optional[np.ndarray] = None
    ascan_support_map: Optional[np.ndarray] = None
    interpolated_distance_map: Optional[np.ndarray] = None
    interpolated_projection: Optional[np.ndarray] = None
    interpolated_value_range: Optional[Tuple[float, float]] = None
    overlay_volume: Optional[np.ndarray] = None
    overlay_label_ids: Optional[Tuple[int, int]] = None
    overlay_palette: Optional[Dict[int, Tuple[int, int, int, int]]] = None
    mask_height: Optional[int] = None
    piece_volume_raw: Optional[np.ndarray] = None
    piece_volume_interpolated: Optional[np.ndarray] = None
    piece_volume_legacy_raw: Optional[np.ndarray] = None
    piece_volume_legacy_interpolated: Optional[np.ndarray] = None
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
        peak_selection_mode_a: str = "max_peak",
        peak_selection_mode_b: Optional[str] = None,
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
        support_volume = (
            nde_model.get_active_raw_volume()
            if nde_model is not None
            else None
        )

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
                support_volume_data=support_volume,
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
                peak_selection_mode_a=peak_selection_mode_a,
                peak_selection_mode_b=peak_selection_mode_b,
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
                message="Analyse corrosion terminée (données brutes)",
                projection=projection,
                value_range=value_range,
                raw_distance_map=result.distance_map,
                peak_index_map_a=result.peak_index_map_a,
                peak_index_map_b=result.peak_index_map_b,
                raw_peak_index_map_a=result.raw_peak_index_map_a,
                raw_peak_index_map_b=result.raw_peak_index_map_b,
                ascan_support_map=result.ascan_support_map,
                interpolated_distance_map=result.interpolated_distance_map,
                interpolated_projection=interpolated_projection,
                interpolated_value_range=interpolated_value_range,
                overlay_volume=result.overlay_volume,
                overlay_label_ids=result.overlay_label_ids,
                overlay_palette=result.overlay_palette,
                mask_height=int(mask_volume.shape[1]),
                piece_volume_raw=result.piece_volume_raw,
                piece_volume_interpolated=result.piece_volume_interpolated,
                piece_volume_legacy_raw=result.piece_volume_legacy_raw,
                piece_volume_legacy_interpolated=result.piece_volume_legacy_interpolated,
                piece_anchor=result.piece_anchor,
            )

        except Exception as exc:
            return CorrosionWorkflowResult(
                ok=False,
                message=f"Corrosion analysis failed: {exc}",
            )

    def _extract_resolutions(self, nde_model: Optional[NdeModel]) -> Tuple[float, float]:
        """Get crosswise and ultrasound resolutions in mm/px, defaults to 1.0."""
        cross = (
            nde_model.get_axis_resolution_mm("VCoordinate", fallback_axis_index=2)
            if nde_model is not None
            else None
        )
        ultra = (
            nde_model.get_axis_resolution_mm("Ultrasound", fallback_axis_index=1)
            if nde_model is not None
            else None
        )
        if cross is None:
            cross = 1.0
        if ultra is None:
            ultra = 1.0
        return float(cross), float(ultra)

    def run_interpolation(
        self,
        *,
        raw_result: CorrosionWorkflowResult,
        algo: str,
        nde_model: Optional[NdeModel] = None,
    ) -> CorrosionWorkflowResult:
        """Apply interpolation on a previous raw workflow result and return a new result."""
        if not raw_result.ok:
            return CorrosionWorkflowResult(ok=False, message="Pas de résultat brut valide.")
        if raw_result.raw_peak_index_map_a is None or raw_result.raw_peak_index_map_b is None:
            return CorrosionWorkflowResult(ok=False, message="Peak maps bruts manquants.")

        mask_height = raw_result.mask_height or 1
        _, resolution_ultra = self._extract_resolutions(nde_model)

        try:
            # Build a temporary CorrosionAnalysisResult to feed apply_interpolation
            raw_analysis = CorrosionAnalysisResult(
                distance_results={},
                distance_map=raw_result.raw_distance_map if raw_result.raw_distance_map is not None else np.empty((0, 0), dtype=np.float32),
                peak_index_map_a=raw_result.peak_index_map_a,
                peak_index_map_b=raw_result.peak_index_map_b,
                raw_peak_index_map_a=raw_result.raw_peak_index_map_a,
                raw_peak_index_map_b=raw_result.raw_peak_index_map_b,
                ascan_support_map=raw_result.ascan_support_map if raw_result.ascan_support_map is not None else np.empty((0, 0), dtype=np.uint8),
                distance_value_range=raw_result.value_range or (0.0, 0.0),
                interpolated_distance_map=np.empty((0, 0), dtype=np.float32),
                interpolated_value_range=(0.0, 0.0),
                overlay_volume=raw_result.overlay_volume if raw_result.overlay_volume is not None else np.empty((0, 0, 0), dtype=np.uint8),
                overlay_label_ids=raw_result.overlay_label_ids or (0, 0),
                overlay_palette=raw_result.overlay_palette or {},
                overlay_npz_path=None,
                piece_volume_legacy_raw=raw_result.piece_volume_legacy_raw,
            )

            interp_result = self.cscan_corrosion_service.apply_interpolation(
                raw_result=raw_analysis,
                algo=algo,
                mask_height=mask_height,
                use_mm=False,
                resolution_ultrasound_mm=resolution_ultra,
            )

            projection, value_range = self.cscan_corrosion_service.compute_corrosion_projection(
                interp_result.distance_map,
                value_range=interp_result.distance_value_range,
            )

            interpolated_projection: Optional[np.ndarray] = None
            interpolated_value_range: Optional[Tuple[float, float]] = None
            if interp_result.interpolated_distance_map.size > 0:
                interpolated_projection, interpolated_value_range = self.cscan_corrosion_service.compute_corrosion_projection(
                    interp_result.interpolated_distance_map,
                    value_range=interp_result.interpolated_value_range,
                )

            algo_label = normalize_interpolation_algo(algo)
            return CorrosionWorkflowResult(
                ok=True,
                message=f"Interpolation ({algo_label}) terminée",
                projection=projection,
                value_range=value_range,
                raw_distance_map=raw_result.raw_distance_map,
                peak_index_map_a=interp_result.peak_index_map_a,
                peak_index_map_b=interp_result.peak_index_map_b,
                raw_peak_index_map_a=raw_result.raw_peak_index_map_a,
                raw_peak_index_map_b=raw_result.raw_peak_index_map_b,
                ascan_support_map=raw_result.ascan_support_map,
                interpolated_distance_map=interp_result.interpolated_distance_map,
                interpolated_projection=interpolated_projection,
                interpolated_value_range=interpolated_value_range,
                overlay_volume=interp_result.overlay_volume,
                overlay_label_ids=interp_result.overlay_label_ids,
                overlay_palette=interp_result.overlay_palette,
                mask_height=mask_height,
                piece_volume_raw=interp_result.piece_volume_raw,
                piece_volume_interpolated=interp_result.piece_volume_interpolated,
                piece_volume_legacy_raw=interp_result.piece_volume_legacy_raw,
                piece_volume_legacy_interpolated=interp_result.piece_volume_legacy_interpolated,
                piece_anchor=interp_result.piece_anchor,
            )

        except Exception as exc:
            return CorrosionWorkflowResult(
                ok=False,
                message=f"Interpolation failed: {exc}",
            )

