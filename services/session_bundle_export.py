"""Orchestrate the `Export all` bundle for the active session."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import numpy as np

from config.constants import CORROSION_STAGE_INTERPOLATED, CORROSION_STAGE_RAW
from models.layer_stack_model import LayerState
from models.view_state_model import ViewStateModel
from services.annotation_session_manager import AnnotationSessionManager
from services.cscan_corrosion_service import CScanCorrosionService
from services.overlay_export import OverlayExport
from services.split_service import SplitFlawNoflawService
from utils.filename_utils import sanitize_filename_component


@dataclass(frozen=True)
class BundleSentinelExportOptions:
    """Sentinel-specific options collected by the bundle dialog."""

    sentinel_source_view: str = "dscan"
    rotation_degrees: int = 0
    rotation_axes: str = ""
    transpose_axes: str = ""
    output_suffix: str = "_sentinel"
    mirror_horizontal: bool = False
    mirror_vertical: bool = False
    mirror_z: bool = False
    strict_mode: bool = False


@dataclass(frozen=True)
class SessionBundleExportRequest:
    """Full export plan for the active session."""

    output_root: str
    primary_axis_name: Optional[str]
    main_layer_id: str = ""
    export_session_npz: bool = True
    export_sentinel_npz: bool = True
    export_nnunet_pngs: bool = True
    export_corrosion_cscan: bool = True
    sentinel: BundleSentinelExportOptions = field(
        default_factory=BundleSentinelExportOptions
    )


@dataclass(frozen=True)
class SessionBundleExportResult:
    """Collected bundle export outputs."""

    output_root: str
    overlay_npz_paths: tuple[str, ...] = ()
    sentinel_npz_path: Optional[str] = None
    nnunet_root: Optional[str] = None
    nnunet_message: Optional[str] = None
    corrosion_cscan_paths: tuple[tuple[str, str], ...] = ()
    notes: tuple[str, ...] = ()

    def build_summary(self) -> str:
        """Return a compact user-facing summary."""
        lines = [f"Export root: {self.output_root}"]
        lines.append(f"Overlay NPZ: {len(self.overlay_npz_paths)}")
        lines.append(
            "Sentinel NPZ: "
            + (self.sentinel_npz_path if self.sentinel_npz_path else "not exported")
        )
        lines.append(
            "nnU-Net: " + (self.nnunet_root if self.nnunet_root else "not exported")
        )
        lines.append(f"Corrosion C-scan: {len(self.corrosion_cscan_paths)}")
        if self.notes:
            lines.append("Notes:")
            lines.extend(f"- {note}" for note in self.notes)
        return "\n".join(lines)


class SessionBundleExportService:
    """Coordinate bundle exports while reusing dedicated export services."""

    def __init__(
        self,
        *,
        overlay_export: Optional[OverlayExport] = None,
        split_export_service: Optional[SplitFlawNoflawService] = None,
        cscan_corrosion_service: Optional[CScanCorrosionService] = None,
    ) -> None:
        self.logger = logging.getLogger(__name__)
        self.overlay_export = overlay_export or OverlayExport()
        self.split_export_service = split_export_service or SplitFlawNoflawService()
        self.cscan_corrosion_service = cscan_corrosion_service or CScanCorrosionService()

    def export_bundle(
        self,
        *,
        session_manager: AnnotationSessionManager,
        nde_model: Any,
        nde_file: Optional[str],
        request: SessionBundleExportRequest,
        signal_processing_options: Optional[Any] = None,
    ) -> SessionBundleExportResult:
        """Export the selected artifacts for the active session."""
        if not any(
            (
                request.export_session_npz,
                request.export_sentinel_npz,
                request.export_nnunet_pngs,
                request.export_corrosion_cscan,
            )
        ):
            raise ValueError("Select at least one export category.")

        layer_stack = session_manager.get_active_layer_stack()
        if layer_stack is None or not layer_stack.layers:
            raise ValueError("No layer is available in the active session.")

        layers = tuple(layer_stack.layers)
        output_root = Path(str(request.output_root)).expanduser()
        output_root.mkdir(parents=True, exist_ok=True)

        nde_path = self._resolve_nde_path(nde_file=nde_file, nde_model=nde_model)
        nde_stem = self._sanitize_output_stem(Path(nde_path).stem, fallback="nde_export")
        volume_shape = self._resolve_volume_shape(nde_model)

        main_layer = None
        if request.export_sentinel_npz or request.export_nnunet_pngs:
            main_layer = self._resolve_main_layer(layers, request.main_layer_id)
            if main_layer.mask_volume is None:
                raise ValueError("The selected main layer has no mask volume.")

        overlay_paths: tuple[str, ...] = ()
        sentinel_path: Optional[str] = None
        nnunet_root: Optional[str] = None
        nnunet_message: Optional[str] = None
        corrosion_paths: tuple[tuple[str, str], ...] = ()
        notes: list[str] = []

        if request.export_session_npz:
            overlay_paths, overlay_notes = self._export_session_npz_layers(
                layers=layers,
                output_root=output_root,
                nde_stem=nde_stem,
                volume_shape=volume_shape,
                primary_axis_name=request.primary_axis_name,
            )
            notes.extend(overlay_notes)

        if request.export_sentinel_npz and main_layer is not None:
            sentinel_path = self._export_main_layer_sentinel_npz(
                main_layer=main_layer,
                output_root=output_root,
                nde_stem=nde_stem,
                volume_shape=volume_shape,
                primary_axis_name=request.primary_axis_name,
                sentinel=request.sentinel,
            )

        if request.export_nnunet_pngs and main_layer is not None:
            nnunet_root, nnunet_message = self._export_main_layer_nnunet(
                nde_model=nde_model,
                nde_file=nde_path,
                main_layer=main_layer,
                output_root=output_root,
                nde_stem=nde_stem,
                signal_processing_options=signal_processing_options,
            )

        if request.export_corrosion_cscan:
            corrosion_paths, corrosion_notes = self._export_corrosion_cscans(
                layers=layers,
                output_root=output_root,
                nde_path=nde_path,
            )
            notes.extend(corrosion_notes)

        return SessionBundleExportResult(
            output_root=str(output_root),
            overlay_npz_paths=overlay_paths,
            sentinel_npz_path=sentinel_path,
            nnunet_root=nnunet_root,
            nnunet_message=nnunet_message,
            corrosion_cscan_paths=corrosion_paths,
            notes=tuple(notes),
        )

    def _export_session_npz_layers(
        self,
        *,
        layers: tuple[LayerState, ...],
        output_root: Path,
        nde_stem: str,
        volume_shape: Optional[tuple[int, int, int]],
        primary_axis_name: Optional[str],
    ) -> tuple[tuple[str, ...], list[str]]:
        overlay_dir = output_root / "overlay_npz"
        overlay_dir.mkdir(parents=True, exist_ok=True)

        exported: list[str] = []
        notes: list[str] = []
        for index, layer in enumerate(layers, start=1):
            if layer.mask_volume is None:
                notes.append(f"Layer skipped (no mask volume): {layer.name}")
                continue
            layer_slug = self._sanitize_output_stem(layer.name, fallback=f"layer_{index:02d}")
            destination = overlay_dir / f"{nde_stem}_{index:02d}_{layer_slug}.npz"
            saved_path = self.overlay_export.save_npz(
                np.asarray(layer.mask_volume, dtype=np.uint8),
                str(destination),
                expected_shape=volume_shape,
                primary_axis_name=primary_axis_name,
            )
            exported.append(saved_path)
        return tuple(exported), notes

    def _export_main_layer_sentinel_npz(
        self,
        *,
        main_layer: LayerState,
        output_root: Path,
        nde_stem: str,
        volume_shape: Optional[tuple[int, int, int]],
        primary_axis_name: Optional[str],
        sentinel: BundleSentinelExportOptions,
    ) -> str:
        sentinel_dir = output_root / "sentinel"
        sentinel_dir.mkdir(parents=True, exist_ok=True)
        layer_slug = self._sanitize_output_stem(main_layer.name, fallback="main_layer")
        destination = sentinel_dir / f"{nde_stem}_{layer_slug}.npz"
        return self.overlay_export.save_sentinel_npz(
            np.asarray(main_layer.mask_volume, dtype=np.uint8),
            str(destination),
            expected_shape=volume_shape,
            primary_axis_name=primary_axis_name,
            sentinel_source_view=sentinel.sentinel_source_view,
            rotation_degrees=sentinel.rotation_degrees,
            rotation_axes=sentinel.rotation_axes,
            transpose_axes=sentinel.transpose_axes,
            output_suffix=sentinel.output_suffix,
            mirror_horizontal=sentinel.mirror_horizontal,
            mirror_vertical=sentinel.mirror_vertical,
            mirror_z=sentinel.mirror_z,
            strict_mode=sentinel.strict_mode,
        )

    def _export_main_layer_nnunet(
        self,
        *,
        nde_model: Any,
        nde_file: str,
        main_layer: LayerState,
        output_root: Path,
        nde_stem: str,
        signal_processing_options: Optional[Any],
    ) -> tuple[str, str]:
        nnunet_dir = output_root / "nnunet"
        nnunet_dir.mkdir(parents=True, exist_ok=True)
        layer_slug = self._sanitize_output_stem(main_layer.name, fallback="main_layer")
        base_name_override = f"{nde_stem}_{layer_slug}"
        success, message = self.split_export_service.export_nnunet_dataset(
            nde_model=nde_model,
            annotation_model=None,
            nde_file=nde_file,
            output_root=str(nnunet_dir),
            filename_prefix=None,
            filename_suffix=None,
            signal_processing_options=signal_processing_options,
            mask_volume_override=np.asarray(main_layer.mask_volume, dtype=np.uint8),
            base_name_override=base_name_override,
        )
        if not success:
            raise RuntimeError(message)
        return str(nnunet_dir), message

    def _export_corrosion_cscans(
        self,
        *,
        layers: tuple[LayerState, ...],
        output_root: Path,
        nde_path: str,
    ) -> tuple[tuple[tuple[str, str], ...], list[str]]:
        cscan_dir = output_root / "cscan"
        cscan_dir.mkdir(parents=True, exist_ok=True)

        exported: list[tuple[str, str]] = []
        notes: list[str] = []
        for index, layer in enumerate(layers, start=1):
            if str(getattr(layer, "layer_kind", "")).strip().casefold() != "corrosion":
                continue
            stage = self._layer_corrosion_stage(layer)
            if stage not in {CORROSION_STAGE_RAW, CORROSION_STAGE_INTERPOLATED}:
                continue
            projection_payload = self._resolve_corrosion_projection_payload(layer, stage)
            if projection_payload is None:
                notes.append(f"Corrosion C-scan skipped (missing projection): {layer.name}")
                continue
            projection, value_range = projection_payload
            layer_slug = self._sanitize_output_stem(layer.name, fallback=f"corrosion_{index:02d}")
            suffix = f"_{index:02d}_{layer_slug}_{stage}"
            exported.append(
                self.cscan_corrosion_service.save_cscan_projection(
                    output_directory=str(cscan_dir),
                    nde_filename=nde_path,
                    projection=projection,
                    value_range=value_range,
                    filename_suffix=suffix,
                )
            )

        if not exported:
            notes.append("No raw/interpolated corrosion layer was exportable.")
        return tuple(exported), notes

    def _resolve_corrosion_projection_payload(
        self,
        layer: LayerState,
        stage: str,
    ) -> Optional[tuple[np.ndarray, tuple[float, float]]]:
        runtime_cache = getattr(layer, "corrosion_runtime_cache", None)
        runtime_projection = getattr(runtime_cache, "projection", None)
        if self._is_projection_payload(runtime_projection):
            projection, value_range = runtime_projection
            return np.asarray(projection, dtype=np.float32), (
                float(value_range[0]),
                float(value_range[1]),
            )

        state = getattr(layer, "corrosion_state", None)
        payload = getattr(state, "payload", None)
        if not isinstance(payload, dict):
            return None

        if stage == CORROSION_STAGE_RAW:
            peak_a = payload.get("corrosion_raw_peak_index_map_a")
            peak_b = payload.get("corrosion_raw_peak_index_map_b")
            if peak_a is None or peak_b is None:
                peak_a = payload.get("corrosion_peak_index_map_a")
                peak_b = payload.get("corrosion_peak_index_map_b")
        else:
            peak_a = payload.get("corrosion_peak_index_map_a")
            peak_b = payload.get("corrosion_peak_index_map_b")
            if peak_a is None or peak_b is None:
                peak_a = payload.get("corrosion_raw_peak_index_map_a")
                peak_b = payload.get("corrosion_raw_peak_index_map_b")

        if peak_a is None or peak_b is None:
            return None

        distance_map = self.cscan_corrosion_service.build_distance_map_from_peak_maps(
            peak_map_a=np.asarray(peak_a),
            peak_map_b=np.asarray(peak_b),
            use_mm=False,
            resolution_ultrasound_mm=1.0,
        )
        projection, value_range = self.cscan_corrosion_service.compute_corrosion_projection(
            distance_map
        )
        return np.asarray(projection, dtype=np.float32), (
            float(value_range[0]),
            float(value_range[1]),
        )

    @staticmethod
    def _is_projection_payload(value: Any) -> bool:
        if not isinstance(value, tuple) or len(value) != 2:
            return False
        projection, value_range = value
        if projection is None or value_range is None:
            return False
        try:
            return len(value_range) == 2
        except Exception:
            return False

    @staticmethod
    def _resolve_main_layer(
        layers: tuple[LayerState, ...],
        requested_layer_id: str,
    ) -> LayerState:
        target_id = str(requested_layer_id or "").strip()
        if target_id:
            for layer in layers:
                if str(layer.id) == target_id:
                    return layer
        for layer in layers:
            if str(getattr(layer, "layer_kind", "")).strip().casefold() != "corrosion":
                return layer
        raise ValueError("No main layer is available for Sentinel/nnU-Net export.")

    @staticmethod
    def _resolve_nde_path(*, nde_file: Optional[str], nde_model: Any) -> str:
        if nde_file:
            return str(nde_file)
        metadata = getattr(nde_model, "metadata", {}) or {}
        path = str(metadata.get("path") or "").strip()
        if path:
            return path
        return str(Path.cwd() / "current.nde")

    @staticmethod
    def _resolve_volume_shape(nde_model: Any) -> Optional[tuple[int, int, int]]:
        volume = None
        if nde_model is not None and hasattr(nde_model, "get_active_volume"):
            volume = nde_model.get_active_volume()
        if volume is None and nde_model is not None:
            volume = getattr(nde_model, "volume", None)
        if volume is None:
            return None
        data = np.asarray(volume)
        if data.ndim != 3:
            return None
        return tuple(int(dim) for dim in data.shape)

    @staticmethod
    def _sanitize_output_stem(value: str, *, fallback: str) -> str:
        return sanitize_filename_component(value, fallback=fallback)

    @staticmethod
    def _layer_corrosion_stage(layer: LayerState) -> str:
        state = getattr(layer, "corrosion_state", None)
        stage = getattr(state, "stage", None)
        return ViewStateModel.normalize_corrosion_session_stage(stage)
