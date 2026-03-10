"""Signal transformation helpers for NDE A-scan volumes."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping, Optional, Sequence

import numpy as np
from scipy.ndimage import uniform_filter1d
from scipy.signal import hilbert

from models.nde_model import NdeModel


@dataclass(frozen=True)
class NdeSignalTransformInfo:
    """Summary of signal transformations declared in NDE process metadata."""

    process_type: str = "unknown"
    rectification: str = "Unknown"
    digital_filter_type: str = "Unknown"
    low_cut_off_frequency: Optional[float] = None
    high_cut_off_frequency: Optional[float] = None
    signal_source: Optional[str] = None
    smoothing_filter: Optional[float] = None
    averaging_factor: Optional[float] = None
    has_rectification: bool = False
    has_digital_filter: bool = False
    has_hilbert_transform: bool = False
    smoothing_parameters_present: bool = False
    has_any_processing_metadata: bool = False
    is_untransformed_ascan: bool = False


@dataclass(frozen=True)
class NdeSignalProcessingOptions:
    """User-selected signal processing options applied on open."""

    apply_hilbert: bool = False
    apply_smoothing: bool = False

    def is_passthrough(self) -> bool:
        return not self.apply_hilbert and not self.apply_smoothing


class NdeSignalProcessingService:
    """Business logic for process metadata inspection and signal processing."""

    GENERAL_SMOOTHING_WINDOW = 5

    def inspect_group_setup(
        self,
        *,
        setup_json: Mapping[str, Any],
        group_id: Optional[int] = None,
        group_idx: Optional[int] = None,
    ) -> NdeSignalTransformInfo:
        """Extract transformation hints from the process section of one group."""
        group = self._resolve_group(setup_json, group_id=group_id, group_idx=group_idx)
        if not isinstance(group, Mapping):
            return NdeSignalTransformInfo()

        processes = group.get("processes") or []
        acquisition_type = "unknown"
        acquisition_payload: Mapping[str, Any] = {}
        if isinstance(processes, Sequence):
            acquisition_type, acquisition_payload = self._find_acquisition_process(processes)
        if acquisition_type == "unknown":
            acquisition_type, acquisition_payload = self._find_legacy_acquisition_process(group)

        rectification = self._coerce_string(
            self._find_first_key(acquisition_payload, "rectification"),
            default="Unknown",
        )

        digital_filter = self._find_first_key(acquisition_payload, "digitalBandPassFilter")
        filter_type = "Unknown"
        low_cut = None
        high_cut = None
        if isinstance(digital_filter, Mapping):
            filter_type = self._coerce_string(
                digital_filter.get("filterType"),
                default="Unknown",
            )
            low_cut = self._coerce_float(digital_filter.get("lowCutOffFrequency"))
            high_cut = self._coerce_float(digital_filter.get("highCutOffFrequency"))

        signal_source = self._coerce_optional_string(
            self._find_first_key(processes, "signalSource")
        )
        smoothing_filter = self._coerce_float(
            self._find_first_key(acquisition_payload, "smoothingFilter")
        )
        averaging_factor = self._coerce_float(
            self._find_first_key(acquisition_payload, "averagingFactor")
        )

        has_rectification = rectification not in {"None", "Unknown"}
        has_digital_filter = filter_type not in {"None", "Unknown"}
        has_hilbert_transform = str(signal_source or "").strip().lower() == "analytic"
        smoothing_parameters_present = (
            (smoothing_filter is not None and smoothing_filter > 0.0)
            or (averaging_factor is not None and averaging_factor > 0.0)
        )
        has_any_processing_metadata = any(
            [
                acquisition_type != "unknown",
                rectification != "Unknown",
                filter_type != "Unknown",
                signal_source is not None,
                smoothing_filter is not None,
                averaging_factor is not None,
            ]
        )
        is_untransformed_ascan = rectification == "None" and filter_type == "None"

        return NdeSignalTransformInfo(
            process_type=acquisition_type,
            rectification=rectification,
            digital_filter_type=filter_type,
            low_cut_off_frequency=low_cut,
            high_cut_off_frequency=high_cut,
            signal_source=signal_source,
            smoothing_filter=smoothing_filter,
            averaging_factor=averaging_factor,
            has_rectification=has_rectification,
            has_digital_filter=has_digital_filter,
            has_hilbert_transform=has_hilbert_transform,
            smoothing_parameters_present=smoothing_parameters_present,
            has_any_processing_metadata=has_any_processing_metadata,
            is_untransformed_ascan=is_untransformed_ascan,
        )

    def serialize_transform_info(self, info: NdeSignalTransformInfo) -> dict[str, Any]:
        """Serialize a transform summary to plain metadata."""
        return asdict(info)

    def coerce_transform_info(self, payload: Any) -> NdeSignalTransformInfo:
        """Normalize arbitrary metadata payloads to a transform info dataclass."""
        if isinstance(payload, NdeSignalTransformInfo):
            return payload
        if not isinstance(payload, Mapping):
            return NdeSignalTransformInfo()

        return NdeSignalTransformInfo(
            process_type=self._coerce_string(payload.get("process_type"), default="unknown"),
            rectification=self._coerce_string(payload.get("rectification"), default="Unknown"),
            digital_filter_type=self._coerce_string(
                payload.get("digital_filter_type"),
                default="Unknown",
            ),
            low_cut_off_frequency=self._coerce_float(payload.get("low_cut_off_frequency")),
            high_cut_off_frequency=self._coerce_float(payload.get("high_cut_off_frequency")),
            signal_source=self._coerce_optional_string(payload.get("signal_source")),
            smoothing_filter=self._coerce_float(payload.get("smoothing_filter")),
            averaging_factor=self._coerce_float(payload.get("averaging_factor")),
            has_rectification=bool(payload.get("has_rectification", False)),
            has_digital_filter=bool(payload.get("has_digital_filter", False)),
            has_hilbert_transform=bool(payload.get("has_hilbert_transform", False)),
            smoothing_parameters_present=bool(
                payload.get("smoothing_parameters_present", False)
            ),
            has_any_processing_metadata=bool(
                payload.get("has_any_processing_metadata", False)
            ),
            is_untransformed_ascan=bool(payload.get("is_untransformed_ascan", False)),
        )

    def default_processing_options(
        self,
        info: NdeSignalTransformInfo,
    ) -> NdeSignalProcessingOptions:
        """Return the default open-time processing selection."""
        if info.is_untransformed_ascan:
            return NdeSignalProcessingOptions(
                apply_hilbert=True,
                apply_smoothing=True,
            )
        return NdeSignalProcessingOptions()

    def build_detection_title(self, info: NdeSignalTransformInfo) -> str:
        """Return a compact title for the open dialog."""
        if info.is_untransformed_ascan:
            return "A-scan brut detecte"
        if info.has_rectification or info.has_digital_filter or info.has_hilbert_transform:
            return "Transformations detectees"
        if info.has_any_processing_metadata:
            return "Metadonnees de process detectees"
        return "Metadonnees de process indisponibles"

    def build_detection_lines(self, info: NdeSignalTransformInfo) -> list[str]:
        """Return human-readable process details for the open dialog."""
        lines = [
            f"Processus: {info.process_type}",
            f"Rectification: {info.rectification}",
            f"Filtre numerique: {info.digital_filter_type}",
        ]
        if info.low_cut_off_frequency is not None or info.high_cut_off_frequency is not None:
            lines.append(
                "Coupures filtre: "
                f"low={self._format_optional_float(info.low_cut_off_frequency)} Hz, "
                f"high={self._format_optional_float(info.high_cut_off_frequency)} Hz"
            )
        if info.signal_source is not None:
            lines.append(f"Signal source: {info.signal_source}")
        if info.smoothing_filter is not None:
            lines.append(f"Smoothing filter: {info.smoothing_filter:g} Hz")
        if info.averaging_factor is not None:
            lines.append(f"Averaging factor: {info.averaging_factor:g}")
        if info.is_untransformed_ascan:
            lines.append("Etat detecte: RF bipolaire non redresse et non filtre.")
        elif info.has_rectification or info.has_digital_filter or info.has_hilbert_transform:
            lines.append("Etat detecte: acquisition deja transformee selon les metadonnees.")
        else:
            lines.append("Etat detecte: impossible de confirmer un A-scan brut.")
        return lines

    def describe_selection(self, options: NdeSignalProcessingOptions) -> str:
        """Return a compact status string for the selected processing."""
        labels: list[str] = []
        if options.apply_hilbert:
            labels.append("Hilbert")
        if options.apply_smoothing:
            labels.append("lissage general")
        if not labels:
            return "signal NDE conserve"
        return " + ".join(labels)

    def apply_processing_to_model(
        self,
        model: Optional[NdeModel],
        options: NdeSignalProcessingOptions,
    ) -> None:
        """Create the processed variant and select the active signal."""
        if model is None:
            return

        model.metadata["signal_processing_selection"] = {
            "apply_hilbert": bool(options.apply_hilbert),
            "apply_smoothing": bool(options.apply_smoothing),
        }

        if options.is_passthrough():
            model.clear_processed_volume()
            model.set_active_signal_variant("source")
            model.metadata["signal_processing_active"] = "source"
            return

        source_volume = model.volume
        if source_volume is None or getattr(source_volume, "ndim", 0) != 3:
            model.clear_processed_volume()
            model.set_active_signal_variant("source")
            model.metadata["signal_processing_active"] = "source"
            return

        ultrasound_axis = self._resolve_ultrasound_axis_index(
            model.metadata.get("axis_order") or [],
            source_volume.ndim,
        )
        processed = self.apply_processing(
            source_volume,
            ultrasound_axis=ultrasound_axis,
            options=options,
        )
        model.set_processed_volume(processed)
        model.set_active_signal_variant("processed")
        model.metadata["signal_processing_active"] = "processed"

    def apply_processing(
        self,
        volume: np.ndarray,
        *,
        ultrasound_axis: int,
        options: NdeSignalProcessingOptions,
    ) -> np.ndarray:
        """Apply the selected generic processing chain to a volume."""
        result = np.asarray(volume, dtype=np.float32)

        if options.apply_hilbert:
            result = self.apply_general_hilbert_envelope(
                result,
                axis=ultrasound_axis,
            )
        if options.apply_smoothing:
            result = self.apply_general_smoothing(
                result,
                axis=ultrasound_axis,
            )

        return np.asarray(result, dtype=np.float32)

    def apply_general_hilbert_envelope(
        self,
        signal: np.ndarray,
        *,
        axis: int,
    ) -> np.ndarray:
        """Return the Hilbert envelope of the signal along the ultrasound axis."""
        analytic = hilbert(np.asarray(signal, dtype=np.float32), axis=axis)
        envelope = np.abs(analytic)
        return np.asarray(envelope, dtype=np.float32)

    def apply_general_smoothing(
        self,
        signal: np.ndarray,
        *,
        axis: int,
        window_size: Optional[int] = None,
    ) -> np.ndarray:
        """Apply a simple 1D moving-average smoothing along the ultrasound axis."""
        size = int(window_size or self.GENERAL_SMOOTHING_WINDOW)
        if size < 1:
            size = 1
        if size % 2 == 0:
            size += 1
        smoothed = uniform_filter1d(
            np.asarray(signal, dtype=np.float32),
            size=size,
            axis=axis,
            mode="nearest",
        )
        return np.asarray(smoothed, dtype=np.float32)

    def smooth_for_rectification_none(
        self,
        signal: np.ndarray,
        *,
        axis: int,
    ) -> Optional[np.ndarray]:
        """Future per-rectification smoothing hook."""
        return None

    def smooth_for_rectification_positive(
        self,
        signal: np.ndarray,
        *,
        axis: int,
    ) -> Optional[np.ndarray]:
        """Future per-rectification smoothing hook."""
        return None

    def smooth_for_rectification_negative(
        self,
        signal: np.ndarray,
        *,
        axis: int,
    ) -> Optional[np.ndarray]:
        """Future per-rectification smoothing hook."""
        return None

    def smooth_for_rectification_full(
        self,
        signal: np.ndarray,
        *,
        axis: int,
    ) -> Optional[np.ndarray]:
        """Future per-rectification smoothing hook."""
        return None

    def smooth_for_filter_none(
        self,
        signal: np.ndarray,
        *,
        axis: int,
    ) -> Optional[np.ndarray]:
        """Future per-filter smoothing hook."""
        return None

    def smooth_for_filter_low_pass(
        self,
        signal: np.ndarray,
        *,
        axis: int,
    ) -> Optional[np.ndarray]:
        """Future per-filter smoothing hook."""
        return None

    def smooth_for_filter_high_pass(
        self,
        signal: np.ndarray,
        *,
        axis: int,
    ) -> Optional[np.ndarray]:
        """Future per-filter smoothing hook."""
        return None

    def smooth_for_filter_band_pass(
        self,
        signal: np.ndarray,
        *,
        axis: int,
    ) -> Optional[np.ndarray]:
        """Future per-filter smoothing hook."""
        return None

    def _resolve_group(
        self,
        setup_json: Mapping[str, Any],
        *,
        group_id: Optional[int],
        group_idx: Optional[int],
    ) -> Optional[Mapping[str, Any]]:
        groups = setup_json.get("groups") or []
        if not isinstance(groups, Sequence):
            return None

        if group_id is not None:
            for group in groups:
                if not isinstance(group, Mapping):
                    continue
                try:
                    if int(group.get("id")) == int(group_id):
                        return group
                except Exception:
                    continue

        if group_idx is not None:
            idx = int(group_idx) - 1
            if 0 <= idx < len(groups):
                group = groups[idx]
                if isinstance(group, Mapping):
                    return group

        for group in groups:
            if isinstance(group, Mapping):
                return group
        return None

    def _find_acquisition_process(
        self,
        processes: Sequence[Any],
    ) -> tuple[str, Mapping[str, Any]]:
        for process in processes:
            if not isinstance(process, Mapping):
                continue
            for key in ("ultrasonicConventional", "ultrasonicPhasedArray"):
                payload = process.get(key)
                if isinstance(payload, Mapping):
                    return key, payload
        return "unknown", {}

    def _find_legacy_acquisition_process(
        self,
        group: Mapping[str, Any],
    ) -> tuple[str, Mapping[str, Any]]:
        labels = {
            "paut": "paut_legacy_3x",
            "ut": "ut_legacy_3x",
        }
        for key, label in labels.items():
            payload = group.get(key)
            if isinstance(payload, Mapping):
                return label, payload
        return "unknown", {}

    def _find_first_key(self, payload: Any, target_key: str) -> Any:
        if isinstance(payload, Mapping):
            if target_key in payload:
                return payload[target_key]
            for value in payload.values():
                found = self._find_first_key(value, target_key)
                if found is not None:
                    return found
        elif isinstance(payload, Sequence) and not isinstance(payload, (str, bytes, bytearray)):
            for item in payload:
                found = self._find_first_key(item, target_key)
                if found is not None:
                    return found
        return None

    @staticmethod
    def _resolve_ultrasound_axis_index(axis_order: Sequence[Any], ndim: int) -> int:
        for idx, name in enumerate(axis_order):
            if isinstance(name, str) and name.lower() == "ultrasound":
                return idx
        return max(0, min(int(ndim) - 1, 2))

    @staticmethod
    def _coerce_string(value: Any, *, default: str) -> str:
        if value is None:
            return default
        text = str(value).strip()
        return text if text else default

    @staticmethod
    def _coerce_optional_string(value: Any) -> Optional[str]:
        if value is None:
            return None
        text = str(value).strip()
        return text or None

    @staticmethod
    def _coerce_float(value: Any) -> Optional[float]:
        if value is None:
            return None
        try:
            result = float(value)
        except Exception:
            return None
        if not np.isfinite(result):
            return None
        return result

    @staticmethod
    def _format_optional_float(value: Optional[float]) -> str:
        if value is None:
            return "-"
        return f"{value:g}"
