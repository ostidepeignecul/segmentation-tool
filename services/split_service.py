"""
Dataset export helpers built from the current endviews and annotation mask.

Supported flows:
1) flaw/noflaw split using RGB + uint8 endview exports
2) autonomous nnU-Net-style export using imagesTr/labelsTr
"""

from __future__ import annotations

import logging
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np

from services.endview_export import EndviewExportService
from services.nde_signal_processing_service import NdeSignalProcessingOptions


@dataclass(frozen=True)
class SplitExportContext:
    """Resolved inputs shared by the dataset export flows."""

    mask_volume: np.ndarray
    image_volume: np.ndarray
    base_name: str
    processing_tag: str
    processing_description: str
    use_active_signal: bool
    min_value: float
    max_value: float
    primary_axis_name: str
    secondary_axis_name: str


class SplitFlawNoflawService:
    """Orchestrate dataset exports from the active NDE volume and mask."""

    def __init__(self, endview_export_service: Optional[EndviewExportService] = None) -> None:
        self.logger = logging.getLogger(__name__)
        self.endview_export = endview_export_service or EndviewExportService()

    def split_endviews(
        self,
        *,
        nde_model,
        annotation_model,
        nde_file: Optional[str],
        output_root: Path | str,
        filename_prefix: Optional[str] = None,
        filename_suffix: Optional[str] = None,
        signal_processing_options: Optional[NdeSignalProcessingOptions] = None,
    ) -> Tuple[bool, str]:
        """Export endviews then split flaw/noflaw buckets from the current mask."""
        try:
            context = self._resolve_export_context(
                nde_model=nde_model,
                annotation_model=annotation_model,
                nde_file=nde_file,
                signal_processing_options=signal_processing_options,
            )
        except ValueError as exc:
            return False, str(exc)

        base_dir = Path(output_root) / f"{context.base_name}{context.processing_tag}"
        rgb_complete = base_dir / "endviews_rgb24" / "complete"
        uint8_complete = base_dir / "endviews_uint8" / "complete"

        prefix = filename_prefix or ""
        suffix = filename_suffix or ""

        targets = {
            "rgb": {
                "flaw": base_dir / "endviews_rgb24" / "flaw",
                "noflaw": base_dir / "endviews_rgb24" / "noflaw",
                "gtmask_flaw": base_dir / "endviews_rgb24" / "gtmask" / "flaw",
                "gtmask_noflaw": base_dir / "endviews_rgb24" / "gtmask" / "noflaw",
            },
            "uint8": {
                "flaw": base_dir / "endviews_uint8" / "flaw",
                "noflaw": base_dir / "endviews_uint8" / "noflaw",
                "gtmask_flaw": base_dir / "endviews_uint8" / "gtmask" / "flaw",
                "gtmask_noflaw": base_dir / "endviews_uint8" / "gtmask" / "noflaw",
            },
        }

        for fmt_targets in targets.values():
            for path in fmt_targets.values():
                path.mkdir(parents=True, exist_ok=True)
        rgb_complete.mkdir(parents=True, exist_ok=True)
        uint8_complete.mkdir(parents=True, exist_ok=True)

        self.logger.info(
            "Export endviews RGB24 -> %s (active_signal=%s)",
            rgb_complete,
            context.use_active_signal,
        )
        success_rgb, msg_rgb = self.endview_export.export_endviews(
            nde_file=nde_file,
            nde_model=nde_model,
            output_folder=str(rgb_complete),
            export_format="rgb",
            use_active_signal=context.use_active_signal,
        )
        if not success_rgb:
            return False, msg_rgb

        self.logger.info(
            "Export endviews UINT8 -> %s (active_signal=%s)",
            uint8_complete,
            context.use_active_signal,
        )
        success_uint8, msg_uint8 = self.endview_export.export_endviews(
            nde_file=nde_file,
            nde_model=nde_model,
            output_folder=str(uint8_complete),
            export_format="uint8",
            use_active_signal=context.use_active_signal,
        )
        if not success_uint8:
            return False, msg_uint8

        stats = {
            "flaw_masks": 0,
            "noflaw_masks": 0,
            "flaw_rgb_images": 0,
            "noflaw_rgb_images": 0,
            "flaw_uint8_images": 0,
            "noflaw_uint8_images": 0,
        }

        depth = int(context.mask_volume.shape[0])
        for idx in range(depth):
            mask_slice = np.asarray(context.mask_volume[idx], dtype=np.uint8)
            bucket = "flaw" if np.any(mask_slice != 0) else "noflaw"
            position_filename = idx * 1500
            base_filename = f"endview_{position_filename:012d}"
            source_filename = f"{base_filename}.png"
            output_filename = f"{prefix}{base_filename}{suffix}.png"

            self._write_png(targets["rgb"][f"gtmask_{bucket}"] / output_filename, mask_slice)
            self._write_png(targets["uint8"][f"gtmask_{bucket}"] / output_filename, mask_slice)
            stats[f"{bucket}_masks"] += 1

            self._safe_copy(rgb_complete / source_filename, targets["rgb"][bucket], output_filename)
            self._safe_copy(uint8_complete / source_filename, targets["uint8"][bucket], output_filename)
            stats[f"{bucket}_rgb_images"] += int((rgb_complete / source_filename).exists())
            stats[f"{bucket}_uint8_images"] += int((uint8_complete / source_filename).exists())

            if (idx + 1) % 100 == 0:
                self.logger.info("Traitement slices: %s/%s", idx + 1, depth)

        summary = "\n".join(
            [
                "=== Resume ===",
                f"Signal utilise: {context.processing_description}",
                *(f"{key}: {value}" for key, value in stats.items()),
            ]
        )
        return True, f"Split flaw/noflaw termine.\n{summary}"

    def export_nnunet_dataset(
        self,
        *,
        nde_model,
        annotation_model,
        nde_file: Optional[str],
        output_root: Path | str,
        filename_prefix: Optional[str] = None,
        filename_suffix: Optional[str] = None,
        signal_processing_options: Optional[NdeSignalProcessingOptions] = None,
    ) -> Tuple[bool, str]:
        """Export a local nnU-Net-style dataset with matching imagesTr/labelsTr."""
        try:
            context = self._resolve_export_context(
                nde_model=nde_model,
                annotation_model=annotation_model,
                nde_file=nde_file,
                signal_processing_options=signal_processing_options,
            )
            base_dir = Path(output_root) / f"{context.base_name}{context.processing_tag}_nnunet"
            images_tr = base_dir / "imagesTr"
            labels_tr = base_dir / "labelsTr"
            images_tr.mkdir(parents=True, exist_ok=True)
            labels_tr.mkdir(parents=True, exist_ok=True)

            prefix = filename_prefix or ""
            suffix = filename_suffix or ""

            primary_stats = self._export_nnunet_axis_dataset(
                image_volume=context.image_volume,
                mask_volume=context.mask_volume,
                images_dir=images_tr,
                labels_dir=labels_tr,
                prefix=prefix,
                suffix=suffix,
                axis_name=context.primary_axis_name,
                min_value=context.min_value,
                max_value=context.max_value,
            )

            secondary_image_volume = self._build_secondary_volume(context.image_volume)
            secondary_mask_volume = self._build_secondary_volume(context.mask_volume)
            secondary_stats = self._export_nnunet_axis_dataset(
                image_volume=secondary_image_volume,
                mask_volume=secondary_mask_volume,
                images_dir=images_tr,
                labels_dir=labels_tr,
                prefix=prefix,
                suffix=suffix,
                axis_name=context.secondary_axis_name,
                min_value=context.min_value,
                max_value=context.max_value,
            )
        except (ValueError, RuntimeError) as exc:
            return False, str(exc)

        summary = "\n".join(
            [
                "=== Resume ===",
                f"Signal utilise: {context.processing_description}",
                f"Dossier dataset: {base_dir}",
                f"imagesTr: {images_tr}",
                f"labelsTr: {labels_tr}",
                (
                    f"{context.primary_axis_name} ({primary_stats['axis_suffix']}): "
                    f"{primary_stats['image_count']} images, {primary_stats['label_count']} labels"
                ),
                (
                    f"{context.secondary_axis_name} ({secondary_stats['axis_suffix']}): "
                    f"{secondary_stats['image_count']} images, {secondary_stats['label_count']} labels"
                ),
            ]
        )
        return True, f"Export nnU-Net termine.\n{summary}"

    def _resolve_export_context(
        self,
        *,
        nde_model,
        annotation_model,
        nde_file: Optional[str],
        signal_processing_options: Optional[NdeSignalProcessingOptions],
    ) -> SplitExportContext:
        """Resolve all shared inputs required by the export flows."""
        if nde_model is None:
            raise ValueError("Aucun modele NDE charge.")

        mask_volume = getattr(annotation_model, "mask_volume", None)
        if mask_volume is None and hasattr(annotation_model, "get_mask_volume"):
            mask_volume = annotation_model.get_mask_volume()
        if mask_volume is None:
            raise ValueError("Aucun masque d'annotation present.")

        use_active_signal, processing_tag, processing_description = self._processing_metadata(
            signal_processing_options
        )

        image_volume = self._resolve_source_volume(
            nde_model=nde_model,
            use_active_signal=use_active_signal,
        )
        if image_volume is None:
            raise ValueError("Volume NDE introuvable.")

        image_volume = np.asarray(image_volume)
        mask_volume = np.asarray(mask_volume)
        if image_volume.ndim != 3 or mask_volume.ndim != 3:
            raise ValueError("Les exports de split exigent un volume 3D et un masque 3D.")
        if mask_volume.shape != image_volume.shape:
            raise ValueError(
                f"Shape masque {mask_volume.shape} different du volume {image_volume.shape}."
            )

        nde_path = nde_file or (getattr(nde_model, "metadata", {}) or {}).get("path")
        base_name = Path(str(nde_path)).stem if nde_path else "nde_export"
        min_value, max_value = self._resolve_min_max(
            nde_model=nde_model,
            image_volume=image_volume,
            use_active_signal=use_active_signal,
        )

        axis_order = list((getattr(nde_model, "metadata", {}) or {}).get("axis_order") or [])
        primary_axis_name = str(axis_order[0]) if len(axis_order) >= 1 else "UCoordinate"
        secondary_axis_name = str(axis_order[2]) if len(axis_order) >= 3 else "VCoordinate"

        return SplitExportContext(
            mask_volume=mask_volume,
            image_volume=image_volume,
            base_name=base_name,
            processing_tag=processing_tag,
            processing_description=processing_description,
            use_active_signal=use_active_signal,
            min_value=min_value,
            max_value=max_value,
            primary_axis_name=primary_axis_name,
            secondary_axis_name=secondary_axis_name,
        )

    def _resolve_source_volume(self, *, nde_model, use_active_signal: bool) -> Optional[np.ndarray]:
        """Return the raw source or processed volume used to build dataset images."""
        if use_active_signal and hasattr(nde_model, "get_active_raw_volume"):
            volume = nde_model.get_active_raw_volume()
            if volume is not None:
                return np.asarray(volume)

        volume = getattr(nde_model, "volume", None)
        if volume is not None:
            return np.asarray(volume)

        if hasattr(nde_model, "get_active_raw_volume"):
            volume = nde_model.get_active_raw_volume()
            if volume is not None:
                return np.asarray(volume)
        return None

    def _processing_metadata(
        self,
        signal_processing_options: Optional[NdeSignalProcessingOptions],
    ) -> tuple[bool, str, str]:
        """Return the folder suffix and human-readable signal description."""
        if signal_processing_options is None or signal_processing_options.is_passthrough():
            return False, "", "signal source (aucun traitement)"

        folder_parts = []
        label_parts = []
        if signal_processing_options.apply_hilbert:
            folder_parts.append("hilbert")
            label_parts.append("Hilbert")
        if signal_processing_options.apply_smoothing:
            folder_parts.append("lissage")
            label_parts.append("lissage")

        processing_tag = "_" + "+".join(folder_parts) if folder_parts else ""
        processing_description = " + ".join(label_parts) if label_parts else "signal source (aucun traitement)"
        return True, processing_tag, processing_description

    def _resolve_min_max(
        self,
        *,
        nde_model,
        image_volume: np.ndarray,
        use_active_signal: bool,
    ) -> tuple[float, float]:
        """Resolve the min/max used to normalize nnU-Net input images."""
        min_value = None
        max_value = None
        if use_active_signal and hasattr(nde_model, "get_active_min_max"):
            min_value, max_value = nde_model.get_active_min_max()
        if min_value is None or max_value is None:
            metadata = getattr(nde_model, "metadata", {}) or {}
            min_value = metadata.get("min_value", min_value)
            max_value = metadata.get("max_value", max_value)
        if min_value is None or max_value is None:
            min_value = float(np.min(image_volume))
            max_value = float(np.max(image_volume))
        return float(min_value), float(max_value)

    def _export_nnunet_axis_dataset(
        self,
        *,
        image_volume: np.ndarray,
        mask_volume: np.ndarray,
        images_dir: Path,
        labels_dir: Path,
        prefix: str,
        suffix: str,
        axis_name: str,
        min_value: float,
        max_value: float,
    ) -> dict[str, int | str]:
        """Write one axis orientation into matching imagesTr/labelsTr PNG files."""
        image_count = 0
        label_count = 0
        axis_suffix = self._axis_suffix(axis_name)
        depth = int(image_volume.shape[0])

        for idx in range(depth):
            position_filename = idx * 1500
            base_filename = f"endview_{position_filename:012d}"
            output_filename = self._build_axis_filename(
                base_filename=base_filename,
                prefix=prefix,
                suffix=suffix,
                axis_suffix=axis_suffix,
            )

            image_slice = self._normalize_to_uint8(
                image_volume[idx],
                min_value=min_value,
                max_value=max_value,
            )
            label_slice = np.asarray(mask_volume[idx], dtype=np.uint8)

            self._write_png(images_dir / output_filename, image_slice)
            self._write_png(labels_dir / output_filename, label_slice)

            image_count += 1
            label_count += 1
            if (idx + 1) % 100 == 0:
                self.logger.info("Export nnU-Net %s: %s/%s", axis_suffix, idx + 1, depth)

        return {
            "axis_suffix": axis_suffix,
            "image_count": image_count,
            "label_count": label_count,
        }

    def _normalize_to_uint8(
        self,
        image_slice: np.ndarray,
        *,
        min_value: float,
        max_value: float,
    ) -> np.ndarray:
        """Normalize an image slice to grayscale uint8."""
        image_array = np.asarray(image_slice, dtype=np.float32)
        if max_value == min_value:
            normalized = np.zeros_like(image_array, dtype=np.float32)
        else:
            normalized = (image_array - float(min_value)) / (float(max_value) - float(min_value))
            normalized = np.clip(normalized, 0.0, 1.0)
        return (normalized * 255.0).astype(np.uint8)

    def _build_secondary_volume(self, volume: np.ndarray) -> np.ndarray:
        """Return the orthogonal secondary view volume used in the UI."""
        return np.transpose(np.asarray(volume), (2, 1, 0))

    def _build_axis_filename(
        self,
        *,
        base_filename: str,
        prefix: str,
        suffix: str,
        axis_suffix: str,
    ) -> str:
        """Compose the output filename while preserving user prefix/suffix."""
        return f"{prefix}{base_filename}{suffix}_{axis_suffix}.png"

    def _axis_suffix(self, axis_name: str) -> str:
        """Map the axis name to the automatic filename suffix requested by the UI."""
        normalized = "".join(
            character.lower() if character.isalnum() else "_"
            for character in str(axis_name or "").strip()
        ).strip("_")
        if "ucoordinate" in normalized or normalized in {"ucoord", "u", "lengthwise"}:
            return "ucoord"
        if "vcoordinate" in normalized or normalized in {"vcoord", "v", "crosswise"}:
            return "vcoord"
        return normalized or "axis"

    def _write_png(self, path: Path, image: np.ndarray) -> None:
        """Write a PNG file and fail loudly when OpenCV refuses the path."""
        path.parent.mkdir(parents=True, exist_ok=True)
        success = cv2.imwrite(str(path), np.asarray(image))
        if not success:
            raise RuntimeError(f"Impossible d'ecrire le fichier PNG: {path}")

    def _safe_copy(self, src: Path, dst_dir: Path, dst_name: Optional[str] = None) -> None:
        """Copy a file when present, logging missing inputs."""
        if src.exists():
            dst_dir.mkdir(parents=True, exist_ok=True)
            target_name = dst_name or src.name
            shutil.copy2(src, dst_dir / target_name)
        else:
            self.logger.warning("Fichier manquant pour copie: %s", src)
