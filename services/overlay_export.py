"""Service d'export des overlays vers un fichier NPZ."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Mapping, Optional, Sequence, Tuple

import numpy as np

from config.constants import (
    LABEL_DISPLAY_NAMES,
    MASK_COLORS_BGRA,
    USER_LABEL_START,
    format_label_text,
)
from utils.filename_utils import sanitize_filename_component


class OverlayExport:
    """Sauvegarde un volume de masques 3D en NPZ pour l'app ou pour Sentinel."""

    FORMAT_VERSION = 2
    LABEL_MANIFEST_VERSION = 1

    def __init__(self) -> None:
        self.logger = logging.getLogger(__name__)

    def save_npz(
        self,
        mask_volume: np.ndarray,
        destination: str,
        expected_shape: Optional[Sequence[int] | Tuple[int, int, int]] = None,
        *,
        primary_axis_name: Optional[str],
        label_palette: Optional[Mapping[int, Sequence[int]]] = None,
        label_visibility: Optional[Mapping[int, bool]] = None,
    ) -> str:
        """
        Save the standard app format containing both annotation orientations.

        The current mask volume is assumed to match the current primary annotation axis, and
        the orthogonal orientation is derived by the standard `(2, 1, 0)` transpose used by
        the rest of the application.
        """
        masks = self._validate_mask_volume(mask_volume, expected_shape=expected_shape)
        normalized_primary = self._normalize_axis_name(primary_axis_name)
        if normalized_primary not in {"ucoord", "vcoord"}:
            raise ValueError(
                "Impossible d'exporter l'overlay: axe primaire courant inconnu "
                f"({primary_axis_name!r})."
            )

        secondary = np.transpose(masks, (2, 1, 0))
        if normalized_primary == "ucoord":
            mask_ucoord = masks
            mask_vcoord = secondary
        else:
            mask_ucoord = secondary
            mask_vcoord = masks

        path = self._normalize_output_path(destination)
        np.savez_compressed(
            path,
            format_version=np.asarray(self.FORMAT_VERSION, dtype=np.int16),
            mask_ucoord=np.asarray(mask_ucoord, dtype=np.uint8),
            mask_vcoord=np.asarray(mask_vcoord, dtype=np.uint8),
        )
        self.save_label_manifest(
            path,
            masks,
            export_target="app",
            primary_axis_name=normalized_primary,
            label_palette=label_palette,
            label_visibility=label_visibility,
        )
        self.logger.info("Overlay app sauvegarde: %s", path)
        return str(path)

    def save_sentinel_npz(
        self,
        mask_volume: np.ndarray,
        destination: str,
        expected_shape: Optional[Sequence[int] | Tuple[int, int, int]] = None,
        *,
        primary_axis_name: Optional[str],
        sentinel_source_view: str = "dscan",
        rotation_degrees: int = 0,
        rotation_axes: str = "",
        transpose_axes: str = "",
        output_suffix: str = "",
        mirror_horizontal: bool = False,
        mirror_vertical: bool = False,
        mirror_z: bool = False,
        strict_mode: bool = False,
        label_palette: Optional[Mapping[int, Sequence[int]]] = None,
        label_visibility: Optional[Mapping[int, bool]] = None,
    ) -> str:
        """Save a Sentinel NPZ from an explicit B-Scan/D-Scan source orientation."""
        masks = self._validate_mask_volume(mask_volume, expected_shape=expected_shape)
        sentinel_source_masks = self._select_sentinel_source_masks(
            masks,
            primary_axis_name=primary_axis_name,
            sentinel_source_view=sentinel_source_view,
        )
        transformed = self._build_sentinel_payload(
            sentinel_source_masks,
            rotation_degrees=rotation_degrees,
            rotation_axes=rotation_axes,
            transpose_axes=transpose_axes,
            mirror_horizontal=mirror_horizontal,
            mirror_vertical=mirror_vertical,
            mirror_z=mirror_z,
            strict_mode=strict_mode,
        )
        path = self._normalize_output_path(destination, suffix=output_suffix)
        np.savez_compressed(path, arr_0=np.asarray(transformed, dtype=np.uint8))
        self.save_label_manifest(
            path,
            transformed,
            export_target="sentinel",
            primary_axis_name=primary_axis_name,
            label_palette=label_palette,
            label_visibility=label_visibility,
        )
        self.logger.info("Overlay Sentinel sauvegarde: %s", path)
        return str(path)

    def save_label_manifest(
        self,
        npz_path: Path,
        mask_volume: np.ndarray,
        *,
        export_target: str,
        primary_axis_name: Optional[str],
        label_palette: Optional[Mapping[int, Sequence[int]]],
        label_visibility: Optional[Mapping[int, bool]],
        label_names: Optional[Mapping[int, str]] = None,
    ) -> Path:
        """Write the JSON label sidecar next to the exported NPZ."""
        manifest_path = self._label_manifest_path(npz_path)
        payload = self._build_label_manifest(
            npz_path=npz_path,
            mask_volume=mask_volume,
            export_target=export_target,
            primary_axis_name=primary_axis_name,
            label_palette=label_palette,
            label_visibility=label_visibility,
            label_names=label_names,
        )
        manifest_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        self.logger.info("Overlay label manifest sauvegarde: %s", manifest_path)
        return manifest_path

    def _build_label_manifest(
        self,
        *,
        npz_path: Path,
        mask_volume: np.ndarray,
        export_target: str,
        primary_axis_name: Optional[str],
        label_palette: Optional[Mapping[int, Sequence[int]]],
        label_visibility: Optional[Mapping[int, bool]],
        label_names: Optional[Mapping[int, str]],
    ) -> dict[str, object]:
        present_labels = {
            int(value)
            for value in np.unique(np.asarray(mask_volume, dtype=np.uint8)).tolist()
        }
        palette = self._normalize_label_palette(label_palette)
        visibility = self._normalize_label_visibility(label_visibility)
        names = self._normalize_label_names(label_names)

        class_ids = set(present_labels)
        class_ids.update(palette.keys())
        class_ids.update(names.keys())

        classes: dict[str, dict[str, object]] = {}
        for label_id in sorted(class_ids):
            label_name = names.get(int(label_id), self._label_name(label_id))
            entry: dict[str, object] = {
                "id": int(label_id),
                "name": label_name,
                "definition": label_name,
                "display_name": (
                    f"{label_name} ({int(label_id)})"
                    if int(label_id) in names
                    else format_label_text(int(label_id))
                ),
                "present_in_mask": int(label_id) in present_labels,
            }
            color = palette.get(int(label_id))
            if color is None:
                color = self._normalize_color(MASK_COLORS_BGRA.get(int(label_id)))
            if color is not None:
                entry["color_bgra"] = color
            if int(label_id) in visibility:
                entry["visible"] = bool(visibility[int(label_id)])
            classes[str(int(label_id))] = entry

        return {
            "format": "segmentation-tool-labels",
            "format_version": self.LABEL_MANIFEST_VERSION,
            "npz_file": npz_path.name,
            "export_target": str(export_target),
            "primary_axis": self._normalize_axis_name(primary_axis_name),
            "present_classes": sorted(present_labels),
            "classes": classes,
        }

    @staticmethod
    def _label_manifest_path(npz_path: Path) -> Path:
        return npz_path.with_name(f"{npz_path.stem}_labels.json")

    @staticmethod
    def _normalize_label_palette(
        label_palette: Optional[Mapping[int, Sequence[int]]],
    ) -> dict[int, list[int]]:
        palette: dict[int, list[int]] = {}
        for raw_label, raw_color in dict(label_palette or {}).items():
            try:
                label_id = int(raw_label)
            except Exception:
                continue
            color = OverlayExport._normalize_color(raw_color)
            if color is not None:
                palette[label_id] = color
        return palette

    @staticmethod
    def _normalize_label_visibility(
        label_visibility: Optional[Mapping[int, bool]],
    ) -> dict[int, bool]:
        visibility: dict[int, bool] = {}
        for raw_label, raw_visible in dict(label_visibility or {}).items():
            try:
                label_id = int(raw_label)
            except Exception:
                continue
            visibility[label_id] = bool(raw_visible)
        return visibility

    @staticmethod
    def _normalize_label_names(
        label_names: Optional[Mapping[int, str]],
    ) -> dict[int, str]:
        names: dict[int, str] = {}
        for raw_label, raw_name in dict(label_names or {}).items():
            try:
                label_id = int(raw_label)
            except Exception:
                continue
            name = str(raw_name or "").strip()
            if name:
                names[label_id] = name
        return names

    @staticmethod
    def label_names_from_mapping(
        labels_mapping: Optional[Mapping[object, object]],
    ) -> dict[int, str]:
        """Extract class id to label name from inference-style labels mappings."""
        if not isinstance(labels_mapping, Mapping):
            return {}
        raw_labels = labels_mapping.get("labels", {})
        if not isinstance(raw_labels, Mapping):
            return {}

        names: dict[int, str] = {}
        for raw_name, raw_label in raw_labels.items():
            try:
                label_id = int(raw_label)
                name = str(raw_name or "").strip()
            except Exception:
                try:
                    label_id = int(raw_name)
                    name = str(raw_label or "").strip()
                except Exception:
                    continue
            if name:
                names[label_id] = name
        return names

    @staticmethod
    def _normalize_color(raw_color: Optional[Sequence[int]]) -> Optional[list[int]]:
        if raw_color is None:
            return None
        try:
            return [int(channel) for channel in list(raw_color)]
        except Exception:
            return None

    @staticmethod
    def _label_name(label_id: int) -> str:
        label = int(label_id)
        display_name = LABEL_DISPLAY_NAMES.get(label)
        if display_name is not None:
            return str(display_name)
        if label >= int(USER_LABEL_START):
            user_index = label - int(USER_LABEL_START) + 1
            return f"BW echo {user_index}"
        return f"Label {label}"

    def _build_sentinel_payload(
        self,
        masks: np.ndarray,
        *,
        rotation_degrees: int,
        rotation_axes: str,
        transpose_axes: str,
        mirror_horizontal: bool,
        mirror_vertical: bool,
        mirror_z: bool,
        strict_mode: bool,
    ) -> np.ndarray:
        """Apply the Sentinel transform chain on a uint8 3D mask volume."""
        result = np.asarray(masks, dtype=np.uint8)

        transpose_perm = self._parse_transpose_axes(
            transpose_axes,
            ndim=result.ndim,
            strict_mode=strict_mode,
        )
        if transpose_perm is not None:
            result = np.transpose(result, axes=transpose_perm)

        rot = int(rotation_degrees) % 360
        if rot not in (0, 90, 180, 270):
            raise ValueError(f"Rotation non supportee: {rotation_degrees} (attendu 0/90/180/270).")
        if rot:
            axes = self._parse_rotation_axes(
                rotation_axes,
                ndim=result.ndim,
                strict_mode=strict_mode,
            )
            if axes is None:
                axes = (result.ndim - 2, result.ndim - 1)
            result = np.rot90(result, k=rot // 90, axes=axes)

        if mirror_horizontal:
            result = np.flip(result, axis=1)
        if mirror_vertical:
            result = np.flip(result, axis=2)
        if mirror_z:
            result = np.flip(result, axis=0)

        return np.asarray(result, dtype=np.uint8)

    @staticmethod
    def _validate_mask_volume(
        mask_volume: np.ndarray,
        *,
        expected_shape: Optional[Sequence[int] | Tuple[int, int, int]],
    ) -> np.ndarray:
        """Validate and normalize an input mask volume to uint8 `(Z,H,W)`."""
        if mask_volume is None:
            raise ValueError("Aucun volume de masque a sauvegarder.")

        masks = np.asarray(mask_volume, dtype=np.uint8)
        if masks.ndim != 3:
            raise ValueError(f"Volume de masque 3D attendu, recu {masks.ndim}D.")
        if masks.size == 0:
            raise ValueError("Volume de masque vide.")

        if expected_shape is not None:
            tgt = tuple(int(x) for x in expected_shape)
            if masks.shape != tgt:
                raise ValueError(f"Shape overlay {masks.shape} different du volume {tgt}.")
        return masks

    @staticmethod
    def _normalize_axis_name(axis_name: Optional[str]) -> str:
        normalized = "".join(
            character.lower() if character.isalnum() else ""
            for character in str(axis_name or "").strip()
        )
        if normalized in {"ucoordinate", "ucoord", "u", "lengthwise"}:
            return "ucoord"
        if normalized in {"vcoordinate", "vcoord", "v", "crosswise"}:
            return "vcoord"
        return normalized

    @classmethod
    def suggested_sentinel_source_view(cls, primary_axis_name: Optional[str]) -> str:
        """Map the current primary annotation axis to the corresponding B/D-Scan label."""
        normalized_primary = cls._normalize_axis_name(primary_axis_name)
        if normalized_primary == "vcoord":
            return "bscan"
        return "dscan"

    @classmethod
    def _normalize_sentinel_source_view(cls, sentinel_source_view: Optional[str]) -> str:
        normalized = "".join(
            character.lower() if character.isalnum() else ""
            for character in str(sentinel_source_view or "").strip()
        )
        if normalized in {"bscan", "b", "vcoordinate", "vcoord", "v", "crosswise"}:
            return "bscan"
        if normalized in {"dscan", "d", "ucoordinate", "ucoord", "u", "lengthwise"}:
            return "dscan"
        return normalized

    @staticmethod
    def _sentinel_source_view_to_primary_axis(sentinel_source_view: str) -> str:
        if sentinel_source_view == "bscan":
            return "vcoord"
        if sentinel_source_view == "dscan":
            return "ucoord"
        raise ValueError(f"Vue source Sentinel invalide: {sentinel_source_view!r}.")

    def _select_sentinel_source_masks(
        self,
        masks: np.ndarray,
        *,
        primary_axis_name: Optional[str],
        sentinel_source_view: str,
    ) -> np.ndarray:
        """
        Resolve the Sentinel source orientation explicitly instead of using the open view.

        The current mask volume is stored in the current primary annotation orientation. When
        the user selects the other endview, the same `(2, 1, 0)` transpose used across the app
        is applied before the Sentinel transform chain.
        """
        normalized_primary = self._normalize_axis_name(primary_axis_name)
        if normalized_primary not in {"ucoord", "vcoord"}:
            raise ValueError(
                "Impossible de resoudre l'orientation source Sentinel: axe primaire courant "
                f"inconnu ({primary_axis_name!r})."
            )

        normalized_source_view = self._normalize_sentinel_source_view(sentinel_source_view)
        if normalized_source_view not in {"bscan", "dscan"}:
            raise ValueError(
                "Impossible de resoudre l'orientation source Sentinel: vue demandee "
                f"invalide ({sentinel_source_view!r})."
            )

        desired_primary = self._sentinel_source_view_to_primary_axis(normalized_source_view)
        if desired_primary == normalized_primary:
            resolved = np.asarray(masks, dtype=np.uint8)
        else:
            resolved = np.transpose(masks, (2, 1, 0))

        self.logger.info(
            "Overlay Sentinel: source=%s, current_primary=%s, resolved_shape=%s",
            normalized_source_view,
            normalized_primary,
            tuple(resolved.shape),
        )
        return np.asarray(resolved, dtype=np.uint8)

    @staticmethod
    def _normalize_output_path(destination: str, *, suffix: str = "") -> Path:
        path = Path(destination)
        if path.suffix.lower() != ".npz":
            path = path.with_suffix(".npz")
        if suffix:
            normalized_suffix = sanitize_filename_component(suffix)
            path = path.with_name(f"{path.stem}{normalized_suffix}{path.suffix}")
        return path

    def _parse_transpose_axes(
        self,
        raw_value: str,
        *,
        ndim: int,
        strict_mode: bool,
    ) -> Optional[Tuple[int, ...]]:
        """Parse a transpose permutation string like `0,2,1`."""
        text = str(raw_value or "").strip()
        if not text:
            return None
        try:
            values = tuple(int(part.strip()) for part in text.split(","))
        except Exception as exc:
            return self._on_transform_parse_error(
                f"Transpose invalide: {raw_value!r}",
                strict_mode=strict_mode,
                exc=exc,
            )
        if len(values) != ndim or sorted(values) != list(range(ndim)):
            return self._on_transform_parse_error(
                f"Transpose invalide: {raw_value!r} (permutation attendue de 0 a {ndim - 1}).",
                strict_mode=strict_mode,
            )
        return values

    def _parse_rotation_axes(
        self,
        raw_value: str,
        *,
        ndim: int,
        strict_mode: bool,
    ) -> Optional[Tuple[int, int]]:
        """Parse rotation axes like `-2,-1` or `1,2`."""
        text = str(raw_value or "").strip()
        if not text:
            return None
        try:
            values = [int(part.strip()) for part in text.split(",")]
        except Exception as exc:
            return self._on_transform_parse_error(
                f"Axes rotation invalides: {raw_value!r}",
                strict_mode=strict_mode,
                exc=exc,
            )
        if len(values) != 2:
            return self._on_transform_parse_error(
                f"Axes rotation invalides: {raw_value!r} (2 valeurs attendues).",
                strict_mode=strict_mode,
            )

        normalized: list[int] = []
        for axis in values:
            resolved = axis + ndim if axis < 0 else axis
            if resolved < 0 or resolved >= ndim:
                return self._on_transform_parse_error(
                    f"Axe rotation hors borne: {axis} pour ndim={ndim}.",
                    strict_mode=strict_mode,
                )
            normalized.append(resolved)
        if normalized[0] == normalized[1]:
            return self._on_transform_parse_error(
                f"Axes rotation invalides: {raw_value!r} (axes distincts requis).",
                strict_mode=strict_mode,
            )
        return int(normalized[0]), int(normalized[1])

    def _on_transform_parse_error(
        self,
        message: str,
        *,
        strict_mode: bool,
        exc: Optional[Exception] = None,
    ) -> Optional[Tuple[int, ...]]:
        """Raise or log transform parse errors depending on strict mode."""
        if strict_mode:
            if exc is not None:
                raise ValueError(message) from exc
            raise ValueError(message)
        self.logger.warning("%s", message, exc_info=exc is not None)
        return None
