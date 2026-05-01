"""Service d'export des overlays vers un fichier NPZ."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Sequence, Tuple

import numpy as np


class OverlayExport:
    """Sauvegarde un volume de masques 3D en NPZ pour l'app ou pour Sentinel."""

    FORMAT_VERSION = 2

    def __init__(self) -> None:
        self.logger = logging.getLogger(__name__)

    def save_npz(
        self,
        mask_volume: np.ndarray,
        destination: str,
        expected_shape: Optional[Sequence[int] | Tuple[int, int, int]] = None,
        *,
        primary_axis_name: Optional[str],
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
        self.logger.info("Overlay app sauvegarde: %s", path)
        return str(path)

    def save_sentinel_npz(
        self,
        mask_volume: np.ndarray,
        destination: str,
        expected_shape: Optional[Sequence[int] | Tuple[int, int, int]] = None,
        *,
        rotation_degrees: int = 0,
        rotation_axes: str = "",
        transpose_axes: str = "",
        output_suffix: str = "",
        mirror_horizontal: bool = False,
        mirror_vertical: bool = False,
        mirror_z: bool = False,
        strict_mode: bool = False,
    ) -> str:
        """Save a Sentinel-oriented compatibility NPZ using a transformed `arr_0` payload."""
        masks = self._validate_mask_volume(mask_volume, expected_shape=expected_shape)
        transformed = self._build_sentinel_payload(
            masks,
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
        self.logger.info("Overlay Sentinel sauvegarde: %s", path)
        return str(path)

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

    @staticmethod
    def _normalize_output_path(destination: str, *, suffix: str = "") -> Path:
        path = Path(destination)
        if path.suffix.lower() != ".npz":
            path = path.with_suffix(".npz")
        if suffix:
            path = path.with_name(f"{path.stem}{suffix}{path.suffix}")
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
