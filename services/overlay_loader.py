"""Service minimal pour charger un volume de masques NPZ/NPY aligne au NDE."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

from services.overlay_debug_logger import overlay_debug_logger


class OverlayLoader:
    """Charge un fichier NPZ/NPY et renvoie un volume de masques uint8 (Z,H,W)."""

    def __init__(self) -> None:
        self.logger = logging.getLogger(__name__)
        self.mask_volume: Optional[np.ndarray] = None

    def load(
        self,
        path: str,
        target_shape: Tuple[int, int, int],
        *,
        preferred_primary_axis: Optional[str] = None,
    ) -> np.ndarray:
        """Charge un overlay et retourne un volume uint8 (Z,H,W) aligne sur `target_shape`."""
        file_path = Path(path)
        if not file_path.exists():
            raise FileNotFoundError(f"Fichier introuvable: {file_path}")

        data = np.load(file_path, allow_pickle=False)
        explicit_orientation = False
        if isinstance(data, np.lib.npyio.NpzFile):
            try:
                keys = list(data.keys())
                if not keys:
                    raise ValueError("NPZ sans donnees utilisables.")
                arr, explicit_orientation = self._select_npz_array(
                    data,
                    keys=keys,
                    preferred_primary_axis=preferred_primary_axis,
                )
            finally:
                data.close()
        else:
            arr = data

        if getattr(arr, "ndim", 0) != 3:
            raise ValueError(f"Overlay attendu 3D, recu {arr.ndim}D.")

        arr, _ = self._align_to_target_shape(
            arr,
            target_shape=target_shape,
            explicit_orientation=explicit_orientation,
        )

        self.mask_volume = np.array(arr, dtype=np.uint8, copy=False)

        overlay_debug_logger.log_overlay_loading(
            overlay_path=str(file_path),
            masks_shape=self.mask_volume.shape,
            num_slices=self.mask_volume.shape[0],
        )
        unique_classes = np.unique(self.mask_volume)
        overlay_debug_logger.log_variable("unique_classes", unique_classes.tolist())

        return self.mask_volume

    def clear(self) -> None:
        """Efface le masque charge."""
        self.mask_volume = None
        self.logger.info("Overlay NPZ reinitialise")

    def _select_npz_array(
        self,
        data: np.lib.npyio.NpzFile,
        *,
        keys: list[str],
        preferred_primary_axis: Optional[str],
    ) -> tuple[np.ndarray, bool]:
        """Select the correct NPZ payload and report whether orientation is explicit."""
        has_ucoord = "mask_ucoord" in keys
        has_vcoord = "mask_vcoord" in keys
        if has_ucoord or has_vcoord:
            if not (has_ucoord and has_vcoord):
                raise ValueError(
                    "NPZ overlay nouveau format incomplet: `mask_ucoord` et `mask_vcoord` "
                    "doivent etre presents ensemble."
                )
            preferred = self._normalize_axis_name(preferred_primary_axis)
            if preferred == "vcoord":
                return np.asarray(data["mask_vcoord"]), True
            if preferred in {"ucoord", ""}:
                return np.asarray(data["mask_ucoord"]), True
            raise ValueError(
                "Impossible de choisir l'orientation du NPZ overlay: axe primaire courant "
                f"invalide ({preferred_primary_axis!r})."
            )

        selected_key = next((key for key in ("mask", "arr_0") if key in keys), keys[0])
        return np.asarray(data[selected_key]), False

    def _align_to_target_shape(
        self,
        arr: np.ndarray,
        *,
        target_shape: Tuple[int, int, int],
        explicit_orientation: bool,
    ) -> Tuple[np.ndarray, Tuple[int, int, int]]:
        """Align overlay shape to target volume with known axis permutations."""
        tgt_shape = tuple(int(x) for x in target_shape)
        arr_shape = tuple(arr.shape)
        if arr_shape == tgt_shape:
            return arr, arr_shape

        transpose_attempts = (
            ((0, 2, 1), "swap H/W"),
            ((2, 1, 0), "swap U/V"),
        )
        matches: list[tuple[np.ndarray, tuple[int, int, int], str]] = []
        for perm, reason in transpose_attempts:
            candidate = np.transpose(arr, perm)
            candidate_shape = tuple(candidate.shape)
            if candidate_shape == tgt_shape:
                matches.append((candidate, perm, reason))

        if not matches:
            attempted = ", ".join(str(perm) for perm, _ in transpose_attempts)
            raise ValueError(
                f"Shape overlay {arr_shape} different du volume {tgt_shape}. "
                f"Permutations testees: {attempted}."
            )

        candidate, perm, reason = matches[0]
        self.logger.info(
            "Overlay shape %s vs volume %s: application transpose %s (%s).",
            arr_shape,
            tgt_shape,
            perm,
            reason,
        )
        return candidate, tuple(candidate.shape)

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
