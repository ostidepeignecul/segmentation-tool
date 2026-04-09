"""Service minimal pour charger un volume de masques NPZ/NPY aligné au NDE."""

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

    def load(self, path: str, target_shape: Tuple[int, int, int]) -> np.ndarray:
        """Charge un overlay et retourne un volume uint8 (Z,H,W) aligné sur target_shape."""
        file_path = Path(path)
        if not file_path.exists():
            raise FileNotFoundError(f"Fichier introuvable: {file_path}")

        data = np.load(file_path, allow_pickle=False)
        if isinstance(data, np.lib.npyio.NpzFile):
            keys = list(data.keys())
            if not keys:
                raise ValueError("NPZ sans données utilisables.")
            selected_key = next(
                (key for key in ("mask", "arr_0") if key in keys),
                keys[0],
            )
            arr = data[selected_key]
        else:
            arr = data

        if getattr(arr, "ndim", 0) != 3:
            raise ValueError(f"Overlay attendu 3D, reçu {arr.ndim}D.")

        arr, _ = self._align_to_target_shape(arr, target_shape=target_shape)

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
        """Efface le masque chargé."""
        self.mask_volume = None
        self.logger.info("Overlay NPZ réinitialisé")

    def _align_to_target_shape(
        self,
        arr: np.ndarray,
        *,
        target_shape: Tuple[int, int, int],
    ) -> Tuple[np.ndarray, Tuple[int, int, int]]:
        """Align overlay shape to target volume with known axis permutations."""
        tgt_shape = tuple(int(x) for x in target_shape)
        arr_shape = tuple(arr.shape)
        if arr_shape == tgt_shape:
            return arr, arr_shape

        # (0,2,1): swap H/W (legacy tolerance)
        # (2,1,0): swap primary/secondary annotation axis (U <-> V)
        transpose_attempts = (
            ((0, 2, 1), "swap H/W"),
            ((2, 1, 0), "swap U/V"),
        )
        for perm, reason in transpose_attempts:
            candidate = np.transpose(arr, perm)
            candidate_shape = tuple(candidate.shape)
            if candidate_shape == tgt_shape:
                self.logger.info(
                    "Overlay shape %s vs volume %s: application transpose %s (%s).",
                    arr_shape,
                    tgt_shape,
                    perm,
                    reason,
                )
                return candidate, candidate_shape

        attempted = ", ".join(str(perm) for perm, _ in transpose_attempts)
        raise ValueError(
            f"Shape overlay {arr_shape} différent du volume {tgt_shape}. "
            f"Permutations testées: {attempted}."
        )
