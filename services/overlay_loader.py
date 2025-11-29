"""Service minimal pour charger un volume de masques NPZ/NPY aligné au NDE."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

from services.npz_debug_logger import npz_debug_logger


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
            arr = data[keys[0]]
        else:
            arr = data

        if getattr(arr, "ndim", 0) != 3:
            raise ValueError(f"Overlay attendu 3D, reçu {arr.ndim}D.")

        arr_shape = tuple(arr.shape)
        tgt_shape = tuple(target_shape)
        if arr_shape != tgt_shape:
            # Tolérer un swap H/W (axes 1 et 2 inversés) si profondeur identique
            if arr_shape[0] == tgt_shape[0] and arr_shape[1] == tgt_shape[2] and arr_shape[2] == tgt_shape[1]:
                self.logger.info(
                    "Overlay shape %s inversé (H/W) vs volume %s, application d'un transpose (0,2,1).",
                    arr_shape,
                    tgt_shape,
                )
                arr = np.transpose(arr, (0, 2, 1))
                arr_shape = tuple(arr.shape)
            else:
                raise ValueError(f"Shape overlay {arr_shape} différent du volume {tgt_shape}.")

        self.mask_volume = np.array(arr, dtype=np.uint8, copy=False)

        npz_debug_logger.log_npz_loading(
            npz_path=str(file_path),
            masks_shape=self.mask_volume.shape,
            num_slices=self.mask_volume.shape[0],
        )
        unique_classes = np.unique(self.mask_volume)
        npz_debug_logger.log_variable("unique_classes", unique_classes.tolist())

        return self.mask_volume

    def clear(self) -> None:
        """Efface le masque chargé."""
        self.mask_volume = None
        self.logger.info("Overlay NPZ réinitialisé")
