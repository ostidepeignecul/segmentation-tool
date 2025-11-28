"""Service pour charger et préparer un overlay NPZ/NPY aligné sur le volume NDE."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np

from config.constants import MASK_COLORS_BGRA
from services.npz_debug_logger import npz_debug_logger


class NPZOverlayService:
    """Gère le chargement et la préparation de l'overlay de masques."""

    def __init__(self) -> None:
        self.logger = logging.getLogger(__name__)
        self.mask_volume: Optional[np.ndarray] = None  # uint8 classes, shape (Z,H,W)
        self.overlay_rgba: Optional[np.ndarray] = None  # uint8 RGBA, shape (Z,H,W,4)

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def initialize_empty(self, shape: Tuple[int, int, int]) -> None:
        """Crée un volume de masques vide (zeros) aligné sur le NDE."""
        depth, height, width = shape
        self.mask_volume = np.zeros((int(depth), int(height), int(width)), dtype=np.uint8)
        self._build_rgba()

    def load(self, path: str, target_shape: Tuple[int, int, int]) -> None:
        """Charge un fichier NPZ/NPY et construit l'overlay RGBA aligné au NDE."""
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

        # Compatibilité numpy < 1.23 : utiliser np.array pour éviter l'argument copy= de asarray
        self.mask_volume = np.array(arr, dtype=np.uint8, copy=False)
        self._build_rgba()

        npz_debug_logger.log_npz_loading(
            npz_path=str(file_path),
            masks_shape=self.mask_volume.shape,
            num_slices=self.mask_volume.shape[0],
        )
        unique_classes = np.unique(self.mask_volume)
        npz_debug_logger.log_variable("unique_classes", unique_classes.tolist())

    def get_rgba_volume(self) -> Optional[np.ndarray]:
        """Retourne le volume RGBA prêt à afficher (ou None si absent)."""
        return self.overlay_rgba

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #
    def _build_rgba(self) -> None:
        """Construit le volume RGBA depuis mask_volume avec la palette BGRA."""
        if self.mask_volume is None:
            self.overlay_rgba = None
            return
        masks = self.mask_volume
        rgba = np.zeros((*masks.shape, 4), dtype=np.uint8)
        palette: Dict[int, Tuple[int, int, int, int]] = MASK_COLORS_BGRA
        for cls_value in np.unique(masks):
            if cls_value == 0:
                continue
            color = palette.get(int(cls_value))
            if color is None:
                # Couleur fallback magenta semi-transparente
                color = (255, 0, 255, 160)
            mask = masks == cls_value
            rgba[mask] = color
        self.overlay_rgba = rgba

    def clear(self) -> None:
        """Efface overlay et masques."""
        self.mask_volume = None
        self.overlay_rgba = None
        self.logger.info("Overlay NPZ réinitialisé")

