"""Service d'export des overlays vers un fichier NPZ."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Sequence, Tuple

import numpy as np


class OverlayExport:
    """Sauvegarde un volume de masques (Z,H,W) en NPZ compressé."""

    def __init__(self) -> None:
        self.logger = logging.getLogger(__name__)

    def save_npz(
        self,
        mask_volume: np.ndarray,
        destination: str,
        expected_shape: Optional[Sequence[int] | Tuple[int, int, int]] = None,
    ) -> str:
        """
        Sauvegarde le volume de masques dans un fichier .npz et retourne le chemin final.

        Args:
            mask_volume: tableau 3D (Z,H,W) des masques (sera casté en uint8).
            destination: chemin cible (l'extension .npz est ajoutée si absente).
            expected_shape: shape à valider contre mask_volume (optionnel).
        """
        if mask_volume is None:
            raise ValueError("Aucun volume de masque à sauvegarder.")

        masks = np.asarray(mask_volume, dtype=np.uint8)
        if masks.ndim != 3:
            raise ValueError(f"Volume de masque 3D attendu, reçu {masks.ndim}D.")
        if masks.size == 0:
            raise ValueError("Volume de masque vide.")

        if expected_shape is not None:
            tgt = tuple(int(x) for x in expected_shape)
            if masks.shape != tgt:
                raise ValueError(f"Shape overlay {masks.shape} différent du volume {tgt}.")

        path = Path(destination)
        if path.suffix.lower() != ".npz":
            path = path.with_suffix(".npz")

        np.savez_compressed(path, arr_0=masks)
        self.logger.info("Overlay sauvegardé: %s", path)
        return str(path)
