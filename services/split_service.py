"""
Service de split flaw/noflaw basé sur les endviews exportées et le masque courant.

Flux :
1) Export des endviews RGB/uint8 (complete) via EndviewExportService.
2) Parcours du mask volume courant (AnnotationModel) slice par slice :
   - bucket = flaw si un pixel non nul, sinon noflaw.
   - copie l'endview correspondante dans flaw/noflaw (RGB et uint8).
   - écrit le masque dans gtmask/flaw ou gtmask/noflaw.
3) Retourne un résumé des compteurs.
"""

from __future__ import annotations

import logging
import os
import shutil
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np

from services.endview_export import EndviewExportService


class SplitFlawNoflawService:
    """Orchestre export endviews + split flaw/noflaw sur base du masque courant."""

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
    ) -> Tuple[bool, str]:
        """Export endviews et sépare flaw/noflaw selon le masque courant.

        Args:
            nde_model: NdeModel déjà chargé (volume orienté).
            annotation_model: AnnotationModel contenant le mask volume actuel.
            nde_file: Chemin du .nde (pour nommer le dossier racine).
            output_root: Dossier parent où créer la structure d'export.
        """
        if nde_model is None:
            return False, "Aucun modèle NDE chargé."
        mask_volume = getattr(annotation_model, "mask_volume", None)
        if mask_volume is None:
            mask_volume = annotation_model.get_mask_volume()
        if mask_volume is None:
            return False, "Aucun masque d'annotation présent."

        volume = getattr(nde_model, "get_active_volume", lambda: None)()
        if volume is None:
            volume = getattr(nde_model, "volume", None)
        if volume is None:
            return False, "Volume NDE introuvable."

        if mask_volume.shape != volume.shape:
            return False, f"Shape masque {mask_volume.shape} différent du volume {volume.shape}."

        base_name = "nde_export"
        nde_path = nde_file or (getattr(nde_model, "metadata", {}) or {}).get("path")
        if nde_path:
            base_name = Path(str(nde_path)).stem

        base_dir = Path(output_root) / base_name
        rgb_complete = base_dir / "endviews_rgb24" / "complete"
        uint8_complete = base_dir / "endviews_uint8" / "complete"

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

        # Créer tous les dossiers cibles
        for fmt_targets in targets.values():
            for path in fmt_targets.values():
                path.mkdir(parents=True, exist_ok=True)
        rgb_complete.mkdir(parents=True, exist_ok=True)
        uint8_complete.mkdir(parents=True, exist_ok=True)

        # Export endviews (toujours, pour rester déterministe)
        self.logger.info("Export endviews RGB24 -> %s", rgb_complete)
        success_rgb, msg_rgb = self.endview_export.export_endviews(
            nde_file=nde_path,
            nde_model=nde_model,
            output_folder=str(rgb_complete),
            export_format="rgb",
        )
        if not success_rgb:
            return False, msg_rgb

        self.logger.info("Export endviews UINT8 -> %s", uint8_complete)
        success_uint8, msg_uint8 = self.endview_export.export_endviews(
            nde_file=nde_path,
            nde_model=nde_model,
            output_folder=str(uint8_complete),
            export_format="uint8",
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

        depth = mask_volume.shape[0]
        for idx in range(depth):
            mask_slice = np.asarray(mask_volume[idx], dtype=np.uint8)
            bucket = "flaw" if np.any(mask_slice != 0) else "noflaw"
            position_filename = idx * 1500
            filename = f"endview_{position_filename:012d}.png"

            # Écrire les masques
            cv2.imwrite(str(targets["rgb"][f"gtmask_{bucket}"] / filename), mask_slice)
            cv2.imwrite(str(targets["uint8"][f"gtmask_{bucket}"] / filename), mask_slice)
            stats[f"{bucket}_masks"] += 1

            # Copier les endviews complètes vers flaw/noflaw
            self._safe_copy(rgb_complete / filename, targets["rgb"][bucket])
            self._safe_copy(uint8_complete / filename, targets["uint8"][bucket])
            stats[f"{bucket}_rgb_images"] += int((rgb_complete / filename).exists())
            stats[f"{bucket}_uint8_images"] += int((uint8_complete / filename).exists())

            if (idx + 1) % 100 == 0:
                self.logger.info("Traitement slices: %s/%s", idx + 1, depth)

        summary = "\n=== Résumé ===\n" + "\n".join(f"{k}: {v}" for k, v in stats.items())
        return True, f"Split flaw/noflaw terminé.\n{summary}"

    def _safe_copy(self, src: Path, dst_dir: Path) -> None:
        """Copie un fichier s'il existe, log sinon."""
        if src.exists():
            dst_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst_dir / src.name)
        else:
            self.logger.warning("Fichier manquant pour copie: %s", src)
