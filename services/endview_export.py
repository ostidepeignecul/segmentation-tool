# === services/endview_export_service.py ===
"""
Service pour exporter les endviews depuis un fichier NDE avec options de transformation.
Toute la logique reste ici : chargement NDE, normalisation et génération d'images.
"""

from __future__ import annotations

import logging
import os
from typing import Optional, Tuple

import cv2
import numpy as np
from PIL import Image

from services.nde_loader import NdeLoader


class EndviewExportService:
    """Service d'exportation des endviews avec options de transformation."""

    def __init__(self, nde_loader: Optional[NdeLoader] = None):
        self.logger = logging.getLogger(__name__)
        self.nde_loader = nde_loader or NdeLoader()

    def export_endviews(
        self,
        nde_file: Optional[str],
        output_folder: str,
        group_idx: int = 1,
        nde_model=None,
        export_format: str = "rgb",  # 'rgb' ou 'uint8'
        apply_flip_horizontal: bool = False,
        apply_flip_vertical: bool = False,
        rotation_angle: int = 0,  # 0, 90, 180, 270
        apply_custom_transpose: bool = False,
    ) -> Tuple[bool, str]:
        """
        Exporte les endviews depuis un fichier NDE avec options de transformation.

        Args:
            nde_file: Chemin vers le fichier NDE (ignoré si nde_model est fourni)
            output_folder: Dossier où sauvegarder les endviews
            group_idx: Index du groupe à charger (commence à 1)
            nde_model: Modèle NDE déjà chargé (optionnel)
            export_format: Format d'export ('rgb' ou 'uint8')
            apply_flip_horizontal: Appliquer un flip horizontal (miroir gauche-droite)
            apply_flip_vertical: Appliquer un flip vertical (miroir haut-bas)
            rotation_angle: Angle de rotation supplémentaire (0, 90, 180, 270)
            apply_custom_transpose: Appliquer une transposition supplémentaire

        Returns:
            Tuple[bool, str]: (succès, message)
        """
        try:
            model = self._load_model(nde_model, nde_file, group_idx)
            if model is None:
                return False, "Impossible de charger le modèle NDE."

            data_array = getattr(model, "volume", None)
            if data_array is None:
                return False, "Volume NDE introuvable dans le modèle."

            if data_array.ndim < 2:
                return False, "Volume NDE invalide pour l'export."

            # Créer le dossier de sortie
            os.makedirs(output_folder, exist_ok=True)

            num_images = data_array.shape[0] if data_array.ndim >= 3 else 1
            self.logger.info("Export des endviews vers: %s", output_folder)
            self.logger.info("Format: %s", export_format.upper())
            self.logger.info(
                "Options: flip_h=%s, flip_v=%s, rotation=%s°, transpose=%s",
                apply_flip_horizontal,
                apply_flip_vertical,
                rotation_angle,
                apply_custom_transpose,
            )

            metadata = getattr(model, "metadata", {}) or {}
            min_value = metadata.get("min_value")
            max_value = metadata.get("max_value")
            if min_value is None or max_value is None:
                min_value = float(np.min(data_array))
                max_value = float(np.max(data_array))

            # Charger la colormap Omniscan si format RGB
            omniscan_cmap = None
            if export_format == "rgb":
                colormap_path = os.path.join(os.path.dirname(__file__), "..", "OmniScanColorMap.npy")
                if os.path.exists(colormap_path):
                    try:
                        from matplotlib.colors import ListedColormap

                        omniscan_colors = np.load(colormap_path)
                        omniscan_cmap = ListedColormap(omniscan_colors)
                        self.logger.info("Colormap Omniscan chargée")
                    except Exception as exc:  # pragma: no cover - best effort
                        self.logger.warning("Impossible de charger la colormap Omniscan: %s", exc)

            # Exporter chaque endview
            for idx in range(num_images):
                if data_array.ndim == 2:
                    img_data = data_array
                else:
                    img_data = data_array[idx]

                # Normaliser les données
                if max_value == min_value:
                    img_data_normalized = np.zeros_like(img_data, dtype=float)
                else:
                    img_data_normalized = (img_data - min_value) / (max_value - min_value)
                    img_data_normalized = np.clip(img_data_normalized, 0.0, 1.0)

                # Générer l'image selon le format
                if export_format == "rgb":
                    # Format RGB avec colormap
                    if omniscan_cmap is not None:
                        colored_data = omniscan_cmap(img_data_normalized)
                        if colored_data.shape[-1] == 4:
                            colored_data = colored_data[:, :, :3]
                        img_array = (colored_data * 255).astype(np.uint8)
                    else:
                        # Fallback: grayscale converti en RGB
                        gray = (img_data_normalized * 255).astype(np.uint8)
                        img_array = np.stack([gray, gray, gray], axis=-1)
                else:  # uint8
                    # Format uint8 grayscale
                    img_array = (img_data_normalized * 255).astype(np.uint8)

                # Appliquer les transformations personnalisées
                img_array = self._apply_transformations(
                    img_array,
                    flip_horizontal=apply_flip_horizontal,
                    flip_vertical=apply_flip_vertical,
                    rotation_angle=rotation_angle,
                    custom_transpose=apply_custom_transpose,
                )

                # Générer le nom de fichier
                position_filename = idx * 1500
                filename = f"endview_{position_filename:012d}.png"
                output_path = os.path.join(output_folder, filename)

                # Sauvegarder l'image
                if export_format == "rgb":
                    img_pil = Image.fromarray(img_array, mode="RGB")
                    img_pil.save(output_path)
                else:
                    cv2.imwrite(output_path, img_array)

                if (idx + 1) % 100 == 0:
                    self.logger.info("Exporté %s/%s endviews...", idx + 1, num_images)

            message = f"✓ {num_images} endviews exportées avec succès dans:\n{output_folder}"
            self.logger.info(message)
            return True, message

        except Exception as exc:  # pragma: no cover - log et remonte l'erreur
            error_msg = f"Erreur lors de l'export des endviews: {exc}"
            self.logger.error(error_msg)
            import traceback

            traceback.print_exc()
            return False, error_msg

    def _apply_transformations(
        self,
        img_array: np.ndarray,
        flip_horizontal: bool = False,
        flip_vertical: bool = False,
        rotation_angle: int = 0,
        custom_transpose: bool = False,
    ) -> np.ndarray:
        """
        Applique les transformations personnalisées à une image.
        """
        result = img_array.copy()

        # Flip horizontal (miroir gauche-droite)
        if flip_horizontal:
            result = np.fliplr(result)

        # Flip vertical (miroir haut-bas)
        if flip_vertical:
            result = np.flipud(result)

        # Transposition personnalisée
        if custom_transpose:
            if result.ndim == 3:
                # Image RGB: transposer seulement les 2 premières dimensions
                result = np.transpose(result, (1, 0, 2))
            else:
                # Image grayscale
                result = result.T

        # Rotation supplémentaire
        if rotation_angle == 90:
            result = np.rot90(result, k=1)
        elif rotation_angle == 180:
            result = np.rot90(result, k=2)
        elif rotation_angle == 270:
            result = np.rot90(result, k=3)

        return result

    def _load_model(self, nde_model, nde_file: Optional[str], group_idx: int):
        """Charge ou retourne le modèle NDE pour l’export."""
        if nde_model is not None:
            return nde_model
        if not nde_file:
            self.logger.error("Aucun chemin NDE fourni pour l'export.")
            return None
        if not os.path.exists(nde_file):
            self.logger.error("Le fichier NDE n'existe pas: %s", nde_file)
            return None
        return self.nde_loader.load(nde_file, group_idx)


def export_endviews_gui(
    nde_file: Optional[str],
    output_folder: str,
    group_idx: int = 1,
    nde_loader=None,
    nde_loader_service=None,
    nde_model=None,
    export_format: str = "rgb",
    flip_horizontal: bool = False,
    flip_vertical: bool = False,
    rotation_angle: int = 0,
    custom_transpose: bool = False,
) -> Tuple[bool, str]:
    """
    Fonction wrapper pour l'interface graphique.
    """
    loader = nde_loader or nde_loader_service
    service = EndviewExportService(nde_loader=loader)
    return service.export_endviews(
        nde_file=nde_file,
        output_folder=output_folder,
        group_idx=group_idx,
        nde_model=nde_model,
        export_format=export_format,
        apply_flip_horizontal=flip_horizontal,
        apply_flip_vertical=flip_vertical,
        rotation_angle=rotation_angle,
        apply_custom_transpose=custom_transpose,
    )
