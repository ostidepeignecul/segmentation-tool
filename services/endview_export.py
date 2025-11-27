# === services/endview_export_service.py ===
"""
Service pour exporter les endviews depuis un fichier NDE avec options de transformation.
"""

import os
import logging
import numpy as np
from PIL import Image
from pathlib import Path
from typing import Tuple, Optional
import cv2


class EndviewExportService:
    """Service d'exportation des endviews avec options de transformation."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def export_endviews(
        self,
        nde_file: str,
        output_folder: str,
        group_idx: int = 1,
        nde_loader_service=None,
        export_format: str = 'rgb',  # 'rgb' ou 'uint8'
        apply_flip_horizontal: bool = False,
        apply_flip_vertical: bool = False,
        rotation_angle: int = 0,  # 0, 90, 180, 270
        apply_custom_transpose: bool = False
    ) -> Tuple[bool, str]:
        """
        Exporte les endviews depuis un fichier NDE avec options de transformation.
        
        Args:
            nde_file: Chemin vers le fichier NDE
            output_folder: Dossier où sauvegarder les endviews
            group_idx: Index du groupe à charger (commence à 1)
            nde_loader_service: Instance du service NdeLoaderService
            export_format: Format d'export ('rgb' ou 'uint8')
            apply_flip_horizontal: Appliquer un flip horizontal (miroir gauche-droite)
            apply_flip_vertical: Appliquer un flip vertical (miroir haut-bas)
            rotation_angle: Angle de rotation supplémentaire (0, 90, 180, 270)
            apply_custom_transpose: Appliquer une transposition supplémentaire
            
        Returns:
            Tuple[bool, str]: (succès, message)
        """
        try:
            if nde_loader_service is None:
                return False, "Service NDE loader non fourni"
            
            # Vérifier que le fichier NDE existe
            if not os.path.exists(nde_file):
                return False, f"Le fichier NDE n'existe pas: {nde_file}"
            
            # Créer le dossier de sortie
            os.makedirs(output_folder, exist_ok=True)
            
            self.logger.info(f"Export des endviews depuis: {nde_file}")
            self.logger.info(f"Format: {export_format.upper()}")
            self.logger.info(f"Options: flip_h={apply_flip_horizontal}, flip_v={apply_flip_vertical}, "
                           f"rotation={rotation_angle}°, transpose={apply_custom_transpose}")
            
            # Charger les données NDE
            nde_data = nde_loader_service.load_nde_data(nde_file, group_idx)
            data_array = nde_data['data_array']
            structure = nde_data.get('structure', 'public')
            
            # Détecter l'orientation optimale
            orientation_config = nde_loader_service.detect_optimal_orientation(data_array, structure)
            orientation = orientation_config['slice_orientation']
            transpose = orientation_config['transpose']
            num_images = orientation_config['num_images']
            
            self.logger.info(f"Exportation de {num_images} endviews...")
            
            # Charger la colormap Omniscan si format RGB
            omniscan_cmap = None
            if export_format == 'rgb':
                colormap_path = os.path.join(os.path.dirname(__file__), '..', 'OmniScanColorMap.npy')
                if os.path.exists(colormap_path):
                    try:
                        from matplotlib.colors import ListedColormap
                        omniscan_colors = np.load(colormap_path)
                        omniscan_cmap = ListedColormap(omniscan_colors)
                        self.logger.info("Colormap Omniscan chargée")
                    except Exception as e:
                        self.logger.warning(f"Impossible de charger la colormap Omniscan: {e}")
            
            # Exporter chaque endview
            for idx in range(num_images):
                # Extraire le slice
                img_data = nde_loader_service.extract_slice(data_array, idx, orientation)
                
                # Appliquer transpose automatique si nécessaire
                if transpose:
                    img_data = img_data.T
                
                # Normaliser les données
                if nde_data['max_value'] == nde_data['min_value']:
                    img_data_normalized = np.zeros_like(img_data, dtype=float)
                else:
                    img_data_normalized = (img_data - nde_data['min_value']) / (nde_data['max_value'] - nde_data['min_value'])
                    img_data_normalized = np.clip(img_data_normalized, 0.0, 1.0)
                
                # Générer l'image selon le format
                if export_format == 'rgb':
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
                
                # Appliquer rotation automatique si transpose n'a pas été appliqué
                if not transpose:
                    img_array = np.rot90(img_array, k=-1)
                
                # Appliquer les transformations personnalisées
                img_array = self._apply_transformations(
                    img_array,
                    flip_horizontal=apply_flip_horizontal,
                    flip_vertical=apply_flip_vertical,
                    rotation_angle=rotation_angle,
                    custom_transpose=apply_custom_transpose
                )
                
                # Générer le nom de fichier
                position_filename = idx * 1500
                filename = f"endview_{position_filename:012d}.png"
                output_path = os.path.join(output_folder, filename)
                
                # Sauvegarder l'image
                if export_format == 'rgb':
                    # RGB: utiliser PIL pour sauvegarder
                    img_pil = Image.fromarray(img_array, mode='RGB')
                    img_pil.save(output_path)
                else:
                    # uint8: utiliser OpenCV
                    cv2.imwrite(output_path, img_array)
                
                if (idx + 1) % 100 == 0:
                    self.logger.info(f"Exporté {idx + 1}/{num_images} endviews...")
            
            message = f"✓ {num_images} endviews exportées avec succès dans:\n{output_folder}"
            self.logger.info(message)
            return True, message
            
        except Exception as e:
            error_msg = f"Erreur lors de l'export des endviews: {str(e)}"
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
        custom_transpose: bool = False
    ) -> np.ndarray:
        """
        Applique les transformations personnalisées à une image.
        
        Args:
            img_array: Image à transformer (numpy array)
            flip_horizontal: Appliquer un flip horizontal
            flip_vertical: Appliquer un flip vertical
            rotation_angle: Angle de rotation (0, 90, 180, 270)
            custom_transpose: Appliquer une transposition
            
        Returns:
            np.ndarray: Image transformée
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


def export_endviews_gui(
    nde_file: str,
    output_folder: str,
    group_idx: int = 1,
    nde_loader_service=None,
    export_format: str = 'rgb',
    flip_horizontal: bool = False,
    flip_vertical: bool = False,
    rotation_angle: int = 0,
    custom_transpose: bool = False
) -> Tuple[bool, str]:
    """
    Fonction wrapper pour l'interface graphique.
    
    Args:
        nde_file: Chemin vers le fichier NDE
        output_folder: Dossier de sortie
        group_idx: Index du groupe NDE
        nde_loader_service: Service NDE loader
        export_format: Format d'export ('rgb' ou 'uint8')
        flip_horizontal: Appliquer un flip horizontal
        flip_vertical: Appliquer un flip vertical
        rotation_angle: Angle de rotation (0, 90, 180, 270)
        custom_transpose: Appliquer une transposition
        
    Returns:
        Tuple[bool, str]: (succès, message)
    """
    service = EndviewExportService()
    return service.export_endviews(
        nde_file=nde_file,
        output_folder=output_folder,
        group_idx=group_idx,
        nde_loader_service=nde_loader_service,
        export_format=export_format,
        apply_flip_horizontal=flip_horizontal,
        apply_flip_vertical=flip_vertical,
        rotation_angle=rotation_angle,
        apply_custom_transpose=custom_transpose
    )

