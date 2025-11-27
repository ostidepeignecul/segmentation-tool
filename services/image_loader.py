# === model/image_loader.py ===
import os
import logging
from typing import List, Tuple


def load_images_from_folder(folder_path: str, extensions: Tuple[str, ...] = (".png", ".jpg", ".jpeg", ".bmp", ".tiff")) -> List[str]:
    """
    Charge la liste des images depuis un dossier.
    
    Args:
        folder_path: Chemin du dossier contenant les images
        extensions: Tuple des extensions de fichiers à inclure
        
    Returns:
        List[str]: Liste triée des chemins des images
        
    Raises:
        ValueError: Si le dossier n'existe pas ou est vide
    """
    logger = logging.getLogger(__name__)
    logger.info(f"Chargement des images depuis: {folder_path}")
    
    # Vérifier si le dossier existe
    if not os.path.exists(folder_path):
        error_msg = f"Le dossier n'existe pas: {folder_path}"
        logger.error(error_msg)
        raise ValueError(error_msg)
        
    # Vérifier si c'est un dossier
    if not os.path.isdir(folder_path):
        error_msg = f"Le chemin n'est pas un dossier: {folder_path}"
        logger.error(error_msg)
        raise ValueError(error_msg)
    
    # Lister les fichiers
    files = []
    for f in os.listdir(folder_path):
        if f.lower().endswith(extensions):
            full_path = os.path.join(folder_path, f)
            if os.path.isfile(full_path):
                files.append(full_path)
    
    # Vérifier si des fichiers ont été trouvés
    if not files:
        error_msg = f"Aucune image trouvée dans le dossier: {folder_path}"
        logger.error(error_msg)
        raise ValueError(error_msg)
    
    # Trier les fichiers
    files.sort()
    logger.info(f"Nombre d'images trouvées: {len(files)}")
    
    return files
