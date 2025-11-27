# === services/empty_mask_generator_service.py ===
"""
Service pour générer des masques vides à partir d'images PNG ou de fichiers NDE.
"""

import os
import logging
import numpy as np
from PIL import Image
from pathlib import Path
from typing import List, Tuple
import cv2


class EmptyMaskGenerator:
    """Générateur de masques vides pour images PNG ou endviews NDE."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def generate_empty_masks_from_pngs(self, png_folder: str, output_folder: str) -> Tuple[bool, str]:
        """
        Génère des masques vides à partir d'un dossier d'images PNG.
        
        Args:
            png_folder: Dossier contenant les images PNG sources
            output_folder: Dossier où sauvegarder les masques vides
            
        Returns:
            Tuple[bool, str]: (succès, message)
        """
        try:
            # Vérifier que le dossier source existe
            if not os.path.exists(png_folder):
                return False, f"Le dossier source n'existe pas: {png_folder}"
            
            # Créer le dossier de sortie
            os.makedirs(output_folder, exist_ok=True)
            
            # Lister tous les fichiers PNG
            png_files = sorted([
                f for f in os.listdir(png_folder)
                if f.lower().endswith('.png')
            ])
            
            if not png_files:
                return False, f"Aucun fichier PNG trouvé dans {png_folder}"
            
            self.logger.info(f"Génération de {len(png_files)} masques vides depuis PNG...")
            
            # Générer un masque vide pour chaque image
            skipped_count = 0
            generated_count = 0
            
            for idx, png_file in enumerate(png_files):
                # Vérifier si le masque existe déjà
                output_path = os.path.join(output_folder, png_file)
                if os.path.exists(output_path):
                    self.logger.info(f"Masque existe déjà: {png_file}, ignoré")
                    skipped_count += 1
                    continue
                
                # Charger l'image source pour obtenir ses dimensions
                source_path = os.path.join(png_folder, png_file)
                source_img = cv2.imread(source_path)
                
                if source_img is None:
                    self.logger.warning(f"Impossible de lire {png_file}, ignoré")
                    continue
                
                # Créer un masque vide (noir) avec les mêmes dimensions
                height, width = source_img.shape[:2]
                empty_mask = np.zeros((height, width), dtype=np.uint8)
                
                # Sauvegarder le masque avec le même nom
                cv2.imwrite(output_path, empty_mask)
                generated_count += 1
                
                if (idx + 1) % 100 == 0:
                    self.logger.info(f"Traité {idx + 1}/{len(png_files)} fichiers...")
            
            # Construire le message de résultat
            if skipped_count > 0:
                message = f"✓ {generated_count} nouveaux masques générés, {skipped_count} ignorés (déjà existants)\n{output_folder}"
            else:
                message = f"✓ {generated_count} masques vides générés avec succès dans:\n{output_folder}"
            self.logger.info(message)
            return True, message
            
        except Exception as e:
            error_msg = f"Erreur lors de la génération des masques: {str(e)}"
            self.logger.error(error_msg)
            return False, error_msg
    
    def generate_empty_masks_from_nde(
        self,
        nde_file: str,
        output_folder: str,
        group_idx: int = 1,
        nde_loader_service=None
    ) -> Tuple[bool, str]:
        """
        Génère des masques vides à partir d'un fichier NDE.
        
        Args:
            nde_file: Chemin vers le fichier NDE
            output_folder: Dossier où sauvegarder les masques vides
            group_idx: Index du groupe à charger (commence à 1)
            nde_loader_service: Instance du service NdeLoaderService
            
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
            
            self.logger.info(f"Chargement du fichier NDE: {nde_file}")
            
            # Charger les données NDE
            nde_data = nde_loader_service.load_nde_data(nde_file, group_idx)
            data_array = nde_data['data_array']
            structure = nde_data.get('structure', 'public')
            
            # Détecter l'orientation optimale
            orientation_config = nde_loader_service.detect_optimal_orientation(data_array, structure)
            orientation = orientation_config['slice_orientation']
            transpose = orientation_config['transpose']
            num_images = orientation_config['num_images']
            
            self.logger.info(f"Génération de {num_images} masques vides depuis NDE...")
            
            # Générer un masque vide pour chaque endview
            skipped_count = 0
            generated_count = 0
            
            for idx in range(num_images):
                # Générer le nom de fichier (même convention que les endviews)
                position_filename = idx * 1500
                filename = f"endview_{position_filename:012d}.png"
                output_path = os.path.join(output_folder, filename)
                
                # Vérifier si le masque existe déjà
                if os.path.exists(output_path):
                    self.logger.info(f"Masque existe déjà: {filename}, ignoré")
                    skipped_count += 1
                    continue
                
                # Extraire le slice selon l'orientation
                if orientation == 'lengthwise':
                    img_data = data_array[idx, :, :]
                elif orientation == 'crosswise':
                    img_data = data_array[:, idx, :]
                else:  # ultrasound
                    img_data = data_array[:, :, idx]
                
                # Appliquer la transposition si nécessaire
                if transpose:
                    img_data = img_data.T

                # Créer un masque vide avec les mêmes dimensions (AVANT rotation)
                height, width = img_data.shape
                empty_mask = np.zeros((height, width), dtype=np.uint8)

                # IMPORTANT: Appliquer rotation UNIQUEMENT si transpose n'a PAS été appliqué
                # Cela doit correspondre exactement au traitement dans load_nde_as_memory_images
                # - Si transpose=False (image déjà plus large que haute) → rotation -90° nécessaire
                # - Si transpose=True (image corrigée par transpose) → PAS de rotation
                if not transpose:
                    # Rotation -90° (équivalent à np.rot90(img, k=-1))
                    empty_mask = cv2.rotate(empty_mask, cv2.ROTATE_90_CLOCKWISE)
                
                # Sauvegarder le masque
                cv2.imwrite(output_path, empty_mask)
                generated_count += 1
                
                if (idx + 1) % 100 == 0:
                    self.logger.info(f"Traité {idx + 1}/{num_images} fichiers...")
            
            # Construire le message de résultat
            if skipped_count > 0:
                message = f"✓ {generated_count} nouveaux masques générés, {skipped_count} ignorés (déjà existants)\n{output_folder}"
            else:
                message = f"✓ {generated_count} masques vides générés avec succès dans:\n{output_folder}"
            self.logger.info(message)
            return True, message
            
        except Exception as e:
            error_msg = f"Erreur lors de la génération des masques depuis NDE: {str(e)}"
            self.logger.error(error_msg)
            import traceback
            traceback.print_exc()
            return False, error_msg


def generate_empty_masks_gui(
    source_type: str,
    source_path: str,
    output_folder: str,
    group_idx: int = 1,
    nde_loader_service=None
) -> Tuple[bool, str]:
    """
    Fonction wrapper pour l'interface graphique.
    
    Args:
        source_type: Type de source ('png' ou 'nde')
        source_path: Chemin vers le dossier PNG ou le fichier NDE
        output_folder: Dossier de sortie pour les masques
        group_idx: Index du groupe NDE (si applicable)
        nde_loader_service: Service NDE loader (si applicable)
        
    Returns:
        Tuple[bool, str]: (succès, message)
    """
    generator = EmptyMaskGenerator()
    
    if source_type == 'png':
        return generator.generate_empty_masks_from_pngs(source_path, output_folder)
    elif source_type == 'nde':
        return generator.generate_empty_masks_from_nde(
            source_path,
            output_folder,
            group_idx,
            nde_loader_service
        )
    else:
        return False, f"Type de source invalide: {source_type}"


def generate_empty_masks_gui_complete(view, current_nde_file, current_folder, model, nde_loader_service, logger=None):
    """
    Gère complètement la génération de masques vides à partir du dossier PNG courant ou du fichier NDE courant.
    Inclut tous les dialogues utilisateur, validations et gestion d'erreurs.
    
    Args:
        view: Instance de la vue avec méthodes de dialogue (select_directory, show_critical, etc.)
        current_nde_file: Chemin vers le fichier NDE courant (peut être None)
        current_folder: Chemin vers le dossier PNG courant (peut être None)
        model: Instance du modèle d'annotation (pour obtenir image_list)
        nde_loader_service: Instance du service NDE loader
        logger: Logger optionnel pour les logs
        
    Returns:
        None
    """
    if logger is None:
        logger = logging.getLogger(__name__)
    
    try:
        # Déterminer la source (PNG ou NDE)
        source_type = None
        source_path = None

        if current_nde_file is not None:
            # Mode NDE
            source_type = 'nde'
            source_path = current_nde_file
            source_name = os.path.basename(current_nde_file)
        elif current_folder is not None:
            # Mode PNG
            source_type = 'png'
            source_path = current_folder
            source_name = os.path.basename(current_folder)
        else:
            view.show_warning(
                "Aucune source",
                "Veuillez d'abord ouvrir un dossier PNG ou un fichier NDE."
            )
            return

        # Demander le dossier de sortie
        output_folder = view.select_directory(
            "Sélectionner le dossier de sortie pour les masques vides",
            ""
        )

        if not output_folder:
            return

        # Afficher les informations et demander confirmation
        if source_type == 'png':
            num_images = len(model.image_list)
            info_msg = (
                f"Génération de masques vides :\n\n"
                f"• Source : Dossier PNG\n"
                f"• Dossier : {source_name}\n"
                f"• Nombre d'images : {num_images}\n"
                f"• Sortie : {output_folder}\n\n"
                f"Continuer ?"
            )
        else:  # nde
            # Compter les endviews
            nde_data = nde_loader_service.load_nde_data(source_path, group_idx=1)
            data_array = nde_data['data_array']
            structure = nde_data.get('structure', 'public')
            orientation_config = nde_loader_service.detect_optimal_orientation(data_array, structure)
            num_images = orientation_config['num_images']

            info_msg = (
                f"Génération de masques vides :\n\n"
                f"• Source : Fichier NDE\n"
                f"• Fichier : {source_name}\n"
                f"• Nombre d'endviews : {num_images}\n"
                f"• Sortie : {output_folder}\n\n"
                f"Continuer ?"
            )

        if not view.ask_question("Confirmer la génération", info_msg, default_yes=True):
            return

        # Générer les masques vides
        success, message = generate_empty_masks_gui(
            source_type=source_type,
            source_path=source_path,
            output_folder=output_folder,
            group_idx=1,
            nde_loader_service=nde_loader_service if source_type == 'nde' else None
        )

        if success:
            view.show_information("Succès", message)
            logger.info(f"Masques vides générés avec succès : {output_folder}")
        else:
            view.show_critical("Erreur", message)
            logger.error(f"Échec de la génération des masques vides : {message}")

    except Exception as e:
        error_msg = f"Erreur lors de la génération des masques vides : {str(e)}"
        view.show_critical("Erreur", error_msg)
        logger.error(error_msg)
        import traceback
        traceback.print_exc()