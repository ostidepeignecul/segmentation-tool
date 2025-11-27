"""
Service simple pour charger et afficher un overlay NPZ.
"""
import numpy as np
import logging
from pathlib import Path
from typing import Optional, Tuple, List
from config.constants import MASK_COLORS_BGR
from services.npz_debug_logger import npz_debug_logger


class NPZOverlayService:
    """Service pour gérer l'overlay NPZ."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.volume = None  # Volume NPZ chargé (N, H, W)
        self.alpha = 0.5  # Transparence
    
    def load_npz(self, npz_path: str, array_key: str = 'arr_0') -> bool:
        """
        Charge un fichier NPZ ou NPY.
        
        Args:
            npz_path: Chemin vers le fichier NPZ ou NPY
            array_key: Clé du tableau pour NPZ ('arr_0', 'volume', etc.) - ignoré pour NPY
            
        Returns:
            True si succès, False sinon
        """
        try:
            file_path = Path(npz_path)
            if not file_path.exists():
                self.logger.error(f"Fichier introuvable: {file_path}")
                return False
            
            # Détecter le format selon l'extension
            file_ext = file_path.suffix.lower()
            
            if file_ext == '.npy':
                # Format NPY : charge directement un array
                self.logger.info(f"Chargement d'un fichier NPY: {file_path}")
                self.volume = np.load(file_path)
                self.logger.info(f"Volume NPY chargé: shape={self.volume.shape}, dtype={self.volume.dtype}")
            elif file_ext == '.npz':
                # Format NPZ : charge un dictionnaire avec des clés
                self.logger.info(f"Chargement d'un fichier NPZ: {file_path}")
                data = np.load(file_path)
                self.logger.info(f"Clés disponibles: {list(data.keys())}")
                
                # Charger le volume
                if array_key not in data:
                    self.logger.warning(f"Clé '{array_key}' introuvable, utilisation de la première clé")
                    array_key = list(data.keys())[0]
                
                self.volume = data[array_key]
                self.logger.info(f"Volume NPZ chargé: shape={self.volume.shape}, dtype={self.volume.dtype}")
            else:
                self.logger.error(f"Format non supporté: {file_ext}. Formats acceptés: .npz, .npy")
                return False

            # Log NPZ loading
            npz_debug_logger.log_npz_loading(
                npz_path=str(npz_path),
                masks_shape=self.volume.shape,
                num_slices=self.volume.shape[0] if len(self.volume.shape) >= 3 else 1
            )

            # Log classes présentes
            unique_classes = np.unique(self.volume)
            npz_debug_logger.log_variable("unique_classes", unique_classes.tolist())
            npz_debug_logger.log_variable("min_value", int(np.min(self.volume)))
            npz_debug_logger.log_variable("max_value", int(np.max(self.volume)))

            return True
            
        except Exception as e:
            self.logger.error(f"Erreur lors du chargement du NPZ: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def get_overlay_slice(self, index: int, target_shape: Tuple[int, int]) -> Optional[np.ndarray]:
        """
        Extrait une slice du volume et la prépare pour l'affichage.

        Args:
            index: Index de la slice
            target_shape: Shape cible (H, W) des images affichées

        Returns:
            Image RGBA (H, W, 4) ou None si erreur
        """
        if self.volume is None:
            return None

        try:
            # Extraire la slice (toujours premier axe)
            if index >= self.volume.shape[0]:
                self.logger.error(f"Index {index} hors limites (max: {self.volume.shape[0]})")
                return None

            slice_data = self.volume[index, :, :]
            self.logger.debug(f"Slice extraite: shape={slice_data.shape}")

            # Appliquer la même transformation que les endviews
            # Si la shape ne correspond pas, essayer rotation -90°
            if slice_data.shape != target_shape:
                # Essayer rotation -90° (comme pour crosswise sans transpose)
                rotated = np.rot90(slice_data, k=-1)
                if rotated.shape == target_shape:
                    slice_data = rotated
                    # self.logger.info(f"Rotation -90° appliquée: {slice_data.shape}")
                # Sinon essayer transpose
                elif slice_data.shape == target_shape[::-1]:
                    slice_data = slice_data.T
                    self.logger.info(f"Transpose appliqué: {slice_data.shape}")
                else:
                    self.logger.warning(f"Shape mismatch non résolu: slice={slice_data.shape}, target={target_shape}")

            # Créer BGRA avec couleurs des classes (BGR car l'image est en BGR)
            bgra = np.zeros((*slice_data.shape, 4), dtype=np.float32)

            # Appliquer les couleurs selon les valeurs de classe
            # Classes 1-4 : Annotations (frontwall, backwall, flaw, indication)
            # Classes 5-9 : Visualisation (plots et lignes)
            for class_value in [1, 2, 3, 4, 5, 6, 7, 8, 9]:
                mask = (slice_data == class_value)
                if np.any(mask):
                    # Récupérer la couleur BGR normalisée (garder l'ordre BGR)
                    if class_value in MASK_COLORS_BGR:
                        bgr_color = np.array(MASK_COLORS_BGR[class_value]) / 255.0
                        # Garder l'ordre BGR car l'image est en BGR
                        bgra[mask, 0] = bgr_color[0]  # B
                        bgra[mask, 1] = bgr_color[1]  # G
                        bgra[mask, 2] = bgr_color[2]  # R
                        bgra[mask, 3] = self.alpha     # Alpha fixe

            self.logger.debug(f"Classes trouvées: {np.unique(slice_data)}")

            return bgra

        except Exception as e:
            self.logger.error(f"Erreur lors de l'extraction de la slice: {e}")
            return None

    def build_overlay_volume(self, masks: List[np.ndarray]) -> Optional[np.ndarray]:
        """
        Construit un volume overlay (flip horizontal + transpose ZXY) cohérent pour l'affichage 2D/3D.
        """
        if masks is None or len(masks) == 0:
            return None
        flipped_masks = [np.fliplr(mask) for mask in masks]
        volume_data = np.stack(flipped_masks, axis=0)
        return np.transpose(volume_data, (0, 2, 1))

    def transform_mask_slice(self, mask_slice: np.ndarray) -> np.ndarray:
        """Applique les transformations overlay (flip + transpose) à une slice unique."""
        flipped = np.fliplr(mask_slice)
        return flipped.T

    def inverse_transform_slice(self, overlay_slice: np.ndarray) -> np.ndarray:
        """Inverse les transformations overlay pour réinjecter une slice dans l'array global."""
        return np.fliplr(overlay_slice.T)

    def update_overlay_volume(
        self,
        masks: List[np.ndarray],
        changed_indices: Optional[List[int]] = None
    ) -> Optional[np.ndarray]:
        """
        Met à jour le volume d'overlay de manière optimisée.

        Stratégie:
        - Si changed_indices est None ou contient beaucoup d'indices: rebuild complet
        - Si changed_indices est petit (< 10% du total): mise à jour partielle

        Args:
            masks: Liste complète des masques (array global)
            changed_indices: Liste des indices modifiés (None = rebuild complet)

        Returns:
            Volume overlay mis à jour avec transformations appliquées
        """
        try:
            if masks is None or len(masks) == 0:
                self.logger.warning("Aucun masque à transformer pour l'overlay")
                return None

            num_total = len(masks)

            # Décider entre rebuild complet ou partiel
            if changed_indices is None or len(changed_indices) > num_total * 0.1:
                # Rebuild complet (plus de 10% des slices modifiées ou indices inconnus)
                self.logger.info(f"Rebuild complet de l'overlay: {num_total} slices")
                return self.build_overlay_volume(masks)
            else:
                # Mise à jour partielle (moins de 10% des slices modifiées)
                self.logger.info(f"Mise à jour partielle de l'overlay: {len(changed_indices)} slices modifiées")

                # Si pas de volume existant, faire un rebuild complet
                if self.volume is None:
                    self.logger.warning("Pas de volume existant pour mise à jour partielle, rebuild complet")
                    return self.build_overlay_volume(masks)

                # Copier le volume existant
                updated_volume = self.volume.copy()

                # Mettre à jour uniquement les slices modifiées
                for idx in changed_indices:
                    if 0 <= idx < num_total:
                        # Appliquer les transformations à la slice
                        transformed_slice = self.transform_mask_slice(masks[idx])
                        updated_volume[idx] = transformed_slice
                    else:
                        self.logger.warning(f"Index hors limites ignoré: {idx}")

                self.logger.info(f"Mise à jour partielle terminée: {len(changed_indices)} slices")
                return updated_volume

        except Exception as e:
            self.logger.error(f"Erreur lors de la mise à jour de l'overlay: {e}")
            import traceback
            traceback.print_exc()
            return None

    def clear(self):
        """Supprime le volume chargé."""
        self.volume = None
        self.logger.info("Overlay NPZ supprimé")

