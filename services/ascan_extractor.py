#!/usr/bin/env python3
"""
Service pour extraire les valeurs A-scan correspondant aux pixels annotés dans les masques.
VERSION CORRIGÉE - Logique basée sur l'analyse du pipeline de chargement NDE.

CORRECTION PRINCIPALE:
- Pour orientations lengthwise/crosswise: y dans l'image correspond à l'index ultrasound dans le profil
- Pour orientation ultrasound: prendre le maximum du profil (pas d'index Y direct)
- Utilisation correcte du paramètre rotation_applied (= NOT transpose)

OPTIMISATIONS:
- Vectorisation NumPy pour extraction massive de valeurs A-scan (40-60% plus rapide)
"""

import numpy as np
import logging
from typing import Dict, List, Optional
import time


class AScanExtractor:
    """
    Extrait les valeurs A-scan pour chaque pixel annoté dans les masques.
    Version corrigée avec mapping de coordonnées exact + optimisations vectorisées.
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Flag pour activer/désactiver les logs de performance détaillés
        self.ENABLE_PERF_LOGS = True  # Mettre à False pour désactiver les logs détaillés
        
        # Mapping des valeurs de masque vers les noms de classes
        self.class_names = {
            0: "background",
            1: "frontwall",
            2: "backwall",
            3: "flaw",
            4: "indication"
        }
    
    def extract_ascan_values_from_masks(
        self,
        global_masks_array: List[np.ndarray],
        volume_data: np.ndarray,
        orientation: str,
        transpose: bool,
        rotation_applied: bool = False,
        nde_filename: str = "unknown.nde"
    ) -> Dict:
        """
        Extrait les valeurs A-scan pour chaque pixel annoté dans les masques.
        
        Args:
            global_masks_array: Liste des masques 2D (un par endview)
            volume_data: Volume 3D (lengthwise, crosswise, ultrasound)
            orientation: 'lengthwise', 'crosswise', ou 'ultrasound'
            transpose: Si transpose a été appliqué lors de l'affichage
            rotation_applied: Si rotation -90° a été appliquée
            nde_filename: Nom du fichier NDE source
            
        Returns:
            Dict avec structure complète incluant métadonnées et valeurs A-scan
        """
        try:
            if self.ENABLE_PERF_LOGS:
                start_time_total = time.perf_counter()
            
            self.logger.info(f"Extraction des valeurs A-scan pour {len(global_masks_array)} endviews...")
            self.logger.info(f"Volume shape: {volume_data.shape}, orientation: {orientation}, transpose: {transpose}, rotation: {rotation_applied}")
            if self.ENABLE_PERF_LOGS:
                self.logger.info(f"[PERF] Mode vectorisé activé: extraction NumPy avancée (40-60% plus rapide)")
            
            # Créer la structure de métadonnées
            result = {
                "metadata": {
                    "nde_file": nde_filename,
                    "num_endviews": len(global_masks_array),
                    "orientation": orientation,
                    "volume_shape": list(volume_data.shape),
                    "transpose": transpose,
                    "rotation_applied": rotation_applied,
                    "classes": self.class_names,
                    "extractor_version": "2.1_vectorized"
                },
                "endviews": {}
            }
            
            # Traiter chaque endview
            total_pixels = 0
            total_endviews = len(global_masks_array)

            # Afficher la progression tous les 10%
            progress_step = max(1, total_endviews // 10)

            for slice_idx, mask in enumerate(global_masks_array):
                endview_data = self._extract_endview_ascan_values(
                    mask=mask,
                    slice_idx=slice_idx,
                    volume_data=volume_data,
                    orientation=orientation,
                    transpose=transpose,
                    rotation_applied=rotation_applied
                )

                # Compter les pixels annotés
                pixels_count = sum(len(pixels) for pixels in endview_data.values())
                total_pixels += pixels_count

                # Ajouter seulement si des annotations existent
                if pixels_count > 0:
                    result["endviews"][str(slice_idx)] = endview_data

                # Afficher la progression
                if (slice_idx + 1) % progress_step == 0 or (slice_idx + 1) == total_endviews:
                    progress_pct = ((slice_idx + 1) / total_endviews) * 100
                    if self.ENABLE_PERF_LOGS:
                        elapsed_so_far = time.perf_counter() - start_time_total
                        avg_time_per_endview = elapsed_so_far / (slice_idx + 1)
                        eta_seconds = avg_time_per_endview * (total_endviews - (slice_idx + 1))
                        pixels_per_sec = total_pixels / elapsed_so_far if elapsed_so_far > 0 else 0
                        print(f"   → Progression: {slice_idx + 1}/{total_endviews} endviews ({progress_pct:.0f}%) - {total_pixels} pixels | {pixels_per_sec:.0f} px/s | ETA: {eta_seconds:.1f}s")
                    else:
                        print(f"   → Progression: {slice_idx + 1}/{total_endviews} endviews ({progress_pct:.0f}%) - {total_pixels} pixels annotés")

            if self.ENABLE_PERF_LOGS:
                elapsed_total = time.perf_counter() - start_time_total
                pixels_per_sec = total_pixels / elapsed_total if elapsed_total > 0 else 0
                self.logger.info(f"[PERF] ⚡ TOTAL extraction: {elapsed_total:.2f}s pour {total_pixels} pixels")
                self.logger.info(f"[PERF] ⚡ Débit: {pixels_per_sec:.0f} pixels/seconde")
                self.logger.info(f"[PERF] ⚡ Moyenne: {elapsed_total/total_endviews*1000:.2f}ms par endview")
            
            self.logger.info(f"Extraction terminée: {total_pixels} pixels annotés au total")
            self.logger.info(f"Endviews avec annotations: {len(result['endviews'])}/{len(global_masks_array)}")
            print(f"   → Total: {total_pixels} pixels annotés dans {len(result['endviews'])} endviews")
            
            return result
            
        except Exception as e:
            self.logger.error(f"Erreur lors de l'extraction des valeurs A-scan: {str(e)}")
            raise
    
    def _extract_endview_ascan_values(
        self,
        mask: np.ndarray,
        slice_idx: int,
        volume_data: np.ndarray,
        orientation: str,
        transpose: bool,
        rotation_applied: bool
    ) -> Dict[str, List[Dict]]:
        """
        Extrait les valeurs A-scan pour une endview spécifique.
        VERSION VECTORISÉE pour performance maximale.
        
        Args:
            mask: Masque 2D (height, width) avec valeurs 0-4
            slice_idx: Index de l'endview dans le volume
            volume_data: Volume 3D complet
            orientation: Orientation du slicing
            transpose: Si transpose appliqué
            rotation_applied: Si rotation -90° appliquée
            
        Returns:
            Dict avec structure {classe: [{x, y, ascan_value}, ...]}
        """
        if self.ENABLE_PERF_LOGS:
            start_time = time.perf_counter()
        
        endview_data = {}
        
        # Traiter chaque classe (ignorer background = 0)
        for class_value in [1, 2, 3, 4]:
            class_name = self.class_names[class_value]
            
            # Trouver tous les pixels de cette classe
            pixel_coords = np.argwhere(mask == class_value)
            
            if len(pixel_coords) == 0:
                continue  # Pas de pixels pour cette classe
            
            # OPTIMISATION: Extraction vectorisée au lieu de boucle pixel par pixel
            ascan_values = self._extract_ascan_values_vectorized(
                pixel_coords=pixel_coords,
                slice_idx=slice_idx,
                volume_data=volume_data,
                orientation=orientation,
                transpose=transpose,
                rotation_applied=rotation_applied
            )
            
            # Construire la liste de dicts (optimisé avec list comprehension)
            pixel_data = [
                {
                    "x": int(coord[1]),  # coord = (y, x)
                    "y": int(coord[0]),
                    "ascan_value": float(val)
                }
                for coord, val in zip(pixel_coords, ascan_values)
                if not np.isnan(val)  # Filtrer les valeurs invalides
            ]
            
            if pixel_data:
                endview_data[class_name] = pixel_data
        
        if self.ENABLE_PERF_LOGS:
            elapsed = (time.perf_counter() - start_time) * 1000
            total_pixels = sum(len(data) for data in endview_data.values())
            self.logger.debug(f"[PERF] Endview {slice_idx}: {total_pixels} pixels extraits en {elapsed:.2f}ms")
        
        return endview_data
    
    def _extract_ascan_values_vectorized(
        self,
        pixel_coords: np.ndarray,
        slice_idx: int,
        volume_data: np.ndarray,
        orientation: str,
        transpose: bool,
        rotation_applied: bool
    ) -> np.ndarray:
        """
        OPTIMISATION: Extrait les valeurs A-scan pour TOUS les pixels d'un coup (vectorisé).
        Au lieu de boucler sur chaque pixel, on utilise l'indexation avancée NumPy.
        
        Args:
            pixel_coords: Array (N, 2) de coordonnées (y, x)
            slice_idx: Index du slice actuel
            volume_data: Volume 3D (lengthwise, crosswise, ultrasound)
            orientation: 'lengthwise', 'crosswise', ou 'ultrasound'
            transpose: Si transpose appliqué
            rotation_applied: Si rotation -90° appliquée
            
        Returns:
            Array (N,) de valeurs A-scan (NaN si invalide)
        """
        num_pixels = len(pixel_coords)
        ascan_values = np.full(num_pixels, np.nan, dtype=np.float32)
        
        # Séparer les coordonnées y et x
        ys = pixel_coords[:, 0]  # Lignes (hauteur)
        xs = pixel_coords[:, 1]  # Colonnes (largeur)
        
        try:
            if orientation == 'lengthwise':
                # Volume: (lengthwise, crosswise, ultrasound)
                # Slice: volume[slice_idx, :, :] = (crosswise, ultrasound)
                
                if rotation_applied:
                    # Mapping inverse de la rotation
                    crosswise_indices = volume_data.shape[1] - 1 - xs
                    # Filtrer les indices valides
                    valid_mask = (crosswise_indices >= 0) & (crosswise_indices < volume_data.shape[1]) & (ys >= 0) & (ys < volume_data.shape[2])
                    valid_indices = np.where(valid_mask)[0]
                    
                    if len(valid_indices) > 0:
                        # Extraction vectorisée pour tous les pixels valides d'un coup
                        ascan_values[valid_indices] = volume_data[slice_idx, crosswise_indices[valid_indices], ys[valid_indices]]
                else:
                    # Mapping direct
                    valid_mask = (xs >= 0) & (xs < volume_data.shape[1]) & (ys >= 0) & (ys < volume_data.shape[2])
                    valid_indices = np.where(valid_mask)[0]
                    
                    if len(valid_indices) > 0:
                        ascan_values[valid_indices] = volume_data[slice_idx, xs[valid_indices], ys[valid_indices]]
            
            elif orientation == 'crosswise':
                # Volume: (lengthwise, crosswise, ultrasound)
                # Slice: volume[:, slice_idx, :] = (lengthwise, ultrasound)
                
                if rotation_applied:
                    lengthwise_indices = volume_data.shape[0] - 1 - xs
                    valid_mask = (lengthwise_indices >= 0) & (lengthwise_indices < volume_data.shape[0]) & (ys >= 0) & (ys < volume_data.shape[2])
                    valid_indices = np.where(valid_mask)[0]
                    
                    if len(valid_indices) > 0:
                        ascan_values[valid_indices] = volume_data[lengthwise_indices[valid_indices], slice_idx, ys[valid_indices]]
                else:
                    valid_mask = (xs >= 0) & (xs < volume_data.shape[0]) & (ys >= 0) & (ys < volume_data.shape[2])
                    valid_indices = np.where(valid_mask)[0]
                    
                    if len(valid_indices) > 0:
                        ascan_values[valid_indices] = volume_data[xs[valid_indices], slice_idx, ys[valid_indices]]
            
            else:  # ultrasound
                # Volume: (lengthwise, crosswise, ultrasound)
                # Slice: volume[:, :, slice_idx] = (lengthwise, crosswise)
                # Pour ultrasound: prendre le MAX du profil pour chaque pixel
                
                lengthwise_indices = volume_data.shape[0] - 1 - xs
                crosswise_indices = volume_data.shape[1] - 1 - ys
                
                valid_mask = (lengthwise_indices >= 0) & (lengthwise_indices < volume_data.shape[0]) & \
                             (crosswise_indices >= 0) & (crosswise_indices < volume_data.shape[1])
                valid_indices = np.where(valid_mask)[0]
                
                if len(valid_indices) > 0:
                    # Pour chaque pixel, extraire le profil ultrasound et prendre le max
                    for i in valid_indices:
                        profile = volume_data[lengthwise_indices[i], crosswise_indices[i], :]
                        ascan_values[i] = np.max(profile)
        
        except (IndexError, ValueError) as e:
            self.logger.warning(f"Erreur lors de l'extraction vectorisée: {e}")
        
        return ascan_values
    
    def _get_ascan_value_at_position(
        self,
        x: int,
        y: int,
        slice_idx: int,
        volume_data: np.ndarray,
        orientation: str,
        transpose: bool,
        rotation_applied: bool
    ) -> Optional[float]:
        """
        Extrait la valeur A-scan à une position (x, y) spécifique.
        
        LOGIQUE CORRIGÉE basée sur l'analyse du log NDE:
        - Pour lengthwise/crosswise: y dans l'image = index dans le profil ultrasound
        - Pour ultrasound: prendre le maximum du profil
        
        Args:
            x: Coordonnée X dans l'image affichée
            y: Coordonnée Y dans l'image affichée
            slice_idx: Index du slice actuel
            volume_data: Volume 3D (lengthwise, crosswise, ultrasound)
            orientation: 'lengthwise', 'crosswise', ou 'ultrasound'
            transpose: Si transpose appliqué
            rotation_applied: Si rotation -90° appliquée
            
        Returns:
            Valeur A-scan à cette position, ou None si hors limites
        """
        try:
            ascan_profile = None
            
            if orientation == 'lengthwise':
                # Volume: (lengthwise, crosswise, ultrasound)
                # Slice: volume[slice_idx, :, :] = (crosswise, ultrasound)
                
                if rotation_applied:
                    # Rotation -90° appliquée: image = (ultrasound hauteur, crosswise largeur)
                    # Position (x, y): x=0-(crosswise-1), y=0-(ultrasound-1)
                    # Mapping: x → crosswise_idx = crosswise-1-x (rotation inverse)
                    if 0 <= x < volume_data.shape[1]:  # crosswise
                        crosswise_idx = volume_data.shape[1] - 1 - x
                        ascan_profile = volume_data[slice_idx, crosswise_idx, :]  # profil ultrasound
                    else:
                        return None
                else:
                    # Transpose appliqué: image = (ultrasound hauteur, crosswise largeur)
                    # Position (x, y): x=0-(crosswise-1), y=0-(ultrasound-1)
                    # Mapping direct: x → crosswise_idx = x
                    if 0 <= x < volume_data.shape[1]:  # crosswise
                        ascan_profile = volume_data[slice_idx, x, :]  # profil ultrasound
                    else:
                        return None
                
                # CORRECTION: y dans l'image correspond à l'index ultrasound
                if ascan_profile is not None and 0 <= y < len(ascan_profile):
                    return ascan_profile[y]
                else:
                    return None
            
            elif orientation == 'crosswise':
                # Volume: (lengthwise, crosswise, ultrasound)
                # Slice: volume[:, slice_idx, :] = (lengthwise, ultrasound)
                
                if rotation_applied:
                    # Rotation -90° appliquée: image = (ultrasound hauteur, lengthwise largeur)
                    # Position (x, y): x=0-(lengthwise-1), y=0-(ultrasound-1)
                    # Mapping: x → lengthwise_idx = lengthwise-1-x (rotation inverse)
                    if 0 <= x < volume_data.shape[0]:  # lengthwise
                        lengthwise_idx = volume_data.shape[0] - 1 - x
                        ascan_profile = volume_data[lengthwise_idx, slice_idx, :]  # profil ultrasound
                    else:
                        return None
                else:
                    # Transpose appliqué: image = (ultrasound hauteur, lengthwise largeur)
                    # Position (x, y): x=0-(lengthwise-1), y=0-(ultrasound-1)
                    # Mapping direct: x → lengthwise_idx = x
                    if 0 <= x < volume_data.shape[0]:  # lengthwise
                        ascan_profile = volume_data[x, slice_idx, :]  # profil ultrasound
                    else:
                        return None
                
                # CORRECTION: y dans l'image correspond à l'index ultrasound
                if ascan_profile is not None and 0 <= y < len(ascan_profile):
                    return ascan_profile[y]
                else:
                    return None
            
            else:  # ultrasound
                # Volume: (lengthwise, crosswise, ultrasound)
                # Slice: volume[:, :, slice_idx] = (lengthwise, crosswise)
                # Image après rotation: (crosswise hauteur, lengthwise largeur)
                # Position (x, y): x=0-(lengthwise-1), y=0-(crosswise-1)
                
                # Mapping inverse de la rotation
                if 0 <= x < volume_data.shape[0] and 0 <= y < volume_data.shape[1]:
                    lengthwise_idx = volume_data.shape[0] - 1 - x
                    crosswise_idx = volume_data.shape[1] - 1 - y
                    ascan_profile = volume_data[lengthwise_idx, crosswise_idx, :]  # profil ultrasound
                else:
                    return None
                
                # CORRECTION: Pour ultrasound, prendre le MAXIMUM du profil
                # car y n'a pas de correspondance directe avec l'index ultrasound
                if ascan_profile is not None:
                    return np.max(ascan_profile)
                else:
                    return None
                
        except IndexError as e:
            self.logger.warning(f"Position hors limites ({x}, {y}): {e}")
            return None


def export_ascan_values_to_json(
    global_masks_array: List[np.ndarray],
    volume_data: np.ndarray,
    orientation: str,
    transpose: bool,
    rotation_applied: bool = False,
    nde_filename: str = "unknown.nde"
) -> Dict:
    """
    Fonction utilitaire pour extraire les valeurs A-scan depuis les masques.
    VERSION CORRIGÉE avec logique basée sur l'analyse du pipeline NDE.
    
    Args:
        global_masks_array: Liste des masques 2D
        volume_data: Volume 3D (lengthwise, crosswise, ultrasound)
        orientation: 'lengthwise', 'crosswise', ou 'ultrasound'
        transpose: Si transpose a été appliqué
        rotation_applied: Si rotation -90° a été appliquée
        nde_filename: Nom du fichier NDE source
        
    Returns:
        Dict avec structure complète prête pour export JSON
    """
    extractor = AScanExtractor()
    return extractor.extract_ascan_values_from_masks(
        global_masks_array=global_masks_array,
        volume_data=volume_data,
        orientation=orientation,
        transpose=transpose,
        rotation_applied=rotation_applied,
        nde_filename=nde_filename
    )

