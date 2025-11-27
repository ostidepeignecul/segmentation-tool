# === services/nde_loader_service.py ===
import os
import h5py
import json
import numpy as np
import cv2
import logging
from typing import List, Dict, Tuple, Optional
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import tempfile
import shutil

from utils.helpers import safe_division
from utils.omniscan_div_cmap import get_omniscan_diverging_colormap
from services.nde_debug_logger import nde_debug_logger


class NdeLoaderService:
    """
    Service pour charger et traiter les fichiers NDE.
    Supporte les structures Domain (Suzlon) et Public.
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self._temp_dir = None
        self._current_nde_data = None
        self._current_nde_path = None
        # Cache mémoire pour les images générées
        self._cached_images = None  # Liste de numpy arrays (H, W, 3) en BGR
        self._cached_image_names = None  # Liste des noms de fichiers virtuels
        self._cached_raw_slices = None  # Liste des slices normalisés (float32) alignés sur les endviews
        
    def __del__(self):
        """Nettoyage du répertoire temporaire."""
        self.cleanup_temp_dir()
    
    def cleanup_temp_dir(self):
        """Supprime le répertoire temporaire s'il existe."""
        if self._temp_dir and os.path.exists(self._temp_dir):
            try:
                shutil.rmtree(self._temp_dir)
                self.logger.info(f"Répertoire temporaire supprimé: {self._temp_dir}")
            except Exception as e:
                self.logger.warning(f"Impossible de supprimer le répertoire temporaire: {e}")
            self._temp_dir = None
    
    def detect_nde_structure(self, nde_file: str) -> str:
        """
        Détecte la structure du fichier NDE.
        
        Args:
            nde_file: Chemin vers le fichier NDE
            
        Returns:
            str: 'domain' ou 'public'
        """
        try:
            with h5py.File(nde_file, 'r') as f:
                root_keys = list(f.keys())
                
                if 'Domain' in root_keys:
                    return 'domain'
                elif 'Public' in root_keys:
                    return 'public'
                else:
                    raise ValueError(f"Structure NDE non reconnue: {root_keys}")
        except Exception as e:
            self.logger.error(f"Erreur lors de la détection de structure: {e}")
            raise
    
    def load_nde_data(self, nde_file: str, group_idx: int = 1) -> Dict:
        """
        Charge les données d'un fichier NDE.

        Args:
            nde_file: Chemin vers le fichier NDE
            group_idx: Index du groupe à charger (commence à 1)

        Returns:
            Dict: Dictionnaire contenant les données et métadonnées
        """
        try:
            nde_debug_logger.log_section("LOAD_NDE_DATA")
            nde_debug_logger.log_variable("nde_file", nde_file)
            nde_debug_logger.log_variable("group_idx", group_idx)

            structure = self.detect_nde_structure(nde_file)
            self.logger.info(f"Structure détectée: {structure}")
            nde_debug_logger.log_variable("structure", structure, indent=1)

            with h5py.File(nde_file, 'r') as f:
                if structure == 'domain':
                    result = self._load_domain_structure(f, group_idx)
                else:  # public
                    result = self._load_public_structure(f, group_idx)

                nde_debug_logger.log("Données chargées:", indent=1)
                nde_debug_logger.log_variable("data_array", result['data_array'], indent=2)
                nde_debug_logger.log_variable("structure", result.get('structure'), indent=2)

                return result

        except Exception as e:
            self.logger.error(f"Erreur lors du chargement NDE: {e}")
            nde_debug_logger.log(f"ERREUR: {e}", indent=1)
            raise
    
    def _load_domain_structure(self, f: h5py.File, group_idx: int) -> Dict:
        """Charge les données d'un fichier avec structure Domain (Suzlon)."""
        try:
            # Charger la configuration
            json_str = f['Domain/Setup'][()]
            json_decoded = json.loads(json_str)

            # Extraire les données
            data_path = f"Domain/DataGroups/{group_idx-1}/Datasets/0/Amplitude"
            if data_path not in f:
                raise ValueError(f"Chemin de données non trouvé: {data_path}")

            data_array = f[data_path][:]
            self.logger.info(f"Données chargées: {data_array.shape}")

            # Détecter et corriger les valeurs négatives
            data_array = self._correct_negative_values(data_array)

            # Extraire les métadonnées
            group_info = json_decoded["groups"][group_idx-1]

            # Calculer min/max
            min_value = np.min(data_array)
            max_value = np.max(data_array)

            # Extraire les dimensions - essayer différentes structures
            dimensions = None
            try:
                # Structure standard
                dimensions = group_info['data']['ascan']['dataset']['amplitude']['dimensions']
            except KeyError:
                try:
                    # Structure alternative 1: directement dans dataset
                    dimensions = group_info['dataset']['amplitude']['dimensions']
                except KeyError:
                    try:
                        # Structure alternative 2: directement dans ascan
                        dimensions = group_info['ascan']['dataset']['amplitude']['dimensions']
                    except KeyError:
                        # Dernière tentative: chercher dans toutes les clés
                        self.logger.warning(f"Structure Domain non standard. Clés disponibles: {list(group_info.keys())}")
                        # Créer des dimensions par défaut basées sur la forme du tableau
                        dimensions = [
                            {'offset': 0, 'resolution': 1, 'quantity': data_array.shape[0]},
                            {'offset': 0, 'resolution': 1, 'quantity': data_array.shape[1]},
                            {'offset': 0, 'resolution': 1, 'quantity': data_array.shape[2]}
                        ]
                        self.logger.info(f"Dimensions par défaut créées: {dimensions}")

            # Créer les positions
            positions = {}
            for axis, idx in zip(['lengthwise', 'crosswise', 'ultrasound'], range(3)):
                dim_info = dimensions[idx]
                positions[axis] = np.array([
                    dim_info['offset'] + i * dim_info['resolution']
                    for i in range(dim_info['quantity'])
                ])

            return {
                'data_array': data_array,
                'min_value': min_value,
                'max_value': max_value,
                'dimensions': dimensions,
                'positions': positions,
                'group_info': group_info,
                'structure': 'domain'
            }

        except Exception as e:
            self.logger.error(f"Erreur lors du chargement de la structure Domain: {e}")
            raise
    
    def _load_public_structure(self, f: h5py.File, group_idx: int) -> Dict:
        """Charge les données d'un fichier avec structure Public."""
        try:
            # Charger la configuration
            json_str = f['Public/Setup'][()]
            json_decoded = json.loads(json_str)
            
            # Trouver les données
            datasets_path = f"Public/Groups/{group_idx-1}/Datasets"
            if datasets_path not in f:
                raise ValueError(f"Chemin de données non trouvé: {datasets_path}")
            
            # Chercher le dataset AScanAmplitude
            data_array = None
            for key in f[datasets_path].keys():
                if 'AScanAmplitude' in key:
                    data_array = f[f"{datasets_path}/{key}"][:]
                    break
            
            if data_array is None:
                raise ValueError("Aucun dataset AScanAmplitude trouvé")
            
            self.logger.info(f"Données chargées: {data_array.shape}")
            
            # Calculer min/max
            min_value = np.min(data_array)
            max_value = np.max(data_array)
            
            # Extraire les dimensions depuis la configuration
            group_info = json_decoded["groups"][group_idx-1]

            # Pour la structure Public, les dimensions sont directement dans datasets[0]
            dimensions = group_info['datasets'][0]['dimensions']

            # Créer les positions (structure Public utilise des clés différentes)
            positions = {}
            axis_names = ['lengthwise', 'crosswise', 'ultrasound']

            for axis_name, idx in zip(axis_names, range(3)):
                if idx < len(dimensions):
                    dim_info = dimensions[idx]
                    # Structure Public utilise 'offset', 'resolution', 'quantity'
                    # mais peut aussi avoir d'autres noms de clés
                    offset = dim_info.get('offset', 0)
                    resolution = dim_info.get('resolution', 1)
                    quantity = dim_info.get('quantity', data_array.shape[idx])

                    positions[axis_name] = np.array([
                        offset + i * resolution
                        for i in range(quantity)
                    ])
            
            return {
                'data_array': data_array,
                'min_value': min_value,
                'max_value': max_value,
                'dimensions': dimensions,
                'positions': positions,
                'group_info': group_info,
                'structure': 'public'
            }
            
        except Exception as e:
            self.logger.error(f"Erreur lors du chargement de la structure Public: {e}")
            raise
    
    def _correct_negative_values(self, data: np.ndarray) -> np.ndarray:
        """Corrige les valeurs négatives en les ramenant à zéro."""
        min_val = np.min(data)
        if min_val >= 0:
            return data
        
        # Compter les valeurs négatives
        negative_count = np.sum(data < 0)
        negative_percentage = (negative_count / data.size) * 100
        
        self.logger.info(f"Correction des valeurs négatives: {negative_count} valeurs ({negative_percentage:.3f}%)")
        
        # Clipping à 0
        corrected_data = np.clip(data, 0, None)
        return corrected_data
    
    def detect_optimal_orientation(self, data_array: np.ndarray, structure: str = 'public') -> Dict:
        """
        Détecte l'orientation optimale pour les endviews.

        LOGIQUE DIFFÉRENTE SELON LE TYPE DE FICHIER:
        - Fichiers PUBLIC (standard): TOUJOURS lengthwise (premier axe)
        - Fichiers DOMAIN (Suzlon): Système de score pour choisir l'orientation avec le plus d'endviews

        Args:
            data_array: Array 3D (lengthwise, crosswise, ultrasound)
            structure: Type de structure ('public' ou 'domain')

        Returns:
            Dict: Configuration d'orientation optimale
        """
        nde_debug_logger.log_section("DETECT_OPTIMAL_ORIENTATION")
        nde_debug_logger.log_variable("data_array", data_array)
        nde_debug_logger.log_variable("structure", structure)

        lengthwise_qty, crosswise_qty, ultrasound_qty = data_array.shape
        nde_debug_logger.log(f"Dimensions: lengthwise={lengthwise_qty}, crosswise={crosswise_qty}, ultrasound={ultrasound_qty}", indent=1)

        if structure == 'public':
            # FICHIERS STANDARD (Public): TOUJOURS extraire selon l'axe lengthwise
            # C'est la convention standard pour les endviews NDE
            # data[idx,:,:] → (crosswise, ultrasound)
            nde_debug_logger.log("[PUBLIC] Extraction selon axe lengthwise", indent=1)
            slice_lengthwise = data_array[0,:,:]
            nde_debug_logger.log_variable("slice_lengthwise.shape", slice_lengthwise.shape, indent=2)

            aspect_lengthwise = slice_lengthwise.shape[1] / slice_lengthwise.shape[0] if slice_lengthwise.shape[0] > 0 else 1.0
            nde_debug_logger.log_variable("aspect_lengthwise", aspect_lengthwise, indent=2)

            # Détermine si transpose est nécessaire
            transpose = aspect_lengthwise < 1.0
            nde_debug_logger.log_variable("transpose", transpose, indent=2)
            nde_debug_logger.log_variable("rotation_applied", not transpose, indent=2)

            self.logger.info(f"[PUBLIC] Orientation: lengthwise, shape={slice_lengthwise.shape}, aspect={aspect_lengthwise:.3f}, transpose={transpose}")

            result = {
                'slice_orientation': 'lengthwise',
                'transpose': transpose,
                'num_images': lengthwise_qty,
                'shape': slice_lengthwise.shape,
                'aspect': aspect_lengthwise
            }
            nde_debug_logger.log_variable("orientation_config", result, indent=1)
            return result

        else:  # structure == 'domain' (Suzlon)
            # FICHIERS SUZLON: Utiliser système de score pour choisir la meilleure orientation
            # Basé sur nde_NdeSuzlonToPng.py
            orientations = []

            # 1. Slice lengthwise: data[idx,:,:] → (crosswise, ultrasound)
            slice_lengthwise = data_array[0,:,:]
            aspect_lengthwise = slice_lengthwise.shape[1] / slice_lengthwise.shape[0] if slice_lengthwise.shape[0] > 0 else 1.0
            orientations.append({
                'name': 'lengthwise',
                'shape': slice_lengthwise.shape,
                'aspect': aspect_lengthwise,
                'num_images': lengthwise_qty
            })

            # 2. Slice crosswise: data[:,idx,:] → (lengthwise, ultrasound)
            slice_crosswise = data_array[:,0,:]
            aspect_crosswise = slice_crosswise.shape[1] / slice_crosswise.shape[0] if slice_crosswise.shape[0] > 0 else 1.0
            orientations.append({
                'name': 'crosswise',
                'shape': slice_crosswise.shape,
                'aspect': aspect_crosswise,
                'num_images': crosswise_qty
            })

            # 3. Slice ultrasound: data[:,:,idx] → (lengthwise, crosswise)
            slice_ultrasound = data_array[:,:,0]
            aspect_ultrasound = slice_ultrasound.shape[1] / slice_ultrasound.shape[0] if slice_ultrasound.shape[0] > 0 else 1.0
            orientations.append({
                'name': 'ultrasound',
                'shape': slice_ultrasound.shape,
                'aspect': aspect_ultrasound,
                'num_images': ultrasound_qty
            })

            # Système de score: privilégier le nombre d'images
            best_orientation = None
            best_score = -1

            for orient in orientations:
                num_images = orient['num_images']
                aspect = orient['aspect']

                # Score basé sur le nombre d'images
                if num_images >= 1000:
                    score = 20
                elif num_images >= 500:
                    score = 15
                elif num_images >= 100:
                    score = 10
                else:
                    score = 5

                # Bonus pour aspect ratio raisonnable
                if 0.1 <= aspect <= 50.0:
                    score += 2
                elif 0.05 <= aspect <= 100.0:
                    score += 1

                if score > best_score:
                    best_score = score
                    best_orientation = orient

            # Détermine si transpose est nécessaire
            transpose = best_orientation['aspect'] < 1.0

            self.logger.info(f"[DOMAIN/Suzlon] Orientation: {best_orientation['name']}, shape={best_orientation['shape']}, "
                           f"aspect={best_orientation['aspect']:.3f}, num_images={best_orientation['num_images']}, transpose={transpose}")

            return {
                'slice_orientation': best_orientation['name'],
                'transpose': transpose,
                'num_images': best_orientation['num_images'],
                'shape': best_orientation['shape'],
                'aspect': best_orientation['aspect']
            }
    
    def extract_slice(self, data_array: np.ndarray, idx: int, orientation: str) -> np.ndarray:
        """
        Extrait un slice selon l'orientation spécifiée.
        
        Args:
            data_array: Array 3D des données
            idx: Index du slice à extraire
            orientation: 'lengthwise', 'crosswise', ou 'ultrasound'
            
        Returns:
            np.ndarray: Le slice extrait
        """
        if orientation == 'crosswise':
            return data_array[:, idx, :]
        elif orientation == 'ultrasound':
            return data_array[:, :, idx]
        else:  # lengthwise
            return data_array[idx, :, :]
    
    def generate_endview_image(self, img_data: np.ndarray, min_value: float, max_value: float,
                             colorize: bool = True, colormap_path: str = None) -> Image.Image:
        """
        Génère une image endview avec normalisation et colormap Omniscan.

        Args:
            img_data: Données de l'image (2D array)
            min_value: Valeur minimale pour la normalisation
            max_value: Valeur maximale pour la normalisation
            colorize: True pour RGB, False pour grayscale
            colormap_path: Chemin vers la colormap OmniScan (.npy)

        Returns:
            PIL.Image: Image générée
        """
        # Normaliser les données
        if max_value == min_value:
            img_data_normalized = np.zeros_like(img_data, dtype=float)
        else:
            img_data_normalized = (img_data - min_value) / (max_value - min_value)
            img_data_normalized = np.clip(img_data_normalized, 0.0, 1.0)

        # Charger la colormap Omniscan
        if colorize and colormap_path and os.path.exists(colormap_path):
            try:
                # Déterminer si on a des valeurs négatives (diverging colormap)
                if min_value < 0:
                    # Utiliser la colormap divergente Omniscan
                    omniscan_cmap = get_omniscan_diverging_colormap(colormap_path)
                    self.logger.debug("Utilisation de la colormap Omniscan divergente")
                else:
                    # Utiliser la colormap standard Omniscan
                    OmniScanColorMap = np.load(colormap_path)
                    omniscan_cmap = ListedColormap(OmniScanColorMap)
                    self.logger.debug("Utilisation de la colormap Omniscan standard")
            except Exception as e:
                self.logger.warning(f"Impossible de charger la colormap Omniscan: {e}")
                omniscan_cmap = plt.cm.viridis
        elif colorize:
            self.logger.warning("Colormap Omniscan non trouvée, utilisation de viridis")
            omniscan_cmap = plt.cm.viridis
        else:
            omniscan_cmap = None

        # Générer l'image
        if colorize and omniscan_cmap:
            # Appliquer la colormap et convertir en uint8
            colored_data = omniscan_cmap(img_data_normalized)
            # Si la colormap retourne RGBA, convertir en RGB
            if colored_data.shape[-1] == 4:
                colored_data = colored_data[:, :, :3]
            img = Image.fromarray(np.uint8(colored_data * 255))
        else:
            img = Image.fromarray(np.uint8(img_data_normalized * 255), 'L')

        return img
    
    def export_nde_to_pngs(self, nde_file: str, output_dir: str, group_idx: int = 1, 
                          colorize: bool = True, colormap_path: str = None) -> List[str]:
        """
        Exporte un fichier NDE vers des images PNG.
        
        Args:
            nde_file: Chemin vers le fichier NDE
            output_dir: Répertoire de sortie
            group_idx: Index du groupe à exporter
            colorize: True pour RGB, False pour grayscale
            colormap_path: Chemin vers la colormap OmniScan
            
        Returns:
            List[str]: Liste des chemins des images générées
        """
        try:
            # Charger les données NDE
            nde_data = self.load_nde_data(nde_file, group_idx)
            data_array = nde_data['data_array']
            structure = nde_data.get('structure', 'public')

            # Détecter l'orientation optimale
            orientation_config = self.detect_optimal_orientation(data_array, structure)
            orientation = orientation_config['slice_orientation']
            transpose = orientation_config['transpose']
            
            # Créer le répertoire de sortie
            os.makedirs(output_dir, exist_ok=True)
            
            # Générer les images
            image_paths = []
            num_images = orientation_config['num_images']
            
            self.logger.info(f"Génération de {num_images} endviews...")
            
            for idx in range(num_images):
                # Extraire le slice
                img_data = self.extract_slice(data_array, idx, orientation)
                
                # Appliquer transpose si nécessaire
                if transpose:
                    img_data = img_data.T
                
                # Générer l'image
                img = self.generate_endview_image(
                    img_data,
                    nde_data['min_value'],
                    nde_data['max_value'],
                    colorize=colorize,
                    colormap_path=colormap_path
                )

                # CORRECTION: Appliquer rotation UNIQUEMENT si transpose n'a PAS été appliqué
                # Logique:
                # - Si transpose=False (image déjà plus large que haute) → rotation -90° nécessaire
                # - Si transpose=True (image corrigée par transpose) → PAS de rotation
                if not transpose:
                    img = img.transpose(Image.ROTATE_270)

                # Sauvegarder
                position_filename = idx * 1500  # Format standard
                filename = f"endview_{position_filename:012d}.png"
                filepath = os.path.join(output_dir, filename)
                img.save(filepath)
                image_paths.append(filepath)
                
                if idx % 100 == 0:
                    self.logger.info(f"Généré {idx+1}/{num_images} images...")
            
            self.logger.info(f"Export terminé: {len(image_paths)} images générées")
            return image_paths
            
        except Exception as e:
            self.logger.error(f"Erreur lors de l'export NDE: {e}")
            raise
    
    def load_nde_as_temp_pngs(self, nde_file: str, group_idx: int = 1) -> List[str]:
        """
        Charge un fichier NDE et l'exporte temporairement en PNG.
        
        Args:
            nde_file: Chemin vers le fichier NDE
            group_idx: Index du groupe à charger
            
        Returns:
            List[str]: Liste des chemins des images PNG temporaires
        """
        try:
            # Nettoyer l'ancien répertoire temporaire
            self.cleanup_temp_dir()
            
            # Créer un nouveau répertoire temporaire
            self._temp_dir = tempfile.mkdtemp(prefix="nde_temp_")
            self.logger.info(f"Répertoire temporaire créé: {self._temp_dir}")
            
            # Exporter vers PNG
            image_paths = self.export_nde_to_pngs(
                nde_file, 
                self._temp_dir, 
                group_idx,
                colorize=True,  # Version colorisée pour l'interface
                colormap_path="./OmniScanColorMap.npy"
            )
            
            # Stocker les informations pour le nettoyage
            self._current_nde_data = self.load_nde_data(nde_file, group_idx)
            self._current_nde_path = nde_file
            
            return image_paths
            
        except Exception as e:
            self.logger.error(f"Erreur lors du chargement NDE temporaire: {e}")
            self.cleanup_temp_dir()
            raise
    
    def get_nde_metadata(self) -> Optional[Dict]:
        """Retourne les métadonnées du fichier NDE actuellement chargé."""
        return self._current_nde_data
    
    def get_nde_path(self) -> Optional[str]:
        """Retourne le chemin du fichier NDE actuellement chargé."""
        return self._current_nde_path

    def load_nde_as_memory_images(self, nde_file: str, group_idx: int = 1) -> Tuple[List[np.ndarray], List[str]]:
        """
        Charge un fichier NDE et génère les images directement en mémoire (pas de fichiers temporaires).

        Cette méthode est beaucoup plus rapide que load_nde_as_temp_pngs car elle évite:
        - L'écriture de fichiers PNG sur disque (~15 secondes pour 526 images)
        - La lecture ultérieure depuis le disque
        - La gestion de fichiers temporaires

        Args:
            nde_file: Chemin vers le fichier NDE
            group_idx: Index du groupe à charger

        Returns:
            Tuple[List[np.ndarray], List[str]]:
                - Liste des images en mémoire (numpy arrays BGR uint8)
                - Liste des noms de fichiers virtuels (pour compatibilité)
        """
        try:
            # Nettoyer l'ancien cache
            self._cached_images = []
            self._cached_image_names = []
            self._cached_raw_slices = []

            # Charger les données NDE
            nde_data = self.load_nde_data(nde_file, group_idx)
            data_array = nde_data['data_array']
            structure = nde_data.get('structure', 'public')

            # Détecter l'orientation optimale
            orientation_config = self.detect_optimal_orientation(data_array, structure)
            orientation = orientation_config['slice_orientation']
            transpose = orientation_config['transpose']
            num_images = orientation_config['num_images']

            self.logger.info(f"Génération de {num_images} endviews en mémoire...")

            # Générer toutes les images en mémoire
            nde_debug_logger.log_section("GENERATION DES ENDVIEWS EN MEMOIRE")
            nde_debug_logger.log_variable("num_images", num_images)
            nde_debug_logger.log_variable("orientation", orientation)
            nde_debug_logger.log_variable("transpose", transpose)

            for idx in range(num_images):
                # Log détaillé seulement pour la première image
                if idx == 0:
                    nde_debug_logger.log(f"Génération de l'endview {idx} (détails complets):", indent=1)

                # Extraire le slice
                img_data = self.extract_slice(data_array, idx, orientation)

                if idx == 0:
                    nde_debug_logger.log_variable("img_data (après extract_slice)", img_data, indent=2)

                # Appliquer transpose si nécessaire
                if transpose:
                    img_data_before = img_data.shape
                    img_data = img_data.T
                    if idx == 0:
                        nde_debug_logger.log_transformation("Transpose", img_data_before, img_data.shape, indent=2)

                # Normaliser les données
                if nde_data['max_value'] == nde_data['min_value']:
                    img_data_normalized = np.zeros_like(img_data, dtype=float)
                else:
                    img_data_normalized = (img_data - nde_data['min_value']) / (nde_data['max_value'] - nde_data['min_value'])
                    img_data_normalized = np.clip(img_data_normalized, 0.0, 1.0)

                if idx == 0:
                    nde_debug_logger.log_variable("img_data_normalized", img_data_normalized, indent=2)

                # Convertir en uint8 (grayscale)
                img_uint8 = (img_data_normalized * 255).astype(np.uint8)

                # CORRECTION: Appliquer rotation UNIQUEMENT si transpose n'a PAS été appliqué
                # Logique:
                # - Si transpose=False (image déjà plus large que haute) → rotation -90° nécessaire
                # - Si transpose=True (image corrigée par transpose) → PAS de rotation
                # Cela permet de gérer tous les types de fichiers NDE correctement
                rotation_applied = not transpose
                if rotation_applied:
                    img_uint8_before = img_uint8.shape
                    img_uint8 = np.rot90(img_uint8, k=-1)
                    img_data_normalized = np.rot90(img_data_normalized, k=-1)
                    if idx == 0:
                        nde_debug_logger.log_transformation("Rotation -90°", img_uint8_before, img_uint8.shape, indent=2)
                        nde_debug_logger.log_variable("rotation_applied", True, indent=2)
                else:
                    if idx == 0:
                        nde_debug_logger.log_variable("rotation_applied", False, indent=2)

                # Convertir en BGR pour OpenCV (grayscale → BGR)
                # Utiliser np.ascontiguousarray pour garantir un array continu en mémoire
                img_bgr = cv2.cvtColor(np.ascontiguousarray(img_uint8), cv2.COLOR_GRAY2BGR)

                if idx == 0:
                    nde_debug_logger.log_variable("img_bgr (final)", img_bgr, indent=2)

                # Stocker la version normalisée brute (float32) alignée avec l'endview affichée
                self._cached_raw_slices.append(np.ascontiguousarray(img_data_normalized.astype(np.float32)))

                # Stocker en mémoire
                self._cached_images.append(img_bgr)

                # Générer le nom de fichier virtuel (pour compatibilité)
                position_filename = idx * 1500
                filename = f"endview_{position_filename:012d}.png"
                self._cached_image_names.append(filename)

                if idx % 100 == 0 and idx > 0:
                    self.logger.info(f"Généré {idx}/{num_images} images en mémoire...")

            self.logger.info(f"Génération terminée: {len(self._cached_images)} images en mémoire")

            # Stocker les métadonnées
            self._current_nde_data = nde_data
            self._current_nde_path = nde_file

            return self._cached_images, self._cached_image_names

        except Exception as e:
            self.logger.error(f"Erreur lors du chargement NDE en mémoire: {e}")
            self._cached_images = None
            self._cached_image_names = None
            raise

    def get_cached_image(self, index: int) -> Optional[np.ndarray]:
        """
        Récupère une image depuis le cache mémoire.

        Args:
            index: Index de l'image

        Returns:
            np.ndarray: Image BGR uint8, ou None si pas en cache
        """
        if self._cached_images is None or index < 0 or index >= len(self._cached_images):
            return None
        return self._cached_images[index]

    def get_cached_image_name(self, index: int) -> Optional[str]:
        """
        Récupère le nom d'une image depuis le cache.

        Args:
            index: Index de l'image

        Returns:
            str: Nom du fichier virtuel, ou None si pas en cache
        """
        if self._cached_image_names is None or index < 0 or index >= len(self._cached_image_names):
            return None
        return self._cached_image_names[index]

    def get_cached_raw_slice(self, index: int) -> Optional[np.ndarray]:
        """
        Récupère la slice brute normalisée alignée avec une endview donnée.

        Args:
            index: Index de l'image/endview

        Returns:
            np.ndarray: Slice float32 normalisé (0-1) ou None si indisponible.
        """
        if self._cached_raw_slices is None or index < 0 or index >= len(self._cached_raw_slices):
            return None
        return self._cached_raw_slices[index].copy()

    def clear_memory_cache(self):
        """Libère le cache mémoire des images."""
        self._cached_images = None
        self._cached_image_names = None
        self._cached_raw_slices = None
        self.logger.info("Cache mémoire libéré")

    # === Métadonnées NDE / résolutions ===

    @staticmethod
    def canonical_axis_name(axis: Optional[str]) -> Optional[str]:
        if not axis:
            return None
        axis_lower = axis.strip().lower()
        if axis_lower in ("lengthwise", "ucoordinate", "u coordinate", "u-axis", "u"):
            return "lengthwise"
        if axis_lower in ("crosswise", "vcoordinate", "v coordinate", "v-axis", "v"):
            return "crosswise"
        if "ultra" in axis_lower:
            return "ultrasound"
        return None

    @staticmethod
    def extract_ultrasound_velocity(group_info: Optional[Dict]) -> Optional[float]:
        if not isinstance(group_info, dict):
            return None

        def _deep_get(container: Dict, path: Tuple[str, ...]) -> Optional[float]:
            current = container
            for key in path:
                if not isinstance(current, dict):
                    return None
                current = current.get(key)
            return current if isinstance(current, (int, float)) else None

        for path in (
            ("dataset", "ascan", "velocity"),
            ("data", "ascan", "velocity"),
            ("ascan", "velocity"),
            ("paut", "velocity"),
        ):
            value = _deep_get(group_info, path)
            if value:
                return float(value)

        processes = group_info.get("processes") if isinstance(group_info.get("processes"), list) else []
        for process in processes:
            if not isinstance(process, dict):
                continue
            pha = process.get("ultrasonicPhasedArray")
            if pha and isinstance(pha, dict):
                velocity = pha.get("velocity")
                if isinstance(velocity, (int, float)) and velocity > 0:
                    return float(velocity)
        return None

    def get_axis_resolution_mm(self, nde_data: Dict) -> Dict[str, float]:
        """Retourne un dict {'crosswise': mm, 'ultrasound': mm, 'lengthwise': mm} basé sur les métadonnées."""
        dimensions = nde_data.get("dimensions") or []
        group_info = nde_data.get("group_info")
        axis_entries: Dict[str, Dict] = {}
        for dim in dimensions:
            if not isinstance(dim, dict):
                continue
            canonical = self.canonical_axis_name(dim.get("axis"))
            if canonical:
                axis_entries[canonical] = dim

        def _linear_mm(axis_key: str) -> Optional[float]:
            entry = axis_entries.get(axis_key)
            if entry is None:
                return None
            resolution = entry.get("resolution")
            if resolution is None:
                return None
            return float(resolution) * 1000.0

        def _ultrasound_mm() -> Optional[float]:
            entry = axis_entries.get("ultrasound")
            if entry is None:
                return None
            resolution = entry.get("resolution")
            if resolution is None:
                return None
            velocity = self.extract_ultrasound_velocity(group_info)
            meters = float(resolution)
            if velocity:
                meters = meters * float(velocity) / 2.0
            return meters * 1000.0

        return {
            "lengthwise": _linear_mm("lengthwise") or 0.0,
            "crosswise": _linear_mm("crosswise") or 0.0,
            "ultrasound": _ultrasound_mm() or 0.0,
        }

