#!/usr/bin/env python3
"""
Exporteur JSON pour générer des fichiers d'annotation compatibles avec Kili et réutilisables.
Génère le même format que HydroFORM_Gen2_Demo_2_gr1.json.
"""

import json
import os
import time
from datetime import datetime
from typing import Dict, List, Tuple, Any
import numpy as np
import cv2
import logging
from pathlib import Path
from utils.morphology import smooth_mask_contours
from services.profile import generate_profile_mask


class JsonExporter:
    """Exporteur pour générer des fichiers JSON d'annotations."""
    
    def __init__(self, output_folder=None):
        """
        output_folder: dossier où écrire le JSON (pour export).
                       Peut être None si on ne fait que du chargement.
        """
        self.output_folder = output_folder
        self.logger = logging.getLogger(__name__)
        
        # Mapping des labels vers les noms de catégories
        self.label_categories = {
            "frontwall": "FRONTWALL",
            "backwall": "BACKWALL", 
            "flaw": "FLAW",
            "indication": "INDICATION"
        }
    
    def export_annotations_to_json(self, model, label_settings: Dict, output_path: str, 
                                 image_list: List[str] = None) -> str:
        """
        Exporte toutes les annotations vers un fichier JSON.
        
        Args:
            model: Modèle contenant les polygones et images
            label_settings: Paramètres individuels par label
            output_path: Chemin de sortie pour le fichier JSON
            image_list: Liste optionnelle des images à traiter
            
        Returns:
            Chemin du fichier JSON généré
        """
        try:
            # Utiliser toutes les images si aucune liste fournie
            if image_list is None:
                image_list = model.image_list
            
            # Structure principale du JSON
            json_data = {}
            
            # Traiter chaque image
            for image_index, image_path in enumerate(image_list):
                self.logger.info(f"Traitement image {image_index + 1}/{len(image_list)}: {os.path.basename(image_path)}")
                
                # Charger l'image pour obtenir ses dimensions
                image = cv2.imread(image_path)
                if image is None:
                    self.logger.warning(f"Impossible de charger l'image: {image_path}")
                    continue
                
                height, width = image.shape[:2]
                
                # Obtenir les polygones pour cette image
                # Note: Pour l'instant, on utilise les polygones actuels du modèle
                # Dans une version complète, il faudrait charger les polygones spécifiques à chaque image
                polygons = model.get_all_polygons()
                
                # Générer les annotations pour cette image (avec données image pour threshold)
                annotations = self._generate_annotations_for_image(
                    polygons, label_settings, width, height, image_index, image
                )
                
                # Ajouter à la structure JSON si des annotations existent
                if annotations:
                    json_data[str(image_index)] = {
                        "OBJECT_DETECTION_JOB": {
                            "annotations": annotations
                        }
                    }
            
            # Sauvegarder le fichier JSON
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(json_data, f, indent=4, ensure_ascii=False)
            
            self.logger.info(f"Fichier JSON exporté: {output_path}")
            self.logger.info(f"Nombre d'images avec annotations: {len(json_data)}")
            
            return output_path
            
        except Exception as e:
            self.logger.error(f"Erreur lors de l'export JSON: {str(e)}")
            raise
    
    def _generate_annotations_for_image(self, polygons: Dict, label_settings: Dict,
                                      width: int, height: int, image_index: int,
                                      image_data: np.ndarray) -> List[Dict]:
        """
        Génère les annotations pour une image spécifique.
        Utilise la même logique de construction des polygones que l'application.

        Args:
            polygons: Dictionnaire des polygones par label
            label_settings: Paramètres par label
            width: Largeur de l'image
            height: Hauteur de l'image
            image_index: Index de l'image

        Returns:
            Liste des annotations pour cette image
        """
        annotations = []
        annotation_counter = 0

        # Traiter chaque label
        for label, label_polygons in polygons.items():
            if not label_polygons:  # Pas de polygones pour ce label
                continue

            # Obtenir les paramètres pour ce label
            settings = label_settings.get(label, {})
            mask_type = settings.get("mask_type", "polygon")

            # Traiter chaque polygone de ce label
            for poly_index, polygon_points in enumerate(label_polygons):
                if len(polygon_points) < 3:  # Polygone invalide
                    continue

                # Générer un ID unique pour cette annotation
                timestamp = int(time.time() * 1000)
                mid = f"{timestamp}-{annotation_counter}"
                annotation_counter += 1

                # Extraire les contours du masque final (avec threshold + profil + lissage)
                final_contour_points = self._extract_final_mask_contours(
                    polygon_points, width, height, settings, image_data
                )

                if final_contour_points:
                    # Utiliser les contours du masque final
                    normalized_vertices = self._normalize_polygon_coordinates(
                        final_contour_points, width, height
                    )
                else:
                    # Fallback vers les points originaux si extraction échoue
                    self.logger.warning(f"Fallback vers points originaux pour {label}")
                    normalized_vertices = self._normalize_polygon_coordinates(
                        polygon_points, width, height
                    )

                # Créer l'annotation
                annotation = {
                    "children": {},
                    "isKeyFrame": True,
                    "categories": [
                        {
                            "name": self.label_categories.get(label, label.upper())
                        }
                    ],
                    "mid": mid,
                    "type": "semantic",
                    "boundingPoly": [
                        {
                            "normalizedVertices": normalized_vertices
                        }
                    ]
                }

                annotations.append(annotation)

                self.logger.debug(f"Annotation créée: {label} ({mask_type}) avec {len(normalized_vertices)} points")

        return annotations

    def _extract_final_mask_contours(self, polygon_points: List[Tuple], width: int, height: int,
                                   settings: Dict, image_data: np.ndarray) -> List[Tuple]:
        """
        Extrait les contours du masque final après application complète de l'algorithme.
        Suit exactement la même logique que l'application : threshold + profil + lissage.

        Args:
            polygon_points: Points du polygone original
            width: Largeur de l'image
            height: Hauteur de l'image
            settings: Paramètres du label (threshold, mask_type, smooth_contours)
            image_data: Données de l'image pour le threshold

        Returns:
            Liste des points du contour final [(x, y), ...]
        """
        try:
            import numpy as np
            import cv2

            # 1. APPLIQUER LE THRESHOLD BINAIRE (même logique que l'application)
            threshold = settings.get("threshold", 150)
            gray = cv2.cvtColor(image_data, cv2.COLOR_BGR2GRAY)
            binary_mask = (gray < threshold).astype(np.uint8)

            # 2. CRÉER LE MASQUE AVEC LE POLYGONE ORIGINAL
            polygon_mask = np.zeros((height, width), dtype=np.uint8)
            points = np.array(polygon_points, np.int32)
            cv2.fillPoly(polygon_mask, [points], 1)

            # 3. APPLIQUER LE THRESHOLD AU POLYGONE (étape cruciale manquante !)
            polygon_mask &= binary_mask

            # 4. APPLIQUER L'ALGORITHME SELON LE TYPE DE MASQUE
            mask_type = settings.get("mask_type", "polygon")

            if mask_type == "polygon":
                # MASQUE POLYGONAL : Générer le profil automatique
                final_mask = self._generate_profile_mask_for_export(polygon_mask, 1)
            else:
                # MASQUE STANDARD : Utiliser directement le polygone avec threshold
                final_mask = polygon_mask.copy()

            # 5. APPLIQUER LE LISSAGE SI ACTIVÉ (même logique que l'application)
            smooth_enabled = settings.get("smooth_contours", False)
            if smooth_enabled:
                final_mask = smooth_mask_contours(final_mask, kernel_size=5)

            # 6. EXTRAIRE LES CONTOURS DU MASQUE FINAL
            contours, _ = cv2.findContours(final_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            if not contours:
                self.logger.warning("Aucun contour trouvé dans le masque final")
                return []

            # Prendre le plus grand contour (le principal)
            largest_contour = max(contours, key=cv2.contourArea)

            # Convertir en liste de points
            contour_points = [(int(point[0][0]), int(point[0][1])) for point in largest_contour]

            self.logger.debug(f"Contour final extrait: {len(contour_points)} points (threshold={threshold}, type={mask_type}, smooth={smooth_enabled})")

            return contour_points

        except Exception as e:
            self.logger.warning(f"Erreur lors de l'extraction du contour final: {str(e)}")
            return []  # Retourner une liste vide en cas d'erreur

    def _generate_profile_mask_for_export(self, single_mask, val):
        """
        Génère un masque de profil pour l'export (même logique que l'application).
        Respecte strictement le contour original et applique la verticalisation.

        Args:
            single_mask: Masque binaire du polygone avec threshold appliqué
            val: Valeur du label (1, 2, 3, 4)

        Returns:
            Masque avec profil généré
        """
        try:
            # Créer un masque temporaire pour la fusion
            fusion_mask = np.zeros_like(single_mask)
            
            # Trouver les contours du masque binaire
            contours, _ = cv2.findContours(single_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            
            if not contours:
                return single_mask
            
            # Fusionner tous les contours en un seul
            if len(contours) > 1:
                # Remplir tous les contours dans le masque de fusion
                cv2.drawContours(fusion_mask, contours, -1, 1, -1)
                # Trouver le contour externe de la fusion
                contours, _ = cv2.findContours(fusion_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            
            # Créer le masque final
            profile_mask = np.zeros_like(single_mask)
            
            # Pour chaque contour (maintenant fusionné)
            for contour in contours:
                # Remplir le contour avec la valeur du label
                cv2.fillPoly(profile_mask, [contour], val)
            
            # Appliquer la verticalisation pixel par pixel
            h, w = profile_mask.shape
            # Pour chaque colonne x
            for x in range(w):
                # Trouver tous les pixels non-nuls dans cette colonne
                col = profile_mask[:, x]
                idx = np.where(col == val)[0]
                if idx.size > 0:
                    # Prendre le plus petit et le plus grand y
                    y_min = idx.min()
                    y_max = idx.max()
                    # Remplir verticalement entre ces points
                    profile_mask[y_min:y_max+1, x] = val
            
            return profile_mask
            
        except Exception as e:
            self.logger.error(f"Erreur lors de la génération du masque de profil pour l'export: {str(e)}")
            return single_mask

    def _normalize_polygon_coordinates(self, polygon_points: List[Tuple],
                                     width: int, height: int) -> List[Dict]:
        """
        Convertit les coordonnées de polygone en coordonnées normalisées (0-1).

        Args:
            polygon_points: Liste des points du polygone [(x, y), ...]
            width: Largeur de l'image
            height: Hauteur de l'image

        Returns:
            Liste des vertices normalisés [{"x": 0.5, "y": 0.3}, ...]
        """
        normalized_vertices = []

        for point in polygon_points:
            x, y = point

            # Normaliser les coordonnées directement (0-1)
            # Les contours extraits du masque sont déjà dans le bon système de coordonnées
            normalized_x = max(0.0, min(1.0, x / width))
            normalized_y = max(0.0, min(1.0, y / height))

            normalized_vertices.append({
                "x": normalized_x,
                "y": normalized_y
            })

        return normalized_vertices
    
    def load_annotations_from_json(self, json_path: str) -> Dict:
        """
        Charge les annotations depuis un fichier JSON.
        
        Args:
            json_path: Chemin vers le fichier JSON
            
        Returns:
            Dictionnaire des annotations chargées
        """
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            self.logger.info(f"Annotations chargées depuis: {json_path}")
            self.logger.info(f"Nombre d'images: {len(data)}")
            
            return data
            
        except Exception as e:
            self.logger.error(f"Erreur lors du chargement JSON: {str(e)}")
            raise
    
    def convert_json_to_polygons(self, json_data: Dict, image_width: int,
                               image_height: int, image_index: int = 0, margin: int = 0) -> Dict:
        """
        Convertit les annotations JSON en polygones utilisables par l'application.

        Args:
            json_data: Données JSON chargées
            image_width: Largeur de l'image
            image_height: Hauteur de l'image
            image_index: Index de l'image à traiter
            margin: Marge (ignorée, conservée pour compatibilité)

        Returns:
            Dictionnaire des polygones par label
        """
        polygons = {"frontwall": [], "backwall": [], "flaw": [], "indication": []}
        
        try:
            # Obtenir les annotations pour cette image
            image_key = str(image_index)
            if image_key not in json_data:
                self.logger.warning(f"Aucune annotation trouvée pour l'image {image_index}")
                return polygons
            
            annotations = json_data[image_key]["OBJECT_DETECTION_JOB"]["annotations"]
            
            # Traiter chaque annotation
            for annotation in annotations:
                try:
                    # Validation et obtention de la catégorie
                    if "categories" not in annotation or not isinstance(annotation["categories"], list) or len(annotation["categories"]) == 0:
                        self.logger.warning(f"Annotation sans catégorie valide: {annotation}")
                        continue

                    if "name" not in annotation["categories"][0]:
                        self.logger.warning(f"Catégorie sans nom: {annotation['categories'][0]}")
                        continue

                    category_name = annotation["categories"][0]["name"]

                    # Mapper vers notre système de labels
                    label = self._map_category_to_label(category_name)
                    if label not in polygons:
                        self.logger.warning(f"Label inconnu: {label}")
                        continue

                    # Validation et extraction des coordonnées normalisées
                    if "boundingPoly" not in annotation or not isinstance(annotation["boundingPoly"], list) or len(annotation["boundingPoly"]) == 0:
                        self.logger.warning(f"Annotation sans boundingPoly valide: {annotation}")
                        continue

                    if "normalizedVertices" not in annotation["boundingPoly"][0]:
                        self.logger.warning(f"boundingPoly sans normalizedVertices: {annotation['boundingPoly'][0]}")
                        continue

                    normalized_vertices = annotation["boundingPoly"][0]["normalizedVertices"]

                    if not isinstance(normalized_vertices, list):
                        self.logger.warning(f"normalizedVertices n'est pas une liste: {normalized_vertices}")
                        continue

                    # Convertir en coordonnées pixel avec validation et ajustement de marge
                    pixel_coordinates = []
                    for vertex in normalized_vertices:
                        # Validation du vertex
                        if not isinstance(vertex, dict) or "x" not in vertex or "y" not in vertex:
                            self.logger.warning(f"Vertex invalide: {vertex}")
                            continue

                        # Convertir les coordonnées normalisées en coordonnées pixel
                        # Utiliser round() au lieu de int() pour éviter le décalage dû à la troncature
                        x = round(vertex["x"] * image_width)
                        y = round(vertex["y"] * image_height)

                        # Utiliser les coordonnées directement (pas de marge)
                        pixel_coordinates.append((x, y))

                    # Valider et nettoyer le polygone
                    cleaned_polygon = self._validate_imported_polygon(pixel_coordinates)

                    if len(cleaned_polygon) >= 3:  # Polygone valide
                        polygons[label].append(cleaned_polygon)
                        self.logger.debug(f"Polygone importé: {label} avec {len(cleaned_polygon)} points")
                    else:
                        self.logger.warning(f"Polygone invalide ignoré: {label} avec {len(pixel_coordinates)} points")

                    self.logger.debug(f"Polygone converti: {label} avec {len(pixel_coordinates)} points")

                except Exception as e:
                    self.logger.error(f"Erreur lors du traitement d'une annotation: {str(e)}")
                    continue
            
            return polygons
            
        except Exception as e:
            self.logger.error(f"Erreur lors de la conversion JSON vers polygones: {str(e)}")
            return polygons
    
    def _map_category_to_label(self, category_name: str) -> str:
        """
        Mappe un nom de catégorie vers un label interne.
        
        Args:
            category_name: Nom de la catégorie (ex: "FRONTWALL")
            
        Returns:
            Label interne (ex: "frontwall")
        """
        category_to_label = {
            "FRONTWALL": "frontwall",
            "BACKWALL": "backwall",
            "FLAW": "flaw", 
            "INDICATION": "indication"
        }
        
        return category_to_label.get(category_name, category_name.lower())
    
    def export_current_image_to_json(self, model, label_settings: Dict, 
                                   output_path: str) -> str:
        """
        Exporte uniquement l'image actuelle vers JSON.
        
        Args:
            model: Modèle contenant les polygones
            label_settings: Paramètres par label
            output_path: Chemin de sortie
            
        Returns:
            Chemin du fichier généré
        """
        # Obtenir l'image actuelle
        current_image_path = model.image_list[model.current_index]
        
        return self.export_annotations_to_json(
            model, label_settings, output_path, [current_image_path]
        )

    def export_mask_to_json(self, final_mask: np.ndarray, model, label_settings: Dict,
                          output_path: str) -> str:
        """
        Exporte un masque final calculé vers JSON (utilise les contours du masque, pas les polygones originaux).

        Args:
            final_mask: Masque final avec tous les traitements appliqués
            model: Modèle contenant les informations d'image
            label_settings: Paramètres par label
            output_path: Chemin de sortie

        Returns:
            Chemin du fichier généré
        """
        try:
            # Obtenir l'image actuelle pour les dimensions
            current_image_path = model.image_list[model.current_index]
            image = cv2.imread(current_image_path)
            if image is None:
                raise ValueError(f"Impossible de charger l'image: {current_image_path}")

            height, width = image.shape[:2]

            # Extraire les contours de chaque label du masque final
            annotations = self._extract_annotations_from_mask(
                final_mask, label_settings, width, height
            )

            # Créer la structure JSON avec l'index de l'image actuelle
            image_index = getattr(model, 'current_index', 0)
            json_data = {
                str(image_index): {
                    "OBJECT_DETECTION_JOB": {
                        "annotations": annotations
                    }
                }
            }

            # Sauvegarder le fichier JSON
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(json_data, f, indent=2, ensure_ascii=False)

            self.logger.info(f"Masque exporté vers JSON: {output_path}")
            return output_path

        except Exception as e:
            self.logger.error(f"Erreur lors de l'export masque vers JSON: {str(e)}")
            raise

    def export_polygons_to_json(self, model, label_settings: Dict,
                              output_path: str) -> str:
        """
        Exporte les polygones dessinés directement vers JSON.
        Utilise exactement les polygones de l'interface, pas les contours du masque.

        Args:
            model: Modèle contenant les polygones dessinés
            label_settings: Paramètres pour chaque label
            output_path: Chemin de sortie pour le fichier JSON

        Returns:
            Chemin du fichier JSON créé
        """
        try:
            # self.logger.info(f"Export polygones vers JSON: {output_path}")

            # Obtenir les dimensions de l'image depuis le modèle (compatible mode mémoire)
            if model.current_image is not None:
                # Mode mémoire : utiliser l'image déjà chargée
                image = model.current_image
                height, width = image.shape[:2]
            else:
                # Mode fichier : charger depuis le disque
                current_image_path = model.image_list[model.current_index]
                image = cv2.imread(current_image_path)
                if image is None:
                    raise ValueError(f"Impossible de charger l'image: {current_image_path}")
                height, width = image.shape[:2]

            # Obtenir les polygones directement du modèle (exactement ce qui est dessiné)
            # Utiliser le nouveau format avec paramètres et extraire seulement les points
            polygons = model.get_all_polygons_points_only()

            # Générer les annotations à partir des polygones dessinés
            annotations = self._generate_annotations_from_polygons(
                polygons, label_settings, width, height
            )

            # Créer la structure JSON avec l'index de l'image actuelle
            image_index = getattr(model, 'current_index', 0)
            json_data = {
                str(image_index): {
                    "OBJECT_DETECTION_JOB": {
                        "annotations": annotations
                    }
                }
            }

            # Sauvegarder le fichier JSON
            os.makedirs(os.path.dirname(output_path), exist_ok=True)

            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(json_data, f, indent=2, ensure_ascii=False)

            # self.logger.info(f"JSON exporté avec succès: {output_path}")
            # self.logger.info(f"Nombre d'annotations: {len(annotations)}")

            return output_path

        except Exception as e:
            self.logger.error(f"Erreur lors de l'export polygones vers JSON: {str(e)}")
            raise

    def _generate_annotations_from_polygons(self, polygons: Dict, label_settings: Dict,
                                          width: int, height: int) -> List[Dict]:
        """
        Génère les annotations JSON à partir des polygones dessinés dans l'interface.

        Args:
            polygons: Dictionnaire des polygones par label
            label_settings: Paramètres pour chaque label
            width: Largeur de l'image
            height: Hauteur de l'image

        Returns:
            Liste des annotations JSON
        """
        annotations = []
        annotation_counter = 0

        # Mapping des labels
        label_mapping = {
            "frontwall": "FRONTWALL",
            "backwall": "BACKWALL",
            "flaw": "FLAW",
            "indication": "INDICATION"
        }

        for label, label_polygons in polygons.items():
            if label not in label_mapping:
                continue

            # Traiter chaque polygone de ce label
            for polygon_points in label_polygons:
                if len(polygon_points) < 3:  # Polygone invalide
                    continue

                # Générer un ID unique pour cette annotation
                timestamp = int(time.time() * 1000)
                mid = f"{timestamp}-{annotation_counter}"
                annotation_counter += 1

                # Utiliser les coordonnées directement (pas de marge)
                # Normaliser les coordonnées
                normalized_vertices = self._normalize_polygon_coordinates(
                    polygon_points, width, height
                )

                # Créer l'annotation JSON
                annotation = {
                    "children": {},
                    "isKeyFrame": True,
                    "categories": [{"name": label_mapping[label]}],
                    "mid": mid,
                    "type": "semantic",
                    "boundingPoly": [{
                        "normalizedVertices": normalized_vertices
                    }]
                }

                annotations.append(annotation)
                self.logger.debug(f"Annotation créée: {label_mapping[label]} avec {len(normalized_vertices)} points")

        return annotations

    def _extract_annotations_from_mask(self, final_mask: np.ndarray, label_settings: Dict,
                                     width: int, height: int) -> List[Dict]:
        """
        Extrait les annotations directement du masque final calculé.
        Utilise les contours réels du masque (avec tous les traitements appliqués).

        Args:
            final_mask: Masque final avec valeurs 1,2,3,4 pour frontwall,backwall,flaw,indication
            label_settings: Paramètres par label
            width: Largeur de l'image
            height: Hauteur de l'image

        Returns:
            Liste des annotations extraites du masque
        """
        annotations = []
        annotation_counter = 0

        # Mapping des valeurs de masque vers les labels
        label_mapping = {1: "frontwall", 2: "backwall", 3: "flaw", 4: "indication"}

        # Traiter chaque label présent dans le masque
        for mask_value, label in label_mapping.items():
            # Extraire les pixels de ce label
            label_mask = (final_mask == mask_value).astype(np.uint8)

            if not np.any(label_mask):
                continue  # Pas de pixels pour ce label

            # Extraire les contours de ce label
            contours, _ = cv2.findContours(label_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            if not contours:
                continue  # Pas de contours trouvés

            # Traiter chaque contour (il peut y avoir plusieurs régions séparées)
            for contour in contours:
                if cv2.contourArea(contour) < 10:  # Ignorer les très petits contours
                    continue

                # Générer un ID unique pour cette annotation
                timestamp = int(time.time() * 1000)
                mid = f"{timestamp}-{annotation_counter}"
                annotation_counter += 1

                # Simplifier et nettoyer le contour pour éviter les points collés
                cleaned_contour = self._clean_contour_points(contour)

                if len(cleaned_contour) < 3:  # Pas assez de points pour un polygone
                    continue

                # Normaliser les coordonnées
                normalized_vertices = self._normalize_polygon_coordinates(
                    cleaned_contour, width, height
                )

                # Créer l'annotation
                annotation = {
                    "children": {},
                    "isKeyFrame": True,
                    "categories": [
                        {
                            "name": self.label_categories.get(label, label.upper())
                        }
                    ],
                    "mid": mid,
                    "type": "semantic",
                    "boundingPoly": [
                        {
                            "normalizedVertices": normalized_vertices
                        }
                    ]
                }

                annotations.append(annotation)

                self.logger.debug(f"Contour extrait du masque: {label} avec {len(normalized_vertices)} points")

        return annotations

    def _clean_contour_points(self, contour) -> List[Tuple]:
        """
        Nettoie les points du contour pour créer des polygones simples et utilisables.

        Args:
            contour: Contour OpenCV

        Returns:
            Liste de points nettoyés [(x, y), ...]
        """
        # Appliquer une simplification modérée pour réduire le nombre de points
        # tout en préservant la forme générale
        epsilon = 0.01 * cv2.arcLength(contour, True)  # 1% de la longueur du contour
        simplified = cv2.approxPolyDP(contour, epsilon, True)

        # Convertir en liste de points
        points = [(int(point[0][0]), int(point[0][1])) for point in simplified]

        if len(points) < 3:
            return points

        # Supprimer les points dupliqués consécutifs
        cleaned_points = []
        min_distance = 2  # Distance minimale entre points

        for i, point in enumerate(points):
            if i == 0:
                cleaned_points.append(point)
            else:
                # Calculer la distance avec le point précédent
                prev_point = cleaned_points[-1]
                distance = np.sqrt((point[0] - prev_point[0])**2 + (point[1] - prev_point[1])**2)

                # Garder le point seulement s'il n'est pas trop proche du précédent
                if distance >= min_distance:
                    cleaned_points.append(point)

        # Vérifier que le polygone est fermé (dernier point = premier point)
        if len(cleaned_points) >= 3:
            first_point = cleaned_points[0]
            last_point = cleaned_points[-1]
            distance_to_start = np.sqrt((last_point[0] - first_point[0])**2 + (last_point[1] - first_point[1])**2)

            # Si le dernier point n'est pas proche du premier, fermer le polygone
            if distance_to_start > min_distance:
                cleaned_points.append(first_point)

        return cleaned_points

    def _validate_imported_polygon(self, polygon_points: List[Tuple]) -> List[Tuple]:
        """
        Valide et nettoie un polygone importé depuis JSON.

        Args:
            polygon_points: Points du polygone importé

        Returns:
            Points du polygone validé et nettoyé
        """
        if len(polygon_points) < 3:
            return []

        # Supprimer les points dupliqués consécutifs
        cleaned_points = []
        for i, point in enumerate(polygon_points):
            if i == 0 or point != polygon_points[i-1]:
                cleaned_points.append(point)

        if len(cleaned_points) < 3:
            return []

        # Vérifier si le polygone est fermé (dernier point = premier point)
        is_closed = len(cleaned_points) > 3 and cleaned_points[-1] == cleaned_points[0]
        if is_closed:
            # Supprimer le dernier point dupliqué pour éviter la duplication
            cleaned_points = cleaned_points[:-1]

        # S'assurer qu'on a encore assez de points
        if len(cleaned_points) < 3:
            return []

        # Supprimer les points trop proches pour éviter les problèmes d'affichage
        final_points = []
        min_distance = 1  # Distance minimale entre points

        for i, point in enumerate(cleaned_points):
            if i == 0:
                final_points.append(point)
            else:
                # Calculer la distance avec le point précédent
                prev_point = final_points[-1]
                distance = np.sqrt((point[0] - prev_point[0])**2 + (point[1] - prev_point[1])**2)

                if distance >= min_distance:
                    final_points.append(point)

        # Vérifier qu'on a encore un polygone valide
        if len(final_points) < 3:
            final_points = cleaned_points  # Retourner les points originaux si le nettoyage a trop réduit

        # S'assurer que le polygone est fermé pour l'affichage
        if len(final_points) >= 3 and final_points[-1] != final_points[0]:
            final_points.append(final_points[0])

        return final_points

    def export(self, data):
        # Example of using profile generator if needed
        if data.get('use_profile'):
            mask = data['mask']
            val = data['label_val']
            data['profile_mask'] = generate_profile_mask(mask, val).tolist()
        with open(os.path.join(self.output_folder, data['filename']), 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

def generate_json_filename(base_name: str = "annotations") -> str:
    """
    Génère un nom de fichier JSON avec timestamp.
    
    Args:
        base_name: Nom de base du fichier
        
    Returns:
        Nom de fichier avec timestamp
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{base_name}_{timestamp}.json"
