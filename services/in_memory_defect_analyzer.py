#!/usr/bin/env python3
"""
Service pour analyser les défauts directement depuis les données en mémoire.
Réutilise le pipeline de calculate_distances() sans dépendre de fichiers JSON.
"""
import numpy as np
import logging
import cv2
from typing import Dict, List, Tuple, Optional

from services.ascan_extractor import export_ascan_values_to_json
from services.distance_measurement import DistanceMeasurementService


class InMemoryDefectAnalyzer:
    """
    Analyse des défauts en temps réel depuis les données NDE et masques en mémoire.

    CONFIGURATION DES TYPES DE DÉFAUTS:
    Pour ajouter un nouveau type de défaut, ajouter une entrée dans DEFECT_TYPES_CONFIG.
    """

    # Configuration des types de défauts (facilite l'ajout de nouveaux types)
    DEFECT_TYPES_CONFIG = {
        'manque': {
            'id': 'manque',
            'label': 'Manque critique',
            'description': 'Position manquante dans frontwall ou backwall',
            'color_class': 1,  # Rouge (classe NPZ)
            'severity': 'critique',
            'icon': '⚠️',
            'enabled': True
        },
        'amincissement': {
            'id': 'amincissement',
            'label': 'Amincissement',
            'description': 'Épaisseur réduite (<60% de la moyenne)',
            'color_class': 4,  # Orange/Jaune (classe NPZ)
            'severity': 'moderee',
            'icon': '⬇️',
            'enabled': True
        }
        # Pour ajouter un nouveau type:
        # 'nouveau_type': {
        #     'id': 'nouveau_type',
        #     'label': 'Nom Affiché',
        #     'description': 'Description du défaut',
        #     'color_class': 6,  # Classe NPZ (1=rouge, 4=jaune, 6=vert, etc.)
        #     'severity': 'moderee',
        #     'icon': '🔸',
        #     'enabled': True
        # }
    }

    # Types de défauts (IDs pour le code)
    DEFECT_TYPE_MISSING = "manque"
    DEFECT_TYPE_THINNING = "amincissement"

    # Niveaux de sévérité (conservés pour compatibilité)
    SEVERITY_CRITICAL = "critique"
    SEVERITY_MODERATE = "moderee"

    def __init__(self):
        self.logger = logging.getLogger(__name__)

        # Seuils configurables pour l'amincissement
        self.thinning_threshold = 0.70  # 70% de la moyenne (nominal)
        self.thinning_tolerance = 0.10  # ±10% tolérance (60%-80%)
        self.effective_thinning_threshold = self.thinning_threshold - self.thinning_tolerance  # 0.60

        # Seuils pour les manques
        self.min_missing_width = 6  # Largeur minimale pour considérer un manque (>5 pixels)

    @classmethod
    def get_enabled_defect_types(cls) -> List[Dict]:
        """
        Retourne la liste des types de défauts activés.

        Returns:
            Liste des configurations de types de défauts activés
        """
        return [config for config in cls.DEFECT_TYPES_CONFIG.values() if config.get('enabled', True)]

    @classmethod
    def get_defect_type_config(cls, defect_type_id: str) -> Optional[Dict]:
        """
        Retourne la configuration d'un type de défaut.

        Args:
            defect_type_id: ID du type de défaut

        Returns:
            Configuration du type ou None
        """
        return cls.DEFECT_TYPES_CONFIG.get(defect_type_id)

    def analyze_defects(
        self,
        global_masks: List[np.ndarray],
        volume_data: np.ndarray,
        nde_data: Dict,
        orientation: str,
        transpose: bool,
        rotation_applied: bool,
        class_A: str = 'frontwall',
        class_B: str = 'backwall',
        nde_filename: str = 'unknown.nde',
        preserve_all_defects: bool = False
    ) -> Dict:
        """
        Analyse les défauts directement depuis les données en mémoire.

        Args:
            global_masks: Array des masques de segmentation
            volume_data: Données NDE brutes
            nde_data: Métadonnées NDE (dimensions, résolutions, etc.)
            orientation: Orientation des slices ('lengthwise', 'crosswise', 'ultrasound')
            transpose: Si transpose a été appliqué
            rotation_applied: Si rotation a été appliquée
            class_A: Première classe (défaut: 'frontwall')
            class_B: Deuxième classe (défaut: 'backwall')
            nde_filename: Nom du fichier NDE

        Returns:
            Dict contenant les défauts détectés, statistiques et overlay NPZ
        """
        try:
            self.logger.info("=== ANALYSE DES DÉFAUTS EN MÉMOIRE ===")

            # Étape 1: Extraire les valeurs A-scan
            self.logger.info("Extraction des valeurs A-scan...")
            ascan_data = export_ascan_values_to_json(
                global_masks_array=global_masks,
                volume_data=volume_data,
                orientation=orientation,
                transpose=transpose,
                rotation_applied=rotation_applied,
                nde_filename=nde_filename
            )

            # Étape 2: Calculer les distances
            self.logger.info("Calcul des distances entre classes...")
            dimensions = nde_data.get('dimensions', [])
            resolution_crosswise = dimensions[1].get('resolution', 1.0) if len(dimensions) >= 3 else 1.0
            resolution_ultrasound = dimensions[2].get('resolution', 1.0) if len(dimensions) >= 3 else 1.0

            distance_service = DistanceMeasurementService()
            distance_results = distance_service.measure_distance_all_endviews(
                ascan_data=ascan_data,
                class_A=class_A,
                class_B=class_B,
                resolution_crosswise=resolution_crosswise,
                resolution_ultrasound=resolution_ultrasound
            )

            return self._finalize_analysis(
                distance_results=distance_results,
                global_masks=global_masks,
                class_A=class_A,
                class_B=class_B,
                nde_filename=nde_filename,
                preserve_all_defects=preserve_all_defects
            )

        except Exception as e:
            self.logger.error(f"Erreur lors de l'analyse des défauts: {e}")
            import traceback
            traceback.print_exc()
            return {
                'status': 'error',
                'error': str(e),
                'defects': [],
                'statistics': {},
                'overlay': None
            }

    def analyze_from_distance_results(
        self,
        distance_results: Dict,
        global_masks: List[np.ndarray],
        class_A: str = 'frontwall',
        class_B: str = 'backwall',
        nde_filename: str = 'unknown.nde',
        preserve_all_defects: bool = False
    ) -> Dict:
        """Analyse les défauts à partir de résultats de distance déjà calculés."""
        try:
            return self._finalize_analysis(
                distance_results=distance_results,
                global_masks=global_masks,
                class_A=class_A,
                class_B=class_B,
                nde_filename=nde_filename,
                preserve_all_defects=preserve_all_defects
            )
        except Exception as e:
            self.logger.error(f"Erreur lors de l'analyse (données distance existantes): {e}")
            import traceback
            traceback.print_exc()
            return {
                'status': 'error',
                'error': str(e),
                'defects': [],
                'statistics': {},
                'overlay': None
            }

    def _finalize_analysis(
        self,
        distance_results: Dict,
        global_masks: List[np.ndarray],
        class_A: str,
        class_B: str,
        nde_filename: str,
        preserve_all_defects: bool = False
    ) -> Dict:
        """Pipeline commun de détection/statistiques/overlay à partir des distances."""

        self.logger.info("Analyse des défauts...")
        defects = []

        # Détecter les positions manquantes
        defects.extend(self._detect_missing_positions(distance_results, class_A, class_B))

        # Détecter les défauts d'épaisseur
        defects.extend(self._detect_thickness_defects(distance_results))

        if not preserve_all_defects:
            defects = self._merge_adjacent_defects(defects)
            defects = self._filter_small_missing_defects(defects)

        # Trier les défauts par sévérité
        defects = self._sort_defects(defects)

        # Générer les statistiques
        statistics = self._generate_statistics(defects, distance_results)

        # Étape Overlay
        self.logger.info("Génération de l'overlay NPZ...")
        image_shape = global_masks[0].shape if global_masks else (0, 0)
        num_endviews = len(global_masks)
        overlay_raw = self._generate_defect_overlay(defects, (num_endviews, *image_shape))

        overlay = overlay_raw
        if overlay.size > 0:
            overlay = np.stack([np.rot90(slice_img, k=1) for slice_img in overlay], axis=0)

        result = {
            'defects': defects,
            'statistics': statistics,
            'overlay': overlay,
            'metadata': {
                'nde_file': nde_filename,
                'class_A': class_A,
                'class_B': class_B,
                'num_endviews': num_endviews,
                'num_defects': len(defects),
                'parameters': {
                    'thinning_threshold': self.thinning_threshold,
                    'thinning_tolerance': self.thinning_tolerance,
                    'effective_thinning_threshold': self.effective_thinning_threshold,
                    'min_missing_width': self.min_missing_width
                },
                'defect_types': [
                    {
                        'id': config['id'],
                        'label': config['label'],
                        'enabled': config.get('enabled', True)
                    }
                    for config in self.DEFECT_TYPES_CONFIG.values()
                    if config.get('enabled', True)
                ]
            },
            'status': 'success'
        }

        self.logger.info(f"Analyse terminée: {len(defects)} défauts détectés")
        return result

    def _detect_missing_positions(self, distance_results: Dict, class_A: str, class_B: str) -> List[Dict]:
        """Détecte les positions manquantes critiques."""
        defects = []
        defect_config = self.DEFECT_TYPES_CONFIG[self.DEFECT_TYPE_MISSING]

        for endview_id_str, endview_result in distance_results.get('endviews', {}).items():
            endview_id = int(endview_id_str)

            # Positions manquantes dans classe A
            for x in endview_result.get('missing_in_A', []):
                defects.append({
                    "type": self.DEFECT_TYPE_MISSING,
                    "severity": defect_config['severity'],
                    "endview_id": endview_id,
                    "x": x,
                    "missing_class": class_A,
                    "present_class": class_B,
                    "description": f"Manque dans {class_A} à X={x}",
                    "recommendation": f"Inspection manuelle requise - {class_A} absent(e) à cette position"
                })

            # Positions manquantes dans classe B
            for x in endview_result.get('missing_in_B', []):
                defects.append({
                    "type": self.DEFECT_TYPE_MISSING,
                    "severity": defect_config['severity'],
                    "endview_id": endview_id,
                    "x": x,
                    "missing_class": class_B,
                    "present_class": class_A,
                    "description": f"Manque dans {class_B} à X={x}",
                    "recommendation": f"Inspection manuelle requise - {class_B} absent(e) à cette position"
                })

        self.logger.info(f"Détecté {len(defects)} positions manquantes")
        return defects

    def _detect_thickness_defects(self, distance_results: Dict) -> List[Dict]:
        """Détecte les défauts d'épaisseur (amincissement uniquement)."""
        defects = []
        defect_config = self.DEFECT_TYPES_CONFIG[self.DEFECT_TYPE_THINNING]

        for endview_id_str, endview_result in distance_results.get('endviews', {}).items():
            if endview_result.get('status') != 'success':
                continue

            endview_id = int(endview_id_str)
            measurements = endview_result.get('measurements', [])
            statistics = endview_result.get('statistics', {})
            mean_distance = statistics.get('mean_distance_mm')

            if mean_distance is None or mean_distance <= 0:
                continue

            # Trier par X
            measurements_sorted = sorted(
                [m for m in measurements if m.get('status') == 'success'],
                key=lambda m: m.get('x', 0)
            )

            for measurement in measurements_sorted:
                x = measurement.get('x')
                distance = measurement.get('distance_mm')

                if distance is None:
                    continue

                # Extraire les positions Y
                point_A = measurement.get('point_A', {})
                point_B = measurement.get('point_B', {})
                y_A = point_A.get('y')
                y_B = point_B.get('y')

                # Détecter amincissement avec tolérance ±10%
                # Seuil: 70% ± 10% = entre 60% et 80%
                # On ne détecte que si distance < 60% (en dessous de la zone de tolérance)
                if distance < mean_distance * self.effective_thinning_threshold:
                    percentage = (distance / mean_distance) * 100
                    defects.append({
                        "type": self.DEFECT_TYPE_THINNING,
                        "severity": defect_config['severity'],
                        "endview_id": endview_id,
                        "x": x,
                        "y_A": y_A,
                        "y_B": y_B,
                        "distance_mm": distance,
                        "mean_distance_mm": mean_distance,
                        "percentage_of_mean": percentage,
                        "description": f"Amincissement à X={x}: {percentage:.1f}% de l'épaisseur moyenne",
                        "recommendation": "Surveiller - épaisseur réduite"
                    })

        self.logger.info(f"Détecté {len(defects)} défauts d'amincissement")
        return defects

    def _merge_adjacent_defects(self, defects: List[Dict]) -> List[Dict]:
        """
        Fusionne les défauts adjacents de même type et même sévérité.

        Critères de fusion:
        - Même endview_id
        - Même type de défaut
        - Même sévérité
        - Positions X consécutives (X, X+1, X+2...)

        NOUVEAU: Fusionne aussi les défauts côte à côte même si les pourcentages
        sont différents (ex: 68% et 72% deviennent le même défaut fusionné)
        """
        if not defects:
            return defects

        # Trier par endview, type, sévérité, puis X
        sorted_defects = sorted(
            defects,
            key=lambda d: (
                d.get('endview_id', 0),
                d.get('type', ''),
                d.get('severity', ''),
                d.get('x', 0)
            )
        )

        merged = []
        current_group = [sorted_defects[0]]

        for i in range(1, len(sorted_defects)):
            prev = current_group[-1]
            curr = sorted_defects[i]

            # Vérifier si on peut fusionner avec le groupe actuel
            can_merge = (
                curr.get('endview_id') == prev.get('endview_id') and
                curr.get('type') == prev.get('type') and
                curr.get('severity') == prev.get('severity') and
                curr.get('x', 0) == prev.get('x', 0) + 1  # X consécutif
            )

            if can_merge:
                # Ajouter au groupe actuel
                current_group.append(curr)
            else:
                # Finaliser le groupe actuel et créer un défaut fusionné
                merged_defect = self._create_merged_defect(current_group)
                merged.append(merged_defect)

                # Commencer un nouveau groupe
                current_group = [curr]

        # Ne pas oublier le dernier groupe
        if current_group:
            merged_defect = self._create_merged_defect(current_group)
            merged.append(merged_defect)

        self.logger.info(f"Fusion: {len(defects)} défauts → {len(merged)} défauts après regroupement")
        return merged

    def _create_merged_defect(self, defect_group: List[Dict]) -> Dict:
        """
        Crée un défaut fusionné à partir d'un groupe de défauts adjacents.
        """
        if len(defect_group) == 1:
            # Pas de fusion nécessaire, retourner tel quel
            return defect_group[0]

        # Prendre les attributs communs du premier défaut
        first = defect_group[0]
        merged = {
            'type': first.get('type'),
            'severity': first.get('severity'),
            'endview_id': first.get('endview_id'),
        }

        # Positions X
        x_positions = [d.get('x') for d in defect_group if d.get('x') is not None]
        merged['x_start'] = min(x_positions)
        merged['x_end'] = max(x_positions)
        merged['x_positions'] = sorted(x_positions)
        merged['width'] = len(x_positions)

        # Pour la navigation, utiliser le X central
        merged['x'] = x_positions[len(x_positions) // 2]

        # Coordonnées Y (pour le tracé de lignes)
        # Prendre le min/max de tous les y_A et y_B
        y_A_values = [d.get('y_A') for d in defect_group if d.get('y_A') is not None]
        y_B_values = [d.get('y_B') for d in defect_group if d.get('y_B') is not None]

        if y_A_values:
            merged['y_A'] = min(y_A_values)
        if y_B_values:
            merged['y_B'] = max(y_B_values)

        # Classes manquantes (pour positions manquantes)
        if 'missing_class' in first:
            merged['missing_class'] = first['missing_class']
        if 'present_class' in first:
            merged['present_class'] = first['present_class']

        # Distances (calculer la moyenne)
        distances = [d.get('distance_mm') for d in defect_group if d.get('distance_mm') is not None]
        if distances:
            merged['distance_mm'] = sum(distances) / len(distances)
            merged['min_distance_mm'] = min(distances)
            merged['max_distance_mm'] = max(distances)

        if 'mean_distance_mm' in first:
            merged['mean_distance_mm'] = first['mean_distance_mm']

        if 'percentage_of_mean' in first:
            percentages = [d.get('percentage_of_mean') for d in defect_group if d.get('percentage_of_mean') is not None]
            if percentages:
                merged['percentage_of_mean'] = sum(percentages) / len(percentages)

        # Description mise à jour
        if merged['width'] == 1:
            merged['description'] = first.get('description', '')
        else:
            # Pour les manques, inclure la classe manquante dans la description
            if 'missing_class' in first:
                missing_class = first.get('missing_class', '')
                merged['description'] = f"Manque dans {missing_class}: zone X={merged['x_start']} à X={merged['x_end']} ({merged['width']} pixels)"
            else:
                # Pour les autres types (amincissement, etc.)
                merged['description'] = f"{first.get('type', 'Défaut')}: zone X={merged['x_start']} à X={merged['x_end']} ({merged['width']} pixels)"

        # Recommandation
        merged['recommendation'] = first.get('recommendation', '')

        return merged

    def _sort_defects(self, defects: List[Dict]) -> List[Dict]:
        """Trie les défauts par sévérité puis par position."""
        severity_order = {
            self.SEVERITY_CRITICAL: 0,
            self.SEVERITY_MODERATE: 1
        }

        return sorted(
            defects,
            key=lambda d: (
                severity_order.get(d.get('severity'), 999),
                d.get('endview_id', 0),
                d.get('x', 0)
            )
        )

    def _filter_small_missing_defects(self, defects: List[Dict]) -> List[Dict]:
        """
        Filtre les défauts de type 'manque critique' dont la largeur est ≤ 5 pixels.

        Règle: Un manque de 1 à 5 pixels n'est pas considéré comme un vrai défaut.
        Seuls les manques de 6 pixels ou plus (width >= 6) sont conservés.
        """
        filtered = []
        removed_count = 0

        for defect in defects:
            # Si ce n'est pas un défaut de type manque, on le garde
            if defect.get('type') != self.DEFECT_TYPE_MISSING:
                filtered.append(defect)
                continue

            # Pour les manques, vérifier la largeur
            width = defect.get('width', 1)  # Par défaut 1 si non fusionné

            if width >= self.min_missing_width:
                # Manque suffisamment large, on le garde
                filtered.append(defect)
            else:
                # Manque trop petit (1-5 pixels), on le retire
                removed_count += 1
                self.logger.debug(
                    f"Défaut manquant filtré (trop petit): endview={defect.get('endview_id')} "
                    f"x={defect.get('x')} width={width}"
                )

        self.logger.info(f"Filtre des petits manques: {removed_count} défauts retirés, {len(filtered)} conservés")
        return filtered

    def _generate_statistics(self, defects: List[Dict], distance_results: Dict) -> Dict:
        """Génère des statistiques globales."""
        defect_counts = {}
        severity_counts = {}

        for defect in defects:
            dtype = defect.get('type', 'unknown')
            severity = defect.get('severity', 'unknown')

            defect_counts[dtype] = defect_counts.get(dtype, 0) + 1
            severity_counts[severity] = severity_counts.get(severity, 0) + 1

        num_endviews = distance_results.get('metadata', {}).get('num_endviews', 1)
        critical_count = severity_counts.get(self.SEVERITY_CRITICAL, 0)
        moderate_count = severity_counts.get(self.SEVERITY_MODERATE, 0)

        penalty = (critical_count * 0.5 + moderate_count * 0.2)
        integrity_score = max(0.0, min(1.0, 1.0 - (penalty / max(1, num_endviews * 10))))

        if integrity_score >= 0.9:
            status = "excellent"
        elif integrity_score >= 0.7:
            status = "bon"
        elif integrity_score >= 0.5:
            status = "acceptable"
        elif integrity_score >= 0.3:
            status = "attention_requise"
        else:
            status = "critique"

        return {
            "total_defects": len(defects),
            "defects_by_type": defect_counts,
            "defects_by_severity": severity_counts,
            "integrity_score": round(integrity_score, 3),
            "status": status,
            "critical_defects": critical_count,
            "moderate_defects": moderate_count
        }

    def _generate_defect_overlay(self, defects: List[Dict], nde_shape: Tuple[int, int, int]) -> np.ndarray:
        """
        Génère un overlay NPZ avec les défauts marqués.
        IMPORTANT: Pas de transformation - les coordonnées sont déjà dans le bon système.
        """
        num_endviews, height, width = nde_shape
        overlay = np.zeros(nde_shape, dtype=np.uint8)

        for defect in defects:
            try:
                endview_id = int(defect.get('endview_id', 0))
                defect_type = defect.get('type')

                if endview_id >= num_endviews:
                    continue

                # Récupérer la couleur depuis la configuration du type
                defect_config = self.get_defect_type_config(defect_type)
                if defect_config is None:
                    self.logger.warning(f"Type de défaut inconnu: {defect_type}, défaut ignoré")
                    continue

                defect_class = defect_config['color_class']

                # Récupérer les positions X (défaut simple ou fusionné)
                x_positions = defect.get('x_positions')
                if x_positions is None:
                    # Défaut simple (non fusionné)
                    x = defect.get('x')
                    if x is None or x >= width:
                        continue
                    x_positions = [x]

                # Récupérer les coordonnées Y
                y_A = defect.get('y_A')
                y_B = defect.get('y_B')

                # Tracer des lignes verticales pour chaque position X
                if y_A is not None and y_B is not None:
                    y_A_int = int(round(y_A))
                    y_B_int = int(round(y_B))

                    if 0 <= y_A_int < height and 0 <= y_B_int < height:
                        temp_mask = overlay[endview_id].copy()

                        # Tracer une ligne pour chaque X
                        for x in x_positions:
                            if 0 <= x < width:
                                cv2.line(
                                    temp_mask,
                                    (x, y_A_int),
                                    (x, y_B_int),
                                    color=defect_class,
                                    thickness=2
                                )

                        overlay[endview_id] = temp_mask
                else:
                    # Position manquante - marquer toutes les colonnes
                    for x in x_positions:
                        if 0 <= x < width:
                            overlay[endview_id, :, x] = defect_class

            except Exception as e:
                self.logger.warning(f"Erreur lors du marquage du défaut: {e}")

        self.logger.info(f"Overlay généré avec {np.count_nonzero(overlay)} pixels marqués")
        return overlay

    def generate_defects_overlay_by_type_and_endview(
        self,
        defects: List[Dict],
        defect_type: str,
        endview_id: int,
        nde_shape: Tuple[int, int, int]
    ) -> np.ndarray:
        """
        Génère un overlay NPZ avec TOUS les défauts d'un type donné sur un endview.
        Utilisé pour afficher tous les défauts du même type en une seule fois.

        Args:
            defects: Liste de tous les défauts
            defect_type: Type de défaut à afficher (ex: 'amincissement_modere')
            endview_id: ID de l'endview à afficher
            nde_shape: Shape du volume NDE (num_endviews, height, width)

        Returns:
            Array numpy avec tous les défauts du type marqués
        """
        self.logger.info(f"[2D-DEFECT-GROUP] Génération overlay groupé | type={defect_type} endview={endview_id} nde_shape={nde_shape}")

        _, height, width = nde_shape
        overlay = np.zeros(nde_shape, dtype=np.uint8)

        # Récupérer la classe de couleur depuis la configuration du type
        defect_config = self.get_defect_type_config(defect_type)
        if defect_config is None:
            self.logger.warning(f"[2D-DEFECT-GROUP] Type de défaut inconnu: {defect_type}")
            return overlay

        default_color_class = defect_config['color_class']

        # Filtrer les défauts du type et endview demandés
        matching_defects = [
            d for d in defects
            if d.get('type') == defect_type and int(d.get('endview_id', -1)) == endview_id
        ]

        self.logger.info(f"[2D-DEFECT-GROUP] {len(matching_defects)} défaut(s) trouvé(s) pour type={defect_type} endview={endview_id}")

        if not matching_defects:
            # Pas de défauts, retourner overlay vide avec rotation
            if overlay.size > 0:
                rotated_stack = np.array([np.rot90(slice_img, k=1) for slice_img in np.array([np.fliplr(slice_img) for slice_img in overlay])])
                return rotated_stack
            return overlay

        # Tracer tous les défauts sur le même overlay
        for defect in matching_defects:
            try:
                # Utiliser la couleur du type de défaut
                defect_class = default_color_class

                # Récupérer les positions X
                x_positions = defect.get('x_positions')
                if x_positions is None:
                    x = defect.get('x')
                    if x is None or x >= width:
                        continue
                    x_positions = [x]

                # Récupérer les coordonnées Y
                y_A = defect.get('y_A')
                y_B = defect.get('y_B')

                if y_A is not None and y_B is not None:
                    y_A_int = int(round(y_A))
                    y_B_int = int(round(y_B))

                    if 0 <= y_A_int < height and 0 <= y_B_int < height:
                        temp_mask = overlay[endview_id].copy()

                        # Tracer une ligne pour chaque X
                        for x in x_positions:
                            if 0 <= x < width:
                                cv2.line(
                                    temp_mask,
                                    (y_A_int, x),
                                    (y_B_int, x),
                                    color=defect_class,
                                    thickness=3
                                )

                        overlay[endview_id] = temp_mask
                else:
                    # Position manquante - marquer colonnes
                    for x in x_positions:
                        if 0 <= x < width:
                            overlay[endview_id, x, :] = defect_class

            except Exception as e:
                self.logger.warning(f"[2D-DEFECT-GROUP] Erreur lors du marquage défaut: {e}")

        # Appliquer les transformations (flip + rotation)
        if overlay.size > 0:
            flipped = np.array([np.fliplr(slice_img) for slice_img in overlay])
            rotated_stack = np.array([np.rot90(slice_img, k=1) for slice_img in flipped])

            nonzero_count = int((rotated_stack != 0).sum())
            self.logger.info(f"[2D-DEFECT-GROUP] Overlay groupé généré | nonzero_total={nonzero_count}")
            return rotated_stack

        return overlay

    def generate_single_defect_overlay(
        self,
        defect: Dict,
        nde_shape: Tuple[int, int, int]
    ) -> np.ndarray:
        """
        Génère un overlay NPZ avec UN SEUL défaut marqué.
        Utilisé pour afficher un défaut à la fois dans la vue 2D.

        Args:
            defect: Dictionnaire du défaut à afficher
            nde_shape: Shape du volume NDE (num_endviews, height, width)

        Returns:
            Array numpy avec le défaut marqué
        """
        self.logger.info(f"[2D-DEFECT] Génération overlay individuel | nde_shape={nde_shape}")
        self.logger.debug(f"[2D-DEFECT] Paramètres défaut: {defect}")

        num_endviews, height, width = nde_shape
        overlay = np.zeros(nde_shape, dtype=np.uint8)

        # Mapper sévérité → classe NPZ (10 classes: 0-9)
        severity_to_class = {
            self.SEVERITY_CRITICAL: 1,   # Rouge (classe frontwall)
            self.SEVERITY_MODERATE: 4,   # Orange/Jaune (classe indication)
            self.SEVERITY_LOW: 6         # Vert clair (classe plot backwall max)
        }

        try:
            endview_id = int(defect.get('endview_id', 0))
            severity = defect.get('severity')

            if endview_id >= num_endviews:
                self.logger.warning(f"[2D-DEFECT] endview_id hors limites: {endview_id} >= {num_endviews}")
                return overlay

            defect_class = severity_to_class.get(severity, 4)

            # Récupérer les positions X (défaut simple ou fusionné)
            x_positions = defect.get('x_positions')
            if x_positions is None:
                # Défaut simple (non fusionné)
                x = defect.get('x')
                if x is None or x >= width:
                    self.logger.warning(f"[2D-DEFECT] X invalide pour overlay individuel: x={x}, width={width}")
                    return overlay
                x_positions = [x]

            # Récupérer les coordonnées Y
            y_A = defect.get('y_A')
            y_B = defect.get('y_B')

            # Tracer des lignes en tenant compte de la transposition ultérieure (ZXY)
            # Pour conserver l'axe X horizontal dans la vue finale, inverser x/y ici
            if y_A is not None and y_B is not None:
                y_A_int = int(round(y_A))
                y_B_int = int(round(y_B))

                if 0 <= y_A_int < height and 0 <= y_B_int < height:
                    temp_mask = overlay[endview_id].copy()

                    # Tracer une ligne pour chaque X (épaisseur 3 pour meilleure visibilité)
                    for x in x_positions:
                        if 0 <= x < width:
                            cv2.line(
                                temp_mask,
                                (y_A_int, x),  # swap (x,y) → (y,x) pour corriger l'axe après transpose
                                (y_B_int, x),
                                color=defect_class,
                                thickness=3  # Plus épais pour un seul défaut
                            )

                    overlay[endview_id] = temp_mask
                    try:
                        ymin, ymax = min(y_A_int, y_B_int), max(y_A_int, y_B_int)
                        self.logger.info(f"[2D-DEFECT] Ligne tracée | endview={endview_id} x_positions={x_positions} y_range=[{ymin},{ymax}] class={defect_class}")
                    except Exception:
                        pass
            else:
                # Position manquante - marquer toutes les colonnes
                for x in x_positions:
                    if 0 <= x < width:
                        # swap axes pour garder X horizontal après transpose
                        overlay[endview_id, x, :] = defect_class
                self.logger.info(f"[2D-DEFECT] Colonne(s) marquée(s) (position manquante) | endview={endview_id} x_positions={x_positions} class={defect_class}")

        except Exception as e:
            self.logger.warning(f"[2D-DEFECT] Erreur lors du marquage du défaut: {e}")

        # IMPORTANT: Ne pas appliquer de flip/transpose ici non plus.
        # On garde la même orientation/shape que les overlays de défauts complets
        try:
            nonzero_total = int((overlay != 0).sum())
            unique_vals = np.unique(overlay).tolist()
            self.logger.info(f"[2D-DEFECT] Overlay individuel généré | nonzero_total={nonzero_total} classes={unique_vals}")
        except Exception:
            pass

        # Ajustement d'orientation pour la vue 2D: flip horizontal par slice + transpose/rotation
        if overlay.size > 0:
            flipped = np.array([np.fliplr(slice_img) for slice_img in overlay])
            # Transpose ZXY (Z, H, W) -> (Z, W, H)
            transposed = flipped.transpose((0, 2, 1))
            # Alternative rotation 90° par slice (pour tester un alignement différent)
            rotated_stack = np.array([np.rot90(slice_img, k=1) for slice_img in flipped])

            try:
                nz_after_t = int((transposed != 0).sum())
                nz_after_r = int((rotated_stack != 0).sum())
                self.logger.info(
                    f"[2D-DEFECT] Flip+Transpose vs Rotate | shape_t={transposed.shape} "
                    f"shape_r={rotated_stack.shape} nonzero_t={nz_after_t} nonzero_r={nz_after_r}"
                )
            except Exception:
                pass

            # Retourner la version tournée (k=-1) pour prendre en compte la rotation ajoutée
            return rotated_stack
        return overlay
