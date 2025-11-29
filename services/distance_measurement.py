#!/usr/bin/env python3
"""
Service pour mesurer les distances entre pixels de classes différentes.
Permet de mesurer l'épaisseur entre frontwall et backwall, par exemple.
"""

import numpy as np
import logging
from typing import Dict, List, Optional, Tuple
import math
import cv2
import time


class DistanceMeasurementService:
    """
    Service pour mesurer les distances entre pixels ayant les valeurs A-scan maximales
    de deux classes différentes, pour chaque position X.
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Flag pour activer/désactiver les logs de performance détaillés
        self.ENABLE_PERF_LOGS = True  # Mettre à False pour désactiver les logs détaillés

    def _normalize_class_key(self, class_id: str) -> str:
        """Normalize a class identifier to a comparable string key."""
        try:
            return str(int(class_id))
        except Exception:
            return str(class_id)

    def _resolve_class_key(self, endview_data: Dict, class_id: str) -> Optional[str]:
        """Find the matching key in endview_data for the requested class id/name."""
        key_normalized = self._normalize_class_key(class_id)
        if key_normalized in endview_data:
            return key_normalized
        # Fallback: keep original string if present (ex: legacy 'frontwall')
        if isinstance(class_id, str) and class_id in endview_data:
            return class_id
        return None
    
    def _build_x_index(self, pixels: List[Dict]) -> Dict[int, List[Dict]]:
        """
        OPTIMISATION: Construit un index {x: [pixels]} pour accès O(1) par position X.
        Au lieu de parcourir tous les pixels pour chaque X (O(n)), on fait un seul parcours
        initial et ensuite l'accès est O(1).
        
        Args:
            pixels: Liste de pixels avec clés 'x', 'y', 'ascan_value'
            
        Returns:
            Dict mappant chaque position X à la liste de pixels à cette position
        """
        if self.ENABLE_PERF_LOGS:
            start_time = time.perf_counter()
        
        x_index = {}
        for pixel in pixels:
            x = pixel['x']
            if x not in x_index:
                x_index[x] = []
            x_index[x].append(pixel)
        
        if self.ENABLE_PERF_LOGS:
            elapsed = (time.perf_counter() - start_time) * 1000
            self.logger.debug(f"[PERF] Index X construit: {len(pixels)} pixels → {len(x_index)} positions X uniques en {elapsed:.2f}ms")
        
        return x_index
    
    def measure_distance_at_x(
        self,
        ascan_data: Dict,
        endview_id: str,
        x: int,
        class_A: str,
        class_B: str,
        resolution_crosswise: float = 1.0,
        resolution_ultrasound: float = 1.0,
        x_index_A: Optional[Dict[int, List[Dict]]] = None,
        x_index_B: Optional[Dict[int, List[Dict]]] = None
    ) -> Dict:
        """
        Mesure la distance entre les pixels ayant les valeurs A-scan max de 2 classes à une position X.
        
        OPTIMISATION: Si x_index_A et x_index_B sont fournis, utilise l'accès O(1) au lieu de filtrer O(n).
        
        Args:
            ascan_data: Données A-scan (structure JSON complète)
            endview_id: ID de l'endview (ex: "0", "1", ...)
            x: Position X à analyser
            class_A: Nom de la première classe (ex: "frontwall")
            class_B: Nom de la deuxième classe (ex: "backwall")
            resolution_crosswise: Résolution en mm/pixel pour l'axe X (crosswise)
            resolution_ultrasound: Résolution en mm/pixel pour l'axe Y (ultrasound)
            x_index_A: Index pré-construit pour classe A (optionnel, optimisation)
            x_index_B: Index pré-construit pour classe B (optionnel, optimisation)
            
        Returns:
            Dict avec les résultats de la mesure
        """
        try:
            if self.ENABLE_PERF_LOGS:
                start_time = time.perf_counter()
            
            # Vérifier que l'endview existe
            if endview_id not in ascan_data.get('endviews', {}):
                return {
                    "x": x,
                    "status": "error",
                    "error": f"Endview {endview_id} not found"
                }
            
            endview_data = ascan_data['endviews'][endview_id]

            key_A = self._resolve_class_key(endview_data, class_A)
            key_B = self._resolve_class_key(endview_data, class_B)

            # Vérifier que les classes existent
            if key_A is None:
                return {
                    "x": x,
                    "status": "error",
                    "error": f"Class {class_A} not found in endview {endview_id}"
                }
            
            if key_B is None:
                return {
                    "x": x,
                    "status": "error",
                    "error": f"Class {class_B} not found in endview {endview_id}"
                }
            
            # OPTIMISATION: Utiliser l'index X si fourni, sinon filtrer (ancien comportement)
            if x_index_A is not None:
                pixels_A = x_index_A.get(x, [])
                if self.ENABLE_PERF_LOGS:
                    self.logger.debug(f"[PERF] Utilisation index X pour classe A: accès O(1)")
            else:
                pixels_A = [p for p in endview_data[key_A] if p['x'] == x]
                if self.ENABLE_PERF_LOGS:
                    self.logger.debug(f"[PERF] Filtrage linéaire pour classe A: O(n)")
            
            if x_index_B is not None:
                pixels_B = x_index_B.get(x, [])
                if self.ENABLE_PERF_LOGS:
                    self.logger.debug(f"[PERF] Utilisation index X pour classe B: accès O(1)")
            else:
                pixels_B = [p for p in endview_data[key_B] if p['x'] == x]
                if self.ENABLE_PERF_LOGS:
                    self.logger.debug(f"[PERF] Filtrage linéaire pour classe B: O(n)")
            
            # Vérifier l'existence
            if not pixels_A:
                return {
                    "x": x,
                    "status": "error",
                    "error": f"X={x} not found in class {class_A}"
                }
            
            if not pixels_B:
                return {
                    "x": x,
                    "status": "error",
                    "error": f"X={x} not found in class {class_B}"
                }
            
            # Trouver les pixels avec A-scan max
            max_ascan_A = max(p['ascan_value'] for p in pixels_A)
            max_ascan_B = max(p['ascan_value'] for p in pixels_B)
            
            # Récupérer tous les pixels ayant la valeur max (peut y en avoir plusieurs)
            max_pixels_A = [p for p in pixels_A if p['ascan_value'] == max_ascan_A]
            max_pixels_B = [p for p in pixels_B if p['ascan_value'] == max_ascan_B]
            
            # Calculer la position moyenne si plusieurs pixels ont la même valeur max
            avg_y_A = np.mean([p['y'] for p in max_pixels_A])
            avg_y_B = np.mean([p['y'] for p in max_pixels_B])
            
            # Point A (utiliser le premier pixel pour x, mais y moyen)
            point_A = {
                "x": x,
                "y": float(avg_y_A),
                "ascan_value": float(max_ascan_A),
                "num_pixels_at_max": len(max_pixels_A)
            }
            
            # Point B (utiliser le premier pixel pour x, mais y moyen)
            point_B = {
                "x": x,
                "y": float(avg_y_B),
                "ascan_value": float(max_ascan_B),
                "num_pixels_at_max": len(max_pixels_B)
            }
            
            # Calculer la distance euclidienne en pixels
            dx = point_A['x'] - point_B['x']  # Devrait être 0 si même X
            dy = point_A['y'] - point_B['y']
            distance_pixels = math.sqrt(dx**2 + dy**2)
            
            # Calculer la distance en millimètres
            dx_mm = dx * resolution_crosswise
            dy_mm = dy * resolution_ultrasound
            distance_mm = math.sqrt(dx_mm**2 + dy_mm**2)
            
            if self.ENABLE_PERF_LOGS:
                elapsed = (time.perf_counter() - start_time) * 1000
                self.logger.debug(f"[PERF] Mesure distance à x={x}: {elapsed:.2f}ms")
            
            return {
                "x": x,
                "point_A": point_A,
                "point_B": point_B,
                "distance_pixels": float(distance_pixels),
                "distance_mm": float(distance_mm),
                "delta_x_pixels": float(abs(dx)),
                "delta_y_pixels": float(abs(dy)),
                "delta_x_mm": float(abs(dx_mm)),
                "delta_y_mm": float(abs(dy_mm)),
                "status": "success"
            }
            
        except Exception as e:
            self.logger.error(f"Erreur lors de la mesure à x={x}: {str(e)}")
            return {
                "x": x,
                "status": "error",
                "error": str(e)
            }
    
    def measure_distance_all_x(
        self,
        ascan_data: Dict,
        endview_id: str,
        class_A: str,
        class_B: str,
        resolution_crosswise: float = 1.0,
        resolution_ultrasound: float = 1.0
    ) -> Dict:
        """
        Mesure les distances pour toutes les positions X d'une endview.
        
        OPTIMISATION: Construit un index X au début pour accès O(1) au lieu de O(n) par position.
        
        Args:
            ascan_data: Données A-scan (structure JSON complète)
            endview_id: ID de l'endview
            class_A: Nom de la première classe
            class_B: Nom de la deuxième classe
            resolution_crosswise: Résolution en mm/pixel pour l'axe X
            resolution_ultrasound: Résolution en mm/pixel pour l'axe Y
            
        Returns:
            Dict avec toutes les mesures et statistiques
        """
        try:
            if self.ENABLE_PERF_LOGS:
                start_time_total = time.perf_counter()
            
            # Vérifier que l'endview existe
            if endview_id not in ascan_data.get('endviews', {}):
                return {
                    "endview_id": endview_id,
                    "status": "error",
                    "error": f"Endview {endview_id} not found"
                }
            
            endview_data = ascan_data['endviews'][endview_id]

            key_A = self._resolve_class_key(endview_data, class_A)
            key_B = self._resolve_class_key(endview_data, class_B)

            # OPTIMISATION: Construire les index X pour accès O(1)
            if self.ENABLE_PERF_LOGS:
                self.logger.debug(f"[PERF] Construction des index X pour endview {endview_id}...")
            
            x_index_A = None
            x_index_B = None
            x_positions_A = set()
            x_positions_B = set()
            
            if key_A is not None and key_A in endview_data:
                x_index_A = self._build_x_index(endview_data[key_A])
                x_positions_A = set(x_index_A.keys())
            
            if key_B is not None and key_B in endview_data:
                x_index_B = self._build_x_index(endview_data[key_B])
                x_positions_B = set(x_index_B.keys())

            # Positions X communes aux deux classes
            common_x_positions = sorted(x_positions_A & x_positions_B)

            # Positions X manquantes dans chaque classe
            missing_in_A = sorted(x_positions_B - x_positions_A)
            missing_in_B = sorted(x_positions_A - x_positions_B)

            # Logging des positions communes et manquantes
            if common_x_positions:
                self.logger.info(f"Mesure des distances pour {len(common_x_positions)} positions X communes")
            else:
                self.logger.warning(f"⚠️  Aucune position X commune trouvée entre {class_A} et {class_B}")

            if missing_in_A:
                self.logger.warning(f"⚠️  {len(missing_in_A)} positions X manquantes dans {class_A}")
            if missing_in_B:
                self.logger.warning(f"⚠️  {len(missing_in_B)} positions X manquantes dans {class_B}")

            # Si aucune position commune, retourner quand même les positions manquantes
            if not common_x_positions:
                return {
                    "endview_id": endview_id,
                    "class_A": class_A,
                    "class_B": class_B,
                    "status": "error",
                    "error": "No common X positions found between the two classes",
                    "missing_in_A": missing_in_A,
                    "missing_in_B": missing_in_B,
                    "measurements": []  # Aucune mesure possible
                }
            
            # Mesurer pour chaque position X
            if self.ENABLE_PERF_LOGS:
                start_time_measurements = time.perf_counter()
                self.logger.debug(f"[PERF] Mesure de {len(common_x_positions)} positions X communes...")
            
            measurements = []
            successful_measurements = []
            
            for x in common_x_positions:
                # OPTIMISATION: Passer les index X pour accès O(1)
                result = self.measure_distance_at_x(
                    ascan_data=ascan_data,
                    endview_id=endview_id,
                    x=x,
                    class_A=class_A,
                    class_B=class_B,
                    resolution_crosswise=resolution_crosswise,
                    resolution_ultrasound=resolution_ultrasound,
                    x_index_A=x_index_A,
                    x_index_B=x_index_B
                )
                
                measurements.append(result)
                
                if result['status'] == 'success':
                    successful_measurements.append(result)
            
            if self.ENABLE_PERF_LOGS:
                elapsed_measurements = (time.perf_counter() - start_time_measurements) * 1000
                elapsed_total = (time.perf_counter() - start_time_total) * 1000
                self.logger.debug(f"[PERF] Mesures terminées: {elapsed_measurements:.2f}ms | Total endview: {elapsed_total:.2f}ms")
            
            # Calculer les statistiques
            statistics = self._calculate_statistics(successful_measurements)

            return {
                "endview_id": endview_id,
                "class_A": class_A,
                "class_B": class_B,
                "resolution_crosswise_mm": resolution_crosswise,
                "resolution_ultrasound_mm": resolution_ultrasound,
                "measurements": measurements,
                "statistics": statistics,
                "missing_in_A": missing_in_A,
                "missing_in_B": missing_in_B,
                "status": "success"
            }
            
        except Exception as e:
            self.logger.error(f"Erreur lors de la mesure pour endview {endview_id}: {str(e)}")
            return {
                "endview_id": endview_id,
                "status": "error",
                "error": str(e)
            }
    
    def measure_distance_all_endviews(
        self,
        ascan_data: Dict,
        class_A: str,
        class_B: str,
        resolution_crosswise: float = 1.0,
        resolution_ultrasound: float = 1.0
    ) -> Dict:
        """
        Mesure les distances pour toutes les endviews.
        
        Args:
            ascan_data: Données A-scan (structure JSON complète)
            class_A: Nom de la première classe
            class_B: Nom de la deuxième classe
            resolution_crosswise: Résolution en mm/pixel pour l'axe X
            resolution_ultrasound: Résolution en mm/pixel pour l'axe Y
            
        Returns:
            Dict avec toutes les mesures pour toutes les endviews
        """
        try:
            if self.ENABLE_PERF_LOGS:
                start_time_all = time.perf_counter()
            
            endview_ids = list(ascan_data.get('endviews', {}).keys())
            
            if not endview_ids:
                return {
                    "status": "error",
                    "error": "No endviews found in A-scan data"
                }
            
            self.logger.info(f"Mesure des distances pour {len(endview_ids)} endviews")
            if self.ENABLE_PERF_LOGS:
                self.logger.info(f"[PERF] Mode optimisé activé: indexation X pour accès O(1)")

            results = {
                "metadata": {
                    "nde_file": ascan_data.get('metadata', {}).get('nde_file', 'unknown'),
                    "class_A": class_A,
                    "class_B": class_B,
                    "resolution_crosswise_mm": resolution_crosswise,
                    "resolution_ultrasound_mm": resolution_ultrasound,
                    "num_endviews": len(endview_ids)
                },
                "endviews": {}
            }

            # Afficher la progression tous les 10%
            total_endviews = len(endview_ids)
            progress_step = max(1, total_endviews // 10)

            for idx, endview_id in enumerate(endview_ids):
                result = self.measure_distance_all_x(
                    ascan_data=ascan_data,
                    endview_id=endview_id,
                    class_A=class_A,
                    class_B=class_B,
                    resolution_crosswise=resolution_crosswise,
                    resolution_ultrasound=resolution_ultrasound
                )

                results['endviews'][endview_id] = result

                # Afficher la progression
                if (idx + 1) % progress_step == 0 or (idx + 1) == total_endviews:
                    progress_pct = ((idx + 1) / total_endviews) * 100
                    if self.ENABLE_PERF_LOGS:
                        elapsed_so_far = (time.perf_counter() - start_time_all)
                        avg_time_per_endview = elapsed_so_far / (idx + 1)
                        eta_seconds = avg_time_per_endview * (total_endviews - (idx + 1))
                        print(f"   → Progression: {idx + 1}/{total_endviews} endviews ({progress_pct:.0f}%) | ETA: {eta_seconds:.1f}s")
                    else:
                        print(f"   → Progression: {idx + 1}/{total_endviews} endviews ({progress_pct:.0f}%)")

            # Calculer les statistiques globales
            all_successful = []
            for endview_result in results['endviews'].values():
                if endview_result.get('status') == 'success':
                    all_successful.extend([
                        m for m in endview_result.get('measurements', [])
                        if m.get('status') == 'success'
                    ])

            results['global_statistics'] = self._calculate_statistics(all_successful)
            results['status'] = 'success'

            if self.ENABLE_PERF_LOGS:
                elapsed_total = time.perf_counter() - start_time_all
                self.logger.info(f"[PERF] ⚡ TOTAL: {elapsed_total:.2f}s ({elapsed_total*1000:.0f}ms) pour {len(endview_ids)} endviews")
                self.logger.info(f"[PERF] ⚡ Moyenne: {elapsed_total/len(endview_ids)*1000:.2f}ms par endview")
            
            self.logger.info(f"Mesure terminée: {len(all_successful)} mesures réussies au total")
            print(f"   → Total: {len(all_successful)} mesures réussies")
            
            return results
            
        except Exception as e:
            self.logger.error(f"Erreur lors de la mesure globale: {str(e)}")
            return {
                "status": "error",
                "error": str(e)
            }
    
    def _calculate_statistics(self, measurements: List[Dict]) -> Dict:
        """Calcule les statistiques sur une liste de mesures réussies."""
        if not measurements:
            return {
                "total_measurements": 0,
                "successful_measurements": 0,
                "failed_measurements": 0
            }
        
        distances_pixels = [m['distance_pixels'] for m in measurements]
        distances_mm = [m['distance_mm'] for m in measurements]
        
        return {
            "total_measurements": len(measurements),
            "successful_measurements": len(measurements),
            "mean_distance_pixels": float(np.mean(distances_pixels)),
            "std_distance_pixels": float(np.std(distances_pixels)),
            "min_distance_pixels": float(np.min(distances_pixels)),
            "max_distance_pixels": float(np.max(distances_pixels)),
            "median_distance_pixels": float(np.median(distances_pixels)),
            "mean_distance_mm": float(np.mean(distances_mm)),
            "std_distance_mm": float(np.std(distances_mm)),
            "min_distance_mm": float(np.min(distances_mm)),
            "max_distance_mm": float(np.max(distances_mm)),
            "median_distance_mm": float(np.median(distances_mm))
        }

    def build_distance_map(
        self,
        distance_results: Dict,
        num_endviews: int,
        width: int,
        *,
        use_mm: bool = False
    ) -> np.ndarray:
        """
        Construit une matrice (Z, X) des distances à partir des résultats détaillés.

        Args:
            distance_results: Sortie de measure_distance_all_endviews
            num_endviews: Nombre de slices (Z)
            width: Largeur (X) de l'image endview
            use_mm: Si True, utilise distance_mm au lieu de distance_pixels

        Returns:
            np.ndarray shape (num_endviews, width) rempli de NaN par défaut
        """
        distance_map = np.full((num_endviews, width), np.nan, dtype=np.float32)
        endviews_data = distance_results.get("endviews", {})
        value_key = "distance_mm" if use_mm else "distance_pixels"

        for endview_id_str, endview_result in endviews_data.items():
            if endview_result.get("status") != "success":
                continue
            try:
                z_idx = int(endview_id_str)
            except (TypeError, ValueError):
                continue
            if z_idx < 0 or z_idx >= num_endviews:
                continue

            measurements = endview_result.get("measurements", [])
            for measurement in measurements:
                if measurement.get("status") != "success":
                    continue
                x_idx = int(round(measurement.get("x", -1)))
                if x_idx < 0 or x_idx >= width:
                    continue
                value = measurement.get(value_key)
                if value is None:
                    continue
                distance_map[z_idx, x_idx] = float(value)

        return distance_map


def _save_measurement_errors(
    results: Dict,
    output_json_path: str,
    class_A: str,
    class_B: str
) -> None:
    """
    Extrait et sauvegarde toutes les erreurs de mesure dans un fichier séparé.

    Args:
        results: Résultats complets des mesures
        output_json_path: Chemin du fichier JSON principal
        class_A: Nom de la première classe
        class_B: Nom de la deuxième classe
    """
    import json

    logger = logging.getLogger(__name__)

    # Collecter toutes les erreurs
    all_errors = []

    endviews_data = results.get('endviews', {})
    for endview_id, endview_result in endviews_data.items():
        # Récupérer les positions X manquantes (présentes même en cas d'erreur)
        missing_in_A = endview_result.get('missing_in_A', [])
        missing_in_B = endview_result.get('missing_in_B', [])

        # Ajouter les positions manquantes comme erreurs individuelles
        for x in missing_in_A:
            all_errors.append({
                "endview_id": endview_id,
                "x": x,
                "error_type": "missing_position",
                "error": f"X={x} found in {class_B} but missing in {class_A}",
                "missing_in_class": class_A,
                "present_in_class": class_B
            })

        for x in missing_in_B:
            all_errors.append({
                "endview_id": endview_id,
                "x": x,
                "error_type": "missing_position",
                "error": f"X={x} found in {class_A} but missing in {class_B}",
                "missing_in_class": class_B,
                "present_in_class": class_A
            })

        # Si l'endview a une erreur globale (ex: aucune position commune)
        if endview_result.get('status') != 'success':
            # Ajouter aussi une erreur d'endview pour traçabilité
            all_errors.append({
                "endview_id": endview_id,
                "error_type": "endview_error",
                "error": endview_result.get('error', 'Unknown error'),
                "total_missing_in_A": len(missing_in_A),
                "total_missing_in_B": len(missing_in_B)
            })
            # Ne pas continuer, car il peut y avoir des mesures partielles
            # Seulement si measurements est vide
            if not endview_result.get('measurements'):
                continue

        # Vérifier les mesures individuelles (erreurs sur positions communes)
        measurements = endview_result.get('measurements', [])
        for measurement in measurements:
            if measurement.get('status') == 'error':
                all_errors.append({
                    "endview_id": endview_id,
                    "x": measurement.get('x'),
                    "error_type": "measurement_error",
                    "error": measurement.get('error', 'Unknown error'),
                    "class_A": class_A,
                    "class_B": class_B
                })

    # Si aucune erreur, ne pas créer de fichier
    if not all_errors:
        logger.info("Aucune erreur de mesure détectée")
        return

    # Créer le fichier d'erreurs
    error_path = output_json_path.replace('.json', '_errors.json')

    # Compter les différents types d'erreurs
    missing_position_errors = [e for e in all_errors if e.get('error_type') == 'missing_position']
    measurement_errors = [e for e in all_errors if e.get('error_type') == 'measurement_error']
    endview_errors = [e for e in all_errors if e.get('error_type') == 'endview_error']

    missing_in_A_count = len([e for e in missing_position_errors if e.get('missing_in_class') == class_A])
    missing_in_B_count = len([e for e in missing_position_errors if e.get('missing_in_class') == class_B])

    error_data = {
        "metadata": {
            "source_file": output_json_path,
            "class_A": class_A,
            "class_B": class_B,
            "total_errors": len(all_errors),
            "missing_positions_in_A": missing_in_A_count,
            "missing_positions_in_B": missing_in_B_count,
            "measurement_errors": len(measurement_errors),
            "endview_errors": len(endview_errors),
            "nde_file": results.get('metadata', {}).get('nde_file', 'unknown')
        },
        "errors": all_errors
    }

    # OPTIMISATION: Pas d'indentation pour les gros fichiers = 50-70% plus rapide
    with open(error_path, 'w', encoding='utf-8') as f:
        json.dump(error_data, f, ensure_ascii=False)

    logger.warning(f"⚠️  {len(all_errors)} erreurs de mesure détectées et sauvegardées: {error_path}")


def measure_distances_from_json(
    ascan_json_path: str,
    class_A: str,
    class_B: str,
    resolution_crosswise: float = 1.0,
    resolution_ultrasound: float = 1.0,
    output_json_path: Optional[str] = None
) -> Dict:
    """
    Fonction utilitaire pour mesurer les distances depuis un fichier JSON A-scan.

    Args:
        ascan_json_path: Chemin vers le fichier JSON A-scan
        class_A: Nom de la première classe
        class_B: Nom de la deuxième classe
        resolution_crosswise: Résolution en mm/pixel pour l'axe X
        resolution_ultrasound: Résolution en mm/pixel pour l'axe Y
        output_json_path: Chemin de sortie (optionnel, auto-généré si None)

    Returns:
        Dict avec toutes les mesures
    """
    import json

    # Charger le JSON A-scan
    with open(ascan_json_path, 'r', encoding='utf-8') as f:
        ascan_data = json.load(f)

    # Créer le service et mesurer
    service = DistanceMeasurementService()
    results = service.measure_distance_all_endviews(
        ascan_data=ascan_data,
        class_A=class_A,
        class_B=class_B,
        resolution_crosswise=resolution_crosswise,
        resolution_ultrasound=resolution_ultrasound
    )

    # Sauvegarder si chemin de sortie fourni
    if output_json_path:
        # OPTIMISATION: Pas d'indentation pour les gros fichiers = 50-70% plus rapide
        start_save = time.perf_counter()
        with open(output_json_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False)
        save_time = time.perf_counter() - start_save

        logging.info(f"Résultats sauvegardés en {save_time:.2f}s: {output_json_path}")
    else:
        # Auto-générer le nom de fichier
        base_path = ascan_json_path.replace('_ascan_values.json', '')
        output_json_path = f"{base_path}_distances_{class_A}_{class_B}.json"

        # OPTIMISATION: Pas d'indentation pour les gros fichiers = 50-70% plus rapide
        start_save = time.perf_counter()
        with open(output_json_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False)
        save_time = time.perf_counter() - start_save

        logging.info(f"Résultats sauvegardés en {save_time:.2f}s: {output_json_path}")

    # Extraire et sauvegarder les erreurs dans un fichier séparé
    _save_measurement_errors(results, output_json_path, class_A, class_B)

    return results


def calculate_distances_workflow(
    global_masks: List[np.ndarray],
    volume_data: np.ndarray,
    nde_data: Dict,
    orientation: str,
    transpose: bool,
    rotation_applied: bool,
    nde_filename: str,
    output_directory: str,
    class_A: str = 'frontwall',
    class_B: str = 'backwall',
    save_json: bool = True,
    save_lines_npz: bool = True,
    lines_mode: str = 'contours'
) -> Dict:
    """
    Workflow complet de calcul des distances entre deux classes.

    Ce workflow orchestre :
    1. Extraction des valeurs A-scan depuis les masques
    2. Calcul des distances entre class_A et class_B
    3. Sauvegarde optionnelle des fichiers JSON (A-scan, distances, erreurs)
    4. Génération optionnelle du NPZ de lignes de mesure

    Args:
        global_masks: Liste des masques de segmentation
        volume_data: Volume 3D NDE
        nde_data: Métadonnées NDE (dimensions, résolutions, etc.)
        orientation: Orientation des slices ('lengthwise', 'crosswise', 'ultrasound')
        transpose: Si transpose a été appliqué
        rotation_applied: Si rotation a été appliquée
        nde_filename: Nom du fichier NDE
        output_directory: Répertoire de sortie pour les fichiers
        class_A: Première classe (défaut: 'frontwall')
        class_B: Deuxième classe (défaut: 'backwall')
        save_json: Si True, sauvegarde les fichiers JSON (défaut: True)
        save_lines_npz: Si True, génère le NPZ de lignes (défaut: True)
        lines_mode: Mode de tracé des lignes ('vertical', 'contours', 'both')

    Returns:
        Dict contenant:
            - ascan_data: Données A-scan extraites
            - distance_results: Résultats des calculs de distance
            - ascan_json_path: Chemin du fichier A-scan JSON (si sauvegardé)
            - distances_json_path: Chemin du fichier distances JSON (si sauvegardé)
            - errors_json_path: Chemin du fichier erreurs JSON (si sauvegardé)
            - lines_npz_path: Chemin du NPZ de lignes (si généré)
            - status: 'success' ou 'error'
            - error: Message d'erreur (si status='error')
    """
    import json
    import time
    from services.ascan_service import export_ascan_values_to_json

    logger = logging.getLogger(__name__)
    result = {
        'ascan_data': None,
        'distance_results': None,
        'ascan_json_path': None,
        'distances_json_path': None,
        'errors_json_path': None,
        'lines_npz_path': None,
        'status': 'error',
        'error': None
    }

    try:
        logger.info("=== WORKFLOW CALCUL DISTANCES : DÉBUT ===")

        # Étape 1: Extraction A-scan
        logger.info("Étape 1/4: Extraction des valeurs A-scan...")
        ascan_data = export_ascan_values_to_json(
            global_masks_array=global_masks,
            volume_data=volume_data,
            orientation=orientation,
            transpose=transpose,
            rotation_applied=rotation_applied,
            nde_filename=nde_filename
        )
        result['ascan_data'] = ascan_data
        logger.info(f"A-scan extrait: {len(ascan_data.get('endviews', {}))} endviews")

        # Étape 2: Calcul des distances
        logger.info("Étape 2/4: Calcul des distances...")
        dimensions = nde_data.get('dimensions', [])
        resolution_crosswise = dimensions[1].get('resolution', 1.0) if len(dimensions) >= 3 else 1.0
        resolution_ultrasound = dimensions[2].get('resolution', 1.0) if len(dimensions) >= 3 else 1.0

        logger.info(f"Résolutions: crosswise={resolution_crosswise:.6f} mm/px, ultrasound={resolution_ultrasound:.6f} mm/px")

        distance_service = DistanceMeasurementService()
        distance_results = distance_service.measure_distance_all_endviews(
            ascan_data=ascan_data,
            class_A=class_A,
            class_B=class_B,
            resolution_crosswise=resolution_crosswise,
            resolution_ultrasound=resolution_ultrasound
        )
        result['distance_results'] = distance_results

        if distance_results.get('status') != 'success':
            result['error'] = distance_results.get('error', 'Erreur lors du calcul des distances')
            logger.error(result['error'])
            return result

        logger.info("Distances calculées avec succès")

        # Étape 3: Génération du NPZ de lignes (optionnel)
        if save_lines_npz and global_masks:
            logger.info("Étape 3/4: Génération du NPZ de lignes...")
            try:
                import os
                base_name = os.path.splitext(nde_filename)[0]
                lines_npz_path = os.path.join(output_directory, f"{base_name}_distances_lines.npz")

                image_shape = global_masks[0].shape
                create_distance_lines_npz(
                    distances_data=distance_results,
                    image_shape=image_shape,
                    num_endviews=len(global_masks),
                    output_npz_path=lines_npz_path,
                    line_thickness=2,
                    mode=lines_mode,
                    class_A=class_A,
                    class_B=class_B
                )
                result['lines_npz_path'] = lines_npz_path
                logger.info(f"NPZ de lignes créé: {lines_npz_path}")
            except Exception as e:
                logger.warning(f"Erreur lors de la création du NPZ de lignes: {e}")
                # Ne pas bloquer le workflow si la création du NPZ échoue
        else:
            logger.info("Étape 3/4: Génération du NPZ de lignes ignorée (désactivée)")

        # Étape 4: Sauvegarde JSON (optionnel)
        if save_json:
            logger.info("Étape 4/4: Sauvegarde des fichiers JSON...")
            try:
                import os
                base_name = os.path.splitext(nde_filename)[0]

                # Sauvegarder A-scan JSON
                ascan_json_path = os.path.join(output_directory, f"{base_name}_ascan_values.json")
                start_time = time.perf_counter()
                with open(ascan_json_path, 'w', encoding='utf-8') as f:
                    json.dump(ascan_data, f, ensure_ascii=False)
                save_time = time.perf_counter() - start_time
                result['ascan_json_path'] = ascan_json_path
                logger.info(f"A-scan JSON sauvegardé en {save_time:.2f}s: {ascan_json_path}")

                # Sauvegarder distances JSON
                distances_json_path = os.path.join(output_directory, f"{base_name}_distances_{class_A}_{class_B}.json")
                start_time = time.perf_counter()
                with open(distances_json_path, 'w', encoding='utf-8') as f:
                    json.dump(distance_results, f, ensure_ascii=False)
                save_time = time.perf_counter() - start_time
                result['distances_json_path'] = distances_json_path
                logger.info(f"Distances JSON sauvegardé en {save_time:.2f}s: {distances_json_path}")

                # Sauvegarder erreurs JSON
                _save_measurement_errors(distance_results, distances_json_path, class_A, class_B)
                errors_json_path = distances_json_path.replace('.json', '_errors.json')
                if os.path.exists(errors_json_path):
                    result['errors_json_path'] = errors_json_path
                    logger.info(f"Erreurs JSON sauvegardé: {errors_json_path}")

            except Exception as e:
                logger.warning(f"Erreur lors de la sauvegarde JSON: {e}")
                # Ne pas bloquer le workflow si la sauvegarde échoue
        else:
            logger.info("Étape 4/4: Sauvegarde JSON ignorée (désactivée)")

        result['status'] = 'success'
        logger.info("=== WORKFLOW CALCUL DISTANCES : SUCCÈS ===")
        return result

    except Exception as e:
        logger.error(f"Erreur dans le workflow de calcul des distances: {e}")
        import traceback
        traceback.print_exc()
        result['error'] = str(e)
        return result


def create_distance_lines_npz(
    distances_data: Dict,
    image_shape: Tuple[int, int],
    num_endviews: int,
    output_npz_path: str,
    line_thickness: int = 1,
    mode: str = 'both',
    class_A: str = 'frontwall',
    class_B: str = 'backwall'
) -> None:
    """
    Crée un NPZ contenant des masques avec les lignes de mesure tracées.

    Args:
        distances_data: Données de distances (structure JSON complète)
        image_shape: Shape des images (height, width)
        num_endviews: Nombre total d'endviews
        output_npz_path: Chemin de sortie pour le NPZ
        line_thickness: Épaisseur des lignes en pixels
        mode: Mode de tracé - 'vertical', 'contours', ou 'both' (défaut: 'both')
            - 'vertical': Lignes verticales entre classe A et classe B (classe 9, rouge vif)
            - 'contours': Lignes continues suivant les contours (classe A → classe 5+, classe B → classe 6+)
            - 'both': Les deux modes combinés
        class_A: Nom de la première classe (défaut: 'frontwall')
        class_B: Nom de la deuxième classe (défaut: 'backwall')

    Mapping des classes pour les contours:
        - frontwall → classe 5 (mauve)
        - backwall → classe 6 (vert clair)
        - flaw → classe 7 (orange)
        - indication → classe 8 (cyan)
    """
    logger = logging.getLogger(__name__)

    # Mapping des classes vers les valeurs de visualisation
    CLASS_TO_PLOT_VALUE = {
        'frontwall': 5,   # Mauve
        'backwall': 6,    # Vert clair
        'flaw': 7,        # Orange
        'indication': 8   # Cyan
    }

    try:
        logger.info(f"Création du NPZ de lignes de mesure: {num_endviews} endviews, shape={image_shape}")
        logger.info(f"Classes: {class_A} vs {class_B}, Mode: {mode}")

        # Créer un array vide pour toutes les endviews
        height, width = image_shape
        lines_volume = np.zeros((num_endviews, height, width), dtype=np.uint8)

        # Traiter chaque endview
        endviews_data = distances_data.get('endviews', {})

        for endview_id_str, endview_result in endviews_data.items():
            endview_id = int(endview_id_str)

            # Vérifier que l'endview est dans les limites
            if endview_id >= num_endviews:
                logger.warning(f"Endview {endview_id} hors limites (max={num_endviews-1}), ignoré")
                continue

            # Vérifier que des mesures existent
            if endview_result.get('status') != 'success':
                logger.debug(f"Endview {endview_id}: pas de mesures réussies, masque vide")
                continue

            measurements = endview_result.get('measurements', [])
            successful_measurements = [m for m in measurements if m.get('status') == 'success']

            if not successful_measurements:
                logger.debug(f"Endview {endview_id}: aucune mesure réussie, masque vide")
                continue

            # Créer le masque pour cette endview
            mask = np.zeros((height, width), dtype=np.uint8)

            # === MODE 'VERTICAL' : Lignes verticales entre frontwall et backwall ===
            if mode in ['vertical', 'both']:
                for measurement in successful_measurements:
                    point_A = measurement.get('point_A', {})
                    point_B = measurement.get('point_B', {})

                    # Récupérer les coordonnées (arrondir si moyenne)
                    x_A = int(round(point_A.get('x', 0)))
                    y_A = int(round(point_A.get('y', 0)))
                    x_B = int(round(point_B.get('x', 0)))
                    y_B = int(round(point_B.get('y', 0)))

                    # Tracer ligne verticale en rouge vif (classe 9)
                    cv2.line(
                        mask,
                        (x_A, y_A),  # Point de départ (x, y)
                        (x_B, y_B),  # Point d'arrivée (x, y)
                        color=9,  # Classe 9 = rouge highlight
                        thickness=line_thickness
                    )

            # === MODE 'CONTOURS' : Lignes continues suivant les contours ===
            if mode in ['contours', 'both']:
                # Extraire tous les points max de classe A et B
                points_A = []
                points_B = []

                for measurement in successful_measurements:
                    point_A = measurement.get('point_A', {})
                    point_B = measurement.get('point_B', {})

                    x_A = point_A.get('x', 0)
                    y_A = point_A.get('y', 0)
                    x_B = point_B.get('x', 0)
                    y_B = point_B.get('y', 0)

                    points_A.append((x_A, y_A))
                    points_B.append((x_B, y_B))

                # Trier par X pour avoir une ligne continue
                points_A_sorted = sorted(points_A, key=lambda p: p[0])
                points_B_sorted = sorted(points_B, key=lambda p: p[0])

                # Déterminer les valeurs de classe pour les contours
                color_A = CLASS_TO_PLOT_VALUE.get(class_A, 5)  # Par défaut classe 5
                color_B = CLASS_TO_PLOT_VALUE.get(class_B, 6)  # Par défaut classe 6

                # Tracer ligne continue pour classe A
                for i in range(len(points_A_sorted) - 1):
                    pt1 = (int(round(points_A_sorted[i][0])), int(round(points_A_sorted[i][1])))
                    pt2 = (int(round(points_A_sorted[i+1][0])), int(round(points_A_sorted[i+1][1])))
                    cv2.line(mask, pt1, pt2, color=color_A, thickness=line_thickness)

                # Tracer ligne continue pour classe B
                for i in range(len(points_B_sorted) - 1):
                    pt1 = (int(round(points_B_sorted[i][0])), int(round(points_B_sorted[i][1])))
                    pt2 = (int(round(points_B_sorted[i+1][0])), int(round(points_B_sorted[i+1][1])))
                    cv2.line(mask, pt1, pt2, color=color_B, thickness=line_thickness)

            # Stocker le masque dans le volume
            lines_volume[endview_id] = mask

            # Log selon le mode
            if mode == 'vertical':
                logger.debug(f"Endview {endview_id}: {len(successful_measurements)} lignes verticales tracées")
            elif mode == 'contours':
                logger.debug(f"Endview {endview_id}: contours frontwall et backwall tracés")
            else:  # both
                logger.debug(f"Endview {endview_id}: {len(successful_measurements)} lignes verticales + contours tracés")

        # Appliquer les mêmes transformations que pour les masques (flip + transpose ZXY)
        logger.info("Application des transformations (flip + transpose ZXY)...")
        volume_flipped = np.array([np.fliplr(slice_img) for slice_img in lines_volume])
        volume_transposed = volume_flipped.transpose((0, 2, 1))  # ZXY

        # Sauvegarder en NPZ
        np.savez_compressed(output_npz_path, arr_0=volume_transposed)

        logger.info(f"NPZ de lignes créé avec succès: {output_npz_path}")

    except Exception as e:
        logger.error(f"Erreur lors de la création du NPZ de lignes: {str(e)}")
        raise

