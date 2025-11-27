"""
Service pour le calcul des masques.
Sépare la logique métier de calcul des masques de la vue.
"""
import cv2
import numpy as np
import logging
from typing import Dict, List, Tuple, Any, Callable, Optional
from config.constants import MASK_COLORS_BGR
from services.profile import generate_profile_mask
from utils.morphology import smooth_mask_contours
from utils.helpers import calculate_auto_threshold


class MaskCalculationService:
    """Service pour calculer les masques avec différents paramètres."""
    
    def __init__(self):
        self._logger = logging.getLogger(__name__)

    # --- Sources de threshold -------------------------------------------------

    def get_inverted_threshold_source(
        self,
        base_image: Optional[np.ndarray],
        raw_slice: Optional[np.ndarray] = None,
        legacy_max: float = 255.0,
    ) -> Tuple[Optional[np.ndarray], float]:
        """
        Retourne une image uint8 inversée (noir <-> blanc) pour le calcul de threshold.

        Args:
            base_image: Image BGR si disponible.
            raw_slice: Slice normalisée (0-1) optionnelle.
            legacy_max: Valeur max du seuil historique.
        """
        if raw_slice is not None:
            return self.convert_raw_slice_to_inverted(raw_slice, legacy_max), legacy_max

        if base_image is None:
            return None, legacy_max

        gray = cv2.cvtColor(base_image, cv2.COLOR_BGR2GRAY)
        inverted_gray = 255 - gray
        return inverted_gray, legacy_max

    def convert_raw_slice_to_inverted(
        self, raw_slice: np.ndarray, legacy_max: float = 255.0
    ) -> np.ndarray:
        """Convertit une slice normalisée (0-1) en image uint8 inversée."""
        raw_uint8 = np.clip(raw_slice * legacy_max, 0, legacy_max).astype(np.uint8)
        return legacy_max - raw_uint8

    def get_threshold_source_for_slice(
        self,
        slice_index: int,
        raw_slice_provider: Callable[[int], Optional[np.ndarray]],
        legacy_max: float = 255.0,
    ) -> Tuple[Optional[np.ndarray], float]:
        """Construit la source inversée pour une slice fournie par un callback."""
        raw_slice = raw_slice_provider(slice_index)
        if raw_slice is None:
            return None, legacy_max
        return self.convert_raw_slice_to_inverted(raw_slice, legacy_max), legacy_max

    # --- Opérations sur contours/polygones -----------------------------------

    def merge_vertical_polygons(
        self,
        contours: List[np.ndarray],
        vertical_gap: int,
        image_shape: Optional[Tuple[int, int]] = None,
    ) -> List[np.ndarray]:
        """
        Fusionne des contours proches verticalement (même logique que la vue mais généralisée).
        """
        if not contours:
            return []

        if image_shape:
            h, w = image_shape
        else:
            all_points = np.vstack([c.reshape(-1, 2) for c in contours])
            max_y = np.max(all_points[:, 1]) + 10
            max_x = np.max(all_points[:, 0]) + 10
            h, w = int(max_y), int(max_x)

        all_contours_mask = np.zeros((h, w), dtype=np.uint8)
        for contour in contours:
            cv2.drawContours(all_contours_mask, [contour], -1, 255, -1)

        for x in range(w):
            col = all_contours_mask[:, x]
            idx = np.where(col > 0)[0]
            if idx.size == 0:
                continue

            idx = np.sort(idx)
            groups = []
            current = [idx[0]]
            for i in range(1, len(idx)):
                if idx[i] - idx[i - 1] <= vertical_gap:
                    current.append(idx[i])
                else:
                    groups.append(current)
                    current = [idx[i]]
            if current:
                groups.append(current)

            for group in groups:
                y_min = min(group)
                y_max = max(group)
                all_contours_mask[y_min : y_max + 1, x] = 255

        merged_contours, _ = cv2.findContours(
            all_contours_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
        )
        return list(merged_contours)

    def transform_to_pixelized_contour(
        self,
        polygon_points: List[Tuple[int, int]],
        threshold: int,
        threshold_source: np.ndarray,
        mask_type: str,
        smooth_contours: bool,
        profile_value: Optional[int] = None,
    ) -> List[Tuple[int, int]]:
        """
        Transforme un polygone en contour aligné sur les pixels en appliquant le threshold,
        le lissage et la génération de profil (si mask_type=polygon).
        """
        if not polygon_points:
            return polygon_points

        h, w = threshold_source.shape[:2]
        binary_mask = (threshold_source <= threshold).astype(np.uint8) * 255

        points = np.array(polygon_points[:-1], np.int32)
        temp_mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(temp_mask, [points], 1)
        temp_mask &= binary_mask

        if mask_type == "polygon" and profile_value is not None:
            temp_mask = self._generate_profile_mask(temp_mask, profile_value, mask_type)

        if smooth_contours and np.any(temp_mask):
            kernel_size = max(1, int(0.5 * 2))
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
            temp_mask = cv2.morphologyEx(temp_mask, cv2.MORPH_CLOSE, kernel)

        pixelized_points = self._extract_pixel_perfect_contour(temp_mask)
        if pixelized_points and len(pixelized_points) >= 3 and pixelized_points[-1] != pixelized_points[0]:
            pixelized_points.append(pixelized_points[0])
        return pixelized_points or polygon_points

    def _extract_pixel_perfect_contour(self, mask: np.ndarray) -> List[Tuple[int, int]]:
        """
        Extrait un contour carré (pixel perfect) depuis un masque binaire.
        """
        if mask is None or mask.size == 0 or not np.any(mask > 0):
            return []

        scale_factor = 3
        h, w = mask.shape
        enlarged_mask = np.zeros((h * scale_factor, w * scale_factor), dtype=np.uint8)
        y_coords, x_coords = np.where(mask > 0)
        if len(y_coords) > 0:
            y_starts = y_coords * scale_factor
            x_starts = x_coords * scale_factor
            for y_start, x_start in zip(y_starts, x_starts):
                enlarged_mask[y_start : y_start + scale_factor, x_start : x_start + scale_factor] = 255

        contours, _ = cv2.findContours(enlarged_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if not contours:
            return []

        largest = max(contours, key=cv2.contourArea)
        contour_array = largest.reshape(-1, 2)
        original_points = [(x / scale_factor, y / scale_factor) for x, y in contour_array]
        simplified_points = self._simplify_to_pixel_corners(original_points)

        if len(simplified_points) >= 3 and simplified_points[-1] != simplified_points[0]:
            simplified_points.append(simplified_points[0])
        return simplified_points

    def _simplify_to_pixel_corners(self, points: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
        """Garde uniquement les coins pertinents d'un contour agrandi."""
        if len(points) < 3:
            return points

        simplified = [points[0]]
        for i in range(1, len(points) - 1):
            prev = simplified[-1]
            curr = points[i]
            next_pt = points[i + 1]
            dir1 = (curr[0] - prev[0], curr[1] - prev[1])
            dir2 = (next_pt[0] - curr[0], next_pt[1] - curr[1])
            if dir1 != dir2:
                simplified.append(curr)

        simplified.append(points[-1])
        final_points = [simplified[0]]
        for i in range(1, len(simplified) - 1):
            prev = final_points[-1]
            curr = simplified[i]
            next_pt = simplified[i + 1]
            cross = (curr[0] - prev[0]) * (next_pt[1] - prev[1]) - (curr[1] - prev[1]) * (next_pt[0] - prev[0])
            if cross != 0:
                final_points.append(curr)
        return final_points

    # --- ROI / Region growing -------------------------------------------------

    def compute_roi_region_mask(
        self,
        roi_points: List[Tuple[int, int]],
        threshold_source: np.ndarray,
        threshold: int,
        vertical_gap: int,
    ) -> Optional[np.ndarray]:
        """Calcule un masque binaire pour une ROI (rectangle/polygone) sur une slice donnée."""
        if roi_points is None or len(roi_points) < 3 or threshold_source is None:
            return None

        h, w = threshold_source.shape[:2]
        binary_mask = (threshold_source <= threshold).astype(np.uint8)
        temp_mask = np.zeros((h, w), dtype=np.uint8)

        pts = roi_points
        if pts[0] != pts[-1]:
            pts = pts + [pts[0]]
        points_np = np.array(pts[:-1], np.int32)
        cv2.fillPoly(temp_mask, [points_np], 1)
        temp_mask &= binary_mask

        contours, _ = cv2.findContours(temp_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        merged_contours = self.merge_vertical_polygons(contours, vertical_gap, threshold_source.shape[:2])
        if not merged_contours:
            return None

        region_mask = np.zeros((h, w), dtype=np.uint8)
        for contour in merged_contours:
            cv2.drawContours(region_mask, [contour], -1, 255, -1)

        if not np.any(region_mask):
            return None
        return region_mask

    def grow_seed_region_from_point(
        self,
        seed_x: int,
        seed_y: int,
        threshold: int,
        threshold_source: np.ndarray,
        vertical_gap: int,
        mask_type: str,
        smooth_contours: bool,
        profile_value: Optional[int] = None,
        record_polygons: bool = True,
        return_region_mask: bool = False,
    ) -> Dict[str, Any]:
        """
        Croissance de région à partir d'une graine. Retourne les polygones et le masque de région.
        """
        h, w = threshold_source.shape[:2]
        seed_x = int(np.clip(seed_x, 0, w - 1))
        seed_y = int(np.clip(seed_y, 0, h - 1))

        binary_mask = (threshold_source <= threshold).astype(np.uint8)
        if not (0 <= seed_y < h and 0 <= seed_x < w):
            return {"success": False, "region_mask": None, "polygons": []}

        if binary_mask[seed_y, seed_x] == 0:
            return {"success": False, "region_mask": None, "polygons": []}

        num_labels, labels = cv2.connectedComponents(binary_mask, connectivity=8)
        seed_label = labels[seed_y, seed_x]
        if seed_label == 0:
            return {"success": False, "region_mask": None, "polygons": []}

        region_mask = np.zeros_like(binary_mask, dtype=np.uint8)
        region_mask[labels == seed_label] = 255

        contours, _ = cv2.findContours(region_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        merged_contours = self.merge_vertical_polygons(contours, vertical_gap, threshold_source.shape[:2])

        polygons_created = []
        for contour in merged_contours:
            epsilon = 0.001 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            polygon_points = [(int(p[0][0]), int(p[0][1])) for p in approx]
            if len(polygon_points) == 1:
                px, py = polygon_points[0]
                polygon_points = [
                    (px - 1, py - 1),
                    (px + 1, py - 1),
                    (px + 1, py + 1),
                    (px - 1, py + 1),
                    (px - 1, py - 1),
                ]
            elif len(polygon_points) == 2:
                (x1, y1), (x2, y2) = polygon_points
                polygon_points = [
                    (x1 - 1, y1 - 1),
                    (x2 + 1, y1 - 1),
                    (x2 + 1, y2 + 1),
                    (x1 - 1, y2 + 1),
                    (x1 - 1, y1 - 1),
                ]
            else:
                polygon_points.append(polygon_points[0])

            pixelized_polygon = self.transform_to_pixelized_contour(
                polygon_points,
                threshold,
                threshold_source,
                mask_type,
                smooth_contours,
                profile_value,
            )
            if len(pixelized_polygon) >= 1:
                polygons_created.append(pixelized_polygon)

        success = len(polygons_created) > 0
        return {
            "success": success,
            "region_mask": region_mask if return_region_mask else None,
            "polygons": polygons_created if record_polygons else [],
        }

    def apply_seed_volume_to_adjacent_slices(
        self,
        seed_x: int,
        seed_y: int,
        threshold: int,
        parameters: Dict[str, Any],
        current_index: int,
        total_slices: int,
        threshold_source_provider: Callable[[int], Optional[np.ndarray]],
        apply_region_mask_cb: Callable[[int, np.ndarray, str], bool],
        label: str,
        pause_cb: Optional[Callable[[], None]] = None,
        resume_cb: Optional[Callable[[], None]] = None,
    ) -> int:
        """
        Applique automatiquement la croissance seed sur les slices adjacentes tant qu'une région existe.
        """
        processed_slices = 0
        vertical_gap = parameters.get("vertical_gap", 0)
        mask_type = parameters.get("mask_type", "standard")
        smooth_contours = parameters.get("smooth_contours", False)

        if pause_cb:
            pause_cb()

        try:
            for direction in (1, -1):
                idx = current_index + direction
                while 0 <= idx < total_slices:
                    threshold_source = threshold_source_provider(idx)
                    if threshold_source is None:
                        break

                    seed_result = self.grow_seed_region_from_point(
                        seed_x,
                        seed_y,
                        threshold,
                        threshold_source,
                        vertical_gap,
                        mask_type,
                        smooth_contours,
                        parameters.get("class_value"),
                        record_polygons=False,
                        return_region_mask=True,
                    )

                    if not seed_result.get("success"):
                        break
                    region_mask = seed_result.get("region_mask")
                    if region_mask is None:
                        break
                    if not apply_region_mask_cb(idx, region_mask, label):
                        break
                    processed_slices += 1
                    idx += direction
        finally:
            if resume_cb:
                resume_cb()

        return processed_slices

    def apply_roi_volume_to_adjacent_slices(
        self,
        roi_points: List[Tuple[int, int]],
        threshold: int,
        parameters: Dict[str, Any],
        current_index: int,
        total_slices: int,
        threshold_source_provider: Callable[[int], Optional[np.ndarray]],
        apply_region_mask_cb: Callable[[int, np.ndarray, str], bool],
        label: str,
        pause_cb: Optional[Callable[[], None]] = None,
        resume_cb: Optional[Callable[[], None]] = None,
    ) -> int:
        """
        Propagation ROI (rectangle/polygone) vers les slices adjacentes.
        """
        processed_slices = 0
        vertical_gap = parameters.get("vertical_gap", 0)

        if pause_cb:
            pause_cb()

        try:
            for direction in (1, -1):
                idx = current_index + direction
                while 0 <= idx < total_slices:
                    threshold_source = threshold_source_provider(idx)
                    if threshold_source is None:
                        break

                    region_mask = self.compute_roi_region_mask(
                        roi_points, threshold_source, threshold, vertical_gap
                    )
                    if region_mask is None:
                        break

                    if not apply_region_mask_cb(idx, region_mask, label):
                        break
                    processed_slices += 1
                    idx += direction
        finally:
            if resume_cb:
                resume_cb()

        return processed_slices

    # --- Masques et overlay ---------------------------------------------------

    def calculate_individual_mask(self, image: np.ndarray, polygons: Dict[str, List], 
                                label_settings: Dict[str, Dict[str, Any]]) -> np.ndarray:
        """
        Calcule le masque coloré avec paramètres individuels pour chaque label.
        
        Args:
            image: Image originale
            polygons: Polygones par label
            label_settings: Paramètres individuels par label
            
        Returns:
            Masque coloré avec valeurs de classe
        """
        if image is None:
            return np.zeros((100, 100), dtype=np.uint8)
        
        h, w = image.shape[:2]
        final_mask = np.zeros((h, w), dtype=np.uint8)
        
        # Traiter chaque label avec ses propres paramètres
        for label, label_value in [("frontwall", 1), ("backwall", 2), ("flaw", 3), ("indication", 4)]:
            if label not in label_settings:
                continue
                
            settings = label_settings[label]
            threshold = settings["threshold"]
            mask_type = settings["mask_type"]
            smooth_contours = settings.get("smooth_contours", False)
            
            # Calculer le masque binaire pour ce seuil
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            binary_mask = (gray < threshold).astype(np.uint8) * 255
            
            # Calculer le masque pour ce label
            if mask_type == "standard":
                label_mask = self._calculate_standard_mask_for_label(
                    binary_mask, h, w, label, polygons[label], label_value
                )
            else:  # polygon
                label_mask = self._calculate_polygon_mask_for_label(
                    binary_mask, h, w, label, polygons[label], label_value, settings
                )
            
            # Appliquer le lissage si activé
            if smooth_contours and np.any(label_mask):
                kernel_size = settings.get("smooth_kernel", 5)
                # Créer un masque binaire temporaire
                temp_binary = (label_mask == label_value).astype(np.uint8) * 255
                smoothed_binary = smooth_mask_contours(temp_binary, kernel_size)
                # Remettre la valeur de classe
                label_mask = np.where(smoothed_binary > 0, label_value, 0).astype(np.uint8)
            
            # Fusionner avec le masque final
            final_mask = np.maximum(final_mask, label_mask)
        
        return final_mask
    
    def _calculate_standard_mask_for_label(self, binary_mask: np.ndarray, h: int, w: int,
                                         label: str, polygons: list, label_value: int) -> np.ndarray:
        """Calcule le masque standard pour un label spécifique."""
        mask = np.zeros((h, w), dtype=np.uint8)

        for pts in polygons:
            if len(pts) >= 3:
                points = np.array(pts, np.int32)
                # Clipper les coordonnées aux limites de l'image
                points[:, 0] = np.clip(points[:, 0], 0, w - 1)
                points[:, 1] = np.clip(points[:, 1], 0, h - 1)

                tmp = np.zeros_like(mask, dtype=np.uint8)
                cv2.fillPoly(tmp, [points], 1)
                # Appliquer le threshold
                tmp &= binary_mask
                # Assigner la valeur de classe
                tmp[tmp > 0] = label_value
                mask = np.maximum(mask, tmp)

        return mask
    
    def _calculate_polygon_mask_for_label(self, binary_mask: np.ndarray, h: int, w: int,
                                        label: str, polygons: list, label_value: int,
                                        settings: Dict[str, Any]) -> np.ndarray:
        """Calcule le masque polygonal pour un label spécifique."""
        mask = np.zeros((h, w), dtype=np.uint8)

        for pts in polygons:
            if len(pts) >= 3:
                points = np.array(pts, np.int32)
                # Clipper les coordonnées aux limites de l'image
                points[:, 0] = np.clip(points[:, 0], 0, w - 1)
                points[:, 1] = np.clip(points[:, 1], 0, h - 1)

                tmp = np.zeros_like(mask, dtype=np.uint8)
                cv2.fillPoly(tmp, [points], 1)
                # Appliquer le threshold
                tmp &= binary_mask
                # Générer le masque de profil avec verticalisation pour ce label spécifique
                profile_mask = self._generate_profile_mask_for_label(tmp, label_value, settings)
                mask = np.maximum(mask, profile_mask)
        
        return mask
    
    def _generate_profile_mask_for_label(self, single_mask: np.ndarray, val: int, 
                                       settings: Dict[str, Any]) -> np.ndarray:
        """
        Génère un masque de profil pour un label spécifique avec ses propres paramètres.
        
        Args:
            single_mask: Masque binaire initial
            val: Valeur du label à appliquer
            settings: Paramètres du label
            
        Returns:
            Masque respectant strictement le contour avec paramètres du label
        """
        try:
            mask_type = settings["mask_type"]
            
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
            
            # Appliquer la verticalisation seulement si ce label est en mode polygon
            if mask_type == "polygon":
                h, w = profile_mask.shape
                
                # Version optimisée avec colonnes actives
                mask_bool = (profile_mask == val)
                any_col = mask_bool.any(axis=0)
                cols = np.nonzero(any_col)[0]
                
                if len(cols) > 0:
                    # Y minimum et maximum par colonne (vectorisé)
                    y_min = np.where(any_col, mask_bool.argmax(axis=0), 0)
                    y_max = np.where(any_col, h - 1 - mask_bool[::-1].argmax(axis=0), 0)
                    
                    # Remplir verticalement - itération sur colonnes actives seulement
                    for x in cols:
                        profile_mask[y_min[x]:y_max[x]+1, x] = val
            
            return profile_mask
            
        except Exception as e:
            self._logger.error(f"Erreur lors de la génération du masque de profil: {str(e)}")
            return single_mask

    def _generate_profile_mask_with_params(self, single_mask: np.ndarray, val: int, parameters: dict) -> np.ndarray:
        """
        Génère un masque de profil avec les paramètres spécifiques du polygone.
        """
        try:
            fusion_mask = np.zeros_like(single_mask)

            contours, _ = cv2.findContours(single_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

            if not contours:
                return single_mask

            if len(contours) > 1:
                cv2.drawContours(fusion_mask, contours, -1, 1, -1)
                contours, _ = cv2.findContours(fusion_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

            profile_mask = np.zeros_like(single_mask)

            for contour in contours:
                cv2.fillPoly(profile_mask, [contour], val)

            mask_type = parameters.get("mask_type", "standard")
            if mask_type == "polygon":
                h, w = profile_mask.shape
                mask_bool = profile_mask == val
                any_col = mask_bool.any(axis=0)
                cols = np.nonzero(any_col)[0]
                if len(cols) > 0:
                    y_min = np.where(any_col, mask_bool.argmax(axis=0), 0)
                    y_max = np.where(any_col, h - 1 - mask_bool[::-1].argmax(axis=0), 0)
                    for x in cols:
                        profile_mask[y_min[x] : y_max[x] + 1, x] = val

            return profile_mask

        except Exception as e:
            self._logger.error(f"Erreur lors de la génération du masque de profil avec paramètres: {str(e)}")
            return single_mask

    def _generate_profile_mask(self, single_mask: np.ndarray, val: int, mask_type: str) -> np.ndarray:
        """Version générique pour un seul masque avec mask_type."""
        try:
            if single_mask is None or not np.any(single_mask):
                return single_mask

            fusion_mask = np.zeros_like(single_mask)
            contours, _ = cv2.findContours(single_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            if not contours:
                return single_mask
            if len(contours) > 1:
                cv2.drawContours(fusion_mask, contours, -1, 1, -1)
                contours, _ = cv2.findContours(fusion_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

            profile_mask = np.zeros_like(single_mask)
            for contour in contours:
                cv2.fillPoly(profile_mask, [contour], val)

            if mask_type == "polygon":
                h, w = profile_mask.shape
                mask_bool = profile_mask == val
                any_col = mask_bool.any(axis=0)
                cols = np.nonzero(any_col)[0]
                if len(cols) > 0:
                    y_min = np.where(any_col, mask_bool.argmax(axis=0), 0)
                    y_max = np.where(any_col, h - 1 - mask_bool[::-1].argmax(axis=0), 0)
                    for x in cols:
                        profile_mask[y_min[x] : y_max[x] + 1, x] = val
            return profile_mask
        except Exception as e:
            self._logger.error(f"Erreur lors de la génération du masque de profil générique: {str(e)}")
            return single_mask
    
    def apply_overlay(self, image: np.ndarray, mask: np.ndarray, alpha: float) -> np.ndarray:
        """
        Applique un overlay coloré sur l'image.
        
        Args:
            image: Image de base
            mask: Masque avec valeurs de classe
            alpha: Transparence (0-100)
            
        Returns:
            Image avec overlay
        """
        if image is None or mask is None:
            return image
        
        alpha_normalized = alpha / 100.0

        overlay = np.zeros_like(image)
        for class_value, bgr_color in MASK_COLORS_BGR.items():
            class_pixels = mask == class_value
            if np.any(class_pixels):
                overlay[class_pixels] = bgr_color

        mask_any = np.any(mask > 0, axis=2)
        if not np.any(mask_any):
            return image

        mask_alpha = mask_any.astype(np.float32) * alpha_normalized
        mask_alpha_exp = mask_alpha[..., None]

        result = image.astype(np.float32) * (1 - mask_alpha_exp) + overlay.astype(np.float32) * mask_alpha_exp
        return result.astype(np.uint8)

    # --- Masque global à partir de polygones (avec paramètres) --------------

    def calculate_mask_from_polygons_with_params(
        self,
        image: np.ndarray,
        class_map: Dict[str, int],
        polygons_by_label: Dict[str, List[Dict[str, Any]]],
        threshold_source: Optional[np.ndarray],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Construit le masque final (valeurs de classe) et la zone affectée à partir de polygones paramétrés.

        Args:
            image: Image BGR (utilisée pour fallback threshold si threshold_source est None)
            class_map: Mapping label -> valeur de classe
            polygons_by_label: Dict label -> liste de dicts {points, parameters}
            threshold_source: Image inversée uint8 (H, W) ou None (fallback: 255 - gray(image))

        Returns:
            final_mask: masque uint8 (H, W) avec valeurs de classe
            affected_region: masque bool (H, W) indiquant les pixels modifiés
        """
        if image is None:
            return np.zeros((0, 0), dtype=np.uint8), np.zeros((0, 0), dtype=bool)

        h, w = image.shape[:2]
        final_mask = np.zeros((h, w), dtype=np.uint8)
        affected_region = np.zeros((h, w), dtype=bool)

        if threshold_source is None:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            threshold_source = 255 - gray

        # Inclure un label spécial "erase" avec valeur 0
        label_value_map = {**class_map, "erase": 0}

        for label, class_value in label_value_map.items():
            polygons = polygons_by_label.get(label, [])
            if not polygons:
                continue

            label_mask = np.zeros((h, w), dtype=np.uint8)
            label_region = np.zeros((h, w), dtype=bool)
            is_erase = label == "erase"

            for poly_data in polygons:
                pts = poly_data.get("points") or []
                parameters = poly_data.get("parameters") or {}
                if len(pts) < 3:
                    continue

                threshold = parameters.get("threshold", 150)
                mask_type = parameters.get("mask_type", "standard")
                smooth_enabled = parameters.get("smooth_contours", False)

                binary_mask = (threshold_source <= threshold).astype(np.uint8) * 255

                pts_clipped = np.array(pts, dtype=np.int32)
                pts_clipped[:, 0] = np.clip(pts_clipped[:, 0], 0, w - 1)
                pts_clipped[:, 1] = np.clip(pts_clipped[:, 1], 0, h - 1)

                tmp = np.zeros_like(label_mask, dtype=np.uint8)
                cv2.fillPoly(tmp, [pts_clipped], 1)
                tmp &= binary_mask
                if not np.any(tmp):
                    continue

                current_region = np.zeros_like(label_region, dtype=bool)

                if mask_type == "standard":
                    if is_erase:
                        current_region = tmp > 0
                    else:
                        label_mask[tmp > 0] = class_value
                        current_region = tmp > 0
                else:
                    profile_value = class_value if not is_erase else 1
                    profile_mask = self._generate_profile_mask_with_params(tmp, profile_value, parameters)
                    if is_erase:
                        current_region = profile_mask > 0
                    else:
                        label_mask = np.maximum(label_mask, profile_mask)
                        current_region = profile_mask == profile_value

                if smooth_enabled and np.any(tmp > 0):
                    polygon_binary = (tmp > 0).astype(np.uint8)
                    smooth_value = class_value if not is_erase else 1
                    smoothed = smooth_mask_contours(polygon_binary * smooth_value, 0.5)
                    if is_erase:
                        current_region = smoothed > 0
                    else:
                        label_mask[tmp > 0] = 0
                        label_mask[smoothed == smooth_value] = class_value
                        current_region = smoothed == smooth_value

                label_region |= current_region

            if is_erase:
                affected_region |= label_region
            else:
                final_mask = np.maximum(final_mask, label_mask)
                affected_region |= label_region

        return final_mask, affected_region

    # --- Helpers seed/auto ---------------------------------------------------

    def build_seed_auto_polygon(self, x: int, y: int, width: int, height: int, radius: int = 10) -> List[Tuple[int, int]]:
        """Construit un petit rectangle autour de la graine pour l'auto-threshold."""
        half = max(2, radius)
        left = max(0, x - half)
        right = min(width - 1, x + half)
        top = max(0, y - half)
        bottom = min(height - 1, y + half)

        if left == right:
            right = min(width - 1, left + 1)
        if top == bottom:
            bottom = min(height - 1, top + 1)

        return [
            (left, top),
            (right, top),
            (right, bottom),
            (left, bottom),
            (left, top),
        ]

    def build_seed_marker_polygon(self, x: int, y: int, size: int = 6) -> List[Tuple[int, int]]:
        """Retourne un carré fermé autour de la graine pour visualiser la ROI persistante."""
        half = max(2, size // 2)
        return [
            (x - half, y - half),
            (x + half, y - half),
            (x + half, y + half),
            (x - half, y + half),
            (x - half, y - half),
        ]

    def calculate_colored_mask(
        self,
        image: np.ndarray,
        class_map: Dict[str, int],
        polygons_by_label: Dict[str, List[Dict[str, Any]]],
        temporary_polygons: List[Dict[str, Any]],
        threshold_source: Optional[np.ndarray],
    ) -> np.ndarray:
        """
        Construit le masque final (valeurs de classe) à partir des polygones persistants et temporaires.
        """
        if image is None:
            return np.zeros((100, 100, 3), dtype=np.uint8)

        h, w = image.shape[:2]
        if threshold_source is None:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            threshold_source = 255 - gray

        final_mask = np.zeros((h, w), dtype=np.uint8)

        def _fill_from_polygons(poly_list: List[Dict[str, Any]], target_mask: np.ndarray):
            for poly_data in poly_list:
                label = poly_data.get("label")
                if label not in class_map:
                    continue
                class_value = class_map[label]
                points = poly_data.get("points", [])
                params = poly_data.get("parameters", {})
                if len(points) < 3:
                    continue
                pts_list = points[:-1] if len(points) > 3 and points[0] == points[-1] else points
                if len(pts_list) < 3:
                    continue
                pts = np.array(pts_list, np.int32)
                temp_mask = np.zeros((h, w), dtype=np.uint8)
                cv2.fillPoly(temp_mask, [pts], class_value)

                threshold_val = params.get("threshold")
                if threshold_val is not None:
                    binary_mask = (threshold_source <= threshold_val).astype(np.uint8)
                    temp_mask = temp_mask * binary_mask

                mask_type = params.get("mask_type", "standard")
                if mask_type == "polygon":
                    temp_mask = self._generate_profile_mask_with_params(temp_mask, class_value, params)

                np.maximum(target_mask, temp_mask, out=target_mask)

        _fill_from_polygons(
            [
                {"label": label, "points": poly.get("points"), "parameters": poly.get("parameters")}
                for label, polys in polygons_by_label.items()
                for poly in polys
            ],
            final_mask,
        )
        _fill_from_polygons(temporary_polygons, final_mask)

        colored_mask = np.zeros((h, w, 3), dtype=np.uint8)
        for label_value, color in MASK_COLORS_BGR.items():
            colored_mask[final_mask == label_value] = tuple(color)
        return colored_mask

    # --- ROI builders --------------------------------------------------------

    def build_rectangle_polygons(
        self,
        rectangle_polygon: List[Tuple[int, int]],
        threshold_source: np.ndarray,
        settings: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Calcule les polygones issus d'une ROI rectangle (auto/manuel).
        settings attend : threshold, auto_threshold, threshold_percentage, max_threshold,
        threshold_method, vertical_gap, mask_type, smooth_contours, class_value.
        """
        if threshold_source is None or len(rectangle_polygon) < 4:
            return {"polygons": [], "threshold_used": settings.get("threshold", 0)}

        threshold_used = settings.get("threshold", 0)
        vertical_gap = settings.get("vertical_gap", 0)
        mask_type = settings.get("mask_type", "standard")
        smooth_contours = settings.get("smooth_contours", False)
        class_value = settings.get("class_value")

        if settings.get("auto_threshold"):
            threshold_used = int(
                calculate_auto_threshold(
                    threshold_source,
                    rectangle_polygon,
                    settings.get("threshold_percentage", 0),
                    min(settings.get("max_threshold", 255), threshold_source.max() if threshold_source is not None else 255),
                    settings.get("threshold_method", "percentile"),
                    value_range=(0.0, settings.get("max_threshold", 255)),
                )
            )

        h, w = threshold_source.shape[:2]
        binary_mask = (threshold_source <= threshold_used).astype(np.uint8) * 255

        points = np.array(rectangle_polygon[:-1], np.int32)
        temp_mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(temp_mask, [points], 1)
        temp_mask &= binary_mask

        contours, _ = cv2.findContours(temp_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        merged_contours = self.merge_vertical_polygons(contours, vertical_gap, threshold_source.shape[:2])

        polygons_created = []
        for contour in merged_contours:
            epsilon = 0.001 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            polygon_points = [(int(p[0][0]), int(p[0][1])) for p in approx]
            if len(polygon_points) == 1:
                x, y = polygon_points[0]
                polygon_points = [(x - 1, y - 1), (x + 1, y - 1), (x + 1, y + 1), (x - 1, y + 1), (x - 1, y - 1)]
            elif len(polygon_points) == 2:
                x1, y1 = polygon_points[0]
                x2, y2 = polygon_points[1]
                polygon_points = [
                    (x1 - 1, y1 - 1),
                    (x2 + 1, y1 - 1),
                    (x2 + 1, y2 + 1),
                    (x1 - 1, y2 + 1),
                    (x1 - 1, y1 - 1),
                ]
            else:
                polygon_points.append(polygon_points[0])

            pixelized_polygon = self.transform_to_pixelized_contour(
                polygon_points,
                threshold_used,
                threshold_source,
                mask_type,
                smooth_contours,
                class_value,
            )
            if pixelized_polygon:
                polygons_created.append(pixelized_polygon)

        return {"polygons": polygons_created, "threshold_used": threshold_used}

    def build_polygon_polygons(
        self,
        polygon_points: List[Tuple[int, int]],
        threshold_source: np.ndarray,
        settings: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Calcule les polygones issus d'une ROI polygone libre (auto/manuel).
        settings attend : threshold, auto_threshold, threshold_percentage, max_threshold,
        threshold_method, vertical_gap, mask_type, smooth_contours, class_value.
        """
        if threshold_source is None or len(polygon_points) < 3:
            return {"polygons": [], "threshold_used": settings.get("threshold", 0)}

        threshold_used = settings.get("threshold", 0)
        vertical_gap = settings.get("vertical_gap", 0)
        mask_type = settings.get("mask_type", "standard")
        smooth_contours = settings.get("smooth_contours", False)
        class_value = settings.get("class_value")

        if settings.get("auto_threshold"):
            threshold_used = int(
                calculate_auto_threshold(
                    threshold_source,
                    polygon_points,
                    settings.get("threshold_percentage", 0),
                    min(settings.get("max_threshold", 255), threshold_source.max() if threshold_source is not None else 255),
                    settings.get("threshold_method", "percentile"),
                    value_range=(0.0, settings.get("max_threshold", 255)),
                )
            )

        h, w = threshold_source.shape[:2]
        binary_mask = (threshold_source <= threshold_used).astype(np.uint8) * 255

        pts_array = np.array(polygon_points[:-1], np.int32)
        temp_mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(temp_mask, [pts_array], 1)
        temp_mask &= binary_mask

        contours, _ = cv2.findContours(temp_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        merged_contours = self.merge_vertical_polygons(contours, vertical_gap, threshold_source.shape[:2])

        polygons_created = []
        for contour in merged_contours:
            epsilon = 0.001 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            poly_pts = [(int(p[0][0]), int(p[0][1])) for p in approx]
            if len(poly_pts) == 1:
                x, y = poly_pts[0]
                poly_pts = [(x - 1, y - 1), (x + 1, y - 1), (x + 1, y + 1), (x - 1, y + 1), (x - 1, y - 1)]
            elif len(poly_pts) == 2:
                x1, y1 = poly_pts[0]
                x2, y2 = poly_pts[1]
                poly_pts = [
                    (x1 - 1, y1 - 1),
                    (x2 + 1, y1 - 1),
                    (x2 + 1, y2 + 1),
                    (x1 - 1, y2 + 1),
                    (x1 - 1, y1 - 1),
                ]
            else:
                poly_pts.append(poly_pts[0])

            pixelized = self.transform_to_pixelized_contour(
                poly_pts,
                threshold_used,
                threshold_source,
                mask_type,
                smooth_contours,
                class_value,
            )
            if pixelized:
                polygons_created.append(pixelized)

        return {"polygons": polygons_created, "threshold_used": threshold_used}
