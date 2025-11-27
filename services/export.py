# === model/mask_exporter.py ===
import os
import numpy as np
from PIL import Image
import cv2
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Any
from utils.helpers import normalize_polygon_coords, apply_binary_threshold, clip_polygon_coords
from utils.morphology import smooth_mask_contours
from services.profile import generate_profile_mask
from .json_service import JsonExporter, generate_json_filename
from config.constants import MASK_COLORS_RGB, MASK_COLORS_BGR

def export_masks(model, threshold, mask_type="standard", smooth_enabled=False, smooth_radius=0, output_folder=None):
    """
    Exporte les masques avec le seuil, le type et l'arrondissement spécifiés.

    Args:
        model: Modèle contenant les données
        threshold: Seuil de binarisation
        mask_type: Type de masque ("standard" ou "polygon")
        smooth_enabled: Activer l'arrondissement des contours
        smooth_radius: Rayon d'arrondissement (si activé)
        output_folder: Dossier de sortie (optionnel, sinon utilise le dossier de l'image)
    """
    gray = cv2.cvtColor(model.image_original, cv2.COLOR_BGR2GRAY)
    binary_mask = apply_binary_threshold(gray, threshold)
    h, w = binary_mask.shape
    mask_std = np.zeros((h, w), dtype=np.uint8)
    mask_poly = np.zeros((h, w), dtype=np.uint8)

    for label, val in {'frontwall': 1, 'backwall': 2, 'flaw': 3, 'indication': 4}.items():
        for pts in model.polygons[label]:
            if len(pts) >= 3:
                # Clipper les coordonnées aux limites de l'image au lieu d'utiliser la marge
                pts_clipped = clip_polygon_coords(pts, w, h)
                tmp = np.zeros_like(mask_std, dtype=np.uint8)
                cv2.fillPoly(tmp, [np.array(pts_clipped, dtype=np.int32)], 1)
                tmp &= binary_mask
                mask_std[tmp > 0] = val
                # Pour le masque polygonal, générer des profils pour TOUS les types (frontwall, backwall, flaw, indication)
                mask_poly = np.maximum(mask_poly, generate_profile_mask(tmp, val))

    # Appliquer l'arrondissement des contours si activé
    if smooth_enabled and smooth_radius > 0:
        mask_std = smooth_mask_contours(mask_std, kernel_size=int(2 * smooth_radius + 1))
        mask_poly = smooth_mask_contours(mask_poly, kernel_size=int(2 * smooth_radius + 1))

    # Utiliser le dossier de sortie fourni ou déduire du chemin de l'image
    if output_folder is not None:
        base = output_folder
    else:
        base = os.path.dirname(model.image_list[model.current_index])

    name = os.path.basename(model.image_list[model.current_index])

    # Choisir le masque à exporter selon le type sélectionné
    save_all_masks(mask_std, mask_poly, base, name, mask_type)


def save_all_masks(m1, m2, folder, fname, mask_type="standard"):
    """
    Sauvegarde les masques selon le type sélectionné.

    Args:
        m1: Masque standard
        m2: Masque polygonal
        folder: Dossier de destination
        fname: Nom du fichier
        mask_type: Type de masque à privilégier ("standard" ou "polygon")
    """
    # Créer les dossiers selon le type sélectionné
    logger = logging.getLogger(__name__)
    if mask_type == "polygon":
        selected_mask = m2
        logger.info(f"POLYGON masks exported: {fname}")
    else:  # standard
        selected_mask = m1
        logger.info(f"STANDARD masks exported: {fname}")

    # Créer aussi les dossiers pour l'autre type (pour compatibilité)
    std_dir = os.path.join(folder, "masks_standard")
    poly_dir = os.path.join(folder, "masks_polygon")
    vis_poly_dir = os.path.join(folder, "masks_polygon_visual")
    vis_std_dir = os.path.join(folder, "masks_standard_visual")

    os.makedirs(std_dir, exist_ok=True)
    os.makedirs(poly_dir, exist_ok=True)
    os.makedirs(vis_poly_dir, exist_ok=True)
    os.makedirs(vis_std_dir, exist_ok=True)

    # Sauvegarder les deux types pour compatibilité
    Image.fromarray(m1).save(os.path.join(std_dir, fname))
    Image.fromarray(m2).save(os.path.join(poly_dir, fname))

    # Créer les versions visuelles avec les couleurs BGR définies dans les constantes
    rgb_poly = np.zeros((*m2.shape, 3), dtype=np.uint8)
    for class_value, bgr_color in MASK_COLORS_BGR.items():
        rgb_poly[m2 == class_value] = bgr_color
    Image.fromarray(rgb_poly).save(os.path.join(vis_poly_dir, fname))

    rgb_std = np.zeros((*m1.shape, 3), dtype=np.uint8)
    for class_value, bgr_color in MASK_COLORS_BGR.items():
        rgb_std[m1 == class_value] = bgr_color
    Image.fromarray(rgb_std).save(os.path.join(vis_std_dir, fname))

    # Calculer les statistiques du masque sélectionné
    total_pixels = selected_mask.size
    frontwall_pixels = np.sum(selected_mask == 1)
    backwall_pixels = np.sum(selected_mask == 2)
    flaw_pixels = np.sum(selected_mask == 3)
    indication_pixels = np.sum(selected_mask == 4)
    logger.info(f"Type: {mask_type} | Frontwall: {frontwall_pixels} | Backwall: {backwall_pixels} | Flaw: {flaw_pixels} | Indication: {indication_pixels} pixels")

    # Retourner les informations pour que le contrôleur puisse les afficher
    return {
        'type': mask_type,
        'filename': fname,
        'stats': {
            'frontwall': frontwall_pixels,
            'backwall': backwall_pixels,
            'flaw': flaw_pixels,
            'indication': indication_pixels,
            'total': total_pixels
        }
    }


def export_individual_masks(model, label_settings=None, output_folder=None):
    """
    Exporte un masque unifié avec les paramètres individuels de chaque polygone.
    Génère un masque binaire, un masque visible ET un fichier JSON selon les choix de l'utilisateur.

    Args:
        model: Modèle contenant les données avec paramètres individuels par polygone
        label_settings: Paramètres par label (obsolète, maintenant dans les polygones)
        output_folder: Dossier de sortie (optionnel, sinon utilise le dossier de l'image)

    Returns:
        np.ndarray: Le masque final généré (pour stockage dans l'array global)
    """
    h, w = model.image_original.shape[:2]
    final_mask = np.zeros((h, w), dtype=np.uint8)

    # Obtenir les polygones avec leurs paramètres individuels
    polygons_with_params = model.get_all_polygons()

    # Traiter chaque label
    for label, val in {'frontwall': 1, 'backwall': 2, 'flaw': 3, 'indication': 4}.items():
        polygons = polygons_with_params[label]

        if not polygons:
            continue

        # Créer le masque temporaire pour ce label
        label_mask = np.zeros((h, w), dtype=np.uint8)

        # Traiter chaque polygone avec ses paramètres individuels
        for polygon_data in polygons:
            pts = polygon_data['points']
            parameters = polygon_data['parameters']

            if len(pts) < 3:
                continue

            # Utiliser les paramètres spécifiques de ce polygone
            threshold = parameters.get('threshold', 150)
            mask_type = parameters.get('mask_type', 'standard')
            smooth_enabled = parameters.get('smooth_contours', False)

            # Calculer le masque binaire avec le threshold spécifique de ce polygone
            gray = cv2.cvtColor(model.image_original, cv2.COLOR_BGR2GRAY)
            binary_mask = apply_binary_threshold(gray, threshold)

            # Traiter ce polygone spécifique
            # Clipper les coordonnées aux limites de l'image
            h, w = model.image_original.shape[:2]
            pts_clipped = clip_polygon_coords(pts, w, h)
            tmp = np.zeros_like(label_mask, dtype=np.uint8)
            cv2.fillPoly(tmp, [np.array(pts_clipped, dtype=np.int32)], 1)
            tmp &= binary_mask

            # Appliquer le type de masque choisi pour ce polygone spécifique
            if mask_type == "standard":
                # Masque standard : utiliser directement le polygone
                label_mask[tmp > 0] = val
            else:  # polygon
                # Masque polygonal : générer des profils automatiques
                profile_mask = generate_profile_mask(tmp, val)
                label_mask = np.maximum(label_mask, profile_mask)

            # Appliquer l'arrondissement spécifique à ce polygone
            if smooth_enabled and np.any(tmp > 0):
                # Extraire seulement les pixels de ce polygone
                polygon_binary = (tmp > 0).astype(np.uint8)
                smoothed = smooth_mask_contours(polygon_binary * val, 0.5)
                # Remplacer les pixels de ce polygone par la version lissée
                label_mask[tmp > 0] = 0  # Effacer l'ancien
                label_mask[smoothed == val] = val  # Ajouter le nouveau

        # Ajouter ce label au masque final
        final_mask = np.maximum(final_mask, label_mask)

    # Sauvegarder le masque unifié
    # Utiliser le dossier de sortie fourni ou déduire du chemin de l'image
    if output_folder is not None:
        base = output_folder
    else:
        base = os.path.dirname(model.image_list[model.current_index])

    name = os.path.basename(model.image_list[model.current_index])

    # Créer un dictionnaire de paramètres pour la compatibilité avec save_unified_masks
    # (utilise les paramètres du premier polygone de chaque label comme exemple)
    compatibility_settings = {}
    for label in ['frontwall', 'backwall', 'flaw', 'indication']:
        polygons = polygons_with_params[label]
        if polygons:
            # Utiliser les paramètres du premier polygone comme exemple
            first_params = polygons[0]['parameters']
            compatibility_settings[label] = first_params
        else:
            # Paramètres par défaut si aucun polygone
            compatibility_settings[label] = {
                'threshold': 150,
                'mask_type': 'standard',
                'smooth_contours': False
            }

    # Sauvegarder selon vos choix (passer le modèle pour l'export JSON)
    save_unified_masks(final_mask, base, name, compatibility_settings, model)

    # Retourner le masque pour qu'il puisse être stocké dans l'array global
    return final_mask


def save_unified_masks(mask, folder, fname, label_settings, model):
    """
    Sauvegarde un masque unifié avec paramètres individuels dans 3 dossiers séparés.
    Génère un masque binaire, un masque visible ET un fichier JSON selon les choix utilisateur.

    Args:
        mask: Masque unifié avec tous les labels
        folder: Dossier de destination
        fname: Nom du fichier
        label_settings: Paramètres utilisés pour chaque label
        model: Modèle contenant les données pour l'export JSON
    """
    # Créer 3 dossiers séparés
    binary_dir = os.path.join(folder, "masks_binary")
    visual_dir = os.path.join(folder, "masks_visual")
    json_dir = os.path.join(folder, "json")

    os.makedirs(binary_dir, exist_ok=True)
    os.makedirs(visual_dir, exist_ok=True)
    os.makedirs(json_dir, exist_ok=True)

    # 1. MASQUE BINAIRE : Masque avec valeurs 1, 2, 3, 4 pour frontwall, backwall, flaw, indication
    binary_path = os.path.join(binary_dir, fname)
    Image.fromarray(mask).save(binary_path)

    # 2. MASQUE VISIBLE : Version colorée pour visualisation
    visual_mask = np.zeros((*mask.shape, 3), dtype=np.uint8)
    # Utiliser les couleurs RGB pour l'export PIL
    for class_value, rgb_color in MASK_COLORS_RGB.items():
        visual_mask[mask == class_value] = rgb_color

    # Sauvegarder le masque visible (même nom de fichier)
    visual_path = os.path.join(visual_dir, fname)
    Image.fromarray(visual_mask).save(visual_path)

    # Logger les informations d'export
    logger = logging.getLogger(__name__)
    # logger.info(f"UNIFIED masks exported: {fname}")
    # logger.info(f"Masque binaire: {binary_dir}")
    # logger.info(f"Masque visible: {visual_dir}")
    # logger.info(f"Dossier JSON: {json_dir}")

    # Calculer les paramètres utilisés pour chaque label
    label_stats = {}
    for label, settings in label_settings.items():
        pixels = np.sum(mask == {'frontwall': 1, 'backwall': 2, 'flaw': 3, 'indication': 4}[label])
        color = {'frontwall': 'rouge', 'backwall': 'vert', 'flaw': 'bleu', 'indication': 'orange'}[label]
        label_stats[label] = {
            'pixels': pixels,
            'color': color,
            'threshold': settings['threshold'],
            'mask_type': settings['mask_type'],
            'smooth_contours': settings['smooth_contours']
        }
        # logger.info(f"{label.upper()} ({color}): threshold={settings['threshold']}, "
        #            f"type={settings['mask_type']}, smooth={settings['smooth_contours']}, "
        #            f"pixels={pixels}")

    # Statistiques globales
    frontwall_pixels = np.sum(mask == 1)
    backwall_pixels = np.sum(mask == 2)
    flaw_pixels = np.sum(mask == 3)
    indication_pixels = np.sum(mask == 4)
    total_annotated = frontwall_pixels + backwall_pixels + flaw_pixels + indication_pixels
    total_image = mask.size
    coverage = (total_annotated / total_image) * 100

    global_stats = {
        'frontwall': frontwall_pixels,
        'backwall': backwall_pixels,
        'flaw': flaw_pixels,
        'indication': indication_pixels,
        'total_annotated': total_annotated,
        'total_image': total_image,
        'coverage': coverage
    }

    # logger.info(f"Statistiques globales: Frontwall={frontwall_pixels}, Backwall={backwall_pixels}, "
    #            f"Flaw={flaw_pixels}, Indication={indication_pixels}, "
    #            f"Total annoté={total_annotated} pixels ({coverage:.1f}% de l'image)")

    # 3. EXPORT JSON AUTOMATIQUE avec le même nom que l'image
    json_result = None
    # logger.info("Export JSON automatique...")
    try:
        # Obtenir le nom de l'image actuelle (sans extension)
        image_name = os.path.splitext(fname)[0]
        json_filename = f"{image_name}.json"
        json_path = os.path.join(json_dir, json_filename)

        # Créer l'exporteur JSON
        json_exporter = JsonExporter()

        # Exporter les polygones originaux pour préserver la forme exacte dessinée
        result_path = json_exporter.export_polygons_to_json(
            model, label_settings, json_path
        )

        if result_path:
            # logger.info(f"JSON exporté: {json_filename}")
            # logger.info("Compatible avec Kili et réutilisable")
            # logger.info("Exporté depuis le masque final (cohérent avec les masques)")
            json_result = {
                'success': True,
                'filename': json_filename,
                'path': result_path
            }
        else:
            logger.error("Erreur lors de l'export JSON")
            json_result = {'success': False, 'error': 'Export failed'}

    except Exception as e:
        logger.error(f"Erreur export JSON: {str(e)}")
        json_result = {'success': False, 'error': str(e)}

    # Retourner toutes les informations pour que le contrôleur puisse les afficher
    return {
        'filename': fname,
        'directories': {
            'binary': binary_dir,
            'visual': visual_dir,
            'json': json_dir
        },
        'label_stats': label_stats,
        'global_stats': global_stats,
        'json_export': json_result
    }


def save_individual_masks(m1, m2, folder, fname, label_settings):
    """
    Sauvegarde les masques avec paramètres individuels.

    Args:
        m1: Masque standard
        m2: Masque polygonal
        folder: Dossier de destination
        fname: Nom du fichier
        label_settings: Paramètres utilisés pour chaque label
    """
    # Créer les sous-dossiers
    std_dir = os.path.join(folder, "standard")
    poly_dir = os.path.join(folder, "polygon")
    vis_dir = os.path.join(folder, "visual")

    os.makedirs(std_dir, exist_ok=True)
    os.makedirs(poly_dir, exist_ok=True)
    os.makedirs(vis_dir, exist_ok=True)

    # Sauvegarder les masques
    Image.fromarray(m1).save(os.path.join(std_dir, fname))
    Image.fromarray(m2).save(os.path.join(poly_dir, fname))

    # Créer la version visuelle avec les couleurs RGB pour l'export
    rgb = np.zeros((*m1.shape, 3), dtype=np.uint8)
    for class_value, rgb_color in MASK_COLORS_RGB.items():
        rgb[m1 == class_value] = rgb_color
    Image.fromarray(rgb).save(os.path.join(vis_dir, fname))

    # Logger les paramètres utilisés
    logger = logging.getLogger(__name__)
    logger.info(f"INDIVIDUAL masks exported: {fname}")
    for label, settings in label_settings.items():
        pixels = np.sum(m1 == {'frontwall': 1, 'backwall': 2, 'flaw': 3, 'indication': 4}[label])
        logger.info(f"{label}: threshold={settings['threshold']}, type={settings['mask_type']}, "
                   f"smooth={settings['smooth_contours']}, pixels={pixels}")

    total_pixels = m1.size
    frontwall_pixels = np.sum(m1 == 1)
    backwall_pixels = np.sum(m1 == 2)
    flaw_pixels = np.sum(m1 == 3)
    indication_pixels = np.sum(m1 == 4)
    logger.info(f"Total: Frontwall={frontwall_pixels} | Backwall={backwall_pixels} | Flaw={flaw_pixels} | Indication={indication_pixels} pixels")

    # Retourner les informations pour que le contrôleur puisse les afficher
    return {
        'filename': fname,
        'stats': {
            'frontwall': frontwall_pixels,
            'backwall': backwall_pixels,
            'flaw': flaw_pixels,
            'indication': indication_pixels,
            'total': total_pixels
        }
    }


def export_annotations_to_json(model, label_settings, output_dir="exports"):
    """
    Exporte les annotations actuelles vers un fichier JSON compatible Kili.
    Utilise le masque final pour garantir la cohérence avec les masques exportés.

    Args:
        model: Modèle contenant les polygones et images
        label_settings: Paramètres individuels par label
        output_dir: Dossier de sortie

    Returns:
        Chemin du fichier JSON généré
    """
    logger = logging.getLogger(__name__)

    try:
        # Générer le masque final avec les paramètres individuels
        # Utiliser la même logique que dans export_individual_masks
        final_mask = _generate_individual_mask_for_export(model, label_settings)

        # Créer l'exporteur JSON
        json_exporter = JsonExporter()

        # Générer le nom de fichier
        json_filename = generate_json_filename("mask_annotations")
        json_path = os.path.join(output_dir, "json", json_filename)

        # Exporter depuis le masque final
        result_path = json_exporter.export_mask_to_json(
            final_mask, model, label_settings, json_path
        )

        logger.info(f"JSON exporté: {os.path.basename(result_path)}")
        logger.info(f"Dossier: {os.path.dirname(result_path)}")
        # logger.info("Compatible avec Kili et réutilisable")
        # logger.info("Exporté depuis le masque final (cohérent avec les masques)")

        return result_path

    except Exception as e:
        logger.error(f"Erreur lors de l'export JSON: {str(e)}")
        return None


def _generate_individual_mask_for_export(model, label_settings):
    """
    Génère le masque final avec les paramètres individuels pour l'export.
    Utilise la même logique que export_individual_masks.

    Args:
        model: Modèle contenant les polygones et images
        label_settings: Paramètres par label

    Returns:
        np.ndarray: Masque final avec valeurs 1,2,3,4 pour frontwall,backwall,flaw,indication
    """
    h, w = model.image_original.shape[:2]
    final_mask = np.zeros((h, w), dtype=np.uint8)

    # Obtenir les polygones avec leurs paramètres individuels
    polygons_with_params = model.get_all_polygons()

    for label, label_value in [("frontwall", 1), ("backwall", 2), ("flaw", 3), ("indication", 4)]:
        polygons = polygons_with_params[label]

        if not polygons:
            continue

        # Traiter chaque polygone avec ses propres paramètres figés
        for polygon_data in polygons:
            points = polygon_data['points']
            parameters = polygon_data['parameters']

            if len(points) < 3:
                continue

            # Générer le masque pour ce polygone avec ses paramètres
            poly_mask = _generate_polygon_mask_with_parameters(
                model.image_original, points, parameters, label_value
            )

            # Ajouter au masque final
            final_mask = np.maximum(final_mask, poly_mask)

    return final_mask


def _generate_polygon_mask_with_parameters(image, points, parameters, label_value):
    """
    Génère un masque pour un polygone avec ses paramètres spécifiques.

    Args:
        image: Image originale
        points: Points du polygone
        parameters: Paramètres du polygone (threshold, mask_type, etc.)
        label_value: Valeur du label (1,2,3,4)

    Returns:
        np.ndarray: Masque du polygone
    """
    h, w = image.shape[:2]

    # Obtenir les paramètres
    threshold = parameters.get('threshold', 128)
    mask_type = parameters.get('mask_type', 'standard')

    # Créer le masque binaire avec le threshold
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    binary_mask = (gray < threshold).astype(np.uint8) * 255

    # Créer le masque du polygone
    points_array = np.array(points[:-1], np.int32)  # Enlever le dernier point dupliqué
    poly_mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(poly_mask, [points_array], 1)

    # Appliquer le threshold
    poly_mask &= binary_mask

    # Appliquer le type de masque
    if mask_type == "polygon":
        # Générer le profil avec verticalisation
        poly_mask = _generate_profile_mask_simple(poly_mask, label_value)
    else:
        # Mode standard : juste appliquer la valeur du label
        poly_mask[poly_mask > 0] = label_value

    return poly_mask


def _generate_profile_mask_simple(single_mask, val):
    """
    Génère un masque de profil simple avec verticalisation.

    Args:
        single_mask: Masque binaire initial
        val: Valeur du label à appliquer

    Returns:
        np.ndarray: Masque avec profil
    """
    try:
        # Trouver les contours
        contours, _ = cv2.findContours(single_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        if not contours:
            return single_mask

        # Créer le masque final
        profile_mask = np.zeros_like(single_mask)

        # Remplir les contours
        for contour in contours:
            cv2.fillPoly(profile_mask, [contour], val)

        # Appliquer la verticalisation
        h, w = profile_mask.shape
        for x in range(w):
            col = profile_mask[:, x]
            idx = np.where(col == val)[0]
            if idx.size > 0:
                y_min = idx.min()
                y_max = idx.max()
                profile_mask[y_min:y_max+1, x] = val

        return profile_mask

    except Exception:
        # En cas d'erreur, retourner le masque simple
        result = single_mask.copy()
        result[result > 0] = val
        return result


def export_all_annotations_to_json(model, label_settings, output_dir="exports"):
    """
    Exporte toutes les annotations vers un fichier JSON.
    Note: Cette fonction exporte seulement l'image actuelle car les polygones
    ne sont pas persistés entre les images dans le modèle actuel.

    Args:
        model: Modèle contenant les polygones et images
        label_settings: Paramètres individuels par label
        output_dir: Dossier de sortie

    Returns:
        Chemin du fichier JSON généré
    """
    logger = logging.getLogger(__name__)

    try:
        # Pour l'instant, exporter seulement l'image actuelle
        # car les polygones ne sont pas persistés entre les images
        logger.info("Export JSON pour l'image actuelle seulement")
        logger.info("Les polygones ne sont pas persistés entre les images")

        return export_annotations_to_json(model, label_settings, output_dir)

    except Exception as e:
        logger.error(f"Erreur lors de l'export JSON complet: {str(e)}")
        return None


def load_annotations_from_json(json_path, model):
    """
    Charge les annotations depuis un fichier JSON.

    Args:
        json_path: Chemin vers le fichier JSON
        model: Modèle pour obtenir les dimensions d'image

    Returns:
        Dictionnaire des polygones chargés
    """
    logger = logging.getLogger(__name__)

    try:
        # Créer l'exporteur JSON
        json_exporter = JsonExporter()

        # Charger les données JSON
        json_data = json_exporter.load_annotations_from_json(json_path)

        # Obtenir les dimensions de l'image actuelle (compatible mode mémoire)
        if model.current_image is not None:
            # Mode mémoire : utiliser l'image déjà chargée
            image = model.current_image
            height, width = image.shape[:2]
        else:
            # Mode fichier : charger depuis le disque
            current_image_path = model.image_list[model.current_index]
            import cv2
            image = cv2.imread(current_image_path)
            if image is None:
                raise ValueError(f"Impossible de charger l'image: {current_image_path}")
            height, width = image.shape[:2]

        # Convertir en polygones (sans marge - système de coordonnées direct)
        polygons = json_exporter.convert_json_to_polygons(
            json_data, width, height, model.current_index, margin=0
        )

        # logger.info(f"Annotations chargées depuis: {os.path.basename(json_path)}")
        for label, polys in polygons.items():
            if polys:
                # logger.info(f"{label}: {len(polys)} polygone(s)")
                pass

        return polygons

    except Exception as e:
        logger.error(f"Erreur lors du chargement JSON: {str(e)}")
        return {"frontwall": [], "backwall": [], "flaw": [], "indication": []}



