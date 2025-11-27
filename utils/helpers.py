"""
Fonctions utilitaires pour la conversion et normalisation de données.
"""
import numpy as np


def safe_division(numerator, denominator):
    """
    Division sécurisée pour éviter la division par zéro.
    
    Args:
        numerator: Numérateur
        denominator: Dénominateur
        
    Returns:
        Résultat de la division ou 0.0 si dénominateur est zéro
    """
    try:
        return numerator / denominator
    except ZeroDivisionError:
        return 0.0



def normalize_polygon_coords(pts, margin=0):
    """
    DEPRECATED: Cette fonction est conservée pour compatibilité mais ne fait plus rien.
    La marge a été supprimée du système.

    Args:
        pts: Liste de points (x,y)
        margin: Ignoré (conservé pour compatibilité)

    Returns:
        Points inchangés
    """
    return pts

def clip_polygon_coords(pts, width, height):
    """
    Clippe les coordonnées d'un polygone aux limites de l'image.
    Utilise le pattern de clipping au lieu de rejeter les polygones hors limites.

    Args:
        pts: Liste de points (x,y)
        width: Largeur de l'image
        height: Hauteur de l'image

    Returns:
        Points clippés aux limites [0, width-1] et [0, height-1]
    """
    clipped_pts = []
    for x, y in pts:
        clipped_x = int(np.clip(x, 0, width - 1))
        clipped_y = int(np.clip(y, 0, height - 1))
        clipped_pts.append((clipped_x, clipped_y))
    return clipped_pts

def apply_binary_threshold(gray_img, threshold):
    """
    Applique un seuil binaire à une image en niveaux de gris.

    Args:
        gray_img: Image en niveaux de gris
        threshold: Seuil

    Returns:
        Masque binaire
    """
    return (gray_img < threshold).astype(np.uint8)

def calculate_auto_threshold(gray_img, polygon_points, percentage=20, max_threshold=254,
                             method="percentile", value_range=(0.0, 255.0)):
    """
    Calcule automatiquement le threshold selon la méthode choisie.
    Utilise des méthodes robustes adaptées à différents types d'images.

    Args:
        gray_img: Image en niveaux de gris (numpy array)
        polygon_points: Liste des points du polygone [(x, y), ...]
        percentage: Pourcentage d'extension du threshold (défaut: 20%)
        max_threshold: Threshold maximum autorisé (défaut: 254)
        method: Méthode de calcul (défaut: "percentile")

    Returns:
        float: Valeur du threshold calculé (même échelle que value_range)

    Méthodes disponibles:
        - "percentile": Basé sur le percentile (robuste aux valeurs extrêmes)
        - "otsu": Seuillage d'Otsu (sépare 2 classes)
        - "robust_z": Z-score robuste basé sur médiane/IQR
        - "adaptive": Seuillage adaptatif local
        - "hysteresis": Double seuil avec connectivité
        - "average": Moyenne entre min/max (ancienne méthode)
        - "third_darkest": 3ème pixel le plus foncé (ancienne méthode)
    """
    import cv2

    value_min, value_max = value_range
    if value_max <= value_min:
        value_max = value_min + 1.0
    scale = value_max - value_min

    def _denormalize(value_255: float) -> float:
        return value_min + (value_255 / 255.0) * scale

    default_threshold = _denormalize(150.0)

    if not polygon_points or len(polygon_points) < 3:
        return min(default_threshold, max_threshold)

    try:
        h, w = gray_img.shape[:2]

        # Créer un masque pour la zone de sélection
        mask = np.zeros((h, w), dtype=np.uint8)
        points_array = np.array(
            polygon_points[:-1] if polygon_points[-1] == polygon_points[0] else polygon_points,
            np.int32
        )
        cv2.fillPoly(mask, [points_array], 1)

        # Normaliser l'image en 0-255 pour réutiliser les algos existants
        working_img = gray_img.astype(np.float32)
        working_img = np.clip((working_img - value_min) / scale, 0.0, 1.0)
        working_uint8 = (working_img * 255.0).astype(np.uint8)
        h_img, w_img = working_uint8.shape[:2]

        # Extraire la région d'intérêt (ROI)
        roi = working_uint8[mask == 1]

        if len(roi) == 0:
            return min(default_threshold, max_threshold)

        # Calculer le threshold selon la méthode choisie
        if method == "percentile":
            # Méthode percentile : robuste aux valeurs extrêmes
            # Utilise le 5ème percentile comme seuil de base
            p = 5.0  # Percentile pour zones sombres
            base_threshold = np.percentile(roi, p)

        elif method == "otsu":
            # Méthode d'Otsu simplifiée : utilise l'histogramme pour trouver le seuil optimal
            if len(roi) > 1:
                # Calculer l'histogramme
                hist, bins = np.histogram(roi, bins=256, range=(0, 256))

                # Implémentation simplifiée d'Otsu
                total_pixels = len(roi)
                sum_total = np.sum(np.arange(256) * hist)

                max_variance = 0
                best_threshold = 0

                sum_background = 0
                weight_background = 0

                for t in range(256):
                    weight_background += hist[t]
                    if weight_background == 0:
                        continue

                    weight_foreground = total_pixels - weight_background
                    if weight_foreground == 0:
                        break

                    sum_background += t * hist[t]

                    mean_background = sum_background / weight_background
                    mean_foreground = (sum_total - sum_background) / weight_foreground

                    # Variance inter-classe
                    variance = weight_background * weight_foreground * (mean_background - mean_foreground) ** 2

                    if variance > max_variance:
                        max_variance = variance
                        best_threshold = t

                base_threshold = float(best_threshold)
            else:
                base_threshold = np.min(roi)

        elif method == "robust_z":
            # Z-score robuste basé sur médiane et IQR
            med = np.median(roi)
            q75, q25 = np.percentile(roi, [75, 25])
            iqr = max(1.0, q75 - q25)  # Éviter division par zéro

            # Normalisation robuste
            z_scores = (roi - med) / (iqr / 1.349)

            # Seuil : pixels significativement plus sombres que la médiane
            k = -1.5  # Cut-off pour pixels sombres
            dark_pixels = roi[z_scores <= k]

            if len(dark_pixels) > 0:
                base_threshold = np.max(dark_pixels)  # Le plus clair des pixels sombres
            else:
                base_threshold = np.min(roi)  # Fallback

        elif method == "adaptive":
            # Seuillage adaptatif local : calcul pixel par pixel dans la ROI
            # Retourne un masque binaire directement au lieu d'un seuil global

            # Calculer la taille de bloc adaptée à la ROI
            roi_area = np.sum(mask)
            block_size = min(21, max(3, int(np.sqrt(roi_area) / 6)) | 1)  # Taille adaptée, doit être impaire

            # Créer une image étendue pour éviter les effets de bord
            pad_size = block_size // 2
            padded_img = np.pad(working_uint8, pad_size, mode='reflect')
            padded_mask = np.pad(mask, pad_size, mode='constant', constant_values=0)

            # Appliquer le seuillage adaptatif sur l'image complète
            # Ajuster le paramètre C pour être plus sélectif
            c_param = max(5, int(np.std(roi) * 0.3))  # Paramètre adaptatif
            adaptive_result = cv2.adaptiveThreshold(
                padded_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY_INV, block_size, c_param
            )

            # Extraire seulement la partie correspondant à l'image originale
            adaptive_result = adaptive_result[pad_size:pad_size+h_img, pad_size:pad_size+w_img]

            # Appliquer le masque ROI : garder seulement les pixels dans la zone d'intérêt
            adaptive_result = adaptive_result & (mask * 255)

            # Pour compatibilité avec le reste du code, calculer un seuil représentatif
            # basé sur les pixels détectés par la méthode adaptative
            detected_pixels = working_uint8[(adaptive_result > 0) & (mask == 1)]
            if len(detected_pixels) > 0:
                base_threshold = np.max(detected_pixels)
            else:
                base_threshold = np.percentile(roi, 10)  # Fallback

        elif method == "hysteresis":
            # Double seuil avec connectivité : réduit le bruit
            p_high = 2.0   # Seuil strict (pixels très sombres)
            p_low = 10.0   # Seuil permissif (pixels moyennement sombres)

            t_high = np.percentile(roi, p_high)
            t_low = np.percentile(roi, p_low)

            # Utiliser le seuil bas comme base, le haut servira pour la validation
            base_threshold = t_low

        elif method == "average":
            # Ancienne méthode : moyenne entre min et max
            filtered_pixels = roi[roi < 255]  # Exclure le blanc pur
            if len(filtered_pixels) == 0:
                filtered_pixels = roi

            darkest_pixel = np.min(filtered_pixels)
            lightest_pixel = np.max(filtered_pixels)
            base_threshold = (int(darkest_pixel) + int(lightest_pixel)) / 2.0

        elif method == "third_darkest":
            # Ancienne méthode : 3ème pixel le plus foncé
            unique_pixels = np.unique(roi)
            dark_pixels = unique_pixels[unique_pixels <= 200]

            if len(dark_pixels) >= 3:
                base_threshold = dark_pixels[2]
            elif len(dark_pixels) > 0:
                base_threshold = dark_pixels[-1]  # Le plus clair des pixels foncés
            else:
                base_threshold = np.min(roi)

        elif method == "kmeans":
            # Méthode K-means : clustering en 2 classes (claire/sombre)
            if len(roi) > 1:
                try:
                    from sklearn.cluster import KMeans

                    # Préparer les données pour K-means
                    roi_reshaped = roi.reshape(-1, 1).astype(np.float32)

                    # K-means avec 2 clusters
                    kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
                    labels = kmeans.fit_predict(roi_reshaped)
                    centers = kmeans.cluster_centers_.flatten()

                    # Identifier le cluster le plus sombre
                    dark_cluster_idx = np.argmin(centers)
                    light_cluster_idx = 1 - dark_cluster_idx

                    # Calculer le seuil comme point médian entre les centres
                    base_threshold = (centers[dark_cluster_idx] + centers[light_cluster_idx]) / 2
                    base_threshold = float(base_threshold)
                except ImportError:
                    # Fallback si sklearn n'est pas disponible
                    base_threshold = np.percentile(roi, 10)
            else:
                base_threshold = np.min(roi)

        elif method == "gmm":
            # Méthode GMM (Gaussian Mixture Model) : clustering probabiliste
            if len(roi) > 1:
                try:
                    from sklearn.mixture import GaussianMixture

                    # Préparer les données pour GMM
                    roi_reshaped = roi.reshape(-1, 1).astype(np.float32)

                    # GMM avec 2 composantes
                    gmm = GaussianMixture(n_components=2, random_state=42)
                    gmm.fit(roi_reshaped)

                    # Obtenir les moyennes des composantes
                    means = gmm.means_.flatten()

                    # Identifier la composante la plus sombre
                    dark_mean = np.min(means)
                    light_mean = np.max(means)

                    # Calculer le seuil comme point médian entre les moyennes
                    base_threshold = (dark_mean + light_mean) / 2
                    base_threshold = float(base_threshold)
                except ImportError:
                    # Fallback si sklearn n'est pas disponible
                    base_threshold = np.percentile(roi, 10)
            else:
                base_threshold = np.min(roi)

        elif method == "sauvola":
            # Méthode Sauvola locale : T(x,y) = m(x,y) * [1 + k * (s(x,y)/R - 1)]
            # Calcul pixel par pixel dans la ROI avec fenêtre locale
            if len(roi) > 1:
                # Paramètres Sauvola adaptés
                roi_std = np.std(roi)
                k = 0.15 + (roi_std / 255.0) * 0.2  # Adaptatif entre 0.15 et 0.35
                R = 128  # Plage dynamique (128 pour images 8-bit)

                # Taille de fenêtre adaptée à la ROI
                roi_area = np.sum(mask)
                window_size = min(15, max(3, int(np.sqrt(roi_area) / 8)) | 1)  # Fenêtre impaire

                # Appliquer Sauvola localement dans la ROI
                sauvola_result = _apply_sauvola_local(working_uint8, mask, window_size, k, R)

                # Calculer un seuil représentatif basé sur les pixels détectés
                detected_pixels = working_uint8[(sauvola_result > 0) & (mask == 1)]
                if len(detected_pixels) > 0:
                    base_threshold = np.max(detected_pixels)
                else:
                    # Fallback : utiliser la formule Sauvola globale
                    mean_global = np.mean(roi)
                    std_global = np.std(roi)
                    base_threshold = mean_global * (1 + k * (std_global / R - 1))
                    base_threshold = float(base_threshold)
            else:
                base_threshold = np.min(roi)

        elif method == "niblack":
            # Méthode Niblack locale : T(x,y) = m(x,y) + k * s(x,y)
            # Calcul pixel par pixel dans la ROI avec fenêtre locale
            if len(roi) > 1:
                # Paramètres Niblack adaptés et plus raisonnables
                roi_std = np.std(roi)
                # Utiliser un k plus modéré et adaptatif
                k = -0.2 - (roi_std / 255.0) * 0.2  # Entre -0.2 et -0.4

                # Taille de fenêtre adaptée à la ROI
                roi_area = np.sum(mask)
                window_size = min(15, max(3, int(np.sqrt(roi_area) / 8)) | 1)  # Fenêtre impaire

                # Appliquer Niblack localement dans la ROI
                niblack_result = _apply_niblack_local(working_uint8, mask, window_size, k)

                # Pour Niblack, calculer un seuil représentatif différemment
                # car la méthode locale peut être très variable
                detected_pixels = working_uint8[(niblack_result > 0) & (mask == 1)]
                if len(detected_pixels) > 0 and len(detected_pixels) < len(roi) * 0.3:
                    # Si on a une détection raisonnable (moins de 30% des pixels)
                    # Utiliser le 50ème percentile des pixels détectés
                    base_threshold = np.percentile(detected_pixels, 50)
                else:
                    # Si trop de pixels détectés ou aucun, utiliser une approche beaucoup plus conservative
                    # Utiliser une méthode alternative basée sur les percentiles
                    base_threshold = np.percentile(roi, 5)  # Très conservateur
                    print(f"Niblack: trop de détections ({len(detected_pixels)}/{len(roi)}), fallback percentile")
            else:
                base_threshold = np.min(roi)

        else:
            # Méthode par défaut : percentile
            base_threshold = np.percentile(roi, 5.0)

        # Appliquer l'extension basée sur le pourcentage
        threshold_extension = (percentage / 100.0) * base_threshold
        calculated_threshold = base_threshold + threshold_extension

        # Clamp dans [0,255] puis ramener dans l'échelle réelle
        calculated_threshold = float(np.clip(calculated_threshold, 0.0, 255.0))
        final_threshold = _denormalize(calculated_threshold)
        final_threshold = float(min(max_threshold, final_threshold))

        return final_threshold

    except Exception as e:
        print(f"Erreur lors du calcul du threshold automatique: {e}")
        return min(default_threshold, max_threshold)


def _apply_sauvola_local(gray_img, mask, window_size, k=0.2, R=128):
    """
    Applique la méthode Sauvola locale pixel par pixel dans la ROI.

    Args:
        gray_img: Image en niveaux de gris
        mask: Masque de la ROI (1 dans la zone d'intérêt, 0 ailleurs)
        window_size: Taille de la fenêtre locale (doit être impaire)
        k: Paramètre de sensibilité Sauvola
        R: Plage dynamique

    Returns:
        Masque binaire avec pixels détectés dans la ROI
    """
    h, w = gray_img.shape
    result = np.zeros((h, w), dtype=np.uint8)

    # Padding pour gérer les bords
    pad_size = window_size // 2
    padded_img = np.pad(gray_img, pad_size, mode='reflect')
    padded_mask = np.pad(mask, pad_size, mode='constant', constant_values=0)

    # Parcourir chaque pixel de la ROI
    roi_coords = np.where(mask == 1)

    for i, j in zip(roi_coords[0], roi_coords[1]):
        # Coordonnées dans l'image paddée
        pi, pj = i + pad_size, j + pad_size

        # Extraire la fenêtre locale
        window = padded_img[pi-pad_size:pi+pad_size+1, pj-pad_size:pj+pad_size+1]
        window_mask = padded_mask[pi-pad_size:pi+pad_size+1, pj-pad_size:pj+pad_size+1]

        # Calculer moyenne et écart-type locaux seulement sur les pixels valides
        valid_pixels = window[window_mask == 1]
        if len(valid_pixels) > 0:
            mean_local = np.mean(valid_pixels)
            std_local = np.std(valid_pixels)
        else:
            # Fallback : utiliser toute la fenêtre
            mean_local = np.mean(window)
            std_local = np.std(window)

        # Formule Sauvola : T = m * [1 + k * (s/R - 1)]
        threshold_local = mean_local * (1 + k * (std_local / R - 1))

        # Appliquer le seuil local
        if gray_img[i, j] < threshold_local:
            result[i, j] = 255

    return result


def _apply_niblack_local(gray_img, mask, window_size, k=-0.2):
    """
    Applique la méthode Niblack locale pixel par pixel dans la ROI.

    Args:
        gray_img: Image en niveaux de gris
        mask: Masque de la ROI (1 dans la zone d'intérêt, 0 ailleurs)
        window_size: Taille de la fenêtre locale (doit être impaire)
        k: Paramètre de sensibilité Niblack

    Returns:
        Masque binaire avec pixels détectés dans la ROI
    """
    h, w = gray_img.shape
    result = np.zeros((h, w), dtype=np.uint8)

    # Padding pour gérer les bords
    pad_size = window_size // 2
    padded_img = np.pad(gray_img, pad_size, mode='reflect')
    padded_mask = np.pad(mask, pad_size, mode='constant', constant_values=0)

    # Parcourir chaque pixel de la ROI
    roi_coords = np.where(mask == 1)

    for i, j in zip(roi_coords[0], roi_coords[1]):
        # Coordonnées dans l'image paddée
        pi, pj = i + pad_size, j + pad_size

        # Extraire la fenêtre locale
        window = padded_img[pi-pad_size:pi+pad_size+1, pj-pad_size:pj+pad_size+1]
        window_mask = padded_mask[pi-pad_size:pi+pad_size+1, pj-pad_size:pj+pad_size+1]

        # Calculer moyenne et écart-type locaux seulement sur les pixels valides
        valid_pixels = window[window_mask == 1]
        if len(valid_pixels) > 0:
            mean_local = np.mean(valid_pixels)
            std_local = np.std(valid_pixels)
        else:
            # Fallback : utiliser toute la fenêtre
            mean_local = np.mean(window)
            std_local = np.std(window)

        # Formule Niblack : T = m + k * s
        threshold_local = mean_local + k * std_local

        # Appliquer le seuil local
        if gray_img[i, j] < threshold_local:
            result[i, j] = 255

    return result
