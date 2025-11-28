"""
Global constants and mappings.
"""

# Mapping des classes vers leurs valeurs numériques
CLASS_MAP = {
    'frontwall': 1,
    'backwall': 2,
    'flaw': 3,
    'indication': 4,
    'erase': 0,  # Label d'effacement (remet la zone à l'arrière-plan)
}

# Paramètres par défaut pour chaque label
LABEL_SETTINGS = {
    'frontwall': {'threshold': 128, 'smooth_kernel': 5},
    'backwall':    {'threshold': 128, 'smooth_kernel': 5},
    'flaw':        {'threshold': 128, 'smooth_kernel': 5},
    'indication':  {'threshold': 128, 'smooth_kernel': 5},
    'erase':       {'threshold': 128, 'smooth_kernel': 5},
}

# Couleurs des labels pour l'interface utilisateur (format hexadécimal)
LABEL_COLORS_HEX = {
    'frontwall': '#FF0000',  # Rouge (interchangé avec flaw)
    'backwall': '#00FF00',   # Vert
    'flaw': '#0000FF',       # Bleu (interchangé avec frontwall)
    'indication': '#FFA500',  # Orange
    'erase': '#888888',      # Gris neutre pour les zones d'effacement
}

# Couleurs des masques pour OpenCV (format BGR)
MASK_COLORS_BGR = {
    # === CLASSES D'ANNOTATION (0-4) ===
    1: [0, 0, 255],     # Rouge pour frontwall (BGR) - interchangé avec flaw
    2: [0, 255, 0],     # Vert pour backwall (BGR)
    3: [255, 0, 0],     # Bleu pour flaw (BGR) - interchangé avec frontwall
    4: [0, 165, 255],   # Orange pour indication (BGR)
    # === CLASSES DE VISUALISATION (5-9) ===
    5: [255, 100, 200], # Mauve pour plot frontwall max (BGR)
    6: [100, 255, 100], # Vert clair pour plot backwall max (BGR)
    7: [0, 165, 255],   # Orange pour plot flaw max (BGR)
    8: [255, 255, 0],   # Cyan pour plot indication max (BGR)
    9: [0, 0, 255]      # Rouge vif pour lignes verticales / mesure (BGR)
}

# Couleurs des masques pour PIL/export (format RGB)
MASK_COLORS_RGB = {
    # === CLASSES D'ANNOTATION (0-4) ===
    1: [255, 0, 0],     # Rouge pour frontwall (RGB) - interchangé avec flaw
    2: [0, 255, 0],     # Vert pour backwall (RGB)
    3: [0, 0, 255],     # Bleu pour flaw (RGB) - interchangé avec frontwall
    4: [255, 165, 0],   # Orange pour indication (RGB)
    # === CLASSES DE VISUALISATION (5-9) ===
    5: [200, 100, 255], # Mauve pour plot frontwall max (RGB)
    6: [100, 255, 100], # Vert clair pour plot backwall max (RGB)
    7: [255, 165, 0],   # Orange pour plot flaw max (RGB)
    8: [0, 255, 255],   # Cyan pour plot indication max (RGB)
    9: [255, 0, 0]      # Rouge vif pour lignes verticales / mesure (RGB)
}

# Couleurs des masques avec alpha pour OpenCV (format BGRA)
MASK_COLORS_BGRA = {
    # === CLASSES D'ANNOTATION (0-4) ===
    1: [0, 0, 255, 255],     # Rouge pour frontwall (BGRA) - interchangé avec flaw
    2: [0, 255, 0, 255],     # Vert pour backwall (BGRA)
    3: [255, 0, 0, 255],     # Bleu pour flaw (BGRA) - interchangé avec frontwall
    4: [0, 165, 255, 255],   # Orange pour indication (BGRA)
    # === CLASSES DE VISUALISATION (5-9) ===
    5: [255, 100, 200, 255], # Mauve pour plot frontwall max (BGRA)
    6: [100, 255, 100, 255], # Vert clair pour plot backwall max (BGRA)
    7: [0, 165, 255, 255],   # Orange pour plot flaw max (BGRA)
    8: [255, 255, 0, 255],   # Cyan pour plot indication max (BGRA)
    9: [0, 0, 255, 255]      # Rouge vif pour lignes verticales / mesure (BGRA)
}

