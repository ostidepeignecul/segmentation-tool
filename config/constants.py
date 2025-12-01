# Palette de couleurs par défaut pour les labels (générique, indices numériques).
# Les labels sont créés dynamiquement, sans noms prédéfinis.
# Couleurs des masques pour OpenCV (format BGR)
MASK_COLORS_BGR = {
    1: [0, 0, 255],
    2: [0, 255, 0],
    3: [255, 0, 0],
    4: [0, 165, 255],
    5: [255, 100, 200],
    6: [100, 255, 100],
    7: [0, 165, 255],
    8: [255, 255, 0],
    9: [0, 0, 255],
}

# Couleurs des masques pour PIL/export (format RGB)
MASK_COLORS_RGB = {
    1: [255, 0, 0],
    2: [0, 255, 0],
    3: [0, 0, 255],
    4: [255, 165, 0],
    5: [200, 100, 255],
    6: [100, 255, 100],
    7: [255, 165, 0],
    8: [0, 255, 255],
    9: [255, 0, 0],
}

# Couleurs des masques avec alpha pour OpenCV (format BGRA)
MASK_COLORS_BGRA = {
    1: [0, 0, 255, 255],
    2: [0, 255, 0, 255],
    3: [255, 0, 0, 255],
    4: [0, 165, 255, 255],
    5: [255, 100, 200, 255],
    6: [100, 255, 100, 255],
    7: [0, 165, 255, 255],
    8: [255, 255, 0, 255],
    9: [0, 0, 255, 255],
}

