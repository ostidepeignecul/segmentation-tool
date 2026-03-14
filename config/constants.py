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
    100: [255, 255, 255],
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
    100: [255, 255, 255],
}

# Couleurs des masques avec alpha pour OpenCV (format BGRA)
MASK_COLORS_BGRA = {
    0: [180, 180, 180, 200],
    1: [0, 0, 255, 255],
    2: [0, 255, 0, 255],
    3: [255, 0, 0, 255],
    4: [0, 165, 255, 255],
    5: [255, 100, 200, 255],
    6: [100, 255, 100, 255],
    7: [0, 165, 255, 255],
    8: [255, 255, 0, 255],
    9: [0, 0, 255, 255],
    100: [255, 255, 255, 255],
}

PERSISTENT_LABEL_IDS = (0, 1, 2, 3, 100)
USER_LABEL_START = 4

LABEL_DISPLAY_NAMES = {
    0: "Erase",
    1: "Paint",
    2: "Frontwall",
    3: "Backwall",
    100: "Reflector",
}


def format_label_text(label_id: int) -> str:
    """Return the user-facing label text while preserving the numeric class id."""
    label = int(label_id)
    display_name = LABEL_DISPLAY_NAMES.get(label)
    if display_name is None:
        return f"Label {label}"
    return f"{display_name} ({label})"

