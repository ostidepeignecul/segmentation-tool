from __future__ import annotations

from typing import Optional

# Palette de couleurs par defaut pour les labels d'annotation.

BACKGROUND_LABEL_ID = 0
REFLECTOR_LABEL_ID = 1
PAINT_LABEL_ID = 2
FRONTWALL_LABEL_ID = 3
BACKWALL_LABEL_ID = 4

DEFAULT_ACTIVE_LABEL_ID = PAINT_LABEL_ID
DEFAULT_CORROSION_LABEL_A_ID = FRONTWALL_LABEL_ID
DEFAULT_CORROSION_LABEL_B_ID = BACKWALL_LABEL_ID

# Couleurs des masques pour OpenCV (format BGR)
MASK_COLORS_BGR = {
    REFLECTOR_LABEL_ID: [255, 255, 255],
    PAINT_LABEL_ID: [0, 0, 255],
    FRONTWALL_LABEL_ID: [0, 255, 0],
    BACKWALL_LABEL_ID: [255, 0, 0],
    5: [0, 165, 255],
    6: [255, 100, 200],
    7: [100, 255, 100],
    8: [0, 165, 255],
    9: [255, 255, 0],
}

# Couleurs des masques pour PIL/export (format RGB)
MASK_COLORS_RGB = {
    REFLECTOR_LABEL_ID: [255, 255, 255],
    PAINT_LABEL_ID: [255, 0, 0],
    FRONTWALL_LABEL_ID: [0, 255, 0],
    BACKWALL_LABEL_ID: [0, 0, 255],
    5: [255, 165, 0],
    6: [200, 100, 255],
    7: [100, 255, 100],
    8: [255, 165, 0],
    9: [0, 255, 255],
}

# Couleurs des masques avec alpha pour OpenCV (format BGRA)
MASK_COLORS_BGRA = {
    BACKGROUND_LABEL_ID: [180, 180, 180, 200],
    REFLECTOR_LABEL_ID: [255, 255, 255, 255],
    PAINT_LABEL_ID: [0, 0, 255, 255],
    FRONTWALL_LABEL_ID: [0, 255, 0, 255],
    BACKWALL_LABEL_ID: [255, 0, 0, 255],
    5: [0, 165, 255, 255],
    6: [255, 100, 200, 255],
    7: [100, 255, 100, 255],
    8: [0, 165, 255, 255],
    9: [255, 255, 0, 255],
}

PERSISTENT_LABEL_IDS = (
    BACKGROUND_LABEL_ID,
    REFLECTOR_LABEL_ID,
    PAINT_LABEL_ID,
    FRONTWALL_LABEL_ID,
    BACKWALL_LABEL_ID,
)
USER_LABEL_START = 5

LABEL_DISPLAY_NAMES = {
    BACKGROUND_LABEL_ID: "Background",
    REFLECTOR_LABEL_ID: "Reflector",
    PAINT_LABEL_ID: "Paint",
    FRONTWALL_LABEL_ID: "FW",
    BACKWALL_LABEL_ID: "BW",
}


def format_label_text(label_id: int) -> str:
    """Return the user-facing label text while preserving the numeric class id."""
    label = int(label_id)
    display_name = LABEL_DISPLAY_NAMES.get(label)
    if display_name is None:
        if label >= int(USER_LABEL_START):
            user_index = label - int(USER_LABEL_START) + 1
            return f"BW echo {user_index} ({label})"
        return f"Label {label}"
    return f"{display_name} ({label})"


# Corrosion workflow stages
CORROSION_STAGE_BASE = "base"
CORROSION_STAGE_RAW = "raw"
CORROSION_STAGE_INTERPOLATED = "interpolated"


def normalize_corrosion_peak_selection_mode(mode: Optional[str]) -> str:
    """Normalize the corrosion peak disambiguation mode to a stable internal key."""
    value = str(mode or "").strip().casefold().replace("-", "_").replace(" ", "_")
    aliases = {
        "optimiste": "optimistic",
        "optimistic": "optimistic",
        "pessimiste": "pessimistic",
        "pessimistic": "pessimistic",
        "max_peak": "max_peak",
        "maxpeak": "max_peak",
        "peak_max": "max_peak",
    }
    return aliases.get(value, "max_peak")


def normalize_interpolation_algo(algo: Optional[str]) -> str:
    """Normalize the interpolation algorithm name to a stable internal key."""
    value = str(algo or "").strip().casefold().replace("-", "_").replace(" ", "_")
    aliases = {
        "1d_dual_axis": "1d_dual_axis",
        "1d_pchip_dual_axis": "1d_pchip_dual_axis",
        "2d_linear_nd": "2d_linear_nd",
        "2d_clough_tocher": "2d_clough_tocher",
        "1d_makima_dual_axis": "1d_makima_dual_axis",
        "2d_rbf_thin_plate": "2d_rbf_thin_plate",
        "2d_gaussian_fill": "2d_gaussian_fill",
        "brut": "1d_dual_axis",
    }
    return aliases.get(value, "1d_dual_axis")
