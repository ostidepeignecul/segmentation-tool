import os
import numpy as np
from cmap import Colormap as cmap_colormap

  
# Creates a diverging version of the omniscan colormap
def get_omniscan_diverging_colormap(omniscan_cmap_path):
    if os.path.exists(omniscan_cmap_path):
        omniscan_cmap_array = np.load(omniscan_cmap_path)

    # Diverges on the 128th index, with the 128th value being the 0 value, and the 0th value and 255th value being the same
    # We first ''compress'' the colormap to 128 values by downsampling to not lose the diverging effect, then we mirror the colormap
    # resample from 0,255 to 0,128 with linear interpolation
    omniscan_cmap_array = np.array(
        [
            np.interp(
                np.linspace(0, 255, 128),
                np.linspace(0, 255, 256),
                omniscan_cmap_array[:, i],
            )
            for i in range(3)
        ]
    ).T
    omniscan_cmap_array = np.vstack(
        [omniscan_cmap_array[::-1][0:127], omniscan_cmap_array]
    )
    alpha_channel = np.full((omniscan_cmap_array.shape[0], 1), 1)

    # Set alpha to 0 where all RGB values are 0
    mask = np.all(omniscan_cmap_array == 0, axis=1)
    alpha_channel[mask] = 0
    omniscan_cmap_array = np.hstack([omniscan_cmap_array, alpha_channel])
    omniscan_cmap = cmap_colormap(
        omniscan_cmap_array, name="Omniscan Diverging Colormap"
    )

    return omniscan_cmap