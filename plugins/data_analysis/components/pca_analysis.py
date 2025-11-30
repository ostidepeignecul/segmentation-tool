# type: ignore
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
from scipy.signal import find_peaks
from sklearn.decomposition import PCA

from sentinelai.plugins.data_analysis.DataAnalysisPluginManager import (
    AnalysisType,
    DataAnalysisInput,
    DataAnalysisOutput,
    DataAnalysisPlugin,
    data_analysis_plugin_registry,
    register_data_analysis_plugin,
)
from sentinelai.utils.data_array_preprocessing import resize_and_flatten


@dataclass
class PCAInput:
    data_grey: np.ndarray  # Grey scale data for PCA
    n_dims: int  # Number of dimensions for PCA
    all_labels: list[str] | None = None  # Labels for each data point
    target_label: str | None = None  # Specific label for targeting PCA
    index_data: np.ndarray | None = None  # Position index array


@dataclass
class PCAOutput:
    data_transformed: np.ndarray
    pca_components: np.ndarray
    pca_mean: np.ndarray
    pca_explained_variance_ratio: np.ndarray
    index_data: np.ndarray | None = None
    other_analysis: dict[str, np.ndarray] | None = None


def compute_correlation(
    data_transformed: np.ndarray, step: int = 1, figs: bool = False
) -> tuple[np.ndarray, int, np.ndarray]:
    """
    Computes the correlation between data points separated by a given step index in PCA-transformed data.
    The correlation is defined as an inner product between two "images" (data points), focusing on variations
    from the mean image.

    Args:
        data_transformed (np.ndarray): Data transformed through PCA, where components are orthogonal.
        step (int): Step between images to correlate.
        figs (bool): If True, plots figures related to the correlation process (not implemented in this snippet).

    Returns:
        Tuple[np.ndarray, int, np.ndarray]: A tuple containing the correlation array, the main periodic peaks period,
                                            and the indices marking discontinuities.
    """
    nb_files = data_transformed.shape[0]
    corr_neighbours = calculate_neighbour_correlations(data_transformed, step)
    autocorr, x_acorr = calculate_autocorrelation(corr_neighbours)
    ind_disc = identify_discontinuities(corr_neighbours, autocorr, x_acorr)

    # The main periodic peak period is derived from the discontinuity indices
    correlation_main_period = (
        np.min(np.diff(sorted(ind_disc))) if len(ind_disc) > 1 else 0
    )

    return corr_neighbours, correlation_main_period, ind_disc


def calculate_neighbour_correlations(data: np.ndarray, step: int) -> np.ndarray:
    """
    Calculate correlations between neighbouring data points.
    """
    correlations = np.array(
        [
            np.dot(data[i], data[i + step])
            / np.sqrt(np.dot(data[i], data[i]) * np.dot(data[i + step], data[i + step]))
            for i in range(data.shape[0] - step)
        ]
    )
    return correlations


def calculate_autocorrelation(
    correlations: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Calculate the autocorrelation from neighbour correlations.
    """
    y = 1 - correlations
    y[y < np.mean(y)] = 0
    autocorr = np.correlate(y, y, mode="same")
    autocorr *= autocorr > np.mean(autocorr)
    x_acorr = np.linspace(-0.5 * len(autocorr), 0.5 * len(autocorr), len(correlations))
    return autocorr, x_acorr


def identify_discontinuities(
    corr_neighbours: np.ndarray, autocorr: np.ndarray, x_acorr: np.ndarray
) -> np.ndarray:
    """
    Identify discontinuities in the data based on correlation and autocorrelation analysis.
    """
    peaks, _ = find_peaks(autocorr)
    if len(peaks) == 0:
        return np.array([], dtype=int)

    # Assuming the first peak right of the maximum autocorrelation peak is the main discontinuity indicator
    main_peak = peaks[np.argmax(autocorr[peaks])]
    ind_disc = np.array([main_peak])

    # Additional logic to refine and identify other discontinuities can be added here, not sure if the original script had it

    return ind_disc.astype(int)


@register_data_analysis_plugin(data_analysis_plugin_registry, AnalysisType.PCA)
class PCAPlugin(DataAnalysisPlugin):
    _plugin_type = AnalysisType.PCA

    def perform_analysis(self, input_data: DataAnalysisInput) -> DataAnalysisOutput:
        # Validate input_data to ensure it is appropriate for PCA
        if input_data.analysis_type != AnalysisType.PCA:
            raise ValueError("PCAPlugin can only process PCA analysis types")
        if not isinstance(input_data.data, np.ndarray):
            raise TypeError("Input data must be a numpy array")

        # Convert the input data
        if input_data.additional_params is not None:
            new_shape = input_data.additional_params.get("new_shape", (256, 256))
            nb_components = input_data.additional_params.get("nb_components", 2)
            nb_zones = input_data.additional_params.get("nb_zones", 1)
            index_data = input_data.additional_params.get("index_data")
            flaws_indexes = input_data.additional_params.get("flaws_indexes", [])
            correlation = input_data.additional_params.get(
                "calculate_correlation", False
            )
        else:
            new_shape = (256, 256)
            nb_components = 2
            nb_zones = 1
            index_data = None
            flaws_indexes = []
            correlation = False

        data_flatten = resize_and_flatten(input_data.data, new_shape)

        # Perform PCA
        pca = PCA(n_components=nb_components)
        data_transformed = pca.fit_transform(data_flatten)

        data_zones = []
        corr_zones = []
        data_flaws = []
        data_noflaws = []
        nb_data_per_zone = len(data_transformed) // nb_zones

        corr = None
        if correlation:
            corr, main_period, ind_disc = compute_correlation(data_transformed, step=1)

        for i in range(nb_zones):
            if i < nb_zones - 1:
                data_zones.append(
                    data_transformed[i * nb_data_per_zone : (i + 1) * nb_data_per_zone]
                )
                if correlation:
                    corr_zones.append(
                        corr[i * nb_data_per_zone : (i + 1) * nb_data_per_zone]
                    )
            else:
                data_zones.append(data_transformed[i * nb_data_per_zone :])
                if correlation:
                    corr_zones.append(corr[i * nb_data_per_zone :])

        for i in range(len(data_transformed)):
            if i in flaws_indexes:
                data_flaws.append(data_transformed[i])
            else:
                data_noflaws.append(data_transformed[i])

        # Construct and return the output structure
        # Create PCAOutput instance
        pca_output = PCAOutput(
            data_transformed=data_transformed,
            pca_components=pca.components_,
            pca_mean=pca.mean_,
            pca_explained_variance_ratio=pca.explained_variance_ratio_,
            index_data=index_data,
            other_analysis={
                "correlations": corr,
                "data_zones": data_zones,
                "corr_zones": corr_zones,
                "data_flaws": np.array(data_flaws),
                "data_noflaws": np.array(data_noflaws),
                "flaws_indexes": flaws_indexes,
            },
        )

        # Construct and return the DataAnalysisOutput with PCAOutput
        return DataAnalysisOutput(
            analysis_type=AnalysisType.PCA,
            analysis_display_name="Data variance (PCA)",
            results=pca_output,  # Now directly using PCAOutput instance
        )

    @property
    def plugin_id(self) -> str:
        return "PCA"

    @property
    def plugin_type(self) -> AnalysisType:
        return AnalysisType.PCA

    @property
    def plugin_display_name(self) -> str:
        return "Data variance (PCA)"
