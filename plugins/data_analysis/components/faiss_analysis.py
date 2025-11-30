import numpy as np
from dataclasses import dataclass

import faiss
from sentinelai.plugins.data_analysis.DataAnalysisPluginManager import (
    AnalysisType,
    DataAnalysisInput,
    DataAnalysisOutput,
    DataAnalysisPlugin,
    data_analysis_plugin_registry,
    register_data_analysis_plugin,
)


@dataclass
class FAISSInput:
    complete_data: np.ndarray
    # for example, to map flaws to their closest noflaw slice
    reference_data: np.ndarray
    flaws_indexes: np.ndarray
    noflaws_indexes: np.ndarray
    k: int


@dataclass
class FAISSOutput:
    distances: np.ndarray
    flaws_indexes: np.ndarray
    noflaws_indexes: np.ndarray
    abs_radius: np.ndarray
    radius: np.ndarray
    indexes: np.ndarray


def create_faiss_db(data):
    faiss_list = []
    for i in range(len(data)):
        im = data[i]
        img = im.flatten()
        faiss_list.append(img.astype(np.single))

    db = np.array(faiss_list)

    return db


def build_faiss_index(res, db, faiss_k, faiss_dim):
    # build the index
    faiss_index = faiss.IndexFlatL2(faiss_dim)
    faiss_index.add(db)  # type: ignore

    # sanity check
    D, I = faiss_index.search(db[:0], faiss_k)

    return faiss_index  # gpu_index


# def faiss_index_search(flaw_data, noflaw_data, faiss_index, faiss_k):
def faiss_index_search(complete_data, faiss_index, faiss_k=2):
    distance_list = []
    abs_radius_list = []
    radius_list = []
    index_list = []

    for i in range(len(complete_data)):
        query_img = complete_data[i]
        query_1d = query_img.flatten()
        D, I = faiss_index.search(
            np.tile(query_1d, (1, 1)).astype(np.single),
            faiss_k,
        )

        # if the first neighbor is yourself (distance==0) _and_ FAISS actually returned >1 neighbor,
        # use the second one; otherwise fall back to the first.
        if D.shape[1] > 1 and D[0, 0] == 0.0:
            best_col = 1
        else:
            best_col = 0

        distance_list.append([D[0, best_col]])
        index_list.append([I[0, best_col]])
        # radius_delta = noflaw_data['index_data'][I[0, 0]]/1000000 - flaw_data['index_data'][i]/1000000
        # abs_radius_list.append(abs(radius_delta))
        # radius_list.append(radius_delta)

    return (
        np.array(distance_list).squeeze(),
        np.array(abs_radius_list),
        np.array(radius_list),
        np.array(index_list),
    )


def faiss_search(complete_data, ref_data, faiss_k):
    # FAISS
    faiss_dim = complete_data[0].shape[0] * complete_data[0].shape[1]
    db = create_faiss_db(ref_data)
    gpu_index = build_faiss_index(None, db, faiss_k, faiss_dim)

    distance_list, abs_radius_list, radius_list, indexes_list = faiss_index_search(
        complete_data, gpu_index, faiss_k
    )

    return distance_list, abs_radius_list, radius_list, indexes_list


@register_data_analysis_plugin(data_analysis_plugin_registry, AnalysisType.FAISS)
class FAISSPlugin(DataAnalysisPlugin):
    _plugin_type = AnalysisType.FAISS

    def perform_analysis(
        self, input_data: DataAnalysisInput | FAISSInput
    ) -> DataAnalysisOutput:
        # Validate input_data to ensure it is appropriate for FAISS
        if isinstance(input_data, DataAnalysisInput):
            if input_data.analysis_type != AnalysisType.FAISS:
                raise ValueError("FAISSPlugin can only process FAISS analysis types")
            if not isinstance(input_data.data, np.ndarray):
                raise TypeError("Input data must be a numpy array")
            complete_data = input_data.data
            # Convert the input data
            if input_data.additional_params is not None:
                ref_data = input_data.additional_params.get("ref_data", None)
                # Maybe ref_data key does not exist and is instea complete_data

                flaws_indexes = input_data.additional_params.get(
                    "flaws_indexes", np.array([[]])
                )
                noflaws_indexes = input_data.additional_params.get(
                    "noflaws_indexes", np.array([[]])
                )
                k = input_data.additional_params.get("k", 1)
            else:
                ref_data = np.array([])
                flaws_indexes = np.array([])
                noflaws_indexes = np.array([])
                k = 1
        elif isinstance(input_data, FAISSInput):
            complete_data = input_data.complete_data
            ref_data = input_data.reference_data
            flaws_indexes = input_data.flaws_indexes
            noflaws_indexes = input_data.noflaws_indexes
            k = input_data.k

        # Perform FAISS
        distance, abs_radius, radius, indexes = faiss_search(complete_data, ref_data, k)

        distances = []

        for i in range(len(distance)):
            if i in flaws_indexes:
                distances.append([i, distance[i], 1])
            else:
                distances.append([i, distance[i], 0])

        # Construct and return the output structure
        # Create FAISSOutput instance
        faiss_output = FAISSOutput(
            distances=np.array(distances).squeeze(),
            flaws_indexes=np.array(flaws_indexes),
            noflaws_indexes=np.array(noflaws_indexes),
            abs_radius=abs_radius,
            radius=radius,
            indexes=indexes,
        )

        # Construct and return the DataAnalysisOutput with FAISSOutput
        return DataAnalysisOutput(
            analysis_type=AnalysisType.FAISS,
            analysis_display_name="Difference between flaw and nearest non-flaw",
            results=faiss_output,  # Now directly using FAISSOutput instance
        )

    @property
    def plugin_id(self) -> str:
        return "FAISS"

    @property
    def plugin_type(self) -> AnalysisType:
        return AnalysisType.FAISS

    @property
    def plugin_display_name(self) -> str:
        return "Difference between flaw and nearest non-flaw"
