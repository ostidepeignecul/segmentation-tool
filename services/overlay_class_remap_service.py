from __future__ import annotations

from typing import Iterable, Mapping

import numpy as np


class OverlayClassRemapService:
    """Pure helpers for validating and remapping overlay class ids in memory."""

    MIN_LABEL_ID = 0
    MAX_LABEL_ID = 255

    def normalize_mapping(
        self,
        raw_mapping: Mapping[int, int],
        *,
        allowed_sources: Iterable[int],
    ) -> dict[int, int]:
        """Validate and normalize a complete source->target mapping."""
        allowed = tuple(int(source) for source in allowed_sources)
        allowed_set = set(allowed)
        mapping: dict[int, int] = {}

        for source in allowed:
            if source not in raw_mapping:
                raise ValueError(f"Classe source absente du mapping: {source}")
            try:
                target = int(raw_mapping[source])
            except Exception as exc:
                raise ValueError(f"Valeur cible invalide pour la classe {source}.") from exc
            if target < self.MIN_LABEL_ID or target > self.MAX_LABEL_ID:
                raise ValueError(
                    f"Classe cible hors limites pour {source}: {target} "
                    f"(attendu {self.MIN_LABEL_ID}-{self.MAX_LABEL_ID})."
                )
            if source == 0 and target != 0:
                raise ValueError("La classe 0 doit rester 0 pour conserver le background.")
            mapping[int(source)] = target

        unexpected_sources = sorted(int(source) for source in raw_mapping if int(source) not in allowed_set)
        if unexpected_sources:
            raise ValueError(f"Classes source inattendues dans le mapping: {unexpected_sources}")

        return mapping

    def remap_mask_volume(
        self,
        mask_volume: np.ndarray,
        mapping: Mapping[int, int],
    ) -> np.ndarray:
        """Return a remapped copy of a uint8 mask volume using source-stable comparisons."""
        source_view = np.asarray(mask_volume, dtype=np.uint8)
        if source_view.ndim != 3:
            raise ValueError("Le remap des classes exige un volume 3D.")

        result = source_view.copy()
        for source, target in mapping.items():
            result[source_view == int(source)] = int(target)
        return result

    def extract_classes(self, mask_volume: np.ndarray) -> tuple[int, ...]:
        """Return the sorted unique class ids currently present in a mask volume."""
        return tuple(int(value) for value in np.unique(np.asarray(mask_volume, dtype=np.uint8)).tolist())
