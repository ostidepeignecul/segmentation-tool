"""Service métier pour les opérations d'annotation/ROI (stubs)."""

from __future__ import annotations

from typing import Any, Optional, Sequence, Tuple


class AnnotationService:
    """Encapsule la logique ROI, seuils et propagation (placeholder)."""

    def compute_threshold(
        self,
        gray_slice: Any,
        polygon: Sequence[Tuple[int, int]] | None,
        *,
        auto: bool,
    ) -> Optional[float]:
        """Calcule ou retourne un seuil pour la ROI courante (stub)."""
        return None

    def build_roi_mask(
        self,
        shape: Tuple[int, int],
        polygon: Sequence[Tuple[int, int]] | None = None,
        rectangle: Tuple[int, int, int, int] | None = None,
        point: Tuple[int, int] | None = None,
    ) -> Optional[Any]:
        """Construit un masque binaire pour la ROI courante (stub)."""
        return None

    def apply_label_on_slice(
        self,
        mask: Any,
        label: int,
        *,
        persistence: bool,
    ) -> Optional[Any]:
        """Applique un label sur une slice à partir d'un masque ROI (stub)."""
        return None

    def propagate_volume(
        self,
        slice_mask: Any,
        target_depth: int,
        *,
        persistence: bool,
    ) -> Optional[Any]:
        """Propage un masque de slice dans tout le volume (stub)."""
        return None
