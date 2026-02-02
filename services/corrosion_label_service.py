from __future__ import annotations

from typing import Iterable, Optional, Tuple


class CorrosionLabelService:
    """Pure helper for choosing a valid corrosion label pair."""

    @staticmethod
    def normalize_pair(
        labels: Iterable[int],
        *,
        label_a: Optional[int],
        label_b: Optional[int],
    ) -> Tuple[Optional[int], Optional[int]]:
        ordered = [int(lbl) for lbl in labels]
        available = set(ordered)
        if label_a not in available:
            label_a = None
        if label_b not in available:
            label_b = None
        if not ordered:
            return None, None

        if label_a is None and label_b is None:
            label_a = ordered[0]
            label_b = ordered[1] if len(ordered) > 1 else None
            return label_a, label_b

        if label_a is None:
            for lbl in ordered:
                if label_b is None or lbl != label_b:
                    label_a = lbl
                    break

        if label_b is None:
            for lbl in ordered:
                if label_a is None or lbl != label_a:
                    label_b = lbl
                    break

        if label_a == label_b:
            if len(ordered) > 1:
                for lbl in ordered:
                    if lbl != label_a:
                        label_b = lbl
                        break
            else:
                label_b = None

        return label_a, label_b
