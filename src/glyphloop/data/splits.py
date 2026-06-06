"""Leakage-safe train/test splitting — the safety rail that kept self-training honest.

The retrospective's key point about avoiding confirmation bias: *pseudo-labels
only ever touch train; the held-out set is the anchor*. Two rules enforce that
here:

1. :func:`split_by_group` partitions by a **group key** (e.g. source/template id)
   so near-duplicate variants of one item never straddle the train/test boundary
   — otherwise a "held-out" image is really a memorized training image.
2. :func:`assert_test_is_human_verified` refuses to let any machine-pseudo-labeled
   item into the test split: test labels must come from humans (discovery /
   verification), never from the model being evaluated.
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Generic, TypeVar

T = TypeVar("T")


@dataclass(frozen=True)
class Split(Generic[T]):
    train: list[T]
    test: list[T]

    def __len__(self) -> int:
        return len(self.train) + len(self.test)


def _stable_fraction(key: str, salt: str) -> float:
    """Deterministic value in ``[0, 1)`` from a group key (stable across runs)."""
    digest = hashlib.sha256(f"{salt}:{key}".encode()).hexdigest()
    return int(digest[:8], 16) / 0xFFFFFFFF


def split_by_group(
    items: Sequence[T],
    group_key: Callable[[T], str],
    *,
    test_frac: float = 0.2,
    salt: str = "glyphloop",
) -> Split[T]:
    """Split ``items`` so all members of a group land in the same side.

    Grouping is by a stable hash of the group key, so the assignment is
    reproducible and independent of item order or dataset size.
    """
    train: list[T] = []
    test: list[T] = []
    for item in items:
        bucket = test if _stable_fraction(group_key(item), salt) < test_frac else train
        bucket.append(item)
    return Split(train=train, test=test)


def assert_test_is_human_verified(
    test_items: Sequence[T], is_pseudo_labeled: Callable[[T], bool]
) -> None:
    """Raise if any test item carries a machine pseudo-label rather than a human one."""
    leaked = [t for t in test_items if is_pseudo_labeled(t)]
    if leaked:
        raise ValueError(
            f"{len(leaked)} pseudo-labeled item(s) leaked into the test split; "
            "test labels must be human-verified to keep evaluation honest."
        )
