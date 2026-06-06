"""Verification queue: confirm propagated labels before they enter the test set.

When a known label is propagated across a cluster, a human spot-checks it. The
risk being guarded against is *under-segmentation* — a cluster that looks like one
phrase but actually mixes two — so the items closest to the cluster boundary (the
least confidently-propagated) are surfaced first.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Propagation:
    item_id: str
    cluster_id: int
    label: str
    similarity: float  # how close the item sits to the cluster's labeled core (0..1)


@dataclass(frozen=True)
class VerificationQuery:
    item_id: str
    label: str
    similarity: float


def verification_queue(
    propagations: list[Propagation], *, max_similarity: float = 1.0, limit: int | None = None
) -> list[VerificationQuery]:
    """Propagated items to spot-check, least-similar-to-core first.

    Items with ``similarity > max_similarity`` are treated as safe and skipped.
    """
    candidates = [p for p in propagations if p.similarity <= max_similarity]
    candidates.sort(key=lambda p: p.similarity)
    queries = [VerificationQuery(p.item_id, p.label, p.similarity) for p in candidates]
    return queries[:limit] if limit is not None else queries
