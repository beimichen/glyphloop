"""Discovery queue: dense, unlabeled clusters worth one human "what does this say?".

Each answered discovery query mints a new vocabulary entry, which feeds synthetic
generation and pretraining — this is how the label space grows from zero.
"""

from __future__ import annotations

from dataclasses import dataclass

from glyphloop.active.mine import ClusterStats, summarize_clusters


@dataclass(frozen=True)
class DiscoveryQuery:
    cluster_id: int
    size: int


def discovery_queue(
    labels, known_clusters: set[int], *, min_size: int = 10, limit: int | None = None
) -> list[DiscoveryQuery]:
    """Unlabeled clusters of at least ``min_size``, largest first (most leverage)."""
    stats: list[ClusterStats] = summarize_clusters(labels, known_clusters)
    queries = [
        DiscoveryQuery(s.cluster_id, s.size)
        for s in stats
        if not s.has_known_label and s.size >= min_size
    ]
    return queries[:limit] if limit is not None else queries
