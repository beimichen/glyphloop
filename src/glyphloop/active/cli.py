"""`glyphloop active mine` — cluster an unlabeled pool and surface discovery queries.

Embeds every image in a pool (TensorFlow-free pixel embedding by default),
clusters with OPTICS, and prints the dense, as-yet-unlabeled clusters a human
should identify next. Needs the ``[viz]`` extra for clustering.
"""

from __future__ import annotations

import argparse
from pathlib import Path


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="glyphloop active mine", description=__doc__)
    ap.add_argument("--pool", type=Path, required=True, help="dir of unlabeled images")
    ap.add_argument("--min-size", type=int, default=10, help="minimum cluster size to surface")
    ap.add_argument("--limit", type=int, default=20, help="max discovery queries to print")
    args = ap.parse_args(argv)

    from glyphloop.active.discovery import discovery_queue
    from glyphloop.embedding.cluster import optics_labels
    from glyphloop.embedding.embed import pixel_embedding

    paths = sorted(Path(args.pool).glob("*.png"))
    if not paths:
        print(f"No images found in {args.pool}")
        return

    vectors = pixel_embedding(paths)
    labels = optics_labels(vectors)
    queries = discovery_queue(
        labels, known_clusters=set(), min_size=args.min_size, limit=args.limit
    )

    n_clusters = len({int(c) for c in labels if int(c) != -1})
    print(f"{len(paths)} images -> {n_clusters} clusters; {len(queries)} discovery queries:")
    for q in queries:
        print(f"  cluster {q.cluster_id:>4}  size {q.size:>4}  -> ask a human to identify it")


if __name__ == "__main__":
    main()
