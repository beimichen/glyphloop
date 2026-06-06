"""Generate the committed t-SNE clustering figure — fully reproducible, no weights.

Renders synthetic CAPTCHAs for a handful of phrases, embeds each render's
*text region* (content-cropped and size-normalized, the placement/scale invariance
a learned embedder would provide), projects to 2D with t-SNE, and colors by phrase.

The takeaway it illustrates: once you remove the background noise and random
placement, same-phrase renders fall into the same neighborhood — which is the
premise of the cold-start "embed -> cluster -> ask a human" loop. (A naive
whole-image pixel embedding does *not* separate them, which is exactly why the
real system needed a learned embedder — see docs/METHOD.md.)

Nothing but the resulting PNG is written; no synthetic images are saved.

Run:  uv run --extra viz python scripts/make_cluster_figure.py
"""

from __future__ import annotations

import numpy as np

from glyphloop.config import IMAGE_SIZE, paths, set_seed
from glyphloop.data.synthesis import render_text
from glyphloop.data.synthesis.measurement import get_smallest_box_crop
from glyphloop.embedding.cluster import tsne_2d

PHRASES = [
    "1.21 gigawatts",
    "abracadabra",
    "all your base",
    "ace of spades",
    "zombie attack",
    "against the grain",
    "absolute zero",
    "zig zag",
]
PER_PHRASE = 18
DIFFICULTY = 1
CROP_SIZE = (48, 16)


def embed_text_region(textground) -> np.ndarray:
    """Content-crop the text layer, size-normalize, and flatten to a vector."""
    x0, y0, x1, y1 = get_smallest_box_crop(textground)
    crop = textground.crop((x0, y0, max(x1, x0 + 1), max(y1, y0 + 1))).resize(CROP_SIZE)
    return np.asarray(crop, dtype=np.float32).reshape(-1) / 255.0


def main() -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    set_seed(7)
    vectors: list[np.ndarray] = []
    phrase_of: list[str] = []
    for phrase in PHRASES:
        for _ in range(PER_PHRASE):
            _, _, _, _, textground = render_text(IMAGE_SIZE, DIFFICULTY, phrase)
            vectors.append(embed_text_region(textground))
            phrase_of.append(phrase)

    coords = tsne_2d(np.stack(vectors), perplexity=15)

    fig, ax = plt.subplots(figsize=(8, 6))
    cmap = plt.get_cmap("tab10")
    for idx, phrase in enumerate(PHRASES):
        pts = [coords[i] for i in range(len(phrase_of)) if phrase_of[i] == phrase]
        ax.scatter(
            [p[0] for p in pts],
            [p[1] for p in pts],
            s=30,
            color=cmap(idx % 10),
            label=phrase,
            alpha=0.85,
        )
    ax.set_title(
        "Synthetic CAPTCHAs cluster by phrase\n(t-SNE of size-normalized text-region embeddings)"
    )
    ax.set_xticks([])
    ax.set_yticks([])
    ax.legend(loc="center left", bbox_to_anchor=(1, 0.5), fontsize=8, frameon=False)
    fig.tight_layout()

    out_dir = paths.figures
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "tsne_synthetic_clusters.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
