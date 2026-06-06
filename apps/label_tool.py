"""Streamlit human-in-the-loop labeling tool — the cold-start label propagation UI.

Two tabs mirror the two productive questions:

- **Discover**: dense unlabeled clusters; the human types what one says -> mints a
  new vocabulary entry (grows *train*).
- **Verify**: items that inherited a propagated label; the human confirms/rejects
  -> confirmed items can enter the held-out test set (grows *test*).

Run:  uv run --extra app streamlit run apps/label_tool.py
This is a thin shell over the pure queue logic in glyphloop.active.
"""

from __future__ import annotations

from pathlib import Path

import streamlit as st

from glyphloop.active.discovery import discovery_queue
from glyphloop.embedding.cluster import optics_labels
from glyphloop.embedding.embed import pixel_embedding

st.set_page_config(page_title="glyphloop labeler", layout="wide")
st.title("glyphloop — human-in-the-loop label propagation")

pool = st.sidebar.text_input("Unlabeled pool directory", value="data/raw/unlabeled")
min_size = st.sidebar.slider("Min cluster size", 2, 50, 10)

tab_discover, tab_verify = st.tabs(["Discover (grows train)", "Verify (grows test)"])

with tab_discover:
    st.caption(
        "Dense clusters with no known label nearby — identify one to mint a vocabulary entry."
    )
    paths = sorted(Path(pool).glob("*.png")) if Path(pool).is_dir() else []
    if not paths:
        st.info(f"No images in `{pool}`. Point the sidebar at a pool of CAPTCHA PNGs.")
    else:
        labels = optics_labels(pixel_embedding(paths))
        for q in discovery_queue(labels, known_clusters=set(), min_size=min_size, limit=25):
            cols = st.columns([1, 3])
            cols[0].metric(f"cluster {q.cluster_id}", f"{q.size} imgs")
            cols[1].text_input(
                "What does this cluster say?",
                key=f"discover_{q.cluster_id}",
                placeholder="type the phrase…",
            )

with tab_verify:
    st.caption(
        "Items that inherited a propagated label, least-similar-to-core first. "
        "Confirm to promote into the held-out test set."
    )
    st.info(
        "Wire this tab to a propagation run (glyphloop.active.verification.verification_queue). "
        "Kept intentionally minimal — the queue logic is unit-tested in tests/test_active.py."
    )
