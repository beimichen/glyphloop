"""Human-in-the-loop label propagation — the cold-start sequel.

"What if I had a million images and no phrase list?" Embed, cluster, then ask the
human only the two questions machines are worst at:

- **Discovery** (:mod:`glyphloop.active.discovery`): "Here is a dense cluster with
  no label nearby — what does it say?" One human answer mints a new vocabulary
  entry, which feeds synthetic generation and pretraining. This grows *train*.
- **Verification** (:mod:`glyphloop.active.verification`): "I propagated label X to
  these — was I right?" Confirmation promotes them into the held-out test set,
  keeping evaluation honest as the model drifts. This grows *test*.

Clustering (:mod:`glyphloop.active.mine`) decides which of the two to ask.
"""

from glyphloop.active.mine import margin, prediction_entropy, rank_by_uncertainty

__all__ = ["margin", "prediction_entropy", "rank_by_uncertainty"]
