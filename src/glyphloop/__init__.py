"""glyphloop — learning to read distorted text from almost no labels.

The method, not the dataset, is the artifact: a synthetic bootstrap, then
confidence-gated self-training on a large unlabeled pool, then distillation into
an email-sized model — plus a cold-start active-learning loop (embed -> cluster
-> human-in-the-loop label propagation).

Heavy stacks (TensorFlow, scikit-learn) are imported lazily inside the functions
that need them, so ``import glyphloop`` stays cheap and the synthesis + inference
paths work without the optional ``[train]`` extras installed.
"""

from glyphloop.config import IMAGE_SIZE, PROJECT_ROOT, paths, set_seed

__version__ = "0.1.0"

__all__ = ["IMAGE_SIZE", "PROJECT_ROOT", "paths", "set_seed", "__version__"]
