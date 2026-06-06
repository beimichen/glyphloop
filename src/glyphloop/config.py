"""Project-root-relative paths and reproducibility helpers.

Every path in the project derives from :data:`PROJECT_ROOT` (when running from a
checkout) so the code runs unchanged on any machine — no hardcoded ``/Users/...``
paths, which the original repo had everywhere. Override any location with an
environment variable, e.g. ``GLYPHLOOP_DATA_DIR``.
"""

from __future__ import annotations

import os
import random
from dataclasses import dataclass, field
from pathlib import Path

# src/glyphloop/config.py -> parents[2] is the repository root (when run from a
# source checkout). For an installed wheel there is no repo, so callers should
# rely on :func:`fonts_dir` rather than ``PROJECT_ROOT`` for packaged assets.
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Image geometry the models and synthesis pipeline agree on: 128 wide, 64 tall,
# single (grayscale) channel.
IMAGE_WIDTH = 128
IMAGE_HEIGHT = 64
IMAGE_CHANNELS = 1
IMAGE_SIZE = (IMAGE_WIDTH, IMAGE_HEIGHT)
# Model input tensor shape (W, H, C).
INPUT_SHAPE = (IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_CHANNELS)


def _resolve(env_var: str, default: Path) -> Path:
    """Return an env-var override (if set) else the default, as an absolute Path."""
    value = os.environ.get(env_var)
    return Path(value).expanduser().resolve() if value else default


def fonts_dir() -> Path:
    """Locate the bundled TrueType fonts, in a source checkout or installed wheel."""
    candidates = [
        PROJECT_ROOT / "assets" / "fonts",  # source checkout
        Path(__file__).resolve().parent / "_assets" / "fonts",  # installed wheel
    ]
    for path in candidates:
        if path.is_dir():
            return path
    # Fall back to the first candidate; callers raise a clear error if it's empty.
    return candidates[0]


def vocab_dir() -> Path:
    """Directory holding the phrase word-lists shipped with the package."""
    return Path(__file__).resolve().parent / "data" / "vocab"


@dataclass(frozen=True)
class Paths:
    """Canonical project directories, all relative to :data:`PROJECT_ROOT`."""

    root: Path = PROJECT_ROOT
    data: Path = field(
        default_factory=lambda: _resolve("GLYPHLOOP_DATA_DIR", PROJECT_ROOT / "data")
    )
    models: Path = field(
        default_factory=lambda: _resolve("GLYPHLOOP_MODELS_DIR", PROJECT_ROOT / "models")
    )
    reports: Path = field(default_factory=lambda: PROJECT_ROOT / "reports")
    configs: Path = field(default_factory=lambda: PROJECT_ROOT / "configs")
    runs: Path = field(default_factory=lambda: PROJECT_ROOT / "runs")

    @property
    def data_raw(self) -> Path:
        return self.data / "raw"

    @property
    def data_interim(self) -> Path:
        return self.data / "interim"

    @property
    def data_processed(self) -> Path:
        return self.data / "processed"

    @property
    def synthetic(self) -> Path:
        return self.data_processed / "synthetic"

    @property
    def figures(self) -> Path:
        return self.reports / "figures"


paths = Paths()


def set_seed(seed: int = 42) -> int:
    """Seed Python, NumPy and (if installed) TensorFlow for reproducible runs.

    Returns the seed so callers can log it. TensorFlow is imported lazily so this
    works without the optional ``[train]`` extra.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)

    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:  # pragma: no cover - numpy is a core dep, defensive only
        pass

    try:
        import tensorflow as tf

        tf.random.set_seed(seed)
    except ImportError:
        pass  # tensorflow is an optional extra; seeding the rest is still useful.

    return seed
