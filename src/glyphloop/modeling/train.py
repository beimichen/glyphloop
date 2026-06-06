"""One config-driven trainer — replaces the original ``train.py`` /
``train_model.py`` / ``experimenting.py`` trio of near-duplicate scripts.

Every training recipe is now a composed Hydra config rather than a copy-pasted
file. Examples::

    # show how a recipe composes, without training (no data / no TensorFlow):
    python -m glyphloop.modeling.train train=pretrain_synthetic data=synthetic --cfg job

    # round-0 synthetic pretrain (needs the [train] extra + a synthetic corpus):
    python -m glyphloop.modeling.train train=pretrain_synthetic data=synthetic

TensorFlow is imported lazily inside :func:`run`, so ``--cfg job`` works on a
fresh, lightweight install.
"""

from __future__ import annotations

from pathlib import Path

import hydra
import numpy as np
from omegaconf import DictConfig, OmegaConf

from glyphloop.config import paths, set_seed
from glyphloop.data.features import load_features
from glyphloop.data.generate import label_from_filename
from glyphloop.data.glossary import get_phrase_dict


def load_classification_dataset(data_dir: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """Load ``(features, sparse_phrase_labels)`` from a directory of rendered PNGs.

    Filenames carry the base64 phrase label (see :mod:`glyphloop.data.generate`);
    images whose phrase is outside the closed vocabulary are skipped.
    """
    data_dir = Path(data_dir)
    phrase_dict = get_phrase_dict()

    feats: list[np.ndarray] = []
    labels: list[int] = []
    for png in sorted(data_dir.glob("*.png")):
        phrase = label_from_filename(png.name)
        if phrase is None or phrase not in phrase_dict:
            continue
        feats.append(load_features(png))
        labels.append(phrase_dict[phrase])

    if not feats:
        raise FileNotFoundError(f"No labelled CAPTCHA PNGs found in {data_dir}")
    return np.stack(feats), np.asarray(labels, dtype=np.int64)


def run(cfg: DictConfig) -> Path:
    """Build the model from ``cfg`` and fit it; returns the saved-weights path."""
    from glyphloop.modeling.architectures import ModelSpec, build_model

    set_seed(int(cfg.get("seed", 42)))

    model = build_model(
        ModelSpec(
            kind=cfg.model.kind,
            filters=int(cfg.model.get("filters", 128)),
            learning_rate=float(cfg.model.get("learning_rate", 1e-3)),
        )
    )

    x, y = load_classification_dataset(cfg.data.path)
    print(f"Loaded {len(x)} examples across {len(set(y.tolist()))} phrase classes")

    model.fit(
        x,
        y,
        epochs=int(cfg.train.epochs),
        batch_size=int(cfg.train.batch_size),
        validation_split=float(cfg.train.get("validation_split", 0.1)),
    )

    out_dir = paths.runs / cfg.train.name
    out_dir.mkdir(parents=True, exist_ok=True)
    weights_path = out_dir / f"{cfg.model.kind}.weights.h5"
    model.save_weights(weights_path)
    OmegaConf.save(cfg, out_dir / "config.yaml")
    print(f"Saved weights -> {weights_path}")
    return weights_path


@hydra.main(version_base=None, config_path=str(paths.configs), config_name="config")
def main(cfg: DictConfig) -> None:
    run(cfg)


if __name__ == "__main__":
    main()
