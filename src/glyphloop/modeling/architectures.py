"""The model zoo — the *real* builders, consolidated.

The original repo's trainers imported builders that didn't exist
(``build_a_mobile_hydra``, ``build_unet_captcha_model``); the working ones were
scattered across ``experimenting.py`` and ``model_building_codes.py``. They live
here now, in one place:

- :func:`build_teacher`     — a ResNet phrase classifier (accurate, heavy).
- :func:`build_student`     — a depthwise/DenseNet-style classifier (email-sized).
- :func:`build_autoencoder` — encoder/decoder for the cold-start embedder.
- :func:`build_hydra`       — a shared backbone with symbol/n-gram/word/phrase
                              heads, whose auxiliary outputs drive inference-time
                              pattern-whitelisting.

TensorFlow is imported lazily so ``import glyphloop`` works without the
``[train]`` extra.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from glyphloop.config import IMAGE_CHANNELS, IMAGE_HEIGHT, IMAGE_WIDTH
from glyphloop.data.glossary import (
    get_gram_dict,
    get_phrase_dict,
    get_word_dict,
)

INPUT_SHAPE = (IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_CHANNELS)


# ──────────────────────────── single-head classifiers ────────────────────────────
def build_teacher(num_classes: int | None = None, input_shape=INPUT_SHAPE):
    """ResNet-style phrase classifier — the accurate-but-heavy teacher."""
    import keras
    from tensorflow.keras import initializers, layers
    from tensorflow.keras.metrics import SparseCategoricalAccuracy, SparseTopKCategoricalAccuracy

    num_classes = num_classes or len(get_phrase_dict())
    init = initializers.RandomNormal(0, 0.01)
    filters = 64

    inputs = layers.Input(shape=input_shape)
    x = layers.Conv2D(filters, 3, padding="same", activation="relu", kernel_initializer=init)(
        inputs
    )
    x = layers.BatchNormalization()(x)
    for b in (1, 2, 3):
        x = layers.MaxPooling2D()(x)
        x = layers.Conv2D(
            filters * b, 1, padding="same", activation="relu", kernel_initializer=init
        )(x)
        for _ in range(4):
            y = layers.Conv2D(
                filters * b, 3, padding="same", activation="relu", kernel_initializer=init
            )(x)
            y = layers.BatchNormalization()(y)
            x = layers.Add()([x, y])
    x = layers.GlobalAveragePooling2D()(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = keras.Model(inputs, outputs, name="teacher")
    model.compile(
        loss="sparse_categorical_crossentropy",
        optimizer="adam",
        metrics=[SparseCategoricalAccuracy(), SparseTopKCategoricalAccuracy()],
    )
    return model


def build_student(
    num_classes: int | None = None, input_shape=INPUT_SHAPE, filters=128, learning_rate=1e-3
):
    """Lightweight depthwise/DenseNet-style classifier — the deployable student."""
    import keras
    from tensorflow.keras import initializers, layers, optimizers
    from tensorflow.keras.metrics import SparseCategoricalAccuracy, SparseTopKCategoricalAccuracy

    num_classes = num_classes or len(get_phrase_dict())
    init = initializers.RandomNormal(0, 0.1)

    inputs = layers.Input(shape=input_shape)
    x = inputs
    for block in (2, 3, 4):
        x = layers.Conv2D(filters, 1, padding="same", activation="relu", kernel_initializer=init)(x)
        dense = [x]
        for _ in range(block - 1):
            y = layers.DepthwiseConv2D(
                3, padding="same", activation="relu", kernel_initializer=init
            )(x)
            y = layers.BatchNormalization()(y)
            y = layers.Conv2D(
                filters, 1, padding="same", activation="relu", kernel_initializer=init
            )(y)
            y = layers.BatchNormalization()(y)
            dense.append(y)
            x = layers.Concatenate()(dense)
        x = layers.MaxPooling2D()(x)
    x = layers.GlobalAveragePooling2D()(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = keras.Model(inputs, outputs, name="student")
    model.compile(
        loss="sparse_categorical_crossentropy",
        optimizer=optimizers.Adam(learning_rate=learning_rate),
        metrics=[SparseCategoricalAccuracy(), SparseTopKCategoricalAccuracy()],
    )
    return model


# ──────────────────────────────── autoencoder ────────────────────────────────────
def build_autoencoder(input_shape=INPUT_SHAPE):
    """Convolutional autoencoder; ``encoder`` is the cold-start image embedder."""
    import keras
    from tensorflow.keras import layers

    enc_in = layers.Input(shape=input_shape)
    x = enc_in
    for f in (32, 64, 96, 128, 160, 192, 224, 256):
        x = layers.Conv2D(f, 3, padding="same", activation="relu")(x)
        x = layers.BatchNormalization()(x)
        if f in (64, 128, 192):
            x = layers.MaxPooling2D()(x)
    x = layers.GlobalAveragePooling2D()(x)
    enc_out = layers.Activation("sigmoid")(x)
    encoder = keras.Model(enc_in, enc_out, name="encoder")

    dec_in = layers.Input(shape=(256,))
    h = layers.Dense((IMAGE_WIDTH // 8) * (IMAGE_HEIGHT // 8) * 256, activation="relu")(dec_in)
    h = layers.Reshape((IMAGE_WIDTH // 8, IMAGE_HEIGHT // 8, 256))(h)
    for f in (224, 192, 160, 128, 96, 64, 32):
        h = layers.Conv2D(f, 3, padding="same", activation="relu")(h)
        h = layers.BatchNormalization()(h)
        if f in (224, 160, 96):
            h = layers.UpSampling2D()(h)
    dec_out = layers.Conv2D(IMAGE_CHANNELS, 1, padding="same", activation="sigmoid")(h)
    decoder = keras.Model(dec_in, dec_out, name="decoder")

    ae_in = layers.Input(shape=input_shape)
    ae = keras.Model(ae_in, decoder(encoder(ae_in)), name="autoencoder")
    ae.compile(loss="mse", optimizer="adam")
    return ae, encoder, decoder


# ───────────────────────────────── hydra ─────────────────────────────────────────
@dataclass(frozen=True)
class HeadSpec:
    num_classes: int
    name: str
    loss: str
    weight: float
    activation: str
    gap_size: int = 128


def default_hydra_heads() -> list[HeadSpec]:
    """Symbol/n-gram/word/phrase heads sized from the vocabulary."""
    n_mono = len(get_gram_dict(1))
    n_bi = len(get_gram_dict(2))
    n_tri = len(get_gram_dict(3))
    n_word = len(get_word_dict())
    n_phrase = len(get_phrase_dict())
    return [
        HeadSpec(n_mono, "mono", "binary_crossentropy", n_mono / 8.23, "sigmoid", 128),
        HeadSpec(n_bi, "bi", "binary_crossentropy", n_bi / 8.30, "sigmoid", 128),
        HeadSpec(n_tri, "tri", "binary_crossentropy", n_tri / 6.19, "sigmoid", 128),
        HeadSpec(n_word, "words", "binary_crossentropy", n_word / 2.36, "sigmoid", 128),
        HeadSpec(n_phrase, "phrase", "categorical_crossentropy", 1.0, "softmax", 512),
    ]


def build_hydra(input_shape=INPUT_SHAPE, heads: list[HeadSpec] | None = None):
    """Shared ResNet backbone with multiple classification heads."""
    import keras
    from tensorflow.keras import layers
    from tensorflow.keras.optimizers import Adam

    heads = heads or default_hydra_heads()

    def strided(x, filters, strides):
        x = layers.Conv2D(filters, 3, strides=strides, padding="same")(x)
        x = layers.BatchNormalization()(x)
        return layers.Activation("relu")(x)

    def residual(x, filters, blocks=1):
        for _ in range(blocks):
            y = layers.Conv2D(filters, 3, padding="same")(x)
            y = layers.BatchNormalization()(y)
            y = layers.Activation("relu")(y)
            x = layers.Add()([x, y])
        return x

    inputs = layers.Input(shape=input_shape)
    x = inputs
    for filters in (64, 128, 192):
        x = strided(x, filters, strides=(2, 2))
        x = residual(x, filters, blocks=1)

    outputs = []
    for h in heads:
        head = strided(x, h.gap_size, strides=(1, 1))
        head = layers.GlobalAveragePooling2D()(head)
        outputs.append(layers.Dense(h.num_classes, activation=h.activation, name=h.name)(head))

    model = keras.Model(inputs, outputs, name="hydra")
    model.compile(
        loss={h.name: h.loss for h in heads},
        loss_weights={h.name: h.weight for h in heads},
        optimizer=Adam(),
    )
    return model


@dataclass(frozen=True)
class ModelSpec:
    """Resolved by the trainer from a Hydra ``model`` config group."""

    kind: str = "teacher"  # teacher | student | hydra
    filters: int = 128
    learning_rate: float = 1e-3
    extra: dict = field(default_factory=dict)


def build_model(spec: ModelSpec):
    if spec.kind == "teacher":
        return build_teacher()
    if spec.kind == "student":
        return build_student(filters=spec.filters, learning_rate=spec.learning_rate)
    if spec.kind == "hydra":
        return build_hydra()
    raise ValueError(f"unknown model kind: {spec.kind!r}")
