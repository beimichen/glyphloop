"""End-to-end phrase recognition: ONNX model -> ranked phrase guess.

Runs on ONNX Runtime alone (no TensorFlow). Handles both model shapes:

- a single-output **student** (phrase softmax): argmax / top-k directly;
- the multi-output **hydra**: fuse the heads via
  :func:`glyphloop.inference.postprocess.rerank_phrases`.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from glyphloop.data.features import load_features
from glyphloop.data.glossary import get_gram_dict, get_word_dict, load_phrases
from glyphloop.inference import postprocess

_HEAD_ORDER = ("mono", "bi", "tri", "words", "phrase")


class PhraseRecognizer:
    """Load an ONNX model once and recognize phrases from image paths."""

    def __init__(self, onnx_path: str | Path):
        import onnxruntime as ort

        self.session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name
        self.n_outputs = len(self.session.get_outputs())
        self.phrases = list(load_phrases())
        if self.n_outputs > 1:
            self.head_dicts = {
                "mono": get_gram_dict(1),
                "bi": get_gram_dict(2),
                "tri": get_gram_dict(3),
                "words": get_word_dict(),
            }

    def recognize(self, image_path: str | Path, k: int = 5) -> list[tuple[float, str]]:
        x = load_features(image_path)[None, ...].astype("float32")
        outputs = [np.asarray(o) for o in self.session.run(None, {self.input_name: x})]

        if self.n_outputs == 1:
            probs = outputs[0][0]
            ranked = sorted(
                ((float(probs[i]), p) for i, p in enumerate(self.phrases)), reverse=True
            )
            return postprocess.top_k(ranked, k)

        named = {name: outputs[i][0] for i, name in enumerate(_HEAD_ORDER[: self.n_outputs])}
        ranked = postprocess.rerank_phrases(self.phrases, named["phrase"], named, self.head_dicts)
        return postprocess.top_k(ranked, k)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="glyphloop infer", description=__doc__)
    ap.add_argument("--model", type=Path, required=True, help="ONNX model path")
    ap.add_argument("--image", type=Path, required=True, help="CAPTCHA image to read")
    ap.add_argument("-k", type=int, default=5)
    args = ap.parse_args(argv)

    recognizer = PhraseRecognizer(args.model)
    for score, phrase in recognizer.recognize(args.image, args.k):
        print(f"{score:.4f}  {phrase}")


if __name__ == "__main__":
    main()
