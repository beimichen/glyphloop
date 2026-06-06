"""Export a trained Keras student to ONNX for portable, TensorFlow-free inference.

The lightweight inference path (:mod:`glyphloop.inference`) runs on ONNX Runtime
alone — no TF, no CUDA — so the demo and any service stay small. A parity check
compares Keras vs ONNX logits on random input to catch conversion drift.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from glyphloop.config import INPUT_SHAPE, paths


def export_to_onnx(
    weights: Path, out: Path | None = None, *, opset: int = 17, check: bool = True
) -> Path:
    """Convert a student ``.weights.h5`` to ONNX. Returns the ONNX path."""
    import tf2onnx

    from glyphloop.modeling.architectures import build_student

    model = build_student()
    model.load_weights(weights)

    out = out or (paths.models / "student.onnx")
    out.parent.mkdir(parents=True, exist_ok=True)

    spec = (
        __import__("tensorflow").TensorSpec((None, *INPUT_SHAPE), dtype="float32", name="input"),
    )
    model_proto, _ = tf2onnx.convert.from_keras(
        model, input_signature=spec, opset=opset, output_path=str(out)
    )
    print(f"Exported ONNX -> {out}")

    if check:
        _parity_check(model, out)
    return out


def _parity_check(keras_model, onnx_path: Path, n: int = 4, tol: float = 1e-3) -> None:
    import onnxruntime as ort

    x = np.random.rand(n, *INPUT_SHAPE).astype("float32")
    keras_out = keras_model.predict(x, verbose=0)

    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    onnx_out = sess.run(None, {sess.get_inputs()[0].name: x})[0]

    max_diff = float(np.abs(keras_out - onnx_out).max())
    status = "OK" if max_diff < tol else "WARN"
    print(f"[{status}] Keras vs ONNX max abs diff = {max_diff:.2e} (tol {tol:.0e})")


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="glyphloop export", description=__doc__)
    ap.add_argument("--weights", type=Path, required=True, help="student .weights.h5")
    ap.add_argument(
        "--out", type=Path, default=None, help="output .onnx (default: models/student.onnx)"
    )
    ap.add_argument("--opset", type=int, default=17)
    ap.add_argument("--no-check", action="store_true", help="skip the Keras-vs-ONNX parity check")
    args = ap.parse_args(argv)
    export_to_onnx(args.weights, args.out, opset=args.opset, check=not args.no_check)


if __name__ == "__main__":
    main()
