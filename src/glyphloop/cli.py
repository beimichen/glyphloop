"""Unified ``glyphloop`` command-line entrypoint.

A thin dispatcher over the module CLIs. The Hydra-driven trainer owns its own
argument parsing, so it is invoked directly via ``python -m`` (printed below).
"""

from __future__ import annotations

import sys

_HELP = """\
glyphloop <command> [args]

Data
  synth             Render a balanced synthetic CAPTCHA dataset from the vocabulary

Method
  selftrain         Confidence-gated self-training over an unlabeled pool
  distill           Distill a teacher into the email-sized student
  evaluate          Evaluate a classifier on a held-out, human-verified split
  export            Export a student to ONNX (+ Keras-vs-ONNX parity check)
  infer             Read a CAPTCHA image with an ONNX model

Active learning
  active mine       Rank an unlabeled pool by model uncertainty

Hydra / app entrypoints (run directly):
  python -m glyphloop.modeling.train train=pretrain_synthetic data=synthetic
  streamlit run apps/label_tool.py
  python apps/demo.py
"""


def main(argv: list[str] | None = None) -> None:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] in {"-h", "--help", "help"}:
        print(_HELP)
        return

    command, rest = argv[0], argv[1:]
    if command == "synth":
        from glyphloop.data.generate import main as run
    elif command == "selftrain":
        from glyphloop.selftrain.loop import main as run
    elif command == "distill":
        from glyphloop.modeling.distill import main as run
    elif command == "evaluate":
        from glyphloop.modeling.evaluate import main as run
    elif command == "export":
        from glyphloop.modeling.export import main as run
    elif command == "infer":
        from glyphloop.inference.pipeline import main as run
    elif command == "active":
        from glyphloop.active.cli import main as run

        if rest and rest[0] == "mine":  # "glyphloop active mine ..." (only subcommand)
            rest = rest[1:]
    else:
        print(f"Unknown command: {command}\n\n{_HELP}")
        sys.exit(2)
    run(rest)


if __name__ == "__main__":
    main()
