"""Gradio synthesis + recognition playground.

Renders a synthetic CAPTCHA for a chosen phrase live (the generator is the part
that always runs on a fresh clone — no weights or data needed). If an ONNX model
is present at ``models/student.onnx`` it also shows the top-k phrase guesses.

Run:  python apps/demo.py   (needs the [app] extra: `uv run --extra app python apps/demo.py`)
"""

from __future__ import annotations

import random

import gradio as gr

from glyphloop.config import IMAGE_SIZE, paths
from glyphloop.data.glossary import load_phrases
from glyphloop.data.synthesis import render_text

PHRASES = list(load_phrases())
ONNX_PATH = paths.models / "student.onnx"


def _recognizer():
    if not ONNX_PATH.exists():
        return None
    try:
        from glyphloop.inference.pipeline import PhraseRecognizer

        return PhraseRecognizer(ONNX_PATH)
    except Exception:
        return None


def generate(phrase: str, difficulty: int):
    phrase = phrase.strip() or random.choice(PHRASES)
    captcha, _, _, _, _ = render_text(IMAGE_SIZE, int(difficulty), phrase)
    captcha = captcha.resize((IMAGE_SIZE[0] * 3, IMAGE_SIZE[1] * 3))

    recognizer = _recognizer()
    if recognizer is None:
        note = (
            f"Rendered **{phrase}**.\n\nNo model at `{ONNX_PATH}` — this is the "
            "synthesis playground. Train + export a student to enable recognition "
            "(see models/README.md)."
        )
        return captcha, note

    # NB: recognizer wants a path; save to a temp file under the (gitignored) data dir.
    tmp = paths.data / "_demo_tmp.png"
    tmp.parent.mkdir(parents=True, exist_ok=True)
    captcha.save(tmp)
    guesses = recognizer.recognize(tmp, k=5)
    table = "\n".join(f"{score:.3f}  {p}" for score, p in guesses)
    return captcha, f"Rendered **{phrase}**.\n\nTop guesses:\n```\n{table}\n```"


def build() -> gr.Blocks:
    with gr.Blocks(title="glyphloop") as demo:
        gr.Markdown("# glyphloop — synthesis & recognition playground")
        with gr.Row():
            phrase = gr.Textbox(label="Phrase (blank = random from the vocabulary)")
            difficulty = gr.Slider(0, 4, value=2, step=1, label="Distortion level")
        go = gr.Button("Render")
        image = gr.Image(label="Synthetic CAPTCHA")
        out = gr.Markdown()
        go.click(generate, inputs=[phrase, difficulty], outputs=[image, out])
    return demo


if __name__ == "__main__":
    build().launch()
