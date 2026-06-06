# glyphloop task runner — `just --list` to see everything.
# Requires uv (https://docs.astral.sh/uv). Most recipes wrap `uv run`.

set dotenv-load := true

# Show available recipes
default:
    @just --list

# Install the project + dev tools into a local .venv (lightweight deps)
setup:
    uv sync

# Install with every optional extra (train, viz, app)
setup-all:
    uv sync --all-extras

# Lint + format-check (what CI runs)
lint:
    uv run ruff check .
    uv run ruff format --check .

# Auto-fix lint issues and format
format:
    uv run ruff check --fix .
    uv run ruff format .

# Static type check
typecheck:
    uv run pyright

# Run the test suite
test:
    uv run pytest

# Generate a handful of synthetic CAPTCHA samples to eyeball (output is gitignored)
sample n="8":
    uv run python scripts/make_sample_images.py --n {{n}}

# Regenerate the committed t-SNE / clustering figure from fresh synthetic data
figures:
    uv run --extra viz python scripts/make_cluster_figure.py

# Show how a Hydra training recipe composes (no training; needs no data)
config recipe="pretrain_synthetic":
    uv run python -m glyphloop.modeling.train train={{recipe}} --cfg job

# --- the recipes below need the [train] extra (TensorFlow) and a dataset ---

# Round 0: pretrain the teacher on synthetic data
train-pretrain:
    uv run --extra train python -m glyphloop.modeling.train train=pretrain_synthetic data=synthetic

# Round N: self-train — pseudo-label the unlabeled pool, then fine-tune
selftrain pool:
    uv run --extra train glyphloop selftrain --pool {{pool}}

# Distill the accurate teacher into the email-sized student
distill teacher:
    uv run --extra train glyphloop distill --teacher {{teacher}}

# Export a trained student to ONNX (+ parity check)
export weights:
    uv run --extra train glyphloop export --weights {{weights}}
