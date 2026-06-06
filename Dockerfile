# syntax=docker/dockerfile:1
# Lightweight image for the inference / data-generation path. uv resolves the
# default (no-TensorFlow) dependency set into a venv in the builder stage; the
# runtime stage copies only that venv plus the source. Training is intentionally
# out of scope for this image (the [train] extra pulls a multi-GB TF/CUDA stack).

FROM ghcr.io/astral-sh/uv:python3.11-bookworm-slim AS builder
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=0
WORKDIR /app

# Install dependencies first (cached layer), then the project itself.
COPY pyproject.toml uv.lock* ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --no-install-project --no-dev
COPY . .
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --no-dev

FROM python:3.11-slim-bookworm AS runtime
RUN useradd --create-home --uid 1000 app
WORKDIR /app
COPY --from=builder --chown=app:app /app /app
ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1
USER app

# `docker run <img> --help` prints the CLI; mount weights/data as needed.
ENTRYPOINT ["glyphloop"]
CMD ["--help"]
