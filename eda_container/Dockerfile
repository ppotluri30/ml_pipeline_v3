# syntax=docker/dockerfile:1

ARG PYTHON_VERSION=3.11.4
FROM python:${PYTHON_VERSION}-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Create a non-privileged user.
ARG UID=10001
RUN adduser \
    --disabled-password \
    --gecos "" \
    --home "/nonexistent" \
    --shell "/sbin/nologin" \
    --no-create-home \
    --uid "${UID}" \
    appuser

# Install dependencies.
RUN --mount=type=cache,target=/root/.cache/pip \
    --mount=type=bind,source=requirements.txt,target=requirements.txt \
    python -m pip install -r requirements.txt

# Pre-create the cache directory and set ownership.
RUN mkdir -p /app/.cache/matplotlib && \
    chown -R appuser:appuser /app/.cache

# Give matplotlib and dcor cache directories they can write to
ENV MPLCONFIGDIR=/app/.cache/matplotlib
ENV NUMBA_CACHE_DIR=/app/.cache/dcor

USER appuser

COPY . .

# Expose the port.
EXPOSE 8010

# This command is passed to the entrypoint script.
CMD ["uvicorn", "main:app", "--host=0.0.0.0", "--port=8010"]