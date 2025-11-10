# syntax=docker/dockerfile:1

ARG PYTHON_VERSION=3.11.4
FROM python:${PYTHON_VERSION}-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV GIT_PYTHON_REFRESH=quiet

# --- NEW: Install gosu for privilege dropping ---
# We run as root to install packages
# --- Install gosu and git (git needed for MLflow to capture commit; quiet warning using env var) ---
RUN apt-get update && apt-get install -y --no-install-recommends gosu git && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Create a non-privileged user.
ARG UID=10001   
RUN adduser \
    --disabled-password \
    --gecos "" \
    --home "/home/appuser" \
    --shell "/bin/bash" \
    --uid "${UID}" \
    appuser

# Ensure dir exists and set permissions.
RUN mkdir -p /app/outputs/models && chown -R appuser:appuser /app/outputs

# Install dependencies.
RUN --mount=type=cache,target=/root/.cache/pip \
    --mount=type=bind,source=train_container/requirements.txt,target=requirements.txt \
    python -m pip install -r requirements.txt

# Copy the application code (cache bust with CODE_VERSION build arg)
ARG CODE_VERSION
ENV CODE_VERSION=${CODE_VERSION}
COPY train_container/ .
COPY shared/ /app/shared/
# Assert updated main.py contains fallback logic; fail build otherwise
RUN grep -q "target_fallback" main.py || (echo "ERROR: target_fallback not found in main.py; stale build context" && exit 1)
RUN echo "Building train_container with CODE_VERSION=${CODE_VERSION}" && grep -n "target_fallback" main.py || true

# Pre-create the cache directory and set ownership.
RUN chown -R appuser:appuser /app

# Expose the port.
EXPOSE 5005

CMD ["python", "main.py"]