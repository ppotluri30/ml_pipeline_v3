# syntax=docker/dockerfile:1

ARG PYTHON_VERSION=3.11.4
FROM python:${PYTHON_VERSION}-slim AS base
RUN apt-get update && apt-get install -y --no-install-recommends git && rm -rf /var/lib/apt/lists/*
ENV GIT_PYTHON_REFRESH=quiet

# Prevents Python from writing pyc files.
ENV PYTHONDONTWRITEBYTECODE=1

# Keeps Python from buffering stdout and stderr to avoid situations where
# the application crashes without emitting any logs due to buffering.
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Create a non-privileged user that the app will run under.
# See https://docs.docker.com/go/dockerfile-user-best-practices/
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
    --mount=type=bind,source=inference_container/requirements.txt,target=requirements.txt \
    python -m pip install -r requirements.txt

# Switch to the non-privileged user to run the application.
USER appuser

COPY inference_container/ .
COPY shared/ /app/shared/

EXPOSE 8000

# Default, env-driven Uvicorn HTTP server settings (safe to override in compose)
ENV UVICORN_WORKERS=1 \
    UVICORN_TIMEOUT_KEEP_ALIVE=60 \
    UVICORN_TIMEOUT_GRACEFUL_SHUTDOWN=30 \
    UVICORN_LIMIT_CONCURRENCY=1000 \
    INFERENCE_START_IN_APP=1

# UNIFIED SINGLE-PROCESS ARCHITECTURE:
# Run only uvicorn; Kafka consumer starts within FastAPI via INFERENCE_START_IN_APP=1
# This ensures main.py runtime and api_server.py share the same memory space and service.df
CMD ["sh", "-c", "exec uvicorn api_server:app --host 0.0.0.0 --port 8000 --log-level info --access-log --workers ${UVICORN_WORKERS:-1} --limit-concurrency ${UVICORN_LIMIT_CONCURRENCY:-1000} --timeout-keep-alive ${UVICORN_TIMEOUT_KEEP_ALIVE:-60} --timeout-graceful-shutdown ${UVICORN_TIMEOUT_GRACEFUL_SHUTDOWN:-30}"]
