#!/bin/sh
set -eu

MODE="${LOCUST_MODE:-ui}"
USERS_VAL="${USERS:-20}"
SPAWN_VAL="${SPAWN_RATE:-5}"
RUNTIME_VAL="${RUNTIME:-2m}"

if [ "$MODE" = "headless" ]; then
  echo "[entrypoint] Starting Locust in headless mode: -u ${USERS_VAL} -r ${SPAWN_VAL} --run-time ${RUNTIME_VAL}" >&2
  exec locust -f /mnt/locust/locustfile.py --headless -u "${USERS_VAL}" -r "${SPAWN_VAL}" --run-time "${RUNTIME_VAL}" --host http://inference:8000
else
  echo "[entrypoint] Starting Locust Web UI on 0.0.0.0:8089" >&2
  exec locust -f /mnt/locust/locustfile.py --host http://inference:8000 --web-host 0.0.0.0
fi
