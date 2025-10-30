#!/bin/sh
# Robust MinIO init script – creates required buckets with diagnostics.
set -e

log() { printf '[minio-init] %s\n' "$*"; }
warn() { printf '[minio-init][WARN] %s\n' "$*" >&2; }
err()  { printf '[minio-init][ERROR] %s\n' "$*" >&2; }

log "Starting bucket ensure script..."

# Detect and mitigate potential CRLF (Windows) line ending issues (heuristic)
if grep -q "\r" "$0" 2>/dev/null; then
  warn "Script appears to have CRLF line endings; converting in-memory may avoid subtle issues."
fi

MC_ENDPOINT=${MC_ENDPOINT:-http://minio:9000}
MC_USER=${MC_USER:-minioadmin}
MC_PASS=${MC_PASS:-minioadmin}
RETRIES=${RETRIES:-60}        # Increased from 30 for slower cold boots
SLEEP_SEC=${SLEEP_SEC:-2} 

# Optional network diagnostics (DEBUG_MINIO_INIT=1 to enable verbose)
if [ "${DEBUG_MINIO_INIT}" = "1" ]; then
  log "Debug mode enabled: endpoint=$MC_ENDPOINT"
  ping -c1 minio 2>/dev/null || warn "ping to minio failed (not fatal)"
fi

i=0
last_err=""
while ! out=$(mc alias set minio "$MC_ENDPOINT" "$MC_USER" "$MC_PASS" 2>&1); do
  last_err=$out
  i=$((i+1))
  if [ "$i" -ge "$RETRIES" ]; then
    err "Unable to set alias after $i attempts (endpoint=$MC_ENDPOINT). Last attempt failed."
    if [ -n "$last_err" ]; then
      err "Last mc alias error: $last_err"
    fi
    # Provide extra diagnostics
    if command -v curl >/dev/null 2>&1; then
      warn "curl HEAD:"; curl -I --max-time 5 "$MC_ENDPOINT" || true
    fi
    exit 2
  fi
  log "MinIO not ready yet, retry $i/$RETRIES..."
  sleep "$SLEEP_SEC"
done

log "Alias configured. Ensuring buckets..."
# Added 'inference-logs' (active bucket) and retained legacy 'inference-txt-logs'
BUCKETS="mlflow dataset processed-data inference-logs inference-txt-logs model-promotion"

for b in $BUCKETS; do
  if mc ls "minio/$b" >/dev/null 2>&1; then
    log "Bucket $b exists"
  else
    if mc mb -p "minio/$b" >/dev/null 2>&1; then
      log "Created bucket $b"
    else
      warn "Could not create bucket $b (may already exist or race)"
    fi
  fi
done

log "Current buckets listing:"
mc ls minio || true
log "Buckets ensured. ✅"
