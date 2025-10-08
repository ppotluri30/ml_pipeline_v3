#!/bin/sh
set -e
echo "[minio-init] Starting bucket ensure script..."

# Wait for MinIO to be reachable and alias to succeed
i=0
while ! mc alias set minio http://minio:9000 minioadmin minioadmin >/dev/null 2>&1; do
  i=$((i+1))
  if [ "$i" -ge 30 ]; then
    echo "[minio-init] ERROR: Unable to set alias after $i attempts" >&2
    exit 1
  fi
  echo "[minio-init] MinIO not ready yet, retry $i..."
  sleep 2
done

echo "[minio-init] Alias configured. Ensuring buckets..."
BUCKETS="mlflow dataset processed-data inference-txt-logs model-promotion"
for b in $BUCKETS; do
  if mc ls "minio/$b" >/dev/null 2>&1; then
    echo "[minio-init] Bucket $b exists"
  else
    if mc mb -p "minio/$b" >/dev/null 2>&1; then
      echo "[minio-init] Created bucket $b"
    else
      echo "[minio-init] WARN: Could not create bucket $b (may already exist or race condition)" >&2
    fi
  fi
done

echo "[minio-init] Current buckets:"
mc ls minio || true
echo "✅ Buckets ensured."
