#!/bin/bash
set -e

echo "Starting ITFF API service..."

# Sync artifacts if base URL is set
if [ -n "$ITFF_ARTIFACT_BASE_URL" ]; then
    echo "Syncing artifacts from $ITFF_ARTIFACT_BASE_URL..."
    python scripts/sync_artifacts.py \
        --manifest deploy/artifacts.manifest.json \
        --base-url "$ITFF_ARTIFACT_BASE_URL" \
        --root /app \
        --overwrite || echo "Warning: Artifact sync failed, continuing anyway..."
else
    echo "Warning: ITFF_ARTIFACT_BASE_URL not set, skipping artifact sync"
fi

echo "Starting uvicorn server..."
exec uvicorn api.main:app --host 0.0.0.0 --port 8000 --workers 2

