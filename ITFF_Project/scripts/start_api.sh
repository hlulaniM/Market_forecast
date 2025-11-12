#!/bin/bash

echo "=========================================="
echo "Starting ITFF API service..."
echo "=========================================="
echo "Current directory: $(pwd)"
echo "ITFF_ARTIFACT_BASE_URL: ${ITFF_ARTIFACT_BASE_URL:-NOT SET}"

# Sync artifacts if base URL is set
if [ -n "$ITFF_ARTIFACT_BASE_URL" ]; then
    echo "Syncing artifacts from $ITFF_ARTIFACT_BASE_URL..."
    if python scripts/sync_artifacts.py \
        --manifest deploy/artifacts.manifest.json \
        --base-url "$ITFF_ARTIFACT_BASE_URL" \
        --root /app \
        --overwrite; then
        echo "✓ Artifact sync completed successfully"
        echo "Listing downloaded files:"
        ls -lh /app/models/ 2>/dev/null || echo "No models directory"
        ls -lh /app/data/ 2>/dev/null || echo "No data directory"
    else
        echo "✗ WARNING: Artifact sync failed, continuing anyway..."
    fi
else
    echo "⚠ WARNING: ITFF_ARTIFACT_BASE_URL not set, skipping artifact sync"
fi

echo "Starting uvicorn server..."
exec uvicorn api.main:app --host 0.0.0.0 --port 8000 --workers 2

