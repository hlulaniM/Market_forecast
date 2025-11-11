# Rollback Checklist

1. Identify the last known-good model version (`deploy/artifact_hashes.json` history).
2. Update `deploy/artifacts.manifest.json` to point to the stable artifact URLs/hashes.
3. Run `python scripts/sync_artifacts.py --manifest deploy/artifacts.manifest.json --overwrite`.
4. Redeploy the API container (`render.yaml` service redeploy).
5. Verify `/health`, `/metrics`, and sample `/predict` response.
6. Notify stakeholders and log the rollback in `reports/live/daily_notes.md`.

