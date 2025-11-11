# Production Readiness Plan

This document captures the enhancements required to move the Intelligent Trading Forecast Framework (ITFF) from a research prototype to a production-ready service.

## 1. Inference Correctness
- Apply saved scalers to incoming sequences before model inference.
- Validate sequence shape, feature order, and scaler presence per symbol.
- Add unit tests using stored sample sequences to ensure deterministic predictions.

## 2. API Hardening
- Separate settings module (Pydantic `BaseSettings`) for configuration and secrets. ✅
- Enforce request authentication (static token initially, pluggable JWT/OAuth later). ✅ (`X-API-Token`)
- Add rate limiting via `slowapi` or API gateway rules. ✅ (default `60/minute`, configurable via `ITFF_API_RATE_LIMIT`)
- Provide latency benchmark script and capture p95 metrics.
- Extend logging to JSON structure and ship to cloud log sink.

## 3. Containerization & Runtime
- Maintain `.dockerignore` to keep images lean.
- Produce `Dockerfile.api` (done) and `Dockerfile.dashboard`.
- Create shared base image or builder cache via multi-stage builds.
- Define `docker-compose.yml` for local multi-service runs with volume mounts for model artifacts.

## 4. Data & Model Artifacts
- Store models/scalers in remote object storage (S3/Blob) with signed download at container boot.
- Version artifacts by `symbol-target-model_type` and maintain checksum manifest.
- Automate daily pipeline job to refresh data, train, evaluate, and publish artifacts.
- Include data retention policy and backfill procedure.

## 5. Dashboard & Analytics
- Containerize Plotly Dash with health endpoint for Render/Railway. ✅
- Write ingestion job that pushes evaluation metrics to a lightweight SQLite/Parquet store the dashboard reads.
- Add authentication (basic auth or reverse proxy) before exposing publicly. ✅ (`dash_auth` with `ITFF_DASH_USERNAME/PASSWORD`)

## 6. Deployment Workflows
- GitHub Actions workflow:
  - Lint (ruff/flake8), type check (mypy), unit tests (pytest).
  - Build Docker images for API and dashboard.
  - Publish to container registry (GitHub Container Registry).
- Define IaC (Render `render.yaml` or Railway `railway.json`) for reproducible deployments.
- Provide fallback instructions for Google Cloud Run free tier.

## 7. Monitoring & Observability
- Expose `/metrics` (already in API) and add Dash health endpoint.
- Configure free Grafana Cloud or BetterStack for alerting on latency/error thresholds.
- Add synthetic check (cron hitting `/health`) and logging retention guidance.

## 8. Security & Compliance
- Document threat model (webhook exposure, credential handling).
- Use secrets manager or platform env vars; never bake keys into images.
- Enforce HTTPS (platform-provided) and optional request signing for TradingView webhooks.

## 9. Live Validation
- Prepare automated notebooks/scripts for 30-day live evaluation with profit factor computation.
- Store live run logs in `reports/live/` with daily summary.
- Provide rollback checklist to previous model version.

## 10. Documentation
- Update `README.md` with production quickstart, env vars, and deployment instructions.
- Add runbooks for incident response, retraining failures, and data anomalies.
- Maintain changelog and release process (semantic versioning).

