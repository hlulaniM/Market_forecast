# Intelligent Trading Forecast Framework (ITFF)

This repository contains the implementation assets for the Intelligent Trading Forecast Framework, focused on probabilistic intraday forecasting for BTC/USD and ETH/USD.

## Project Layout

- data/ raw, processed, engineered features, and sequence datasets
- models/ saved model weights, scalers, metrics, and checkpoints (LSTM + Transformer)
- notebooks/ exploratory data analysis and experiment notebooks
- scripts/ data pipelines, feature engineering, and model training utilities
- api/ FastAPI prediction service for deployment
- dashboard/ assets for monitoring dashboards
- tradingview/ webhook templates and integration notes
- proposal/ documentation, literature survey, and project reports

## Phase 1 Checklist

1. Create and activate a Python 3.11 virtual environment.
2. Install project dependencies from `requirements.txt`.
3. Copy `.env.example` to `.env` and populate API credentials (Alpaca/Binance).
4. Run `python scripts/fetch_market_data.py --symbol BTCUSDT` to validate data ingestion.

## Modeling Quickstart

- Optional sentiment enrichment: `python scripts/fetch_sentiment.py --symbol BTCUSDT --limit 50 --output data/sentiment`
- Engineer features with sentiment: `python scripts/engineer_features.py data/processed/btcusdt_..._clean.parquet --symbol BTCUSDT --sentiment-file data/sentiment/btcusdt_sentiment_*.parquet`
- Prepare sequences: `python scripts/prepare_sequences.py --symbol BTCUSDT --features data/features/btcusdt_features.parquet --save-metadata --scaler models/scalers/btcusdt_feature_scaler.pkl`
- Train baseline LSTM: `python scripts/train_lstm.py --symbol BTCUSDT --label direction`
- Train Transformer + MC Dropout: `python scripts/train_transformer.py --symbol BTCUSDT --label direction --mc-samples 30`
- Tackle imbalance on level targets: `python scripts/train_lstm.py --symbol BTCUSDT --label level_up --class-weight balanced --threshold 0.4`
- Generate evaluation plots: `python scripts/evaluate_models.py --symbol BTCUSDT --label direction --model-type transformer --mc-samples 30 --output reports`
- Metrics (including Brier score and uncertainty statistics) are stored in `models/training/*_metrics.json`.
- Threshold sweeps and calibration CSVs are saved to `reports/` (see `scripts/threshold_sweep.py`).

## Pipeline Runner

Automate the full data → feature → model refresh with:

```
python scripts/run_pipeline.py --symbol BTCUSDT --days 60 --include-sentiment --train-direction --train-level --model-type transformer --class-weight balanced --threshold 0.4
```

This sequentially pulls OHLCV candles, scores sentiment, engineers features, rebuilds datasets, and retrains the requested models.

Refer to `docs/operations.md` for scheduling, monitoring, and calibration playbooks.

## Phase 5 Quickstart (API + TradingView)

1. Ensure checkpoints exist in `models/training/` (e.g. `btcusdt_direction_best.keras`, `btcusdt_direction_transformer_best.keras`).
2. Launch the prediction service locally:
   ```
   uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
   ```
3. Submit a prediction request:
   ```python
   import json, numpy as np, requests
   seq = np.load('data/datasets/btcusdt_test_X.npy')[-1].tolist()
   payload = {
       "symbol": "BTCUSDT",
       "target": "direction",
       "model_type": "transformer",
       "mc_samples": 30,
       "sequence": seq
   }
   resp = requests.post('http://localhost:8000/predict', json=payload, timeout=10)
   print(resp.json())
   ```
4. Configure a TradingView alert to POST its payload to `/predict`; see `tradingview/README.md` for template guidance.

## Dashboard

Launch the Plotly Dash app to monitor accuracy, ROC/PR/calibration plots, and threshold sweeps:
```
python dashboard/app.py
```
The dashboard reads evaluation JSON/CSV artifacts in `reports/` and updates automatically whenever you rerun the evaluation scripts.

## Next Steps

- Populate the `data/` directory using the data collection pipeline.
- Extend feature engineering, modeling, and deployment assets in later phases (e.g., sentiment ingestion, calibration dashboards, live latency monitoring).

## Containerization Plan

1. **FastAPI prediction service**
   - Build a slim Python base image (`python:3.11-slim`) with system deps for TensorFlow.
   - Copy only `api/`, `scripts/`, `requirements.txt`, and shared libs into `/app`.
   - Install dependencies with `pip install --no-cache-dir -r requirements.txt`.
   - Expose `8000`, set `CMD` to `uvicorn api.main:app --host 0.0.0.0 --port 8000 --workers 2`.
   - Mount or download model artifacts at runtime (S3/Blob/Drive) to keep the image lean.

2. **Plotly dashboard**
   - Use the same base and dependency layer to avoid duplication.
   - Copy `dashboard/` + shared requirements (reuse the primary `requirements.txt`).
   - Expose `8050`, run with `python dashboard/app.py`.
   - Share data artifacts via mounted volume/remote storage, not baked into the image.

3. **Shared build strategy**
   - Multi-stage Dockerfile or two Dockerfiles with a common `requirements` layer.
   - `.dockerignore` to omit `data/`, `models/`, notebook outputs, caches.
   - Environment variables for secrets/thresholds; reference them in the app configs.
   - Health endpoints: FastAPI already serves `/health`; add a simple `/health` route to Dash or configure `Dash` to respond via Gunicorn worker.

4. **Deployment targets (free tier friendly)**
   - Render / Railway web services (one per container) with persistent volume for models, or download artifacts on boot.
   - Alternative: Google Cloud Run free tier with artifacts in GCS and secrets via Secret Manager.

5. **Next actions**
   - Draft `.dockerignore`.
   - Create `Dockerfile.api`.
   - Create `Dockerfile.dashboard`.
   - Add deployment instructions and Render/Railway manifests once images are verified locally.

## Docker Quickstart

```bash
# Build images (from repository root)
docker build -f ITFF_Project/Dockerfile.api -t itff-api ITFF_Project
docker build -f ITFF_Project/Dockerfile.dashboard -t itff-dashboard ITFF_Project

# Run API with mounted artifacts and secrets
docker run --rm -p 8000:8000 ^
  -e ITFF_API_TOKEN=replace-me ^
  -e ITFF_API_RATE_LIMIT=60/minute ^
  -v %cd%/ITFF_Project/models:/app/models ^
  -v %cd%/ITFF_Project/data:/app/data ^
  itff-api

# Run dashboard (optional volume for reports)
docker run --rm -p 8050:8050 ^
  -e ITFF_DASH_USERNAME=demo ^
  -e ITFF_DASH_PASSWORD=demo ^
  -v %cd%/ITFF_Project/reports:/app/reports ^
  itff-dashboard
```

The containers expect the same folder structure as the repository. In production, mount cloud storage or download artifacts on startup instead of baking models into the image.

### Runtime Configuration

- `ITFF_API_TOKEN`: shared secret required in the `X-API-Token` request header. Leave unset to disable auth (not recommended in production).
- `ITFF_API_RATE_LIMIT`: SlowAPI-compatible string (e.g. `60/minute`, `500/hour`). Omit to disable rate limiting.
- `ITFF_MODELS_DIR`, `ITFF_DATA_DIR`: optional overrides used by startup scripts / volume mounts.
- `ITFF_DASH_USERNAME`, `ITFF_DASH_PASSWORD`: enable HTTP Basic Auth on the dashboard when both are provided.

Clients must include the header `X-API-Token: <value>` when calling `/predict`.

### Artifact Synchronisation

Maintain a manifest of required model/scaler files under `deploy/artifacts.manifest.json`. Each entry contains a `source` (remote path or full URL), `destination` (relative path inside the container), and optional `sha256` checksum. Download artifacts with:

```bash
python scripts/sync_artifacts.py --manifest deploy/artifacts.manifest.json ^
  --base-url https://your-storage-bucket.s3.amazonaws.com ^
  --root ITFF_Project --overwrite
```

Set `ITFF_ARTIFACT_BASE_URL` to avoid passing `--base-url` each time. The script skips existing files unless `--overwrite` is provided.

#### Cloudflare R2 Quickstart

1. Create an R2 bucket (e.g., `itff-artifacts`) from the Cloudflare dashboard.
2. Enable public access under **Settings → Public Access** and note the bucket domain (shown under the “Public Bucket URL” heading, e.g. `https://<accountid>.r2.cloudflarestorage.com/itff-artifacts`).
3. Upload model weights and scalers via the dashboard or `rclone`.
4. Run `python scripts/hash_artifacts.py` to generate `deploy/artifact_hashes.json`, then fill in `sha256` values in `deploy/artifacts.manifest.json`.
5. Update the manifest’s `base_url` with the bucket domain and deploy; set `ITFF_ARTIFACT_BASE_URL` to the same URL in Render/Railway.

## CI & Testing

- Install development dependencies: `pip install -r ITFF_Project/requirements-dev.txt`.
- Lint: `flake8 ITFF_Project/api ITFF_Project/dashboard ITFF_Project/scripts ITFF_Project/tests`.
- Tests: `pytest ITFF_Project/tests`.
- GitHub Actions workflow (`.github/workflows/ci.yml`) runs lint + tests on push/pull requests targeting `main`.

## Render Free-Tier Deployment

1. **Repository setup**
   - Push the `ITFF_Project` folder with Dockerfiles to GitHub (done).
   - Ensure large artifacts (models/data) live in cloud storage or Render persistent disks.

2. **API service**
   - Create a new Web Service in Render, select the repo, and set the root to `ITFF_Project`.
   - Runtime: Docker.
   - Docker build command: default (`docker build` picks `Dockerfile.api` once specified).
   - Start command: `uvicorn api.main:app --host 0.0.0.0 --port 8000 --workers 2`.
   - Environment variables:
     - `ITFF_API_TOKEN` (shared secret for webhook auth, to be enforced in code).
     - `ITFF_API_RATE_LIMIT=60/minute` (or custom limit).
     - `ITFF_MODELS_DIR=/var/data/models` (if using persistent disk).
     - `ITFF_DATA_DIR=/var/data/datasets`.
   - Add a persistent disk (minimum 1 GB) mounted at `/var/data` and populate it with model/scaler artifacts on first deploy.

3. **Dashboard service**
   - Create a second Web Service using `Dockerfile.dashboard`.
   - Start command: `gunicorn --bind 0.0.0.0:8050 dashboard.app:server`.
   - Environment variables: `ITFF_DASH_USERNAME`, `ITFF_DASH_PASSWORD` for HTTP Basic Auth.
   - Mount the same persistent disk (read-only) at `/var/data` if the dashboard needs datasets/reports.
   - Configure IP allowlists or VPN access if additional locking is required.

4. **Secrets management**
   - Store API tokens, webhook secrets, and storage credentials in Render’s secret manager.
   - Rotate keys periodically; never commit them to Git.

5. **Post-deploy checks**
   - Hit `/health` and `/metrics` on the API service.
   - Load the dashboard and confirm curves load using mounted reports.
   - Enable auto-deploy on main branch; rely on GitHub Actions CI to keep builds green before Render deploys.
