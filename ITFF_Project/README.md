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
