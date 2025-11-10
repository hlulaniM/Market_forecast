# Operational Playbook

## Daily Refresh

Run the orchestrated pipeline for both assets once per day:
```
# BTC
python scripts/run_pipeline.py --symbol BTCUSDT --days 60 --include-sentiment --train-direction --train-level --model-type transformer --class-weight balanced --threshold 0.4

# ETH
python scripts/run_pipeline.py --symbol ETHUSDT --days 60 --include-sentiment --train-direction --train-level --model-type transformer --class-weight balanced --threshold 0.4
```

## Windows Task Scheduler (example)

1. Open **Task Scheduler** â†’ **Create Basic Task**.
2. Trigger: **Daily** (choose off-market hour).
3. Action: **Start a Program**.
   - Program/Script: `powershell.exe`
   - Add arguments:
     ```
     -File "C:\\Users\\HlulaniMabunda\\Market_forecast\\ITFF_Project\\scripts\\run_pipeline.ps1"
     ```
4. Create the helper PowerShell script (`run_pipeline.ps1`) containing:
   ```powershell
   $env:Path = "C:\\Users\\HlulaniMabunda\\Market_forecast\\ITFF_Project\\.venv\\Scripts;$env:Path"
   Set-Location "C:\\Users\\HlulaniMabunda\\Market_forecast\\ITFF_Project"
   python scripts/run_pipeline.py --symbol BTCUSDT --days 60 --include-sentiment --train-direction --train-level --model-type transformer --class-weight balanced --threshold 0.4
   python scripts/run_pipeline.py --symbol ETHUSDT --days 60 --include-sentiment --train-direction --train-level --model-type transformer --class-weight balanced --threshold 0.4
   ```

## API Monitoring

- `/metrics` exposes Prometheus-compatible counters and histograms (`itff_api_requests_total`, `itff_api_request_latency_seconds`).
- Add scrape job:
  ```yaml
  - job_name: itff_api
    metrics_path: /metrics
    static_configs:
      - targets: ['localhost:8000']
  ```

## Calibration Review

- Generate plots after each training cycle:
  ```
  python scripts/evaluate_models.py --symbol BTCUSDT --label direction --model-type transformer --mc-samples 30 --output reports
  python scripts/threshold_sweep.py --symbol BTCUSDT --label level_up --model-type lstm --steps 21 --output reports
  ```
- Review outputs saved in `reports/` before promoting a checkpoint to production.

## 30-Day TradingView Validation Checklist

1. **Pre-launch**
   - Deploy FastAPI (`uvicorn api.main:app --host 0.0.0.0 --port 8000`).
   - Start the dashboard (`python dashboard/app.py`) for daily monitoring.
   - Confirm `/health` and `/metrics` endpoints respond.
2. **TradingView Setup**
   - Import `tradingview/pine_template.pine` and map the feature buffer to your indicators.
   - Create alerts for BTCUSDTPERP and ETHUSDTPERP with webhook payload pointing to `/predict`.
   - Set alert frequency to `Once Per Bar Close` to match the 1-minute cadence.
3. **Daily Operations**
   - Each morning verify the pipeline completed; re-run `run_pipeline.py` if missed.
   - Capture API latency stats from Prometheus; flag if p95 exceeds 0.2s.
   - Export dashboard screenshots and store trading logs.
4. **Weekly Review**
   - Run evaluation + threshold scripts; adjust alert threshold if precision/recall drift >5%.
   - Update trading rules in TradingView if new threshold chosen.
5. **Post 30 Days**
   - Aggregate profit factor, MAE, directional accuracy, and latency.
   - Archive Prometheus metrics and dashboard visuals for reporting.
   - Retrain models with the 30-day window appended and re-run validation before production deployment.
