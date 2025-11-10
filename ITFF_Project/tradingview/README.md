# TradingView Integration Notes

1. **Add Webhook URL**: Set the TradingView alert to POST to your deployed FastAPI endpoint, e.g. `https://your-server/api/predict`.
2. **Alert Message Template**: Configure the alert body to send the most recent 60-step feature window (already scaled) that the model expects. Example snippet:
   ```
   {
     "symbol": "BTCUSDT",
     "target": "direction",
     "model_type": "transformer",
     "mc_samples": 20,
     "sequence": {{sequence_buffer|json}},
     "threshold": 0.55
   }
   ```
   Replace `sequence_buffer` with the name of your Pine Script variable that exports the feature matrix in the order defined by `data/datasets/<symbol>_metadata.json`.
3. **Decision Threshold**: Include an optional `threshold` field if you want TradingView to treat probabilities above a custom level as actionable.
4. **Webhook Handler**: The API returns `probability_mean`, `probability_std`, `decision`, and (optionally) raw `samples`. Use these to drive trade logic or dashboards.
5. **Pine Script Template**: See `pine_template.pine` for a ready-to-edit script that serialises the feature window and embeds the webhook payload.
6. **Latency Monitoring**: Log the response time within Pine Script or via your webhook relay to ensure sub-200 ms turnaround.

> **Tip**: Rotate MC sample counts (e.g., 10â€“30) to balance latency versus uncertainty accuracy.
