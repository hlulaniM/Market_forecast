import argparse
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import pandas as pd
from transformers import AutoModelForSequenceClassification, AutoTokenizer, TextClassificationPipeline
from yahoo_fin import news

MODEL_NAME = "ProsusAI/finbert"


SYMBOL_TO_TICKER: Dict[str, str] = {
    "BTCUSDT": "BTC-USD",
    "ETHUSDT": "ETH-USD",
}


def fetch_headlines(ticker: str, limit: int) -> List[Dict[str, str]]:
    items = news.get_yf_rss(ticker) or []
    subset = items[:limit] if limit else items
    return subset


def build_pipeline() -> TextClassificationPipeline:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
    return TextClassificationPipeline(model=model, tokenizer=tokenizer, return_all_scores=True, truncation=True)


def score_headlines(headlines: List[Dict[str, str]], pipe: TextClassificationPipeline) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for item in headlines:
        title = item.get("title") or ""
        summary = item.get("summary") or ""
        text = title if title else summary
        if not text:
            continue

        scores = pipe(text)[0]
        score_map = {entry["label"].lower(): entry["score"] for entry in scores}
        sentiment_score = score_map.get("positive", 0.0) - score_map.get("negative", 0.0)
        top_label = max(scores, key=lambda x: x["score"]) if scores else {"label": "neutral", "score": 0.0}

        published = item.get("published") or item.get("pubDate")
        if published:
            timestamp = pd.to_datetime(published, utc=True, errors="coerce")
        else:
            timestamp = None
        if timestamp is None or pd.isna(timestamp):
            timestamp = pd.Timestamp.utcnow().tz_localize("UTC")

        rows.append(
            {
                "timestamp": timestamp,
                "headline": title,
                "summary": summary,
                "sentiment_label": top_label["label"].lower(),
                "sentiment_confidence": float(top_label["score"]),
                "sentiment_score": float(sentiment_score),
                "source": item.get("source", "yahoo_fin"),
                "link": item.get("link"),
            }
        )

    frame = pd.DataFrame(rows).drop_duplicates(subset=["headline", "timestamp"])
    frame = frame.sort_values("timestamp").reset_index(drop=True)
    return frame


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch news headlines and compute FinBERT sentiment scores.")
    parser.add_argument("--symbol", required=True, help="Internal symbol identifier, e.g. BTCUSDT")
    parser.add_argument("--ticker", help="Yahoo Finance ticker symbol, e.g. BTC-USD")
    parser.add_argument("--limit", type=int, default=100, help="Maximum number of headlines to fetch")
    parser.add_argument("--output", default="data/sentiment", help="Directory for sentiment parquet output")
    args = parser.parse_args()

    ticker = args.ticker or SYMBOL_TO_TICKER.get(args.symbol.upper())
    if not ticker:
        raise ValueError("Ticker not provided and not found in mapping.")

    headlines = fetch_headlines(ticker, args.limit)
    if not headlines:
        raise RuntimeError(f"No headlines retrieved for ticker {ticker}")

    pipe = build_pipeline()
    sentiment_df = score_headlines(headlines, pipe)

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    output_path = output_dir / f"{args.symbol.lower()}_sentiment_{timestamp}.parquet"
    sentiment_df.to_parquet(output_path, index=False)

    print(f"Saved {len(sentiment_df)} sentiment rows to {output_path}")


if __name__ == "__main__":
    main()
