from typing import List, Optional

import numpy as np
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from api.models.registry import registry


router = APIRouter(prefix="/predict", tags=["prediction"])


class PredictRequest(BaseModel):
    symbol: str = Field(..., description="Trading symbol, e.g. BTCUSDT")
    target: str = Field("direction", description="Prediction target: direction or level_up")
    sequence: List[List[float]] = Field(..., description="Most recent sequence window in model feature order")
    threshold: Optional[float] = Field(0.5, description="Decision threshold applied to predicted probability")
    model_type: str = Field("lstm", description="Model variant to use: lstm or transformer")
    mc_samples: Optional[int] = Field(1, ge=1, le=200, description="Number of MC Dropout samples (transformer only)")


class PredictResponse(BaseModel):
    symbol: str
    target: str
    model_type: str
    probability_mean: float
    probability_std: float
    decision: int
    samples: Optional[List[float]] = Field(None, description="Raw probability samples when mc_samples > 1")


@router.post("", response_model=PredictResponse)
async def run_prediction(payload: PredictRequest) -> PredictResponse:
    bundle = registry.load(payload.symbol, payload.target, payload.model_type)
    model = bundle["model"]
    metadata = bundle.get("metadata", {})
    scaler = bundle.get("scaler")
    expected_len = metadata.get("sequence_length")
    expected_features = metadata.get("feature_columns")

    sequence = np.array(payload.sequence, dtype=np.float32)
    if sequence.ndim != 2:
        raise HTTPException(status_code=400, detail="Sequence must be 2D: [timesteps, features]")

    if expected_len is not None and sequence.shape[0] != expected_len:
        raise HTTPException(status_code=400, detail=f"Expected sequence length {expected_len}, received {sequence.shape[0]}")

    if expected_features is not None and sequence.shape[1] != len(expected_features):
        raise HTTPException(
            status_code=400,
            detail=f"Expected {len(expected_features)} features ({expected_features}), received {sequence.shape[1]}",
        )

    if scaler is not None:
        try:
            sequence = scaler.transform(sequence)
        except Exception as exc:  # pragma: no cover - safety net
            raise HTTPException(status_code=400, detail=f"Failed to scale sequence: {exc}") from exc

    sequence = np.expand_dims(sequence.astype(np.float32), axis=0)

    mc_samples = payload.mc_samples or 1
    if payload.model_type == "transformer" and mc_samples > 1:
        probs = []
        for _ in range(mc_samples):
            probs.append(float(model(sequence, training=True).numpy()[0][0]))
    else:
        probs = [float(model.predict(sequence, verbose=0)[0][0])]

    probability_mean = float(np.mean(probs))
    probability_std = float(np.std(probs))
    decision = int(probability_mean >= (payload.threshold or 0.5))

    return PredictResponse(
        symbol=payload.symbol.upper(),
        target=payload.target,
        model_type=payload.model_type,
        probability_mean=probability_mean,
        probability_std=probability_std,
        decision=decision,
        samples=probs if mc_samples > 1 else None,
    )
