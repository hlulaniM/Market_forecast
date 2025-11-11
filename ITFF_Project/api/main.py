import logging
import time
from typing import Any, Dict, List

from fastapi import Depends, FastAPI, HTTPException, Request, Response
from fastapi.responses import PlainTextResponse
from prometheus_client import CollectorRegistry, Counter, Histogram, generate_latest
from slowapi import Limiter
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware
from slowapi.util import get_remote_address

from api.config import get_settings
from api.dependencies import enforce_api_token
from api.models.registry import registry
from api.routers.predict import router as predict_router

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
logger = logging.getLogger("itff.api")

settings = get_settings()

rate_limits: List[str] = []
if settings.api_rate_limit:
    rate_limits.append(settings.api_rate_limit)

if rate_limits:
    limiter = Limiter(key_func=get_remote_address, default_limits=rate_limits)
else:
    limiter = Limiter(key_func=get_remote_address)

METRICS_REGISTRY = CollectorRegistry()

REQUEST_COUNT = Counter(
    "itff_api_requests_total",
    "Total number of API requests",
    ["method", "endpoint", "status_code"],
    registry=METRICS_REGISTRY,
)
REQUEST_LATENCY = Histogram(
    "itff_api_request_latency_seconds",
    "Latency of API requests in seconds",
    ["method", "endpoint"],
    registry=METRICS_REGISTRY,
)

app = FastAPI(title="ITFF Prediction Service", version="0.3.0")
app.state.limiter = limiter
app.add_middleware(SlowAPIMiddleware)


@app.exception_handler(RateLimitExceeded)
async def rate_limit_handler(request: Request, exc: RateLimitExceeded) -> PlainTextResponse:
    return PlainTextResponse("Too Many Requests", status_code=429)


@app.middleware("http")
async def metrics_middleware(request: Request, call_next):
    start_time = time.perf_counter()
    response: Response = await call_next(request)
    latency = time.perf_counter() - start_time

    endpoint = request.url.path
    method = request.method
    status_code = response.status_code

    REQUEST_COUNT.labels(method=method, endpoint=endpoint, status_code=status_code).inc()
    REQUEST_LATENCY.labels(method=method, endpoint=endpoint).observe(latency)

    response.headers["X-Process-Time"] = f"{latency:.4f}"
    logger.info("%s %s %s %.4fs", method, endpoint, status_code, latency)
    return response


app.include_router(predict_router, dependencies=[Depends(enforce_api_token)])


@app.get("/health")
def healthcheck() -> Dict[str, str]:
    return {"status": "ok"}


@app.get("/metadata/{symbol}")
def get_metadata(symbol: str, target: str = "direction", model_type: str = "lstm") -> Dict[str, Any]:
    bundle = registry.load(symbol, target, model_type)
    metadata = bundle.get("metadata", {})
    if not metadata:
        raise HTTPException(status_code=404, detail="Metadata not available for requested symbol")
    return metadata


@app.get("/metrics")
def get_metrics() -> Response:
    return PlainTextResponse(generate_latest(METRICS_REGISTRY), media_type="text/plain")


def get_uvicorn_kwargs(host: str = "0.0.0.0", port: int = 8000) -> Dict[str, Any]:
    return {"app": "api.main:app", "host": host, "port": port, "reload": True}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(**get_uvicorn_kwargs())
