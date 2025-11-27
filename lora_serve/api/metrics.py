# lora_serve/api/metrics.py
from fastapi import APIRouter, Response
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

metrics_router = APIRouter()

@metrics_router.get("/metrics")
async def metrics():
    # /metrics in Prometheus text format
    data = generate_latest()
    return Response(content=data, media_type=CONTENT_TYPE_LATEST)

