
from prometheus_client import CollectorRegistry, CONTENT_TYPE_LATEST, generate_latest
from prometheus_client import Counter, Gauge, Histogram
from fastapi import FastAPI, Response

registry = CollectorRegistry()
requests_total = Counter("lora_requests_total", "Total requests", registry=registry)
tokens_total = Counter("lora_tokens_generated_total", "Total tokens", registry=registry)

metrics_app = FastAPI()

@metrics_app.get("/")
def metrics():
    data = generate_latest(registry)
    return Response(content=data, media_type=CONTENT_TYPE_LATEST)
