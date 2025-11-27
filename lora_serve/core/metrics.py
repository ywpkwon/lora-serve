# lora_serve/core/metrics.py
from prometheus_client import Counter, Histogram

# ---- Request-level metrics ----

REQUESTS_TOTAL = Counter(
    "lora_serve_requests_total",
    "Total number of generation requests",
    ["endpoint", "adapter_id"],
)

REQUEST_LATENCY_MS = Histogram(
    "lora_serve_request_latency_ms",
    "End-to-end latency per request (ms)",
    ["endpoint"],
    buckets=(10, 25, 50, 100, 200, 400, 800, 1600, 3200),
)

# If you want streaming TTFT later:
TTFT_MS = Histogram(
    "lora_serve_ttft_ms",
    "Time-to-first-token for streaming requests (ms)",
)

# ---- Batching / scheduler metrics ----

BATCH_SIZE = Histogram(
    "lora_serve_batch_size",
    "Number of requests per batch",
    buckets=(1, 2, 4, 8, 16, 32, 64),
)

QUEUE_WAIT_MS = Histogram(
    "lora_serve_queue_wait_ms",
    "Time a request spent in the queue before being batched (ms)",
)

# ---- Token throughput ----

TOKENS_GENERATED = Counter(
    "lora_serve_tokens_generated_total",
    "Total number of tokens generated",
    ["endpoint"],
)

