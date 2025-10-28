
# LoRAServe

LoRAServe is a lightweight, educational project exploring how modern LLM inference systems (like vLLM and SGLang) work under the hood — from dynamic batching and adapter hot-loading to streaming responses and KV-cache reuse.

# 🚀 Overview

This repository is a side project to understand and re-implement key design patterns in large-model serving infrastructure.
It’s not a production service — the focus is on clarity, modularity, and learning-by-doing.

LoRAServe aims to demystify how real LLM engines manage:
- Dynamic batching and request queues
- LoRA adapter hot-loading and caching
- Token streaming (incremental responses)
- KV-cache lifecycle management
- Lightweight fairness and scheduling policies
- Speculative decoding and performance instrumentation

# Architecture at a Glance
```
/lora_serve
├── api/                     # REST API endpoints (FastAPI)
│   ├── routes.py            # /v1/generate, /v1/chat, /v1/generate/stream
│   └── schemas.py           # Request/response models (pydantic)
├── core/
│   ├── engines/             # HFEngine wrapper over transformers/PEFT
│   ├── adapters.py          # LoRAAdapterManager (load, cache, evict)
│   ├── config.py            # BaseSettings (env + .env overrides)
│   └── logging.py           # Thread/colorized logging helpers
├── scheduler/
│   ├── queue.py             # Tenant queues (per-tenant request queues)
│   ├── policies.py          # Batching/fairness strategies
│   └── batcher.py           # DynamicBatcher main loop
├── kv_cache/
│   └── manager.py           # Placeholder for KV-cache reuse & stats
└── tests/                   # pytest-based functional/unit tests
```

Each layer is intentionally minimal and commented so you can trace the full path:
```
/v1/generate → router → queue → DynamicBatcher → HFEngine → model.generate()
```

# ⚙️ Usage (Local)
```bash

# Install dependencies
pip install -r requirements.txt

# Configure environment (see .env.example)
cp .env.example .env

# Launch server
uvicorn lora_serve.api.app:app --reload

# Run a simple generation
curl -s -H "Content-Type: application/json" \
  -d '{"prompt":"Hello world","max_tokens":16}' \
  http://localhost:8000/v1/generate
```

Optional: run a streaming request (real-time token output)

```bash
curl -N -H "Content-Type: application/json" \
  -d '{"prompt":"Explain LoRA in one line","max_tokens":64,"stream":true}' \
  http://localhost:8000/v1/generate/stream
```
```

# 🧩 Example Goals for Learners

- See how to implement a vLLM-like batching loop from scratch.
- Experiment with LoRA hot-loading on a single GPU (Titan/A100).
- Explore asyncio patterns (async/await, queues, Futures).
- Add metrics (Prometheus or OpenTelemetry) and visualize throughput.
- Extend to multi-GPU or Kubernetes deployments later.

🧰 Dependencies

- FastAPI / Uvicorn — REST API layer
- Transformers — model/tokenizer backend
- PEFT — LoRA adapter handling
- torch — inference runtime
- sse-starlette — streaming (Server-Sent Events)
- pydantic — configuration and schema validation

# 🧾 License & Disclaimer

⚠️ Disclaimer
This project is an independent, educational exploration.
It is not affiliated with or endorsed by any organization and is not optimized for production workloads.
Use at your own discretion for research or learning purposes.

# 🙏 Acknowledgements

- vLLM: inspiration for batching and memory management
- SGLang: reference for adapter/runtime design
- PEFT: LoRA support
- Transformers: base modeling toolkit



## Quickstart

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
cp .env.example .env
uvicorn lora_serve.app:app --host 0.0.0.0 --port 8000 --reload
```

Then:
```bash
python examples/client_generate.py --prompt "Hello" --adapter demo-adapter
```
or
(continous siumul)
```bash
while True; do
  curl -s -H "Content-Type: application/json" \
    -d "{\"prompt\":\"Hello $i\",\"max_tokens\":16}" \
    http://localhost:8000/v1/generate >/dev/null && sleep 0.5
done
```

(batch test)
```
```bash
for i in {1..5}; do
  curl -s -H "Content-Type: application/json" \
    -d "{\"prompt\":\"Hello $i\",\"max_tokens\":16}" \
    http://localhost:8000/v1/generate >/dev/null &
done; wait
```



