
# LoRAServe

Async multi-tenant LLM inference with dynamic LoRA loading, KV-cache reuse, and speculative decoding.

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

