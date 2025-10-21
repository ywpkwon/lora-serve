
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



