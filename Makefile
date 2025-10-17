
.PHONY: fmt test bench run

fmt:
	ruff check --fix . || true
	python -m pip install -e .

test:
	pytest -q

run:
	uvicorn lora_serve.app:app --host 0.0.0.0 --port 8000

bench:
	bash scripts/bench_local.sh
