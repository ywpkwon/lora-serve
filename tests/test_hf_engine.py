import asyncio
from lora_serve.core.types import GenerateRequest

def test_generate_batch_smoke(hf_engine):
    req = GenerateRequest(prompt="Hello from pytest!", max_tokens=8)
    res = asyncio.run(hf_engine.generate_batch([req]))
    assert isinstance(res[0].text, str)
    assert len(res[0].text.strip()) > 0

def test_multiple_prompts(hf_engine):
    reqs = [
        GenerateRequest(prompt="A short poem.", max_tokens=8),
        GenerateRequest(prompt="List two colors.", max_tokens=8),
    ]
    res = asyncio.run(hf_engine.generate_batch(reqs))
    assert len(res) == 2
    assert all(len(r.text.strip()) > 0 for r in res)
