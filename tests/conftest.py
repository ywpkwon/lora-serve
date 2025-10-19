import pytest
import asyncio
from lora_serve.core.engines.hf_engine import HFEngine

@pytest.fixture(scope="session")
def hf_engine():
    """Load the model once per test session."""
    engine = HFEngine("TinyLlama/TinyLlama-1.1B-Chat-v1.0", dtype="bfloat16")
    return engine

@pytest.fixture(scope="session")
def event_loop():
    """Override pytest-asyncio's loop fixture so we can reuse one loop."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()
