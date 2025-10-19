
import asyncio
import logging
from fastapi import APIRouter, Depends
from ..core.config import settings
from ..core.types import GenerateRequest
from ..core.adapters import LoRAAdapterManager
from ..core.engines.hf_engine import HFEngine
from ..scheduler.queue import TenantQueues
from ..scheduler.batcher import DynamicBatcher
from .schemas import GenerateIn, GenerateOut

logger = logging.getLogger(__name__)
api_router = APIRouter()

# Global singletons for scaffold
_adapters = LoRAAdapterManager(base_dir=__import__("pathlib").Path(settings.adapter_root))
_engine = HFEngine()
_queues = TenantQueues()
_batcher = DynamicBatcher(_engine, _queues, settings.max_batch_tokens, settings.max_wait_ms)
asyncio.get_event_loop().create_task(_batcher.run_forever())

@api_router.post("/generate", response_model=GenerateOut)
async def generate(body: GenerateIn):
    logger.debug("Received /generate request: %s", body.dict())
    req = GenerateRequest(**body.model_dump())
    res = await _batcher.enqueue(req)
    return GenerateOut(text=res.text, tokens=res.tokens)
