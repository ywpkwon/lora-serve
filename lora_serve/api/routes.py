
import asyncio
import logging
from fastapi import APIRouter, Depends, HTTPException
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
_engine = HFEngine(model_id=settings.model_id, dtype=settings.dtype, device=settings.device)
_queues = TenantQueues()
_batcher = DynamicBatcher(_engine, _queues, _adapters, settings.max_batch_tokens, settings.max_wait_ms)
asyncio.get_event_loop().create_task(_batcher.run_forever())

@api_router.post("/generate", response_model=GenerateOut)
async def generate(body: GenerateIn):
    logger.debug("Received /generate request: %s", body.dict())

    # validate adapter up front
    if body.adapter_id:
        try:
            path = await _adapters.ensure_loaded(body.adapter_id)  # may raise FileNotFoundError
        except FileNotFoundError:
            # 404 reads better than 400: it's a missing resource (adapter)
            raise HTTPException(status_code=404, detail=f"Adapter '{body.adapter_id}' not found")
        # optional: pre-attach here, or let batcher do it; either is fine
        await _engine.attach_adapter(body.adapter_id, str(path))

    req = GenerateRequest(**body.model_dump())
    res = await _batcher.enqueue(req)
    return GenerateOut(text=res.text, tokens=res.tokens)
