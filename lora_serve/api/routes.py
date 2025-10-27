
import asyncio
import logging
from sse_starlette.sse import EventSourceResponse
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


@api_router.post("/generate/stream")
async def generate_stream(body: GenerateIn):
    # 1) Validate quick things (adapter exists, ranges, etc.)
    if body.temperature is not None and body.temperature < 0:
        raise HTTPException(400, "temperature must be >= 0")

    adapter_id = body.adapter_id
    if adapter_id:
        try:
            path = await _adapters.resolve_path(adapter_id)  # existence check only
        except FileNotFoundError:
            raise HTTPException(404, f"Adapter '{adapter_id}' not found")
        # Attach now (single-request path; ok to do here)
        await _engine.attach_adapter(adapter_id, str(path))

    req = GenerateRequest(
        prompt=body.prompt,
        max_tokens=body.max_tokens,
        temperature=body.temperature,
        top_p=body.top_p,
        adapter_id=adapter_id,
        tenant_id=body.tenant_id,
        stream=True,
    )

    async def event_gen():
        # Optional: initial hello
        yield {"event": "start", "data": ""}

        async for chunk in _engine.stream_generate_single(req):
            # OpenAI-style: each line is `data: <json>`
            yield {"data": chunk}

        yield {"event": "end", "data": "[DONE]"}

    return EventSourceResponse(event_gen())
