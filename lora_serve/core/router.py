
import asyncio
import logging
from .adapters import LoRAAdapterManager
from ..scheduler.batcher import DynamicBatcher
from .types import GenerateRequest


logger = logging.getLogger(__name__)


class RequestRouter:
    def __init__(self, batcher: DynamicBatcher, adapters: LoRAAdapterManager):
        self.batcher = batcher
        self.adapters = adapters

    async def submit(self, req: GenerateRequest):
        if req.adapter_id:
            logger.debug("Ensuring adapter %s is loaded", req.adapter_id)
            await self.adapters.ensure_loaded(req.adapter_id)
        logger.debug("Enqueueing request for prompt len=%d", len(req.prompt))
        return await self.batcher.enqueue(req)
