
import asyncio
from .adapters import LoRAAdapterManager
from ..scheduler.batcher import DynamicBatcher
from .types import GenerateRequest

class RequestRouter:
    def __init__(self, batcher: DynamicBatcher, adapters: LoRAAdapterManager):
        self.batcher = batcher
        self.adapters = adapters

    async def submit(self, req: GenerateRequest):
        if req.adapter_id:
            await self.adapters.ensure_loaded(req.adapter_id)
        return await self.batcher.enqueue(req)
