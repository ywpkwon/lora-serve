
import asyncio
import logging
from typing import List
from .queue import TenantQueues
from .policies import choose_batch
from ..core.engines.engine import IEngine


logger = logging.getLogger(__name__)


class DynamicBatcher:
    def __init__(self, engine: IEngine, queues: TenantQueues, max_batch_tokens: int, max_wait_ms: int):
        self.engine = engine
        self.queues = queues
        self.max_batch_tokens = max_batch_tokens
        self.max_wait_ms = max_wait_ms

    async def enqueue(self, req):
        fut = self.queues.push(getattr(req, "tenant_id", "default") or "default", req)
        return await fut

    async def run_forever(self):
        logger.info("DynamicBatcher started")
        while True:
            batch = await choose_batch(self.queues, self.max_batch_tokens, self.max_wait_ms)
            if not batch:
                await asyncio.sleep(0.001)
                continue
            logger.debug("Formed batch of %d requests", len(batch))
            results = await self.engine.generate_batch([e.req for e in batch])
            logger.debug("Engine returned %d results", len(results))
            for e, r in zip(batch, results):
                e.fut.set_result(r)
