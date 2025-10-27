
import asyncio
import logging
from typing import List
from .queue import TenantQueues
from .policies import choose_batch
from ..core.engines.engine import IEngine
from ..core.adapters import LoRAAdapterManager


logger = logging.getLogger(__name__)


class DynamicBatcher:
    def __init__(self, engine: IEngine, queues: TenantQueues, adapters: LoRAAdapterManager,
                 max_batch_tokens: int, max_wait_ms: int):
        self.engine = engine
        self.queues = queues
        self.adapters = adapters
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

            # Determine the adapter for this batch (all the same by policy)
            adapter_id = getattr(batch[0].req, "adapter_id", None)
            try:
                if adapter_id:
                    # Ensure loaded (path resolution/caching) at the manager layer
                    path = await self.adapters.ensure_loaded(adapter_id)
                    # Make it active on the engine (idempotent)
                    await self.engine.attach_adapter(adapter_id, str(path))
                    # Now generate
                results = await self.engine.generate_batch([e.req for e in batch])
            except FileNotFoundError:
                # Defensive: adapter vanished between route check and now
                for e in batch:
                    if not e.fut.done():
                        e.fut.set_exception(RuntimeError(f"Adapter '{adapter_id}' not found"))
                continue
            except Exception as ex:
                for e in batch:
                    if not e.fut.done():
                        e.fut.set_exception(ex)
                continue

            logger.debug("Engine returned %d results", len(results))
            for e, r in zip(batch, results):
                e.fut.set_result(r)
