
from typing import List
from .queue import TenantQueues

async def choose_batch(queues: TenantQueues, max_batch_tokens: int, max_wait_ms: int):
    # naive: pick first available one-by-one
    batch = []
    entry = await queues.pop_any()
    if entry:
        batch.append(entry)
    return batch
