
from typing import List
from .queue import TenantQueues
from ..core.types import GenerateRequest


async def choose_batch(queues: TenantQueues, max_batch_tokens: int, max_wait_ms: int):
    # naive: pick first available one-by-one
    batch = []
    first = await queues.pop_any()
    if not first:
        return []

    batch = [first]
    budget = _rough_tokens(first.req)

    # Fill until budget or wait window reached
    # Small spin to let a few more arrive within wait window
    import asyncio
    start = first.enq_ts_ms
    while budget < max_batch_tokens:
        # Stop if we've waited long enough
        now = _now_ms()
        if now - start >= max_wait_ms:
            break
        # Try to grab another if any
        nxt = await queues.pop_any()
        if nxt is None:
            # nothing ready; give others a moment to enqueue
            await asyncio.sleep(0.001)
            continue
        new_cost = _rough_tokens(nxt.req)
        if budget + new_cost > max_batch_tokens:
            # too big; put it back? (naive policy: run it next tick → we just keep it out)
            # In a real system you'd have a put-back. OK to just run current batch now.
            # You could stash it in a local variable and re-enqueue; keeping it simple here.
            # We'll run it next tick naturally since it remains in queue.
            # (Because we didn't remove it; but we actually did pop() it.
            # For minimalism, let's keep it and run this batch; then push it back.)
            # Push-back:
            queues.push("default", nxt.req)  # if you track tenant_id, use it here
            break
        batch.append(nxt)
        budget += new_cost

    return batch


def _rough_tokens(req: GenerateRequest) -> int:
    # Very rough token proxy: chars/4 + max_new_tokens
    return max(1, len(req.prompt) // 4) + max(1, req.max_tokens)


def _now_ms() -> int:
    import asyncio
    return int(asyncio.get_event_loop().time() * 1000)
