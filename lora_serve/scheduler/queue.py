
import asyncio
from dataclasses import dataclass
from typing import Any, Dict, Optional

@dataclass
class _Entry:
    fut: asyncio.Future
    req: Any
    enq_ts_ms: int

class TenantQueues:
    def __init__(self):
        self._q: Dict[str, asyncio.Queue[_Entry]] = {}

    def push(self, tenant: str, req: Any) -> asyncio.Future:
        fut: asyncio.Future = asyncio.get_event_loop().create_future()
        q = self._q.setdefault(tenant, asyncio.Queue())
        q.put_nowait(_Entry(fut=fut, req=req, enq_ts_ms=_now_ms()))
        return fut

    async def pop_any(self) -> Optional[_Entry]:
        # naive scan for scaffold
        for q in self._q.values():
            if not q.empty():
                return await q.get()
        return None


def _now_ms() -> int:
    loop = asyncio.get_event_loop()
    return int(loop.time() * 1000)
