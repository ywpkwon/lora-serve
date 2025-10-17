
import asyncio
from dataclasses import dataclass
from typing import Any, Dict

@dataclass
class _Entry:
    fut: asyncio.Future
    req: Any

class TenantQueues:
    def __init__(self):
        self._q: Dict[str, asyncio.Queue[_Entry]] = {}

    def push(self, tenant: str, req: Any) -> asyncio.Future:
        fut: asyncio.Future = asyncio.get_event_loop().create_future()
        q = self._q.setdefault(tenant, asyncio.Queue())
        q.put_nowait(_Entry(fut=fut, req=req))
        return fut

    async def pop_any(self):
        # naive scan for scaffold
        for q in self._q.values():
            if not q.empty():
                return await q.get()
        return None
