
# Minimal HF-backed engine stub (no actual HF load for scaffold)
from typing import List, AsyncIterator
import asyncio
from ...core.types import GenerateRequest, GenerateResult, VerifyRequest, VerifyResult
from .engine import IEngine

class HFEngine(IEngine):
    def __init__(self):
        self.adapters: dict[str, str] = {}

    async def warmup(self) -> None:
        await asyncio.sleep(0)

    async def attach_adapter(self, adapter_id: str, path: str) -> None:
        self.adapters[adapter_id] = str(path)

    async def detach_adapter(self, adapter_id: str) -> None:
        self.adapters.pop(adapter_id, None)

    async def generate_batch(self, reqs: List[GenerateRequest]) -> List[GenerateResult]:
        # Fake generation for scaffold
        out = []
        for r in reqs:
            text = f"[adapter={r.adapter_id or 'none'}] {r.prompt} ... (generated)"
            out.append(GenerateResult(text=text, tokens=min(r.max_tokens, 32)))
        await asyncio.sleep(0)
        return out

    async def stream_generate_batch(self, reqs: List[GenerateRequest]) -> List[AsyncIterator[GenerateResult]]:
        async def _one(r: GenerateRequest):
            for _ in range(3):
                yield GenerateResult(text="...", tokens=1)
                await asyncio.sleep(0.05)
        return [ _one(r) for r in reqs ]

    async def verify_batch(self, reqs: List[VerifyRequest]) -> List[VerifyResult]:
        # Accept all as a placeholder
        return [ VerifyResult(accepted=len(r.proposed), text="(verified)", tokens=len(r.proposed)) for r in reqs ]
