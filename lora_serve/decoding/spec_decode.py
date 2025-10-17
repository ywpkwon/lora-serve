
from typing import List
from ..core.engines.engine import IEngine
from ..core.types import GenerateRequest, GenerateResult, VerifyRequest

class SpeculativeOrchestrator:
    def __init__(self, draft: IEngine, target: IEngine, max_draft_steps: int = 8):
        self.draft = draft
        self.target = target
        self.max_draft_steps = max_draft_steps

    async def generate(self, reqs: List[GenerateRequest]) -> List[GenerateResult]:
        draft_res = await self.draft.generate_batch(reqs)
        verify = [VerifyRequest(prompt=r.prompt, proposed=list(range(min(self.max_draft_steps, dr.tokens))))
                  for r, dr in zip(reqs, draft_res)]
        verified = await self.target.verify_batch(verify)
        return [v.to_generate_result() for v in verified]
