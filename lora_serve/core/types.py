
from dataclasses import dataclass
from typing import Optional, List

@dataclass
class GenerateRequest:
    prompt: str
    max_tokens: int = 64
    temperature: float = 0.7
    top_p: float = 0.95
    adapter_id: Optional[str] = None
    tenant_id: Optional[str] = None
    stream: bool = False

@dataclass
class GenerateResult:
    text: str
    tokens: int

@dataclass
class VerifyRequest:
    prompt: str
    proposed: List[int]  # token ids

@dataclass
class VerifyResult:
    accepted: int
    text: str
    tokens: int

    def to_generate_result(self) -> GenerateResult:
        return GenerateResult(text=self.text, tokens=self.tokens)
