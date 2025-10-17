
from pydantic import BaseModel

class GenerateIn(BaseModel):
    prompt: str
    max_tokens: int = 64
    temperature: float = 0.7
    top_p: float = 0.95
    adapter_id: str | None = None
    tenant_id: str | None = None
    stream: bool = False

class GenerateOut(BaseModel):
    text: str
    tokens: int
