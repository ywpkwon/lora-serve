
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    model_id: str = "meta-llama/Llama-3-8b"
    adapter_root: str = "./examples/adapters"
    dtype: str = "bfloat16"
    max_batch_tokens: int = 8192
    max_wait_ms: int = 10
    kv_block_tokens: int = 512
    kv_capacity_blocks: int = 4096
    enable_spec_decode: bool = True
    draft_model_id: str | None = None

settings = Settings()
