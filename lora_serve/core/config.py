
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    model_id: str = "meta-llama/Llama-3-8b"
    adapter_root: str = "./examples/adapters"
    dtype: str = "bfloat16"
    device: str = "cuda"
    max_batch_tokens: int = 8192
    max_wait_ms: int = 10
    kv_block_tokens: int = 512
    kv_capacity_blocks: int = 4096
    enable_spec_decode: bool = True
    draft_model_id: str | None = None

    # v2-style config:
    model_config = SettingsConfigDict(
        env_prefix="LORASERVE_",  # keys must start with this
        env_file=".env",          # load from project root .env
        extra="ignore",           # (optional) ignore unknown keys
    )

settings = Settings()
