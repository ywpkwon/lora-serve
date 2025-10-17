
from dataclasses import dataclass
from typing import Optional
from .logging import logger

@dataclass
class ModelHandle:
    model_id: str
    dtype: str = "bfloat16"

class ModelRegistry:
    def __init__(self):
        self.base: Optional[ModelHandle] = None

    async def load_base(self, model_id: str, dtype: str = "bfloat16"):
        # TODO: actually load HF model & tokenizer
        self.base = ModelHandle(model_id, dtype)
        logger.info("Loaded base model %s (%s)", model_id, dtype)
        return self.base
