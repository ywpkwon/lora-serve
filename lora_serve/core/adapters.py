
import asyncio
from pathlib import Path
from collections import OrderedDict

class LoRAAdapterManager:
    def __init__(self, base_dir: Path, max_loaded: int = 8):
        self.base_dir = base_dir
        self.max_loaded = max_loaded
        self.loaded: OrderedDict[str, Path] = OrderedDict()
        self._lock = asyncio.Lock()

    async def resolve_path(self, adapter_id: str) -> Path:
        # existence-only; no cache mutation
        path = self.base_dir / adapter_id
        if not path.exists():
            raise FileNotFoundError(path)
        return path

    async def ensure_loaded(self, adapter_id: str) -> Path:
        async with self._lock:
            if adapter_id in self.loaded:
                self.loaded.move_to_end(adapter_id)
                return self.loaded[adapter_id]
            path = self.base_dir / adapter_id
            if not path.exists():
                raise FileNotFoundError(path)
            if len(self.loaded) >= self.max_loaded:
                self.loaded.popitem(last=False)  # Evict LRU
            self.loaded[adapter_id] = path
            return path
