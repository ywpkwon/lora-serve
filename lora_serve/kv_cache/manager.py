
from .allocator import BlockAllocator

class KVCacheManager:
    def __init__(self, block_size_tokens: int, capacity_blocks: int):
        self.alloc = BlockAllocator(block_size_tokens, capacity_blocks)

    def allocate(self, req_id: str, tokens: int):
        return self.alloc.reserve(req_id, tokens)

    def free(self, req_id: str):
        self.alloc.release(req_id)

    def stats(self):
        return {
            "blocks_free": self.alloc.free_blocks(),
            "blocks_used": self.alloc.used_blocks(),
            "frag_ratio": self.alloc.fragmentation(),
        }
