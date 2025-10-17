
class BlockAllocator:
    def __init__(self, block_tokens: int, capacity_blocks: int):
        self.block_tokens = block_tokens
        self.capacity = capacity_blocks
        self.used = 0

    def reserve(self, key: str, tokens: int):
        blocks = (tokens + self.block_tokens - 1) // self.block_tokens
        if self.used + blocks > self.capacity:
            return None
        self.used += blocks
        return blocks

    def release(self, key: str):
        # scaffold: no per-key tracking
        pass

    def free_blocks(self): return self.capacity - self.used
    def used_blocks(self): return self.used
    def fragmentation(self): return 0.0
