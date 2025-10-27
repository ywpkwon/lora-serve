
# Minimal HF-backed engine stub (no actual HF load for scaffold)
from typing import List, AsyncIterator
import asyncio
import logging
import threading
import torch
from ...core.types import GenerateRequest, GenerateResult, VerifyRequest, VerifyResult
from .engine import IEngine
from transformers import AutoTokenizer, AutoModelForCausalLM, TextIteratorStreamer

try:
    from peft import PeftModel  # core wrapper that can hold/load adapters
    _HAS_PEFT = True
except Exception:
    PeftModel = None  # type: ignore
    _HAS_PEFT = False


logger = logging.getLogger(__name__)


class HFEngine(IEngine):
    def __init__(self, model_id: str, dtype: str = "bfloat16", device: str = "cuda"):
        self.model_id = model_id
        self.device = device if torch.cuda.is_available() else "cpu"

        # dtype handling
        _dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16, "fp16": torch.float16}.get(dtype, torch.float16)

        logger.info("Loading base model %s (dtype=%s, device=%s)", model_id, _dtype, self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True, trust_remote_code=True)
        if self.tokenizer.pad_token_id is None and self.tokenizer.eos_token_id is not None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            dtype=_dtype,
            device_map="auto" if self.device == "cuda" else None,
            trust_remote_code=True,
        )
        self.model.eval()
        torch.set_grad_enabled(False)
        torch.backends.cuda.matmul.allow_tf32 = True

        # adapter_id -> attached flag/path
        self._adapters: dict[str, str] = {}
        self._adapter_lock = asyncio.Lock()

    async def warmup(self) -> None:
        await asyncio.sleep(0)

    async def attach_adapter(self, adapter_id: str, path: str) -> None:
        if not _HAS_PEFT:
            logger.warning("PEFT not installed; skipping adapter %s", adapter_id)
            return

        async with self._adapter_lock:
            # Already have this adapter registered?
            existing = []
            if isinstance(self.model, PeftModel) and hasattr(self.model, "peft_config"):
                existing = list(getattr(self.model, "peft_config", {}).keys())

            if not isinstance(self.model, PeftModel):
                # First adapter: wrap base model with PEFT and register under adapter_id
                try:
                    self.model = PeftModel.from_pretrained(
                        self.model,
                        path,
                        adapter_name=adapter_id,     # <-- key line
                        is_trainable=False,          # inference
                    )
                except TypeError:
                    # Older PEFT: no adapter_name kw; it will be "default"
                    self.model = PeftModel.from_pretrained(self.model, path)
            else:
                # Model already PEFT-wrapped: load this adapter if missing
                if adapter_id not in existing:
                    # Register the new adapter under adapter_id
                    kwargs = {"adapter_name": adapter_id}
                    try:
                        self.model.load_adapter(path, **kwargs)
                    except TypeError:
                        # Older PEFT signatures (no adapter_name kw)
                        self.model.load_adapter(path)
                        # It likely registered as "default" – we'll handle selection below

            # Now select the adapter safely
            try:
                names = list(getattr(self.model, "peft_config", {}).keys())
                if adapter_id in names:
                    self.model.set_adapter(adapter_id)
                elif "default" in names:
                    logger.debug("Adapter '%s' not registered; falling back to 'default'", adapter_id)
                    self.model.set_adapter("default")
                else:
                    raise RuntimeError(f"No selectable adapters found: have {names}")
            except Exception as e:
                logger.exception("set_adapter failed for '%s' (have=%s): %s", adapter_id, existing, e)
                raise

            # book-keeping
            self._adapters[adapter_id] = str(path)
            logger.info("Adapter '%s' active (available=%s)", adapter_id, list(self.model.peft_config.keys()))

    async def detach_adapter(self, adapter_id: str) -> None:
        """Detach and free a specific LoRA adapter if loaded."""
        if not _HAS_PEFT or not isinstance(self.model, PeftModel):
            logger.debug("No PEFT model attached; nothing to detach.")
            return

        async with self._adapter_lock:
            if adapter_id not in self._adapters:
                logger.debug("Adapter %s not found in engine cache", adapter_id)
                return

            logger.info("Detaching adapter %s", adapter_id)
            try:
                if hasattr(self.model, "unload_adapter"):
                    self.model.unload_adapter(adapter_id)
                else:
                    # Fallback: clear from config dicts
                    if hasattr(self.model, "peft_config") and adapter_id in self.model.peft_config:
                        del self.model.peft_config[adapter_id]
            except Exception as e:
                logger.warning("Failed to unload adapter %s: %s", adapter_id, e)

            # remove from cache and LRU
            self._adapters.pop(adapter_id, None)

            # optional: torch.cuda.empty_cache() if GPU mem high
            # import torch; torch.cuda.empty_cache()

    async def generate_batch(self, reqs: List[GenerateRequest]) -> List[GenerateResult]:
        logger.debug("generate_batch called with %d reqs", len(reqs))
        prompts = [r.prompt for r in reqs]
        if not prompts:
            return []

        # Decide generation knobs for the whole batch (simple policy)
        max_new = max(max(1, r.max_tokens) for r in reqs)
        # If ANY request wants sampling (temperature > 0), enable sampling
        do_sample = any(r.temperature and r.temperature > 0 for r in reqs)
        # Since torch generate() can’t take per-row temperatures, pick a batch value.
        # Easiest: use the max across reqs when sampling is on.
        batch_temp = max((r.temperature or 0.0) for r in reqs) if do_sample else None
        batch_top_p = max((r.top_p or 0.0) for r in reqs) if do_sample else None

        # Tokenize as a padded batch
        tok = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,  # truncate *inputs* if needed to model's max position
        )
        input_ids = tok["input_ids"].to(self.model.device)
        attn_mask = tok["attention_mask"].to(self.model.device)
        input_len = input_ids.shape[1]
        logger.debug("tokens ready. going to generation inference.")

        try:
            with torch.inference_mode():
                outputs = self.model.generate(
                    input_ids=input_ids,
                    attention_mask=attn_mask,
                    max_new_tokens=max_new,
                    do_sample=do_sample,
                    temperature=batch_temp,
                    top_p=batch_top_p,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                )
        except RuntimeError as e:
            # Very basic OOM backoff: try halving max_new once.
            if "CUDA out of memory" in str(e) and max_new > 1:
                logger.warning("OOM at max_new=%d; retrying with %d", max_new, max_new // 2)
                with torch.inference_mode():
                    outputs = self.model.generate(
                        input_ids=input_ids,
                        attention_mask=attn_mask,
                        max_new_tokens=max_new // 2,
                        do_sample=do_sample,
                        temperature=batch_temp,
                        top_p=batch_top_p,
                        pad_token_id=self.tokenizer.pad_token_id,
                        eos_token_id=self.tokenizer.eos_token_id,
                    )
            else:
                logger.exception("generate_batch failed")
                raise

        # Split the batched outputs row-wise and decode only *new* tokens
        results: list[GenerateResult] = []
        for i in range(outputs.shape[0]):
            out_ids = outputs[i]
            # Guard against edge cases (e.g., EOS at step 0)
            new_slice = out_ids[input_len:] if out_ids.shape[0] > input_len else out_ids[:0]
            text = self.tokenizer.decode(new_slice, skip_special_tokens=True)
            results.append(GenerateResult(text=text, tokens=len(new_slice)))
        logger.debug("generate_batch: done (max_new=%d, do_sample=%s)", max_new, do_sample)
        return results

        # Fake generation for scaffold
        out = []
        for r in reqs:
            text = f"[adapter={r.adapter_id or 'none'}] {r.prompt} ... (generated)"
            out.append(GenerateResult(text=text, tokens=min(r.max_tokens, 32)))

        # logger.debug("Tokenizing %d prompts", len(prompts))
        # logger.debug("Model.generate done (batch=%d)", len(prompts))
        await asyncio.sleep(0)
        return out

    async def stream_generate_single(self, req: GenerateRequest):
        """
        Async generator yielding text chunks as they are produced.
        NOTE: single-request path (no batching).
        """
        prompt = req.prompt
        do_sample = (req.temperature or 0) > 0
        tok = self.tokenizer(
            [prompt], return_tensors="pt", padding=False, truncation=True
        )
        input_ids = tok["input_ids"].to(self.model.device)
        attn_mask = tok.get("attention_mask")
        if attn_mask is not None:
            attn_mask = attn_mask.to(self.model.device)

        # streamer will yield decoded text *incrementally*
        streamer = TextIteratorStreamer(
            self.tokenizer, skip_prompt=True, skip_special_tokens=True
        )

        gen_kwargs = dict(
            input_ids=input_ids,
            attention_mask=attn_mask,
            max_new_tokens=max(1, req.max_tokens),
            do_sample=do_sample,
            temperature=req.temperature if do_sample else None,
            top_p=req.top_p if do_sample else None,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            streamer=streamer,
        )

        # Run .generate() in a background thread, read tokens here
        def _run():
            try:
                with torch.inference_mode():
                    self.model.generate(**gen_kwargs)
            except Exception as e:
                logger.exception("stream generate failed: %s", e)
                # close streamer by stopping iteration
                try:
                    streamer.on_finalized_text()
                except Exception:
                    pass

        t = threading.Thread(target=_run, daemon=True)
        t.start()

        # The streamer is an iterator that blocks until next chunk is available.
        # Bridge it to an async generator:
        loop = asyncio.get_event_loop()
        while True:
            chunk = await loop.run_in_executor(None, next, streamer, None)
            if chunk is None:
                break
            yield chunk

    async def stream_generate_batch(self, reqs: List[GenerateRequest]) -> List[AsyncIterator[GenerateResult]]:
        async def _one(r: GenerateRequest):
            for _ in range(3):
                yield GenerateResult(text="...", tokens=1)
                await asyncio.sleep(0.05)
        return [ _one(r) for r in reqs ]

    async def verify_batch(self, reqs: List[VerifyRequest]) -> List[VerifyResult]:
        # Accept all as a placeholder
        return [ VerifyResult(accepted=len(r.proposed), text="(verified)", tokens=len(r.proposed)) for r in reqs ]
