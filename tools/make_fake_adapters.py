#!/usr/bin/env python
import argparse, torch, os
from transformers import AutoModelForCausalLM
from peft import LoraConfig, get_peft_model


def make_adapter(base_id: str, out_dir: str, r: int = 8, alpha: int = 16, target=("q_proj","v_proj")):
    print(f"[make] base={base_id} -> {out_dir} (r={r}, alpha={alpha}, targets={target})")
    # tiny, CPU load is fine for saving adapter structure
    model = AutoModelForCausalLM.from_pretrained(base_id, torch_dtype=torch.float16)
    cfg = LoraConfig(
        r=r, lora_alpha=alpha, lora_dropout=0.0,
        target_modules=list(target),
        bias="none",
        task_type="CAUSAL_LM",
    )
    peft_model = get_peft_model(model, cfg)

    # initialize LoRA params with small random values (no training)
    with torch.no_grad():
        for n, p in peft_model.named_parameters():
            if "lora_" in n:  # LoRA A/B weights
                p.uniform_(-1e-3, 1e-3)

    os.makedirs(out_dir, exist_ok=True)
    peft_model.save_pretrained(out_dir)
    print(f"[ok] saved {out_dir}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_id", required=True, help="e.g. microsoft/phi-3-mini-4k-instruct")
    ap.add_argument("--names", default="math,code,jp", help="comma-separated adapter names")
    ap.add_argument("--out_root", default="examples/adapters")
    ap.add_argument("--rank", type=int, default=8)
    args = ap.parse_args()
    for name in args.names.split(","):
        out = os.path.join(args.out_root, name.strip())
        make_adapter(args.model_id, out, r=args.rank)
