#!/usr/bin/env python3
"""Merge a LoRA adapter into its base model for deployment.

Keeps both workflows documented:
  1. LoRA adapter deployment  -> serve base + small adapter (recommended).
  2. Merged deployment        -> single self-contained Tell-4B directory.

Usage:
    python scripts/merge_adapter.py --base-model Qwen/Qwen3.5-4B \\
        --adapter outputs/tell-4b --output-dir outputs/tell-4b-merged
"""

from __future__ import annotations

import argparse
import sys


def main() -> None:
    ap = argparse.ArgumentParser(description="Merge LoRA adapter into base model.")
    ap.add_argument("--base-model", default="Qwen/Qwen3.5-4B")
    ap.add_argument("--adapter", default="outputs/tell-4b")
    ap.add_argument("--output-dir", default="outputs/tell-4b-merged")
    args = ap.parse_args()

    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from peft import PeftModel
    except ImportError as e:
        print(f"ERROR: missing dependency: {e}", file=sys.stderr)
        raise SystemExit(2)

    print(f"Loading base model: {args.base_model}")
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model, trust_remote_code=True,
        torch_dtype=torch.float32, low_cpu_mem_usage=True)
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    print(f"Loading adapter: {args.adapter}")
    model = PeftModel.from_pretrained(model, args.adapter)
    print("Merging adapter into base weights...")
    merged = model.merge_and_unload()
    merged.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"Merged model saved -> {args.output_dir}")
    print("Deploy either:")
    print(f"  1. LoRA: base {args.base_model} + adapter {args.adapter}")
    print(f"  2. Merged: {args.output_dir}")


if __name__ == "__main__":
    main()
