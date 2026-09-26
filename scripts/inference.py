#!/usr/bin/env python3
"""Deterministic single-example inference for Tell-4B.

Example:
    python scripts/inference.py --adapter outputs/tell-4b \\
        --state "Invoice is 1200 EUR but PO is 1000 EUR." \\
        --question "What is the reconciliation result?" \\
        --options "A:match|B:mismatch|C:needs_review"

Prints just the label (e.g. ``B``) on stdout. With ``--json`` prints the
full ``{"label","status","raw"}`` object. Never invents a label: invalid
output yields ``{"label": null, "status": "invalid_output"}``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Run one Tell-4B decision.")
    ap.add_argument("--adapter", default=None, help="Path to LoRA adapter dir (omit for base-only).")
    ap.add_argument("--base-model", default="Qwen/Qwen3.5-4B")
    ap.add_argument("--state", required=True)
    ap.add_argument("--question", required=True)
    ap.add_argument("--options", required=True,
                    help='E.g. "A:match|B:mismatch|C:needs_review"')
    ap.add_argument("--max-new-tokens", type=int, default=8)
    ap.add_argument("--json", action="store_true", help="Print full result object.")
    return ap.parse_args()


def get_device() -> torch.device:
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def main() -> None:
    args = parse_args()
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    from tell.prompt import build_prompt, parse_options_string, get_allowed_labels
    from tell.inference import extract_label

    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as e:
        print(f"ERROR: missing dependency: {e}", file=sys.stderr)
        raise SystemExit(2)

    options = parse_options_string(args.options)
    allowed = get_allowed_labels(options)
    prompt = build_prompt(args.state, args.question, options)

    model_id = args.adapter or args.base_model
    tok_id = args.adapter if args.adapter else args.base_model
    tokenizer = AutoTokenizer.from_pretrained(tok_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if args.adapter:
        from peft import PeftModel
        base = AutoModelForCausalLM.from_pretrained(
            args.base_model, trust_remote_code=True,
            torch_dtype=torch.float32, low_cpu_mem_usage=True)
        model = PeftModel.from_pretrained(base, args.adapter)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            args.base_model, trust_remote_code=True,
            torch_dtype=torch.float32, low_cpu_mem_usage=True)
    model.eval()
    device = get_device()
    try:
        model.to(device)
    except Exception:
        device = torch.device("cpu")

    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model.generate(
            **inputs, max_new_tokens=args.max_new_tokens,
            do_sample=False, temperature=0.0, pad_token_id=tokenizer.pad_token_id,
        )
    gen_ids = out[0][inputs["input_ids"].shape[1]:]
    raw = tokenizer.decode(gen_ids, skip_special_tokens=True)
    result = extract_label(raw, allowed)
    if args.json:
        print(json.dumps(result, ensure_ascii=False))
    else:
        if result["label"] is None:
            print(json.dumps(result, ensure_ascii=False))
        else:
            print(result["label"])


if __name__ == "__main__":
    main()
