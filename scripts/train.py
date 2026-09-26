#!/usr/bin/env python3
"""Fine-tune Tell-4B with LoRA/PEFT on Apple MPS (CUDA/CPU fallbacks).

Answer-only loss (default): only the tokens after ``ANSWER:`` contribute to
the loss (prompt tokens are masked with -100). This stops the model wasting
capacity on reproducing the input prompt and focuses learning on
``state + question + options -> label``. Use ``--full-loss`` to train on the
whole sequence instead (plain causal SFT).

Examples:
    python scripts/train.py --base-model Qwen/Qwen3.5-4B \\
        --train-file data/train.jsonl --valid-file data/valid.jsonl \\
        --output-dir outputs/tell-4b

    # Smoke test (~1 epoch, 100 samples):
    python scripts/train.py --max-train-samples 100 --epochs 1

    # With a YAML config:
    python scripts/train.py --config configs/mac_m3_48gb.yaml
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import yaml

# Imported lazily inside main() where possible so --help works without deps.

FALLBACK_MODELS = [
    "Qwen/Qwen3-4B",
    "Qwen/Qwen2.5-3B-Instruct",
    "Qwen2.5-3B-Instruct",
]

DEFAULT_TARGETS = "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj"


def get_device() -> torch.device:
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def device_name(d: torch.device) -> str:
    if d.type == "mps":
        return "Apple MPS"
    if d.type == "cuda":
        return f"CUDA ({torch.cuda.get_device_name(0)})"
    return "CPU"


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Fine-tune Tell-4B (LoRA) on MPS/CUDA/CPU.")
    ap.add_argument("--base-model", default="Qwen/Qwen3.5-4B")
    ap.add_argument("--train-file", default="data/train.jsonl")
    ap.add_argument("--valid-file", default="data/valid.jsonl")
    ap.add_argument("--output-dir", default="outputs/tell-4b")
    ap.add_argument("--config", default=None, help="YAML config file (CLI flags override it).")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--grad-accum", type=int, default=16)
    ap.add_argument("--max-seq-len", type=int, default=2048)
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--lora-alpha", type=int, default=32)
    ap.add_argument("--lora-dropout", type=float, default=0.05)
    ap.add_argument("--target-modules", default=DEFAULT_TARGETS)
    ap.add_argument("--max-train-samples", type=int, default=None)
    ap.add_argument("--max-valid-samples", type=int, default=None)
    ap.add_argument("--dtype", default="float32",
                    choices=["float32", "float16", "bfloat16", "auto"])
    ap.add_argument("--full-loss", action="store_true",
                    help="Train on the full sequence instead of answer-only loss.")
    ap.add_argument("--seed", type=int, default=7)
    return ap.parse_args()


def apply_config(args: argparse.Namespace) -> argparse.Namespace:
    if not args.config:
        return args
    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    # CLI wins: only fill values that are still at argparse defaults is complex;
    # simpler rule: config provides base, CLI overrides only if explicitly passed.
    import sys as _sys
    cli_keys = {a.split("=")[0].lstrip("-").replace("-", "_") for a in _sys.argv[1:] if a.startswith("--")}
    for k, v in cfg.items():
        key = k.replace("-", "_")
        if key in vars(args) and key not in cli_keys:
            setattr(args, key, v)
    return args


def main() -> None:
    args = apply_config(parse_args())
    device = get_device()
    print("Tell-4B training")
    print(f"Device: {device_name(device)}")
    print(f"Base model: {args.base_model}")

    try:
        from transformers import (
            AutoConfig,
            AutoModelForCausalLM,
            AutoTokenizer,
            Trainer,
            TrainingArguments,
            DataCollatorForLanguageModeling,
        )
        from peft import LoraConfig, get_peft_model, TaskType
        from datasets import load_dataset
    except ImportError as e:
        print(f"ERROR: missing dependency: {e}\nInstall with: pip install -r requirements.txt",
              file=sys.stderr)
        raise SystemExit(2)

    # Base-model availability check with explicit fallback guidance.
    try:
        AutoConfig.from_pretrained(args.base_model, trust_remote_code=True)
    except Exception as e:
        print(f"ERROR: could not resolve base model '{args.base_model}': {e}", file=sys.stderr)
        print("The default is Qwen/Qwen3.5-4B. If that identifier is unavailable with your",
              file=sys.stderr)
        print("transformers version, pick one explicitly instead of silently switching, e.g.:",
              file=sys.stderr)
        for fb in FALLBACK_MODELS:
            print(f"  --base-model {fb}", file=sys.stderr)
        raise SystemExit(2)

    dtype_map = {"float32": torch.float32, "float16": torch.float16,
                 "bfloat16": torch.bfloat16, "auto": "auto"}
    dtype = dtype_map[args.dtype]

    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    model = AutoModelForCausalLM.from_pretrained(
        args.base_model, trust_remote_code=True,
        torch_dtype=dtype, low_cpu_mem_usage=True,
    )
    # LoRA must stay trainable in fp32 even if base weights are fp16.
    if args.dtype in ("float16", "bfloat16"):
        model = model.float() if device.type == "mps" else model

    target_modules = [t.strip() for t in args.target_modules.split(",") if t.strip()]
    peft_cfg = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_r, lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=target_modules,
        bias="none",
    )
    model = get_peft_model(model, peft_cfg)
    trainable, total = 0, 0
    for _, p in model.named_parameters():
        total += p.numel()
        if p.requires_grad:
            trainable += p.numel()
    print(f"Trainable parameters: {trainable:,} / {total:,} ({100 * trainable / max(total, 1):.2f}%)")

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    from tell.prompt import build_training_text, build_prompt  # noqa: E402

    def load_records(path: str) -> list[dict]:
        with open(path, encoding="utf-8") as f:
            return [json.loads(line) for line in f if line.strip()]

    train_records = load_records(args.train_file)
    valid_records = load_records(args.valid_file)
    if args.max_train_samples:
        train_records = train_records[: args.max_train_samples]
    if args.max_valid_samples:
        valid_records = valid_records[: args.max_valid_samples]
    print(f"Dataset size: {len(train_records)} train / {len(valid_records)} valid")
    print(f"Loss mode: {'full-sequence SFT' if args.full_loss else 'answer-only (prompt masked)'}")

    def tokenize_record(r: dict) -> dict:
        full_text = build_training_text(r)
        prompt_text = build_prompt(r["state"], r["question"], r["options"])
        full = tokenizer(full_text, truncation=True, max_length=args.max_seq_len)
        if args.full_loss:
            full["labels"] = list(full["input_ids"])
        else:
            # Mask prompt tokens so cross-entropy focuses on "ANSWER: <label>".
            prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
            n_mask = min(len(prompt_ids), len(full["input_ids"]))
            labels = [-100] * n_mask + list(full["input_ids"][n_mask:])
            # Edge case: truncation removed the answer -> keep last token supervised.
            if all(v == -100 for v in labels) and labels:
                labels[-1] = full["input_ids"][-1]
            full["labels"] = labels
        return full

    train_ds = load_dataset("json", data_files=args.train_file, split="train")
    valid_ds = load_dataset("json", data_files=args.valid_file, split="train")
    if args.max_train_samples:
        train_ds = train_ds.select(range(min(args.max_train_samples, len(train_ds))))
    if args.max_valid_samples:
        valid_ds = valid_ds.select(range(min(args.max_valid_samples, len(valid_ds))))
    train_tok = train_ds.map(tokenize_record, remove_columns=train_ds.column_names)
    valid_tok = valid_ds.map(tokenize_record, remove_columns=valid_ds.column_names)

    out_dir = Path(args.output_dir)
    targs = TrainingArguments(
        output_dir=str(out_dir),
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_steps=10,
        load_best_model_at_end=True,
        save_total_limit=2,
        report_to="none",
        seed=args.seed,
        fp16=False,  # MPS does not support fp16 training reliably; keep fp32.
        bf16=False,
        optim="adamw_torch",
    )
    trainer = Trainer(
        model=model,
        args=targs,
        train_dataset=train_tok,
        eval_dataset=valid_tok,
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
    )
    trainer.train()
    model.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)
    print(f"Saved LoRA adapter + tokenizer -> {out_dir}")


if __name__ == "__main__":
    main()
