#!/usr/bin/env python3
"""Push a Tell-4B adapter (or merged model) to the Hugging Face Hub.

Usage:
    export HF_TOKEN="hf_..."
    python scripts/push_hf.py --adapter outputs/tell-4b --repo-id YOUR_USERNAME/tell-4b

Never hard-code tokens and never commit secrets — the token is read from
the HF_TOKEN environment variable (or `huggingface-cli login`).
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


MODEL_CARD = """---
license: mit
language: en
tags:
- decision-model
- classification
- lora
- qwen
base_model: {base_model}
---

# Tell-4B

**Experimental specialist decision model.** Tell-4B is a small (~4B) model
fine-tuned with LoRA for *structured decisions*, not open-ended chat. Given a
STATE, a QUESTION and a fixed set of OPTIONS it returns exactly one option
label (e.g. `B`).

> Built in a **Jev-style** specialist design: a small model for form-like
> classification tasks (invoice / PO matching, reconciliation triage) that
> returns exactly one label. Independent implementation built only from public
> open-source components.

## Intended use

- Invoice / PO matching, reconciliation triage, and other form-like
  classification tasks where the output must be a single label.
- Research and education on small specialist models on Apple Silicon.
- NOT for: open-ended generation, legal/financial advice, autonomous payments.

## Training methodology

- Base model: `{base_model}` (configurable via `--base-model`).
- Method: LoRA/PEFT supervised fine-tuning (`scripts/train.py`).
- Loss: **answer-only** — prompt tokens before `ANSWER:` are masked (`-100`),
  so cross-entropy focuses on the label. See README for rationale.
- Hardware: Apple Silicon M3 (MPS); CUDA/CPU fallbacks supported.

## LoRA configuration

- r: {lora_r}, alpha: {lora_alpha}, dropout: {lora_dropout}
- target modules: {target_modules}

## Dataset description

- JSONL records: `state + question + options -> answer label`.
- Starter set is synthetic smoke-test data only (train ~120 / valid ~30 /
  test ~60). Meaningful fine-tuning requires a substantially larger,
  licensed or independently generated dataset. `test.jsonl` is never trained on.

## Evaluation methodology

- `scripts/evaluate.py --mode base` vs `--mode tell` on held-out `test.jsonl`.
- Metrics: exact accuracy, invalid-output rate, per-class accuracy, confusion
  matrix. Only measured results are reported — never fabricated.

## Limitations

- Small context, label-only output; long or conflicting evidence degrades
  accuracy (see README robustness section).
- Prompt-injection resistance is best-effort: STATE is untrusted data.
- Starter dataset does NOT yield a production-quality model.

## Responsible use

- Keep a human in the loop for financial decisions.
- Validate structured outputs; treat `invalid_output` as abstention.

## License

MIT (see LICENSE) for the Tell-4B adapter, code and synthetic starter data.
Base-model weights follow their own license.

## How to run

```bash
pip install -r requirements.txt
python scripts/inference.py --adapter . \
  --state "Invoice is 1200 EUR but PO is 1000 EUR." \
  --question "What is the reconciliation result?" \
  --options "A:match|B:mismatch|C:needs_review"
# -> B
```

## Ollama

A Q4_K_M GGUF of the merged model is published for local runs
(see `Modelfile` in the GitHub repo):

```bash
ollama pull seyhunak/tell-4b
ollama run seyhunak/tell-4b "STATE: ... QUESTION: ... OPTIONS: ... ANSWER:"
```

## Live demo

A Next.js one-pager classifying 50 finance tickets in real time
(`demo/` in the GitHub repo, `npx .` after `npm run build`).
"""


def main() -> None:
    ap = argparse.ArgumentParser(description="Push Tell-4B to the Hugging Face Hub.")
    ap.add_argument("--adapter", default="outputs/tell-4b", help="Local adapter or merged dir.")
    ap.add_argument("--repo-id", required=True, help="E.g. YOUR_USERNAME/tell-4b")
    ap.add_argument("--base-model", default="Qwen/Qwen3.5-4B")
    ap.add_argument("--private", action="store_true", help="Create a private repo.")
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--lora-alpha", type=int, default=32)
    ap.add_argument("--lora-dropout", type=float, default=0.05)
    ap.add_argument("--target-modules", default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj")
    args = ap.parse_args()

    token = os.environ.get("HF_TOKEN")
    if not token:
        print("ERROR: HF_TOKEN is not set. Run: export HF_TOKEN=\"hf_...\"", file=sys.stderr)
        raise SystemExit(2)
    try:
        from huggingface_hub import HfApi
    except ImportError as e:
        print(f"ERROR: missing dependency: {e}", file=sys.stderr)
        raise SystemExit(2)

    src = Path(args.adapter)
    if not src.exists():
        print(f"ERROR: adapter dir not found: {src}", file=sys.stderr)
        raise SystemExit(2)

    card = MODEL_CARD.format(base_model=args.base_model, lora_r=args.lora_r,
                             lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout,
                             target_modules=args.target_modules)
    (src / "README.md").write_text(card, encoding="utf-8")
    print(f"Wrote model card -> {src / 'README.md'}")

    api = HfApi(token=token)
    api.create_repo(args.repo_id, exist_ok=True, private=args.private)
    api.upload_folder(folder_path=str(src), repo_id=args.repo_id,
                      commit_message="Upload Tell-4B (LoRA adapter + card)")
    print(f"Pushed -> https://huggingface.co/{args.repo_id}")


if __name__ == "__main__":
    main()
