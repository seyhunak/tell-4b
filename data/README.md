# Starter dataset (synthetic smoke-test data)

This directory holds small **synthetic** JSONL files so the full pipeline
(validate → train → infer → evaluate) can be verified end to end:

- `train.jsonl` — ~120 examples
- `valid.jsonl` — ~30 examples
- `test.jsonl` — ~60 examples (held out; never train on it)

Regenerate with:

```bash
python scripts/make_dataset.py --out-dir data --seed 7
python scripts/validate_dataset.py
```

## Important

- These are **smoke-test examples only**. They do NOT produce a
  production-quality model and must not be presented as such.
- Meaningful fine-tuning requires a substantially larger, licensed or
  independently generated dataset (thousands–tens of thousands of reviewed
  examples, class-balanced, with real-world edge cases).
- Keep `test.jsonl` completely separate from training.

## Format

Each line is one record:

```json
{
  "state": "The invoice total is 1200 EUR while the purchase order total is 1000 EUR.",
  "question": "How should the transaction be classified?",
  "options": [
    {"label": "A", "key": "match", "description": "Amounts match"},
    {"label": "B", "key": "mismatch", "description": "Amounts do not match"},
    {"label": "C", "key": "needs_review", "description": "Insufficient information"}
  ],
  "answer": "B"
}
```

Coverage: invoice/PO matching, duplicate detection, payment and bank/ledger
reconciliation, amount/currency mismatch, missing reference, tolerance
handling, needs-human-review cases, plus sentiment, intent, logic, factual
and support-routing classification.
