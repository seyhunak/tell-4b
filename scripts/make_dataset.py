#!/usr/bin/env python3
"""Generate the synthetic starter/smock-test dataset for Tell-4B.

Creates train.jsonl / valid.jsonl / test.jsonl in the Tell record format:

    {"state": ..., "question": ..., "options": [{"label","key","description"}], "answer": "B"}

The data is *synthetic smoke-test data only* — it exists so the full
pipeline (validate -> train -> infer -> evaluate) can be exercised end to
end on a MacBook. It is NOT sufficient for production-quality fine-tuning.
See data/README.md.

Usage:
    python scripts/make_dataset.py [--out-dir data] [--seed 7]
    python scripts/make_dataset.py --train-n 120 --valid-n 30 --test-n 60
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

VENDORS = ["Vendor X", "Acme GmbH", "Globex Ltd", "Initech", "Umbrella Corp", "Hooli"]
CURRENCIES = ["EUR", "USD", "GBP"]


def rec(state: str, question: str, options: list[tuple], answer: str) -> dict:
    return {
        "state": state,
        "question": question,
        "options": [
            {"label": lab, "key": key, "description": desc} for lab, key, desc in options
        ],
        "answer": answer,
    }


def finance_records(rng: random.Random, n: int, id_base: int) -> list[dict]:
    """Generate n finance records cycling through 10 sub-types."""
    out: list[dict] = []
    i = 0
    while len(out) < n:
        kind = i % 10
        j = id_base + i
        v = rng.choice(VENDORS)
        amt = rng.choice([250, 500, 1000, 1250, 2400, 5000, 9800])
        diff = rng.choice([1, 5, 50, 200, 250])
        cur = rng.choice(CURRENCIES)
        cur2 = "USD" if cur == "EUR" else "EUR"
        inv = f"INV-{j}"
        po = f"PO-{j}"

        if kind == 0:  # invoice matching -> match
            out.append(rec(
                f"Invoice {inv} from {v} is {amt} {cur}. "
                f"The corresponding purchase order {po} is {amt} {cur}. "
                f"Reference numbers agree.",
                "How should this transaction be classified?",
                [("A", "match", "Amounts match"), ("B", "mismatch", "Amounts do not match"),
                 ("C", "needs_review", "Insufficient information")], "A"))
        elif kind == 1:  # PO matching amount mismatch
            out.append(rec(
                f"Invoice {inv} from {v} is {amt + diff} {cur}. "
                f"The corresponding purchase order {po} is {amt} {cur}. "
                f"No tolerance rule applies.",
                "How should this transaction be classified?",
                [("A", "match", "Amounts match"), ("B", "mismatch", "Amounts do not match"),
                 ("C", "needs_review", "Insufficient information")], "B"))
        elif kind == 2:  # duplicate invoice detection
            out.append(rec(
                f"Invoice {inv} from {v} for {amt} {cur} was already paid on 2026-03-0{rng.randint(1,9)}. "
                f"An identical invoice {inv} for {amt} {cur} arrived again today.",
                "Is the new invoice a duplicate?",
                [("A", "unique", "First time seen"), ("B", "duplicate", "Already processed"),
                 ("C", "needs_review", "Insufficient information")], "B"))
        elif kind == 3:  # payment reconciliation
            out.append(rec(
                f"Payment of {amt} {cur} to {v} matches open invoice {inv} of {amt} {cur}. "
                f"Value dates and references agree.",
                "What is the reconciliation result?",
                [("A", "match", "Payment matches invoice"), ("B", "mismatch", "Payment does not match"),
                 ("C", "needs_review", "Insufficient information")], "A"))
        elif kind == 4:  # bank/ledger reconciliation
            out.append(rec(
                f"Bank statement shows {amt} {cur} from {v}. "
                f"Ledger entry for {inv} expects {amt} {cur}. Both agree.",
                "What is the reconciliation result?",
                [("A", "match", "Records agree"), ("B", "mismatch", "Records disagree"),
                 ("C", "needs_review", "Insufficient information")], "A"))
        elif kind == 5:  # amount mismatch
            out.append(rec(
                f"The invoice total is {amt + diff} {cur} while the purchase order total is {amt} {cur}.",
                "How should the transaction be classified?",
                [("A", "match", "Amounts match"), ("B", "mismatch", "Amounts do not match"),
                 ("C", "needs_review", "Insufficient information")], "B"))
        elif kind == 6:  # currency mismatch
            out.append(rec(
                f"Invoice {inv} is billed in {cur} but purchase order {po} is in {cur2}. "
                f"Amounts are numerically equal but currencies differ.",
                "How should this transaction be classified?",
                [("A", "match", "Records agree"), ("B", "mismatch", "Currency mismatch"),
                 ("C", "needs_review", "Insufficient information")], "B"))
        elif kind == 7:  # missing reference (include payment ID for uniqueness)
            out.append(rec(
                f"Payment PAY-{j} of {amt} {cur} arrived from {v} with no invoice reference "
                f"and no remittance advice.",
                "How should this payment be classified?",
                [("A", "match", "Matched to an invoice"), ("B", "mismatch", "Contradicts records"),
                 ("C", "needs_review", "Missing reference, needs human review")], "C"))
        elif kind == 8:  # tolerance handling -> match within tolerance
            out.append(rec(
                f"Invoice {inv} is {amt + 1} {cur}; purchase order {po} is {amt} {cur}. "
                f"A tolerance rule allows differences up to 5 {cur}.",
                "How should this transaction be classified under the tolerance rule?",
                [("A", "match", "Within tolerance"), ("B", "mismatch", "Outside tolerance"),
                 ("C", "needs_review", "Insufficient information")], "A"))
        else:  # needs-human-review: conflicting evidence
            out.append(rec(
                f"Invoice {inv} from {v} shows {amt} {cur} but the attachment is unreadable "
                f"and the PO lookup timed out.",
                "How should this transaction be classified?",
                [("A", "match", "Amounts match"), ("B", "mismatch", "Amounts do not match"),
                 ("C", "needs_review", "Insufficient information")], "C"))
        i += 1
    return out[:n]


def general_records(rng: random.Random, n: int, offset: int) -> list[dict]:
    """Generate n general-classification records cycling through 5 sub-types."""
    pos = ["I love this product, it works perfectly!", "Excellent service, very happy.",
           "This is wonderful, five stars."]
    neg = ["I hate this, it broke immediately.", "Terrible experience, very disappointed.",
           "Awful quality, do not buy."]
    neu = ["The package arrived on Tuesday.", "The item weighs 2 kg.",
           "The meeting is scheduled for 10am."]
    out: list[dict] = []
    i = 0
    while len(out) < n:
        kind = (offset + i) % 5
        k = offset + i
        if kind == 0:  # sentiment (3-way)
            cands = [("A", "positive", "Positive sentiment"), ("B", "negative", "Negative sentiment"),
                     ("C", "neutral", "Neutral sentiment")]
            pick = k % 3
            txt = [pos, neg, neu][pick][rng.randrange(3)]
            out.append(rec(f"Customer review #{k}: \"{txt}\"",
                           "What is the sentiment of the review?",
                           cands, ["A", "B", "C"][pick]))
        elif kind == 1:  # intent (4-way to exercise A-D)
            intents = [("billing question", "A"), ("cancel subscription", "B"),
                       ("technical issue", "C"), ("general inquiry", "D")]
            texts = {"billing question": f"Why was I charged twice on invoice {k}?",
                     "cancel subscription": "Please cancel my subscription effective immediately.",
                     "technical issue": "The app crashes on login since yesterday.",
                     "general inquiry": "What are your opening hours?"}
            key, lab = intents[k % 4]
            out.append(rec(f"Support message #{k}: \"{texts[key]}\"",
                           "Route this message to the correct queue.",
                           [("A", "billing", "Billing question"), ("B", "cancel", "Cancellation request"),
                            ("C", "technical", "Technical issue"), ("D", "general", "General inquiry")], lab))
        elif kind == 2:  # simple logical reasoning
            a, b = rng.randint(2, 20), rng.randint(2, 20)
            if k % 2 == 0:
                out.append(rec(f"All invoices over 1000 EUR need approval. Invoice #{k} is {1000 + a * 10} EUR.",
                               "Does this invoice need approval?",
                               [("A", "yes", "Approval required"), ("B", "no", "No approval needed"),
                                ("C", "needs_review", "Cannot determine")], "A"))
            else:
                out.append(rec(f"All invoices over 1000 EUR need approval. Invoice #{k} is {900 - b} EUR.",
                               "Does this invoice need approval?",
                               [("A", "yes", "Approval required"), ("B", "no", "No approval needed"),
                                ("C", "needs_review", "Cannot determine")], "B"))
        elif kind == 3:  # factual classification
            if k % 2 == 0:
                out.append(rec(f"Statement #{k}: Water boils at 100 degrees Celsius at sea level.",
                               "Is the statement factually correct?",
                               [("A", "true", "Correct statement"), ("B", "false", "Incorrect statement"),
                                ("C", "needs_review", "Cannot verify")], "A"))
            else:
                out.append(rec(f"Statement #{k}: The capital of France is Berlin.",
                               "Is the statement factually correct?",
                               [("A", "true", "Correct statement"), ("B", "false", "Incorrect statement"),
                                ("C", "needs_review", "Cannot verify")], "B"))
        else:  # customer-support routing (3-way)
            queues = [("A", "billing", "Billing team"), ("B", "technical", "Technical team"),
                      ("C", "sales", "Sales team")]
            pick = k % 3
            msgs = [f"My invoice #{k} has the wrong VAT amount.",
                    f"The dashboard shows error 500 for account {k}.",
                    f"We would like a quote for 50 seats starting Q{k % 4 + 1}."]
            out.append(rec(f"Ticket #{k}: \"{msgs[pick]}\"",
                           "Which team should handle this ticket?",
                           queues, ["A", "B", "C"][pick]))
        i += 1
    return out[:n]


def build_split(rng: random.Random, n_fin: int, n_gen: int, id_base: int) -> list[dict]:
    records = finance_records(rng, n_fin, id_base) + general_records(rng, n_gen, id_base)
    rng.shuffle(records)
    return records


def write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate synthetic Tell-4B starter dataset.")
    ap.add_argument("--out-dir", default="data")
    ap.add_argument("--train-n", type=int, default=120)
    ap.add_argument("--valid-n", type=int, default=30)
    ap.add_argument("--test-n", type=int, default=60)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    out = Path(args.out_dir)
    # ~58% finance / ~42% general per split
    def split_counts(n: int) -> tuple[int, int]:
        n_fin = round(n * 0.58)
        return n_fin, n - n_fin

    rng = random.Random(args.seed)
    # Distinct ID bases per split avoid cross-split duplicate states.
    train = build_split(rng, *split_counts(args.train_n), id_base=1000)
    valid = build_split(rng, *split_counts(args.valid_n), id_base=2000)
    test = build_split(rng, *split_counts(args.test_n), id_base=3000)

    write_jsonl(out / "train.jsonl", train)
    write_jsonl(out / "valid.jsonl", valid)
    write_jsonl(out / "test.jsonl", test)
    print(f"Wrote {len(train)} train / {len(valid)} valid / {len(test)} test -> {out}/")


if __name__ == "__main__":
    main()
