#!/usr/bin/env python3
"""Validate Tell-4B JSONL dataset files.

Checks schema, label validity, empty fields, and cross-split state leakage.

Usage:
    python scripts/validate_dataset.py --train-file data/train.jsonl \
        --valid-file data/valid.jsonl --test-file data/test.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path


def load(path: Path) -> list[dict]:
    records = []
    with path.open(encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"ERROR {path}:{lineno}: invalid JSON: {e}", file=sys.stderr)
                raise SystemExit(2)
    return records


def check_file(records: list[dict], name: str) -> Counter:
    errors = 0
    labels = Counter()
    for i, r in enumerate(records):
        loc = f"{name}[{i}]"
        for field in ("state", "question", "options", "answer"):
            if field not in r:
                print(f"ERROR {loc}: missing field '{field}'")
                errors += 1
        if not isinstance(r.get("state", ""), str) or not r.get("state", "").strip():
            print(f"ERROR {loc}: empty 'state'")
            errors += 1
        if not isinstance(r.get("question", ""), str) or not r.get("question", "").strip():
            print(f"ERROR {loc}: empty 'question'")
            errors += 1
        opts = r.get("options", [])
        if not isinstance(opts, list) or len(opts) < 2:
            print(f"ERROR {loc}: 'options' must be a list with >= 2 entries")
            errors += 1
            continue
        seen = set()
        for o in opts:
            if not isinstance(o, dict) or not all(k in o for k in ("label", "key", "description")):
                print(f"ERROR {loc}: each option needs label/key/description: {o!r}")
                errors += 1
                continue
            seen.add(str(o["label"]).upper())
        ans = str(r.get("answer", "")).upper()
        if ans not in seen:
            print(f"ERROR {loc}: answer '{r.get('answer')}' not in option labels {sorted(seen)}")
            errors += 1
        labels[ans] += 1
    if errors:
        raise SystemExit(f"Validation failed for {name}: {errors} error(s)")
    return labels


def main() -> None:
    ap = argparse.ArgumentParser(description="Validate Tell-4B JSONL datasets.")
    ap.add_argument("--train-file", default="data/train.jsonl")
    ap.add_argument("--valid-file", default="data/valid.jsonl")
    ap.add_argument("--test-file", default="data/test.jsonl")
    args = ap.parse_args()

    files = {"train": Path(args.train_file), "valid": Path(args.valid_file), "test": Path(args.test_file)}
    for name, p in files.items():
        if not p.exists():
            print(f"ERROR: {name} file not found: {p}", file=sys.stderr)
            raise SystemExit(2)

    all_records = {name: load(p) for name, p in files.items()}
    for name, records in all_records.items():
        dist = check_file(records, name)
        print(f"{name}: {len(records)} examples, label distribution: {dict(dist)}")

    # Cross-split leakage check on exact state strings.
    states: dict[str, str] = {}
    leaked = 0
    for name, records in all_records.items():
        for r in records:
            s = r["state"]
            if s in states and states[s] != name:
                print(f"WARNING: duplicate state in {states[s]} and {name}: {s[:80]!r}...")
                leaked += 1
            states.setdefault(s, name)
    if leaked:
        print(f"WARNING: {leaked} duplicate state(s) across splits (fix with distinct ID ranges).")
    else:
        print("No cross-split duplicate states. OK.")


if __name__ == "__main__":
    main()
