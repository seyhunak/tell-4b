"""Tests for the dataset format and starter data."""

import json
from pathlib import Path

DATA = Path(__file__).resolve().parents[1] / "data"


def load(name):
    with open(DATA / name, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def check_record(r):
    assert isinstance(r["state"], str) and r["state"].strip()
    assert isinstance(r["question"], str) and r["question"].strip()
    assert isinstance(r["options"], list) and len(r["options"]) >= 2
    labels = set()
    for o in r["options"]:
        assert {"label", "key", "description"} <= set(o)
        labels.add(str(o["label"]).upper())
    assert str(r["answer"]).upper() in labels


def test_files_exist_and_meet_minimums():
    counts = {n: len(load(n)) for n in ("train.jsonl", "valid.jsonl", "test.jsonl")}
    assert counts["train.jsonl"] >= 100, counts
    assert counts["valid.jsonl"] >= 25, counts
    assert counts["test.jsonl"] >= 50, counts


def test_schema_valid():
    for name in ("train.jsonl", "valid.jsonl", "test.jsonl"):
        for r in load(name):
            check_record(r)


def test_no_cross_split_state_leakage():
    seen = {}
    for name in ("train.jsonl", "valid.jsonl", "test.jsonl"):
        for r in load(name):
            assert r["state"] not in seen, f"duplicate state in {name}: {r['state'][:60]!r}"
            seen[r["state"]] = name
