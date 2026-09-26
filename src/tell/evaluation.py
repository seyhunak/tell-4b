"""Metric computation for Tell-4B evaluation (pure Python, no torch)."""

from __future__ import annotations

from collections import Counter


def compute_metrics(expected: list[str | None], predicted: list[str | None]) -> dict:
    """Compute exact accuracy, invalid rate, per-class accuracy, confusion.

    ``predicted`` may contain None for invalid outputs.
    Returns a JSON-serialisable dict.
    """
    if len(expected) != len(predicted):
        raise ValueError("expected and predicted must have the same length")
    n = len(expected)
    if n == 0:
        return {
            "n": 0,
            "accuracy": 0.0,
            "invalid_rate": 0.0,
            "invalid_count": 0,
            "per_class": {},
            "confusion": {},
            "labels": [],
        }

    correct = sum(1 for e, p in zip(expected, predicted) if p is not None and p == e)
    invalid = sum(1 for p in predicted if p is None)
    labels = sorted({e for e in expected if e is not None})

    per_class: dict[str, dict] = {}
    for lab in labels:
        idx = [i for i, e in enumerate(expected) if e == lab]
        c = sum(1 for i in idx if predicted[i] == lab)
        per_class[lab] = {
            "n": len(idx),
            "correct": c,
            "accuracy": (c / len(idx)) if idx else 0.0,
        }

    confusion: dict[str, dict[str, int]] = {}
    for lab in labels:
        row: Counter = Counter()
        for i, e in enumerate(expected):
            if e == lab:
                row[predicted[i] if predicted[i] is not None else "<invalid>"] += 1
        confusion[lab] = dict(row)

    return {
        "n": n,
        "accuracy": correct / n,
        "invalid_rate": invalid / n,
        "invalid_count": invalid,
        "per_class": per_class,
        "confusion": confusion,
        "labels": labels,
    }


def format_report(metrics: dict, title: str = "Tell-4B Evaluation") -> str:
    """Format metrics in the README's human-readable style."""
    lines = [title, "-" * len(title), ""]
    lines.append(f"Examples:           {metrics['n']}")
    lines.append(f"Accuracy:           {metrics['accuracy'] * 100:.1f}%")
    lines.append(f"Invalid outputs:    {metrics['invalid_rate'] * 100:.1f}%")
    lines.append("")
    if metrics["per_class"]:
        lines.append("Per-class:")
        for lab in sorted(metrics["per_class"]):
            acc = metrics["per_class"][lab]["accuracy"] * 100
            lines.append(f"{lab:<20}{acc:.1f}%")
    return "\n".join(lines)
