"""Structured-output validation for Tell-4B.

The model must return exactly one option label (e.g. "B").
This module never invents a label: if no valid option can be safely
extracted it returns ``{"label": None, "status": "invalid_output"}``.
"""

from __future__ import annotations

import re


def _normalize(text: str) -> str:
    return (text or "").strip().upper()


def extract_label(raw_text: str | None, allowed_labels: list[str]) -> dict:
    """Extract a valid option label from raw model output.

    Returns a dict: ``{"label": <str|None>, "status": <str>, "raw": <str>}``
    where status is one of:
      - ``"ok"``: output was exactly the label (modulo case/whitespace).
      - ``"extracted"``: exactly one distinct allowed label was found
        inside a longer string (e.g. ``"The answer is B because..."`` -> B).
      - ``"invalid_output"``: zero or multiple conflicting labels found.
    """
    raw = raw_text if raw_text is not None else ""
    allowed = [_normalize(a) for a in (allowed_labels or [])]
    allowed_set = set(allowed)
    norm = _normalize(raw)

    if norm in allowed_set:
        return {"label": norm, "status": "ok", "raw": raw}

    if not allowed:
        return {"label": None, "status": "invalid_output", "raw": raw}

    # Find standalone allowed labels (word-boundary match).
    pattern = r"\b(" + "|".join(re.escape(a) for a in allowed) + r")\b"
    found = re.findall(pattern, norm)
    distinct = list(dict.fromkeys(found))  # preserve order, dedupe

    if len(distinct) == 1:
        return {"label": distinct[0], "status": "extracted", "raw": raw}
    return {"label": None, "status": "invalid_output", "raw": raw}


def validate_output(raw_text: str | None, allowed_labels: list[str]) -> dict:
    """Alias for :func:`extract_label` (kept for a stable public name)."""
    return extract_label(raw_text, allowed_labels)
