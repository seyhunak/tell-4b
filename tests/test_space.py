"""Tests for the Gradio Space (space/app.py).

The Space app imports torch, gradio and peft at module scope, none of which are
guaranteed on a contributor's machine. The two pieces of logic worth guarding
here — option parsing and the scoring index arithmetic — are pure, so they are
loaded out of app.py's source and exercised without those imports.
"""

import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
APP_PY = ROOT / "space" / "app.py"


class _FakeGradio:
    """Stand-in for the two gradio symbols app.py's helpers touch."""

    class Error(Exception):
        pass


def _load_from_app(start: str, end: str):
    source = APP_PY.read_text()
    assert start in source, f"{start!r} is gone from space/app.py — update this test"
    fragment = source.split(start)[1].split(end)[0]
    namespace = {"re": re, "gr": _FakeGradio}
    exec(compile(f"{start}{fragment}", str(APP_PY), "exec"), namespace)  # noqa: S102
    return namespace[start.split("(")[0].removeprefix("def ")]


parse_options = _load_from_app(
    "def parse_options(text: str)",
    "# --- model loading",
)


def _tail_layout(prompt_ids, full_ids):
    """Mirror of the common-prefix + tail bookkeeping in score_options()."""
    shared = 0
    for a, b in zip(prompt_ids, full_ids):
        if a != b:
            break
        shared += 1
    return shared, len(full_ids) - shared, full_ids[shared:]


def test_parse_options_newline_and_pipe_are_equivalent():
    lines = parse_options("A: match\nB: mismatch\nC: needs_review")
    pipes = parse_options("A: match | B: mismatch | C: needs_review")
    assert lines == pipes
    assert [o["label"] for o in lines] == ["A", "B", "C"]
    assert [o["key"] for o in lines] == ["match", "mismatch", "needs_review"]


def test_parse_options_defaults_labels_and_keeps_descriptions():
    opts = parse_options("match|mismatch")
    assert [(o["label"], o["key"]) for o in opts] == [("A", "match"), ("B", "mismatch")]
    # A description that was not supplied falls back to the key, so the rendered
    # OPTIONS block is byte-identical to what training built.
    assert all(o["description"] == o["key"] for o in opts)

    described = parse_options("A: match: invoice equals PO\nB: mismatch: amounts differ")
    assert described[0]["description"] == "invoice equals PO"
    assert described[1]["description"] == "amounts differ"


def test_parse_options_rejects_unusable_input():
    with pytest.raises(_FakeGradio.Error):
        parse_options("")
    with pytest.raises(_FakeGradio.Error):
        parse_options("A: only one")
    with pytest.raises(_FakeGradio.Error):
        parse_options("A: match\nA: mismatch")  # duplicate label


@pytest.mark.parametrize(
    "prompt_ids, full",
    [
        ([1, 2, 3, 4, 5], [1, 2, 3, 4, 5, 90]),          # label is its own token
        ([1, 2, 3, 4], [1, 2, 3, 4, 90]),                 # "\n" + "A" merged into one
        ([1, 2, 3, 4, 5, 6, 7], [1, 2, 3, 4, 5, 6, 7, 90, 91]),  # multi-token label
    ],
)
def test_scoring_window_covers_exactly_the_answer_tokens(prompt_ids, full):
    shared, n, targets = _tail_layout(prompt_ids, full)
    assert shared == len(prompt_ids)
    assert n == len(targets) == len(full) - shared

    keep = max(n + 1, 2)
    width = len(full)
    # Left padding is what every row shares in one batch; the answer always ends
    # at width-1. logits_to_keep=keep keeps absolute logits [width-keep, width-1],
    # and the logit at index p-1 is the one predicting the token at position p.
    answer_positions = list(range(width - n, width))
    expected_window = [p - 1 - (width - keep) for p in answer_positions]
    assert expected_window == list(range(keep - n - 1, keep - 1))


def test_scoring_window_survives_a_boundary_merge():
    """If BPE fuses the prompt's last token with the label, the tail starts early."""
    prompt_ids = [1, 2, 3, 4]
    full = [1, 2, 3, 7, 90]  # token 4 became part of the answer token
    shared, n, targets = _tail_layout(prompt_ids, full)
    assert shared == 3
    assert (n, targets) == (2, [7, 90])

    keep = n + 1
    width = len(full)
    answer_positions = list(range(width - n, width))
    assert [p - 1 - (width - keep) for p in answer_positions] == list(
        range(keep - n - 1, keep - 1)
    )


def test_space_prompt_matches_training_prompt():
    """The Space must render the exact prompt training used, or the model drifts."""
    sys.path.insert(0, str(ROOT / "src"))
    from tell.prompt import build_prompt

    options = parse_options("A: match\nB: mismatch\nC: needs_review")
    prompt = build_prompt("Invoice is 1250, PO is 1000.", "Classify this.", options)
    assert "OPTIONS:\nA: match\nB: mismatch\nC: needs_review\n\nANSWER:" in prompt


def test_space_readme_declares_zero_gpu_and_links_the_model():
    readme = (ROOT / "space" / "README.md").read_text()
    frontmatter = readme.split("---")[1]
    assert "sdk: gradio" in frontmatter
    assert "app_file: app.py" in frontmatter
    assert "suggested_hardware: zero-a10g" in frontmatter
    assert "seyhunak/tell-4b" in frontmatter
    assert "startup_duration_timeout" in frontmatter