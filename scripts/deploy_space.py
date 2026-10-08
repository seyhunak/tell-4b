#!/usr/bin/env python3
"""Create (or update) the Tell-4B Gradio Space on the Hugging Face Hub.

Uploads ``space/*`` plus ``src/tell/*`` into the Space repo. The Space needs
``src/tell/prompt.py`` because app.py imports the SAME prompt template that
training and pytest use — a copy pasted into the Space would silently drift.

Usage:
    export HF_TOKEN="hf_..."
    python scripts/deploy_space.py                       # -> seyhunak/tell-4b-demo
    python scripts/deploy_space.py --space-id you/tell   # -> custom Space
    python scripts/deploy_space.py --hardware zero-a10g  # default

Hardware is set at creation time, which matters: since July 2026 a Gradio Space
created on ``cpu-basic`` requires a paid plan, while a good-standing free account
(verified email, older than 30 days) may still host up to 2 ZeroGPU Spaces. Pass
``--hardware`` to change it; ``huggingface_hub`` supports these on creation only,
so a later change needs the Settings page.
"""

from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SPACE_DIR = ROOT / "space"
TELL_SRC = ROOT / "src" / "tell"

SPACE_FILES = ("app.py", "README.md", "requirements.txt")
# The whole package: `tell/__init__.py` imports .prompt, .inference and
# .evaluation, so shipping a subset breaks `from tell.prompt import ...`.
TELL_FILES = ("__init__.py", "prompt.py", "inference.py", "evaluation.py")


def main() -> int:
    ap = argparse.ArgumentParser(description="Deploy the Tell-4B Gradio Space.")
    ap.add_argument("--space-id", default="seyhunak/tell-4b-demo")
    ap.add_argument("--hardware", default="zero-a10g",
                    help="Space hardware, e.g. zero-a10g, cpu-basic, t4-small.")
    ap.add_argument("--private", action="store_true")
    ap.add_argument("--dry-run", action="store_true", help="Only print what would be uploaded.")
    args = ap.parse_args()

    missing = [f for f in SPACE_FILES if not (SPACE_DIR / f).exists()]
    missing += [f"src/tell/{f}" for f in TELL_FILES if not (TELL_SRC / f).exists()]
    if missing:
        print(f"ERROR: missing files: {', '.join(missing)}", file=sys.stderr)
        return 2

    try:
        from huggingface_hub import HfApi
    except ImportError:
        print("ERROR: pip install huggingface_hub", file=sys.stderr)
        return 2

    api = HfApi()
    repo_type = "space"

    try:
        info = api.repo_info(args.space_id, repo_type=repo_type)
        hardware = (info.sdk or {}) if isinstance(info.sdk, dict) else {}
        print(f"Space exists: {args.space_id}")
        print(f"  hardware now: {hardware.get('hardware', {}).get('current', 'unknown')}"
              f"  (to change: Space -> Settings -> Hardware)")
    except Exception:
        if args.dry_run:
            print(f"[dry-run] would create space {args.space_id} "
                  f"(sdk=gradio, hardware={args.hardware})")
        else:
            api.create_repo(
                args.space_id,
                repo_type=repo_type,
                space_sdk="gradio",
                space_hardware=args.hardware,
                private=args.private,
                exist_ok=True,
            )
            print(f"Created space: {args.space_id} (hardware={args.hardware})")

    # Stage the exact tree the Space repo should contain.
    with tempfile.TemporaryDirectory() as tmp:
        stage = Path(tmp)
        for name in SPACE_FILES:
            (stage / name).write_bytes((SPACE_DIR / name).read_bytes())
        (stage / "src" / "tell").mkdir(parents=True)
        for name in TELL_FILES:
            (stage / "src" / "tell" / name).write_bytes((TELL_SRC / name).read_bytes())

        staged = sorted(str(p.relative_to(stage)) for p in stage.rglob("*") if p.is_file())
        print("Uploading:")
        for name in staged:
            print(f"  {name}")

        if args.dry_run:
            return 0

        api.upload_folder(
            folder_path=str(stage),
            repo_id=args.space_id,
            repo_type=repo_type,
            commit_message="Deploy Tell-4B Space",
        )

    print(f"\nDone -> https://huggingface.co/spaces/{args.space_id}")
    print("The build downloads ~9.3 GB of Qwen3.5-4B weights; startup allows 30 minutes.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())