"""Render the Tell-4B explainer to frames, then encode with ffmpeg.

Usage:
    python3 media/build_video.py            # full render
    python3 media/build_video.py --preview  # few frames only, fast
"""
from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from render_core import W, H, FPS, Scene, progress_bar, vignette
from scenes_a import (s1_cold_open, s2_problem, s3_answer_only, s4_training,
                      base_frame)
from scenes_b import s5_samples, s6_robust, s7_results, s8_close

OUT = Path(__file__).parent / "build"
FRAMES = OUT / "frames"

SCENES = [
    Scene("cold-open", 4.2, s1_cold_open),
    Scene("problem", 6.4, s2_problem),
    Scene("contract", 6.2, s3_answer_only),
    Scene("training", 6.6, s4_training),
    Scene("samples", 9.8, s5_samples),   # 4 slides x 2.3s + settle
    Scene("robustness", 6.4, s6_robust),
    Scene("results", 6.6, s7_results),
    Scene("close", 6.6, s8_close),
]


def total_frames() -> int:
    return sum(int(s.dur * FPS) for s in SCENES)


def render(preview: bool = False) -> Path:
    FRAMES.mkdir(parents=True, exist_ok=True)
    for old in FRAMES.glob("*.png"):
        old.unlink()
    total = total_frames()
    step = max(1, FPS // 2) if preview else 1
    done = 0
    t0 = time.time()
    idx = 0
    for sc in SCENES:
        n = int(sc.dur * FPS)
        for i in range(n):
            t = i / FPS
            if i % step and not preview:
                done += 1
                continue
            img, d = base_frame(t)
            sc.fn(t, img, d)
            vignette(img)
            img.convert("RGB").save(FRAMES / f"{idx:05d}.png")
            idx += 1
            done += 1
        frac = done / total
        bar = "#" * int(frac * 34)
        print(f"\r  [{bar:<34}] {frac*100:5.1f}%  {sc.name:<11}"
              f" {time.time()-t0:5.1f}s", end="", flush=True)
    print()
    return OUT


def encode() -> Path:
    mp4 = OUT / "tell-4b-explainer.mp4"
    cmd = [
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
        "-framerate", str(FPS), "-i", str(FRAMES / "%05d.png"),
        "-c:v", "libx264", "-preset", "slow", "-crf", "19",
        "-pix_fmt", "yuv420p", "-movflags", "+faststart",
        "-vf", "scale=1920:1080:flags=lanczos",
        str(mp4),
    ]
    subprocess.run(cmd, check=True)
    return mp4


def make_thumb() -> Path:
    """Compose a 1280x720 thumbnail from rendered frames (no ffmpeg needed).

    Uses the cold-open hero frame and adds title bars, so the still reads as
    a title card rather than a random mid-animation frame.
    """
    from PIL import Image, ImageDraw
    from render_core import TEXT, ACCENT, sans, text_c
    # late cold-open frame: all three rings fully expanded (radius 300 about
    # centre 960,372), so the crop contains the whole motif, none clipped.
    for cand in ("00090.png", "00045.png", "00000.png"):
        if (FRAMES / cand).exists():
            src = FRAMES / cand
            break
    else:
        src = sorted(FRAMES.glob("*.png"))[0]
    # The cold-open frame already carries the wordmark and tagline, so the
    # thumbnail is that frame verbatim — no extra overlay text to duplicate it.
    im = Image.open(src).convert("RGB").resize((1280, 720), Image.LANCZOS)
    out = OUT / "tell-4b-thumb.png"
    im.save(out)
    return out


if __name__ == "__main__":
    prev = "--preview" in sys.argv
    if prev:
        OUT.mkdir(parents=True, exist_ok=True)
        i = 0
        for sc in SCENES:
            for frac in (0.35, 0.7, 0.95):
                t = sc.dur * frac
                img, d = base_frame(t)
                sc.fn(t, img, d)
                vignette(img)
                img.convert("RGB").resize((960, 540)).save(
                    OUT / f"preview_{i:02d}_{sc.name}.png")
                i += 1
        print(f"preview frames -> {OUT}")
        sys.exit(0)
    if "--thumb" in sys.argv:
        print(f"thumb -> {make_thumb()}")
        sys.exit(0)
    print(f"rendering {total_frames()} frames at {W}x{H} ...")
    render()
    mp4 = encode()
    print(f"\nvideo -> {mp4}  ({mp4.stat().st_size/1e6:.1f} MB)")
    print(f"thumb -> {make_thumb()}")
