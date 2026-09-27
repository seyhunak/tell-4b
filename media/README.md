# Tell-4B explainer video

Animated product explainer rendered from source with **PIL only** — no After
Effects, no external assets, no third-party Python packages beyond Pillow.
Every frame is drawn programmatically, so the video is reproducible by
re-running the script.

## Output

```text
build/tell-4b-explainer.mp4    1920x1080, 30 fps, 52.8 s, H.264 (yuv420p)
build/tell-4b-thumb.png        1280x720 still for the video thumbnail
```

## Build

```bash
python3 build_video.py            # render all 1584 frames + encode (~85 s)
python3 build_video.py --preview  # 3 small frames per scene, fast
python3 build_video.py --thumb    # rebuild the thumbnail only
```

Requires `pillow` and `ffmpeg` (with `libx264`) on `PATH`.

## Files

| File | Role |
|---|---|
| `render_core.py` | Palette, easing, text/layout and shape primitives |
| `scenes_a.py` | Scenes 1-4 + the shared facts table (`SAMPLES`, `CATS`) |
| `scenes_b.py` | Scenes 5-8 (samples, robustness, results, close) |
| `build_video.py` | Scene timeline, frame renderer, ffmpeg encode, thumbnail |

## Scene timeline

| # | Scene | Length | Content |
|---|---|---|---|
| 1 | cold-open | 4.2 s | "B" motif, tagline |
| 2 | problem | 6.4 s | Invoice vs PO, the open question |
| 3 | contract | 6.2 s | STATE/QUESTION/OPTIONS -> ANSWER |
| 4 | training | 6.6 s | Qwen3.5-4B -> LoRA -> Tell-4B, answer-only loss |
| 5 | samples | 9.8 s | 4 real tickets, crossfaded, one label each |
| 6 | robustness | 6.4 s | Prompt injection ignored (T-045) |
| 7 | results | 6.6 s | 96.7% vs 98.3% base, plus the honest caveat |
| 8 | close | 6.6 s | Demo breakdown, 50/50, outro card |

## Data provenance

Nothing in the video is invented. Every figure is taken from the repo:

- **Ticket text and expected labels** — copied verbatim from `demo/lib/tickets.ts`
  (retrieved from the live `/api/tickets` endpoint).
- **50 / 50 correct, 0 invalid** — measured by replaying all 50 tickets through
  the running `POST /api/classify` endpoint.
- **Category split (24 / 10 / 10 / 6)** — from the same live endpoint.
- **96.7% Tell vs 98.3% base, 0.0% vs 1.7% invalid** — the smoke run documented
  in `README.md` section 11.
- **LoRA r=16, alpha=32; 120/30/60 train/valid/test** — `configs/mac_m3_48gb.yaml`
  and `data/*.jsonl`.

The video states plainly that the base model scores higher on this dataset;
that caveat is in the source README and is not hidden in the cut.
