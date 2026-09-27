"""Frame-rendering toolkit for the Tell-4B explainer video.

Pure PIL + stdlib. No external assets: every glyph, box and chart on screen is
drawn here, so the render is fully reproducible from source.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from PIL import Image, ImageDraw, ImageFont

W, H = 1920, 1080
FPS = 30

# ---- palette (dark, high contrast) ----
BG = (11, 15, 25)
BG_SOFT = (18, 24, 39)
PANEL = (23, 31, 50)
PANEL_HI = (31, 42, 66)
STROKE = (44, 56, 82)
STROKE_HI = (70, 88, 124)
TEXT = (240, 244, 250)
MUTED = (150, 163, 186)
DIM = (104, 116, 140)
ACCENT = (94, 234, 212)      # teal - brand
ACCENT_2 = (129, 140, 248)   # indigo
WARN = (251, 191, 36)        # amber - injection
GOOD = (74, 222, 128)        # green - correct
BAD = (248, 113, 113)        # red
MONO = "/System/Library/Fonts/Supplemental/Courier New Bold.ttf"
SANS = "/System/Library/Fonts/Supplemental/Arial.ttf"
SANS_B = "/System/Library/Fonts/Supplemental/Arial Bold.ttf"
BLACK = "/System/Library/Fonts/Supplemental/Arial Black.ttf"

_fc: dict = {}


def font(path: str, size: int) -> ImageFont.FreeTypeFont:
    k = (path, size)
    if k not in _fc:
        _fc[k] = ImageFont.truetype(path, size)
    return _fc[k]


def sans(size: int, bold: bool = False, black: bool = False) -> ImageFont.FreeTypeFont:
    if black:
        return font(BLACK, size)
    return font(SANS_B if bold else SANS, size)


def mono(size: int, bold: bool = True) -> ImageFont.FreeTypeFont:
    return font(MONO, size)


# ---- easing ----
def clamp(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, x))


def ease_out(x: float) -> float:
    x = clamp(x)
    return 1 - (1 - x) ** 3


def ease_in_out(x: float) -> float:
    x = clamp(x)
    return 4 * x * x * x if x < 0.5 else 1 - ((-2 * x + 2) ** 3) / 2


def ease_out_back(x: float) -> float:
    x = clamp(x)
    c1, c3 = 1.70158, 2.70158
    return 1 + c3 * (x - 1) ** 3 + c1 * (x - 1) ** 2


def seg(t: float, start: float, dur: float) -> float:
    """Normalised 0..1 progress of a sub-animation starting at `start`."""
    if dur <= 0:
        return 1.0
    return clamp((t - start) / dur)



# ---- text helpers ----
def tw(d: ImageDraw.ImageDraw, text: str, f) -> int:
    return int(d.textlength(text, font=f))


def text_c(d, xy, s, f, fill=TEXT, anchor="la", tracking: int = 0):
    """Draw text; `tracking` adds letter-spacing (px) for small-caps labels."""
    if not tracking:
        d.text(xy, s, font=f, fill=fill, anchor=anchor)
        return
    total = sum(tw(d, ch, f) + tracking for ch in s) - tracking
    x, y = xy
    if anchor[0] == "m":
        x -= total / 2
    elif anchor[0] == "r":
        x -= total
    va = "a" if len(anchor) > 1 and anchor[1] == "a" else anchor[1]
    for ch in s:
        d.text((x, y), ch, font=f, fill=fill, anchor="l" + va)
        x += tw(d, ch, f) + tracking


def wrap(d, s: str, f, max_w: int) -> list[str]:
    words, lines, cur = s.split(), [], ""
    for w in words:
        t = (cur + " " + w).strip()
        if tw(d, t, f) <= max_w or not cur:
            cur = t
        else:
            lines.append(cur)
            cur = w
    if cur:
        lines.append(cur)
    return lines


def para(d, xy, s, f, fill, max_w, lh, tracking: int = 0) -> int:
    x, y = xy
    for ln in wrap(d, s, f, max_w):
        text_c(d, (x, y), ln, f, fill, tracking=tracking)
        y += lh
    return y



# ---- primitives ----
def rrect(d, box, r, fill=None, outline=None, width=1):
    d.rounded_rectangle([int(v) for v in box], radius=r, fill=fill,
                        outline=outline, width=width)


def panel(d, box, r=18, fill=PANEL, outline=STROKE, width=2):
    rrect(d, box, r, fill=fill, outline=outline, width=width)


def tint(color, f: float):
    """Blend a colour toward the background by factor f (0..1)."""
    return tuple(int(BG[i] + (color[i] - BG[i]) * f) for i in range(3))


def arrow(d, p0, p1, color, width=5, head=16, head_frac=0.42):
    x0, y0 = p0
    x1, y1 = p1
    d.line([x0, y0, x1, y1], fill=color, width=width)
    ang = math.atan2(y1 - y0, x1 - x0)
    bx, by = x1 - (x1 - x0) * head_frac, y1 - (y1 - y0) * head_frac
    p1a = (bx + head * math.cos(ang - 0.42), by + head * math.sin(ang - 0.42))
    p1b = (bx + head * math.cos(ang + 0.42), by + head * math.sin(ang + 0.42))
    d.polygon([(x1, y1), p1a, p1b], fill=color)


def dot_glow(img, xy, color, r=8, a=255, halo=3.0):
    x, y = xy
    for i in range(int(halo), 0, -1):
        rr = r * (1 + i * 0.85)
        ov = Image.new("RGBA", img.size, (0, 0, 0, 0))
        ImageDraw.Draw(ov).ellipse([x - rr, y - rr, x + rr, y + rr],
                                   fill=color + (int(30 / i),))
        img.alpha_composite(ov)
    ov = Image.new("RGBA", img.size, (0, 0, 0, 0))
    ImageDraw.Draw(ov).ellipse([x - r, y - r, x + r, y + r], fill=color + (a,))
    img.alpha_composite(ov)


def vgrad(img, top=BG, bottom=(8, 11, 20)):
    d = ImageDraw.Draw(img)
    for y in range(H):
        t = y / (H - 1)
        d.line([0, y, W, y],
               fill=tuple(int(top[i] + (bottom[i] - top[i]) * t)
                          for i in range(3)))


def vignette(img, strength=0.30):
    """Soft corner darkening via a blurred mask (no banding artefacts)."""
    import PIL.ImageFilter as F
    mask = Image.new("L", (W, H), 0)
    md = ImageDraw.Draw(mask)
    md.ellipse([-W * 0.30, -H * 0.42, W * 1.30, H * 1.42], fill=255)
    mask = mask.filter(F.GaussianBlur(220))
    dark = Image.new("RGBA", (W, H), (0, 0, 0, int(255 * strength)))
    dark.putalpha(mask.point(lambda v: int(v * strength)))
    img.alpha_composite(dark)



# ---- reusable widgets ----
def kicker(d, xy, s, color=ACCENT, size=22):
    x, y = xy
    d.rounded_rectangle([x, y + 8, x + 46, y + 12], radius=2, fill=color)
    text_c(d, (x + 60, y), s.upper(), sans(size, bold=True), color, tracking=4)


def title(d, s, size=76, y=118, color=TEXT, sub=None, sub_color=MUTED):
    text_c(d, (W // 2, y), s, sans(size, bold=True), color, anchor="ma")
    if sub:
        text_c(d, (W // 2, y + int(size * 1.15)), sub, sans(28), sub_color,
               anchor="ma")


def progress_bar(img, d, t: float, frac: float, y=H - 46, color=ACCENT):
    x0, x1 = 120, W - 120
    d.rounded_rectangle([x0, y, x1, y + 5], radius=3, fill=(30, 38, 58))
    f = clamp(frac)
    if f > 0:
        d.rounded_rectangle([x0, y, x0 + (x1 - x0) * f, y + 5], radius=3,
                            fill=color)
    for i in range(7):
        cx = x0 + (x1 - x0) * (i / 6)
        d.ellipse([cx - 3, y + 1, cx + 3, y + 7], fill=(46, 58, 84))
    cx = x0 + (x1 - x0) * f
    d.ellipse([cx - 8, y - 5, cx + 8, y + 11], fill=color)


def label_chip(d, xy, txt, color, size=20, padx=16, pady=9, fill=None):
    f = sans(size, bold=True)
    w = tw(d, txt, f) + padx * 2
    h = size + pady * 2
    x, y = xy
    bg = fill if fill is not None else tint(color, 0.16)
    rrect(d, [x, y, x + w, y + h], h // 2, fill=bg, outline=color, width=2)
    text_c(d, (x + w / 2, y + h / 2 + 1), txt, f, color, anchor="mm")
    return w, h


def count_up(d, xy, target: float, f, fill, t: float, dur=0.9, suffix="",
             prefix="", anchor="la"):
    v = target * ease_out(clamp(t / dur))
    dec = 0 if float(target).is_integer() else 1
    d.text(xy, f"{prefix}{v:.{dec}f}{suffix}", font=f, fill=fill, anchor=anchor)
    return v


@dataclass
class Scene:
    name: str
    dur: float
    fn: object
