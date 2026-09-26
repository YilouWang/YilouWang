"""Calibration image: every HUD region drawn on a real frame.

Used by ``tft-advisor calibrate``: take a screenshot during a planning phase,
draw all regions, and let the player check that each box covers its HUD
element on their resolution / UI scale. Also runnable directly:

    python -m tft_advisor.capture.calibrate [image.png ...] [--out DIR] [--layout anchored]
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence

from PIL import Image, ImageDraw, ImageFont

from .regions import BASE_ASPECT, REGION_LABELS, REGIONS, current_layout, region_box, viewport

# Distinct, readable on both the dark HUD and the bright arena.
PALETTE: tuple[tuple[int, int, int], ...] = (
    (255, 215, 0),
    (0, 229, 255),
    (255, 64, 129),
    (118, 255, 3),
    (255, 145, 0),
    (213, 0, 249),
    (0, 230, 118),
    (41, 121, 255),
    (255, 23, 68),
    (255, 255, 255),
    (29, 233, 182),
    (255, 234, 0),
    (234, 128, 252),
    (100, 255, 218),
)

_CJK_FONT_CANDIDATES = (
    r"C:\Windows\Fonts\msyh.ttc",
    r"C:\Windows\Fonts\msyhbd.ttc",
    r"C:\Windows\Fonts\simhei.ttf",
    r"C:\Windows\Fonts\simsun.ttc",
    "/System/Library/Fonts/PingFang.ttc",
    "/System/Library/Fonts/STHeiti Medium.ttc",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/noto-cjk/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
    "/usr/share/fonts/wqy-microhei/wqy-microhei.ttc",
)


def _load_font(size: int) -> tuple[Any, bool]:
    """(font, supports_chinese). Falls back to Pillow's built-in font."""
    for path in _CJK_FONT_CANDIDATES:
        if os.path.isfile(path):
            try:
                return ImageFont.truetype(path, size), True
            except Exception:
                continue
    try:
        return ImageFont.load_default(size=size), False  # Pillow >= 10.1 (scalable)
    except TypeError:
        return ImageFont.load_default(), False


def _overlaps(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    return a[0] < b[2] and b[0] < a[2] and a[1] < b[3] and b[1] < a[3]


def draw_regions(
    img: Image.Image,
    names: Optional[Iterable[str]] = None,
    *,
    layout: Optional[str] = None,
    chinese: bool = True,
) -> Image.Image:
    """Copy of ``img`` with a labeled rectangle for every region (same size as ``img``).

    When the frame is not 16:9 the assumed 16:9 HUD area is outlined too.
    Chinese labels are added when a CJK font is available on the machine.
    """
    out = img.convert("RGB") if img.mode != "RGB" else img.copy()
    w, h = out.size
    draw = ImageDraw.Draw(out)
    line = max(2, round(min(w, h) / 360))
    font, has_cjk = _load_font(max(12, round(h / 55)))
    use_zh = chinese and has_cjk

    ox, oy, vw, vh = viewport((w, h))
    if abs(w / h - BASE_ASPECT) > 0.01:
        draw.rectangle(
            [round(ox), round(oy), round(ox + vw) - 1, round(oy + vh) - 1], outline=(255, 255, 255), width=max(1, line // 2)
        )

    placed: list[tuple[int, int, int, int]] = []
    for i, name in enumerate(list(names) if names is not None else list(REGIONS)):
        color = PALETTE[i % len(PALETTE)]
        x0, y0, x1, y1 = region_box((w, h), name, layout=layout)
        draw.rectangle([x0, y0, x1 - 1, y1 - 1], outline=color, width=line)

        label = f"{name} {REGION_LABELS.get(name, '')}".strip() if use_zh else name
        tb = draw.textbbox((0, 0), label, font=font)
        tw, th = tb[2] - tb[0] + 2 * line, tb[3] - tb[1] + 2 * line
        lx = min(max(0, x0 + line), max(0, w - tw))
        ly = min(max(0, y0 + line), max(0, h - th))
        box = (lx, ly, lx + tw, ly + th)
        for _ in range(len(placed) + 1):  # shift down until it does not cover another label
            hit = next((p for p in placed if _overlaps(box, p)), None)
            if hit is None:
                break
            ly = hit[3] + 1
            if ly + th > h:
                ly = max(0, y0 - th)
            box = (lx, ly, lx + tw, ly + th)
        placed.append(box)
        draw.rectangle(box, fill=(0, 0, 0))
        draw.text((lx + line - tb[0], ly + line - tb[1]), label, fill=color, font=font)

    title = f"{w}x{h}  layout={current_layout(layout)}"
    if use_zh:
        title += "  请确认每个框都盖住了对应的界面元素"
    tb = draw.textbbox((0, 0), title, font=font)
    ty = h - (tb[3] - tb[1]) - 3 * line
    draw.rectangle([0, ty - line, tb[2] - tb[0] + 4 * line, h], fill=(0, 0, 0))
    draw.text((2 * line - tb[0], ty - tb[1]), title, fill=(255, 255, 255), font=font)
    return out


def save_calibration(img: Image.Image, directory: str | Path, *, layout: Optional[str] = None) -> tuple[Path, Path]:
    """Save the raw frame and the annotated calibration image; returns (raw, annotated)."""
    from .screen import save_frame

    raw = save_frame(img, directory, "calibrate_raw")
    annotated = save_frame(draw_regions(img, layout=layout), directory, "calibrate")
    return raw, annotated


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Draw TFT HUD regions on a screenshot for calibration")
    parser.add_argument("images", nargs="*", help="screenshots to annotate (default: capture the screen now)")
    parser.add_argument("--out", default="~/.tft_advisor/captures", help="output directory")
    parser.add_argument("--layout", default=None, choices=["inscribed", "anchored"])
    args = parser.parse_args(argv)

    frames: list[Image.Image] = []
    if args.images:
        from .screen import FileCapturer

        cap = FileCapturer(args.images)
        while (frame := cap.grab()) is not None:
            frames.append(frame)
        if cap.missing:
            print(f"找不到这些文件：{', '.join(cap.missing)}")
    else:
        from ..config import CaptureConfig, load_config
        from .screen import ScreenCapturer

        try:  # the user's window title / monitor settings, when a config file exists
            capture_cfg = load_config().capture
        except Exception:  # noqa: BLE001 - calibration must work with a broken config too
            capture_cfg = CaptureConfig()
        with ScreenCapturer(capture_cfg) as cap:
            frame = cap.grab()
            if frame is None:
                print(cap.last_error or "截图失败：原因未知")
                return 1
            frames.append(frame)
    if not frames:
        print("没有可用的图片")
        return 1
    for frame in frames:
        _raw, annotated = save_calibration(frame, args.out, layout=args.layout)
        print(f"校准图已保存：{annotated}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
