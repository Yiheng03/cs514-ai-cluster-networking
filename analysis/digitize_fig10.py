"""Read the bar values of Figure 10 from the paper PDF.

Usage: python3 analysis/digitize_fig10.py nsdi25-wang-xizheng-simai.pdf
Writes data/fig10_digitized.csv (seconds). Requires pdfimages (poppler).
"""

import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

Image.MAX_IMAGE_PIXELS = None
OUT = Path(__file__).resolve().parent.parent / "data" / "fig10_digitized.csv"


def load_figure(pdf):
    # Figure 10 is the only raster on PDF page 11.
    tmp = Path(tempfile.mkdtemp())
    subprocess.run(["pdfimages", "-f", "11", "-l", "11", "-png", pdf,
                    str(tmp / "im")], check=True)
    imgs = sorted(tmp.glob("im-*.png"))
    rgb = np.array(Image.open(imgs[0]).convert("RGB")).astype(int)
    alpha = np.array(Image.open(imgs[1]))
    rgb[alpha < 128] = 255
    return rgb


def tick_rows(dark, x0, x1):
    idx = np.where(dark[:, x0:x1].all(axis=1))[0]
    groups = np.split(idx, np.where(np.diff(idx) > 1)[0] + 1)
    return [g.mean() for g in groups if len(g)]


def segment_top(mask, y_top, y0, a, b):
    # Median over columns so the black hatching lines do not matter.
    sub = mask[y_top:y0, a:b]
    has = sub.any(axis=0)
    if has.sum() < 20:
        return None
    first = sub.argmax(axis=0)[has] + y_top
    return float(np.median(first))


def main(pdf):
    img = load_figure(pdf)
    R, G, B = img[..., 0], img[..., 1], img[..., 2]
    dark = img.sum(axis=2) < 300
    colors = {
        "tp": (B > 200) & (R < 200) & (B - R > 30),
        "dp_top": (G > 200) & (R < 215) & (B < 180) & (G - B > 40),
        "total": (R > 220) & (G < 190) & (B < 190) & (R - G > 50),
    }
    # Pixel geometry measured on the raster. Bars are 330 px wide, 18 px borders.
    panels = {
        "H100": dict(axis=846, ticks=[2e6, 1e6, 0.0], x_range=(850, 6000)),
        "H20": dict(axis=6515, ticks=[6e6, 4e6, 2e6, 0.0], x_range=(6530, 11950)),
    }
    series_order = ["no_comp", "simai_model", "simai", "groundtruth"]
    rows = []
    for gpu, p in panels.items():
        ticks = tick_rows(dark, p["axis"] - 30, p["axis"] - 2)
        y0 = ticks[-1]
        px_per_us = (ticks[-1] - ticks[0]) / (p["ticks"][0] - p["ticks"][-1])
        any_col = (colors["tp"] | colors["dp_top"] | colors["total"])
        cols = np.where(any_col[800:int(y0) - 20, p["x_range"][0]:p["x_range"][1]]
                        .any(axis=0))[0] + p["x_range"][0]
        groups = np.split(cols, np.where(np.diff(cols) > 3)[0] + 1)
        bars = []
        for g in groups:
            if len(g) < 60:
                continue
            n = max(1, round((g[-1] - g[0] + 18) / 348))   # touching bars
            w = (g[-1] - g[0] + 18) / n
            bars += [(int(g[0] + k * w) + 15, int(g[0] + (k + 1) * w) - 33)
                     for k in range(n)]
        # A gap wider than a bar separates the 100G/200G/400G clusters.
        starts = np.array([b[0] for b in bars])
        cluster = np.r_[0, np.cumsum(np.diff(starts) > 500)]
        for (a, b), c in zip(bars, cluster):
            vals = {k: segment_top(m, 800, int(y0), a, b) for k, m in colors.items()}
            vals = {k: None if v is None else (y0 - v) / px_per_us * 1e-6
                    for k, v in vals.items()}
            rows.append(dict(gpu=gpu, bw_gbit=[100, 200, 400][c], x=a, **vals,
                             tick_unc_s=6.5 / px_per_us * 1e-6))
    df = pd.DataFrame(rows)
    df["series"] = df.groupby(["gpu", "bw_gbit"]).cumcount().map(
        dict(enumerate(series_order)))
    df["dp"] = df["dp_top"] - df["tp"]
    df["comp"] = df["total"] - df["dp_top"]
    df.loc[df["series"] == "no_comp", ["comp", "total"]] = [0.0, None]
    df.loc[df["series"] == "no_comp", "total"] = df["dp_top"]
    df = df[["gpu", "bw_gbit", "series", "tp", "dp", "comp", "total", "tick_unc_s"]]
    OUT.parent.mkdir(exist_ok=True)
    df.round(4).to_csv(OUT, index=False)
    print(df.round(4).to_string(index=False))


if __name__ == "__main__":
    main(sys.argv[1])
