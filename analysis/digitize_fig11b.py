"""Read the bar values of Figure 11(b) from the paper PDF.

Usage: python3 analysis/digitize_fig11b.py nsdi25-wang-xizheng-simai.pdf
Writes data/fig11b_digitized.csv. Blue = TP 8, green = TP 16, pink = TP 32.
"""

import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

Image.MAX_IMAGE_PIXELS = None
OUT = Path(__file__).resolve().parent.parent / "data" / "fig11b_digitized.csv"

# Pixel positions measured on the raster: y-axis, rows of the 1.0 and 0.0
# ticks for each panel, and the x-centre of each host-size cluster.
AXIS_X = 1298
PANELS = {"H100": (681, 3173), "H20": (4063, 6555)}
HOST_CENTRES = {8: 2215, 16: 4250, 32: 6250, 64: 8150}


def load(pdf):
    tmp = Path(tempfile.mkdtemp())
    subprocess.run(["pdfimages", "-f", "12", "-l", "12", "-png", pdf,
                    str(tmp / "im")], check=True)
    imgs = sorted(tmp.glob("im-*.png"))
    rgb = np.array(Image.open(imgs[2]).convert("RGB")).astype(int)
    rgb[np.array(Image.open(imgs[3])) < 128] = 255
    return rgb


def main(pdf):
    im = load(pdf)
    R, G, B = im[..., 0], im[..., 1], im[..., 2]
    masks = {
        8: (B > 150) & (B - R > 40) & (G < 200) & (B - G > 20),
        16: (G > 150) & (G - R > 40) & (G - B > 40),
        32: (R > 220) & (R - G > 30) & (R - B > 30),
    }
    rows = []
    for gpu, (one, zero) in PANELS.items():
        px = zero - one
        for tp, m in masks.items():
            sl = m[one - 150:zero - 10, AXIS_X + 10:9400]
            cols = np.where(sl.sum(axis=0) > 80)[0] + AXIS_X + 10
            groups = np.split(cols, np.where(np.diff(cols) > 5)[0] + 1)
            # The grid hatching splits pink bars in two; merge the pieces.
            merged = []
            for g in groups:
                if merged and g[0] - merged[-1][-1] < 120:
                    merged[-1] = np.r_[merged[-1], g]
                else:
                    merged.append(g)
            for g in merged:
                if len(g) < 100:
                    continue
                sub = m[one - 150:zero, g[0] + 40:g[-1] - 40]
                has = sub.any(axis=0)
                top = np.median(sub.argmax(axis=0)[has]) + one - 150
                host = min(HOST_CENTRES, key=lambda h: abs(HOST_CENTRES[h] - g.mean()))
                rows.append(dict(gpu=gpu, gpus_per_host=host, tp=tp,
                                 norm_time=round((zero - top) / px, 4)))
    df = pd.DataFrame(rows).sort_values(["gpu", "gpus_per_host", "tp"])
    df["tick_unc"] = round(6.5 / (PANELS["H100"][1] - PANELS["H100"][0]), 4)
    OUT.parent.mkdir(exist_ok=True)
    df.to_csv(OUT, index=False)
    print(df.to_string(index=False))


if __name__ == "__main__":
    main(sys.argv[1])
