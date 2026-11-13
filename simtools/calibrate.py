"""Calibrate per-GPU throughput against the paper's Figure 10.

The computation segment of each 100 Gbit/s SimAI bar is the target. SimAI
reports computation and pipeline bubble separately and both scale with 1/R,
while the paper shows a single segment, so we match comp + bubble:
    R = R0 * (comp + bubble at R0) / target
"""

import json

import pandas as pd

from . import config as C
from . import sim
from . import workloads as W

CAL_FILE = C.SUMMARY_DIR / "calibration.json"
REF_BW = 100


def target_compute(gpu):
    f = pd.read_csv(C.REPO_ROOT / "data" / "fig10_digitized.csv")
    row = f[(f.gpu == gpu) & (f.bw_gbit == REF_BW) & (f.series == "simai")]
    return float(row["comp"].iloc[0])


def calibrate(workload_df, header, model, tp, gpus, gps, tag, r0=1e14):
    out = {}
    for gpu in C.GPU_TYPES:
        d = W.fill_compute(workload_df, model, r0, tp=tp)
        path = W.write_workload(header, d, C.WORKLOAD_DIR / f"{tag}_r0.txt")
        r = sim.run(path, gpus, gps, REF_BW, gpu, f"{tag}_cal_{gpu}")
        out[gpu] = r0 * (r["comp"] + r["bubble"]) / target_compute(gpu)
    return out


def save(cal, meta):
    CAL_FILE.parent.mkdir(parents=True, exist_ok=True)
    CAL_FILE.write_text(json.dumps({"throughput_flops": cal, **meta}, indent=2))


def load():
    return json.loads(CAL_FILE.read_text())["throughput_flops"]
