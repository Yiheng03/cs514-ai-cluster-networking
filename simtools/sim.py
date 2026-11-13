"""Run SimAI-Analytical and parse its summary row."""

import re
import shutil
import subprocess
from pathlib import Path

from . import config as C

SUMMARY_KEYS = ["dp", "dp_ep", "tp", "ep", "pp", "bubble", "comp",
                "exposed_comm", "total"]


def run(workload, gpus, gpus_per_server, nic_gbit, gpu_type, tag,
        nvlink=C.NVLINK_GBPS, dp_overlap=0.0, ep_overlap=0.0, tp_overlap=0.0):
    """Simulate one iteration and return the breakdown in seconds."""
    cmd = [str(C.SIMAI_BIN), "-w", str(Path(workload).resolve()),
           "-g", str(gpus), "-g_p_s", str(gpus_per_server),
           "-nv", str(nvlink), "-nic", str(C.NIC_GBPS[nic_gbit]),
           "-n_p_s", str(gpus_per_server), "-g_type", gpu_type,
           "-dp_o", str(dp_overlap), "-ep_o", str(ep_overlap),
           "-tp_o", str(tp_overlap), "-r", f"{tag}-"]
    # SimAI reads its ratio tables via relative paths, so run from SIMAI_ROOT.
    (C.SIMAI_ROOT / "results").mkdir(exist_ok=True)
    proc = subprocess.run(cmd, cwd=C.SIMAI_ROOT, capture_output=True, text=True)
    out_csv = C.SIMAI_ROOT / "results" / f"{tag}-EndToEnd.csv"
    if proc.returncode != 0 or not out_csv.exists():
        raise RuntimeError(f"SimAI failed for {tag}: {proc.stderr[-1500:]}")
    C.RAW_DIR.mkdir(parents=True, exist_ok=True)
    dest = C.RAW_DIR / out_csv.name
    shutil.move(str(out_csv), dest)
    return parse(dest)


def parse(csv_path):
    # Row 2 looks like "<tag>, 854297 (42.86%), ..., 1993267"; values are in us.
    row = Path(csv_path).read_text().splitlines()[1]
    vals = [int(re.match(r"\s*(-?\d+)", f).group(1)) for f in row.split(",")[1:]]
    res = {k: v * 1e-6 for k, v in zip(SUMMARY_KEYS, vals)}
    if res["total"] > 1e6:
        raise RuntimeError(f"overflow in {csv_path}")
    return res
