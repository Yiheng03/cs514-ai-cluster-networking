"""Figure 11(b): GPT-3 175B, GPUs per host (8-64) x TP size.

Same workload settings and calibrated throughput as fig10. The paper does not
state the bandwidth for this figure; we use the production designs from its
Section 5.1 (H100 400 Gbit/s, H20 200 Gbit/s) and also run 100 and 200.
"""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from simtools import calibrate, config as C, sim, workloads as W  # noqa: E402

GPUS, PP, GBS = 1024, 8, 64
PROD_BW = {"H100": 400, "H20": 200}
CONFIGS = [(8, 8), (16, 8), (16, 16), (32, 8), (32, 16), (32, 32),
           (64, 8), (64, 16), (64, 32)]          # (GPUs per host, TP)


def load_workload(tp):
    raw = W.generate_gpt3(GPUS, tp, PP, GBS, 1,
                          C.WORKLOAD_DIR / "raw" / f"gpt3_tp{tp}_pp{PP}_gbs{GBS}.txt")
    header, df = W.read_workload(raw)
    return header, W.optimizer_allgather_to_dp(df)


def run_all(cal, tag, bw=PROD_BW, scale=1.0, tp_efficiency=False):
    """Simulate every config. scale multiplies compute time.

    With tp_efficiency=True the per-GPU compute time stays at its TP 8 value
    instead of shrinking as TP grows.
    """
    rows = []
    for tp in sorted({t for _, t in CONFIGS}):
        header, df = load_workload(tp)
        eff = 8 / tp if tp_efficiency else 1.0
        for gpu in C.GPU_TYPES:
            d = W.fill_compute(df, "gpt3", cal[gpu] * eff / scale, tp=tp)
            path = W.write_workload(header, d, C.WORKLOAD_DIR / f"{tag}_tp{tp}_{gpu}.txt")
            for gps in [g for g, t in CONFIGS if t == tp]:
                r = sim.run(path, GPUS, gps, bw[gpu], gpu, f"{tag}_{gpu}_g{gps}_tp{tp}")
                rows.append(dict(gpu=gpu, gpus_per_host=gps, tp=tp,
                                 dp=GPUS // (tp * PP),
                                 **{f"t_{k}": v for k, v in r.items()}))
    out = pd.DataFrame(rows)
    # Normalize like the paper: slowest config of each GPU = 1.
    out["norm_time"] = out["t_total"] / out.groupby("gpu")["t_total"].transform("max")
    return out.sort_values(["gpu", "gpus_per_host", "tp"])


def main():
    cal = calibrate.load()
    paper = pd.read_csv(C.REPO_ROOT / "data" / "fig11b_digitized.csv")

    out = run_all(cal, "fig11b")
    out.to_csv(C.SUMMARY_DIR / "fig11b_ours.csv", index=False)
    cmp_ = out.merge(paper, on=["gpu", "gpus_per_host", "tp"], suffixes=("", "_paper"))
    cmp_["abs_err"] = cmp_["norm_time"] - cmp_["norm_time_paper"]
    cols = ["gpu", "gpus_per_host", "tp", "t_total", "t_tp", "t_dp", "t_comp",
            "t_bubble", "norm_time", "norm_time_paper", "abs_err"]
    print(cmp_[cols].round(3).to_string(index=False))
    cmp_[cols].to_csv(C.SUMMARY_DIR / "fig11b_compare.csv", index=False)

    rows = [run_all(cal, f"fig11b_comp{s}", scale=s).assign(comp_scale=s)
            for s in (0.75, 1.25)]
    pd.concat(rows).to_csv(C.SUMMARY_DIR / "fig11b_compute_sensitivity.csv", index=False)

    for bw in (100, 200):
        run_all(cal, f"fig11b_bw{bw}", bw={"H100": bw, "H20": bw}).to_csv(
            C.SUMMARY_DIR / f"fig11b_bw{bw}_sensitivity.csv", index=False)

    rows = []
    for name, bw in (("prod", PROD_BW), ("bw100", {"H100": 100, "H20": 100})):
        o = run_all(cal, f"fig11b_pen_{name}", bw=bw, tp_efficiency=True)
        rows.append(o.assign(variant=name))
    pen = pd.concat(rows).merge(paper, on=["gpu", "gpus_per_host", "tp"],
                                suffixes=("", "_paper"))
    pen.to_csv(C.SUMMARY_DIR / "fig11b_tp_efficiency_sensitivity.csv", index=False)


if __name__ == "__main__":
    main()
