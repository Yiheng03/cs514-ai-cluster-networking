"""Figure 10: GPT-3 175B iteration time vs. per-GPU NIC bandwidth.

1024 GPUs, 8 per host, TP 8, PP 8, seq 2048, micro-batch 1. The paper does
not give the global batch; 64 gives the TP time closest to the 100G bar, and
32-256 are checked in the batch sensitivity run.
"""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from simtools import calibrate, config as C, sim, workloads as W  # noqa: E402

GPUS, GPS, TP, PP = 1024, 8, 8, 8
GBS_MAIN = 64


def build(gbs):
    raw = W.generate_gpt3(GPUS, TP, PP, gbs, 1,
                          C.WORKLOAD_DIR / "raw" / f"gpt3_tp{TP}_pp{PP}_gbs{gbs}.txt")
    header, df = W.read_workload(raw)
    return header, W.optimizer_allgather_to_dp(df)


def sweep(header, df, cal, tag, scale=1.0, dp_overlap=0.0):
    rows = []
    for gpu in C.GPU_TYPES:
        d = W.fill_compute(df, "gpt3", cal[gpu] / scale, tp=TP)
        path = W.write_workload(header, d, C.WORKLOAD_DIR / f"{tag}_{gpu}.txt")
        for bw in C.NIC_GBPS:
            r = sim.run(path, GPUS, GPS, bw, gpu, f"{tag}_{gpu}_{bw}G",
                        dp_overlap=dp_overlap)
            rows.append(dict(gpu=gpu, bw_gbit=bw, **r))
    out = pd.DataFrame(rows)
    # The paper has no bubble segment, so compare against comp + bubble.
    out["comp_incl_bubble"] = out["comp"] + out["bubble"]
    return out


def gains(df, col="total"):
    """Fractional time reduction per bandwidth doubling."""
    p = df.pivot(index="gpu", columns="bw_gbit", values=col)
    return pd.DataFrame({"100->200": 1 - p[200] / p[100],
                         "200->400": 1 - p[400] / p[200]})


def main():
    header, df = build(GBS_MAIN)
    cal = calibrate.calibrate(df, header, "gpt3", TP, GPUS, GPS, "fig10")
    calibrate.save(cal, dict(
        reference="Fig.10 SimAI bar at 100 Gbit/s, computation segment",
        target_s={g: calibrate.target_compute(g) for g in C.GPU_TYPES},
        global_batch=GBS_MAIN, flops_per_gpu_iter=W.total_flops(df, "gpt3", tp=TP),
        simai_commit=C.SIMAI_COMMIT, aicb_commit=C.AICB_COMMIT))
    print("throughput (TFLOP/s):", {g: round(v / 1e12, 1) for g, v in cal.items()})

    ours = sweep(header, df, cal, "fig10")
    ours.to_csv(C.SUMMARY_DIR / "fig10_ours.csv", index=False)
    print(ours.round(4).to_string(index=False))
    print(gains(ours).round(4))

    # Global batch sensitivity (recalibrated for each batch).
    rows = []
    for gbs in (32, 64, 128, 256):
        h, d = build(gbs)
        c = calibrate.calibrate(d, h, "gpt3", TP, GPUS, GPS, f"fig10_gbs{gbs}")
        s = sweep(h, d, c, f"fig10_gbs{gbs}")
        g = gains(s)
        for gpu in C.GPU_TYPES:
            rows.append(dict(global_batch=gbs, gpu=gpu,
                             tp_s=float(s[(s.gpu == gpu) & (s.bw_gbit == 100)].tp.iloc[0]),
                             gain_100_200=g.loc[gpu, "100->200"],
                             gain_200_400=g.loc[gpu, "200->400"]))
    pd.DataFrame(rows).to_csv(C.SUMMARY_DIR / "fig10_batch_sensitivity.csv", index=False)

    # Computation +-25%.
    rows = []
    for scale in (0.75, 1.0, 1.25):
        g = gains(sweep(header, df, cal, f"fig10_comp{scale}", scale=scale))
        for gpu in C.GPU_TYPES:
            rows.append(dict(comp_scale=scale, gpu=gpu,
                             gain_100_200=g.loc[gpu, "100->200"],
                             gain_200_400=g.loc[gpu, "200->400"]))
    pd.DataFrame(rows).to_csv(C.SUMMARY_DIR / "fig10_compute_sensitivity.csv", index=False)

    # Half of DP communication overlapped with computation.
    s = sweep(header, df, cal, "fig10_dpo05", dp_overlap=0.5)
    gains(s).reset_index().to_csv(C.SUMMARY_DIR / "fig10_overlap_sensitivity.csv", index=False)


if __name__ == "__main__":
    main()
