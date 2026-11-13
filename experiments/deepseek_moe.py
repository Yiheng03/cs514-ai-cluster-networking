"""DeepSeek-V2-Lite MoE extension.

Seq 4096, micro-batch 1, TP 1, PP 1, global batch 1024 sequences in every run.
Compute uses the throughput calibrated on GPT-3.

Sweeps:
  A  bandwidth x EP on 8-GPU hosts, 1024 GPUs and 256 GPUs; at 1024 GPUs
     also load skew (1.5, 2) and compute +-25%
  B  GPUs per host (8/16/32) x EP
  C  TP 1/2/4 at EP 16, 200 Gbit/s
"""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from simtools import calibrate, config as C, sim, workloads as W  # noqa: E402

GBS = 1024
EPS = (8, 16, 32, 64)
PROD_BW = {"H100": 400, "H20": 200}
SEQ = C.DEEPSEEK_16B["seq_length"]

_cache = {}


def workload(world, ep, tp):
    key = (world, ep, tp)
    if key not in _cache:
        raw = W.generate_deepseek(world, ep, tp, GBS, 1,
                                  C.WORKLOAD_DIR / "raw" / f"ds16_ws{world}_ep{ep}_tp{tp}.txt")
        header, df = W.read_workload(raw)
        df = W.optimizer_allgather_to_dp(df)
        if tp == 1:
            df = W.fix_tp1(df)
        _cache[key] = (header, df)
    return _cache[key]


def run_point(cal, *, world, ep, tp, gps, bw, gpu, tag, skew=1.0, comp_scale=1.0):
    header, df = workload(world, ep, tp)
    d = W.fill_compute(df, "deepseek16", cal[gpu] / comp_scale, tp=tp)
    if skew != 1.0:
        d = W.apply_skew(d, skew)
    path = W.write_workload(header, d, C.WORKLOAD_DIR / "moe" / f"{tag}.txt")
    r = sim.run(path, world, gps, bw, gpu, tag)
    return dict(gpu=gpu, world=world, ep=ep, tp=tp, gpus_per_host=gps,
                bw_gbit=bw, skew=skew, comp_scale=comp_scale,
                tokens_per_gpu=GBS * SEQ // world,
                a2a_intra_host_frac=W.placement_summary(ep, tp, gps),
                comm_frac=r["exposed_comm"] / r["total"],
                **{f"t_{k}": v for k, v in r.items()})


def main():
    cal = calibrate.load()

    # A: (skew, compute scale) variants; the extra ones only at 1024 GPUs.
    rows = []
    for world in (1024, 256):
        variants = [(1.0, 1.0)]
        if world == 1024:
            variants += [(1.5, 1.0), (2.0, 1.0), (1.0, 0.75), (1.0, 1.25)]
        for ep in EPS:
            for gpu in C.GPU_TYPES:
                for bw in C.NIC_GBPS:
                    for skew, cs in variants:
                        tag = f"moeA_{gpu}_ws{world}_ep{ep}_{bw}G_s{skew}_c{cs}"
                        rows.append(run_point(cal, world=world, ep=ep, tp=1, gps=8,
                                              bw=bw, gpu=gpu, tag=tag,
                                              skew=skew, comp_scale=cs))
    pd.DataFrame(rows).to_csv(C.SUMMARY_DIR / "moe_bandwidth_ep.csv", index=False)

    # B
    rows = []
    for ep in EPS:
        for gpu in C.GPU_TYPES:
            for gps in (8, 16, 32):
                for bw in sorted({PROD_BW[gpu], 200}):
                    rows.append(run_point(cal, world=1024, ep=ep, tp=1, gps=gps,
                                          bw=bw, gpu=gpu,
                                          tag=f"moeB_{gpu}_g{gps}_ep{ep}_{bw}G"))
    pd.DataFrame(rows).to_csv(C.SUMMARY_DIR / "moe_hostsize_ep.csv", index=False)

    # C
    rows = [run_point(cal, world=1024, ep=16, tp=tp, gps=8, bw=200, gpu=gpu,
                      tag=f"moeC_{gpu}_tp{tp}_ep16_200G")
            for tp in (1, 2, 4) for gpu in C.GPU_TYPES]
    pd.DataFrame(rows).to_csv(C.SUMMARY_DIR / "moe_tp_control.csv", index=False)


if __name__ == "__main__":
    main()
