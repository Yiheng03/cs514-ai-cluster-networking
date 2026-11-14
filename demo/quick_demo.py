"""Small subset of the experiments, for checking that everything runs."""

import sys
import time

import pandas as pd

sys.path.insert(0, ".")
sys.path.insert(0, "experiments")
import fig10_bandwidth as F10          # noqa: E402
import deepseek_moe as DS              # noqa: E402
from simtools import calibrate         # noqa: E402

t0 = time.time()
print("== GPT-3 175B: generate workload and calibrate compute ==")
header, df = F10.build(F10.GBS_MAIN)
cal = calibrate.calibrate(df, header, "gpt3", F10.TP, F10.GPUS, F10.GPS, "demo")
calibrate.save(cal, {"note": "quick demo run"})
print({g: f"{v / 1e12:.1f} TFLOP/s" for g, v in cal.items()})

print("\n== Fig. 10 sweep: 100 / 200 / 400 Gbit/s ==")
out = F10.sweep(header, df, cal, "demo_fig10")
out.to_csv("results/demo_fig10.csv", index=False)
print(out[["gpu", "bw_gbit", "tp", "dp", "comp_incl_bubble", "total"]].round(3).to_string(index=False))
print("\nTime saved per bandwidth doubling:")
print(F10.gains(out).round(3).to_string())

print("\n== DeepSeek-V2-Lite: EP 8 vs EP 32, H100, 1024 GPUs ==")
rows = [DS.run_point(cal, world=1024, ep=ep, tp=1, gps=8, bw=bw, gpu="H100",
                     tag=f"demo_ds_ep{ep}_{bw}G")
        for ep in (8, 32) for bw in (200, 400)]
moe = pd.DataFrame(rows)
moe.to_csv("results/demo_moe.csv", index=False)
print(moe[["ep", "bw_gbit", "a2a_intra_host_frac", "t_comp", "t_ep",
           "t_dp_ep", "t_total"]].round(3).to_string(index=False))

print(f"\nFinished in {time.time() - t0:.0f} s")
