# CS514: AI Cluster Networking

**Designing AI Training Clusters with Simulation: From GPT-3 to Mixture-of-Experts Workloads**

Fall 2026 course project by **Yiheng Fan** and **Ying Xiong**.

We reproduce the SimAI paper's GPT-3 175B bandwidth and tensor-parallelism case
studies (Figures 10 and 11(b)) with the public SimAI-Analytical simulator and
AICB workload generator, then study a DeepSeek-V2-Lite MoE training workload.

## Main results

| | Paper | Ours |
|---|---|---|
| H100 iteration time, 100/200/400 Gbit/s (s) | 2.350 / 1.703 / 1.380 | 2.349 / 1.708 / 1.388 |
| H20 iteration time, 100/200/400 Gbit/s (s) | 5.904 / 5.258 / 4.935 | 5.890 / 5.249 / 4.929 |
| H100 time saved 200→400 | 19% | 18.8% |
| H20 time saved 100→200 / 200→400 | 11% / 6% | 10.9% / 6.1% |

Only the 100 Gbit/s computation segment is calibrated; every other point is a
prediction. Figure 11(b) reproduces only in direction (see the report).
For DeepSeek, the H20 200→400 Gbit/s gain is 9–10% at 4K tokens/GPU (vs. 6% for
GPT-3) and 3–7% at 16K tokens/GPU, rising as EP groups leave the host.

## Repository layout

```
proposal/              project proposal (LaTeX)
simtools/              shared code
  config.py            every parameter and assumption, in one place
  workloads.py         AICB generation, workload fixes, compute filling, skew
  flops.py             per-row FLOP counts (GPT-3, DeepSeek-V2-Lite)
  calibrate.py         single-point computation calibration against Fig. 10
  sim.py               run SimAI-Analytical and parse its summary
experiments/           one script per study
  fig10_bandwidth.py   Figure 10 + batch / compute / overlap sensitivity
  fig11b_tp_hostsize.py Figure 11(b) + bandwidth / compute / TP-efficiency sensitivity
  deepseek_moe.py      MoE: bandwidth x EP, host size x EP, TP control, skew, +-25% compute
analysis/
  digitize_fig10.py    read Figure 10 bar values from the paper PDF
  digitize_fig11b.py   read Figure 11(b) bar values from the paper PDF
  memory_check.py      per-GPU model-state memory estimate
  make_figures.py      all report figures
data/                  digitized paper values (CSV)
results/               tidy result tables (CSV) + calibration.json
results/raw/           every SimAI-Analytical output CSV, one per run
figures/               report figures (PDF)
demo/                  Docker-based quick demo (see demo/README.md)
patches/               small AICB patch (DeepSeek workloads without a GPU)
third_party/SimAI/     SimAI + AICB source (git submodule)
```

## Rerun everything

Requirements: Linux (or WSL2/Docker), `gcc`, `cmake`, Python 3 with `numpy`,
`pandas`, `matplotlib`, `pillow`; `poppler-utils` only for re-digitizing.
No GPU and no PyTorch are needed.

SimAI is pinned as a submodule at `third_party/SimAI` (commit `a57d765`, AICB
submodule `23eec3c`).

```sh
git clone --recurse-submodules https://github.com/Yiheng03/cs514-ai-cluster-networking.git
cd cs514-ai-cluster-networking
git -C third_party/SimAI/aicb apply ../../../patches/aicb_no_aiob.patch
(cd third_party/SimAI && ./scripts/build.sh -c analytical)
sh run_all.sh                     # ~45 min; optional arg: paper PDF to re-digitize
```

`SIMAI_ROOT` can point to a SimAI checkout elsewhere.

For a short check that everything builds and runs (about 5 minutes, Docker
only), see [demo/README.md](demo/README.md).

## Assumptions we made (details in `simtools/config.py` and the report)

* One NIC per GPU; 100/200/400 Gbit/s per GPU = `-nic` 12.5/25/50 GB/s; NVLink `-nv 360`.
* GPT-3: TP 8, PP 8, seq 2048, micro-batch 1, **global batch 64** (not in the paper;
  chosen to match the paper's TP time; 32–256 tested).
* Figure 11(b) bandwidth: 400 Gbit/s (H100) / 200 Gbit/s (H20), production designs.
* The distributed-optimizer parameter all-gather is placed in the DP group.
* DeepSeek-V2-Lite with its published hidden size 2048 (AICB's preset uses 10944).
* Computation = FLOPs / calibrated throughput (H100 164, H20 35 TFLOP/s).

## References

- [SimAI paper (NSDI 2025)](https://www.usenix.org/conference/nsdi25/presentation/wang-xizheng-simai)
- [SimAI](https://github.com/aliyun/SimAI)
- [AICB](https://github.com/aliyun/aicb)
