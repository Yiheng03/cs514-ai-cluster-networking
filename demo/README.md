# Demo

A short end-to-end run that shows the code works on a fresh machine: it builds
SimAI-Analytical from the official source inside a clean Ubuntu 24.04
container, generates GPT-3 175B and DeepSeek-V2-Lite workloads with AICB,
calibrates compute, runs the Figure 10 bandwidth sweep and a few MoE points,
and redraws the figures. Everything runs on CPU; no GPU is needed.

## Run

Requires Docker (Docker Desktop on macOS works, including Apple Silicon).
From the repository root:

```sh
git submodule update --init --recursive        # SimAI + AICB source
git -C third_party/SimAI/aicb apply ../../../patches/aicb_no_aiob.patch
sh demo/run.sh
```

The first run builds the image and compiles SimAI (a few minutes); the
experiments themselves take about 30 seconds. Results are written to
`demo/output/`.

## Files

| File | Purpose |
|---|---|
| `Dockerfile` | Ubuntu 24.04 with gcc, cmake, Python, numpy, pandas, matplotlib |
| `run.sh` | Builds the image and runs the demo in a container (run on the host) |
| `quick_demo.sh` | Inside the container: builds SimAI, runs the demo, collects outputs |
| `build_simai.sh` | Compiles SimAI-Analytical with SimAI's own build script |
| `quick_demo.py` | Picks the configurations; calls the experiment code in `experiments/` |
| `paths.sh` | Finds the repo and SimAI (submodule, or `SIMAI_ROOT`) |
| `output/` | Outputs from our recorded run on a MacBook Air (Apple M5, Docker) |

The numbers in `output/demo_fig10.csv` and `output/demo_moe.csv` match
`results/fig10_ours.csv` and `results/moe_bandwidth_ep.csv`.
