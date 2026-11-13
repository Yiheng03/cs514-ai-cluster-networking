#!/bin/sh
# Run every experiment and regenerate the figures.
# Usage: sh run_all.sh [nsdi25-wang-xizheng-simai.pdf]
set -e
cd "$(dirname "$0")"
if [ -n "$1" ]; then    # optional: re-read the figures from the PDF
  python3 analysis/digitize_fig10.py "$1"
  python3 analysis/digitize_fig11b.py "$1"
fi
python3 experiments/fig10_bandwidth.py      # calibration + Figure 10 (~5 min)
python3 experiments/fig11b_tp_hostsize.py   # Figure 11(b) + sensitivities (~20 min)
python3 experiments/deepseek_moe.py         # MoE extension (~15 min)
python3 analysis/memory_check.py
python3 analysis/make_figures.py
