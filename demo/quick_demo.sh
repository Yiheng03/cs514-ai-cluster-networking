#!/bin/sh
# Short end-to-end run (about 3-5 minutes, mostly compiling SimAI):
#   build SimAI-Analytical, generate GPT-3 and DeepSeek workloads with AICB,
#   calibrate compute, run the Fig. 10 sweep and a few MoE points, redraw
#   the figures. Runs on a copy of the repo so committed results stay as they
#   are; outputs go to demo/output/.
set -e
. "$(dirname "$0")/paths.sh"
sh "$DEMO_DIR/build_simai.sh"

rm -rf /tmp/demo && mkdir -p /tmp/demo
tar -C "$REPO_DIR" --exclude=./third_party --exclude=./results/raw --exclude=./.git -cf - . \
  | tar -C /tmp/demo -xf -
cd /tmp/demo
python3 "$DEMO_DIR/quick_demo.py"
python3 analysis/make_figures.py

mkdir -p "$DEMO_DIR/output"
cp results/calibration.json results/demo_*.csv figures/*.pdf "$DEMO_DIR/output/"
echo "Done. Outputs in $DEMO_DIR/output/"
