#!/bin/sh
# Build the Docker image and run the quick demo inside it (run on the host).
# Usage: sh demo/run.sh
set -e
. "$(dirname "$0")/paths.sh"

# Mount the folder that holds both the repo and SimAI.
ROOT="$(cd "$DEMO_DIR/.." && pwd)"
REL="${DEMO_DIR#$ROOT/}"

docker build -t cs514-simai "$DEMO_DIR"
docker run --rm -v "$ROOT":/work -w /work cs514-simai sh "$REL/quick_demo.sh"
