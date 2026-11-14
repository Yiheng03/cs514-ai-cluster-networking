#!/bin/sh
# Build SimAI-Analytical (inside the container). Skips if already built.
set -e
. "$(dirname "$0")/paths.sh"
cd "$SIMAI_ROOT"
if [ -x bin/SimAI_analytical ]; then
  echo "SimAI_analytical already built"
else
  ./scripts/build.sh -c analytical
fi
ls -l bin/SimAI_analytical
