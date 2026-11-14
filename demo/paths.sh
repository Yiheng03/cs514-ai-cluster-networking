# Locate the repo and SimAI. Sourced by the other demo scripts.
# Works both inside the GitHub repo (SimAI = third_party/SimAI submodule) and
# in a project folder where the repo and SimAI sit side by side.
DEMO_DIR="$(cd "$(dirname "$0")" && pwd)"
if [ -d "$DEMO_DIR/../simtools" ]; then
  REPO_DIR="$(cd "$DEMO_DIR/.." && pwd)"
else
  REPO_DIR="$(cd "$DEMO_DIR/../2_project_repo/cs514-ai-cluster-networking" && pwd)"
fi
if [ -z "$SIMAI_ROOT" ]; then
  if [ -f "$REPO_DIR/third_party/SimAI/scripts/build.sh" ]; then
    SIMAI_ROOT="$REPO_DIR/third_party/SimAI"
  else
    SIMAI_ROOT="$(cd "$DEMO_DIR/../3_third_party_source/SimAI" && pwd)"
  fi
fi
export SIMAI_ROOT
