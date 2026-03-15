#!/usr/bin/env bash
# .brev/setup.sh
# Runs once on instance first boot (or launchable deploy).
# Idempotent: safe to re-run.
set -euo pipefail

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
REPO_DIR="${REPO_DIR:-$HOME/cuopt-opt}"
VENV="$HOME/coo-venv"
NV_PYPI="https://pypi.nvidia.com"

echo ">>> [1/5] Creating Python virtual environment at $VENV"
python3 -m venv "$VENV" --system-site-packages
# shellcheck source=/dev/null
source "$VENV/bin/activate"

# Upgrade pip/wheel first so binary wheels resolve correctly
pip install --quiet --upgrade pip wheel

# ---------------------------------------------------------------------------
# Install cuOpt (binary wheel — no build from source required)
# ---------------------------------------------------------------------------
echo ">>> [2/5] Installing cuOpt and CUDA runtime libraries"
pip install \
  --extra-index-url "$NV_PYPI" \
  --quiet \
  cuopt-cu12 \
  nvidia-nvjitlink-cu12

# ---------------------------------------------------------------------------
# Install the cuoptopt-agent + dashboard
# ---------------------------------------------------------------------------
echo ">>> [3/5] Installing cuoptopt-agent + dashboard dependencies"
pip install --quiet -e "$REPO_DIR/cuoptopt-agent[dev]"

# ---------------------------------------------------------------------------
# nvJitLink LD_PRELOAD fix
# Without this, libcuopt.so fails to import due to symbol version mismatch
# between the system CUDA toolkit's libnvJitLink and the pip-installed one.
# ---------------------------------------------------------------------------
echo ">>> [4/5] Configuring LD_PRELOAD for libnvJitLink"
JITLINK=$(python3 - <<'PYEOF'
import pathlib
try:
    import nvidia.nvjitlink as m
    base = pathlib.Path(m.__file__).parent
    # Find the versioned .so (e.g. libnvJitLink.so.12)
    candidates = sorted(base.glob("libnvJitLink.so.*"))
    print(candidates[-1])
except Exception as e:
    print("")
PYEOF
)

if [[ -n "$JITLINK" && -f "$JITLINK" ]]; then
    MARKER="export LD_PRELOAD=$JITLINK"
    if ! grep -qF "$JITLINK" "$HOME/.bashrc"; then
        echo "${MARKER}\${LD_PRELOAD:+:\$LD_PRELOAD}" >> "$HOME/.bashrc"
        echo "    Added LD_PRELOAD to ~/.bashrc: $JITLINK"
    else
        echo "    LD_PRELOAD already set in ~/.bashrc — skipping"
    fi
else
    echo "    WARNING: could not locate libnvJitLink.so — skipping LD_PRELOAD fix"
fi

# Also export for the remainder of this script session
if [[ -n "$JITLINK" && -f "$JITLINK" ]]; then
    export LD_PRELOAD="$JITLINK${LD_PRELOAD:+:$LD_PRELOAD}"
fi

# ---------------------------------------------------------------------------
# Write .env.template (API keys — never committed with real values)
# ---------------------------------------------------------------------------
echo ">>> [5/5] Writing $REPO_DIR/.env.template"
cat > "$REPO_DIR/.env.template" << 'ENVEOF'
# Copy this file to .env and fill in your keys.
# Source it before starting the agent or dashboard:
#   source .env
#
# Only the key for the model you use is required.
# GITHUB_TOKEN is required for the branch sidebar and PR creation.

export ANTHROPIC_API_KEY=""       # https://console.anthropic.com
export OPENAI_API_KEY=""          # https://platform.openai.com
export NVIDIA_API_KEY=""          # https://build.nvidia.com

export GITHUB_TOKEN=""            # GitHub → Settings → Developer settings → PAT (scope: repo)

# Optional: change the dashboard port (default 8080)
# export DASHBOARD_PORT=8080
ENVEOF

# ---------------------------------------------------------------------------
# Smoke-test: verify cuOpt imports cleanly
# ---------------------------------------------------------------------------
echo ""
echo ">>> Verifying cuOpt import..."
if python3 -c "import cuopt; print('  cuopt', cuopt.__version__, '— OK')" 2>/dev/null; then
    :
else
    echo "  WARNING: cuopt import failed — check CUDA driver compatibility."
    echo "  Run: python3 -c \"import cuopt\" to diagnose."
fi

# ---------------------------------------------------------------------------
# Done
# ---------------------------------------------------------------------------
cat << 'BANNER'

=============================================================
  cuoptopt-agent setup complete!

  Next steps:
    1. Copy and fill in API keys:
         cp .env.template .env && nano .env
    2. Source the keys:
         source .env
    3. Start the dashboard:
         source ~/coo-venv/bin/activate
         cuoptopt-dashboard
    4. Open the dashboard:
         Use the Brev Secure Link for port 8080, or
         http://localhost:8080 if connecting via brev port-forward.

  Connect Cursor:
    On your local machine:
         brev refresh
         brev open cursor <instance-name>
=============================================================

BANNER
