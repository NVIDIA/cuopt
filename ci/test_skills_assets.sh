#!/bin/bash

# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Run all assets under skills/ (Python, C, CLI) as part of conda Python test.
# Python: run each .py from its directory (server API clients need server on port 8000).
# C: compile and run each .c with libcuopt.
# CLI: run cuopt_cli on each sample .mps in API-CLI skill assets.

set -euo pipefail

# Use rapids-logger in CI; fall back to echo for local testing
if command -v rapids-logger &>/dev/null; then
  log() { rapids-logger "$*"; }
else
  log() { echo "[rapids-logger] $*"; }
fi

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
SKILLS_ASSETS="${REPO_ROOT}/skills"
FAILED=()
SERVER_PID=""

if [[ ! -d "${SKILLS_ASSETS}" ]]; then
  log "No skills directory found, skipping skills asset tests"
  exit 0
fi

# ---- Start cuOpt server for server API Python assets (port 8000) ----
start_server() {
  if ! python -c "import cuopt_server" 2>/dev/null; then
    log "cuopt_server not available, server API assets will skip"
    return
  fi
  python -m cuopt_server.cuopt_service --ip 127.0.0.1 --port 8000 &>/dev/null &
  SERVER_PID=$!
  for _ in {1..30}; do
    if curl -s -o /dev/null http://127.0.0.1:8000/cuopt/health 2>/dev/null; then
      log "cuOpt server started (port 8000) for server API assets"
      return
    fi
    sleep 1
  done
  log "cuOpt server did not become ready; server API assets will skip"
  kill "${SERVER_PID}" 2>/dev/null || true
  SERVER_PID=""
}
stop_server() {
  if [[ -n "${SERVER_PID}" ]] && kill -0 "${SERVER_PID}" 2>/dev/null; then
    log "Stopping cuOpt server (PID ${SERVER_PID})"
    kill "${SERVER_PID}" 2>/dev/null || true
    wait "${SERVER_PID}" 2>/dev/null || true
    SERVER_PID=""
  fi
}
trap stop_server EXIT
start_server

# ---- Python assets ----
log "Testing Python assets in skills/"
while IFS= read -r -d '' script; do
  dir=$(dirname "$script")
  name=$(basename "$script")
  rel="${script#"$REPO_ROOT/"}"
  log "Running Python asset: $rel"
  if (cd "$dir" && python "$name"); then
    log "PASS: $rel"
  else
    FAILED+=("$rel")
    log "FAIL: $rel"
  fi
done < <(find "${SKILLS_ASSETS}" -path "*/assets/*" -name "*.py" -type f -print0 | sort -z)

# ---- C assets (compile and run; requires CONDA_PREFIX and a C compiler) ----
CC="${CC:-}"
if [[ -z "${CC}" ]]; then
  for c in gcc cc clang; do
    if command -v "$c" &>/dev/null; then
      CC="$c"
      break
    fi
  done
fi
if [[ -n "${CONDA_PREFIX:-}" ]]; then
  if [[ -z "${CC}" ]]; then
    log "No C compiler found; installing c-compiler in conda environment"
    if command -v mamba &>/dev/null; then
      mamba install -y -c conda-forge c-compiler
    else
      conda install -y -c conda-forge c-compiler
    fi
    for c in gcc cc clang; do
      if command -v "$c" &>/dev/null; then
        CC="$c"
        break
      fi
    done
    if [[ -z "${CC}" ]]; then
      log "C compiler still not found after install. Set CC or install gcc/cc/clang."
      exit 1
    fi
  fi
  INCLUDE_PATH="${CONDA_PREFIX}/include"
  LIB_PATH="${CONDA_PREFIX}/lib"
  export LD_LIBRARY_PATH="${LIB_PATH}:${LD_LIBRARY_PATH:-}"

  log "Testing C assets in skills (using ${CC})"
  while IFS= read -r -d '' cfile; do
    dir=$(dirname "$cfile")
    base=$(basename "$cfile" .c)
    rel="${cfile#"$REPO_ROOT/"}"
    log "Building and running C asset: $rel"
    if ! (cd "$dir" && "${CC}" -I"${INCLUDE_PATH}" -L"${LIB_PATH}" -o "$base" "$(basename "$cfile")" -lcuopt); then
      FAILED+=("$rel (build)")
      log "FAIL: $rel (build)"
      continue
    fi
    if [[ "$base" == "mps_solver" ]]; then
      run_cmd=(./"$base" data/sample.mps)
    else
      run_cmd=(./"$base")
    fi
    if (cd "$dir" && "${run_cmd[@]}"); then
      log "PASS: $rel"
    else
      FAILED+=("$rel")
      log "FAIL: $rel"
    fi
  done < <(find "${SKILLS_ASSETS}" -path "*/assets/*" -name "*.c" -type f -print0 | sort -z)
else
  log "CONDA_PREFIX not set, skipping C asset tests"
fi

# ---- CLI assets (cuopt_cli with sample MPS files) ----
log "Testing CLI assets in skills/"
while IFS= read -r -d '' mps; do
  rel="${mps#"$REPO_ROOT/"}"
  log "Running CLI asset: $rel"
  if cuopt_cli "$mps" --time-limit 10; then
    log "PASS: $rel"
  else
    FAILED+=("$rel")
    log "FAIL: $rel"
  fi
done < <(find "${SKILLS_ASSETS}" -path "*/cuopt-*-api-cli/assets/*" -name "*.mps" -type f -print0 | sort -z)

if [[ ${#FAILED[@]} -gt 0 ]]; then
  log "The following skills assets failed:"
  printf '%s\n' "${FAILED[@]}"
  exit 1
fi

log "All skills assets (Python, C, CLI) passed."
exit 0
