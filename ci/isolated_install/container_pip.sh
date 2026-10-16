#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Runs INSIDE a container. Installs a single cuOpt component wheel into a clean venv, builds the
# sample gtest against it, runs it.
#
# Env:
#   COMPONENT    client|mathopt|routing (required)
#   CUDA_MAJOR   default 13
#   WHEEL_DIRS   optional, colon-separated directories holding prebuilt wheels (the client wheel
#                plus the component wheel). When set, exactly those wheel files are installed (this
#                is what PR CI uses); otherwise the nightly wheel index is used.
#   NIGHTLY_WHEEL_INDEX  index for the nightly path and for third-party dependencies
set -euo pipefail

: "${COMPONENT:?}"
CUDA_MAJOR="${CUDA_MAJOR:-13}"
INDEX="${NIGHTLY_WHEEL_INDEX:-https://pypi.anaconda.org/rapidsai-wheels-nightly/simple}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORK="$(mktemp -d)"

case "${COMPONENT}" in
  client)  PKG="libcuopt-client"; EXPECTED="libcuopt-client" ;;
  mathopt) PKG="libcuopt-mathopt-cu${CUDA_MAJOR}"; EXPECTED="libcuopt-client libcuopt-mathopt-cu${CUDA_MAJOR}" ;;
  routing) PKG="libcuopt-routing-cu${CUDA_MAJOR}"; EXPECTED="libcuopt-client libcuopt-routing-cu${CUDA_MAJOR}" ;;
  *) echo "bad COMPONENT=${COMPONENT}" >&2; exit 2 ;;
esac

# Build tooling is a test fixture, not part of the package under test.
ensure_system_tools() {
  local missing=()
  command -v g++ >/dev/null 2>&1 || missing+=(g++)
  command -v make >/dev/null 2>&1 || missing+=(make)
  python3 -c 'import venv, ensurepip' >/dev/null 2>&1 || missing+=(python3-venv)
  [[ ${#missing[@]} -eq 0 ]] && return 0
  if command -v apt-get >/dev/null 2>&1; then
    export DEBIAN_FRONTEND=noninteractive
    apt-get update -qq
    apt-get install -y -qq --no-install-recommends "${missing[@]}" python3-pip ca-certificates >/dev/null
  elif command -v dnf >/dev/null 2>&1; then
    dnf install -y -q "${missing[@]/g++/gcc-c++}" >/dev/null
  else
    echo "missing tools (${missing[*]}) and no known package manager" >&2
    exit 1
  fi
}
ensure_system_tools

python3 -m venv "${WORK}/venv"
# shellcheck disable=SC1091
. "${WORK}/venv/bin/activate"
pip install -q cmake

if [[ -n "${WHEEL_DIRS:-}" ]]; then
  # Install exactly the prebuilt wheel files; third-party dependencies still come from the index.
  FILES=()
  IFS=: read -r -a DIRS <<< "${WHEEL_DIRS}"
  pick() { # pick <glob>: first match across WHEEL_DIRS
    local d f
    for d in "${DIRS[@]}"; do
      for f in "${d}"/${1}; do [[ -f "${f}" ]] && { echo "${f}"; return 0; }; done
    done
    echo "no wheel matching ${1} in ${WHEEL_DIRS}" >&2
    return 1
  }
  FILES+=("$(pick 'libcuopt_client-*.whl')")
  [[ "${COMPONENT}" != "client" ]] && FILES+=("$(pick "libcuopt_${COMPONENT}_*.whl")")
  echo "== installing local wheels: ${FILES[*]}"
  # Pin each cuopt package to its exact local file, so that pip cannot satisfy it from a public
  # index instead (--extra-index-url keeps PyPI enabled, which hosts 0.0.0a0 placeholders).
  : > "${WORK}/constraints.txt"
  for f in "${FILES[@]}"; do
    name="$(basename "${f}" | cut -d- -f1 | tr '_' '-')"
    echo "${name} @ file://$(realpath "${f}")" >> "${WORK}/constraints.txt"
  done
  pip install -q --pre --extra-index-url "${INDEX}" --constraint "${WORK}/constraints.txt" "${FILES[@]}"
else
  pip install -q --pre --extra-index-url "${INDEX}" "${PKG}"
fi

echo "== installed cuopt packages"
INSTALLED="$(pip list --format=freeze | grep -i '^libcuopt\|^cuopt' | cut -d= -f1 | tr '[:upper:]_' '[:lower:]-' | sort | xargs)"
echo "${INSTALLED}"
WANT="$(echo "${EXPECTED}" | tr ' ' '\n' | sort | xargs)"
if [[ "${INSTALLED}" != "${WANT}" ]]; then
  echo "FAIL: expected exactly [${WANT}] but found [${INSTALLED}]" >&2
  exit 1
fi

# The public PyPI placeholders are 0.0.0a0; installing one means no real cuopt wheel was found.
if pip list --format=freeze | grep -i '^libcuopt' | grep -q '==0\.0\.0'; then
  echo "FAIL: a 0.0.0 placeholder cuopt package was installed" >&2
  exit 1
fi

SP="$(python -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')"

# mathopt/routing consumers compile CUDA, so they need nvcc. The client wheel is declared as
# needing no CUDA, so it deliberately gets none (a client-only user will not have it either).
if [[ "${COMPONENT}" != "client" ]] && ! command -v nvcc >/dev/null 2>&1; then
  pip install -q "cuda-toolkit[nvcc,cudart,cccl]==${CUDA_MAJOR}.*"
  NVCC="$(find "${SP}/nvidia" -name nvcc -type f -path '*/bin/*' 2>/dev/null | head -1 || true)"
  if [[ -z "${NVCC}" ]]; then
    echo "FAIL: nvcc not found after installing cuda-toolkit[nvcc]" >&2
    exit 1
  fi
  export CUDACXX="${NVCC}"
  CUDAToolkit_ROOT="$(dirname "$(dirname "${NVCC}")")"
  export CUDAToolkit_ROOT
  # pip CUDA wheels ship only versioned libraries (libcudart.so.13); FindCUDAToolkit looks for the
  # unversioned name. This is venv-local scratch, only needed to make the test project configure.
  for lib in "${CUDAToolkit_ROOT}"/lib/lib*.so.[0-9]*; do
    [[ -e "${lib%.so.*}.so" ]] || ln -s "$(basename "${lib}")" "${lib%.so.*}.so"
  done
fi

# Wheels install headers + CMake configs under <pkg>/lib64/cmake/<name>/. Third-party configs
# (rmm, raft, ...) are only found via their config directory; cuopt's config finds its sibling
# component targets (mathopt/routing) through each wheel's root, so pass both.
ROOTS="$(find "${SP}" -mindepth 3 -maxdepth 3 -path '*/lib64/cmake' -printf '%h\n' | xargs -r -n1 dirname | sort -u | tr '\n' ';')"
CONFIG_DIRS="$(find "${SP}" \( -name '*-config.cmake' -o -name '*Config.cmake' \) -printf '%h\n' | sort -u | tr '\n' ';')"
PREFIXES="${ROOTS}${CONFIG_DIRS}"
CUDA_LIBS="$(find "${SP}/nvidia" -maxdepth 3 -type d -name lib 2>/dev/null | tr '\n' ':' || true)"
export LD_LIBRARY_PATH="${CUDA_LIBS}${LD_LIBRARY_PATH:-}"

cmake -S "${HERE}/tests" -B "${WORK}/build" -DCOMPONENT="${COMPONENT}" \
      -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH="${PREFIXES}"
cmake --build "${WORK}/build" -j"$(nproc)"
"${WORK}/build/isolated_${COMPONENT}_test"
