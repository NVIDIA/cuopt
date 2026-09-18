#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Asserts that libcuopt_client.so stays a CUDA-free leaf (#1890).
#
# ci/check_symbols.sh answers a different question: it looks at *exported* symbols and
# fails when internal ones leak out. It filters undefined symbols away and never reads
# DT_NEEDED, so a change that reintroduces a dependency on rmm, raft, CUDA or another
# cuOpt component passes it unnoticed.

set -eEuo pipefail

LIBRARY="${1}"
status=0

echo "Checking that '${LIBRARY}' resolves without CUDA"

# Weak undefined symbols are allowed to stay unresolved, so only strong ones count.
undefined="$(
    nm --dynamic --undefined-only --with-symbol-versions "${LIBRARY}" \
        | awk '$1 == "U" { print $2 }' \
        | sed 's/@.*//' \
        | c++filt \
        | grep -E '^(rmm|raft)::|^cuda[A-Z_]|^cu[A-Z]' || true
)"
if [[ -n "${undefined}" ]]; then
    echo "ERROR: undefined GPU-stack symbols in ${LIBRARY}:"
    sed 's/^/    /' <<< "${undefined}"
    status=1
fi

needed="$(objdump -p "${LIBRARY}" | awk '/NEEDED/ { print $2 }')"

gpu_needed="$(grep -E '^(librmm|libraft|libcudart|libcuda|libcublas|libcusparse|libcudss|libnccl)' <<< "${needed}" || true)"
if [[ -n "${gpu_needed}" ]]; then
    echo "ERROR: ${LIBRARY} has a GPU-stack DT_NEEDED entry:"
    sed 's/^/    /' <<< "${gpu_needed}"
    status=1
fi

# The client is the leaf every other component links, so it must depend on none of them.
cuopt_needed="$(grep -E '^libcuopt' <<< "${needed}" || true)"
if [[ -n "${cuopt_needed}" ]]; then
    echo "ERROR: ${LIBRARY} depends on another cuOpt library:"
    sed 's/^/    /' <<< "${cuopt_needed}"
    status=1
fi

if [[ "${status}" -eq 0 ]]; then
    echo "OK: no GPU-stack symbols, no GPU-stack DT_NEEDED, no cuOpt dependency"
fi
exit "${status}"
