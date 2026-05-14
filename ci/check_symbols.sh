#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

set -eEuo pipefail

echo "checking for symbol visibility issues"

LIBRARY="${1}"

echo ""
echo "Checking exported symbols in '${LIBRARY}'"
symbol_file="$(mktemp)"
readelf --dyn-syms --wide "${LIBRARY}" \
    | awk '$7 != "UND"' \
    | c++filt \
    > "${symbol_file}"

patterns=(
    'cub::'
    'thrust::'
    'raft::'
    'rmm::'
    'cuopt::linear_programming::detail'
    'cuopt::routing::detail'
    'grpc::'
    'google::protobuf'
    'tbb::'
)

for pattern in "${patterns[@]}"; do
    echo "Checking for '${pattern}' symbols..."
    matches=$(grep -F -c "${pattern}" "${symbol_file}" || true)
    if [[ "${matches}" -ne 0 ]]; then
        grep -F -m 20 "${pattern}" "${symbol_file}"
        echo "ERROR: Found exported symbols in ${LIBRARY} matching the pattern ${pattern}."
        echo "ERROR: Total matching symbols: ${matches}"
        rm "${symbol_file}"
        exit 1
    fi
done

rm "${symbol_file}"
echo "No symbol visibility issues found in ${LIBRARY}"
