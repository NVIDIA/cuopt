#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

set -eEuo pipefail

echo "checking for symbol visibility issues"

LIBRARY="${1}"

echo ""
echo "Checking exported symbols in '${LIBRARY}'"
symbol_file="$(mktemp)"
match_file="$(mktemp)"
trap 'rm -f "${symbol_file}" "${match_file}"' EXIT

# Ignore WEAK and UNIQUE symbols since UNIQUE symbols should be exported and
# WEAK symbols may come from template instantiations.
# Ignore symbols containing "_error" since these are likely exception types
# and should be exported.

readelf --dyn-syms --wide "${LIBRARY}" \
    | awk '$7 != "UND" && $5 != "WEAK" && $5 != "UNIQUE"' \
    | c++filt --no-params \
    | awk '$0 !~ /_error/' \
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

failed=0

for pattern in "${patterns[@]}"; do
    echo "Checking for '${pattern}' symbols..."

    awk -v pattern="${pattern}" '
        BEGIN { has_trailing_scope = (substr(pattern, length(pattern) - 1) == "::") }
        $1 ~ /^[0-9]+:/ {
            symbol = ""
            for (i = 8; i <= NF; ++i) {
                symbol = symbol (i == 8 ? "" : " ") $i
            }

            sub(/<.*/, "", symbol)
            sub(/^.*[[:space:]](for|to)[[:space:]]+/, "", symbol)

            if (has_trailing_scope) {
                matched = (index(symbol, pattern) == 1)
            } else {
                matched = (symbol == pattern || index(symbol, pattern "::") == 1)
            }

            if (matched) { print }
        }
    ' "${symbol_file}" > "${match_file}"

    matches=$(awk 'END { print NR }' "${match_file}")
    if [[ "${matches}" -ne 0 ]]; then
        sed -n '1,20p' "${match_file}"
        echo "ERROR: Found exported symbols in ${LIBRARY} matching the pattern ${pattern}."
        echo "ERROR: Total matching symbols: ${matches}"
        failed=1
    fi
done

if [[ "${failed}" -ne 0 ]]; then
    exit 1
fi

echo "No symbol visibility issues found in ${LIBRARY}"
