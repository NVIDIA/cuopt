#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -eEuo pipefail

echo "checking for symbol visibility issues"

LIBRARY="${1}"

# The forbidden-symbol checks apply to every cuOpt library. The other two are opt-in:
#   --no-public-api-check  for components that do not provide the C API (routing, client)
#   --require-leaf         for cuopt_client, which must resolve without the GPU stack
CHECK_PUBLIC_API=1
REQUIRE_LEAF=0
shift
for arg in "$@"; do
    case "${arg}" in
        --no-public-api-check) CHECK_PUBLIC_API=0 ;;
        --require-leaf)        REQUIRE_LEAF=1 ;;
        *) echo "unknown argument: ${arg}" >&2; exit 2 ;;
    esac
done

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
    'cuopt::mathematical_optimization::detail'
    'cuopt::routing::detail'
    'cuopt::detail'
    'grpc::'
    'google::protobuf'
    'tbb::'
    'absl::'
    'dejavu::'
    'papilo::'
    'boost::'
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

# Required public API symbols that must stay exported. This is a small stability
# anchor (core C API lifecycle entrypoints), not an exhaustive list: without it,
# a library whose visibility was over-tightened so the public API is entirely
# hidden would still pass the forbidden-symbol checks above while being unusable.
# Keep this set minimal and limited to entrypoints guaranteed to exist.
required_symbols=(
    cuOptReadProblem
    cuOptCreateProblem
    cuOptSolve
    cuOptDestroyProblem
)

if [[ "${CHECK_PUBLIC_API}" -eq 1 ]]; then
    exported_funcs="$(readelf --dyn-syms --wide "${LIBRARY}" | awk '$7 != "UND" && $4 == "FUNC" { print $8 }')"

    for sym in "${required_symbols[@]}"; do
        echo "Checking that required symbol '${sym}' is exported..."
        if ! grep -qxF "${sym}" <<< "${exported_funcs}"; then
            echo "ERROR: Required public API symbol '${sym}' is not exported from ${LIBRARY}."
            echo "ERROR: Symbol visibility may be over-restricted and hiding the public API."
            failed=1
        fi
    done
fi

# Each component library keeps its own logger only while this state stays hidden; exporting it
# silently merges them back into one. Nothing else catches that.
logger_state_symbols=(
    "cuopt::default_logger()"
    "cuopt::global_log_buffer()"
    "cuopt::reset_default_logger()"
)

demangled_dyn_syms="$(readelf --dyn-syms --wide "${LIBRARY}" | awk '$7 != "UND" { print $8 }' | c++filt)"

for sym in "${logger_state_symbols[@]}"; do
    echo "Checking that logger state '${sym}' is NOT exported..."
    if grep -qF "${sym}" <<< "${demangled_dyn_syms}"; then
        echo "ERROR: Logger state '${sym}' is exported from ${LIBRARY}."
        echo "ERROR: Per-component loggers collapse into one. Check that logger.hpp's namespace"
        echo "ERROR: is not marked CUOPT_EXPORT and hidden visibility is still set on the target."
        failed=1
    fi
done

# cuopt_client has to stay a CUDA-free leaf (#1890). The checks above look at exported
# symbols only: they filter undefined ones away and never read DT_NEEDED, so a change
# reintroducing rmm, raft, CUDA or another component would pass them unnoticed.
if [[ "${REQUIRE_LEAF}" -eq 1 ]]; then
    echo ""
    echo "Checking that '${LIBRARY}' resolves without the GPU stack"

    # Weak undefined symbols are allowed to stay unresolved, so only strong ones count,
    # and @VERSION suffixes are stripped before matching. rmm::/raft:: are deliberately
    # unanchored: a leak often demangles to "typeinfo for rmm::..." or "vtable for
    # rmm::...", which an anchored pattern would miss.
    undefined="$(
        nm --dynamic --undefined-only --with-symbol-versions "${LIBRARY}" \
            | awk '$1 == "U" { print $2 }' \
            | sed 's/@.*//' \
            | c++filt \
            | grep -E 'rmm::|raft::|^__cuda|^cuda[A-Z_]|^cu[A-Z]' || true
    )"
    if [[ -n "${undefined}" ]]; then
        echo "ERROR: undefined GPU-stack symbols in ${LIBRARY}:"
        sed 's/^/    /' <<< "${undefined}"
        failed=1
    fi

    needed="$(objdump -p "${LIBRARY}" | awk '/NEEDED/ { print $2 }')"

    gpu_needed="$(grep -E '^(librmm|libraft|libcudart|libcuda|libcublas|libcusparse|libcudss|libnccl|libnvrtc|libnvJitLink)' <<< "${needed}" || true)"
    if [[ -n "${gpu_needed}" ]]; then
        echo "ERROR: ${LIBRARY} has a GPU-stack DT_NEEDED entry:"
        sed 's/^/    /' <<< "${gpu_needed}"
        failed=1
    fi

    # It is the leaf every other component links, so it must depend on none of them.
    cuopt_needed="$(grep -E '^libcuopt' <<< "${needed}" || true)"
    if [[ -n "${cuopt_needed}" ]]; then
        echo "ERROR: ${LIBRARY} depends on another cuOpt library:"
        sed 's/^/    /' <<< "${cuopt_needed}"
        failed=1
    fi
fi

if [[ "${failed}" -ne 0 ]]; then
    exit 1
fi

echo "No symbol visibility issues found in ${LIBRARY}"
