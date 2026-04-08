#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Derives CUOPT_TEST_COMPONENTS from changed-files env vars
# (CUOPT_ROUTING_CHANGED, CUOPT_LP_CHANGED, CUOPT_MIP_CHANGED).
#
# When none of these env vars are set (e.g. nightly / non-PR builds),
# defaults to "all" so every test runs.
#
# Usage:  source ./ci/utils/derive_test_components.sh

if [[ -z "${CUOPT_ROUTING_CHANGED:-}" && -z "${CUOPT_LP_CHANGED:-}" && -z "${CUOPT_MIP_CHANGED:-}" ]]; then
    export CUOPT_TEST_COMPONENTS="all"
else
    components=""
    [[ "${CUOPT_ROUTING_CHANGED:-}" == "true" ]] && components="${components:+${components},}routing"
    [[ "${CUOPT_LP_CHANGED:-}" == "true" ]]      && components="${components:+${components},}lp"
    [[ "${CUOPT_MIP_CHANGED:-}" == "true" ]]      && components="${components:+${components},}mip"
    # MIP is validated through LP tests (no separate MIP Python tests),
    # so always include LP when MIP changes.
    if [[ "${CUOPT_MIP_CHANGED:-}" == "true" && "${components}" != *"lp"* ]]; then
        components="${components:+${components},}lp"
    fi
    # Fallback to "all" if all components are false (defensive — the job-level
    # 'if' gate in pr.yaml would normally skip the job entirely in this case).
    export CUOPT_TEST_COMPONENTS="${components:-all}"
fi
echo "CUOPT_TEST_COMPONENTS=${CUOPT_TEST_COMPONENTS}"
