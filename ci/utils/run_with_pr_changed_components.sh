#!/bin/bash

# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Export changed-file group flags from pr.yaml, then run a test script from the repo root.
#
# RAPIDS reusable workflows (conda-*-tests, wheels-test) run the job `script` as unquoted
# $INPUTS_SCRIPT, so multiline shell is word-split and breaks. In pr.yaml use a folded
# block scalar (>-) so this invocation is one physical line to Actions but readable:
#
#   script: >-
#     ci/utils/run_with_pr_changed_components.sh
#     ${{ fromJSON(needs.changed-files.outputs.changed_file_groups).test_routing || fromJSON(needs.changed-files.outputs.changed_file_groups).test_shared }}
#     ${{ fromJSON(needs.changed-files.outputs.changed_file_groups).test_lp || fromJSON(needs.changed-files.outputs.changed_file_groups).test_shared }}
#     ${{ fromJSON(needs.changed-files.outputs.changed_file_groups).test_mip || fromJSON(needs.changed-files.outputs.changed_file_groups).test_shared }}
#     ci/test_example.sh
#
# Optional extra args are forwarded to the test script.

set -euo pipefail

export CUOPT_ROUTING_CHANGED="${1}"
export CUOPT_LP_CHANGED="${2}"
export CUOPT_MIP_CHANGED="${3}"
shift 3

cd "$(dirname "${BASH_SOURCE[0]}")/../.."
exec bash "$@"
