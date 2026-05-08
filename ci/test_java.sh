#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Run cuopt-java tests in CI.
#
# Currently a thin wrapper that delegates to ci/build_java.sh with the
# --run-java-tests flag (matches cuvs's pattern). The build/test split
# tracked in cuvs Issue #868 is a deferred follow-up — switching to
# pre-built artifact consumption requires shared-workflow surface area
# for cross-job artifact download that doesn't exist yet.

set -euo pipefail

EXITCODE=0
trap "EXITCODE=1" ERR
set +e

rapids-logger "Check GPU usage"
nvidia-smi

rapids-logger "Run Java build and tests"

# TODO(java-build-test-split): switch to consuming a prebuilt JAR from
# conda-java-build job. Requires custom-job.yaml to expose artifact
# download or use of actions/download-artifact in this script.
# Ref: cuvs Issue rapidsai/cuvs#868
ci/build_java.sh --run-java-tests

rapids-logger "Test script exiting with value: $EXITCODE"
exit ${EXITCODE}
