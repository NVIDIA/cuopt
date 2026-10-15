#!/bin/bash

# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

rapids-logger "Checking GPU availability"
nvidia-smi

rapids-logger "Running the Java build and tests"
ci/build_java.sh --run-java-tests
