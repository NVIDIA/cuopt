#!/bin/bash

# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

set -euo pipefail

# Install dependencies
su cuopt -c "pip install --user pytest"

# Download test data
bash datasets/linear_programming/download_pdlp_test_dataset.sh
bash datasets/mip/download_miplib_test_dataset.sh
cd datasets && ./get_test_data.sh --solomon && ./get_test_data.sh --tsp && cd -

# Test CLI
echo "Testing CLI"
su cuopt -c "export PATH=$PATH:/home/cuopt/.local/bin && export RAPIDS_DATASET_ROOT_DIR=$(realpath datasets) && bash python/libcuopt/libcuopt/tests/test_cli.sh"

# Test cuopt
echo "Testing cuopt"
su cuopt -c "RAPIDS_DATASET_ROOT_DIR=./datasets python -m pytest python/cuopt/cuopt/tests/routing/test_vehicle_routing.py"
su cuopt -c "RAPIDS_DATASET_ROOT_DIR=./datasets python -m pytest python/cuopt/cuopt/tests/linear_programming/test_lp_solver.py::test_solver"

# Test cuopt server
echo "Testing cuopt server"
su cuopt -c "RAPIDS_DATASET_ROOT_DIR=./datasets python -m pytest python/cuopt_server/cuopt_server/tests/test_server.py"