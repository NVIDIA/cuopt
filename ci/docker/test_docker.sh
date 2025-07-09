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
su cuopt -c "pip install --user pytest pexpect"

# Download test data
#bash datasets/linear_programming/download_pdlp_test_dataset.sh
#bash datasets/mip/download_miplib_test_dataset.sh
#cd datasets && ./get_test_data.sh --solomon && ./get_test_data.sh --tsp && cd -

ln -sf "$(pwd)" /home/cuopt/cuopt



# Test CLI
echo "----------------- CLI TEST START ---------------"
su - cuopt -c "cd cuopt && export PATH=$PATH:/home/cuopt/.local/bin && export RAPIDS_DATASET_ROOT_DIR=$(realpath datasets) && bash python/libcuopt/libcuopt/tests/test_cli.sh"
echo "----------------- CLI TEST END ---------------"

# Test cuopt
echo "----------------- CUOPT TEST START ---------------"
su - cuopt -c "cd cuopt && RAPIDS_DATASET_ROOT_DIR=./datasets python -m pytest python/cuopt/cuopt/tests/"
echo "----------------- CUOPT TEST END ---------------"

# Test cuopt server
echo "----------------- CUOPT SERVER TEST START ---------------"
su - cuopt -c "cd cuopt && RAPIDS_DATASET_ROOT_DIR=./datasets python -m pytest python/cuopt_server/cuopt_server/tests/"
echo "----------------- CUOPT SERVER TEST END ---------------"