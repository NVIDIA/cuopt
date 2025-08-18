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

chsh -s /bin/bash cuopt

# Install dependencies
apt-get update
DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends file bzip2

# Download test data
bash datasets/linear_programming/download_pdlp_test_dataset.sh
bash datasets/mip/download_miplib_test_dataset.sh
pushd ./datasets
./get_test_data.sh --solomon
./get_test_data.sh --tsp
popd

# Create symlink to cuopt
ln -sf "$(pwd)" /home/cuopt/cuopt

# Set permissions since the repo is mounted on root
chmod -R a+w "$(pwd)"

# If this script is being run as root, use 'su - cuopt -c "<command>"' to run each command as cuopt.

# Change to cuopt home directory and then to cuopt repo
cat > /home/cuopt/test.sh <<EOF
cd ~/cuopt
pip install --user pytest pexpect
export PATH=\$PATH:/home/cuopt/.local/bin
export RAPIDS_DATASET_ROOT_DIR=\$(realpath datasets)
echo '----------------- CLI TEST START ---------------'
bash python/libcuopt/libcuopt/tests/test_cli.sh
echo '----------------- CLI TEST END ---------------'
echo '----------------- CUOPT TEST START ---------------'
python -m pytest python/cuopt/cuopt/tests/linear_programming
python -m pytest python/cuopt/cuopt/tests/routing
echo '----------------- CUOPT TEST END ---------------'
echo '----------------- CUOPT SERVER TEST START ---------------'
python -m pytest python/cuopt_server/cuopt_server/tests/
echo '----------------- CUOPT SERVER TEST END ---------------'
EOF
su - cuopt -c "bash ~/test.sh"
