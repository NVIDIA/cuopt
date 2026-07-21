#!/bin/bash

# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

# Detect distro and install test dependencies
if [ -f /etc/redhat-release ]; then
    dnf install -y file bzip2 gcc wget unzip
    dnf clean all
    # pip-installed CUDA wheels land in a non-standard prefix on UBI/RHEL.
    # su - resets the environment, so carry the path forward explicitly.
    EXTRA_LD_PATH="/usr/local/lib/python3.12/site-packages/nvidia/cu13/lib"
else
    apt-get update
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends file bzip2 gcc
    EXTRA_LD_PATH=""
fi

# Download test data
bash datasets/linear_programming/download_pdlp_test_dataset.sh
bash datasets/mip/download_miplib_test_dataset.sh
pushd ./datasets
./get_test_data.sh --solomon
./get_test_data.sh --tsp
popd

# Create symlink to cuopt
ln -sf "$(pwd)" /opt/cuopt/cuopt

# Set permissions since the repo is mounted on root
chmod -R a+w "$(pwd)"

cat > /opt/cuopt/test.sh <<EOF
set -euo pipefail

# Restore library path stripped by su - login shell
if [ -n "${EXTRA_LD_PATH}" ]; then
    export LD_LIBRARY_PATH="${EXTRA_LD_PATH}:\${LD_LIBRARY_PATH:-}"
fi

cd /opt/cuopt/cuopt
python -m pip install pytest pexpect
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

# Create a temporary user with UID 1001 (within standard range for both Ubuntu and RHEL)
useradd -m -u 1001 -s /bin/bash tempuser1001 2>/dev/null || true

# Switch to it
su - tempuser1001 -c "bash /opt/cuopt/test.sh"
