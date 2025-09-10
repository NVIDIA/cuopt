#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2021-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

# Creates a conda environment to be used for cuopt benchmarking.

# Abort script on first error
set -e

# Must ensure PROJECT_DIR is exported first then load rapids-mg-tools env
export PROJECT_DIR=${PROJECT_DIR:-$(cd $(dirname ${BASH_SOURCE[0]}) && pwd)}

################################################################################

# Test
logger "Testing container image $IMAGE"
python -c "import cuopt; print(cuopt)"

# Other scripts look for this to be the last line to determine if this
# script completed successfully. This is only possible because of the
# "set -e" above.
echo "done."
logger "done."
