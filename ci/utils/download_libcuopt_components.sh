#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Downloads the libcuopt component wheels built earlier in this run and constrains pip to
# them.
#
# libcuopt is a metapackage over libcuopt-client / -mathopt / -routing. Those are new
# packages that no index carries yet, so without this any `pip install libcuopt` resolves
# its dependencies against PyPI and fails with "No matching distribution found for
# libcuopt-client-<suffix>".
#
# Source this after PIP_CONSTRAINT exists and after anything that truncates it. Appends one
# constraint per component and leaves their paths in LIBCUOPT_COMPONENT_WHEELS.

LIBCUOPT_COMPONENT_WHEELS=()

for _component in client mathopt routing; do
    _wheelhouse=$(rapids-download-from-github \
        "$(rapids-artifact-name wheel_cpp "libcuopt_${_component}" cuopt --cuda "$RAPIDS_CUDA_VERSION")")
    _wheel=$(echo "${_wheelhouse}"/libcuopt_"${_component}"_*.whl)
    echo "libcuopt-${_component}-${RAPIDS_PY_CUDA_SUFFIX} @ file://${_wheel}" >> "${PIP_CONSTRAINT}"
    LIBCUOPT_COMPONENT_WHEELS+=("${_wheel}")
done

export LIBCUOPT_COMPONENT_WHEELS
