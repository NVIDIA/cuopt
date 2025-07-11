#!/bin/bash

# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


git clone -b cuopt_solver https://github.com/tmckayus/cvxpy
pushd cvxpy || exit 1
pip install pytest-error-for-skips
pip install -e .
python -m pytest --error-for-skips cvxpy/tests/test_conic_solvers.py -k "TestCUOPT"
EXITCODE="$?"
if [ "$EXITCODE" -eq 0 ]; then
    echo PASSED smoketest
else
    echo FAILED smoketest
fi
popd || exit 1
exit $EXITCODE
