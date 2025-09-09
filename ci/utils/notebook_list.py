# SPDX-FileCopyrightText: Copyright (c) 2021-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.  # noqa
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

import glob
import re
import sys

from numba import cuda

#
# Not strictly true... however what we mean is
# Pascal or earlier
#
pascal = False

device = cuda.get_current_device()
cc = device.compute_capability
if cc[0] < 7:
    pascal = True

for filename in glob.iglob("**/*.ipynb", recursive=True):
    skip = False

    if "/build/" in filename:
        skip = True

    for line in open(filename, "r"):
        if re.search("# Skip notebook test", line):
            skip = True
            print(f"SKIPPING {filename} (marked as skip)", file=sys.stderr)
            break

    if not skip:
        print(filename)
