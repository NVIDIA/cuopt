# SPDX-FileCopyrightText: Copyright (c) 2021-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

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
