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

n_gpus = len(cuda.list_devices())

for filename in glob.iglob("**/*.ipynb", recursive=True):
    skip = False

    if "/build/" in filename:
        skip = True

    # A notebook that asks the solver for more GPUs than are visible (num_gpus=4) is aborted by
    # cuopt_cli, so it can only run on a node that has them.
    required_gpus = 1
    for line in open(filename, "r"):
        if re.search("# Skip notebook test", line):
            skip = True
            print(f"SKIPPING {filename} (marked as skip)", file=sys.stderr)
            break
        for n in re.findall(r"num_gpus\s*=\s*(\d+)", line):
            required_gpus = max(required_gpus, int(n))

    if not skip and required_gpus > n_gpus:
        skip = True
        print(
            f"SKIPPING {filename} (needs {required_gpus} GPUs, {n_gpus} visible)",
            file=sys.stderr,
        )

    if not skip:
        print(filename)
