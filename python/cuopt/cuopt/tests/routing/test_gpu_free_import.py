# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import subprocess
import sys


def test_routing_imports_without_a_gpu():
    """cuopt.routing (and its dataset helpers) must import on a GPU-less host.

    The dataset helpers build empty cudf.Series objects; doing so in a default
    argument would run at import time and fail with cudaErrorNoDevice where no
    GPU is visible. This guards that the construction stays inside the function
    bodies. Run in a subprocess with no visible GPU for a faithful check.
    """
    code = (
        "import cuopt.routing as r\n"
        "assert callable(r.generate_dataset)\n"
        "assert r.DatasetDistribution.CLUSTERED is not None\n"
    )
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": ""}
    subprocess.run([sys.executable, "-c", code], check=True, env=env)
