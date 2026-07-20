# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import subprocess
import sys


def _run_without_gpu(code):
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": ""}
    subprocess.run([sys.executable, "-c", code], check=True, env=env)


def test_routing_imports_without_a_gpu():
    """cuopt.routing must import and build a model on a GPU-less host.

    The GPU dataset/batch helpers live in cuopt.routing.utils and are not
    re-exported by the package, so importing cuopt.routing must not pull in
    utils / utils_wrapper (which need a GPU). A CPU-only client can then build
    and serialize a problem to submit to a remote solver. Run in a subprocess
    with no visible GPU for a faithful check.
    """
    _run_without_gpu(
        "import sys\n"
        "import cuopt.routing as r\n"
        "assert 'cuopt.routing.utils' not in sys.modules\n"
        "assert 'cuopt.routing.utils_wrapper' not in sys.modules\n"
        # the GPU helpers are intentionally not part of the public surface
        "assert not hasattr(r, 'generate_dataset')\n"
        "import numpy as np\n"
        "dm = r.DataModel(3, 2)\n"
        "dm.add_cost_matrix(np.eye(3, dtype=np.float32))\n"
    )
