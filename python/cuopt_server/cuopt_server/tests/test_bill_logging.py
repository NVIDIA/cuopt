# SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import os
import platform
import shutil
import signal

import pexpect
import pytest

from cuopt_server import cuopt_service
from cuopt_server.tests.utils import utils
from cuopt_server.tests.utils.utils import RequestClient

# Find where is server script
server_script = cuopt_service.__file__
python_path = shutil.which("python")


@pytest.mark.skipif(
    "x86" not in platform.uname().machine,
    reason="Billing logs are not currently supported on non-x86_64 architectures (they take too long to run)",  # noqa: E501
)
def test_time_limit_logs():
    dataset_path = (
        utils.RAPIDS_DATASET_ROOT_DIR
        + "/cuopt_service_data/cuopt_problem_data.json"
    )
    dataset = json.load(open(dataset_path))
    client = RequestClient(port=5010)
    env = os.environ.copy()
    env["CUOPT_MAX_SOLVE_TIME_LIMIT"] = "5"

    url = "/cuopt/"

    cmd = python_path + " " + server_script + " -i 127.0.0.1 -p 5010 -l debug"
    process = pexpect.spawn(cmd, env=env)

    try:
        process.expect("Uvicorn running on", timeout=60, searchwindowsize=500)
    except Exception:
        pass
    finally:
        assert client.get(url + "health").status_code == 200

        # /cuopt/cuopt endpoint, time limit omitted
        dataset["solver_config"]["time_limit"] = None
        client.post(
            url + "cuopt",
            json={"action": "cuOpt_OptimizedRouting", "data": dataset},
        )
        process.expect(
            "Solver time limit not specified, setting to", timeout=60
        )

        # /cuopt/cuopt endpoint, time limit modified (too high)
        dataset["solver_config"]["time_limit"] = 100
        client.post(
            url + "cuopt",
            json={"action": "cuOpt_OptimizedRouting", "data": dataset},
        )
        process.expect("Solver time modified to", timeout=60)

        # /cuopt/cuopt endpoint, use specified (low)
        dataset["solver_config"]["time_limit"] = 0.1
        client.post(
            url + "cuopt",
            json={"action": "cuOpt_OptimizedRouting", "data": dataset},
        )
        process.expect("Using specified solver time 0.1", timeout=60)

        # /cuopt/cuopt endpoint, solver_config omitted
        save_config = dataset["solver_config"]
        del dataset["solver_config"]
        client.post(
            url + "cuopt",
            json={"action": "cuOpt_OptimizedRouting", "data": dataset},
        )
        process.expect("Creating default solver config")
        process.expect(
            "Solver time limit not specified, setting to", timeout=60
        )

        # /cuopt/cuopt solver_config none
        dataset["solver_config"] = None
        client.post(
            url + "cuopt",
            json={"action": "cuOpt_OptimizedRouting", "data": dataset},
        )
        process.expect("Creating default solver config")
        process.expect(
            "Solver time limit not specified, setting to", timeout=60
        )

        # /cuopt/request, time limit omitted
        dataset["solver_config"] = save_config
        dataset["solver_config"]["time_limit"] = None
        client.post(url + "request", json=dataset)
        process.expect(
            "Solver time limit not specified, setting to", timeout=60
        )

        # /cuopt/request, use specified time
        dataset["solver_config"]["time_limit"] = 0.1
        client.post(url + "request", json=dataset)
        process.expect("Using specified solver time 0.1", timeout=60)

        # /cuopt/request, solver config omitted
        del dataset["solver_config"]
        client.post(url + "request", json=dataset)
        process.expect("Creating default solver config")
        process.expect(
            "Solver time limit not specified, setting to", timeout=60
        )

        # /cuopt/request, solver config omitted
        dataset["solver_config"] = None
        client.post(url + "request", json=dataset)
        process.expect("Creating default solver config")
        process.expect(
            "Solver time limit not specified, setting to", timeout=60
        )
        process.kill(signal.SIGINT)
        process.wait()
        if process.isalive():
            process.kill(signal.SIGTERM)
