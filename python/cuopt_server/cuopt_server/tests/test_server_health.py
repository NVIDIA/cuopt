# SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import shutil

import pexpect
import pytest
import requests

from cuopt_server import cuopt_service

# Find where is server script
server_script = cuopt_service.__file__
python_path = shutil.which("python")


@pytest.mark.parametrize("args", ["default", "cmdline", "env"])
def test_server_health(args):
    env = None
    if args == "env":
        env = os.environ.copy()
        env.update(
            {
                "CUOPT_SERVER_IP": "127.0.0.1",
                "CUOPT_SERVER_PORT": "5001",
                "CUOPT_SERVER_LOG_LEVEL": "error",
            }
        )
        url = (
            "http://"
            + env["CUOPT_SERVER_IP"]
            + ":"
            + env["CUOPT_SERVER_PORT"]
            + "/cuopt/"
        )
        cmd = python_path + " " + server_script
    elif args == "cmdline":
        ip = "127.0.0.1"
        port = "5002"
        url = "http://" + ip + ":" + port + "/cuopt/"
        cmd = (
            python_path
            + " "
            + server_script
            + " -i "
            + ip
            + " -p "
            + port
            + " -l info"
        )  # noqa
    else:
        url = "http://0.0.0.0:5000/cuopt/"
        cmd = python_path + " " + server_script

    process = pexpect.spawn(cmd, env=env)
    try:
        process.expect(".*Uvicorn running on.*", timeout=60)
    except Exception:
        pass
    res = requests.get(url + "health").status_code
    process.terminate()
    assert res == 200
