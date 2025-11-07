# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os

from cuopt_sh_client import CuOptServiceSelfHostClient


def test_log_callback():
    port = os.environ.get("CUOPT_SERVER_PORT", 5000)

    client_cert = os.environ.get("CLIENT_CERT", "")
    use_https = client_cert != ""

    data = {
        "csr_constraint_matrix": {
            "offsets": [0, 2],
            "indices": [0, 1],
            "values": [1.0, 1.0],
        },
        "constraint_bounds": {"upper_bounds": [5000.0], "lower_bounds": [0.0]},
        "objective_data": {
            "coefficients": [1.2, 1.7],
            "scalability_factor": 1.0,
            "offset": 0.0,
        },
        "variable_bounds": {
            "upper_bounds": [3000.0, 5000.0],
            "lower_bounds": [0.0, 0.0],
        },
        "maximize": "True",
        "variable_names": ["x", "y"],
        "variable_types": ["I", "I"],
        "solver_config": {
            "time_limit": 10,
            "tolerances": {"optimality": 0.0001},
        },
    }

    def callback(saw_callback):
        def callback(log):
            assert isinstance(log, list)
            assert len(log) > 0
            assert isinstance(log[0], str)
            saw_callback["yes"] = True

        return callback

    client = CuOptServiceSelfHostClient(
        port=port, use_https=use_https, self_signed_cert=client_cert
    )

    saw_callback = {"yes": False}
    lp_exception = True
    try:
        client.get_LP_solve(data, logging_callback=callback(saw_callback))
        lp_exception = False
    except Exception as e:
        # Add the assert this way so pytest will show us the exception
        # msg if any in the log and still fail
        assert str(e) == ""

    assert not lp_exception
    assert saw_callback["yes"]
