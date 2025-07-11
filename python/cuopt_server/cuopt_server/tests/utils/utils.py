# SPDX-FileCopyrightText: Copyright (c) 2022-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.  # noqa
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

import os
import shutil
import signal
import time
from subprocess import Popen, TimeoutExpired
from typing import Dict, List, Optional

import pytest
import requests

RAPIDS_DATASET_ROOT_DIR = os.getenv("RAPIDS_DATASET_ROOT_DIR")
if RAPIDS_DATASET_ROOT_DIR is None:
    RAPIDS_DATASET_ROOT_DIR = os.path.dirname(os.getcwd())
    RAPIDS_DATASET_ROOT_DIR = os.path.join(RAPIDS_DATASET_ROOT_DIR, "datasets")


def generate_json_data(**args):
    return {arg[0]: arg[1] for arg in args.items() if arg[1] is not None}


def delete_request(client, reqId, queued=None, running=None):

    params = {}
    if queued is not None:
        params["queued"] = queued

    if running is not None:
        params["running"] = running
    headers = {"Accept": "application/json"}
    return client.delete(
        f"/cuopt/request/{reqId}", headers=headers, params=params
    )


def poll_request(client, reqId):
    return client.get(f"/cuopt/solution/{reqId}")


def get_lp(client, data):

    headers = {"CLIENT-VERSION": "custom"}

    return client.post("/cuopt/request", headers=headers, params={}, json=data)


def get_routes(
    client,
    cost_waypoint_graph: Optional[Dict] = None,
    travel_time_waypoint_graph: Optional[Dict] = None,
    cost_matrix: Optional[Dict[int, List[List[float]]]] = None,
    travel_time_matrix: Optional[Dict[int, List[List[float]]]] = None,
    vehicle_locations: Optional[List[List[int]]] = None,
    vehicle_ids: Optional[List[str]] = None,
    capacities: Optional[List[List[int]]] = None,
    vehicle_time_windows: Optional[List[List[float]]] = None,
    vehicle_breaks: Optional[List[Dict]] = None,
    vehicle_break_time_windows: Optional[List[List[List[int]]]] = None,
    vehicle_break_durations: Optional[List[List[int]]] = None,
    vehicle_break_locations: Optional[List[int]] = None,
    vehicle_types: Optional[List[int]] = None,
    vehicle_order_match: Optional[List[dict]] = None,
    skip_first_trips: Optional[List[bool]] = None,
    drop_return_trips: Optional[List[bool]] = None,
    min_vehicles: Optional[int] = None,
    vehicle_max_costs: Optional[List[float]] = None,
    vehicle_max_times: Optional[List[float]] = None,
    vehicle_fixed_costs: Optional[List[float]] = None,
    task_locations: Optional[List[int]] = None,
    demand: Optional[List[List[int]]] = None,
    pickup_and_delivery_pairs: Optional[List[List[int]]] = None,
    task_time_windows: Optional[List[List[float]]] = None,
    service_times: Optional[List[float]] = None,
    prizes: Optional[List[float]] = None,
    order_vehicle_match: Optional[List[dict]] = None,
    time_limit: Optional[float] = None,
    objectives: Optional[dict] = None,
    config_file: Optional[str] = None,
    verbose_mode: Optional[bool] = None,
    error_logging: Optional[bool] = None,
    validation_only: Optional[bool] = False,
    cache: Optional[bool] = None,
    reqId: Optional[str] = None,
    result_timeout: Optional[int] = None,
    initialId: Optional[List[str]] = None,
    delete=True,
):
    options = {}
    options["cost_waypoint_graph_data"] = generate_json_data(
        waypoint_graph=cost_waypoint_graph
    )

    options["travel_time_waypoint_graph_data"] = generate_json_data(
        waypoint_graph=travel_time_waypoint_graph
    )

    options["cost_matrix_data"] = generate_json_data(data=cost_matrix)

    options["travel_time_matrix_data"] = generate_json_data(
        data=travel_time_matrix
    )

    # fleet data
    options["fleet_data"] = generate_json_data(
        vehicle_ids=vehicle_ids,
        vehicle_locations=vehicle_locations,
        capacities=capacities,
        vehicle_time_windows=vehicle_time_windows,
        vehicle_breaks=vehicle_breaks,
        vehicle_break_time_windows=vehicle_break_time_windows,
        vehicle_break_durations=vehicle_break_durations,
        vehicle_break_locations=vehicle_break_locations,
        vehicle_types=vehicle_types,
        vehicle_order_match=vehicle_order_match,
        skip_first_trips=skip_first_trips,
        drop_return_trips=drop_return_trips,
        min_vehicles=min_vehicles,
        vehicle_max_costs=vehicle_max_costs,
        vehicle_max_times=vehicle_max_times,
        vehicle_fixed_costs=vehicle_fixed_costs,
    )

    # task data
    options["task_data"] = generate_json_data(
        task_locations=task_locations,
        demand=demand,
        pickup_and_delivery_pairs=pickup_and_delivery_pairs,
        task_time_windows=task_time_windows,
        service_times=service_times,
        prizes=prizes,
        order_vehicle_match=order_vehicle_match,
    )

    # solver config
    options["solver_config"] = generate_json_data(
        time_limit=time_limit,
        objectives=objectives,
        config_file=config_file,
        verbose_mode=verbose_mode,
        error_logging=error_logging,
    )

    params = {"validation_only": validation_only}

    if cache is not None:
        params["cache"] = cache
    if reqId is not None:
        params["reqId"] = reqId
    if initialId is not None:
        params["initialId"] = initialId

    headers = {"CLIENT-VERSION": "custom"}

    return client.post(
        "/cuopt/request",
        headers=headers,
        params=params,
        json=options,
        block=cache is None and result_timeout is None,
        delete=delete,
    )


def cuopt_service_sync(
    client,
    action: str,
    cost_waypoint_graph: Optional[Dict] = None,
    travel_time_waypoint_graph: Optional[Dict] = None,
    cost_matrix: Optional[Dict[int, List[List[float]]]] = None,
    travel_time_matrix: Optional[Dict[int, List[List[float]]]] = None,
    vehicle_locations: Optional[List[List[int]]] = None,
    vehicle_ids: Optional[List[str]] = None,
    capacities: Optional[List[List[int]]] = None,
    vehicle_time_windows: Optional[List[List[float]]] = None,
    vehicle_breaks: Optional[List[Dict]] = None,
    vehicle_break_time_windows: Optional[List[List[List[int]]]] = None,
    vehicle_break_durations: Optional[List[List[int]]] = None,
    vehicle_break_locations: Optional[List[int]] = None,
    vehicle_types: Optional[List[int]] = None,
    vehicle_order_match: Optional[List[dict]] = None,
    skip_first_trips: Optional[List[bool]] = None,
    drop_return_trips: Optional[List[bool]] = None,
    min_vehicles: Optional[int] = None,
    vehicle_max_costs: Optional[List[float]] = None,
    vehicle_max_times: Optional[List[float]] = None,
    vehicle_fixed_costs: Optional[List[float]] = None,
    task_locations: Optional[List[int]] = None,
    demand: Optional[List[List[int]]] = None,
    pickup_and_delivery_pairs: Optional[List[List[int]]] = None,
    task_time_windows: Optional[List[List[float]]] = None,
    service_times: Optional[List[float]] = None,
    prizes: Optional[List[float]] = None,
    order_vehicle_match: Optional[List[dict]] = None,
    time_limit: Optional[float] = None,
    objectives: Optional[dict] = None,
    config_file: Optional[str] = None,
    verbose_mode: Optional[bool] = None,
    error_logging: Optional[bool] = None,
):
    options = {}
    options["cost_waypoint_graph_data"] = generate_json_data(
        waypoint_graph=cost_waypoint_graph
    )

    options["travel_time_waypoint_graph_data"] = generate_json_data(
        waypoint_graph=travel_time_waypoint_graph
    )

    options["cost_matrix_data"] = generate_json_data(data=cost_matrix)

    options["travel_time_matrix_data"] = generate_json_data(
        data=travel_time_matrix
    )

    # fleet data
    options["fleet_data"] = generate_json_data(
        vehicle_ids=vehicle_ids,
        vehicle_locations=vehicle_locations,
        capacities=capacities,
        vehicle_time_windows=vehicle_time_windows,
        vehicle_breaks=vehicle_breaks,
        vehicle_break_time_windows=vehicle_break_time_windows,
        vehicle_break_durations=vehicle_break_durations,
        vehicle_break_locations=vehicle_break_locations,
        vehicle_types=vehicle_types,
        vehicle_order_match=vehicle_order_match,
        skip_first_trips=skip_first_trips,
        drop_return_trips=drop_return_trips,
        min_vehicles=min_vehicles,
        vehicle_max_costs=vehicle_max_costs,
        vehicle_max_times=vehicle_max_times,
        vehicle_fixed_costs=vehicle_fixed_costs,
    )

    # task data
    options["task_data"] = generate_json_data(
        task_locations=task_locations,
        demand=demand,
        pickup_and_delivery_pairs=pickup_and_delivery_pairs,
        task_time_windows=task_time_windows,
        service_times=service_times,
        prizes=prizes,
        order_vehicle_match=order_vehicle_match,
    )

    # solver config
    options["solver_config"] = generate_json_data(
        time_limit=time_limit,
        objectives=objectives,
        config_file=config_file,
        verbose_mode=verbose_mode,
        error_logging=error_logging,
    )

    cuopt_service_data = generate_json_data(
        action=action,
        data=options,
    )

    return client.post(
        "/cuopt/cuopt",
        headers={"CLIENT-VERSION": "custom"},
        json=cuopt_service_data,
    )


# Fixture and client to allow full cuopt service
# to run as a separate process for multiple tests
cuoptmain = None
# Use module name instead of file path to ensure we use the installed package
server_script = "-m"
server_module = "cuopt_server.cuopt_service"
python_path = shutil.which("python")


def cleanup_cuopt_process():
    """Clean up the cuopt process if it's still running"""
    global cuoptmain
    if cuoptmain and cuoptmain.poll() is None:
        cuoptmain.terminate()
        try:
            cuoptmain.wait(timeout=5)
        except TimeoutExpired:
            cuoptmain.kill()
            cuoptmain.wait()


def signal_handler(signum, frame):
    """Handle interrupt signals to ensure cleanup"""
    cleanup_cuopt_process()
    exit(1)


# Register signal handlers for cleanup
signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)


def spinup_wait():
    client = RequestClient()
    count = 0
    result = None
    while True:
        count += 1
        if count == 30:
            break
        try:
            result = client.get("/cuopt/health")
            break
        except Exception:
            time.sleep(1)
    assert result.status_code == 200


@pytest.fixture(scope="session")
def cuoptproc(request):
    global cuoptmain
    env = {
        "CUOPT_SERVER_IP": "0.0.0.0",
        "CUOPT_SERVER_PORT": "5555",
        "CUOPT_SERVER_LOG_LEVEL": "debug",
    }
    cuoptmain = Popen([python_path, server_script, server_module], env=env)
    spinup_wait()

    def shutdown():
        cleanup_cuopt_process()

    request.addfinalizer(shutdown)


class RequestClient:
    def __init__(self, port=5555):
        self.ip = "127.0.0.1"
        self.port = port
        self.url = f"http://{self.ip}:{self.port}"

    def poll_for_completion(self, reqId, delete=True):
        # Wait to complete
        cnt = 0
        headers = {"Accept": "application/json"}
        while True:
            res = requests.get(
                self.url + f"/cuopt/solution/{reqId}", headers=headers
            )
            if "response" in res.json() or "error" in res.json():
                break
            time.sleep(1)
            cnt += 1
            if cnt == 600:
                break
        if delete:
            try:
                requests.delete(
                    self.url + f"/cuopt/solution/{reqId}", headers=headers
                )
            except Exception:
                pass
        return res

    def post(
        self,
        endpoint,
        params=None,
        headers=None,
        json=None,
        data=None,
        block=True,
        delete=True,
    ):
        res = requests.post(
            self.url + endpoint,
            params=params,
            headers=headers,
            json=json,
            data=data,
        )

        # cuopt/cuot is already blocking, don't ever poll
        if endpoint == "/cuopt/cuopt":
            block = False

        if (
            not block
            or res.status_code != 200
            or (headers and "cache" in headers and headers["cache"] is True)
        ):
            return res

        return self.poll_for_completion(res.json()["reqId"], delete)

    def get(self, endpoint, headers=None, json=None):
        return requests.get(self.url + endpoint, headers=headers, json=json)

    def delete(self, endpoint, headers=None, json=None, params=None):
        return requests.delete(
            self.url + endpoint, params=params, headers=headers, json=json
        )
