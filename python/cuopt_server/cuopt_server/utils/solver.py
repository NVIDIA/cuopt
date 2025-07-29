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

import logging
import os
import queue
import time
from typing import List, Optional, Union

from fastapi import HTTPException
from fastapi.exceptions import RequestValidationError
from pydantic import ValidationError

import cuopt_server.utils.request_filter as request_filter
import cuopt_server.utils.settings as settings
from cuopt_server.utils.data_definition import (
    CostMatrices,
    FleetData,
    InitialSolution,
    LPData,
    SolverSettingsConfig,
    TaskData,
    WaypointGraphData,
)
from cuopt_server.utils.exceptions import (
    exception_handler,
    http_exception_handler,
    validation_exception_handler,
)
from cuopt_server.utils.job_queue import (
    CudaUnhealthy,
    SolverBinaryResponse,
    SolverIntermediateResponse,
)
from cuopt_server.utils.logutil import set_ncaid, set_requestid, set_solverid


# Return exception if validation fails
def check_valid(is_valid):
    if not is_valid[0]:
        raise HTTPException(status_code=400, detail=f"{is_valid[1]}")


# Wrap the solver response in a dictionary with a "response"
# field and add total_solve_time, request id, notes, and warnings to the
# dictionary if those values are set.
def make_response(
    response, warnings=[], notes=[], reqId="", total_solve_time=0
):
    r = {"response": response}
    if total_solve_time:
        r["response"]["total_solve_time"] = total_solve_time
    if reqId:
        r["reqId"] = reqId
    if warnings:
        r["warnings"] = warnings
    if notes:
        r["notes"] = notes
    return r


# Validate LP data and call the LP solver
def solve_LP_sync(
    LP_data: Union[LPData, List[LPData]],
    warmstart_data=None,
    warnings=[],
    validation_only=False,
    reqId="",
    intermediate_sender=None,
    solver_logging=False,
):
    from cuopt_server.utils.linear_programming.data_validation import (
        validate_LP_data,
    )
    from cuopt_server.utils.linear_programming.solver import solve as LP_solve

    begin_time = time.time()

    if isinstance(LP_data, list):
        for i_data in LP_data:
            validate_LP_data(i_data)
    else:
        validate_LP_data(LP_data)

    etl_end_time = time.time()
    logging.debug(f"etl_time {etl_end_time - begin_time}")

    if not validation_only:
        # log_file setting is ignored in the service,
        # instead we control it and use it as the basis for callbacks
        if isinstance(LP_data, list):
            # clear log_file setting for all because
            # we don't support callbacks for batch mode
            # and otherwise we ignore log_file
            for i_data in LP_data:
                i_data.solver_config.log_file = ""
        elif solver_logging:
            log_dir, _, _ = settings.get_result_dir()
            log_fname = "log_" + reqId
            log_file = os.path.join(log_dir, log_fname)
            logging.info(f"Writing logs to {log_file}")
            LP_data.solver_config.log_file = log_file
        elif LP_data.solver_config.log_file:
            warnings.append(
                "solver config log_file ignored in the cuopt service"
            )
            LP_data.solver_config.log_file = ""

        notes, addl_warnings, res, total_solve_time = LP_solve(
            LP_data,
            reqId,
            intermediate_sender,
            warmstart_data,
        )
        warnings.extend(addl_warnings)
    else:
        res = {"status": 0, "solution": {}}
        notes = ["Input is valid"]
        total_solve_time = 0

    solve_time = time.time() - etl_end_time
    solver_response = {"solver_response": res}
    etl_time = etl_end_time - begin_time

    full_response = make_response(
        solver_response, warnings, notes, reqId, total_solve_time
    )
    return full_response, etl_time, solve_time


def populate_optimization_data(
    cost_waypoint_graph_data: Optional[WaypointGraphData] = None,
    travel_time_waypoint_graph_data: Optional[WaypointGraphData] = None,
    cost_matrix_data: Optional[CostMatrices] = None,
    travel_time_matrix_data: Optional[CostMatrices] = None,
    fleet_data: Optional[FleetData] = None,
    task_data: Optional[TaskData] = None,
    # Use the update data structure for the sync endpoint because
    # it makes the time_limit value Optional
    initial_solution: Optional[List[InitialSolution]] = None,
    solver_config: Optional[SolverSettingsConfig] = None,
    warnings=[],
):
    from cuopt_server.utils.routing.optimization_data_model import (
        OptimizationDataModel,
    )
    from cuopt_server.utils.routing.solver import warn_on_objectives

    optimization_data = OptimizationDataModel()

    if (
        not cost_waypoint_graph_data
        or not cost_waypoint_graph_data.waypoint_graph
    ) and (not cost_matrix_data or not cost_matrix_data.data):
        raise HTTPException(
            status_code=400,
            detail="cost_matrix/waypoint_graph needs to be provided to find any route",  # noqa
        )

    if (
        cost_waypoint_graph_data and cost_waypoint_graph_data.waypoint_graph
    ) and (cost_matrix_data and cost_matrix_data.data):
        raise HTTPException(
            status_code=400,
            detail="only one of cost_matrix or waypoint_graph needs to be provided, not both",  # noqa
        )

    if (travel_time_matrix_data and travel_time_matrix_data.data) and (
        travel_time_waypoint_graph_data
        and travel_time_waypoint_graph_data.waypoint_graph
    ):
        raise HTTPException(
            status_code=400,
            detail="only one of travel_time_matrix_data or travel_time_waypoint_graph_data needs to be provided, not both",  # noqa
        )

    if cost_waypoint_graph_data and cost_waypoint_graph_data.waypoint_graph:
        check_valid(
            optimization_data.set_cost_waypoint_graph(
                cost_waypoint_graph_data.waypoint_graph
            )
        )
    elif cost_matrix_data and cost_matrix_data.data:
        check_valid(optimization_data.set_cost_matrix(cost_matrix_data.data))

    if (
        travel_time_waypoint_graph_data
        and travel_time_waypoint_graph_data.waypoint_graph
    ):
        check_valid(
            optimization_data.set_travel_time_waypoint_graph(
                travel_time_waypoint_graph_data.waypoint_graph
            )
        )
    elif travel_time_matrix_data and travel_time_matrix_data.data:
        check_valid(
            optimization_data.set_travel_time_matrix(
                travel_time_matrix_data.data
            )
        )

    if fleet_data is not None:
        check_valid(
            optimization_data.set_fleet_data(
                fleet_data.vehicle_ids,
                fleet_data.vehicle_locations,
                fleet_data.capacities,
                fleet_data.vehicle_time_windows,
                fleet_data.vehicle_breaks,
                fleet_data.vehicle_break_time_windows,
                fleet_data.vehicle_break_durations,
                fleet_data.vehicle_break_locations,
                fleet_data.vehicle_types,
                fleet_data.vehicle_order_match,
                fleet_data.skip_first_trips,
                fleet_data.drop_return_trips,
                fleet_data.min_vehicles,
                fleet_data.vehicle_max_costs,
                fleet_data.vehicle_max_times,
                fleet_data.vehicle_fixed_costs,
            )
        )

    if task_data is not None:
        check_valid(
            optimization_data.set_task_data(
                task_data.task_ids,
                task_data.task_locations,
                task_data.demand,
                task_data.pickup_and_delivery_pairs,
                task_data.task_time_windows,
                task_data.service_times,
                task_data.prizes,
                task_data.order_vehicle_match,
            )
        )

    if initial_solution is not None:
        check_valid(optimization_data.set_initial_solution(initial_solution))

    if solver_config is not None:
        if solver_config.time_limit is None:
            num_tasks = len(task_data.task_locations)
            solver_config.time_limit = request_filter.std_solver_time_calc(
                num_tasks
            )
            logging.debug(
                "Solver time limit not specified, "
                f"setting to {solver_config.time_limit}"
            )
        else:
            logging.debug(
                f"Using specified solver time {solver_config.time_limit}"
            )
        owarn, solver_config = warn_on_objectives(solver_config)
        warnings.extend(owarn)
        check_valid(
            optimization_data.set_solver_config(
                solver_config.time_limit,
                solver_config.objectives,
                solver_config.config_file,
                solver_config.verbose_mode,
                solver_config.error_logging,
            )
        )

    return optimization_data


def solve_optimized_routes_sync(
    cost_waypoint_graph_data: Optional[WaypointGraphData] = None,
    travel_time_waypoint_graph_data: Optional[WaypointGraphData] = None,
    cost_matrix_data: Optional[CostMatrices] = None,
    travel_time_matrix_data: Optional[CostMatrices] = None,
    fleet_data: Optional[FleetData] = None,
    task_data: Optional[TaskData] = None,
    initial_solution: Optional[List[InitialSolution]] = None,
    solver_config: Optional[SolverSettingsConfig] = None,
    validation_only: Optional[bool] = False,
    warnings=[],
    reqId="",
):

    from cuopt_server.utils.routing.solver import solve as routing_solve

    begin_time = time.time()

    optimization_data = populate_optimization_data(
        cost_waypoint_graph_data,
        travel_time_waypoint_graph_data,
        cost_matrix_data,
        travel_time_matrix_data,
        fleet_data,
        task_data,
        initial_solution,
        solver_config,
    )

    etl_end_time = time.time()

    logging.debug(f"etl_time {etl_end_time - begin_time}")

    total_solve_time = 0

    if not validation_only:
        notes, addl_warnings, res, total_solve_time = routing_solve(
            optimization_data
        )
        warnings.extend(addl_warnings)
    else:
        res = {
            "status": 0,
            "msg": "Input is Valid",
            "num_vehicles": -1,
            "solution_cost": -1,
            "objective_values": {},
            "vehicle_data": {},
        }
        notes = ["Input is valid"]

    solve_time = time.time() - etl_end_time

    if res["status"] == 0:
        solver_response = {"solver_response": res}
    else:
        solver_response = {"solver_infeasible_response": res}

    full_response = make_response(
        solver_response, warnings, notes, reqId, total_solve_time
    )

    etl_time = etl_end_time - begin_time
    return full_response, etl_time, solve_time


def process_async_solve(
    solver_exit, solver_complete, job_queue, results_queue, abort_list, gpu_id
):

    # Send incumbent solutions
    def send_solution(id, solution, cost):
        results_queue.put(
            SolverIntermediateResponse(
                id, {"solution": solution, "cost": cost}
            )
        )

    import os
    import signal

    if gpu_id:
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id
        set_solverid(gpu_id)

    signal.signal(signal.SIGCHLD, signal.SIG_IGN)
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    signal.signal(signal.SIGTERM, signal.SIG_IGN)

    # Initialize memory resource to use pool memory
    # upfront to make sure all memory allocations are
    # using the same pool allocator. cuOpt performs
    # significant number of memory allocations and
    # deallocations, especially when there are high
    # number of vehicles. Pool memory is efficient in
    # handling such memory allocation patterns
    import rmm

    pool_gigs = int(os.environ.get("CUOPT_GIGABYTES_PER_PROC", 1))

    # limit the pool size so that we allow running
    # multiple processes on same GPU simultaneously
    pool = rmm.mr.PoolMemoryResource(
        rmm.mr.CudaMemoryResource(), initial_pool_size=2**30 * pool_gigs
    )

    rmm.mr.set_current_device_resource(pool)

    # These are all the loggers touched by CUDA that we do not
    # want to hear from normally. The only practical way to build
    # this list is look at log output and find messages from
    # loggers that we do not want ...
    for lname in [
        "numba.cuda.cudadrv.driver",
        "ptxcompiler.patch",
        "ucx",
    ]:
        logging.getLogger(lname).setLevel(logging.WARN)

    def cuda_health_check():
        try:
            import cudf

            cudf.Series([1, 2, 3, 4])
            return True, ""
        except Exception as e:
            return False, str(e)

    def job_id(job):
        try:
            return job.id
        except Exception:
            return ""

    def check_and_send(cuda_healthy):
        if cuda_healthy:
            cuda_healthy, msg = cuda_health_check()
            if not cuda_healthy:
                logging.error(f"solver process unhealthy: {msg}")
                results_queue.put(CudaUnhealthy())
        return cuda_healthy

    logging.info(f"solver rmm pool size in gigabytes {pool_gigs}")

    try:
        # only log cuda health check message 1 per hour
        # set it to log once on startup
        job_queue_timeout = 30  # should be between 1 and 60
        cuda_log_threshold = int(60 / job_queue_timeout) * 60
        cuda_count = cuda_log_threshold

        cuda_healthy = True
        logging.info("solver waiting on job queue")
        while True:
            try:
                import os

                job = job_queue.get(timeout=job_queue_timeout)
                logging.info(
                    f"solver with {os.getpid()} received job {job_id(job)}"
                )
            except queue.Empty:
                cuda_count += 1
                if cuda_count >= cuda_log_threshold:
                    logging.info(
                        f"solver checking cuda health {cuda_log_threshold} "
                        "time(s) per hour"
                    )
                    cuda_count = 0
                cuda_healthy = check_and_send(cuda_healthy)
                if not cuda_healthy:
                    logging.error("solver process exiting")
                    break
                continue

            if solver_exit.is_set():
                break

            ncaid, reqid = job.get_nvcf_ids()
            set_ncaid(ncaid)
            set_requestid(reqid)

            value = abort_list.add_id_or_return(job.id, os.getpid())
            if value is not None:
                # Just skip this job, it's already been aborted
                logging.info(f"solver skipping {job.id}")
                # data might be shared memory so call delete ...
                job.delete_data()
                results_queue.put(SolverBinaryResponse(job.id))
                continue

            cuda_healthy = check_and_send(cuda_healthy)
            if not cuda_healthy:
                logging.error("solver process exiting")
                break

            try:
                success = False
                etl = slv = 0
                ans, etl, slv = job.solve(send_solution)
                success = True
            except (RequestValidationError, ValidationError) as e:
                ans = validation_exception_handler(e)
            except HTTPException as e:
                ans = http_exception_handler(e)
            except Exception as e:
                ans = exception_handler(e)
            logging.info(
                f"solver sending response for job {job.id} success {success}"
            )
            abort_list.update(job.id)
            results_queue.put(
                SolverBinaryResponse(
                    job.id,
                    ans,
                    job.get_result_mime_type(),
                    etl,
                    slv,
                    job.get_sku(),
                    ncaid,
                    reqid,
                    # In the case of cuopt/cuopt endpoint, there
                    # are values that we do not get until the data is
                    # loaded. Read those here and pass them back to
                    # the webserver in the result
                    job.get_action(),
                    job.is_validator_enabled(),
                )
            )
            logging.info("solver waiting on job queue")

    except Exception as e:
        exception_handler(e)
        logging.error(f"solver process exiting on exception {str(e)}")
    solver_complete.set()
    logging.info("solver process finished")
