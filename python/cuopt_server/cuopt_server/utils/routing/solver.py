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

import time
from typing import Optional

import numpy as np
from fastapi import HTTPException

import cudf
from cuopt import distance_engine, routing
from cuopt.routing import ErrorStatus
from cuopt.utilities import (
    InputRuntimeError,
    InputValidationError,
    OutOfMemoryError,
)

from cuopt_server.utils.routing.initial_solution import parse_initial_sol
from cuopt_server.utils.routing.optimization_data_model import (
    OptimizationDataModel,
    objective_names,
)

dep_warning = (
    "{field} is deprecated and will be removed in the next release. Ignored."
)


def warn_on_objectives(solver_config):
    warnings = []
    return warnings, solver_config


# Create routes as waypoint sequence from sequence of task locations
def create_waypoint_sequence_routes(
    optimization_data, solution_routes, waypoint_graph
):
    v_routes = {}
    way_point_seq_df = {}
    if optimization_data.fleet_data["vehicle_types"] is not None:
        v_types = [
            optimization_data.fleet_data["vehicle_types"].iloc[i]
            for i in solution_routes["truck_id"].to_arrow().to_pylist()
        ]
    else:
        v_types = [list(waypoint_graph.keys())[0]] * len(solution_routes)

    solution_routes["vehicle_types"] = v_types
    for v_type in set(v_types):
        v_routes[v_type] = solution_routes[
            solution_routes["vehicle_types"] == v_type
        ]
        way_point_seq_df[v_type] = waypoint_graph[
            v_type
        ].compute_waypoint_sequence(
            optimization_data.locations, v_routes[v_type]
        )

    routes = {}
    for v_type, route in v_routes.items():
        route = route.groupby("truck_id").agg(list).to_pandas().to_dict()
        waypoint_sequence = way_point_seq_df[v_type][
            "waypoint_sequence"
        ].to_numpy()
        waypoint_type = way_point_seq_df[v_type]["waypoint_type"].to_numpy()

        routes.update(
            {
                str(
                    optimization_data.fleet_data["vehicle_ids"].iloc[veh_id]
                ): {
                    "task_id": [
                        route["type"][veh_id][idx]
                        if route["type"][veh_id][idx] in ["Depot", "Break"]
                        else str(
                            optimization_data.task_data["task_ids"].iloc[
                                route["route"][veh_id][idx]
                            ]
                        )
                        for idx in range(len(route["route"][veh_id]))
                    ],
                    "arrival_stamp": route["arrival_stamp"][veh_id],
                    "route": sum(
                        [
                            route["location"][veh_id][idx : idx + 1]
                            if idx == 0 or offsets[idx] == offsets[idx - 1] + 1
                            else waypoint_sequence[
                                offsets[idx - 1] + 1 : offsets[idx]
                            ].tolist()
                            for idx in range(len(offsets))
                        ],
                        [],
                    ),
                    "type": sum(
                        [
                            route["type"][veh_id][idx : idx + 1]
                            if idx == 0 or offsets[idx] == offsets[idx - 1] + 1
                            else waypoint_type[
                                offsets[idx - 1] + 1 : offsets[idx]
                            ].tolist()
                            for idx in range(len(offsets))
                        ],
                        [],
                    ),
                }
                for veh_id, offsets in route["sequence_offset"].items()
            }
        )
    return routes


def create_data_model(
    optimization_data: OptimizationDataModel,
    cost_matrix: Optional[dict] = None,
    travel_time_matrix: Optional[dict] = None,
):
    warnings = []
    # Make sure that we are using pool memory allocator
    import rmm

    assert isinstance(
        rmm.mr.get_current_device_resource(), rmm.mr.StatisticsResourceAdaptor
    ) or isinstance(
        rmm.mr.get_current_device_resource(), rmm.mr.PoolMemoryResource
    )

    n_fleet = len(optimization_data.fleet_data["vehicle_locations"])

    n_locations = list(cost_matrix.values())[0].shape[0]

    locations = cudf.Series(
        list(range(len(optimization_data.locations))),
        index=optimization_data.locations,
    )

    n_orders = len(optimization_data.task_data["task_locations"])

    # Create data model object
    data_model = routing.DataModel(n_locations, n_fleet, n_orders)

    for key, value in cost_matrix.items():
        data_model.add_cost_matrix(value, key)
    if travel_time_matrix is not None:
        for key, value in travel_time_matrix.items():
            data_model.add_transit_time_matrix(value, key)

    if optimization_data.fleet_data["vehicle_locations"] is not None:
        if len(optimization_data.locations) > 0:
            start_location_id = locations.loc[
                optimization_data.fleet_data["vehicle_locations"][
                    "start_location"
                ]
            ]
            end_location_id = locations.loc[
                optimization_data.fleet_data["vehicle_locations"][
                    "end_location"
                ]
            ]
            data_model.set_vehicle_locations(
                start_location_id, end_location_id
            )
        else:
            data_model.set_vehicle_locations(
                optimization_data.fleet_data["vehicle_locations"][
                    "start_location"
                ],
                optimization_data.fleet_data["vehicle_locations"][
                    "end_location"
                ],
            )

    if optimization_data.fleet_data["vehicle_time_windows"] is not None:
        v_time_windows = optimization_data.fleet_data["vehicle_time_windows"]
        data_model.set_vehicle_time_windows(
            v_time_windows["earliest"], v_time_windows["latest"]
        )

    if optimization_data.fleet_data["skip_first_trips"] is not None:
        data_model.set_skip_first_trips(
            optimization_data.fleet_data["skip_first_trips"]
        )

    if (
        optimization_data.fleet_data["vehicle_break_time_windows"] is not None
        and optimization_data.fleet_data["vehicle_break_durations"] is not None
    ):
        for index in range(
            len(optimization_data.fleet_data["vehicle_break_time_windows"])
        ):
            v_break_time_windows = optimization_data.fleet_data[
                "vehicle_break_time_windows"
            ][index]
            v_break_durations = optimization_data.fleet_data[
                "vehicle_break_durations"
            ][index]
            data_model.add_break_dimension(
                v_break_time_windows["earliest"],
                v_break_time_windows["latest"],
                v_break_durations,
            )

    if optimization_data.fleet_data["vehicle_break_locations"] is not None:

        if len(optimization_data.locations) > 0:
            break_location_id = locations.loc[
                optimization_data.fleet_data["vehicle_break_locations"]
            ]
            data_model.set_break_locations(break_location_id)
        else:
            data_model.set_break_locations(
                optimization_data.fleet_data["vehicle_break_locations"]
            )

    if optimization_data.fleet_data["vehicle_types"] is not None:
        data_model.set_vehicle_types(
            optimization_data.fleet_data["vehicle_types"]
        )

    if optimization_data.fleet_data["vehicle_breaks"] is not None:
        for data in optimization_data.fleet_data["vehicle_breaks"]:
            data_model.add_vehicle_break(
                data["vehicle_id"],
                data["earliest"],
                data["latest"],
                data["duration"],
                cudf.Series(data["locations"]),
            )

    if optimization_data.fleet_data["vehicle_order_match"] is not None:
        for data in optimization_data.fleet_data["vehicle_order_match"]:
            data_model.add_vehicle_order_match(
                data["vehicle_id"], cudf.Series(data["order_ids"])
            )

    if optimization_data.fleet_data["drop_return_trips"] is not None:
        data_model.set_drop_return_trips(
            optimization_data.fleet_data["drop_return_trips"]
        )

    if optimization_data.fleet_data["vehicle_max_costs"] is not None:
        data_model.set_vehicle_max_costs(
            optimization_data.fleet_data["vehicle_max_costs"]
        )

    if optimization_data.fleet_data["vehicle_max_times"] is not None:
        data_model.set_vehicle_max_times(
            optimization_data.fleet_data["vehicle_max_times"]
        )

    if optimization_data.fleet_data["vehicle_fixed_costs"] is not None:
        data_model.set_vehicle_fixed_costs(
            optimization_data.fleet_data["vehicle_fixed_costs"]
        )

    if optimization_data.fleet_data["min_vehicles"] is not None:
        data_model.set_min_vehicles(
            optimization_data.fleet_data["min_vehicles"]
        )

    if optimization_data.task_data["task_locations"] is not None:
        if len(optimization_data.locations) > 0:
            task_index = locations.loc[
                optimization_data.task_data["task_locations"]
            ]
            data_model.set_order_locations(task_index)
        else:
            data_model.set_order_locations(
                optimization_data.task_data["task_locations"]
            )

    if optimization_data.task_data["pickup_and_delivery_pairs"] is not None:
        pickup_delivery = optimization_data.task_data[
            "pickup_and_delivery_pairs"
        ]
        data_model.set_pickup_delivery_pairs(
            pickup_delivery["pickup_ind"], pickup_delivery["delivery_ind"]
        )

    if (
        optimization_data.task_data["demand"] is not None
        and optimization_data.fleet_data["capacities"] is not None
    ):

        if (
            optimization_data.task_data["demand"].shape[1]
            != optimization_data.fleet_data["capacities"].shape[1]
        ):
            demand_dim = optimization_data.task_data["demand"].shape[1]
            cap_dim = optimization_data.fleet_data["capacities"].shape[1]
            raise HTTPException(
                status_code=400,
                detail=(
                    f"Mismatch in Capacity and Demand dimension, (capacity_dim) {cap_dim} != (demand_dim) {demand_dim}"  # noqa
                ),
            )
        for col in optimization_data.task_data["demand"].columns:
            demand_name = "demand_" + str(col)
            demand = optimization_data.task_data["demand"][col]
            capacities = optimization_data.fleet_data["capacities"][col]
            data_model.add_capacity_dimension(demand_name, demand, capacities)

    if optimization_data.task_data["task_time_windows"] is not None:
        t_time_windows = optimization_data.task_data["task_time_windows"]

        data_model.set_order_time_windows(
            t_time_windows["earliest"], t_time_windows["latest"]
        )

    if optimization_data.task_data["service_times"] is not None:
        service_times = optimization_data.task_data["service_times"]

        if service_times is not None:
            if type(service_times) is dict:
                for v_id, service_time in service_times.items():
                    data_model.set_order_service_times(
                        cudf.Series(service_time, dtype=np.int32), int(v_id)
                    )
            else:
                data_model.set_order_service_times(
                    cudf.Series(service_times, dtype=np.int32)
                )

    if optimization_data.solver_config["objectives"] is not None:
        data_model.set_objective_function(
            optimization_data.solver_config["objectives"],
            optimization_data.solver_config["objective_weights"],
        )

    if optimization_data.task_data["prizes"] is not None:
        data_model.set_order_prizes(optimization_data.task_data["prizes"])

    if optimization_data.task_data["order_vehicle_match"] is not None:
        for data in optimization_data.task_data["order_vehicle_match"]:
            data_model.add_order_vehicle_match(
                data["order_id"], cudf.Series(data["vehicle_ids"])
            )

    if optimization_data.initial_solution is not None:
        vehicle_ids, routes, types, sol_offsets = parse_initial_sol(
            optimization_data.initial_solution
        )
        data_model.add_initial_solutions(
            cudf.Series(vehicle_ids),
            cudf.Series(routes),
            cudf.Series(types),
            cudf.Series(sol_offsets),
        )
    return warnings, data_model


def create_solver(optimization_data: OptimizationDataModel):
    warnings = []
    solver_settings = routing.SolverSettings()

    if optimization_data.solver_config["time_limit"] is not None:
        solver_settings.set_time_limit(
            optimization_data.solver_config["time_limit"]
        )

    if optimization_data.solver_config["config_file"] is not None:
        solver_settings.dump_config_file(
            optimization_data.solver_config["config_file"]
        )
    if optimization_data.solver_config["verbose_mode"] is not None:
        solver_settings.set_verbose_mode(
            optimization_data.solver_config["verbose_mode"]
        )
    if optimization_data.solver_config["error_logging"] is not None:
        solver_settings.set_error_logging_mode(
            optimization_data.solver_config["error_logging"]
        )

    return warnings, solver_settings


def prep_optimization_data(optimization_data):

    if optimization_data.task_data["task_locations"] is None:
        raise ValueError("task location is None")
    elif optimization_data.fleet_data["vehicle_locations"] is None:
        raise ValueError("vehicle location is None")

    cost_matrix = {}
    cost_waypoint_graph = {}
    travel_time_matrix = {}
    travel_time_waypoint_graph = {}

    if len(optimization_data.cost_matrix) != 0:
        cost_matrix = optimization_data.cost_matrix
    elif len(optimization_data.waypoint_graph) != 0:
        optimization_data.locations = np.append(
            optimization_data.task_data["task_locations"].to_numpy(),
            optimization_data.fleet_data["vehicle_locations"]
            .to_numpy()
            .flatten(),
        )

        if optimization_data.fleet_data["vehicle_break_locations"] is not None:
            optimization_data.locations = np.append(
                optimization_data.locations,
                optimization_data.fleet_data[
                    "vehicle_break_locations"
                ].to_numpy(),
            )
        optimization_data.locations = np.unique(optimization_data.locations)

        for v_type, graph in optimization_data.waypoint_graph.items():
            cost_waypoint_graph[v_type] = distance_engine.WaypointMatrix(
                graph["offsets"], graph["edges"], graph["weights"]
            )

            cost_matrix[v_type] = cost_waypoint_graph[
                v_type
            ].compute_cost_matrix(optimization_data.locations)
    else:
        raise ValueError("No cost matrix or way point graph provided")

    if len(optimization_data.travel_time_matrix) != 0:
        travel_time_matrix = optimization_data.travel_time_matrix
    elif len(optimization_data.travel_time_waypoint_graph) != 0:
        for (
            v_type,
            graph,
        ) in optimization_data.travel_time_waypoint_graph.items():
            travel_time_waypoint_graph[
                v_type
            ] = distance_engine.WaypointMatrix(
                graph["offsets"], graph["edges"], graph["weights"]
            )
            travel_time_matrix[v_type] = travel_time_waypoint_graph[
                v_type
            ].compute_cost_matrix(optimization_data.locations)
    else:
        travel_time_matrix = None

    return (
        optimization_data,
        cost_matrix,
        travel_time_matrix,
        cost_waypoint_graph,
    )


def get_solver_exception_type(status, message):
    msg = f"error_status: {status}, msg: {message}"

    if status == ErrorStatus.Success:
        return None
    elif status == ErrorStatus.ValidationError:
        return InputValidationError(msg)
    elif status == ErrorStatus.OutOfMemoryError:
        return OutOfMemoryError(msg)
    elif status == ErrorStatus.RuntimeError:
        return InputRuntimeError(msg)
    else:
        return RuntimeError(msg)


def solve(
    optimization_data: OptimizationDataModel,
):
    notes = []
    total_solve_time = 0
    try:
        (
            optimization_data,
            cost_matrix,
            travel_time_matrix,
            cost_waypoint_graph,
        ) = prep_optimization_data(optimization_data)

        warnings, data_model = create_data_model(
            optimization_data,
            cost_matrix=cost_matrix,
            travel_time_matrix=travel_time_matrix,
        )

        cswarnings, solver_settings = create_solver(optimization_data)
        warnings.extend(cswarnings)

        solve_time_start = time.time()
        sol = routing.Solve(data_model, solver_settings)
        if sol is not None and sol.get_error_status() != ErrorStatus.Success:
            raise get_solver_exception_type(
                sol.get_error_status(), sol.get_error_message()
            )

        total_solve_time = time.time() - solve_time_start

        valid_solve_status = [0, 1]

        if sol.get_status() not in valid_solve_status:
            raise HTTPException(
                status_code=409,
                detail=sol.get_message(),
            )
        else:
            routes = sol.get_route()
            accepted = sol.get_accepted_solutions().to_arrow().to_pylist()
            dropped_tasks = {
                "task_id": (
                    optimization_data.task_data["task_ids"]
                    .iloc[sol.get_infeasible_orders()]
                    .to_arrow()
                    .to_pylist()
                ),
                "task_index": sol.get_infeasible_orders()
                .to_arrow()
                .to_pylist(),
            }

            # Compute waypoint sequence df for each vehicle type
            if len(optimization_data.waypoint_graph) != 0:
                routes = create_waypoint_sequence_routes(
                    optimization_data, routes, cost_waypoint_graph
                )
            else:
                routes = (
                    routes.groupby("truck_id").agg(list).to_pandas().to_dict()
                )

                routes = {
                    str(
                        optimization_data.fleet_data["vehicle_ids"].iloc[
                            veh_id
                        ]
                    ): {
                        "task_id": [
                            routes["type"][veh_id][idx]
                            if routes["type"][veh_id][idx]
                            in ["Depot", "Break"]
                            else str(
                                optimization_data.task_data["task_ids"].iloc[
                                    routes["route"][veh_id][idx]
                                ]
                            )
                            for idx in range(len(routes["route"][veh_id]))
                        ],
                        "arrival_stamp": routes["arrival_stamp"][veh_id],
                        "type": routes["type"][veh_id],
                        "route": routes["location"][veh_id],
                    }
                    for veh_id in list(routes["route"].keys())
                }

            objective_values_temp = sol.get_objective_values()
            objective_values = {
                objective_names[obj]: float(val)
                for obj, val in objective_values_temp.items()
            }

            initial_sol_map = ["not accepted", "accepted", "not evaluated"]

            res = {
                "status": sol.get_status(),
                "num_vehicles": sol.get_vehicle_count(),
                "solution_cost": sol.get_total_objective(),
                "objective_values": objective_values,
                "vehicle_data": routes,
                "initial_solutions": [initial_sol_map[i] for i in accepted],
                "dropped_tasks": dropped_tasks,
            }
            if res["status"] == 1:
                notes.append(sol.get_message())

        return notes, warnings, res, total_solve_time

    except (InputValidationError, ValueError) as e:
        raise HTTPException(status_code=400, detail=str(e))
    except (InputRuntimeError, OutOfMemoryError) as e:
        raise HTTPException(status_code=422, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))
