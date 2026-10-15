# SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import logging
from typing import List, Optional

import numpy as np
import pandas as pd
from fastapi import HTTPException

from cuopt import distance_engine, routing

from cuopt_server.utils.data_definition import (
    CostMatrices,
    FleetData,
    InitialSolution,
    SolverSettingsConfig,
    TaskData,
    WaypointGraphData,
)
from cuopt_server.utils.routing.host_optimization_data_model import (
    HostOptimizationDataModel,
)
from cuopt_server.utils.routing.initial_solution import parse_initial_sol
from cuopt_server.utils.routing.optimization_data_model import (
    OptimizationDataModel,
    objective_names,
)


# Return exception if validation fails
def check_valid(is_valid):
    if not is_valid[0]:
        raise HTTPException(status_code=400, detail=f"{is_valid[1]}")


def warn_on_objectives(solver_config):
    warnings = []
    return warnings, solver_config


# Standard solve time for VRP
def std_solver_time_calc(num_tasks):
    return 10 + num_tasks / 6


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
    optimization_data = HostOptimizationDataModel()

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
                vehicle_distance_breaks=fleet_data.vehicle_distance_breaks,
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
            solver_config.time_limit = std_solver_time_calc(num_tasks)
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


def create_data_model(
    optimization_data: HostOptimizationDataModel,
    cost_matrix: Optional[dict] = None,
    travel_time_matrix: Optional[dict] = None,
):
    warnings = []

    n_fleet = len(optimization_data.fleet_data["vehicle_locations"])

    n_locations = list(cost_matrix.values())[0].shape[0]

    locations = pd.Series(
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
                pd.Series(data["locations"]),
            )

    if optimization_data.fleet_data["vehicle_distance_breaks"] is not None:
        for data in optimization_data.fleet_data["vehicle_distance_breaks"]:
            if data["locations"] is not None:
                if len(optimization_data.locations) > 0:
                    break_locations = locations.loc[data["locations"]].astype(
                        "int32"
                    )
                else:
                    break_locations = pd.Series(
                        data["locations"], dtype="int32"
                    )
            else:
                break_locations = None
            data_model.add_vehicle_distance_break(
                data["vehicle_id"],
                data["distance_min"],
                data["distance_max"],
                data["duration"],
                break_locations,
            )

    if optimization_data.fleet_data["vehicle_order_match"] is not None:
        for data in optimization_data.fleet_data["vehicle_order_match"]:
            data_model.add_vehicle_order_match(
                data["vehicle_id"], pd.Series(data["order_ids"])
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
                        pd.Series(service_time, dtype=np.int32), int(v_id)
                    )
            else:
                data_model.set_order_service_times(
                    pd.Series(service_times, dtype=np.int32)
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
                data["order_id"], pd.Series(data["vehicle_ids"])
            )

    if optimization_data.initial_solution is not None:
        vehicle_ids, routes, types, sol_offsets = parse_initial_sol(
            optimization_data.initial_solution
        )
        data_model.add_initial_solutions(
            pd.Series(vehicle_ids),
            pd.Series(routes),
            pd.Series(types),
            pd.Series(sol_offsets),
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
        if optimization_data.fleet_data["vehicle_distance_breaks"] is not None:
            for d in optimization_data.fleet_data["vehicle_distance_breaks"]:
                break_locs = d.get("locations")
                if break_locs is not None and len(break_locs) > 0:
                    optimization_data.locations = np.append(
                        optimization_data.locations,
                        np.asarray(break_locs),
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
            travel_time_waypoint_graph[v_type] = (
                distance_engine.WaypointMatrix(
                    graph["offsets"], graph["edges"], graph["weights"]
                )
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


def _as_python_list(values):
    if values is None:
        return []
    if hasattr(values, "tolist"):
        return values.tolist()
    return list(values)


_NODE_TYPE_NAMES = {
    0: "Depot",
    1: "Pickup",
    2: "Delivery",
    3: "Break",
}


def _node_type_name(value):
    try:
        return _NODE_TYPE_NAMES[int(value)]
    except (ValueError, TypeError, KeyError):
        return str(value)


def solution_to_http(
    sol: dict,
    vehicle_ids: Optional[List] = None,
    task_ids: Optional[List] = None,
) -> dict:
    """Map a gRPC routing result dict onto the HTTP solver_response.

    ``sol`` is the routing GetResult dict. ``vehicle_ids`` / ``task_ids`` are
    optional sidecar lists from submit so HTTP keys match the request. After a
    proxy restart they may be absent and numeric indices are used.

    Returns the inner ``solver_response`` dict (integer status 0 or 1).
    Raises ``HTTPException`` with status 409 if ``status`` is neither 0 nor 1.
    """
    status = int(sol.get("status", 0))
    message = sol.get("status_message") or sol.get("error_message") or ""
    if status not in (0, 1):
        raise HTTPException(
            status_code=409,
            detail=message or "routing job did not find a feasible solution",
        )

    route = _as_python_list(sol.get("route"))
    truck_id = _as_python_list(sol.get("truck_id"))
    locations = _as_python_list(sol.get("locations"))
    node_types = _as_python_list(sol.get("node_types"))
    arrival = _as_python_list(sol.get("arrival_stamp"))
    type_names = [_node_type_name(t) for t in node_types]

    grouped = {}
    for i, tid in enumerate(truck_id):
        grouped.setdefault(int(tid), []).append(i)

    vehicle_data = {}
    for tid, idxs in grouped.items():
        if vehicle_ids is not None and 0 <= tid < len(vehicle_ids):
            key = str(vehicle_ids[tid])
        else:
            key = str(tid)
        task_id_col = []
        for i in idxs:
            tname = type_names[i] if i < len(type_names) else ""
            if tname in ("Depot", "Break"):
                task_id_col.append(tname)
            elif task_ids is not None and i < len(route):
                r = int(route[i])
                if 0 <= r < len(task_ids):
                    task_id_col.append(str(task_ids[r]))
                else:
                    task_id_col.append(str(r))
            elif i < len(route):
                task_id_col.append(str(int(route[i])))
            else:
                task_id_col.append("")
        vehicle_data[key] = {
            "task_id": task_id_col,
            "arrival_stamp": [
                float(arrival[i]) for i in idxs if i < len(arrival)
            ],
            "type": [type_names[i] for i in idxs if i < len(type_names)],
            "route": [
                int(locations[i]) if i < len(locations) else int(route[i])
                for i in idxs
            ],
        }

    objective_values = {}
    for key, val in (sol.get("objective_values") or {}).items():
        try:
            name = objective_names[routing.Objective(int(key))]
        except Exception:
            name = str(key)
        objective_values[name] = float(val)

    initial_sol_map = ["not accepted", "accepted", "not evaluated"]
    accepted = [
        initial_sol_map[int(i)] if 0 <= int(i) < 3 else str(int(i))
        for i in _as_python_list(sol.get("accepted"))
    ]
    dropped = [int(i) for i in _as_python_list(sol.get("unserviced_nodes"))]
    dropped_ids = [
        str(task_ids[i])
        if task_ids is not None and 0 <= i < len(task_ids)
        else str(i)
        for i in dropped
    ]

    return {
        "status": status,
        "num_vehicles": int(sol.get("vehicle_count", len(vehicle_data))),
        "solution_cost": float(sol.get("total_objective_value", 0.0)),
        "objective_values": objective_values,
        "vehicle_data": vehicle_data,
        "initial_solutions": accepted,
        "dropped_tasks": {"task_id": dropped_ids, "task_index": dropped},
    }
