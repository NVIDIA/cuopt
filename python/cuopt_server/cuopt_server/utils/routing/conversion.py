# SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import logging
from typing import List, Optional

import numpy as np
from fastapi import HTTPException

import cudf
from cuopt import distance_engine, routing

from cuopt_server.utils.data_definition import (
    CostMatrices,
    FleetData,
    InitialSolution,
    SolverSettingsConfig,
    TaskData,
    WaypointGraphData,
)
from cuopt_server.utils.routing.initial_solution import parse_initial_sol
from cuopt_server.utils.routing.optimization_data_model import (
    OptimizationDataModel,
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
