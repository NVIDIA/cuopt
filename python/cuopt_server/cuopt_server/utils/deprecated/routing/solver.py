# SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import time

from fastapi import HTTPException

from cuopt import routing
from cuopt.routing import ErrorStatus
from cuopt.utilities import (
    InputRuntimeError,
    InputValidationError,
    OutOfMemoryError,
)

from cuopt_server.utils.routing.conversion import (  # noqa: F401
    create_data_model,
    create_solver,
    prep_optimization_data,
    warn_on_objectives,
)
from cuopt_server.utils.routing.optimization_data_model import (
    OptimizationDataModel,
    objective_names,
)


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
