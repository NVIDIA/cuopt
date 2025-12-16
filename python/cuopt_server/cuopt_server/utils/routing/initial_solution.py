# SPDX-FileCopyrightText: Copyright (c) 2022-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import logging

from cuopt_server.utils.data_definition import InitialSolution


def add_initial_sol(data, initial_sols):
    data["initial_solution"] = []
    for initial_sol in initial_sols:
        if "response" in initial_sol:
            logging.info("Reading cuopt routing response as initial solution")
            vehicle_data = initial_sol["response"]["solver_response"][
                "vehicle_data"
            ]
            initial_sol_data = InitialSolution.parse_obj(vehicle_data)
            data["initial_solution"].append(initial_sol_data)

        else:
            logging.info("Reading external solution as initial solution")
            initial_sol_data = InitialSolution.parse_obj(initial_sol)
            data["initial_solution"].append(initial_sol_data)


def parse_initial_sol(initial_sols):
    vehicle_ids = []
    types = []
    routes = []
    sol_offsets = [0]
    for initial_sol in initial_sols:
        initial_sol = initial_sol.model_dump()
        for veh_id, key in enumerate(initial_sol.keys()):
            stop_type = [i for i in initial_sol[key]["type"] if i != "w"]
            types += stop_type
            routes += [
                0 if i in ["Depot", "Break"] else int(i)
                for i in initial_sol[key]["task_id"]
            ]
            vehicle_ids += [veh_id] * len(stop_type)
        sol_offsets.append(len(routes))
    return vehicle_ids, routes, types, sol_offsets
