# SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import time

from fastapi import HTTPException

from cuopt import linear_programming
from cuopt.linear_programming.internals import (
    GetSolutionCallback,
    SetSolutionCallback,
)
from cuopt.linear_programming.solver.solver_wrapper import (
    ErrorStatus,
)
from cuopt.utilities import (
    InputRuntimeError,
    InputValidationError,
    OutOfMemoryError,
)

# Conversion of request data into cuopt data models and solver settings lives
# in conversion.py. The names below are re-exported so existing importers of
# this module keep working.
from cuopt_server.utils.linear_programming.conversion import (  # noqa: F401
    create_data_model,
    create_solver,
    ignored_warning,
    solution_to_legacy_http,
)


def dep_warning(field):
    return (
        f"solver config {field} is deprecated and will "
        "be removed in a future release"
    )


def warn_on_objectives(solver_config):
    warnings = []
    return warnings, solver_config


class CustomGetSolutionCallback(GetSolutionCallback):
    def __init__(self, sender, req_id):
        super().__init__()
        self.req_id = req_id
        self.sender = sender
        self.solutions = []

    def get_solution(self, solution, solution_cost, solution_bound, user_data):
        if user_data is not None:
            assert user_data == self.req_id
        solution_list = solution.tolist()
        solution_cost_val = float(solution_cost[0])
        solution_bound_val = float(solution_bound[0])
        self.solutions.append(
            {
                "solution": solution_list,
                "cost": solution_cost_val,
                "bound": solution_bound_val,
            }
        )
        self.sender(
            self.req_id,
            solution_list,
            solution_cost_val,
            solution_bound_val,
        )


class CustomSetSolutionCallback(SetSolutionCallback):
    def __init__(self, get_callback, req_id):
        super().__init__()
        self.req_id = req_id
        self.get_callback = get_callback
        self.n_callbacks = 0

    def set_solution(self, solution, solution_cost, solution_bound, user_data):
        if user_data is not None:
            assert user_data == self.req_id
        self.n_callbacks += 1
        if self.get_callback.solutions:
            solution[:] = self.get_callback.solutions[-1]["solution"]
            solution_cost[0] = float(self.get_callback.solutions[-1]["cost"])


def get_solver_exception_type(status, message):
    msg = f"error_status: {status}, msg: {message}"

    # TODO change these to enums once we have a clear place
    # to map them from for both routing and lp
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
    LP_data,
    reqId,
    intermediate_sender,
    warmstart_data,
    incumbent_set_solutions,
):
    notes = []

    def create_solution(sol):
        res = solution_to_legacy_http(sol)
        notes.append(sol.get_termination_reason())
        return res

    try:
        is_batch = False
        sol = None
        total_solve_time = None
        if type(LP_data) is list:
            is_batch = True
            data_model_list = []
            warnings = [
                "LP batch mode is deprecated and will be removed in a future release. "
                "Use sequential Solve() calls or implement your own parallelism."
            ]
            for i_data in LP_data:
                i_warnings, data_model = create_data_model(i_data)
                data_model_list.append(data_model)
                warnings.extend(i_warnings)
            cswarnings, solver_settings = create_solver(
                LP_data[0], warmstart_data
            )
            warnings.extend(cswarnings)
            sol, total_solve_time = linear_programming.BatchSolve(
                data_model_list, solver_settings
            )
        else:
            warnings, data_model = create_data_model(LP_data)
            cswarnings, solver_settings = create_solver(
                LP_data, warmstart_data
            )
            warnings.extend(cswarnings)
            callback = (
                CustomGetSolutionCallback(intermediate_sender, reqId)
                if intermediate_sender is not None
                else None
            )
            if callback is not None:
                solver_settings.set_mip_callback(callback, reqId)
                if incumbent_set_solutions:
                    set_callback = CustomSetSolutionCallback(callback, reqId)
                    solver_settings.set_mip_callback(set_callback, reqId)
            solve_begin_time = time.time()
            sol = linear_programming.Solve(
                data_model, solver_settings=solver_settings
            )
            total_solve_time = time.time() - solve_begin_time

        res = None
        if is_batch:
            res = []
            for i_sol in sol:
                if i_sol is None:
                    continue
                if i_sol.get_error_status() != ErrorStatus.Success:
                    res.append(
                        {
                            "status": i_sol.get_error_status(),
                            "solution": i_sol.get_error_message(),
                        }
                    )
                else:
                    res.append(create_solution(i_sol))
        elif sol is not None:
            if sol.get_error_status() != ErrorStatus.Success:
                raise get_solver_exception_type(
                    sol.get_error_status(), sol.get_error_message()
                )
            res = create_solution(sol)

        return notes, warnings, res, total_solve_time

    except (InputValidationError, ValueError) as e:
        raise HTTPException(status_code=400, detail=str(e))
    except (InputRuntimeError, OutOfMemoryError) as e:
        raise HTTPException(status_code=422, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))
