# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""MPS/LP file readers and cuOpt JSON dict converters for LP/MIP.

``Read`` / ``ParseMps`` load files into :class:`DataModel`. ``toDict``
(alias ``toDictFromDataModel``) serializes a model to the cuOpt LP/MIP
JSON schema, ``toDataModelAndSettings`` reads that schema back, and
``toDictFromSolution`` maps a
:class:`~cuopt.linear_programming.solution.Solution` to the cuOpt
JSON response format.
"""

import json as json_module
import os
import zlib

import numpy as np
from cuopt.linear_programming.data_model import DataModel
from cuopt.linear_programming.io import parser_wrapper
from cuopt.linear_programming.io.utilities import (
    catch_io_exception,
)
from cuopt.linear_programming.solution.solution import (
    LPTerminationStatus,
    MILPTerminationStatus,
    ProblemCategory,
)
from cuopt.linear_programming.solver_settings import (
    SolverSettings,
    solver_params,
)

_SOLUTION_STATUSES = (
    LPTerminationStatus.Optimal,
    LPTerminationStatus.IterationLimit,
    LPTerminationStatus.TimeLimit,
    MILPTerminationStatus.Optimal,
    MILPTerminationStatus.FeasibleFound,
)

# cuOpt JSON nests these under solver_config["tolerances"]. Names ending in
# "_tolerance" are looked up there by convention; MIP gaps are the
# exceptions that do not follow that suffix.
_TOLERANCE_EXCEPTIONS = frozenset(
    {
        "mip_absolute_gap",
        "mip_relative_gap",
    }
)


@catch_io_exception
def Read(file_path: str, fixed_mps_format: bool = False) -> DataModel:
    """Read an optimization problem from a file, dispatching on extension.

    Dispatches to the MPS/QPS or LP reader based on the filename suffix
    (case-insensitive), matching the C++ ``read`` entry point:

    - ``.mps``, ``.mps.gz``, ``.mps.bz2``, ``.qps``, ``.qps.gz``, ``.qps.bz2``
      → MPS/QPS reader
    - ``.lp``, ``.lp.gz``, ``.lp.bz2`` → LP reader

    Parameters
    ----------
    file_path : str
        Path to an MPS, QPS, or LP file (optionally ``.gz`` / ``.bz2``
        compressed).
    fixed_mps_format : bool
        If the MPS/QPS reader should parse as fixed MPS format. Ignored for
        LP inputs. False by default.

    Returns
    -------
    data_model : DataModel
        A fully formed LP/MILP/QP problem.

    Raises
    ------
    InputValidationError, InputRuntimeError, OutOfMemoryError
        Parser errors from the underlying C++ readers (via
        ``catch_io_exception``).
    RuntimeError
        If the file extension is not one of the supported suffixes (raised by
        the C++ ``read`` dispatch).
    """
    return parser_wrapper.Read(file_path, fixed_mps_format)


@catch_io_exception
def ParseMps(mps_file_path: str, fixed_mps_format: bool = False) -> DataModel:
    """Read an MPS or QPS file directly via the MPS/QPS reader.

    Unlike :func:`Read`, this function bypasses extension-based dispatch
    and always invokes the MPS/QPS reader (``read_mps`` on the C++ side),
    regardless of the filename suffix. Compressed inputs (``.mps.gz``,
    ``.mps.bz2``, ``.qps.gz``, ``.qps.bz2``) are still supported when
    zlib / libbz2 are available, because compression is detected from
    the file path inside the reader.

    Parameters
    ----------
    mps_file_path : str
        Path to an MPS or QPS file (optionally ``.gz`` / ``.bz2``
        compressed).
    fixed_mps_format : bool
        If the MPS/QPS reader should parse the file as fixed MPS format.
        False by default.

    Returns
    -------
    data_model : DataModel
        A fully formed LP/MILP/QP problem.

    Raises
    ------
    InputValidationError, InputRuntimeError, OutOfMemoryError
        Parser errors from the underlying C++ reader (via
        ``catch_io_exception``).
    """
    return parser_wrapper.ParseMps(mps_file_path, fixed_mps_format)


def _tolist(value):
    """Coerce a numpy array to a list; pass lists and None through.

    Setters store whatever the caller supplied, so a model built from a
    cuOpt dict holds plain lists where an MPS-parsed model holds arrays.
    """
    if value is None:
        return None
    return value.tolist() if hasattr(value, "tolist") else value


def _initial_solution(model, json):
    """Return the initial_solution section, or None if the model has none."""
    primal = model.initial_primal_solution
    dual = model.initial_dual_solution
    if len(primal) == 0 and len(dual) == 0:
        return None
    initial_solution = {}
    if len(primal) > 0:
        initial_solution["primal"] = _tolist(primal) if json else primal
    if len(dual) > 0:
        initial_solution["dual"] = _tolist(dual) if json else dual
    return initial_solution


def toDict(model, json=False):
    """Serialize a ``DataModel`` to a cuOpt LP/MIP JSON dict.

    Parameters
    ----------
    model : DataModel
    json : bool, default False
        If True, numpy arrays become lists and infinities become
        ``"inf"`` / ``"ninf"`` strings.
    """
    if not isinstance(model, parser_wrapper.DataModel):
        raise ValueError(
            "model must be a cuopt.linear_programming.io.parser_wrapper.DataModel"
        )

    def transform(data):
        for key, value in data.items():
            if isinstance(value, dict):
                transform(value)
            elif isinstance(value, list):
                if np.inf in data[key] or -np.inf in data[key]:
                    data[key] = [
                        "inf" if x == np.inf else "ninf" if x == -np.inf else x
                        for x in data[key]
                    ]

    if json is True:
        problem_data = {
            "csr_constraint_matrix": {
                "offsets": _tolist(model.A_offsets),
                "indices": _tolist(model.A_indices),
                "values": _tolist(model.A_values),
            },
            "constraint_bounds": {
                "bounds": _tolist(model.b),
                "upper_bounds": _tolist(model.constraint_upper_bounds),
                "lower_bounds": _tolist(model.constraint_lower_bounds),
                "types": _tolist(model.host_row_types),
            },
            "objective_data": {
                "coefficients": _tolist(model.c),
                "scalability_factor": model.objective_scaling_factor,
                "offset": model.objective_offset,
            },
            "variable_bounds": {
                "upper_bounds": _tolist(model.variable_upper_bounds),
                "lower_bounds": _tolist(model.variable_lower_bounds),
            },
            "maximize": model.maximize,
            "variable_types": _tolist(model.variable_types),
            "variable_names": _tolist(model.variable_names),
        }
        initial_solution = _initial_solution(model, json=True)
        if initial_solution is not None:
            problem_data["initial_solution"] = initial_solution
        transform(problem_data)
    else:
        problem_data = {
            "csr_constraint_matrix": {
                "offsets": model.A_offsets,
                "indices": model.A_indices,
                "values": model.A_values,
            },
            "constraint_bounds": {
                "bounds": model.b,
                "upper_bounds": model.constraint_upper_bounds,
                "lower_bounds": model.constraint_lower_bounds,
                "types": model.host_row_types,
            },
            "objective_data": {
                "coefficients": model.c,
                "scalability_factor": model.objective_scaling_factor,
                "offset": model.objective_offset,
            },
            "variable_bounds": {
                "upper_bounds": model.variable_upper_bounds,
                "lower_bounds": model.variable_lower_bounds,
            },
            "maximize": model.maximize,
            "variable_types": model.variable_types,
            "variable_names": model.variable_names,
        }
        initial_solution = _initial_solution(model, json=False)
        if initial_solution is not None:
            problem_data["initial_solution"] = initial_solution
    return problem_data


def toDictFromDataModel(model, json=False):
    """Alias of :func:`toDict`, named to match :func:`toDictFromSolution`."""
    return toDict(model, json=json)


def _load_mapping(data):
    if isinstance(data, dict):
        payload = data
    elif isinstance(data, str):
        if os.path.isfile(data):
            extension = os.path.splitext(data)[1].lower()
            if extension == ".msgpack":
                import msgpack
                import msgpack_numpy

                msgpack_numpy.patch()
                with open(data, "rb") as f:
                    payload = msgpack.load(f, strict_map_key=False)
            elif extension == ".zlib":
                with open(data, "rb") as f:
                    payload = json_module.loads(zlib.decompress(f.read()))
            else:
                with open(data, "r", encoding="utf-8") as f:
                    payload = json_module.load(f)
        else:
            payload = json_module.loads(data)
    else:
        raise TypeError(
            f"Unsupported input type {type(data)!r}; expected dict, JSON string, "
            "or .json/.msgpack/.zlib path"
        )

    if not isinstance(payload, dict):
        raise TypeError(
            f"Expected a cuOpt JSON object (dict), got {type(payload).__name__}"
        )
    return payload


def _as_array(value, dtype=None):
    if value is None:
        return None
    if isinstance(value, list):
        if any(x in ("inf", "ninf") for x in value):
            value = [
                np.inf if x == "inf" else -np.inf if x == "ninf" else x
                for x in value
            ]
        return np.array(value) if dtype is None else np.array(value, dtype)
    return value


def _section(payload, name):
    value = payload.get(name)
    return value if isinstance(value, dict) else {}


def _fill_data_model(payload):
    data_model = DataModel()
    csr = _section(payload, "csr_constraint_matrix")
    if not csr:
        raise ValueError("cuOpt LP dict is missing csr_constraint_matrix")
    data_model.set_csr_constraint_matrix(
        _as_array(csr.get("values"), np.float64),
        _as_array(csr.get("indices"), np.int32),
        _as_array(csr.get("offsets"), np.int32),
    )

    constraint_bounds = _section(payload, "constraint_bounds")
    bounds = _as_array(constraint_bounds.get("bounds"), np.float64)
    if bounds is not None:
        data_model.set_constraint_bounds(bounds)
    types = _as_array(constraint_bounds.get("types"))
    if types is not None and len(types):
        data_model.set_row_types(types)
    upper = _as_array(constraint_bounds.get("upper_bounds"), np.float64)
    if upper is not None and len(upper):
        data_model.set_constraint_upper_bounds(upper)
    lower = _as_array(constraint_bounds.get("lower_bounds"), np.float64)
    if lower is not None and len(lower):
        data_model.set_constraint_lower_bounds(lower)

    objective = _section(payload, "objective_data")
    coefficients = _as_array(objective.get("coefficients"), np.float64)
    if coefficients is not None:
        data_model.set_objective_coefficients(coefficients)
    if objective.get("scalability_factor") is not None:
        data_model.set_objective_scaling_factor(
            objective["scalability_factor"]
        )
    if objective.get("offset") is not None:
        data_model.set_objective_offset(objective["offset"])

    variable_bounds = _section(payload, "variable_bounds")
    v_upper = _as_array(variable_bounds.get("upper_bounds"), np.float64)
    if v_upper is not None:
        data_model.set_variable_upper_bounds(v_upper)
    v_lower = _as_array(variable_bounds.get("lower_bounds"), np.float64)
    if v_lower is not None:
        data_model.set_variable_lower_bounds(v_lower)

    initial = _section(payload, "initial_solution")
    primal = _as_array(initial.get("primal"), np.float64)
    if primal is not None:
        data_model.set_initial_primal_solution(primal)
    dual = _as_array(initial.get("dual"), np.float64)
    if dual is not None:
        data_model.set_initial_dual_solution(dual)

    if payload.get("maximize") is not None:
        data_model.set_maximize(payload["maximize"])
    if payload.get("variable_types") is not None:
        data_model.set_variable_types(_as_array(payload["variable_types"]))
    if payload.get("variable_names") is not None:
        data_model.set_variable_names(payload["variable_names"])
    return data_model


def _fill_solver_settings(payload, warmstart_data=None):
    solver_settings = SolverSettings()
    solver_config = _section(payload, "solver_config")
    if not solver_config and warmstart_data is None:
        return solver_settings

    tolerances = _section(solver_config, "tolerances")
    if tolerances.get("optimality") is not None:
        solver_settings.set_optimality_tolerance(tolerances["optimality"])
    for param in solver_params:
        if param in _TOLERANCE_EXCEPTIONS or param.endswith("_tolerance"):
            param_value = tolerances.get(param)
        else:
            param_value = solver_config.get(param)
        if param_value is not None and param_value != "":
            if isinstance(param_value, bool):
                param_value = int(param_value)
            solver_settings.set_parameter(param, param_value)

    if warmstart_data is not None:
        solver_settings.set_pdlp_warm_start_data(warmstart_data)
    return solver_settings


def toDataModelAndSettings(data, warmstart_data=None):
    """Convert a cuOpt LP/MIP dict to ``(DataModel, SolverSettings)``.

    Lists become numpy arrays and the strings
    ``"inf"`` / ``"ninf"`` become IEEE infinities.

    Parameters
    ----------
    data : dict or str
        A cuOpt JSON dictionary, a JSON string, or the path to a ``.json``,
        ``.msgpack``, or ``.zlib`` file holding one.
    warmstart_data : optional
        PDLP warm-start blob passed to
        :meth:`SolverSettings.set_pdlp_warm_start_data`.

    Returns
    -------
    (data_model, solver_settings) : tuple
        ``DataModel`` and ``SolverSettings`` ready for
        ``cuopt.linear_programming.Solve`` or
        ``cuopt.grpc.linear_programming.Client.submit``. The settings are
        built from the payload's ``solver_config``, and are left at their
        defaults when it is absent.
    """
    payload = _load_mapping(data)
    return _fill_data_model(payload), _fill_solver_settings(
        payload, warmstart_data=warmstart_data
    )


def toDictFromSolution(sol):
    """Map a cuOpt ``Solution`` to a cuOpt http server response dictionary.

    Produces the same envelope as ``cuopt_server`` (``reqId``,
    ``response.solver_response``, ``vars``, list-encoded arrays), so a
    locally or gRPC-obtained solution can be handed to code written
    against the http response. Fields that ``Solution`` does not
    provide (``reqId``, ``warnings``, ``total_solve_time``) are ``None``;
    fill them in afterward if you need them. The PDLP warm-start blob
    is omitted; the server serves it from a separate endpoint.

    Parameters
    ----------
    sol : cuopt.linear_programming.solution.Solution
        Result from a local ``Solve`` or ``Client.result``.
    """
    solution = {}
    status = sol.get_termination_status()

    if status in _SOLUTION_STATUSES:
        is_lp = sol.get_problem_category() == ProblemCategory.LP
        solution["problem_category"] = sol.get_problem_category().name
        solution["primal_solution"] = _tolist(sol.get_primal_solution())
        solution["primal_objective"] = sol.get_primal_objective()
        solution["solver_time"] = sol.get_solve_time()
        solution["solved_by"] = sol.get_solved_by().name
        solution["vars"] = sol.get_vars()
        if is_lp:
            solution["dual_solution"] = _tolist(sol.get_dual_solution())
            solution["dual_objective"] = sol.get_dual_objective()
            solution["reduced_cost"] = _tolist(sol.get_reduced_cost())
            lp_stats = sol.get_lp_stats()
            solution["lp_statistics"] = {} if lp_stats is None else lp_stats
            solution["milp_statistics"] = {}
        else:
            solution["dual_solution"] = None
            solution["dual_objective"] = None
            solution["reduced_cost"] = None
            solution["lp_statistics"] = {}
            milp_stats = sol.get_milp_stats()
            solution["milp_statistics"] = (
                {} if milp_stats is None else milp_stats
            )

    return {
        "reqId": None,
        "response": {
            "solver_response": {"status": status.name, "solution": solution},
            "total_solve_time": None,
        },
        "warnings": None,
        "notes": [sol.get_termination_reason()],
    }
