# SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import copy
import json
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

import jsonref
import numpy as np
from pydantic import BaseModel, ConfigDict, Extra, Field, PlainValidator
from typing_extensions import Annotated

from ..._version import __version_major_minor__

# INPUT DATA DEFINITIONS


class StrictModel(BaseModel):
    class Config:
        extra = Extra.forbid


def listint(v):
    if isinstance(v, np.ndarray):
        if not np.issubdtype(v.dtype, np.integer):
            raise ValueError("dtype must be an integer type")
    elif not isinstance(v, list) or not all(isinstance(x, int) for x in v):
        raise ValueError("must be a list of ints")
    return v


def listfloat(v):
    if isinstance(v, np.ndarray):
        if not np.issubdtype(v.dtype, np.floating):
            raise ValueError("dtype must be a float type")
    elif not isinstance(v, list) or not all(
        isinstance(x, float) or isinstance(x, int) for x in v
    ):
        raise ValueError("must be a list of floats")
    return v


def listfloatinf(v):
    if isinstance(v, np.ndarray):
        if not np.issubdtype(v.dtype, np.floating):
            raise ValueError("dtype must be a float type")
    elif not isinstance(v, list) or not all(
        isinstance(x, float)
        or isinstance(x, int)
        or (isinstance(x, str) and x in ["inf", "ninf"])
        for x in v
    ):
        raise ValueError("must be a list of floats")
    return v


def liststr(v):
    if isinstance(v, np.ndarray):
        if not np.issubdtype(v.dtype, str):
            raise ValueError("dtype must be a string type")
    elif not isinstance(v, list) or not all(isinstance(x, str) for x in v):
        raise ValueError("must be a list of strings")
    return v


def addint(s):
    s["type"] = "array"
    s["items"] = {"type": "integer"}


def addfloat(s):
    s["type"] = "array"
    s["items"] = {"type": "number"}


def addfloatinf(s):
    s["anyOf"] = [
        {
            "items": {
                "anyOf": [
                    {"type": "number"},
                    {"enum": ["inf", "ninf"], "type": "string"},
                ]
            },
            "type": "array",
        },
        {"type": "null"},
    ]


def addstr(s):
    s["type"] = "array"
    s["items"] = {"type": "string"}


class CSRConstraintMatrix(StrictModel):
    """
    Constraints depicted in CSR format.
    For example lets consider following problem,

    # Constraints

    3x + 4y <= 5.4
    2.7x + 10.1y <= 4.9

    # variables
    x, y >= 0

    # Objective
    minimize(0.2x + 0.1y)

    The constraints on the top are converted to CSR and their
    indices, offsets and values are shared in this matrix.

    Offsets depict number of coefficients/values for each constraint.
    Indices signify variable with which the coefficients/values are associated.
    And coefficients 3, 4, 2.7, 10.1 become values.

    For more details please refer here <https://docs.nvidia.com/nvpl/_static/sparse/storage_format/sparse_matrix.html#compressed-sparse-row-csr> # noqa
    """

    indices: Annotated[List[int], PlainValidator(listint)] = Field(
        default=None,
        description="Indices of constraint matrix",
        json_schema_extra=addint,
    )
    offsets: Annotated[List[int], PlainValidator(listint)] = Field(
        default=None,
        description="Offsets of constraint matrix.",
        json_schema_extra=addint,
    )
    values: Annotated[List[float], PlainValidator(listfloat)] = Field(
        default=None,
        description="Values in constraint matrix",
        json_schema_extra=addfloat,
    )

    class Config:
        arbitrary_types_allowed = True


class ObjectiveData(StrictModel):
    """
    Coefficients of the objective data is passed as a list,
    For example lets consider following problem,

    # Constraints

    3x + 4y <= 5.4
    2.7x + 10.1y <= 4.9

    # variables
    x, y >= 0

    # Objective
    minimize(0.2x + 0.1y)

    Coefficients of the objective 0.2 and 0.1 is passed in the same order.
    """

    coefficients: Annotated[List[float], PlainValidator(listfloat)] = Field(
        default=None,
        description="Objective coefficients (c) array",
        json_schema_extra=addfloat,
    )
    scalability_factor: Optional[float] = Field(
        default=None,
        description="Scaling factor of the objective function",
    )
    offset: Optional[float] = Field(
        default=None,
        description="Offset of the objective function",
    )


class ConstraintBounds(StrictModel):
    """
    Bounds for the constraints will be set here,
    For example lets consider following problem,

    # Constraints

    3x + 4y <= 5.4 # Constraint-1
    2.7x + 10.1y <= 4.9 # Constraint-2

    # variables
    x, y >= 0

    # Objective
    minimize(0.2x + 0.1y)

    Two ways to set this,
    Option #1: using upper and lower bounds,
    (Higher Priority in case both ways are used).

    Bounds for the constraints are added here as a list,

    Constraint-1 upper bound is 5.4
    Constraint-2 upper bound is 4.9

    Since no lower bound is explicitly set,

    Constraint-1 lower bound is negative infinity
    Constraint-2 lower bound is negative infinity

    Option #2: Set using bounds and types

    Set values 5.4, 4.9 as list

    set types as "L", "L" which means Less than of equal to.

    """

    bounds: Optional[
        Annotated[
            List[Union[float, Literal["inf", "ninf"]]],
            PlainValidator(listfloatinf),
        ]
    ] = Field(
        default=None,
        description="Constraint bounds (b / right-hand side) array",
        json_schema_extra=addfloatinf,
    )
    upper_bounds: Optional[
        Annotated[
            List[Union[float, Literal["inf", "ninf"]]],
            PlainValidator(listfloatinf),
        ]
    ] = Field(
        default=None,
        description="Constraints upper bounds",
        json_schema_extra=addfloatinf,
    )
    lower_bounds: Optional[
        Annotated[
            List[Union[float, Literal["inf", "ninf"]]],
            PlainValidator(listfloatinf),
        ]
    ] = Field(
        default=None,
        description="Constraints lower bounds",
        json_schema_extra=addfloatinf,
    )
    types: Optional[Annotated[List[str], PlainValidator(liststr)]] = Field(
        default=None,
        description="Type of each row (constraint)."
        "Supported options are,"
        "'E' for equality ( = ): lower & upper constrains bound equal to b"
        "'L' for less-than ( <= ): lower constrains bound equal to -infinity,"
        "upper constrains bound equal to b"
        "'G' for greater-than ( >= ): lower constrains bound equal to b,"
        "upper constrains bound equal to +infinity",
        json_schema_extra=addstr,
    )


class VariableBounds(StrictModel):
    """
    Bounds for the variable will be set here,
    For example lets consider following problem,

    # Constraints

    3x + 4y <= 5.4 # Constraint-1
    2.7x + 10.1y <= 4.9 # Constraint-2

    # variables
    x, y >= 0

    # Objective
    minimize(0.2x + 0.1y)

    # Since there is not specific upper bound is set for variables
    x - upper bound is infinity
    y - upper bound is infinity

    x - lower bound is 0
    y - lower bound is 0
    """

    upper_bounds: Optional[
        Annotated[
            List[Union[float, Literal["inf", "ninf"]]],
            PlainValidator(listfloatinf),
        ]
    ] = Field(
        default=None,
        description="Variables (x) upper bounds",
        json_schema_extra=addfloatinf,
    )
    lower_bounds: Optional[
        Annotated[
            List[Union[float, Literal["inf", "ninf"]]],
            PlainValidator(listfloatinf),
        ]
    ] = Field(
        default=None,
        description="Variables (x) lower bounds",
        json_schema_extra=addfloatinf,
    )


class InitialSolution(StrictModel):
    """
    Initial solution for the solver.
    """

    primal: Optional[Annotated[List[float], PlainValidator(listfloat)]] = (
        Field(
            default=None,
            description="Initial primal solution",
            json_schema_extra=addfloat,
        )
    )
    dual: Optional[Annotated[List[float], PlainValidator(listfloat)]] = Field(
        default=None,
        description="Initial dual solution<br>Note: Not supported for MILP. ",
        json_schema_extra=addfloat,
    )


class SolverConfig(BaseModel):
    """
    LP/MILP solver options for self-hosted REST requests.

    Keys are passed through to the engine (see ``create_solver`` in
    ``cuopt_server.utils.linear_programming.solver``). Descriptions and
    defaults are documented in the User Guide (*LP, QP, and MILP Settings*).
    """

    model_config = ConfigDict(
        extra="allow",
        json_schema_extra={
            "description": (
                "Solver parameters for LP and MILP. The service accepts every "
                "parameter name registered by the cuOpt engine (snake_case "
                "strings as in "
                "``cpp/include/cuopt/linear_programming/constants.h`` without "
                "the ``CUOPT_`` macro). Human-readable documentation lives in "
                "**LP, QP, and MILP Settings**. Keys not listed in this schema "
                "are still forwarded to the solver. Optional nested tolerance "
                "fields may be supplied either under ``tolerances`` (recommended) "
                "or as top-level keys, depending on the parameter."
            ),
        },
    )

    tolerances: Optional[Dict[str, Any]] = Field(
        default=None,
        description=(
            "Optional map of tolerance parameters (for example "
            "``optimality``, ``absolute_primal_tolerance``, ``mip_absolute_gap``). "
            "See **LP, QP, and MILP Settings** in the User Guide."
        ),
    )


def get_solver_config_value(solver_config: SolverConfig, param: str) -> Any:
    """Resolve *param* from nested ``tolerances`` and/or top-level ``solver_config`` keys."""
    tol = solver_config.tolerances
    if isinstance(tol, dict) and param in tol and tol[param] is not None:
        return tol[param]
    if tol is not None and not isinstance(tol, dict):
        nested = getattr(tol, param, None)
        if nested is not None:
            return nested
    return getattr(solver_config, param, None)


class LPData(StrictModel):
    csr_constraint_matrix: CSRConstraintMatrix = Field(
        default=CSRConstraintMatrix(),
        examples=[
            {
                "offsets": [0, 2, 4],
                "indices": [0, 1, 0, 1],
                "values": [3.0, 4.0, 2.7, 10.1],
            }
        ],
        description=CSRConstraintMatrix.__doc__,
    )
    objective_data: Optional[ObjectiveData] = Field(
        default=ObjectiveData(),
        examples=[
            {
                "coefficients": [0.2, 0.1],
                "scalability_factor": 1.0,
                "offset": 0.0,
            }
        ],
        description=ObjectiveData.__doc__,
    )
    constraint_bounds: Optional[ConstraintBounds] = Field(
        default=ConstraintBounds(),
        examples=[
            {"upper_bounds": [5.4, 4.9], "lower_bounds": ["ninf", "ninf"]}
        ],
        description=ConstraintBounds.__doc__,
    )
    variable_bounds: Optional[VariableBounds] = Field(
        default=VariableBounds(),
        examples=[
            {"upper_bounds": ["inf", "inf"], "lower_bounds": [0.0, 0.0]}
        ],
        description=VariableBounds.__doc__,
    )
    initial_solution: Optional[InitialSolution] = Field(
        default=InitialSolution(),
        description=" ",
    )
    maximize: Optional[bool] = Field(
        default=False,
        examples=[False],
        description="If set to True, solver tries to maximize "
        "objective function else it will try to minimize",
    )
    variable_types: Optional[Annotated[List[str], PlainValidator(liststr)]] = (
        Field(
            default=None,
            description="Type of each variable, this is must for MILP,"
            "Available options are, "
            "'I' - Integer"
            "'C' - Continuous",
            json_schema_extra=addstr,
        )
    )
    variable_names: Optional[Annotated[List[str], PlainValidator(liststr)]] = (
        Field(
            default=None,
            description="Name of variables",
            json_schema_extra=addstr,
        )
    )
    solver_config: Optional[SolverConfig] = Field(
        default=SolverConfig(),
        examples=[{"tolerances": {"optimality": 0.0001}}],
        description=" ",
    )


class LPTupleData(StrictModel):
    data_list: List[Tuple[str, bytes]] = Field(
        default=[],
        description="List of tuples containing mime_types and bytes elements",
    )


class WarmStartData(StrictModel):
    """
    PDLP warm start data to restart PDLP with a previous solution context.
    This should be used when you solve a new problem which is similar to
    the previous one.
    """

    current_primal_solution: Annotated[List[float], PlainValidator(listfloat)]
    current_dual_solution: Annotated[List[float], PlainValidator(listfloat)]
    initial_primal_average: Annotated[List[float], PlainValidator(listfloat)]
    initial_dual_average: Annotated[List[float], PlainValidator(listfloat)]
    current_ATY: Annotated[List[float], PlainValidator(listfloat)]
    sum_primal_solutions: Annotated[List[float], PlainValidator(listfloat)]
    sum_dual_solutions: Annotated[List[float], PlainValidator(listfloat)]
    last_restart_duality_gap_primal_solution: Annotated[
        List[float], PlainValidator(listfloat)
    ]
    last_restart_duality_gap_dual_solution: Annotated[
        List[float], PlainValidator(listfloat)
    ]
    initial_primal_weight: float
    initial_step_size: float
    total_pdlp_iterations: int
    total_pdhg_iterations: int
    last_candidate_kkt_score: float
    last_restart_kkt_score: float
    sum_solution_weight: float
    iterations_since_last_restart: int


class SolutionData(StrictModel):
    problem_category: int = Field(
        default=None, description=("Category of the solution, LP-0/MIP-1/IP-2")
    )
    primal_solution: List[float] = Field(
        default=[],
        description=("Primal solution of the LP problem"),
    )
    dual_solution: List[float] = Field(
        default=[],
        description=(
            "Note: Only applicable to LP \nDual solution of the LP problem\n"
        ),
    )
    solver_time: float = Field(
        default=None,
        description=("Returns the engine solve time in seconds"),
    )
    solved_by: int = Field(
        default=None,
        description=(
            "Returns whether problem was solved by PDLP, Barrier or Dual Simplex"
        ),
    )
    primal_objective: float = Field(
        default=None,
        description=("Primal objective of the LP problem"),
    )
    dual_objective: float = Field(
        default=None,
        description=(
            "Note: Only applicable to LP \nDual objective of the LP problem \n"
        ),
    )
    vars: Dict = Field(
        default={},
        description=("Dictionary mapping each variable (name) to its value"),
    )
    lp_statistics: Dict = Field(
        default={},
        description=(
            "Note: Only applicable to LP \n"
            "Convergence statistics of the solution \n"
            "Includes primal residual, dual residual, \n"
            "reduced cost and gap \n"
        ),
    )
    milp_statistics: Dict = Field(
        default={},
        description=(
            "Note: Only applicable to MILP \n"
            "Convergence statistics of the solution \n"
            "Includes mip gap, solution_bound, pre-solve time, \n"
            "max_constraint_violation, max_int_violatione and \n "
            "max_variable_bound_violation \n"
        ),
    )


# LP termination status values
# NOTE: These must match LPTerminationStatus from
# cuopt.linear_programming.solver.solver_wrapper
# We cannot import them directly because it triggers CUDA/RMM initialization
# before the server has configured memory management.
# See test_termination_status_enum_sync() in test_lp.py to ensure these stay in sync.
LP_STATUS_NAMES = frozenset(
    {
        "NoTermination",
        "NumericalError",
        "Optimal",
        "PrimalInfeasible",
        "DualInfeasible",
        "IterationLimit",
        "TimeLimit",
        "PrimalFeasible",
        "UnboundedOrInfeasible",
    }
)

# MILP termination status values
# NOTE: These must match MILPTerminationStatus from
# cuopt.linear_programming.solver.solver_wrapper
MILP_STATUS_NAMES = frozenset(
    {
        "NoTermination",
        "Optimal",
        "FeasibleFound",
        "Infeasible",
        "Unbounded",
        "TimeLimit",
        "UnboundedOrInfeasible",
    }
)

# Combined set of all valid status names
ALL_STATUS_NAMES = LP_STATUS_NAMES | MILP_STATUS_NAMES


def validate_termination_status(v):
    """Validate that status is a valid LP or MILP termination status name."""
    if v not in ALL_STATUS_NAMES:
        raise ValueError(
            f"status must be one of {sorted(ALL_STATUS_NAMES)}, got '{v}'"
        )
    return v


class SolutionResultData(StrictModel):
    status: Annotated[str, PlainValidator(validate_termination_status)] = (
        Field(
            default="NoTermination",
            examples=["Optimal"],
            description=(
                "In case of LP : \n\n"
                "NoTermination - No Termination \n\n"
                "NumericalError - Numerical Error \n\n"
                "Optimal - Optimal solution is available \n\n"
                "PrimalInfeasible - Primal Infeasible solution \n\n"
                "DualInfeasible - Dual Infeasible solution \n\n"
                "IterationLimit - Iteration Limit reached \n\n"
                "TimeLimit - TimeLimit reached \n\n"
                "PrimalFeasible - Primal Feasible \n\n"
                "---------------------- \n\n"
                "In case of MILP/IP : \n\n"
                "NoTermination - No Termination \n\n"
                "Optimal - Optimal solution is available \n\n"
                "FeasibleFound - Feasible solution is available \n\n"
                "Infeasible - Infeasible \n\n"
                "Unbounded - Unbounded \n\n"
                "TimeLimit - TimeLimit reached \n\n"
            ),
        )
    )
    solution: SolutionData = Field(
        default=SolutionData(), description=("Solution of the LP problem")
    )


class LPSolve(StrictModel):
    solver_response: Union[SolutionResultData, List[SolutionResultData]] = (
        Field(default=SolutionResultData(), description="LP solution")
    )
    perf_times: Optional[Dict] = Field(
        default=None, description=("Etl and Solve times of the solve call")
    )
    total_solve_time: Optional[float] = Field(
        default=None, description=("Total Solve time of batch problem")
    )


class IncumbentSolution(StrictModel):
    solution: List[float]
    cost: Union[float, None]
    bound: Union[float, None]


lp_example_data = {
    "csr_constraint_matrix": {
        "offsets": [0, 2, 4],
        "indices": [0, 1, 0, 1],
        "values": [3.0, 4.0, 2.7, 10.1],
    },
    "constraint_bounds": {
        "upper_bounds": [5.4, 4.9],
        "lower_bounds": ["ninf", "ninf"],
    },
    "objective_data": {
        "coefficients": [0.2, 0.1],
        "scalability_factor": 1.0,
        "offset": 0.0,
    },
    "variable_bounds": {
        "upper_bounds": ["inf", "inf"],
        "lower_bounds": [0.0, 0.0],
    },
    "maximize": False,
    "solver_config": {"tolerances": {"optimality": 0.0001}},
}


# fmt: off
lp_msgpack_example_data = "\x86\xb5csr_constraint_matrix\x83\xa7offsets\x93\x00\x02\x04\xa7indices\x94\x00\x01\x00\x01\xa6values\x94\x03\x04\xcb@\x05\x99\x99\x99\x99\x99\x9a\xcb@$333333\xb1constraint_bounds\x82\xacupper_bounds\x92\xcb@\x15\x99\x99\x99\x99\x99\x9a\xcb@\x13\x99\x99\x99\x99\x99\x9a\xaclower_bounds\x92\xa4ninf\xa4ninf\xaeobjective_data\x83\xaccoefficients\x92\xcb?\xc9\x99\x99\x99\x99\x99\x9a\xcb?\xb9\x99\x99\x99\x99\x99\x9a\xb2scalability_factor\x01\xa6offset\x00\xafvariable_bounds\x82\xacupper_bounds\x92\xa3inf\xa3inf\xaclower_bounds\x92\x00\x00\xa8maximize\xc2\xadsolver_config\x81\xaatolerances\x81\xaaoptimality\xcb?\x1a6\xe2\xeb\x1cC-".encode("unicode_escape") # noqa

lp_zlib_example_data = 'x\x01\x8dR\xd1j\xc4 \x10|\xcfW\x88\xcf%$\xd7\x94\xd2\xfeJ9\xc2\xc6h\xd9b4\xa8\xc9]{\xe4\xdf\xab1\x9a\x0b\x1c\xa5o\xbb#\xb3;\xb3\xe3\xad \x842kZ\xa6\x95u\x06P\xb9v\x00g\xf0J\xdf\xc9\xcd?\xfag-\x84\xe5\xcez\xe0c\x05\x08\xa9\x9e\xb6\xe2\x94\x8af\x05\xce\xb1\xa5\xa8zd\xfc!\xa3N\x8c<\xa3>Pg\x90\xd3\x81\xf9\x9c\x08M*N\xe5k*\xeb\xaa\xdc\xe8~\xc8\x12Pzg\xa4\xd3\x93\xea\x83\x8a\xcd\xc84\x8e\xdc\xb4\x19Mn^\xca<\xba)\xdf\x0ej\xa4\xbe<bP\x85J\xd0$"v\x91\x97e\xe8\xee\x8b3\x873o{p\xb0k`\x9a\x0b\x81\x0c\xb9:^\xb4\xcc\xa7\xcc\x96\xe2|j\x19H\xe8P\xa2\xfbn\x050\xa7\x8d\x9f\xb6\x9dq\xcb\xc6\x03U\xde<\x83A\xe8$\xdf\x8d\xfem\xdf\xa7u\xe7%4\xff9A\x8e/,&\xe4\x9c\xd7\x0fp\xc5\x01\x7f\xb8\xd7$@Z\xbe\x86b\xb5\x9c\xfd!}6\x02?\xf7c8-\xb9\x01\x15\xbfJT\x19\xfe\xdb\xe8p\x80\xe07\xf8*\xab\xaa\x8a\x19/aG\xb1\x14E\xf1\x0b;\x07\xa4W'.encode("unicode_escape") # noqa
# fmt: on

managed_lp_example_data = {
    "action": "cuOpt_LP",
    "data": lp_example_data,
    "client_version": __version_major_minor__,
}

# cut and pasted from actual run of LP example data.
# don't reformat :)
lp_response = {
    "value": {
        "response": {
            "solver_response": {
                "status": "Optimal",
                "solution": {
                    "problem_category": 0,
                    "primal_solution": [0.0, 0.0],
                    "dual_solution": [0.0, 0.0],
                    "primal_objective": 0.0,
                    "dual_objective": 0.0,
                    "solver_time": 43.0,
                    "vars": {},
                    "lp_statistics": {
                        "primal_residual": 0.0,
                        "dual_residual": 0.0,
                        "gap": 0.0,
                    },
                    "reduced_cost": [0.2, 0.1],
                    "mip_statistics": {},
                },
            }
        },
        "reqId": "644dac46-5198-4293-ae3e-0f02a6d41863",
    }  # noqa
}

# cut and pasted from actual run of LP example data.
# don't reformat :)
milp_response = {
    "value": {
        "response": {
            "solver_response": {
                "status": "FeasibleFound",
                "solution": {
                    "problem_category": 1,
                    "primal_solution": [0.0, 0.0],
                    "dual_solution": None,
                    "primal_objective": 0.0,
                    "dual_objective": None,
                    "solver_time": 43.0,
                    "vars": {},
                    "lp_statistics": {},
                    "mip_statistics": {
                        "mip_gap": 0.0,
                        "presolve_time": 0.0,
                        "solution_bound": 0.0,
                        "max_constraint_violation": 0.0,
                        "max_int_violation": 0.0,
                        "max_variable_bound_violation": 0.0,
                    },
                },
            }
        },
        "reqId": "644dac46-5198-4293-ae3e-0f02a6d41863",
    }  # noqa
}

# cut and pasted from actual run of LP example data.
# don't reformat :)
milp_response = {
    "value": {
        "response": {
            "solver_response": {
                "status": "FeasibleFound",
                "solution": {
                    "problem_category": 1,
                    "primal_solution": [0.0, 0.0],
                    "dual_solution": None,
                    "primal_objective": 0.0,
                    "dual_objective": None,
                    "solver_time": 43.0,
                    "vars": {},
                    "lp_statistics": {},
                    "mip_statistics": {
                        "mip_gap": 0.0,
                        "presolve_time": 0.0,
                        "solution_bound": 0.0,
                        "max_constraint_violation": 0.0,
                        "max_int_violation": 0.0,
                        "max_variable_bound_violation": 0.0,
                    },
                },
            }
        },
        "reqId": "644dac46-5198-4293-ae3e-0f02a6d41863",
    }  # noqa
}

managed_lp_response = copy.deepcopy(lp_response)
del managed_lp_response["value"]["reqId"]

managed_milp_response = copy.deepcopy(lp_response)
del managed_milp_response["value"]["reqId"]

lpschema = jsonref.loads(json.dumps(LPData.model_json_schema()), proxies=False)
del lpschema["$defs"]
