# SPDX-FileCopyrightText: Copyright (c) 2023-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved. # noqa
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


# cython: profile=False
# distutils: language = c++
# cython: embedsignature = True
# cython: language_level = 3

from libcpp.memory cimport unique_ptr
from libcpp.string cimport string

from cuopt.linear_programming.solver.solver cimport solver_settings_t


def get_solver_setting(name):
    cdef unique_ptr[solver_settings_t[int, double]] unique_solver_settings

    unique_solver_settings.reset(new solver_settings_t[int, double]())

    cdef solver_settings_t[int, double]* c_solver_settings = (
        unique_solver_settings.get()
    )
    return c_solver_settings.get_parameter_as_string(
        name.encode('utf-8')
    ).decode('utf-8')

# Get the C #defines from the constants.h file
cdef extern from "cuopt/linear_programming/constants.h": # noqa
    cdef const char* c_CUOPT_ABSOLUTE_DUAL_TOLERANCE  "CUOPT_ABSOLUTE_DUAL_TOLERANCE" # noqa
    cdef const char* c_CUOPT_RELATIVE_DUAL_TOLERANCE "CUOPT_RELATIVE_DUAL_TOLERANCE" # noqa
    cdef const char* c_CUOPT_ABSOLUTE_PRIMAL_TOLERANCE "CUOPT_ABSOLUTE_PRIMAL_TOLERANCE" # noqa
    cdef const char* c_CUOPT_RELATIVE_PRIMAL_TOLERANCE "CUOPT_RELATIVE_PRIMAL_TOLERANCE" # noqa
    cdef const char* c_CUOPT_ABSOLUTE_GAP_TOLERANCE "CUOPT_ABSOLUTE_GAP_TOLERANCE" # noqa
    cdef const char* c_CUOPT_RELATIVE_GAP_TOLERANCE "CUOPT_RELATIVE_GAP_TOLERANCE" # noqa
    cdef const char* c_CUOPT_INFEASIBILITY_DETECTION "CUOPT_INFEASIBILITY_DETECTION" # noqa
    cdef const char* c_CUOPT_STRICT_INFEASIBILITY "CUOPT_STRICT_INFEASIBILITY" # noqa
    cdef const char* c_CUOPT_PRIMAL_INFEASIBLE_TOLERANCE "CUOPT_PRIMAL_INFEASIBLE_TOLERANCE" # noqa
    cdef const char* c_CUOPT_DUAL_INFEASIBLE_TOLERANCE "CUOPT_DUAL_INFEASIBLE_TOLERANCE" # noqa
    cdef const char* c_CUOPT_ITERATION_LIMIT "CUOPT_ITERATION_LIMIT" # noqa
    cdef const char* c_CUOPT_TIME_LIMIT "CUOPT_TIME_LIMIT" # noqa
    cdef const char* c_CUOPT_PDLP_SOLVER_MODE "CUOPT_PDLP_SOLVER_MODE" # noqa
    cdef const char* c_CUOPT_METHOD "CUOPT_METHOD" # noqa
    cdef const char* c_CUOPT_PER_CONSTRAINT_RESIDUAL "CUOPT_PER_CONSTRAINT_RESIDUAL" # noqa
    cdef const char* c_CUOPT_SAVE_BEST_PRIMAL_SO_FAR "CUOPT_SAVE_BEST_PRIMAL_SO_FAR" # noqa
    cdef const char* c_CUOPT_FIRST_PRIMAL_FEASIBLE "CUOPT_FIRST_PRIMAL_FEASIBLE" # noqa
    cdef const char* c_CUOPT_LOG_FILE "CUOPT_LOG_FILE" # noqa
    cdef const char* c_CUOPT_LOG_TO_CONSOLE "CUOPT_LOG_TO_CONSOLE" # noqa
    cdef const char* c_CUOPT_CROSSOVER "CUOPT_CROSSOVER" # noqa
    cdef const char* c_CUOPT_PRESOLVE "CUOPT_PRESOLVE" # noqa
    cdef const char* c_CUOPT_DUAL_POSTSOLVE "CUOPT_DUAL_POSTSOLVE" # noqa
    cdef const char* c_CUOPT_MIP_ABSOLUTE_TOLERANCE "CUOPT_MIP_ABSOLUTE_TOLERANCE" # noqa
    cdef const char* c_CUOPT_MIP_RELATIVE_TOLERANCE "CUOPT_MIP_RELATIVE_TOLERANCE" # noqa
    cdef const char* c_CUOPT_MIP_INTEGRALITY_TOLERANCE "CUOPT_MIP_INTEGRALITY_TOLERANCE" # noqa
    cdef const char* c_CUOPT_MIP_ABSOLUTE_GAP "CUOPT_MIP_ABSOLUTE_GAP" # noqa
    cdef const char* c_CUOPT_MIP_RELATIVE_GAP "CUOPT_MIP_RELATIVE_GAP" # noqa
    cdef const char* c_CUOPT_MIP_HEURISTICS_ONLY "CUOPT_MIP_HEURISTICS_ONLY" # noqa
    cdef const char* c_CUOPT_MIP_SCALING "CUOPT_MIP_SCALING" # noqa
    cdef const char* c_CUOPT_SOLUTION_FILE "CUOPT_SOLUTION_FILE" # noqa
    cdef const char* c_CUOPT_NUM_CPU_THREADS "CUOPT_NUM_CPU_THREADS" # noqa
    cdef const char* c_CUOPT_USER_PROBLEM_FILE "CUOPT_USER_PROBLEM_FILE" # noqa
    cdef const char* c_CUOPT_AUGMENTED "CUOPT_AUGMENTED"
    cdef const char* c_CUOPT_FOLDING "CUOPT_FOLDING"
    cdef const char* c_CUOPT_DUALIZE "CUOPT_DUALIZE"
    cdef const char* c_CUOPT_ELIMINATE_DENSE_COLUMNS "CUOPT_ELIMINATE_DENSE_COLUMNS" # noqa
    cdef const char* c_CUOPT_CUDSS_DETERMINISTIC "CUOPT_CUDSS_DETERMINISTIC" # noqa
    cdef const char* c_CUOPT_ORDERING "CUOPT_ORDERING" # noqa

# Create Python string constants from C string literals
CUOPT_ABSOLUTE_DUAL_TOLERANCE = c_CUOPT_ABSOLUTE_DUAL_TOLERANCE.decode('utf-8') # noqa
CUOPT_RELATIVE_DUAL_TOLERANCE = c_CUOPT_RELATIVE_DUAL_TOLERANCE.decode('utf-8') # noqa
CUOPT_ABSOLUTE_PRIMAL_TOLERANCE = c_CUOPT_ABSOLUTE_PRIMAL_TOLERANCE.decode('utf-8') # noqa
CUOPT_RELATIVE_PRIMAL_TOLERANCE = c_CUOPT_RELATIVE_PRIMAL_TOLERANCE.decode('utf-8') # noqa
CUOPT_ABSOLUTE_GAP_TOLERANCE = c_CUOPT_ABSOLUTE_GAP_TOLERANCE.decode('utf-8') # noqa
CUOPT_RELATIVE_GAP_TOLERANCE = c_CUOPT_RELATIVE_GAP_TOLERANCE.decode('utf-8') # noqa
CUOPT_INFEASIBILITY_DETECTION = c_CUOPT_INFEASIBILITY_DETECTION.decode('utf-8') # noqa
CUOPT_STRICT_INFEASIBILITY = c_CUOPT_STRICT_INFEASIBILITY.decode('utf-8') # noqa
CUOPT_PRIMAL_INFEASIBLE_TOLERANCE = c_CUOPT_PRIMAL_INFEASIBLE_TOLERANCE.decode('utf-8') # noqa
CUOPT_DUAL_INFEASIBLE_TOLERANCE = c_CUOPT_DUAL_INFEASIBLE_TOLERANCE.decode('utf-8') # noqa
CUOPT_ITERATION_LIMIT = c_CUOPT_ITERATION_LIMIT.decode('utf-8') # noqa
CUOPT_TIME_LIMIT = c_CUOPT_TIME_LIMIT.decode('utf-8') # noqa
CUOPT_PDLP_SOLVER_MODE = c_CUOPT_PDLP_SOLVER_MODE.decode('utf-8') # noqa
CUOPT_METHOD = c_CUOPT_METHOD.decode('utf-8') # noqa
CUOPT_PER_CONSTRAINT_RESIDUAL = c_CUOPT_PER_CONSTRAINT_RESIDUAL.decode('utf-8') # noqa
CUOPT_SAVE_BEST_PRIMAL_SO_FAR = c_CUOPT_SAVE_BEST_PRIMAL_SO_FAR.decode('utf-8') # noqa
CUOPT_FIRST_PRIMAL_FEASIBLE = c_CUOPT_FIRST_PRIMAL_FEASIBLE.decode('utf-8') # noqa
CUOPT_LOG_FILE = c_CUOPT_LOG_FILE.decode('utf-8') # noqa
CUOPT_LOG_TO_CONSOLE = c_CUOPT_LOG_TO_CONSOLE.decode('utf-8') # noqa
CUOPT_CROSSOVER = c_CUOPT_CROSSOVER.decode('utf-8') # noqa
CUOPT_PRESOLVE = c_CUOPT_PRESOLVE.decode('utf-8') # noqa
CUOPT_DUAL_POSTSOLVE = c_CUOPT_DUAL_POSTSOLVE.decode('utf-8') # noqa
CUOPT_MIP_ABSOLUTE_TOLERANCE = c_CUOPT_MIP_ABSOLUTE_TOLERANCE.decode('utf-8') # noqa
CUOPT_MIP_RELATIVE_TOLERANCE = c_CUOPT_MIP_RELATIVE_TOLERANCE.decode('utf-8') # noqa
CUOPT_MIP_INTEGRALITY_TOLERANCE = c_CUOPT_MIP_INTEGRALITY_TOLERANCE.decode('utf-8') # noqa
CUOPT_MIP_ABSOLUTE_GAP = c_CUOPT_MIP_ABSOLUTE_GAP.decode('utf-8') # noqa
CUOPT_MIP_RELATIVE_GAP = c_CUOPT_MIP_RELATIVE_GAP.decode('utf-8') # noqa
CUOPT_MIP_HEURISTICS_ONLY = c_CUOPT_MIP_HEURISTICS_ONLY.decode('utf-8') # noqa
CUOPT_MIP_SCALING = c_CUOPT_MIP_SCALING.decode('utf-8') # noqa
CUOPT_SOLUTION_FILE = c_CUOPT_SOLUTION_FILE.decode('utf-8') # noqa
CUOPT_NUM_CPU_THREADS = c_CUOPT_NUM_CPU_THREADS.decode('utf-8') # noqa
CUOPT_USER_PROBLEM_FILE = c_CUOPT_USER_PROBLEM_FILE.decode('utf-8') # noqa
CUOPT_AUGMENTED = c_CUOPT_AUGMENTED.decode('utf-8') # noqa
CUOPT_FOLDING = c_CUOPT_FOLDING.decode('utf-8') # noqa
CUOPT_DUALIZE = c_CUOPT_DUALIZE.decode('utf-8') # noqa
CUOPT_ELIMINATE_DENSE_COLUMNS = c_CUOPT_ELIMINATE_DENSE_COLUMNS.decode('utf-8') # noqa
CUOPT_CUDSS_DETERMINISTIC = c_CUOPT_CUDSS_DETERMINISTIC.decode('utf-8') # noqa
CUOPT_ORDERING = c_CUOPT_ORDERING.decode('utf-8') # noqa
