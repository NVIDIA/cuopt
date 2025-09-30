# SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.  # noqa
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

import os
import time

from . import data_model_wrapper
from .utilities import catch_cuopt_exception


class DataModel(data_model_wrapper.DataModel):
    """
    Initialize a DataModel which represents a Linear Program.

    A linear programming optimization problem is defined as follows:
    Minimize :

      dot(c, x)

    Subject to :

      matmul(A, x) (= or >= or)<= b

    Where :

      x : Decision Variables

      c : Objective Coefficients

      A : Constraint Matrix

      b : Constraint Bounds

    With :

      n = number of variables

      m = number of constraints

      x = n-dim vector

      c = n-dim vector

      A = mxn-dim sparse matrix

      b = m-dim vector


    Notes
    -----
    By default, this assumes objective minimization.
    To solve a maximization problem, see set_maximization()

    Objective value can be scaled and offseted accordingly:
    objective_scaling_factor * (dot(c, x) + objective_offset)
    please refer to to the `set_objective_scaling_factor()`
    and `set_objective_offset()` method.

    Examples
    --------
    Minimize:

        cost = 0.2 * VAR1 + 0.1 * VAR2

    Subject to

        3   * VAR1 + 4    * VAR2 <= 5.4

        2.7 * VAR1 + 10.1 * VAR2 <= 4.9

        0 <= VAR1 <= 2

        0 <= VAR2 <= inf

    >>> from cuopt import linear_programming
    >>>
    >>> import numpy as np
    >>>
    >>> data_model = linear_programming.DataModel()
    >>>
    >>> # Set the CSR matrix representation, for more information about CSR
    >>> # checkout:
    >>> # https://docs.nvidia.com/cuda/cusparse/index.html
    #compressed-sparse-row-csr
    >>>
    >>> # Define the different np.array for the CSR representation
    >>> # The 4 values of the constraint matrix (A)
    >>> A_values = np.array([3.0, 4.0, 2.7, 10.1], dtype=np.float64)
    >>>
    >>> # The CSR index vector
    >>> # Here we associate each value in the A_values to its variable index
    >>> # First value correspond to the first variable
    >>> # (3.0 -> variable[*0*], constraint[0])
    >>> # Second value correspond to the second variable
    >>> # (4.0 -> variable[*1*], constraint[0])
    >>> # Third value correspond to the first variable
    >>> # (2.7 -> variable[*0*], constraint[1])
    >>> # Fourth value correspond to the second variable
    >>> # (10.1 -> variable[*1*], constraint[1])
    >>> A_indices = np.array([0, 1, 0, 1], dtype=np.int32)
    >>>
    >>> # The CSR offset vector
    >>> # Here we specify the range of values for each constraint
    >>> # [0, 2) corresponds to the range of values for the first constraints,
    >>> # here [0:3.0, 1:4.0]
    >>> # [2, 4) corresponds to the range of values for the second constraint,
    >>> # here [1:2.7, 2:10.1]
    >>> A_offsets = np.array([0, 2, 4], dtype=np.int32)
    >>>
    >>> data_model.set_csr_constraint_matrix(A_values, A_indices, A_offsets)
    >>>
    >>> # Set the constraint bounds (b / right-hand side) array
    >>> b = np.array([5.4, 4.9], dtype=np.float64)
    >>> data_model.set_constraint_bounds(b)
    >>>
    >>> # Set the objective coefficient (c) array.
    >>> c = np.array([0.2, 0.1], dtype=np.float64)
    >>> data_model.set_objective_coefficients(c)
    >>>
    >>> # Set the constraints/rows equalities, either using the row_type format
    >>> # or by directly setting the bounds
    >>>
    >>> # Method 0: constraints/rows type
    >>> # Set both constraints/rows to less-than (<=)
    >>> row_types = np.array(['L', 'L'])
    >>> data_model.set_row_types(row_types)
    >>>
    >>> # Method 1: directly set bounds
    >>> # Set lower bounds to -infinity and upper bounds to b
    >>> constraint_lower_bounds = np.array([np.NINF, np.NINF],
    >>>                                       dtype=np.float64)
    >>> constraint_upper_bounds = np.array(b, dtype=np.float64)
    >>> data_model.set_constraint_lower_bounds(constraint_lower_bounds)
    >>> data_model.set_constraint_upper_bounds(constraint_upper_bounds)
    >>>
    >>>
    >>> # Set variable lower and upper bounds
    >>> variable_lower_bounds = np.array([0.0, 0.0], dtype=np.float64)
    >>> variable_upper_bounds = np.array([2.0, np.PINF], dtype=np.float64)
    >>> data_model.set_variable_lower_bounds(variable_lower_bounds)
    >>> data_model.set_variable_upper_bounds(variable_upper_bounds)
    """

    def __init__(self):
        if os.environ.get("CUOPT_EXTRA_TIMESTAMPS", False) in (
            True,
            "True",
            "true",
        ):
            print(f"CUOPT_CREATE_PROBLEM: {time.time()}")
        super().__init__()

    @catch_cuopt_exception
    def set_maximize(self, maximize):
        """
        Set the sense of optimization to maximize.

        Parameters
        ----------
        maximize : bool
            True means to maximize the objective function, else minimize.

        Notes
        -----
        Setting before calling the solver is optional, default value if false
        (minimize).
        """
        super().set_maximize(maximize)

    @catch_cuopt_exception
    def set_csr_constraint_matrix(self, A_values, A_indices, A_offsets):
        """
        Set the constraint matrix (A) in CSR format.
        For more information about CSR checkout:
        https://docs.nvidia.com/cuda/cusparse/index.html
        compressed-sparse-row-csr

        Parameters
        ----------
        A_values : np.array dtype - float64
            Values of the CSR representation of the constraint matrix as a
            device floating point array.
        A_indices : np.array dtype - int32
            Indices of the CSR representation of the constraint matrix as a
            device integer array.
        A_offsets : np.array dtype - int32
            Offsets of the CSR representation of the constraint matrix as a
            device integer array.

        Notes
        -----
        Setting before calling the solver is mandatory.

        """
        super().set_csr_constraint_matrix(A_values, A_indices, A_offsets)

    @catch_cuopt_exception
    def set_constraint_bounds(self, b):
        """
        Set the constraint bounds (b / right-hand side) array.

        Parameters
        ----------
        b : np.array dtype - float64
            Device floating point array.

        Notes
        -----
        Setting before calling the solver is mandatory.

        """
        super().set_constraint_bounds(b)

    @catch_cuopt_exception
    def set_objective_coefficients(self, c):
        """
        Set the objective coefficients (c) array.

        Parameters
        ----------
        c : np.array dtype - float64
            Device floating point array.

        Notes
        -----
        Setting before calling the solver is mandatory.

        """
        super().set_objective_coefficients(c)

    @catch_cuopt_exception
    def set_objective_scaling_factor(self, objective_scaling_factor):
        """
        Set the scaling factor of the objective function
        (scaling_factor * objective_value).

        Parameters
        ----------
        objective_scaling_factor : float64
            The scaling factor to apply.

        Notes
        -----
        Setting before calling the solver is optional.
        """
        super().set_objective_scaling_factor(objective_scaling_factor)

    @catch_cuopt_exception
    def set_objective_offset(self, objective_offset):
        """
        Set the offset of the objective function
        (objective_offset + objective_value).

        Parameters
        ----------
        objective_offset : float64
            The constant objective_offset to add.

        Notes
        -----
        Setting before calling the solver is optional.
        """
        super().set_objective_offset(objective_offset)

    @catch_cuopt_exception
    def set_variable_lower_bounds(self, variable_lower_bounds):
        """
        Set the variables (x) lower bounds.

        Parameters
        ----------
        variable_lower_bounds : np.array dtype - float64
            Device floating point array.

        Notes
        -----
        Setting before calling the solver is optional, default value for all is
        0.

        """
        super().set_variable_lower_bounds(variable_lower_bounds)

    @catch_cuopt_exception
    def set_variable_upper_bounds(self, variable_upper_bounds):
        """
        Set the variables (x) upper bounds.

        Parameters
        ----------
        variable_upper_bounds : np.array dtype - float64
            Device floating point array.

        Notes
        -----
        Setting before calling the solver is optional, default value for all is
        +infinity.

        """
        super().set_variable_upper_bounds(variable_upper_bounds)

    @catch_cuopt_exception
    def set_constraint_lower_bounds(self, constraint_lower_bounds):
        """
        Set the constraints lower bounds.

        Parameters
        ----------
        constraint_lower_bounds : np.array dtype - float64
            Device floating point array.

        Notes
        -----
        Setting before calling the solver is optional if you set the row type,
        else it's mandatory along with the upper bounds.

        """
        super().set_constraint_lower_bounds(constraint_lower_bounds)

    @catch_cuopt_exception
    def set_constraint_upper_bounds(self, constraint_upper_bounds):
        """
        Set the constraints upper bounds.

        Parameters
        ----------
        constraint_upper_bounds : np.array dtype - float64
            Device floating point array.

        Notes
        -----
        Setting before calling the solver is optional if you set the row type,
        else it's mandatory along with the lower bounds.

        """
        super().set_constraint_upper_bounds(constraint_upper_bounds)

    @catch_cuopt_exception
    def set_variable_types(self, variable_types):
        """
        Set the variable types.

        Parameters
        ----------
        variables_types : np.array dtype - unicode string (<U1)
            Host character array.
        """
        super().set_variable_types(variable_types)

    @catch_cuopt_exception
    def set_row_types(self, row_types):
        """
        Set the type of each row (constraint). Possible values are:
        'E' for equality ( = ): lower & upper constrains bound equal to b
        'L' for less-than ( <= ): lower constrains bound equal to -infinity,
        upper constrains bound equal to b
        'G' for greater-than ( >= ): lower constrains bound equal to b,
        upper constrains bound equal to +infinity

        Parameters
        ----------
        row_types : np.array dtype - unicode string (<U1)
            Host character array.

        Notes
        -----
        Setting before calling the solver is optional if you set the constraint
        lower and upper bounds, else it's mandatory. If both are set,
        priority goes to set_constraint_lower/upper_bounds.

        Examples
        --------
        >>> row_types = np.array(['L', 'L'])
        >>> data_model.set_row_types(row_types)
        """
        super().set_row_types(row_types)

    @catch_cuopt_exception
    def set_variable_names(self, variables_names):
        """
        Set the variables names.

        Parameters
        ----------
        variables_names : np.array dtype - unicode string
            Host string array.

        Notes
        -----
        Setting before calling the solver is optional. Value is only used for
        file generation of the solution.
        """
        super().set_variable_names(variables_names)

    @catch_cuopt_exception
    def set_row_names(self, row_names):
        """
        Set the row names.

        Parameters
        ----------
        row_names : np.array dtype - unicode string
            Host string array.

        Notes
        -----
        Setting before calling the solver is optional. Value is only used for
        file generation of the solution.
        """
        super().set_row_names(row_names)

    @catch_cuopt_exception
    def set_objective_name(self, objective_name):
        """
        Set the objective name as string.
        """
        super().set_objective_name(objective_name)

    @catch_cuopt_exception
    def set_problem_name(self, problem_name):
        """
        Set the problem name as string.
        """
        super().set_problem_name(problem_name)

    @catch_cuopt_exception
    def set_initial_primal_solution(self, initial_primal_solution):
        """
        Set the initial primal solution.

        Parameters
        ----------
        initial_primal_solution : np.array dtype - float64
            Device floating point array.

        Notes
        -----
        Setting before calling the solver is optional.

        """
        return super().set_initial_primal_solution(initial_primal_solution)

    @catch_cuopt_exception
    def set_initial_dual_solution(self, initial_dual_solution):
        """
        NOTE: Not supported for MILP.

        Set the initial dual solution.

        Parameters
        ----------
        initial_dual_solution : np.array dtype - float64
            Device floating point array.

        Notes
        -----
        Setting before calling the solver is optional.

        """
        return super().set_initial_dual_solution(initial_dual_solution)

    @catch_cuopt_exception
    def get_sense(self):
        """
        Get the sense of optimization as a bool.
        True means maximize the objective function, false means minimize.
        """
        return super().get_sense()

    @catch_cuopt_exception
    def get_constraint_matrix_values(self):
        """
        Get the values of the CSR representation of the constraint matrix as
        numpy.array with float64 type.

        """
        return super().get_constraint_matrix_values()

    @catch_cuopt_exception
    def get_constraint_matrix_indices(self):
        """
        Get the indices of the CSR representation of the constraint matrix as
        numpy.array with int type.

        """
        return super().get_constraint_matrix_indices()

    @catch_cuopt_exception
    def get_constraint_matrix_offsets(self):
        """
        Get the indices of the CSR representation of the constraint matrix as
        numpy.array with int type.

        """
        return super().get_constraint_matrix_offsets()

    @catch_cuopt_exception
    def get_constraint_bounds(self):
        """
        Get the constraint bounds (b / right-hand side) as numpy.array with
        float64 type.

        """
        return super().get_constraint_bounds()

    @catch_cuopt_exception
    def get_objective_coefficients(self):
        """
        Get the objective_coefficients (c) as numpy.array with float64 type.

        """
        return super().get_objective_coefficients()

    @catch_cuopt_exception
    def get_objective_scaling_factor(self):
        """
        Get the scaling factor of the objective function as a float64.
        """
        return super().get_objective_scaling_factor()

    @catch_cuopt_exception
    def get_objective_offset(self):
        """
        Get the offset of the objective function as a float64.
        """
        return super().get_objective_offset()

    @catch_cuopt_exception
    def get_variable_lower_bounds(self):
        """
        Get the variables (x) lower bounds as numpy.array with float64 type.

        """
        return super().get_variable_lower_bounds()

    @catch_cuopt_exception
    def get_variable_upper_bounds(self):
        """
        Get the variables (x) upper bounds as numpy.array with float64 type.

        """
        return super().get_variable_upper_bounds()

    @catch_cuopt_exception
    def get_constraint_lower_bounds(self):
        """
        Get the constraints lower bounds as numpy.array with float64 type.

        """
        return super().get_constraint_lower_bounds()

    @catch_cuopt_exception
    def get_constraint_upper_bounds(self):
        """
        Get the constraints upper bounds as numpy.array with float64 type.

        """
        return super().get_constraint_upper_bounds()

    @catch_cuopt_exception
    def get_row_types(self):
        """
        Get the type of each row (constraint) as numpy.array with char8 type.

        """
        return super().get_row_types()

    @catch_cuopt_exception
    def get_ascii_row_types(self):
        """
        Get the type of each row (constraint) converted to a numpy.array with
        int8 type (ascii value).

        """
        return super().get_ascii_row_types()

    @catch_cuopt_exception
    def get_initial_primal_solution(self):
        """
        Get the initial primal solution as numpy.array with float64 type.

        """
        return super().get_initial_primal_solution()

    @catch_cuopt_exception
    def get_initial_dual_solution(self):
        """
        NOTE: Not supported for MILP.

        Get the initial dual solution as numpy.array with float64 type.

        """
        return super().get_initial_dual_solution()

    @catch_cuopt_exception
    def get_variable_types(self):
        """
        Get the variable types as numpy.array with char type.

        """
        return super().get_variable_types()

    @catch_cuopt_exception
    def get_variable_names(self):
        """
        Get the variable names as numpy.array with string type.

        """
        return super().get_variable_names()

    @catch_cuopt_exception
    def get_row_names(self):
        """
        Get the row names as numpy.array with string type.

        """
        return super().get_row_names()

    @catch_cuopt_exception
    def get_objective_name(self):
        """
        Get the objective name as string.
        """
        return super().get_objective_name()

    @catch_cuopt_exception
    def get_problem_name(self):
        """
        Get the problem name as string.
        """
        return super().get_problem_name()

    @catch_cuopt_exception
    def writeMPS(self, user_problem_file):
        return super().writeMPS(user_problem_file)
