# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.  # noqa
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

from enum import Enum

import numpy as np

import cuopt.linear_programming.data_model as data_model
import cuopt.linear_programming.solver as solver
import cuopt.linear_programming.solver_settings as solver_settings


class VType(str, Enum):
    """
    The type of a variable is either continuous or integer.
    Variable Types can be directly used as a constant.
    CONTINUOUS is  VType.CONTINUOUS
    INTEGER is VType.INTEGER
    """

    CONTINUOUS = "C"
    INTEGER = "I"


CONTINUOUS = VType.CONTINUOUS
INTEGER = VType.INTEGER


class CType(str, Enum):
    """
    The sense of a constraint is either LE, GE or EQ.
    Constraint Sense Types can be directly used as a constant.
    LE is CType.LE
    GE is CType.GE
    EQ is CType EQ
    """

    LE = "L"
    GE = "G"
    EQ = "E"


LE = CType.LE
GE = CType.GE
EQ = CType.EQ


class sense(int, Enum):
    """
    The sense of a model is either MINIMIZE or MAXIMIZE.
    Model objective sense can be directly used as a constant.
    MINIMIZE is sense.MINIMIZE
    MAXIMIZE is sense.MAXIMIZE
    """

    MAXIMIZE = -1
    MINIMIZE = 1


MAXIMIZE = sense.MAXIMIZE
MINIMIZE = sense.MINIMIZE


class Variable:
    """
    cuOpt variable object initialized with details of the variable
    such as lower bound, upper bound, type and name.
    Variables are always associated with a problem and can be
    created using problem.addVariable (See problem class).

    Parameters
    ----------
    lb : float
        Lower bound of the variable. Defaults to  0.
    ub : float
        Upper bound of the variable. Defaults to infinity.
    vtype : enum
        CONTINUOUS or INTEGER. Defaults to CONTINUOUS.
    obj : float
        Coefficient of the Variable in the objective.
    name : str
        Name of the variable. Optional.

    Attributes
    ----------
    VariableName : str
        Name of the Variable.
    VariableType : CONTINUOUS or INTEGER
        Variable type.
    LB : float
        Lower Bound of the Variable.
    UB : float
        Upper Bound of the Variable.
    Obj : float
        Coefficient of the variable in the Objective function.
    Value : float
        Value of the variable after solving.
    ReducedCost : float
        Reduced Cost after solving an LP problem.
    """

    def __init__(
        self,
        lb=0.0,
        ub=float("inf"),
        obj=0.0,
        vtype=CONTINUOUS,
        vname="",
    ):
        self.index = -1
        self.LB = lb
        self.UB = ub
        self.Obj = obj
        self.Value = float("nan")
        self.ReducedCost = float("nan")
        self.VariableType = vtype
        self.VariableName = vname

    def getIndex(self):
        """
        Get the index position of the variable in the problem.
        """
        return self.index

    def getValue(self):
        """
        Returns the Value of the variable computed in current solution.
        Defaults to 0
        """
        return self.Value

    def getObjectiveCoefficient(self):
        """
        Returns the objective coefficient of the variable.
        """
        return self.Obj

    def setObjectiveCoefficient(self, val):
        """
        Sets the objective cofficient of the variable.
        """
        self.Obj = val

    def setLowerBound(self, val):
        """
        Sets the lower bound of the variable.
        """
        self.LB = val

    def getLowerBound(self):
        """
        Returns the lower bound of the variable.
        """
        return self.LB

    def setUpperBound(self, val):
        """
        Sets the upper bound of the variable.
        """
        self.UB = val

    def getUpperBound(self):
        """
        Returns the upper bound of the variable.
        """
        return self.UB

    def setVariableType(self, val):
        """
        Sets the variable type of the variable.
        Variable types can be either CONTINUOUS or INTEGER.
        """
        self.VariableType = val

    def getVariableType(self):
        """
        Returns the type of the variable.
        """
        return self.VariableType

    def setVariableName(self, val):
        """
        Sets the name of the variable.
        """
        self.VariableName = val

    def getVariableName(self):
        """
        Returns the name of the variable.
        """
        return self.VariableName

    def __add__(self, other):
        match other:
            case int() | float():
                return LinearExpression([self], [1.0], float(other))
            case Variable():
                # Change?
                return LinearExpression([self, other], [1.0, 1.0], 0.0)
            case LinearExpression():
                return other + self
            case _:
                raise ValueError(
                    "Cannot add type %s to variable" % type(other).__name__
                )

    def __radd__(self, other):
        return self + other

    def __sub__(self, other):
        match other:
            case int() | float():
                return LinearExpression([self], [1.0], -float(other))
            case Variable():
                return LinearExpression([self, other], [1.0, -1.0], 0.0)
            case LinearExpression():
                # self - other ->   other * -1.0 + self
                return other * -1.0 + self
            case _:
                raise ValueError(
                    "Cannot subtract type %s from variable"
                    % type(other).__name__
                )

    def __rsub__(self, other):
        # other - self  -> other + self * -1.0
        return other + self * -1.0

    def __mul__(self, other):
        match other:
            case int() | float():
                return LinearExpression([self], [float(other)], 0.0)
            case _:
                raise ValueError(
                    "Cannot multiply type %s with variable"
                    % type(other).__name__
                )

    def __rmul__(self, other):
        return self * other

    def __le__(self, other):
        match other:
            case int() | float():
                expr = LinearExpression([self], [1.0], 0.0)
                return Constraint(expr, LE, float(other))
            case Variable() | LinearExpression():
                # var1 <= var2   -> var1 - var2 <= 0
                expr = self - other
                return Constraint(expr, LE, 0.0)
            case _:
                raise ValueError("Unsupported operation")

    def __ge__(self, other):
        match other:
            case int() | float():
                expr = LinearExpression([self], [1.0], 0.0)
                return Constraint(expr, GE, float(other))
            case Variable() | LinearExpression():
                # var1 >= var2   ->  var1 - var2 >= 0
                expr = self - other
                return Constraint(expr, GE, 0.0)
            case _:
                raise ValueError("Unsupported operation")

    def __eq__(self, other):
        match other:
            case int() | float():
                expr = LinearExpression([self], [1.0], 0.0)
                return Constraint(expr, EQ, float(other))
            case Variable() | LinearExpression():
                # var1 == var2   -> var1 - var2 == 0
                expr = self - other
                return Constraint(expr, EQ, 0.0)
            case _:
                raise ValueError("Unsupported operation")


class LinearExpression:
    """
    LinearExpressions contain a set of variables, the coefficients
    for the variables, and a constant.
    LinearExpressions can be used to create constraints and the
    objective in the Problem.
    LinearExpressions can be added and subtracted with other
    LinearExpressions and Variables and can also be multiplied and
    divided by scalars.
    LinearExpressions can be compared with scalars, Variables, and
    other LinearExpressions to create Constraints.

    Parameters
    ----------
    vars : List
        List of Variables in the linear expression.
    coefficients : List
        List of coefficients corresponding to the variables.
    constant : float
        Constant of the linear expression.
    """

    def __init__(self, vars, coefficients, constant):
        self.vars = vars
        self.coefficients = coefficients
        self.constant = constant

    def getVariables(self):
        """
        Returns all the variables in the linear expression.
        """
        return self.vars

    def getVariable(self, i):
        """
        Gets Variable at ith index in the linear expression.
        """
        return self.vars[i]

    def getCoefficients(self):
        """
        Returns all the coefficients in the linear expression.
        """
        return self.coefficients

    def getCoefficient(self, i):
        """
        Gets the coefficient of the variable at ith index of the
        linear expression.
        """
        return self.coefficients[i]

    def getConstant(self):
        """
        Returns the constant in the linear expression.
        """
        return self.constant

    def zipVarCoefficients(self):
        return zip(self.vars, self.coefficients)

    def getValue(self):
        """
        Returns the value of the expression computed with the
        current solution.
        """
        value = 0.0
        for i, var in enumerate(self.vars):
            value += var.Value * self.coefficients[i]
        return value + self.constant

    def __len__(self):
        return len(self.vars)

    def __iadd__(self, other):
        # Compute expr1 += expr2
        match other:
            case int() | float():
                # Update just the constant value
                self.constant += float(other)
                return self
            case Variable():
                # Append just a variable with coefficient 1.0
                self.vars.append(other)
                self.coefficients.append(1.0)
                return self
            case LinearExpression():
                # Append all variables, coefficients and constants
                self.vars.extend(other.vars)
                self.coefficients.extend(other.coefficients)
                self.constant += other.constant
                return self
            case _:
                raise ValueError(
                    "Can't add type %s to Linear Expression"
                    % type(other).__name__
                )

    def __add__(self, other):
        # Compute expr3 = expr1 + expr2
        match other:
            case int() | float():
                # Update just the constant value
                return LinearExpression(
                    self.vars, self.coefficients, self.constant + float(other)
                )
            case Variable():
                # Append just a variable with coefficient 1.0
                vars = self.vars + [other]
                coeffs = self.coefficients + [1.0]
                return LinearExpression(vars, coeffs, self.constant)
            case LinearExpression():
                # Append all variables, coefficients and constants
                vars = self.vars + other.vars
                coeffs = self.coefficients + other.coefficients
                constant = self.constant + other.constant
                return LinearExpression(vars, coeffs, constant)

    def __radd__(self, other):
        return self + other

    def __isub__(self, other):
        # Compute expr1 -= expr2
        match other:
            case int() | float():
                # Update just the constant value
                self.constant -= float(other)
                return self
            case Variable():
                # Append just a variable with coefficient -1.0
                self.vars.append(other)
                self.coefficients.append(-1.0)
                return self
            case LinearExpression():
                # Append all variables, coefficients and constants
                self.vars.extend(other.vars)
                for coeff in other.coefficients:
                    self.coefficients.append(-coeff)
                self.constant -= other.constant
                return self
            case _:
                raise ValueError(
                    "Can't sub type %s from LinearExpression"
                    % type(other).__name__
                )

    def __sub__(self, other):
        # Compute expr3 = expr1 - expr2
        match other:
            case int() | float():
                # Update just the constant value
                return LinearExpression(
                    self.vars, self.coefficients, self.constant - float(other)
                )
            case Variable():
                # Append just a variable with coefficient -1.0
                vars = self.vars + [other]
                coeffs = self.coefficients + [-1.0]
                return LinearExpression(vars, coeffs, self.constant)
            case LinearExpression():
                # Append all variables, coefficients and constants
                vars = self.vars + other.vars
                coeffs = []
                for i in self.coefficients:
                    coeffs.append(i)
                for i in other.coefficients:
                    coeffs.append(-1.0 * i)
                constant = self.constant - other.constant
                return LinearExpression(vars, coeffs, constant)

    def __rsub__(self, other):
        # other - self  -> other + self * -1.0
        return other + self * -1.0

    def __imul__(self, other):
        # Compute expr *= constant
        match other:
            case int() | float():
                self.coefficients = [
                    coeff * float(other) for coeff in self.coefficients
                ]
                self.constant = self.constant * float(other)
                return self
            case _:
                raise ValueError(
                    "Can't multiply type %s by LinearExpresson"
                    % type(other).__name__
                )

    def __mul__(self, other):
        # Compute expr2 = expr1 * constant
        match other:
            case int() | float():
                coeffs = [coeff * float(other) for coeff in self.coefficients]
                constant = self.constant * float(other)
                return LinearExpression(self.vars, coeffs, constant)
            case _:
                raise ValueError(
                    "Can't multiply type %s by LinearExpresson"
                    % type(other).__name__
                )

    def __rmul__(self, other):
        return self * other

    def __itruediv__(self, other):
        # Compute expr /= constant
        match other:
            case int() | float():
                self.coefficients = [
                    coeff / float(other) for coeff in self.coefficients
                ]
                self.constant = self.constant / float(other)
                return self
            case _:
                raise ValueError(
                    "Can't divide LinearExpression by type %s"
                    % type(other).__name__
                )

    def __truediv__(self, other):
        # Compute expr2 = expr1 / constant
        match other:
            case int() | float():
                coeffs = [coeff / float(other) for coeff in self.coefficients]
                constant = self.constant / float(other)
                return LinearExpression(self.vars, coeffs, constant)
            case _:
                raise ValueError(
                    "Can't divide LinearExpression by type %s"
                    % type(other).__name__
                )

    def __le__(self, other):
        match other:
            case int() | float():
                return Constraint(self, LE, float(other))
            case Variable() | LinearExpression():
                # expr1 <= expr2   -> expr1 - expr2 <= 0
                expr = self - other
                return Constraint(expr, LE, 0.0)

    def __ge__(self, other):
        match other:
            case int() | float():
                return Constraint(self, GE, float(other))
            case Variable() | LinearExpression():
                # expr1 >= expr2   ->  expr1 - expr2 >= 0
                expr = self - other
                return Constraint(expr, GE, 0.0)

    def __eq__(self, other):
        match other:
            case int() | float():
                return Constraint(self, EQ, float(other))
            case Variable() | LinearExpression():
                # expr1 == expr2   -> expr1 - expr2 == 0
                expr = self - other
                return Constraint(expr, EQ, 0.0)


class Constraint:
    """
    cuOpt constraint object containing a linear expression,
    the sense of the constraint, and the right-hand side of
    the constraint.
    Constraints are associated with a problem and can be
    created using problem.addConstraint (See problem class).

    Parameters
    ----------
    expr : LinearExpression
        Linear expression corresponding to a problem.
    sense : enum
        Sense of the constraint. Either LE for <=,
        GE for >= or EQ for == .
    rhs : float
        Constraint right-hand side value.
    name : str, Optional
        Name of the constraint. Optional.

    Attributes
    ----------
    ConstraintName : str
        Name of the constraint.
    Sense : LE, GE or EQ
        Row sense. LE for >=, GE for <= or EQ for == .
    RHS : float
        Constraint right-hand side value.
    Slack : float
        Computed LHS - RHS with current solution.
    DualValue : float
        Constraint dual value in the current solution.
    """

    def __init__(self, expr, sense, rhs, name=""):
        self.vindex_coeff_dict = {}
        nz = len(expr)
        self.vars = expr.vars
        self.index = -1
        for i in range(nz):
            v_idx = expr.vars[i].index
            v_coeff = expr.coefficients[i]
            self.vindex_coeff_dict[v_idx] = (
                self.vindex_coeff_dict[v_idx] + v_coeff
                if v_idx in self.vindex_coeff_dict
                else v_coeff
            )
        self.Sense = sense
        self.RHS = rhs - expr.getConstant()
        self.ConstraintName = name
        self.DualValue = float("nan")
        self.Slack = float("nan")

    def __len__(self):
        return len(self.vindex_coeff_dict)

    def getConstraintName(self):
        """
        Returns the name of the constraint.
        """
        return self.ConstraintName

    def getSense(self):
        """
        Returns the sense of the constraint.
        Constraint sense can be LE(<=), GE(>=) or EQ(==).
        """
        return self.Sense

    def getRHS(self):
        """
        Returns the right-hand side value of the constraint.
        """
        return self.RHS

    def getCoefficient(self, var):
        """
        Returns the coefficient of a variable in the constraint.
        """
        v_idx = var.index
        return self.vindex_coeff_dict[v_idx]

    def compute_slack(self):
        # Computes the constraint Slack in the current solution.
        lhs = 0.0
        for var in self.vars:
            lhs += var.Value * self.vindex_coeff_dict[var.index]
        return self.RHS - lhs


class Problem:
    """
    A Problem defines a Linear Program or Mixed Integer Program
    Variable can be be created by calling addVariable()
    Constraints can be added by calling addConstraint()
    The objective can be set by calling setObjective()
    The problem data is formed when calling solve().

    Parameters
    ----------
    model_name : str, optional
        Name of the model. Default is an empty string.

    Attributes
    ----------
    Name : str
        Name of the model.
    ObjSense : sense
        Objective sense (MINIMIZE or MAXIMIZE).
    ObjConstant : float
        Constant term in the objective.
    Status : int
        Status of the problem after solving.
    SolveTime : float
        Time taken to solve the problem.
    SolutionStats : object
        Solution statistics for LP or MIP problem.
    ObjValue : float
        Objective value of the problem.
    IsMIP : bool
        Indicates if the problem is a Mixed Integer Program.
    NumVariables : int
        Number of Variables in the problem.
    NumConstraints : int
        Number of constraints in the problem.
    NumNZs : int
        Number of non-zeros in the problem.

    Examples
    --------
    >>> problem = problem.Problem("MIP_model")
    >>> x = problem.addVariable(lb=-2.0, ub=8.0, vtype=INTEGER)
    >>> y = problem.addVariable(name="Var2")
    >>> problem.addConstraint(2*x - 3*y <= 10, name="Constr1")
    >>> expr = 3*x + y
    >>> problem.addConstraint(expr + x == 20, name="Constr2")
    >>> problem.setObjective(x + y, sense=MAXIMIZE)
    >>> problem.solve()
    """

    def __init__(self, model_name=""):
        self.Name = model_name
        self.vars = []
        self.constrs = []
        self.ObjSense = MINIMIZE
        self.Obj = None
        self.ObjConstant = 0.0
        self.Status = -1
        self.ObjValue = float("nan")

        self.solved = False
        self.rhs = None
        self.row_sense = None
        self.row_pointers = None
        self.column_indicies = None
        self.values = None
        self.lower_bound = None
        self.upper_bound = None
        self.var_type = None

    class dict_to_object:
        def __init__(self, mdict):
            for key, value in mdict.items():
                setattr(self, key, value)

    def reset_solved_values(self):
        # Resets all post solve values
        for var in self.vars:
            var.Value = float("nan")
            var.ReducedCost = float("nan")

        for constr in self.constrs:
            constr.Slack = float("nan")
            constr.DualValue = float("nan")

        self.ObjValue = float("nan")
        self.solved = False

    def addVariable(
        self, lb=0.0, ub=float("inf"), obj=0.0, vtype=CONTINUOUS, name=""
    ):
        """
        Adds a variable to the problem defined by lower bound,
        upper bound, type and name.

        Parameters
        ----------
        lb : float
            Lower bound of the variable. Defaults to  0.
        ub : float
            Upper bound of the variable. Defaults to infinity.
        vtype : enum
            vtype.CONTINUOUS or vtype.INTEGER. Defaults to CONTINUOUS.
        name : string
            Name of the variable. Optional.

        Examples
        --------
        >>> problem = problem.Problem("MIP_model")
        >>> x = problem.addVariable(lb=-2.0, ub=8.0, vtype=INTEGER,
                name="Var1")
        """
        if self.solved:
            self.reset_solved_values()  # Reset all solved values
        n = len(self.vars)
        var = Variable(lb, ub, obj, vtype, name)
        var.index = n
        self.vars.append(var)
        return var

    def addConstraint(self, constr, name=""):
        """
        Adds a constraint to the problem defined by constraint object
        and name. A constraint is generated using LinearExpression,
        Sense and RHS.

        Parameters
        ----------
        constr : Constraint
            Constructed using LinearExpressions (See Examples)
        name : string
            Name of the variable. Optional.

        Examples
        --------
        >>> problem = problem.Problem("MIP_model")
        >>> x = problem.addVariable(lb=-2.0, ub=8.0, vtype=INTEGER)
        >>> y = problem.addVariable(name="Var2")
        >>> problem.addConstraint(2*x - 3*y <= 10, name="Constr1")
        >>> expr = 3*x + y
        >>> problem.addConstraint(expr + x == 20, name="Constr2")
        """
        if self.solved:
            self.reset_solved_values()  # Reset all solved values
        n = len(self.constrs)
        match constr:
            case Constraint():
                constr.index = n
                constr.ConstraintName = name
                self.constrs.append(constr)
            case _:
                raise ValueError("addConstraint requires a Constraint object")

    def setObjective(self, expr, sense=MINIMIZE):
        """
        Set the Objective of the problem with an expression that needs to
        be MINIMIZED or MAXIMIZED.

        Parameters
        ----------
        expr : LinearExpression or Variable or Constant
            Objective expression that needs maximization or minimization.
        sense : enum
            Sets whether the problem is a maximization or a minimization
            problem. Values passed can either be MINIMIZE or MAXIMIZE.
            Defaults to MINIMIZE.

        Examples
        --------
        >>> problem = problem.Problem("MIP_model")
        >>> x = problem.addVariable(lb=-2.0, ub=8.0, vtype=INTEGER)
        >>> y = problem.addVariable(name="Var2")
        >>> problem.addConstraint(2*x - 3*y <= 10, name="Constr1")
        >>> expr = 3*x + y
        >>> problem.addConstraint(expr + x == 20, name="Constr2")
        >>> problem.setObjective(x + y, sense=MAXIMIZE)
        """
        if self.solved:
            self.reset_solved_values()  # Reset all solved values
        self.ObjSense = sense
        match expr:
            case int() | float():
                for var in self.vars:
                    var.setObjectiveCoefficient(0.0)
                self.ObjCon = float(expr)
            case Variable():
                for var in self.vars:
                    var.setObjectiveCoefficient(0.0)
                    if var.getIndex() == expr.getIndex():
                        var.setObjectiveCoefficient(1.0)
            case LinearExpression():
                for var, coeff in expr.zipVarCoefficients():
                    self.vars[var.getIndex()].setObjectiveCoefficient(coeff)
            case _:
                raise ValueError(
                    "Objective must be a LinearExpression or a constant"
                )
        self.Obj = expr

    def getObjective(self):
        """
        Get the Objective expression of the problem.
        """
        return self.Obj

    def getVariables(self):
        """
        Get a list of all the variables in the problem.
        """
        return self.vars

    def getConstraints(self):
        """
        Get a list of all the Constraints in a problem.
        """
        return self.constrs

    @property
    def NumVariables(self):
        # Returns number of variables in the problem
        return len(self.vars)

    @property
    def NumConstraints(self):
        # Returns number of contraints in the problem.
        return len(self.constrs)

    @property
    def NumNZs(self):
        # Returns number of non-zeros in the problem.
        nnz = 0
        for constr in self.constrs:
            nnz += len(constr)
        return nnz

    @property
    def IsMIP(self):
        # Returns if the problem is a MIP problem.
        for var in self.vars:
            if var.VariableType == "I":
                return True
        return False

    def getCSR(self):
        """
        Computes and returns the CSR representation of the
        constraint matrix.
        """
        csr_dict = {"row_pointers": [0], "column_indices": [], "values": []}
        for constr in self.constrs:
            csr_dict["column_indices"].extend(
                list(constr.vindex_coeff_dict.keys())
            )
            csr_dict["values"].extend(list(constr.vindex_coeff_dict.values()))
            csr_dict["row_pointers"].append(len(csr_dict["column_indices"]))
        return self.dict_to_object(csr_dict)

    def get_incumbent_values(self, solution, vars):
        """
        Extract incumbent values of the vars from a problem solution.
        """
        values = []
        for var in vars:
            values.append(solution[var.index])
        return values

    def post_solve(self, solution):
        self.Status = solution.get_termination_status()
        self.SolveTime = solution.get_solve_time()

        IsMIP = False
        if solution.problem_category == 0:
            self.SolutionStats = self.dict_to_object(solution.get_lp_stats())
        else:
            IsMIP = True
            self.SolutionStats = self.dict_to_object(solution.get_milp_stats())

        primal_sol = solution.get_primal_solution()
        reduced_cost = solution.get_reduced_cost()
        if len(primal_sol) > 0:
            for var in self.vars:
                var.Value = primal_sol[var.index]
                if not IsMIP:
                    var.ReducedCost = reduced_cost[var.index]
        dual_sol = None
        if not IsMIP:
            dual_sol = solution.get_dual_solution()
        for i, constr in enumerate(self.constrs):
            if dual_sol is not None:
                constr.DualValue = dual_sol[i]
            constr.Slack = constr.compute_slack()
        self.ObjValue = self.Obj.getValue()
        self.solved = True

    def solve(self, settings=solver_settings.SolverSettings()):
        """
        Optimizes the LP or MIP problem with the added variables,
        constraints and objective.

        Examples
        --------
        >>> problem = problem.Problem("MIP_model")
        >>> x = problem.addVariable(lb=-2.0, ub=8.0, vtype=INTEGER)
        >>> y = problem.addVariable(name="Var2")
        >>> problem.addConstraint(2*x - 3*y <= 10, name="Constr1")
        >>> expr = 3*x + y
        >>> problem.addConstraint(expr + x == 20, name="Constr2")
        >>> problem.setObjective(x + y, sense=MAXIMIZE)
        >>> problem.solve()
        """

        # iterate through the constraints and construct the constraint matrix
        n = len(self.vars)
        self.row_pointers = [0]
        self.column_indicies = []
        self.values = []
        self.rhs = []
        self.row_sense = []
        for constr in self.constrs:
            self.column_indicies.extend(list(constr.vindex_coeff_dict.keys()))
            self.values.extend(list(constr.vindex_coeff_dict.values()))
            self.row_pointers.append(len(self.column_indicies))
            self.rhs.append(constr.RHS)
            self.row_sense.append(constr.Sense)

        self.objective = np.zeros(n)
        self.lower_bound, self.upper_bound = np.zeros(n), np.zeros(n)
        self.var_type = np.empty(n, dtype="S1")

        for j in range(n):
            self.objective[j] = self.vars[j].getObjectiveCoefficient()
            self.var_type[j] = self.vars[j].getVariableType()
            self.lower_bound[j] = self.vars[j].getLowerBound()
            self.upper_bound[j] = self.vars[j].getUpperBound()

        # Initialize datamodel
        dm = data_model.DataModel()
        dm.set_csr_constraint_matrix(
            np.array(self.values),
            np.array(self.column_indicies),
            np.array(self.row_pointers),
        )
        if self.ObjSense == -1:
            dm.set_maximize(True)
        dm.set_constraint_bounds(np.array(self.rhs))
        dm.set_row_types(np.array(self.row_sense, dtype="S1"))
        dm.set_objective_coefficients(self.objective)
        dm.set_variable_lower_bounds(self.lower_bound)
        dm.set_variable_upper_bounds(self.upper_bound)
        dm.set_variable_types(self.var_type)

        # Call Solver
        solution = solver.Solve(dm, settings)

        # Post Solve
        self.post_solve(solution)
