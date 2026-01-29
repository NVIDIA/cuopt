# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


import copy
from enum import Enum

import cuopt_mps_parser
import numpy as np
from scipy.sparse import coo_matrix

import cuopt.linear_programming.data_model as data_model
import cuopt.linear_programming.solver as solver
import cuopt.linear_programming.solver_settings as solver_settings
import warnings


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
    EQ is CType.EQ
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
    created using :py:meth:`Problem.addVariable`.

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
            case Variable():
                return QuadraticExpression(
                    qvars1=[self], qvars2=[other], qcoefficients=[1.0]
                )
            case LinearExpression():
                qvars1 = [self] * len(other.vars)
                qvars2 = other.vars
                qcoeffs = other.coefficients
                vars = [self]
                coeffs = [other.constant]
                return QuadraticExpression(
                    qvars1=qvars1,
                    qvars2=qvars2,
                    qcoefficients=qcoeffs,
                    vars=vars,
                    coefficients=coeffs,
                )
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


class QuadraticExpression:
    """
    QuadraticExpressions contain quadratic terms, linear terms, and a constant.
    QuadraticExpressions can be used to create quadratic objectives in
    the Problem.
    QuadraticExpressions can be added and subtracted with other
    QuadraticExpressions, LinearExpressions, and Variables, and can also
    be multiplied and divided by scalars.

    Parameters
    ----------
    qmatrix : List[List[float]] or 2D numpy array.
        Matrix containing quadratic coefficient matrix terms.
        Should be a square matrix with shape as (num_vars, num_vars).
    qvars : List[Variable]
        List of variables denoting the rows and cols in qmatrix. It is
        a mandatory field when providing qmatrix.
        qvars should be in the order of variables added to the problem
        and can be obtained using problem.getVariables(). The length
        of qvars should be equal to length of row/col in qmatrix.
    qvars1 : List[Variable]
        List of first variables for quadratic terms.
        This should be used if adding quadratic terms in triplet (i, j, x)
        format where i is the row variable, j is the column variable and
        x is the corresponding coefficient. qvars1 contains all i variables
        representing the row.
    qvars2 : List[Variable]
        List of second variables for quadratic terms.
        This should be used if adding quadratic terms in triplet (i, j, x)
        format where i is the row variable, j is the column variable and
        x is the corresponding coefficient. qvars2 contains all j variables
        representing the column.
    qcoefficients : List[float]
        List of coefficients for the quadratic terms.
        This should be used if adding quadratic terms in triplet (i, j, x)
        format where i is the row variable, j is the column variable and
        x is the corresponding coefficient. qcoefficients contains all x
        values representing coefficients for (i,j)
    vars : List[Variable]
        List of Variables for linear terms.
    coefficients : List[float]
        List of coefficients for linear terms.
    constant : float
        Constant of the quadratic expression.

    Examples
    --------
    >>> x = problem.addVariable()
    >>> y = problem.addVariable()
    >>> # Create objective x^2 + 2*x*y + 3*x + 4 using matrix
    >>> quad_matrix = QuadraticExpression(
    ...     qmatrix=[[1.0, 2.0], [0.0, 0.0]],
    ...     qvars=[x, y]
    ... )
    >>> quad_obj_using_matrix = quad_matrix + 3*x + 4
    >>> # Create objective x^2 + 2*x*y + 3*x + 4 using expression
    >>> quad_obj_using_expr = x*x + 2*x*y + 3*x + 4
    """

    def __init__(
        self,
        qmatrix=None,
        qvars=[],
        qvars1=[],
        qvars2=[],
        qcoefficients=[],
        vars=[],
        coefficients=[],
        constant=0.0,
    ):
        self.qmatrix = None
        self.qvars = qvars
        if qmatrix is not None:
            self.qmatrix = coo_matrix(qmatrix)
            mshape = self.qmatrix.shape
            if mshape[0] != mshape[1]:
                raise ValueError("qmatrix should be a square matrix")
            if len(qvars) != mshape[0]:
                raise ValueError(
                    "qvars length mismatch. Should match with qmatrix length. Please check docs for more details."
                )
        self.qvars1 = qvars1
        self.qvars2 = qvars2
        self.qcoefficients = qcoefficients
        self.vars = vars
        self.coefficients = coefficients
        self.constant = constant

    def getVariables(self):
        """
        Returns all the quadractic variables in the expression as list
        of tuples containing Variable 1 and Variable 2 for each term.
        """
        qvars1 = self.qvars1
        qvars2 = self.qvars2
        if self.qmatrix is not None:
            qvars1 = self.qvars1 + [self.qvars[i] for i in self.qmatrix.row]
            qvars2 = self.qvars2 + [self.qvars[i] for i in self.qmatrix.col]
        return list(zip(qvars1, qvars2))

    def getVariable1(self, i):
        """
        Gets first Variable at ith index in the quadratic term.
        """
        qvars1 = self.qvars1
        if self.qmatrix is not None:
            qvars1 = self.qvars1 + [self.qvars[r] for r in self.qmatrix.row]
        return qvars1[i]

    def getVariable2(self, i):
        """
        Gets second Variable at ith index in the quadratic term.
        """
        qvars2 = self.qvars2
        if self.qmatrix is not None:
            qvars2 = self.qvars2 + [self.qvars[c] for c in self.qmatrix.col]
        return qvars2[i]

    def getCoefficients(self):
        """
        Returns all the coefficients of the quadratic term.
        """
        qcoefficients = self.qcoefficients
        if self.qmatrix is not None:
            qcoefficients = self.qcoefficients + self.qmatrix.data.tolist()
        return qcoefficients

    def getCoefficient(self, i):
        """
        Gets the coefficient of the quadratic term at ith index.
        """
        qcoefficients = self.qcoefficients
        if self.qmatrix is not None:
            qcoefficients = self.qcoefficients + self.qmatrix.data.tolist()
        return qcoefficients[i]

    def getLinearExpression(self):
        """
        Returns the Linear Expression associated with the
        Quadratic Expression.
        """
        return LinearExpression(self.vars, self.coefficients, self.constant)

    def getValue(self):
        """
        Returns the value of the expression computed with the
        current solution.
        """
        value = 0.0
        for i, var in enumerate(self.vars):
            value += var.Value * self.coefficients[i]
        for i, var1 in enumerate(self.qvars1):
            var2 = self.qvars2[i]
            value += var1.Value * var2.Value * self.qcoefficients[i]
        if self.qmatrix is not None:
            rows_idx = self.qmatrix.row
            cols_idx = self.qmatrix.col
            data = self.qmatrix.data
            for i, row_idx in enumerate(rows_idx):
                var1 = self.qvars[row_idx]
                var2 = self.qvars[cols_idx[i]]
                value += var1.Value * var2.Value * data[i]
        return value + self.constant

    def __len__(self):
        return len(self.vars) + len(self.qvars1) + len(self.qvars)

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
                # Append all linear variables, coefficients and constants
                self.vars.extend(other.vars)
                self.coefficients.extend(other.coefficients)
                self.constant += other.constant
                return self
            case QuadraticExpression():
                # Linear expression terms
                self.vars.extend(other.vars)
                self.coefficients.extend(other.coefficients)
                self.constant += other.constant
                # Quadratic expression terms
                self.qvars2.extend(other.qvars2)
                self.qvars1.extend(other.qvars1)
                self.qcoefficients.extend(other.qcoefficients)
                # Quadratic matrix terms
                if self.qmatrix is not None and other.qmatrix is not None:
                    self.qmatrix += other.qmatrix
                elif other.qmatrix is not None:
                    self.qmatrix = other.qmatrix
                    self.qvars = other.qvars
                return self
            case _:
                raise ValueError(
                    "Can't add type %s to Quadratic Expression"
                    % type(other).__name__
                )

    def __add__(self, other):
        # Compute expr3 = expr1 + expr2
        match other:
            case int() | float():
                # Update just the constant value
                return QuadraticExpression(
                    self.qmatrix,
                    self.qvars,
                    self.qvars1,
                    self.qvars2,
                    self.qcoefficients,
                    self.vars,
                    self.coefficients,
                    self.constant + float(other),
                )
            case Variable():
                # Append just a variable with coefficient 1.0
                vars = self.vars + [other]
                coeffs = self.coefficients + [1.0]
                return QuadraticExpression(
                    self.qmatrix,
                    self.qvars,
                    self.qvars1,
                    self.qvars2,
                    self.qcoefficients,
                    vars,
                    coeffs,
                    self.constant,
                )
            case LinearExpression():
                # Append all linear variables, coefficients and constants
                vars = self.vars + other.vars
                coeffs = self.coefficients + other.coefficients
                constant = self.constant + other.constant
                return QuadraticExpression(
                    self.qmatrix,
                    self.qvars,
                    self.qvars1,
                    self.qvars2,
                    self.qcoefficients,
                    vars,
                    coeffs,
                    constant,
                )
            case QuadraticExpression():
                # Linear expression terms
                vars = self.vars + other.vars
                coeffs = self.coefficients + other.coefficients
                constant = self.constant + other.constant
                # Quadratic expression terms
                qvars1 = self.qvars1 + other.qvars1
                qvars2 = self.qvars2 + other.qvars2
                qcoeffs = self.qcoefficients + other.qcoefficients
                # Quadratic matrix terms
                qmatrix = self.qmatrix
                qvars = self.qvars
                if self.qmatrix is not None and other.qmatrix is not None:
                    qmatrix = self.qmatrix + other.qmatrix
                elif other.qmatrix is not None:
                    qmatrix = other.qmatrix
                    qvars = other.qvars
                return QuadraticExpression(
                    qmatrix,
                    qvars,
                    qvars1,
                    qvars2,
                    qcoeffs,
                    vars,
                    coeffs,
                    constant,
                )
            case _:
                raise ValueError(
                    "Can't add type %s to Quadratic Expression"
                    % type(other).__name__
                )

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
                # Append all linear variables, coefficients and constants
                self.vars.extend(other.vars)
                for coeff in other.coefficients:
                    self.coefficients.append(-coeff)
                self.constant -= other.constant
                return self
            case QuadraticExpression():
                # Linear expression terms
                self.vars.extend(other.vars)
                for coeff in other.coefficients:
                    self.coefficients.append(-coeff)
                self.constant -= other.constant
                # Quadratic expression terms
                self.qvars2.extend(other.qvars2)
                self.qvars1.extend(other.qvars1)
                for qcoeff in other.qcoefficients:
                    self.qcoefficients.append(-qcoeff)
                # Quadratic matrix terms
                if self.qmatrix is not None and other.qmatrix is not None:
                    self.qmatrix -= other.qmatrix
                elif other.qmatrix is not None:
                    self.qmatrix = -other.qmatrix
                    self.qvars = other.qvars
                return self
            case _:
                raise ValueError(
                    "Can't sub type %s from Quadratic Expression"
                    % type(other).__name__
                )

    def __sub__(self, other):
        # Compute expr3 = expr1 - expr2
        match other:
            case int() | float():
                # Update just the constant value
                return QuadraticExpression(
                    self.qmatrix,
                    self.qvars,
                    self.qvars1,
                    self.qvars2,
                    self.qcoefficients,
                    self.vars,
                    self.coefficients,
                    self.constant - float(other),
                )
            case Variable():
                # Append just a variable with coefficient -1.0
                vars = self.vars + [other]
                coeffs = self.coefficients + [-1.0]
                return QuadraticExpression(
                    self.qmatrix,
                    self.qvars,
                    self.qvars1,
                    self.qvars2,
                    self.qcoefficients,
                    vars,
                    coeffs,
                    self.constant,
                )
            case LinearExpression():
                # Append all linear variables, coefficients and constants
                vars = self.vars + other.vars
                coeffs = []
                for i in self.coefficients:
                    coeffs.append(i)
                for i in other.coefficients:
                    coeffs.append(-1.0 * i)
                constant = self.constant - other.constant
                return QuadraticExpression(
                    self.qmatrix,
                    self.qvars,
                    self.qvars1,
                    self.qvars2,
                    self.qcoefficients,
                    vars,
                    coeffs,
                    constant,
                )
            case QuadraticExpression():
                # Linear expression terms
                vars = self.vars + other.vars
                coeffs = []
                for i in self.coefficients:
                    coeffs.append(i)
                for i in other.coefficients:
                    coeffs.append(-1.0 * i)
                constant = self.constant - other.constant
                # Quadratic expression terms
                qvars1 = self.qvars1 + other.qvars1
                qvars2 = self.qvars2 + other.qvars2
                qcoeffs = []
                for i in self.qcoefficients:
                    qcoeffs.append(i)
                for i in other.qcoefficients:
                    qcoeffs.append(-1.0 * i)
                # Quadratic matrix terms
                qmatrix = self.qmatrix
                qvars = self.qvars
                if self.qmatrix is not None and other.qmatrix is not None:
                    qmatrix = self.qmatrix - other.qmatrix
                elif other.qmatrix is not None:
                    qmatrix = -other.qmatrix
                    qvars = other.qvars
                return QuadraticExpression(
                    qmatrix,
                    qvars,
                    qvars1,
                    qvars2,
                    qcoeffs,
                    vars,
                    coeffs,
                    constant,
                )
            case _:
                raise ValueError(
                    "Can't sub type %s from Quadratic Expression"
                    % type(other).__name__
                )

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
                self.qcoefficients = [
                    qcoeff * float(other) for qcoeff in self.qcoefficients
                ]
                if self.qmatrix is not None:
                    self.qmatrix *= float(other)
                return self
            case _:
                raise ValueError(
                    "Can't multiply type %s by QuadraticExpresson"
                    % type(other).__name__
                )

    def __mul__(self, other):
        # Compute expr2 = expr1 * constant
        match other:
            case int() | float():
                coeffs = [coeff * float(other) for coeff in self.coefficients]
                qcoeffs = [
                    qcoeff * float(other) for qcoeff in self.qcoefficients
                ]
                constant = self.constant * float(other)
                qmatrix = None
                if self.qmatrix is not None:
                    qmatrix = self.qmatrix * float(other)
                return QuadraticExpression(
                    qmatrix,
                    self.qvars,
                    self.qvars1,
                    self.qvars2,
                    qcoeffs,
                    self.vars,
                    coeffs,
                    constant,
                )
            case _:
                raise ValueError(
                    "Can't multiply type %s by QuadraticExpresson"
                    % type(other).__name__
                )

    def __rmul__(self, other):
        return self * other

    def __itruediv__(self, other):
        # Compute expr /= constant
        match other:
            case int() | float():
                self.coefficients = [
                    qcoeff / float(other) for qcoeff in self.coefficients
                ]
                self.qcoefficients = [
                    coeff / float(other) for coeff in self.qcoefficients
                ]
                self.constant = self.constant / float(other)
                if self.qmatrix is not None:
                    self.qmatrix = self.qmatrix / float(other)
                return self
            case _:
                raise ValueError(
                    "Can't divide QuadraticExpression by type %s"
                    % type(other).__name__
                )

    def __truediv__(self, other):
        # Compute expr2 = expr1 / constant
        match other:
            case int() | float():
                coeffs = [coeff / float(other) for coeff in self.coefficients]
                qcoeffs = [
                    qcoeff / float(other) for qcoeff in self.qcoefficients
                ]
                constant = self.constant / float(other)
                qmatrix = None
                if self.qmatrix is not None:
                    qmatrix = self.qmatrix / float(other)
                return QuadraticExpression(
                    qmatrix,
                    self.qvars,
                    self.qvars1,
                    self.qvars2,
                    qcoeffs,
                    self.vars,
                    coeffs,
                    constant,
                )
            case _:
                raise ValueError(
                    "Can't divide LinearExpression by type %s"
                    % type(other).__name__
                )

    def __le__(self, other):
        raise Exception("Quadratic constraints not supported")

    def __ge__(self, other):
        raise Exception("Quadratic constraints not supported")

    def __eq__(self, other):
        raise Exception("Quadratic constraints not supported")


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
            case QuadraticExpression():
                return other + self
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
            case QuadraticExpression():
                return other + self
            case _:
                raise ValueError(
                    "Can't add type %s to Linear Expression"
                    % type(other).__name__
                )

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
            case QuadraticExpression():
                return other * -1 + self
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
            case QuadraticExpression():
                return other * -1 + self
            case _:
                raise ValueError(
                    "Can't sub type %s from LinearExpression"
                    % type(other).__name__
                )

    def __rsub__(self, other):
        # other - self  -> other + self * -1.0
        return self * -1.0 + other

    def __imul__(self, other):
        # Compute expr *= other
        match other:
            case int() | float():
                self.coefficients = [
                    coeff * float(other) for coeff in self.coefficients
                ]
                self.constant = self.constant * float(other)
                return self
            case Variable():
                return other * self
            case LinearExpression():
                qvars1, qvars2, qcoeffs = [], [], []
                for i in range(len(self.vars)):
                    for j in range(len(other.vars)):
                        qvars1.append(self.vars[i])
                        qvars2.append(other.vars[j])
                        qcoeffs.append(
                            self.coefficients[i] * other.coefficients[j]
                        )
                vars = self.vars + other.vars
                coeffs = [other.constant * i for i in self.coefficients] + [
                    self.constant * i for i in other.coefficients
                ]
                constant = self.constant * other.constant
                return QuadraticExpression(
                    qvars1=qvars1,
                    qvars2=qvars2,
                    qcoefficients=qcoeffs,
                    vars=vars,
                    coefficients=coeffs,
                    constant=constant,
                )
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
            case Variable():
                return other * self
            case LinearExpression():
                qvars1, qvars2, qcoeffs = [], [], []
                for i in range(len(self.vars)):
                    for j in range(len(other.vars)):
                        qvars1.append(self.vars[i])
                        qvars2.append(other.vars[j])
                        qcoeffs.append(
                            self.coefficients[i] * other.coefficients[j]
                        )
                vars = self.vars + other.vars
                coeffs = [other.constant * i for i in self.coefficients] + [
                    self.constant * i for i in other.coefficients
                ]
                constant = self.constant * other.constant
                return QuadraticExpression(
                    qvars1=qvars1,
                    qvars2=qvars2,
                    qcoefficients=qcoeffs,
                    vars=vars,
                    coefficients=coeffs,
                    constant=constant,
                )
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
    created using :py:meth:`Problem.addConstraint`.

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
        self.ObjConstant = 0.0
        self.Status = -1
        self.warmstart_data = None

        self.model = None
        self.solved = False
        self.rhs = None
        self.row_sense = None
        self.constraint_csr_matrix = None
        self.objective_qmatrix = None
        self.lower_bound = None
        self.upper_bound = None
        self.var_type = None

    class dict_to_object:
        def __init__(self, mdict):
            for key, value in mdict.items():
                setattr(self, key, value)

    def _from_data_model(self, dm):
        self.Name = dm.get_problem_name()
        obj_coeffs = dm.get_objective_coefficients()
        obj_constant = dm.get_objective_offset()
        num_vars = len(obj_coeffs)
        sense = dm.get_sense()
        if sense:
            sense = MAXIMIZE
        else:
            sense = MINIMIZE
        v_lb = dm.get_variable_lower_bounds()
        v_ub = dm.get_variable_upper_bounds()
        v_types = dm.get_variable_types()
        v_names = dm.get_variable_names().tolist()

        # Add all Variables and Objective Coefficients
        for i in range(num_vars):
            v_name = ""
            if v_names:
                v_name = v_names[i]
            self.addVariable(v_lb[i], v_ub[i], vtype=v_types[i], name=v_name)
        vars = self.getVariables()
        expr = LinearExpression(vars, obj_coeffs, obj_constant)
        self.setObjective(expr, sense)

        # Add all Constraints
        c_lb = dm.get_constraint_lower_bounds()
        c_ub = dm.get_constraint_upper_bounds()
        c_b = dm.get_constraint_bounds()
        c_names = dm.get_row_names()
        offsets = dm.get_constraint_matrix_offsets()
        indices = dm.get_constraint_matrix_indices()
        values = dm.get_constraint_matrix_values()

        num_constrs = len(offsets) - 1
        for i in range(num_constrs):
            start = offsets[i]
            end = offsets[i + 1]
            c_coeffs = values[start:end]
            c_indices = indices[start:end]
            c_vars = [vars[j] for j in c_indices]
            expr = LinearExpression(c_vars, c_coeffs, 0.0)
            if c_lb[i] == c_ub[i]:
                self.addConstraint(expr == c_b[i], name=c_names[i])
            elif c_lb[i] == c_b[i]:
                self.addConstraint(expr >= c_b[i], name=c_names[i])
            elif c_ub[i] == c_b[i]:
                self.addConstraint(expr <= c_b[i], name=c_names[i])
            else:
                raise Exception("Couldn't initialize constraints")

    def _to_data_model(self):
        dm = data_model.DataModel()

        # iterate through the constraints and construct the constraint matrix
        n = len(self.vars)
        self.rhs = []
        self.row_sense = []
        self.row_names = []

        if self.constraint_csr_matrix is None:
            csr_dict = {
                "row_pointers": [0],
                "column_indices": [],
                "values": [],
            }
            for constr in self.constrs:
                csr_dict["column_indices"].extend(
                    list(constr.vindex_coeff_dict.keys())
                )
                csr_dict["values"].extend(
                    list(constr.vindex_coeff_dict.values())
                )
                csr_dict["row_pointers"].append(
                    len(csr_dict["column_indices"])
                )
                self.rhs.append(constr.RHS)
                self.row_sense.append(constr.Sense)
                constr_name = constr.ConstraintName
                if constr_name == "":
                    constr_name = "R" + str(constr.index)
                self.row_names.append(constr_name)
            self.constraint_csr_matrix = csr_dict

        else:
            for constr in self.constrs:
                self.rhs.append(constr.RHS)
                self.row_sense.append(constr.Sense)

        self.objective = np.zeros(n)
        self.lower_bound, self.upper_bound = np.zeros(n), np.zeros(n)
        self.var_type = np.empty(n, dtype="S1")
        self.var_names = []

        for j in range(n):
            self.objective[j] = self.vars[j].getObjectiveCoefficient()
            self.var_type[j] = self.vars[j].getVariableType()
            self.lower_bound[j] = self.vars[j].getLowerBound()
            self.upper_bound[j] = self.vars[j].getUpperBound()
            var_name = self.vars[j].VariableName
            if var_name == "":
                var_name = "C" + str(self.vars[j].index)
            self.var_names.append(var_name)

        # Initialize datamodel
        dm.set_csr_constraint_matrix(
            np.array(self.constraint_csr_matrix["values"]),
            np.array(self.constraint_csr_matrix["column_indices"]),
            np.array(self.constraint_csr_matrix["row_pointers"]),
        )
        if self.ObjSense == -1:
            dm.set_maximize(True)
        dm.set_constraint_bounds(np.array(self.rhs))
        dm.set_row_types(np.array(self.row_sense, dtype="S1"))
        dm.set_objective_coefficients(self.objective)
        dm.set_objective_offset(self.ObjConstant)
        if self.objective_qmatrix is not None:
            dm.set_quadratic_objective_matrix(
                self.objective_qmatrix.data,
                self.objective_qmatrix.indices,
                self.objective_qmatrix.indptr,
            )
        dm.set_variable_lower_bounds(self.lower_bound)
        dm.set_variable_upper_bounds(self.upper_bound)
        dm.set_variable_types(self.var_type)
        dm.set_variable_names(self.var_names)
        dm.set_row_names(self.row_names)
        dm.set_problem_name(self.Name)

        self.model = dm

    def update(self):
        """
        Update the problem. This is mandatory if attributes of
        existing Variables, Constraints or Objective has been
        modified.
        """
        self.reset_solved_values()

    def reset_solved_values(self):
        # Resets all post solve values
        for var in self.vars:
            var.Value = float("nan")
            var.ReducedCost = float("nan")

        for constr in self.constrs:
            constr.Slack = float("nan")
            constr.DualValue = float("nan")

        self.model = None
        self.constraint_csr_matrix = None
        self.objective_qmatrix = None
        self.warmstart_data = None
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
        vtype : enum :py:class:`VType`
            vtype.CONTINUOUS or vtype.INTEGER. Defaults to CONTINUOUS.
        name : string
            Name of the variable. Optional.

        Returns
        -------
        variable : :py:class:`Variable`
            Variable object added to the problem.

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
        constr : :py:class:`Constraint`
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
        return constr

    def updateConstraint(self, constr, coeffs=[], rhs=None):
        """
        Updates a previously added constraint. Values that can be updated are
        constraint coefficients and RHS.

        Parameters
        ----------
        constr : :py:class:`Constraint`
            Constraint to be updated.
        coeffs : List[Tuple[:py:class:`Variable`, coefficient]]
            List of Tuples containing variable and corresponding coefficient.
            Optional.
        rhs : int|float
            New RHS value for the constraint.

        Examples
        --------
        >>> problem = problem.Problem("MIP_model")
        >>> x = problem.addVariable(lb=0.0, vtype=INTEGER)
        >>> y = problem.addVariable(lb=0.0, vtype=INTEGER)
        >>> c1 = problem.addConstraint(2 * x + y <= 7, name="c1")
        >>> c2 = problem.addConstraint(x + y <= 5, name="c2")
        >>> problem.updateConstraint(c1, coeffs=[(x, 1)], rhs=10)
        """
        self.reset_solved_values()
        if isinstance(constr, Constraint):
            if isinstance(coeffs, dict):
                coeffs = coeffs.items()
            for var, coeff in coeffs:
                idx = var.index
                constr.vindex_coeff_dict[idx] = coeff
            if rhs:
                constr.RHS = rhs
        else:
            raise ValueError("Object to update must be a Constraint")

    def setObjective(self, expr, sense=MINIMIZE):
        """
        Set the Objective of the problem with an expression that needs to
        be MINIMIZED or MAXIMIZED.

        Parameters
        ----------
        expr : :py:class:`LinearExpression` or :py:class:`Variable` or Constant
            Objective expression that needs maximization or minimization.
        sense : enum :py:class:`sense`
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
                self.ObjConstant = float(expr)
            case Variable():
                for var in self.vars:
                    var.setObjectiveCoefficient(0.0)
                    if var.getIndex() == expr.getIndex():
                        var.setObjectiveCoefficient(1.0)
            case LinearExpression():
                for var in self.vars:
                    var.setObjectiveCoefficient(0.0)
                for var, coeff in expr.zipVarCoefficients():
                    c_val = self.vars[var.getIndex()].getObjectiveCoefficient()
                    sum_coeff = coeff + c_val
                    self.vars[var.getIndex()].setObjectiveCoefficient(
                        sum_coeff
                    )
                self.ObjConstant = expr.getConstant()
            case QuadraticExpression():
                for var in self.vars:
                    var.setObjectiveCoefficient(0.0)
                for var, coeff in zip(expr.vars, expr.coefficients):
                    c_val = self.vars[var.getIndex()].getObjectiveCoefficient()
                    sum_coeff = coeff + c_val
                    self.vars[var.getIndex()].setObjectiveCoefficient(
                        sum_coeff
                    )
                self.ObjConstant = expr.constant
                qrows = [var.getIndex() for var in expr.qvars1]
                qcols = [var.getIndex() for var in expr.qvars2]
                self.objective_qmatrix = coo_matrix(
                    (
                        np.array(expr.qcoefficients),
                        (np.array(qrows), np.array(qcols)),
                    ),
                    shape=(self.NumVariables, self.NumVariables),
                )
                if expr.qmatrix is not None:
                    self.objective_qmatrix += expr.qmatrix
                self.objective_qmatrix = self.objective_qmatrix.tocsr()
            case _:
                raise ValueError(
                    "Objective must be a Variable, Expression or a constant"
                )

    def updateObjective(self, coeffs=[], constant=None, sense=None):
        """
        Updates the objective of the problem. Values that can be updated are
        objective coefficients, constant and sense.

        Parameters
        ----------
        coeffs : List[Tuple[:py:class:`Variable`, coefficient]]
            List of Tuples containing variable and corresponding coefficient.
            Optional.
        constant : int|float
            New Objective constant for the problem. Optional.
        sense : enum :py:class:`sense`
            Sets the objective sense to either maximize or minimize. Optional.

        Examples
        --------
        >>> problem = problem.Problem("MIP_model")
        >>> x = problem.addVariable(lb=0.0, vtype=INTEGER)
        >>> y = problem.addVariable(lb=0.0, vtype=INTEGER)
        >>> problem.setObjective(4*x + y + 4, MAXIMIZE)
        >>> problem.updateObjective(coeffs=[(x1, 1.0), (x2, 3.0)], constant=5,
                sense=MINIMIZE)
        """
        self.reset_solved_values()
        if isinstance(coeffs, dict):
            coeffs = coeffs.items()
        for var, coeff in coeffs:
            var.setObjectiveCoefficient(coeff)
        if constant:
            self.ObjConstant = constant
        if sense:
            self.ObjSense = sense

    def getIncumbentValues(self, solution, vars):
        """
        Extract incumbent values of the vars from a problem solution.
        """
        values = []
        for var in vars:
            values.append(solution[var.index])
        return values

    def get_incumbent_values(self, solution, vars):
        warnings.warn(
            "This function is deprecated and will be removed."
            "Please use getIncumbentValues instead.",
            DeprecationWarning,
        )
        return self.getIncumbentValues(solution, vars)

    def getWarmstartData(self):
        """
        Note: Applicable to only LP.
        Allows to retrieve the warm start data from the PDLP solver
        once the problem is solved.
        This data can be used to warmstart the next PDLP solve by setting it
        in :py:meth:`cuopt.linear_programming.solver_settings.SolverSettings.set_pdlp_warm_start_data`  # noqa

        Examples
        --------
        >>> problem = problem.Problem.readMPS("LP.mps")
        >>> problem.solve()
        >>> warmstart_data = problem.getWarmstartData()
        >>> settings.set_pdlp_warm_start_data(warmstart_data)
        >>> updated_problem = problem.Problem.readMPS("updated_LP.mps")
        >>> updated_problem.solve(settings)
        """
        return self.warmstart_data

    def get_pdlp_warm_start_data(self):
        warnings.warn(
            "This function is deprecated and will be removed."
            "Please use getWarmstartData instead.",
            DeprecationWarning,
        )
        return self.getWarmstartData()

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

    def getVariable(self, identifier):
        """
        Get a Variable by its index or name.
        """
        for v in self.vars:
            if v.index == identifier or v.VariableName == identifier:
                return v

    def getConstraints(self):
        """
        Get a list of all the Constraints in a problem.
        """
        return self.constrs

    def getConstraint(self, identifier):
        """
        Get a Constraint by its index or name.
        """
        for c in self.constrs:
            if c.index == identifier or c.ConstraintName == identifier:
                return c

    @classmethod
    def readMPS(cls, mps_file):
        """
        Initiliaze a problem from an `MPS <https://en.wikipedia.org/wiki/MPS_(format)>`__ file.  # noqa
        Examples
        --------
        >>> problem = problem.Problem.readMPS("model.mps")
        """
        problem = cls()
        data_model = cuopt_mps_parser.ParseMps(mps_file)
        problem._from_data_model(data_model)
        problem.model = data_model
        return problem

    def writeMPS(self, mps_file):
        """
        Write the problem into an `MPS <https://en.wikipedia.org/wiki/MPS_(format)>`__ file.  # noqa
        Examples
        --------
        >>> problem.writeMPS("model.mps")
        """
        if self.model is None:
            self._to_data_model()
        self.model.writeMPS(mps_file)

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

    @property
    def Obj(self):
        # Returns objective expression of the problem.
        coeffs = []
        for var in self.vars:
            coeffs.append(var.getObjectiveCoefficient())
        if self.objective_qmatrix is None:
            return LinearExpression(self.vars, coeffs, self.ObjConstant)
        else:
            return QuadraticExpression(
                qmatrix=self.objective_qmatrix,
                qvars=self.vars,
                vars=self.vars,
                coefficients=coeffs,
                constant=self.ObjConstant,
            )

    @property
    def ObjValue(self):
        return self.Obj.getValue()

    def getCSR(self):
        """
        Computes and returns the CSR representation of the
        constraint matrix.
        """
        if self.constraint_csr_matrix is not None:
            return self.dict_to_object(self.constraint_csr_matrix)
        csr_dict = {"row_pointers": [0], "column_indices": [], "values": []}
        for constr in self.constrs:
            csr_dict["column_indices"].extend(
                list(constr.vindex_coeff_dict.keys())
            )
            csr_dict["values"].extend(list(constr.vindex_coeff_dict.values()))
            csr_dict["row_pointers"].append(len(csr_dict["column_indices"]))
        self.constraint_csr_matrix = csr_dict
        return self.dict_to_object(csr_dict)

    def getQCSR(self):
        """
        Computes and returns the CSR matrix representation of the
        quadratic objective.
        """
        if self.objective_qmatrix is None:
            return None
        qcsr_matrix = {
            "row_pointers": self.objective_qmatrix.indptr,
            "column_indices": self.objective_qmatrix.indices,
            "values": self.objective_qmatrix.data,
        }
        return self.dict_to_object(qcsr_matrix)

    def getQcsr(self):
        warnings.warn(
            "This function is deprecated and will be removed."
            "Please use getQCSR instead.",
            DeprecationWarning,
        )
        return self.getQCSR()

    def relax(self):
        """
        Relax a MIP problem into an LP problem and return the relaxed model.
        The relaxed model has all variable types set to CONTINUOUS.

        Examples
        --------
        >>> mip_problem = problem.Problem.readMPS("MIP.mps")
        >>> lp_problem = problem.relax()
        """
        self.reset_solved_values()
        relaxed_problem = copy.deepcopy(self)
        vars = relaxed_problem.getVariables()
        for v in vars:
            v.VariableType = CONTINUOUS
        return relaxed_problem

    def populate_solution(self, solution):
        self.Status = solution.get_termination_status()
        self.SolveTime = solution.get_solve_time()
        self.warmstart_data = solution.get_pdlp_warm_start_data()

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
                if (
                    not IsMIP
                    and reduced_cost is not None
                    and len(reduced_cost) > 0
                ):
                    var.ReducedCost = reduced_cost[var.index]
        dual_sol = None
        if not IsMIP:
            dual_sol = solution.get_dual_solution()
        for i, constr in enumerate(self.constrs):
            if dual_sol is not None and len(dual_sol) > 0:
                constr.DualValue = dual_sol[i]
            constr.Slack = constr.compute_slack()
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
        if self.model is None:
            self._to_data_model()
        # Call Solver
        solution = solver.Solve(self.model, settings)
        # Post Solve
        self.populate_solution(solution)
