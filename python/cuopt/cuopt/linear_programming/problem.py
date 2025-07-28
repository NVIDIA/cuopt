import cuopt.linear_programming.data_model as data_model
import cuopt.linear_programming.solver as solver
import cuopt.linear_programming.solver_settings as solver_settings

import enum
import numpy as np

# The type of a variable is either continuous, binary, or integer
CONTINUOUS = 'C'
INTEGER = 'I'

# Variable objects hold a reference to the problem they were created from as well as info
# about there index, lower bound, upper bound, objective coefficient and variable type.
# You can add and subtract Variables to scalars and other Variables to form LinearExpression objects.
# You can multiply variables by a scalar to form a LinearExpression object
class Variable:
    def __init__(self, problem, index, lb=0.0, ub=float('inf'), obj=0.0, vtype=CONTINUOUS, vname=''):
        self.problem = problem
        self.index = index
        self.LB = lb
        self.UB = ub
        self.Obj = obj
        self.Value = 0.0
        self.ReducedCost = float('nan')
        self.VariableType = vtype
        self.VariableName = vname
    def getIndex(self):
        return self.index
    def getValue(self):
        return self.Value
    def getObjectiveCoefficient(self):
        return self.Obj
    def setObjectiveCoefficient(self, val):
        self.Obj = val
    def setLowerBound(self, val):
        self.LB = val
    def getLowerBound(self):
        return self.LB
    def setUpperBound(self, val):
        self.UB = val
    def getUpperBound(self):
        return self.UB
    def setVariableType(self, val):
        self.VariableType = val
    def getVariableType(self):
        return self.VariableType
    def setVariableName(self, val):
        self.VariableName = val
    def getVariableName(self):
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
                raise ValueError('Cannot add type %s to variable' % type(other).__name__)
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
                raise ValueError('Cannot subtract type %s from variable' % type(other).__name__)
    def __rsub__(self, other):
        # other - self  -> other + self * -1.0
        return other + self * -1.0
    def __mul__(self, other):
        match other:
            case int() | float():
                return LinearExpression([self], [float(other)], 0.0)
            case _:
                raise ValueError('Cannot multiply type %s with variable' % type(other).__name__)
    def __rmul__(self, other):
        return self * other


# LinearExpressions contain a set of variables, the coefficients for the variables, and a constant
# LinearExpressions can be used to create constraints and the objective in the Problem
# LinearExpressions can be added and subtracted with other LinearExpressions and Variables
# LinearExpressions can be multiplied and divided by scalars
# LinearExpressions can be compared with scalars, Variables, and other LinearExpressions to create Constraints
class LinearExpression:
    def __init__(self, vars, coefficients, constant):
        self.vars = vars
        self.coefficients = coefficients
        self.constant = constant
    def getVariables(self):
        return self.vars
    def getVariable(self, i):
        return self.vars[i]
    def getCoefficients(self):
        return self.coefficients
    def getCoefficient(self, i):
        return self.coefficients[i]
    def getConstant(self):
        return self.constant
    def zipVarCoefficients(self):
        return zip(self.vars, self.coefficients)
    def getValue(self):
        value = 0.0
        for i, var in enumerate(self.vars):
            value += var.Value * self.coefficients[i]
        return value
    def __len__(self):
        return len(self.vars)
    def __iadd__(self, other):
        match other:
            case int() | float():
                self.constant += float(other)
                return self
            case Variable():
                self.vars.append(other)
                self.coefficients.append(1.0)
                return self
            case LinearExpression():
                self.vars.extend(other.vars)
                self.coefficients.extend(other.coefficients)
                self.constant += other.constant
                return self
            case _:
                raise ValueError("Can't add type %s to Linear Expression" % type(other).__name__)
    def __add__(self, other):
        match other:
            case int() | float():
                return LinearExpression(self.vars, self.coefficients, self.constant + float(other))
            case Variable():
                vars = self.vars + [other]
                coeffs = self.coefficients + [1.0]
                return LinearExpression(vars, coeffs, self.constant)
            case LinearExpression():
                vars = self.vars + [other.vars]
                coeffs = self.coefficients + [other.coefficients]
                constant = self.constant + other.constant
                return LinearExpression(vars, coeffs, constant)
    def __radd__(self, other):
        return self + other
    def __isub__(self, other):
        match other:
            case int() | float():
                self.constant -= float(other)
                return self
            case Variable():
                self.vars.append(other)
                self.coefficients.append(-1.0)
                return self
            case LinearExpression():
                self.vars.extend(other.vars)
                for coeff in other.coefficients:  # Same Time Complexity as extend O(k), k = nelements to append
                    self.coefficients.append(-coeff)
                self.constant -= other.constant
                return self
            case _:
                raise ValueError("Can't sub type %s from LinearExpression" % type(other).__name__)
    def __sub__(self, other):
        match other:
            case int() | float():
                return LinearExpression(self.vars, self.coefficients, self.constant - float(other))
            case Variable():
                vars = self.vars + [other]
                coeffs = self.coefficients + [-1.0]
                return LinearExpression(vars, coeffs, self.constant)
            case LinearExpression():
                vars = self.vars + [other.vars]
                coeffs = []
                for i in self.coefficients:
                    coeffs.append(i)
                for i in other.coefficients:
                    coeffs.append[-1.0*i]
                constant = self.constant - other.constant
                return LinearExpression(vars, coeffs, constant)
    def __rsub__(self, other):
        # other - self  -> other + self * -1.0
        return other + self * -1.0
    def __mul__(self, other):
        match other:
            case int() | float():
                self.coefficients = [coeff * float(other) for coeff in self.coefficients]
                self.constant = self.constant * float(other)
                return self
            case _:
                raise ValueError("Can't multiply type %s by LinearExpresson" % type(other).__name__)
    def __rmul__(self, other):
        return self * other
    def __div__(self, other):
        match other:
            case int() | float():
                self.coefficients = [coeff / float(other) for coeff in self.coefficients]
                self.constant = self.constant / float(other)
                return self
            case _:
                raise ValueError("Can't divide LinearExpression by type %s" % type(other).__name__)
    def __le__(self, other):
        match other:
            case int() | float():
                return Constraint(self, CONSTRAINT_LE, float(other))
            case Variable() | LinearExpression():
                # expr1 <= expr2   -> expr1 - expr2 <= 0
                expr = self - other
                return Constraint(expr, CONSTRAINT_LE, 0.0)
    def __ge__(self, other):
        match other:
            case int() | float():
                return Constraint(self, CONSTRAINT_GE, float(other))
            case Variable() | LinearExpression():
                # expr1 >= expr2   ->  expr1 - expr2 >= 0
                expr = self - other
                return Constraint(expr, CONSTRAINT_GE, 0.0)
    def __eq__(self, other):
        match other:
            case int() | float():
                return Constraint(self, CONSTRAINT_EQ, float(other))
            case Variable() | LinearExpression():
                # expr1 == expr2   -> expr1 - expr2 == 0
                expr = self - other
                return Constraint(expr, CONSTRAINT_EQ, 0.0)

# The sense of a constraint is either less than or equal, greater than or equal, or equal
CONSTRAINT_LE = 'L'
CONSTRAINT_GE = 'G'
CONSTRAINT_EQ = 'E'

# A constraint contains a linear expression, the sense of the constraint, and the right-hand side of the constraint
class Constraint:
    def __init__(self, expr, sense, rhs, name=''):
        self.vindex_coeff_dict = {}
        nz = len(expr)
        self.vars = expr.vars
        for i in range(nz):
            v_idx = expr.vars[i].index
            v_coeff = expr.coefficients[i]
            self.vindex_coeff_dict[v_idx] = self.vindex_coeff_dict[v_idx] + v_coeff if v_idx in self.vindex_coeff_dict else v_coeff
        self.Sense = sense
        self.RHS = rhs - expr.getConstant()
        self.ConstraintName = name
        self.DualValue = float('nan')
    def __len__(self):
        return len(self.vindex_coeff_dict)
    def getName(self):
        return ConstraintName
    def getSense(self):
        return self.Sense
    def getRHS(self):
        return self.RHS
    def getCoefficient(self, var):
        v_idx = var.index
        return vindex_coeff_dict[v_idx]
    @property
    def Slack(self):
        lhs = 0.0
        for var in self.vars:
            lhs += var.Value * self.vindex_coeff_dict[var.index]
        return self.RHS - lhs

# The sense of a problem is either minimize or maximize
MINIMIZE = 0
MAXIMIZE = 1

# A Problem defines a Linear Program or Mixed Integer Program
# Variable can be be created by calling addVariable()
# Constraints can be added by calling addConstraint()
# The objective can be set by calling setObjective()
# The problem data is formed when calling optimize()
class Problem:
    def __init__(self, model_name=''):
        self.Name = model_name
        self.vars = []
        self.constrs = []
        self.ObjSense = MINIMIZE
        self.Obj = None
        self.ObjConstant = 0.0
        self.Status = -1
        self.IsMIP = False
        self.Settings = solver_settings.SolverSettings()

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

    def addVariable(self, lb=0.0, ub=float('inf'), obj=0.0, vtype=CONTINUOUS, name=''):
        n = len(self.vars)
        if vtype == INTEGER or vtype == BINARY:
            self.IsMIP = True
        var = Variable(self, n, lb, ub, obj, vtype, name)
        self.vars.append(var)
        return var

    def addConstraint(self, constr, name=''):
        n = len(self.constrs)
        match constr:
            case Constraint():
                constr.index = n
                constr.ConstraintName = name
                self.constrs.append(constr)
            case _:
                raise ValueError("addConstraint requires a Constraint object")

    def setObjective(self, expr, sense=MINIMIZE):
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
                raise ValueError('Objective must be a LinearExpression or a constant')
        self.Obj = expr

    def getObjective(self):
        return self.Obj

    def getVariabless(self):
        return self.vars

    def getConstraints(self):
        return self.constrs

    @property
    def NumVariables(self):
        return len(self.vars)

    @property
    def NumConstraints(self):
        return len(self.constrs)

    @property
    def NumNZs(self):
        nnz = 0
        for constr in self.constrs:
            nnz += len(constr)
        return nnz

    def getCSR(self):
        csr_dict = {'row_pointers' : [0],
                    'column_indices' : [],
                    'values' : []}
        for constr in self.constrs:
            csr_dict['column_indices'].extend(list(constr.vindex_coeff_dict.keys()))
            csr_dict['values'].extend(list(constr.vindex_coeff_dict.values()))
            csr_dict['row_pointers'].append(len(csr_dict['column_indices']))
        return self.dict_to_object(csr_dict)

    def post_solve(self, solution):
        self.Status = solution.get_termination_status()
        self.SolveTime = solution.get_solve_time()

        if solution.problem_category == 0:
            self.SolutionStats = solution.get_lp_stats()
        else:
            self.SolutionStats = solution.get_milp_stats()

        primal_sol = solution.get_primal_solution()
        reduced_cost = solution.get_reduced_cost()
        if len(primal_sol) > 0:
            for var in self.vars:
                var.Value = primal_sol[var.index]
                if not self.IsMIP:
                    var.ReducedCost =  reduced_cost[var.index]
        if not self.IsMIP:
            dual_sol = solution.get_dual_solution()
            if len(dual_sol) > 0:
                for i, constr in enumerate(self.constrs):
                    constr.DualValue = dual_sol[i]
        self.ObjVal = self.Obj.getValue()

    def solve(self):
        # iterate through the constraints and construct the constraint matrix and the rhs
        m = len(self.constrs)
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

        self.objective = []
        self.lower_bound, self.upper_bound = [], []
        self.var_type = []

        for j in range(n):
            self.objective.append(self.vars[j].getObjectiveCoefficient())
            self.var_type.append(self.vars[j].getVariableType())
            self.lower_bound.append(self.vars[j].getLowerBound())
            self.upper_bound.append(self.vars[j].getUpperBound())

        # Initialize datamodel
        dm = data_model.DataModel()
        dm.set_csr_constraint_matrix(np.array(self.values), np.array(self.column_indicies), np.array(self.row_pointers))
        dm.set_maximize(self.ObjSense)
        dm.set_constraint_bounds(np.array(self.rhs))
        dm.set_row_types(np.array(self.row_sense))
        dm.set_objective_coefficients(np.array(self.objective))
        dm.set_variable_lower_bounds(np.array(self.lower_bound))
        dm.set_variable_upper_bounds(np.array(self.upper_bound))
        dm.set_variable_types(np.array(self.var_type))

        # Call Solver
        solution = solver.Solve(dm, self.Settings)

        # Post Solve
        self.post_solve(solution)
