====================
LP and MILP Examples
====================

This section contains examples of how to use the cuOpt linear programming and mixed integer linear programming Python API.

.. note::

    The examples in this section are not exhaustive. They are provided to help you get started with the cuOpt linear programming and mixed integer linear programming Python API. For more examples, please refer to the `cuopt-examples GitHub repository <https://github.com/NVIDIA/cuopt-examples>`_.


Simple Linear Programming Example
---------------------------------

.. code-block:: python

    from cuopt.linear_programming.problem import Problem, VType, CType, sense

    # Create a new problem
    prob = Problem("Simple LP")
    
    # Add variables
    x = prob.addVariable(lb=0, vtype=VType.CONTINUOUS, name="x")
    y = prob.addVariable(lb=0, vtype=VType.CONTINUOUS, name="y")

    # Add constraints
    prob.addConstraint(x + y <= 10, name="c1")
    prob.addConstraint(x - y >= 0, name="c2")

    # Set objective function
    prob.setObjective(x + y, sense=sense.MAXIMIZE)
    
    # Solve the problem
    prob.solve()
    
    # Check solution status
    if prob.Status.name == "Optimal":
        print(f"Optimal solution found in {prob.SolveTime:.2f} seconds")
        print(f"x = {x.getValue()}")
        print(f"y = {y.getValue()}")
        print(f"Objective value = {prob.ObjVal}")

The response is as follows:

.. code-block:: text

    Optimal solution found in 0.00 seconds
    x = 5.0
    y = 5.0
    Objective value = 10.0

Mixed Integer Linear Programming Example
----------------------------------------

.. code-block:: python

    from cuopt.linear_programming.problem import Problem, VType, CType, sense

    # Create a new MIP problem
    prob = Problem("Simple MIP")
    
    # Add integer variables with bounds
    x = prob.addVariable(lb=0, vtype=VType.INTEGER, name="V_x")
    y = prob.addVariable(lb=10, ub=50, vtype=VType.INTEGER, name="V_y")

    # Verify variable properties
    print(f"Variable x name: {x.getVariableName()}")
    print(f"Variable y upper bound: {y.getUpperBound()}")
    print(f"Variable y lower bound: {y.getLowerBound()}")
    print(f"Variable x type: {x.getVariableType()}")
    print(f"Variable y type: {y.getVariableType()}")  # Returns "I" for integer

    # Add constraints
    prob.addConstraint(2 * x + 4 * y >= 230, name="C1")
    prob.addConstraint(3 * x + 2 * y <= 190, name="C2")

    # Verify constraint properties
    expected_name = ["C1", "C2"]
    expected_coefficient_x = [2, 3]
    expected_coefficient_y = [4, 2]
    expected_sense = [CType.GE, "L"]  # GE = Greater Equal, L = Less Equal
    expected_rhs = [230, 190]
    
    for i, c in enumerate(prob.getConstraints()):
        print(f"Constraint {c.getConstraintName()}:")
        print(f"  Sense: {c.getSense()}")
        print(f"  RHS: {c.getRHS()}")
        print(f"  Coefficient of x: {c.getCoefficient(x)}")
        print(f"  Coefficient of y: {c.getCoefficient(y)}")

    # Check problem statistics
    print(f"Number of variables: {prob.NumVariables}")
    print(f"Number of constraints: {prob.NumConstraints}")
    print(f"Number of non-zeros: {prob.NumNZs}")

    # Set objective function
    expr = 5 * x + 3 * y
    prob.setObjective(expr, sense=sense.MAXIMIZE)

    # Verify objective properties
    expected_obj_coeff = [5, 3]
    print(f"Objective variables: {expr.getVariables()}")
    print(f"Objective coefficients: {expr.getCoefficients()}")
    print(f"Objective sense: {prob.ObjSense}")
    print(f"Objective expression: {prob.getObjective()}")

    # Configure solver settings
    prob.Settings.set_parameter("time_limit", 60)

    # Solve the problem
    prob.solve()
    
    # Check solution status and results
    if prob.Status.name == "Optimal":
        print(f"Optimal solution found in {prob.SolveTime:.2f} seconds")
        print(f"x = {x.getValue()}")
        print(f"y = {y.getValue()}")
        print(f"Objective value = {prob.getObjectiveValue()}")
    else:
        print(f"Problem status: {prob.Status.name}")

The response is as follows:

.. code-block:: text

    Optimal solution found in 0.00 seconds
    x = 5.0
    y = 5.0
    Objective value = 10.0

Advanced Example: Production Planning
-------------------------------------

.. code-block:: python

    from cuopt.linear_programming.problem import Problem, VType, CType, sense

    # Production planning problem
    prob = Problem("Production Planning")
    
    # Decision variables: production quantities
    # x1 = units of product A
    # x2 = units of product B
    x1 = prob.addVariable(lb=0, vtype=VType.INTEGER, name="Product_A")
    x2 = prob.addVariable(lb=0, vtype=VType.INTEGER, name="Product_B")
    
    # Resource constraints
    # Machine time: 2 hours per unit of A, 1 hour per unit of B, max 100 hours
    prob.addConstraint(2 * x1 + x2 <= 100, name="Machine_Time")
    
    # Labor: 1 hour per unit of A, 3 hours per unit of B, max 120 hours
    prob.addConstraint(x1 + 3 * x2 <= 120, name="Labor_Hours")
    
    # Material: 4 units per unit of A, 2 units per unit of B, max 200 units
    prob.addConstraint(4 * x1 + 2 * x2 <= 200, name="Material")
    
    # Demand constraints
    prob.addConstraint(x1 >= 10, name="Min_Demand_A")
    prob.addConstraint(x2 >= 15, name="Min_Demand_B")
    
    # Objective: maximize profit
    # Profit: $50 per unit of A, $30 per unit of B
    prob.setObjective(50 * x1 + 30 * x2, sense=sense.MAXIMIZE)
    
    # Solve with time limit
    prob.Settings.set_parameter("time_limit", 30)
    prob.solve()
    
    # Display results
    if prob.Status.name == "Optimal":
        print("=== Production Planning Solution ===")
        print(f"Status: {prob.Status.name}")
        print(f"Solve time: {prob.SolveTime:.2f} seconds")
        print(f"Product A production: {x1.getValue()} units")
        print(f"Product B production: {x2.getValue()} units")
        print(f"Total profit: ${prob.ObjVal:.2f}")
        
        # Check constraint satisfaction
        print("\n=== Constraint Analysis ===")
        for constraint in prob.getConstraints():
            print(f"{constraint.getConstraintName()}: {constraint.getSense()} {constraint.getRHS()}")
    else:
        print(f"Problem not solved optimally. Status: {prob.Status.name}")

The response is as follows:

.. code-block:: text

    Optimal solution found in 0.00 seconds
    x = 5.0
    y = 5.0
    Objective value = 10.0

Working with Expressions and Constraints
----------------------------------------

.. code-block:: python

    from cuopt.linear_programming.problem import Problem, VType, CType, sense

    prob = Problem("Expression Example")
    
    # Create variables
    x = prob.addVariable(lb=0, name="x")
    y = prob.addVariable(lb=0, name="y")
    z = prob.addVariable(lb=0, name="z")
    
    # Create complex expressions
    expr1 = 2 * x + 3 * y - z
    expr2 = x + y + z
    
    # Add constraints using expressions
    prob.addConstraint(expr1 <= 100, name="Complex_Constraint_1")
    prob.addConstraint(expr2 >= 20, name="Complex_Constraint_2")
    
    # Add constraint with different senses
    prob.addConstraint(x + y == 50, name="Equality_Constraint")  # Equality
    prob.addConstraint(x <= 30, name="Upper_Bound")              # Less than
    prob.addConstraint(y >= 10, name="Lower_Bound")              # Greater than
    
    # Set objective
    prob.setObjective(expr1 + expr2, sense=sense.MAXIMIZE)
    
    # Solve
    prob.solve()
    
    if prob.Status.name == "Optimal":
        print("=== Expression Example Results ===")
        print(f"x = {x.getValue()}")
        print(f"y = {y.getValue()}")
        print(f"z = {z.getValue()}")
        print(f"Objective value = {prob.ObjVal}")
        
        # Show constraint details
        print("\n=== Constraint Details ===")
        for constraint in prob.getConstraints():
            print(f"{constraint.getConstraintName()}: {constraint.getSense()} {constraint.getRHS()}")

The response is as follows:

.. code-block:: text

    Optimal solution found in 0.00 seconds
    x = 5.0
    y = 5.0
    z = 5.0
    Objective value = 10.0