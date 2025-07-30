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
    problem = Problem("Simple LP")
    
    # Add variables
    x = problem.addVariable(lb=0, vtype=VType.CONTINUOUS, name="x")
    y = problem.addVariable(lb=0, vtype=VType.CONTINUOUS, name="y")

    # Add constraints
    problem.addConstraint(x + y <= 10, name="c1")
    problem.addConstraint(x - y >= 0, name="c2")

    # Set objective function
    problem.setObjective(x + y, sense=sense.MAXIMIZE)
    
    # Solve the problem
    problem.solve()
    
    # Check solution status
    if problem.Status.name == "Optimal":
        print(f"Optimal solution found in {problem.SolveTime:.2f} seconds")
        print(f"x = {x.getValue()}")
        print(f"y = {y.getValue()}")
        print(f"Objective value = {problem.ObjVal}")

The response is as follows:

.. code-block:: text

    Optimal solution found in 0.01 seconds
    x = 10.0
    y = 0.0
    Objective value = 10.0

Mixed Integer Linear Programming Example
----------------------------------------

.. code-block:: python

    from cuopt.linear_programming.problem import Problem, VType, CType, sense

    # Create a new MIP problem
    problem = Problem("Simple MIP")
    
    # Add integer variables with bounds
    x = problem.addVariable(vtype=VType.INTEGER, name="V_x")
    y = problem.addVariable(lb=10, ub=50, vtype=VType.INTEGER, name="V_y")

    # Add constraints
    problem.addConstraint(2 * x + 4 * y >= 230, name="C1")
    problem.addConstraint(3 * x + 2 * y <= 190, name="C2")

    # Set objective function
    problem.setObjective(5 * x + 3 * y, sense=sense.MAXIMIZE)

    # Configure solver settings
    problem.Settings.set_parameter("time_limit", 60)

    # Solve the problem
    problem.solve()
    
    # Check solution status and results
    if problem.Status.name == "Optimal":
        print(f"Optimal solution found in {problem.SolveTime:.2f} seconds")
        print(f"x = {x.getValue()}")
        print(f"y = {y.getValue()}")
        print(f"Objective value = {problem.ObjVal}")
    else:
        print(f"Problem status: {problem.Status.name}")

The response is as follows:

.. code-block:: text

    Optimal solution found in 0.00 seconds
    x = 36.0
    y = 40.99999999999999
    Objective value = 303.0


Advanced Example: Production Planning
-------------------------------------

.. code-block:: python

    from cuopt.linear_programming.problem import Problem, VType, CType, sense

    # Production planning problem
    problem = Problem("Production Planning")
    
    # Decision variables: production quantities
    # x1 = units of product A
    # x2 = units of product B
    x1 = problem.addVariable(lb=10, vtype=VType.INTEGER, name="Product_A")
    x2 = problem.addVariable(lb=15, vtype=VType.INTEGER, name="Product_B")
    
    # Resource constraints
    # Machine time: 2 hours per unit of A, 1 hour per unit of B, max 100 hours
    problem.addConstraint(2 * x1 + x2 <= 100, name="Machine_Time")
    
    # Labor: 1 hour per unit of A, 3 hours per unit of B, max 120 hours
    problem.addConstraint(x1 + 3 * x2 <= 120, name="Labor_Hours")
    
    # Material: 4 units per unit of A, 2 units per unit of B, max 200 units
    problem.addConstraint(4 * x1 + 2 * x2 <= 200, name="Material")
    
    # Objective: maximize profit
    # Profit: $50 per unit of A, $30 per unit of B
    problem.setObjective(50 * x1 + 30 * x2, sense=sense.MAXIMIZE)
    
    # Solve with time limit
    problem.Settings.set_parameter("time_limit", 30)
    problem.solve()
    
    # Display results
    if problem.Status.name == "Optimal":
        print("=== Production Planning Solution ===")
        print(f"Status: {problem.Status.name}")
        print(f"Solve time: {problem.SolveTime:.2f} seconds")
        print(f"Product A production: {x1.getValue()} units")
        print(f"Product B production: {x2.getValue()} units")
        print(f"Total profit: ${problem.ObjVal:.2f}")
        
    else:
        print(f"Problem not solved optimally. Status: {problem.Status.name}")

The response is as follows:

.. code-block:: text

    === Production Planning Solution ===

    Status: Optimal
    Solve time: 0.09 seconds
    Product A production: 36.0 units
    Product B production: 28.000000000000004 units
    Total profit: $2640.00

Working with Expressions and Constraints
----------------------------------------

.. code-block:: python

    from cuopt.linear_programming.problem import Problem, VType, CType, sense

    problem = Problem("Expression Example")
    
    # Create variables
    x = problem.addVariable(lb=0, name="x")
    y = problem.addVariable(lb=0, name="y")
    z = problem.addVariable(lb=0, name="z")
    
    # Create complex expressions
    expr1 = 2 * x + 3 * y - z
    expr2 = x + y + z
    
    # Add constraints using expressions
    problem.addConstraint(expr1 <= 100, name="Complex_Constraint_1")
    problem.addConstraint(expr2 >= 20, name="Complex_Constraint_2")
    
    # Add constraint with different senses
    problem.addConstraint(x + y == 50, name="Equality_Constraint")
    problem.addConstraint(1 * x <= 30, name="Upper_Bound_X")
    problem.addConstraint(1 * y >= 10, name="Lower_Bound_Y")
    problem.addConstraint(1 * z <= 100, name="Upper_Bound_Z")
    
    # Set objective
    problem.setObjective(x + 2 * y + 3 * z, sense=sense.MAXIMIZE)

    problem.Settings.set_parameter("time_limit", 20) 

    problem.solve()
    
    
    if problem.Status.name == "Optimal":
        print("=== Expression Example Results ===")
        print(f"x = {x.getValue()}")
        print(f"y = {y.getValue()}")
        print(f"z = {z.getValue()}")
        print(f"Objective value = {problem.ObjVal}")
        
The response is as follows:

.. code-block:: text

    === Expression Example Results ===
    x = 0.0
    y = 50.0
    z = 99.99999999999999
    Objective value = 399.99999999999994
