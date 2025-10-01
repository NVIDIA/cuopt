====================
LP and MILP Examples
====================

This section contains examples of how to use the cuOpt linear programming and mixed integer linear programming Python API.

.. note::

    The examples in this section are not exhaustive. They are provided to help you get started with the cuOpt linear programming and mixed integer linear programming Python API. For more examples, please refer to the `cuopt-examples GitHub repository <https://github.com/NVIDIA/cuopt-examples>`_.


Simple Linear Programming Example
---------------------------------

.. code-block:: python

    from cuopt.linear_programming.problem import Problem, CONTINUOUS, MAXIMIZE
    from cuopt.linear_programming.solver_settings import SolverSettings

    # Create a new problem
    problem = Problem("Simple LP")

    # Add variables
    x = problem.addVariable(lb=0, vtype=CONTINUOUS, name="x")
    y = problem.addVariable(lb=0, vtype=CONTINUOUS, name="y")

    # Add constraints
    problem.addConstraint(x + y <= 10, name="c1")
    problem.addConstraint(x - y >= 0, name="c2")

    # Set objective function
    problem.setObjective(x + y, sense=MAXIMIZE)

    # Configure solver settings
    settings = SolverSettings()
    settings.set_parameter("time_limit", 60)

    # Solve the problem
    problem.solve(settings)

    # Check solution status
    if problem.Status.name == "Optimal":
        print(f"Optimal solution found in {problem.SolveTime:.2f} seconds")
        print(f"x = {x.getValue()}")
        print(f"y = {y.getValue()}")
        print(f"Objective value = {problem.ObjValue}")

The response is as follows:

.. code-block:: text

    Optimal solution found in 0.01 seconds
    x = 10.0
    y = 0.0
    Objective value = 10.0

Mixed Integer Linear Programming Example
----------------------------------------

.. code-block:: python

    from cuopt.linear_programming.problem import Problem, INTEGER, MAXIMIZE
    from cuopt.linear_programming.solver_settings import SolverSettings

    # Create a new MIP problem
    problem = Problem("Simple MIP")

    # Add integer variables with bounds
    x = problem.addVariable(vtype=INTEGER, name="V_x")
    y = problem.addVariable(lb=10, ub=50, vtype=INTEGER, name="V_y")

    # Add constraints
    problem.addConstraint(2 * x + 4 * y >= 230, name="C1")
    problem.addConstraint(3 * x + 2 * y <= 190, name="C2")

    # Set objective function
    problem.setObjective(5 * x + 3 * y, sense=MAXIMIZE)

    # Configure solver settings
    settings = SolverSettings()
    settings.set_parameter("time_limit", 60)

    # Solve the problem
    problem.solve(settings)

    # Check solution status and results
    if problem.Status.name == "Optimal":
        print(f"Optimal solution found in {problem.SolveTime:.2f} seconds")
        print(f"x = {x.getValue()}")
        print(f"y = {y.getValue()}")
        print(f"Objective value = {problem.ObjValue}")
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

    from cuopt.linear_programming.problem import Problem, INTEGER, MAXIMIZE
    from cuopt.linear_programming.solver_settings import SolverSettings

    # Production planning problem
    problem = Problem("Production Planning")

    # Decision variables: production quantities
    # x1 = units of product A
    # x2 = units of product B
    x1 = problem.addVariable(lb=10, vtype=INTEGER, name="Product_A")
    x2 = problem.addVariable(lb=15, vtype=INTEGER, name="Product_B")

    # Resource constraints
    # Machine time: 2 hours per unit of A, 1 hour per unit of B, max 100 hours
    problem.addConstraint(2 * x1 + x2 <= 100, name="Machine_Time")

    # Labor: 1 hour per unit of A, 3 hours per unit of B, max 120 hours
    problem.addConstraint(x1 + 3 * x2 <= 120, name="Labor_Hours")

    # Material: 4 units per unit of A, 2 units per unit of B, max 200 units
    problem.addConstraint(4 * x1 + 2 * x2 <= 200, name="Material")

    # Objective: maximize profit
    # Profit: $50 per unit of A, $30 per unit of B
    problem.setObjective(50 * x1 + 30 * x2, sense=MAXIMIZE)

    # Solve with time limit
    settings = SolverSettings()
    settings.set_parameter("time_limit", 30)
    problem.solve(settings)

    # Display results
    if problem.Status.name == "Optimal":
        print("=== Production Planning Solution ===")
        print(f"Status: {problem.Status.name}")
        print(f"Solve time: {problem.SolveTime:.2f} seconds")
        print(f"Product A production: {x1.getValue()} units")
        print(f"Product B production: {x2.getValue()} units")
        print(f"Total profit: ${problem.ObjValue:.2f}")

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

    from cuopt.linear_programming.problem import Problem, MAXIMIZE
    from cuopt.linear_programming.solver_settings import SolverSettings

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
    problem.setObjective(x + 2 * y + 3 * z, sense=MAXIMIZE)

    settings = SolverSettings()
    settings.set_parameter("time_limit", 20)

    problem.solve(settings)


    if problem.Status.name == "Optimal":
        print("=== Expression Example Results ===")
        print(f"x = {x.getValue()}")
        print(f"y = {y.getValue()}")
        print(f"z = {z.getValue()}")
        print(f"Objective value = {problem.ObjValue}")

The response is as follows:

.. code-block:: text

    === Expression Example Results ===
    x = 0.0
    y = 50.0
    z = 99.99999999999999
    Objective value = 399.99999999999994

Working with Incumbent Solutions
--------------------------------

Incumbent solutions are intermediate feasible solutions found during the MIP solving process. They represent the best integer-feasible solution discovered so far and can be accessed through callback functions.

.. note::
    Incumbent solutions are only available for Mixed Integer Programming (MIP) problems, not for pure Linear Programming (LP) problems.

.. code-block:: python

    from cuopt.linear_programming.problem import Problem, INTEGER, MAXIMIZE
    from cuopt.linear_programming.solver_settings import SolverSettings
    from cuopt.linear_programming.solver.solver_parameters import CUOPT_TIME_LIMIT
    from cuopt.linear_programming.internals import GetSolutionCallback, SetSolutionCallback

    # Create a callback class to receive incumbent solutions
    class IncumbentCallback(GetSolutionCallback):
        def __init__(self):
            super().__init__()
            self.solutions = []
            self.n_callbacks = 0

        def get_solution(self, solution, solution_cost):
            """
            Called whenever the solver finds a new incumbent solution.

            Parameters
            ----------
            solution : array-like
                The variable values of the incumbent solution
            solution_cost : array-like
                The objective value of the incumbent solution
            """
            self.n_callbacks += 1

            # Store the incumbent solution
            incumbent = {
                "solution": solution.copy_to_host(),
                "cost": solution_cost.copy_to_host()[0],
                "iteration": self.n_callbacks
            }
            self.solutions.append(incumbent)

            print(f"Incumbent {self.n_callbacks}: {incumbent['solution']}, cost: {incumbent['cost']:.2f}")

    # Create a more complex MIP problem that will generate multiple incumbents
    problem = Problem("Incumbent Example")

    # Add integer variables
    x = problem.addVariable(vtype=INTEGER)
    y = problem.addVariable(vtype=INTEGER)

    # Add constraints to create a problem that will generate multiple incumbents
    problem.addConstraint(2 * x + 4 * y >= 230)
    problem.addConstraint(3 * x + 2 * y <= 190)

    # Set objective to maximize
    problem.setObjective(5 * x + 3 * y, sense=MAXIMIZE)

    # Configure solver settings with callback
    settings = SolverSettings()
    # Set the incumbent callback
    incumbent_callback = IncumbentCallback()
    settings.set_mip_callback(incumbent_callback)
    settings.set_parameter(CUOPT_TIME_LIMIT, 30)  # Allow enough time to find multiple incumbents

    # Solve the problem
    problem.solve(settings)

    # Display final results
    print(f"\n=== Final Results ===")
    print(f"Problem status: {problem.Status.name}")
    print(f"Solve time: {problem.SolveTime:.2f} seconds")
    print(f"Final solution: x={x.getValue()}, y={y.getValue()}")
    print(f"Final objective value: {problem.ObjValue:.2f}")

The response is as follows:

.. code-block:: text

    Optimal solution found.
    Incumbent 1: [ 0. 58.], cost: 174.00
    Incumbent 2: [36. 41.], cost: 303.00
    Generated fast solution in 0.158467 seconds with objective 303.000000
    Consuming B&B solutions, solution queue size 2
    Solution objective: 303.000000 , relative_mip_gap 0.000000 solution_bound 303.000000 presolve_time 0.043211 total_solve_time 0.160270 max constraint violation 0.000000 max int violation 0.000000 max var bounds violation 0.000000 nodes 4 simplex_iterations 3

    === Final Results ===
    Problem status: Optimal
    Solve time: 0.16 seconds
    Final solution: x=36.0, y=40.99999999999999
    Final objective value: 303.00

Working with PDLP Warmstart Data
--------------------------------

Warmstart data allows to restart PDLP with a previous solution context. This should be used when you solve a new problem which is similar to the previous one.

.. note::
    Warmstart data is only available for Linear Programming (LP) problems, not for Mixed Integer Linear Programming (MILP) problems.

.. code-block:: python

    from cuopt.linear_programming.problem import Problem, CONTINUOUS, MAXIMIZE
    from cuopt.linear_programming.solver.solver_parameters import CUOPT_METHOD
    from cuopt.linear_programming.solver_settings import SolverSettings, SolverMethod

    # Create a new problem
    problem = Problem("Simple LP")

    # Add variables
    x = problem.addVariable(lb=0, vtype=CONTINUOUS, name="x")
    y = problem.addVariable(lb=0, vtype=CONTINUOUS, name="y")

    # Add constraints
    problem.addConstraint(4*x + 10*y <= 130, name="c1")
    problem.addConstraint(8*x - 3*y >= 40, name="c2")

    # Set objective function
    problem.setObjective(2*x + y, sense=MAXIMIZE)

    # Configure solver settings
    settings = SolverSettings()
    settings.set_parameter(CUOPT_METHOD, SolverMethod.PDLP)

    # Solve the problem
    problem.solve(settings)

    # Get the warmstart data
    warmstart_data = problem.get_pdlp_warm_start_data()

    print(warmstart_data.current_primal_solution)
    # Create a new problem
    new_problem = Problem("Warmstart LP")

    # Add variables
    x = new_problem.addVariable(lb=0, vtype=CONTINUOUS, name="x")
    y = new_problem.addVariable(lb=0, vtype=CONTINUOUS, name="y")

    # Add constraints
    new_problem.addConstraint(4*x + 10*y <= 100, name="c1")
    new_problem.addConstraint(8*x - 3*y >= 50, name="c2")

    # Set objective function
    new_problem.setObjective(2*x + y, sense=MAXIMIZE)

    # Configure solver settings
    settings.set_pdlp_warm_start_data(warmstart_data)

    # Solve the problem
    new_problem.solve(settings)

    # Check solution status
    if new_problem.Status.name == "Optimal":
        print(f"Optimal solution found in {new_problem.SolveTime:.2f} seconds")
        print(f"x = {x.getValue()}")
        print(f"y = {y.getValue()}")
        print(f"Objective value = {new_problem.ObjValue}")

The response is as follows:

.. code-block:: text

    Optimal solution found in 0.01 seconds
    x = 25.000000000639382
    y = 0.0
    Objective value = 50.000000001278764
