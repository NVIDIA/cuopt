# Python API examples (cuOpt)

## Python: Routing with Time Windows & Capacities (VRP)

```python
"""
Vehicle Routing Problem with:
- 1 depot (location 0)
- 5 customer locations (1-5)
- 2 vehicles with capacity 100 each
- Time windows for each location
- Demand at each customer
"""
import cudf
from cuopt import routing

# Cost/distance matrix (6x6: depot + 5 customers)
cost_matrix = cudf.DataFrame([
    [0,  10, 15, 20, 25, 30],  # From depot
    [10,  0, 12, 18, 22, 28],  # From customer 1
    [15, 12,  0, 10, 15, 20],  # From customer 2
    [20, 18, 10,  0,  8, 15],  # From customer 3
    [25, 22, 15,  8,  0, 10],  # From customer 4
    [30, 28, 20, 15, 10,  0],  # From customer 5
], dtype="float32")

# Also use as transit time matrix (same values for simplicity)
transit_time_matrix = cost_matrix.copy(deep=True)

# Order data (customers 1-5)
order_locations = cudf.Series([1, 2, 3, 4, 5])  # Location indices for orders

# Demand at each customer (single capacity dimension)
demand = cudf.Series([20, 30, 25, 15, 35], dtype="int32")  # Units to deliver

# Vehicle capacities (must match demand dimensions)
vehicle_capacity = cudf.Series([100, 100], dtype="int32")  # Each vehicle can carry 100 units

# Time windows for orders [earliest, latest]
order_earliest = cudf.Series([0,  10, 20,  0, 30], dtype="int32")
order_latest = cudf.Series([50, 60, 70, 80, 90], dtype="int32")

# Service time at each customer
service_times = cudf.Series([5, 5, 5, 5, 5], dtype="int32")

# Fleet configuration
n_fleet = 2

# Vehicle start/end locations (both start and return to depot)
vehicle_start = cudf.Series([0, 0], dtype="int32")
vehicle_end = cudf.Series([0, 0], dtype="int32")

# Vehicle time windows (operating hours)
vehicle_earliest = cudf.Series([0, 0], dtype="int32")
vehicle_latest = cudf.Series([200, 200], dtype="int32")

# Build the data model
dm = routing.DataModel(
    n_locations=cost_matrix.shape[0],
    n_fleet=n_fleet,
    n_orders=len(order_locations)
)

# Add matrices
dm.add_cost_matrix(cost_matrix)
dm.add_transit_time_matrix(transit_time_matrix)

# Add order data
dm.set_order_locations(order_locations)
dm.set_order_time_windows(order_earliest, order_latest)
dm.set_order_service_times(service_times)

# Add capacity dimension (name, demand_per_order, capacity_per_vehicle)
dm.add_capacity_dimension("weight", demand, vehicle_capacity)

# Add fleet data
dm.set_vehicle_locations(vehicle_start, vehicle_end)
dm.set_vehicle_time_windows(vehicle_earliest, vehicle_latest)

# Configure solver
ss = routing.SolverSettings()
ss.set_time_limit(10)  # seconds

# Solve
solution = routing.Solve(dm, ss)

# Check solution status
print(f"Status: {solution.get_status()}")

# Display routes
if solution.get_status() == 0:  # Success
    print("\n--- Solution Found ---")
    solution.display_routes()

    # Get detailed route data
    route_df = solution.get_route()
    print("\nDetailed route data:")
    print(route_df)

    # Get objective value (total cost)
    print(f"\nTotal cost: {solution.get_total_objective()}")
else:
    print("No feasible solution found (status != 0).")
```

## Python: Pickup and Delivery Problem (PDP)

```python
"""
Pickup and Delivery Problem:
- Items must be picked up from one location and delivered to another
- Same vehicle must do both pickup and delivery
- Pickup must occur before delivery
"""
import cudf
from cuopt import routing

# Cost matrix (depot + 4 locations)
cost_matrix = cudf.DataFrame([
    [0, 10, 20, 30, 40],
    [10, 0, 15, 25, 35],
    [20, 15, 0, 10, 20],
    [30, 25, 10, 0, 15],
    [40, 35, 20, 15, 0],
], dtype="float32")

transit_time_matrix = cost_matrix.copy(deep=True)

n_fleet = 2
n_orders = 4  # 2 pickup-delivery pairs = 4 orders

# Orders: pickup at loc 1 -> deliver at loc 2, pickup at loc 3 -> deliver at loc 4
order_locations = cudf.Series([1, 2, 3, 4])

# Pickup and delivery pairs (indices into order array)
# Order 0 (pickup) pairs with Order 1 (delivery)
# Order 2 (pickup) pairs with Order 3 (delivery)
pickup_indices = cudf.Series([0, 2])
delivery_indices = cudf.Series([1, 3])

# Demand: positive for pickup, negative for delivery (must sum to 0 per pair)
demand = cudf.Series([10, -10, 15, -15], dtype="int32")
vehicle_capacity = cudf.Series([50, 50], dtype="int32")

# Build model
dm = routing.DataModel(
    n_locations=cost_matrix.shape[0],
    n_fleet=n_fleet,
    n_orders=n_orders
)

dm.add_cost_matrix(cost_matrix)
dm.add_transit_time_matrix(transit_time_matrix)
dm.set_order_locations(order_locations)

# Add capacity dimension
dm.add_capacity_dimension("load", demand, vehicle_capacity)

# Set pickup and delivery constraints
dm.set_pickup_delivery_pairs(pickup_indices, delivery_indices)

# Fleet setup
dm.set_vehicle_locations(
    cudf.Series([0, 0]),  # Start at depot
    cudf.Series([0, 0])   # Return to depot
)

# Solve
ss = routing.SolverSettings()
ss.set_time_limit(10)
solution = routing.Solve(dm, ss)

print(f"Status: {solution.get_status()}")
if solution.get_status() == 0:
    solution.display_routes()
```

## Python: Linear Programming (LP)

```python
"""
Production Planning LP:
    maximize    40*chairs + 30*tables  (profit)
    subject to  2*chairs + 3*tables <= 240  (wood constraint)
                4*chairs + 2*tables <= 200  (labor constraint)
                chairs, tables >= 0
"""
from cuopt.linear_programming.problem import Problem, CONTINUOUS, MAXIMIZE
from cuopt.linear_programming.solver_settings import SolverSettings

# Create problem
problem = Problem("ProductionPlanning")

# Decision variables (continuous, non-negative)
chairs = problem.addVariable(lb=0, vtype=CONTINUOUS, name="chairs")
tables = problem.addVariable(lb=0, vtype=CONTINUOUS, name="tables")

# Constraints
problem.addConstraint(2 * chairs + 3 * tables <= 240, name="wood")
problem.addConstraint(4 * chairs + 2 * tables <= 200, name="labor")

# Objective: maximize profit
problem.setObjective(40 * chairs + 30 * tables, sense=MAXIMIZE)

# Solver settings
settings = SolverSettings()
settings.set_parameter("time_limit", 60)
settings.set_parameter("log_to_console", 1)

# Solve
problem.solve(settings)

# Check status and extract results
status = problem.Status.name
print(f"Status: {status}")

if status in ["Optimal", "PrimalFeasible"]:
    print(f"Optimal profit: ${problem.ObjValue:.2f}")
    print(f"Chairs to produce: {chairs.getValue():.1f}")
    print(f"Tables to produce: {tables.getValue():.1f}")

    # Get dual values (shadow prices)
    wood_constraint = problem.getConstraint("wood")
    labor_constraint = problem.getConstraint("labor")
    print(f"\nShadow price (wood): ${wood_constraint.DualValue:.2f} per unit")
    print(f"Shadow price (labor): ${labor_constraint.DualValue:.2f} per unit")
else:
    print(f"No optimal solution found. Status: {status}")
```

## Python: Mixed-Integer Linear Programming (MILP)

```python
"""
Facility Location MILP:
- Decide which warehouses to open (binary)
- Assign customers to open warehouses
- Minimize fixed costs + transportation costs
"""
from cuopt.linear_programming.problem import (
    Problem, CONTINUOUS, INTEGER, MINIMIZE
)
from cuopt.linear_programming.solver_settings import SolverSettings

# Problem data
warehouses = ["W1", "W2", "W3"]
customers = ["C1", "C2", "C3", "C4"]

fixed_costs = {"W1": 100, "W2": 150, "W3": 120}
capacities = {"W1": 50, "W2": 70, "W3": 60}
demands = {"C1": 20, "C2": 25, "C3": 15, "C4": 30}

# Transportation cost from warehouse to customer
transport_cost = {
    ("W1", "C1"): 5, ("W1", "C2"): 8, ("W1", "C3"): 6, ("W1", "C4"): 10,
    ("W2", "C1"): 7, ("W2", "C2"): 4, ("W2", "C3"): 9, ("W2", "C4"): 5,
    ("W3", "C1"): 6, ("W3", "C2"): 7, ("W3", "C3"): 4, ("W3", "C4"): 8,
}

# Create problem
problem = Problem("FacilityLocation")

# Decision variables
# y[w] = 1 if warehouse w is open (binary: INTEGER with bounds 0-1)
y = {w: problem.addVariable(lb=0, ub=1, vtype=INTEGER, name=f"open_{w}") for w in warehouses}

# x[w,c] = units shipped from w to c
x = {
    (w, c): problem.addVariable(lb=0, vtype=CONTINUOUS, name=f"ship_{w}_{c}")
    for w in warehouses for c in customers
}

# Objective: minimize fixed + transportation costs
problem.setObjective(
    sum(fixed_costs[w] * y[w] for w in warehouses) +
    sum(transport_cost[w, c] * x[w, c] for w in warehouses for c in customers),
    sense=MINIMIZE
)

# Constraints
# 1. Meet customer demand
for c in customers:
    problem.addConstraint(
        sum(x[w, c] for w in warehouses) == demands[c],
        name=f"demand_{c}"
    )

# 2. Respect warehouse capacity (only if open)
for w in warehouses:
    problem.addConstraint(
        sum(x[w, c] for c in customers) <= capacities[w] * y[w],
        name=f"capacity_{w}"
    )

# Solver settings
settings = SolverSettings()
settings.set_parameter("time_limit", 120)
settings.set_parameter("mip_relative_gap", 0.01)  # 1% optimality gap

# Solve
problem.solve(settings)

# Check status and extract results
status = problem.Status.name
print(f"Status: {status}")

if status in ["Optimal", "FeasibleFound"]:
    print(f"Total cost: ${problem.ObjValue:.2f}")
    print("\nOpen warehouses:")
    for w in warehouses:
        if y[w].getValue() > 0.5:
            print(f"  {w} (fixed cost: ${fixed_costs[w]})")

    print("\nShipments:")
    for w in warehouses:
        for c in customers:
            shipped = x[w, c].getValue()
            if shipped > 0.01:
                print(f"  {w} -> {c}: {shipped:.1f} units")
else:
    print(f"No solution found. Status: {status}")
```

## Python: Quadratic Programming (QP) - Beta

```python
"""
Portfolio Optimization QP (more complex):
    minimize    x^T * Q * x  (variance/risk)
    subject to  sum(x) = 1         (fully invested)
                r^T * x >= target  (minimum return)
                x >= 0             (no short selling)
"""
from cuopt.linear_programming.problem import Problem, CONTINUOUS, MINIMIZE
from cuopt.linear_programming.solver_settings import SolverSettings

# Create problem
problem = Problem("PortfolioOptimization")

# Decision variables: portfolio weights for 3 assets
x1 = problem.addVariable(lb=0.0, ub=1.0, vtype=CONTINUOUS, name="stock_a")
x2 = problem.addVariable(lb=0.0, ub=1.0, vtype=CONTINUOUS, name="stock_b")
x3 = problem.addVariable(lb=0.0, ub=1.0, vtype=CONTINUOUS, name="stock_c")

# Expected returns
r1, r2, r3 = 0.12, 0.08, 0.05  # 12%, 8%, 5%
target_return = 0.08

# Quadratic objective: minimize variance = x^T * Q * x
# Expand: 0.04*x1^2 + 0.02*x2^2 + 0.01*x3^2 + 2*0.01*x1*x2 + 2*0.005*x1*x3 + 2*0.008*x2*x3
problem.setObjective(
    0.04 * x1 * x1 + 0.02 * x2 * x2 + 0.01 * x3 * x3 +
    0.02 * x1 * x2 + 0.01 * x1 * x3 + 0.016 * x2 * x3,
    sense=MINIMIZE
)

# Constraints
problem.addConstraint(x1 + x2 + x3 == 1, name="fully_invested")
problem.addConstraint(r1 * x1 + r2 * x2 + r3 * x3 >= target_return, name="min_return")

# Solve
settings = SolverSettings()
settings.set_parameter("time_limit", 60)
problem.solve(settings)

# Check status and extract results
status = problem.Status.name
print(f"Status: {status}")

if status in ["Optimal", "PrimalFeasible"]:
    print(f"Portfolio variance: {problem.ObjValue:.6f}")
    print(f"Portfolio std dev: {problem.ObjValue**0.5:.4f}")
    print(f"\nOptimal allocation:")
    print(f"  Stock A: {x1.getValue()*100:.2f}%")
    print(f"  Stock B: {x2.getValue()*100:.2f}%")
    print(f"  Stock C: {x3.getValue()*100:.2f}%")
    exp_return = r1*x1.getValue() + r2*x2.getValue() + r3*x3.getValue()
    print(f"\nExpected return: {exp_return*100:.2f}%")
else:
    print(f"No optimal solution found. Status: {status}")
```
