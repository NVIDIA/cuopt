# Routing Python Tests – API Coverage

Summary of which APIs from `assignment.py` and `vehicle_routing.py` are exercised by tests under `cuopt/tests/routing/`.

---

## assignment.py

| API | Covered | Where |
|-----|---------|--------|
| **SolutionStatus** (enum) | Indirect | Tests assert `get_status() == 0` etc.; enum not referenced by name |
| **Assignment** (class) | Yes | Returned by `routing.Solve()` and used across tests |
| `get_vehicle_count()` | Yes | test_vehicle_properties, test_solver, test_solver_settings, test_batch_solve, etc. |
| `get_total_objective()` | Yes | test_vehicle_properties, test_solver, test_initial_solutions, etc. |
| `get_objective_values()` | Yes | test_vehicle_properties, test_solver |
| `get_route()` | Yes | Most routing tests |
| `get_status()` | Yes | All solve tests |
| `get_message()` | Yes | test_solver |
| `get_error_status()` | Yes | test_vehicle_properties (test_vehicle_max_times_fail) |
| `get_error_message()` | Yes | test_vehicle_properties (test_vehicle_max_times_fail) |
| `get_infeasible_orders()` | Yes | test_solver (test_pdptw) |
| `get_accepted_solutions()` | Yes | test_solver (test_pdptw) |
| `display_routes()` | Yes | test_vehicle_properties (test_vehicle_fixed_costs) |

---

## vehicle_routing.py – DataModel

### Setters

| API | Covered | Where |
|-----|---------|--------|
| `add_cost_matrix()` | Yes | test_data_model, test_vehicle_properties, test_solver, test_batch_solve, etc. |
| `add_transit_time_matrix()` | Yes | test_vehicle_properties, test_solver, test_initial_solutions, test_re_routing, etc. |
| `set_break_locations()` | Yes | test_vehicle_properties (test_empty_routes_with_breaks) |
| `add_break_dimension()` | Yes | test_vehicle_properties (test_empty_routes_with_breaks), test_solver, test_initial_solutions |
| `add_vehicle_break()` | Yes | test_vehicle_properties (test_heterogenous_breaks) |
| `set_objective_function()` | Yes | test_data_model, test_initial_solutions |
| `add_initial_solutions()` | Yes | test_initial_solutions |
| `set_order_locations()` | Yes | test_vehicle_properties, test_solver, test_initial_solutions, test_warnings, etc. |
| `set_vehicle_types()` | Yes | test_vehicle_properties, test_initial_solutions, test_warnings |
| `set_pickup_delivery_pairs()` | Yes | test_vehicle_properties, test_solver, test_warnings, test_re_routing |
| `set_vehicle_time_windows()` | Yes | test_vehicle_properties, test_initial_solutions, test_re_routing |
| `set_vehicle_locations()` | Yes | test_vehicle_properties, test_initial_solutions |
| `set_order_time_windows()` | Yes | test_data_model, test_vehicle_properties, test_solver, etc. |
| `set_order_prizes()` | Yes | test_solver, test_initial_solutions |
| `set_drop_return_trips()` | Yes | test_re_routing, test_initial_solutions |
| `set_skip_first_trips()` | Yes | test_initial_solutions |
| `add_vehicle_order_match()` | Yes | test_vehicle_properties, test_initial_solutions, test_re_routing |
| `add_order_vehicle_match()` | Yes | test_vehicle_properties, test_initial_solutions, test_solver |
| `set_order_service_times()` | Yes | test_vehicle_properties, test_data_model, test_solver, etc. |
| `add_capacity_dimension()` | Yes | test_data_model, test_vehicle_properties, test_solver, etc. |
| `set_vehicle_max_costs()` | Yes | test_vehicle_properties, test_solver_settings |
| `set_vehicle_max_times()` | Yes | test_vehicle_properties |
| `set_vehicle_fixed_costs()` | Yes | test_vehicle_properties |
| `set_min_vehicles()` | Yes | test_vehicle_properties, test_solver_settings, test_initial_solutions, test_solver |

### Getters

| API | Covered | Where |
|-----|---------|--------|
| `get_num_locations()` | Yes | test_solver (getter check), test_re_routing |
| `get_fleet_size()` | Yes | test_data_model, test_vehicle_properties, test_solver_settings |
| `get_num_orders()` | Yes | test_data_model |
| `get_cost_matrix()` | Yes | test_data_model (test_order_constraints, test_multi_cost_and_transit_matrices_getters) |
| `get_transit_time_matrix()` | Yes | test_data_model, test_solver (test_pdptw) |
| `get_transit_time_matrices()` | **No** | — |
| `get_initial_solutions()` | Yes | test_initial_solutions |
| `get_order_locations()` | Yes | test_vehicle_properties (test_single_vehicle_with_match, test_empty_routes_with_breaks) |
| `get_vehicle_types()` | Yes | test_vehicle_properties (test_vehicle_types) |
| `get_pickup_delivery_pairs()` | Yes | test_solver (test_pdptw) |
| `get_vehicle_time_windows()` | Yes | test_vehicle_properties (test_time_windows) |
| `get_vehicle_locations()` | Yes | test_vehicle_properties (test_vehicle_locations) |
| `get_drop_return_trips()` | Yes | test_initial_solutions (SKIP_DEPOTS) |
| `get_skip_first_trips()` | Yes | test_initial_solutions (SKIP_DEPOTS) |
| `get_capacity_dimensions()` | Yes | test_data_model |
| `get_order_time_windows()` | Yes | test_vehicle_properties (test_time_windows uses order TW from model) |
| `get_order_prizes()` | Yes | test_solver |
| `get_break_locations()` | Yes | test_vehicle_properties (test_empty_routes_with_breaks) |
| `get_break_dimensions()` | Yes | test_vehicle_properties (test_empty_routes_with_breaks) |
| `get_non_uniform_breaks()` | Yes | test_vehicle_properties (test_heterogenous_breaks) |
| `get_objective_function()` | Yes | test_data_model |
| `get_vehicle_max_costs()` | Yes | test_vehicle_properties (test_vehicle_max_costs) |
| `get_vehicle_max_times()` | Yes | test_vehicle_properties (test_vehicle_max_times) |
| `get_vehicle_fixed_costs()` | Yes | test_vehicle_properties (test_vehicle_fixed_costs) |
| `get_vehicle_order_match()` | Yes | test_vehicle_properties (test_vehicle_to_order_match) |
| `get_order_vehicle_match()` | Yes | test_vehicle_properties (test_order_to_vehicle_match) |
| `get_order_service_times()` | Yes | test_data_model |
| `get_min_vehicles()` | Yes | test_vehicle_properties (test_vehicle_types, test_pickup_delivery_orders) |

---

## vehicle_routing.py – SolverSettings

| API | Covered | Where |
|-----|---------|--------|
| `set_time_limit()` | Yes | All solve tests |
| `set_verbose_mode()` | Yes | test_solver_settings (test_verbose_mode) |
| `set_error_logging_mode()` | **No** | — |
| `dump_best_results()` | Yes | test_solver_settings (test_dump_results) |
| `dump_config_file()` | Yes | test_solver_settings (test_dump_config) |
| `get_time_limit()` | Yes | test_solver_settings (test_solver_settings_getters) |
| `get_best_results_file_path()` | Yes | test_solver_settings (test_dump_results) |
| `get_config_file_name()` | Yes | test_solver_settings (test_dump_config) |
| `get_best_results_interval()` | Yes | test_solver_settings (test_dump_results) |

---

## vehicle_routing.py – Solve / BatchSolve

| API | Covered | Where |
|-----|---------|--------|
| `Solve(data_model, solver_settings)` | Yes | All routing tests that run the solver |
| `BatchSolve(data_model_list, solver_settings)` | Yes | test_batch_solve |

---

## Summary

- **assignment.py:** All listed APIs are covered.
- **DataModel not covered:** getter `get_transit_time_matrices()` only.
- **SolverSettings not covered:** `set_error_logging_mode()`.

All other **setters** on DataModel (and all except `set_error_logging_mode` on SolverSettings) are covered by at least one test.
