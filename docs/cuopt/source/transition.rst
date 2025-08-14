=======================================
Transition Guide for Change in Features
=======================================

In addition to the quality improvements,  some new features were added, and some features were deprecated to improve user experience. For any questions, please reach out to the cuOpt team through github issues.

Parameter/option statuses are listed below, they express how each of these options are affected:

  **New** - A new feature has been added.

  **Update** - A change in definition of feature.

  **Deprecated** - These options will be accepted but will be removed in the future. In the case of the cuOpt service, the server will also return a warning noting that a feature is deprecated.

  **Limited** - These options are limited with respect to the number of dimensions that can be provided.

  **Removed** - These features were deprecated in a previous release and completely removed in this one.

For all solver_configs fields, see the LP/MILP settings guide :doc:`lp-milp-settings` or the service openapi spec :doc:`open-api`.

Changes to solver_configs.tolerances
------------------------------------

The following fields are **Deprecated** in ``solver_configs.tolerances`` for the service:

- absolute_primal
- absolute_dual
- absolute_gap
- relative_primal
- relative_dual
- relative_gap
- primal_infeasible
- dual_infeasible
- integrality_tolerance
- absolute_mip_gap
- relative_mip_gap

The following fields are **New** in ``solver_configs.tolerances`` for the service and replace the deprecated fields above:

- absolute_primal_tolerance
- absolute_dual_tolerance
- absolute_gap_tolerance
- relative_primal_tolerance
- relative_dual_tolerance
- relative_gap_tolerance
- primal_infeasible_tolerance
- dual_infeasible_tolerance
- mip_integrality_tolerance
- mip_absolute_gap
- mip_relative_gap

The following fields are **New** in ``solver_configs.tolerances`` for the service but were available in the C API in 25.05

- mip_absolute_tolerance
- mip_relative_tolerance

Changes to solver_configs
-------------------------

The following fields are **Deprecated** in ``solver_configs`` for the service:

- solver_mode
- heuristics_only

The following fields are **New** in ``solver_configs`` for the service and replace the deprecated fields above:

- pdlp_solver_mode
- mip_heuristics_only
  
The following are **New** in ``solver_configs`` for the service but were available in the C API in 25.05

- strict_infeasibility
- user_problem_file
- per_constraint_residual
- save_best_primal_so_far
- first_primal_feasible
- log_file
- solution_file
  

 
