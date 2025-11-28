/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#include "ges_solver.cuh"
#include "routing_helpers.cuh"

#include "diversity/diverse_solver.hpp"

#include "adapters/adapted_generator.cuh"
#include "adapters/adapted_modifier.cuh"
#include "adapters/adapted_sol.cuh"
#include "adapters/assignment_adapter.cuh"
#include "ges/guided_ejection_search.cuh"

#include <rmm/mr/device_memory_resource.hpp>

namespace cuopt {
namespace routing {

template <typename i_t, typename f_t, request_t REQUEST>
ges_solver_t<i_t, f_t, REQUEST>::ges_solver_t(const data_model_view_t<i_t, f_t>& data_model,
                                              const solver_settings_t<i_t, f_t>& solver_settings,
                                              double time_limit_,
                                              i_t expected_route_count_,
                                              std::ofstream* intermediate_file_)
  : timer(time_limit_),
    problem(data_model, solver_settings),
    // override for now
    pool_allocator(problem, max_sol_per_population, expected_route_count_),
    expected_route_count(expected_route_count_),
    intermediate_file(intermediate_file_)
{
}

template <typename i_t, typename f_t, request_t REQUEST>
assignment_t<i_t> ges_solver_t<i_t, f_t, REQUEST>::compute_ges_solution(
  std::string diversity_manager_file)
{
  cuopt_expects(expected_route_count <= pool_allocator.problem.get_num_requests(),
                error_type_t::ValidationError,
                "Route count cannot be bigger than number of PD pairs");
  cuopt_expects(expected_route_count <= pool_allocator.problem.get_fleet_size(),
                error_type_t::ValidationError,
                "Route count cannot be bigger than number vehicles");

  // Weights for dimensions that do not have any constraints don't matter
  const double initial_weights[] = {
    10000., 10000., 100., 1000., 1000., 1000., 10000., 10000., 10000.};
  detail::infeasible_cost_t weights(initial_weights);
  auto cpu_weights = detail::get_cpu_cost(weights);
  solve<detail::pool_allocator_t<i_t,
                                 f_t,
                                 detail::solution_t<i_t, f_t, REQUEST>,
                                 detail::problem_t<i_t, f_t>>,
        detail::adapted_sol_t<i_t, f_t, REQUEST>,
        detail::problem_t<i_t, f_t>,
        detail::adapted_generator_t<i_t, f_t, REQUEST>,
        detail::adapted_modifier_t<i_t, f_t, REQUEST>>
    solver(&(pool_allocator.problem), cpu_weights, pool_allocator, diversity_manager_file, timer);
  bool feasible_only = false;

  solver.perform_search(expected_route_count, feasible_only);
  if (solver.reserve_population.is_feasible()) {
    auto best_sol = solver.reserve_population.best_feasible().sol;
    return get_ges_assignment(best_sol, solver.injection_info.accepted);
  } else {
    auto best_sol = solver.reserve_population.best().sol;
    return get_ges_assignment(best_sol, solver.injection_info.accepted);
  }
}

template class ges_solver_t<int, float, request_t::PDP>;
template class ges_solver_t<int, float, request_t::VRP>;

}  // namespace routing
}  // namespace cuopt
