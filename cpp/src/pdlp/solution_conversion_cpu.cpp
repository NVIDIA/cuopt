/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

// Host-side solution conversions, split out of solution_conversion.cu.
//
// cpu_lp_solution_t / cpu_mip_solution_t hold std::vector data and simply move it into
// the cython ret structs -- no device memory involved. Keeping them in a .cu TU forced
// the gRPC client to depend on libcuopt.so purely to resolve these two symbols, so they
// live in cuopt_client instead. The GPU counterparts stay in solution_conversion.cu.

#include <cuopt/export.hpp>
#include <cuopt/mathematical_optimization/cpu_optimization_problem_solution.hpp>
#include <cuopt/mathematical_optimization/utilities/cython_solve.hpp>

#include <string>
#include <utility>

namespace cuopt::mathematical_optimization {

// CPU LP Solution Conversion
// ===========================

template <typename i_t, typename f_t>
cuopt::cython::linear_programming_ret_t
cpu_lp_solution_t<i_t, f_t>::to_cpu_linear_programming_ret_t()
{
  using cpu_solutions_t = cuopt::cython::linear_programming_ret_t::cpu_solutions_t;
  cuopt::cython::linear_programming_ret_t ret;

  cpu_solutions_t cpu;
  cpu.primal_solution_ = std::move(primal_solution_);
  cpu.dual_solution_   = std::move(dual_solution_);
  cpu.reduced_cost_    = std::move(reduced_cost_);

  if (!pdlp_warm_start_data_.current_primal_solution_.empty()) {
    cpu.current_primal_solution_ = std::move(pdlp_warm_start_data_.current_primal_solution_);
    cpu.current_dual_solution_   = std::move(pdlp_warm_start_data_.current_dual_solution_);
    cpu.initial_primal_average_  = std::move(pdlp_warm_start_data_.initial_primal_average_);
    cpu.initial_dual_average_    = std::move(pdlp_warm_start_data_.initial_dual_average_);
    cpu.current_ATY_             = std::move(pdlp_warm_start_data_.current_ATY_);
    cpu.sum_primal_solutions_    = std::move(pdlp_warm_start_data_.sum_primal_solutions_);
    cpu.sum_dual_solutions_      = std::move(pdlp_warm_start_data_.sum_dual_solutions_);
    cpu.last_restart_duality_gap_primal_solution_ =
      std::move(pdlp_warm_start_data_.last_restart_duality_gap_primal_solution_);
    cpu.last_restart_duality_gap_dual_solution_ =
      std::move(pdlp_warm_start_data_.last_restart_duality_gap_dual_solution_);

    ret.initial_primal_weight_         = pdlp_warm_start_data_.initial_primal_weight_;
    ret.initial_step_size_             = pdlp_warm_start_data_.initial_step_size_;
    ret.total_pdlp_iterations_         = pdlp_warm_start_data_.total_pdlp_iterations_;
    ret.total_pdhg_iterations_         = pdlp_warm_start_data_.total_pdhg_iterations_;
    ret.last_candidate_kkt_score_      = pdlp_warm_start_data_.last_candidate_kkt_score_;
    ret.last_restart_kkt_score_        = pdlp_warm_start_data_.last_restart_kkt_score_;
    ret.sum_solution_weight_           = pdlp_warm_start_data_.sum_solution_weight_;
    ret.iterations_since_last_restart_ = pdlp_warm_start_data_.iterations_since_last_restart_;
  }

  ret.solutions_ = std::move(cpu);

  ret.termination_status_ = termination_status_;
  ret.error_status_       = error_status_.get_error_type();
  ret.error_message_      = std::string(error_status_.what());
  ret.l2_primal_residual_ = l2_primal_residual_;
  ret.l2_dual_residual_   = l2_dual_residual_;
  ret.primal_objective_   = primal_objective_;
  ret.dual_objective_     = dual_objective_;
  ret.gap_                = gap_;
  ret.nb_iterations_      = num_iterations_;
  ret.solve_time_         = solve_time_;
  ret.solved_by_          = solved_by_;

  return ret;
}

// ===========================
// CPU MIP Solution Conversion
// ===========================

template <typename i_t, typename f_t>
cuopt::cython::mip_ret_t cpu_mip_solution_t<i_t, f_t>::to_cpu_mip_ret_t()
{
  cuopt::cython::mip_ret_t ret;

  ret.solution_ = std::move(solution_);

  ret.termination_status_           = termination_status_;
  ret.error_status_                 = error_status_.get_error_type();
  ret.error_message_                = std::string(error_status_.what());
  ret.objective_                    = objective_;
  ret.mip_gap_                      = mip_gap_;
  ret.solution_bound_               = solution_bound_;
  ret.total_solve_time_             = total_solve_time_;
  ret.presolve_time_                = presolve_time_;
  ret.max_constraint_violation_     = max_constraint_violation_;
  ret.max_int_violation_            = max_int_violation_;
  ret.max_variable_bound_violation_ = max_variable_bound_violation_;
  ret.nodes_                        = num_nodes_;
  ret.simplex_iterations_           = num_simplex_iterations_;

  return ret;
}

// Explicit template instantiations
template CUOPT_EXPORT cuopt::cython::linear_programming_ret_t
cpu_lp_solution_t<int, double>::to_cpu_linear_programming_ret_t();
template CUOPT_EXPORT cuopt::cython::mip_ret_t cpu_mip_solution_t<int, double>::to_cpu_mip_ret_t();

}  // namespace cuopt::mathematical_optimization
