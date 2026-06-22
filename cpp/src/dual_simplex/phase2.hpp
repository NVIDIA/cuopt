/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#pragma once

#include <dual_simplex/basis_updates.hpp>
#include <dual_simplex/initial_basis.hpp>
#include <dual_simplex/logger.hpp>
#include <dual_simplex/presolve.hpp>
#include <dual_simplex/simplex_solver_settings.hpp>
#include <dual_simplex/types.hpp>
#include <utilities/memory_instrumentation.hpp>

#include <vector>

namespace cuopt {
struct work_limit_context_t;
}

namespace cuopt::mathematical_optimization::simplex {

enum class dual_status_t {
  OPTIMAL          = 0,
  DUAL_UNBOUNDED   = 1,
  NUMERICAL        = 2,
  CUTOFF           = 3,
  TIME_LIMIT       = 4,
  ITERATION_LIMIT  = 5,
  CONCURRENT_LIMIT = 6,
  WORK_LIMIT       = 7,
  UNSET            = 8
};

static std::string dual_status_to_string(dual_status_t status)
{
  switch (status) {
    case dual_status_t::OPTIMAL: return "OPTIMAL";
    case dual_status_t::DUAL_UNBOUNDED: return "DUAL_UNBOUNDED";
    case dual_status_t::NUMERICAL: return "NUMERICAL";
    case dual_status_t::CUTOFF: return "CUTOFF";
    case dual_status_t::TIME_LIMIT: return "TIME_LIMIT";
    case dual_status_t::ITERATION_LIMIT: return "ITERATION_LIMIT";
    case dual_status_t::CONCURRENT_LIMIT: return "CONCURRENT_LIMIT";
    case dual_status_t::WORK_LIMIT: return "WORK_LIMIT";
    case dual_status_t::UNSET: return "UNSET";
  }
  return "UNKNOWN";
}

template <typename i_t, typename f_t>
dual_status_t dual_phase2(i_t phase,
                          i_t slack_basis,
                          f_t start_time,
                          const lp_problem_t<i_t, f_t>& lp,
                          const simplex_solver_settings_t<i_t, f_t>& settings,
                          std::vector<variable_status_t>& vstatus,
                          lp_solution_t<i_t, f_t>& sol,
                          i_t& iter,
                          std::vector<f_t>& steepest_edge_norms,
                          work_limit_context_t* work_unit_context = nullptr);

template <typename i_t, typename f_t>
dual_status_t dual_phase2_with_advanced_basis(i_t phase,
                                              i_t slack_basis,
                                              bool initialize_basis,
                                              f_t start_time,
                                              const lp_problem_t<i_t, f_t>& lp,
                                              const simplex_solver_settings_t<i_t, f_t>& settings,
                                              std::vector<variable_status_t>& vstatus,
                                              basis_update_mpf_t<i_t, f_t>& ft,
                                              std::vector<i_t>& basic_list,
                                              std::vector<i_t>& nonbasic_list,
                                              lp_solution_t<i_t, f_t>& sol,
                                              i_t& iter,
                                              std::vector<f_t>& delta_y_steepest_edge,
                                              work_limit_context_t* work_unit_context = nullptr);

template <typename i_t, typename f_t>
void compute_reduced_cost_update(const lp_problem_t<i_t, f_t>& lp,
                                 const std::vector<i_t>& basic_list,
                                 const std::vector<i_t>& nonbasic_list,
                                 const std::vector<f_t>& delta_y,
                                 i_t leaving_index,
                                 i_t direction,
                                 std::vector<i_t>& delta_z_mark,
                                 std::vector<i_t>& delta_z_indices,
                                 std::vector<f_t>& delta_z,
                                 f_t& work_estimate);

template <typename i_t, typename f_t>
void compute_delta_z(const csr_matrix_t<i_t, f_t>& Arow,
                     const sparse_vector_t<i_t, f_t>& delta_y,
                     i_t leaving_index,
                     i_t direction,
                     const std::vector<i_t>& nonbasic_end,
                     std::vector<i_t>& delta_z_mark,
                     std::vector<i_t>& delta_z_indices,
                     std::vector<f_t>& delta_z,
                     f_t& work_estimate);

template <typename i_t, typename f_t>
void compute_initial_nonbasic_end(const std::vector<i_t>& basic_mark,
                                  csr_matrix_t<i_t, f_t>& Arow,
                                  std::vector<i_t>& nonbasic_end);

}  // namespace cuopt::mathematical_optimization::simplex
