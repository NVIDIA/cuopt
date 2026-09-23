/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#pragma once

#include <dual_simplex/basis_updates.hpp>
#include <dual_simplex/simplex_solver_settings.hpp>
#include <dual_simplex/solution.hpp>
#include <dual_simplex/user_problem.hpp>
#include <linear_algebra/sparse_vector.hpp>

#include <vector>

namespace cuopt::mathematical_optimization::mip {

template <typename i_t, typename f_t>
bool check_for_dual_degeneracy(const simplex::lp_solution_t<i_t, f_t>& solution,
                               const simplex::simplex_solver_settings_t<i_t, f_t>& settings,
                               const std::vector<i_t>& nonbasic_list,
                               std::vector<i_t>& zero_reduced_costs_vars,
                               std::vector<i_t>& zero_reduced_costs_vars_nonbasic_index);

template <typename i_t, typename f_t>
void fast_slack_integer_pivots(const simplex::lp_problem_t<i_t, f_t>& lp,
                               const simplex::simplex_solver_settings_t<i_t, f_t>& settings,
                               const std::vector<i_t>& fractional,
                               const std::vector<i_t>& row_to_slack,
                               const simplex::lp_solution_t<i_t, f_t>& solution,
                               const std::vector<simplex::variable_type_t>& var_types,
                               f_t start_time,
                               std::vector<i_t>& basic_list,
                               std::vector<i_t>& nonbasic_list,
                               std::vector<i_t>& nonbasic_index,
                               std::vector<i_t>& variable_to_basic,
                               std::vector<simplex::variable_status_t>& vstatus,
                               simplex::lp_solution_t<i_t, f_t>& soln,
                               simplex::basis_update_mpf_t<i_t, f_t>& basis_update,
                               f_t& work_estimate);

template <typename i_t, typename f_t>
i_t pivot_out_integer_variables(const simplex::lp_problem_t<i_t, f_t>& lp,
                                const simplex::simplex_solver_settings_t<i_t, f_t>& settings,
                                const std::vector<i_t>& new_slacks,
                                const std::vector<simplex::variable_type_t>& var_types,
                                f_t start_time,
                                std::vector<i_t>& basic_list,
                                std::vector<i_t>& nonbasic_list,
                                std::vector<simplex::variable_status_t>& vstatus,
                                simplex::lp_solution_t<i_t, f_t>& solution,
                                simplex::basis_update_mpf_t<i_t, f_t>& basis_update,
                                i_t& num_fractional,
                                std::vector<i_t>& fractional);

template <typename i_t, typename f_t>
i_t apply_delta_x_for_integer_pivot(const simplex::lp_problem_t<i_t, f_t>& lp,
                                    const simplex::simplex_solver_settings_t<i_t, f_t>& settings,
                                    i_t entering_index,
                                    i_t nonbasic_entering,
                                    i_t direction,
                                    const sparse_vector_t<i_t, f_t>& delta_x,
                                    const std::vector<simplex::variable_type_t>& var_types,
                                    f_t start_time,
                                    std::vector<i_t>& basic_list,
                                    std::vector<i_t>& nonbasic_list,
                                    std::vector<i_t>& nonbasic_index,
                                    std::vector<i_t>& variable_to_basic,
                                    std::vector<simplex::variable_status_t>& vstatus,
                                    sparse_vector_t<i_t, f_t>& utilde_sparse,
                                    simplex::lp_solution_t<i_t, f_t>& solution,
                                    simplex::basis_update_mpf_t<i_t, f_t>& basis_update,
                                    f_t& work_estimate);

template <typename i_t, typename f_t>
void dual_degenerate_feasibility_pump(const simplex::lp_problem_t<i_t, f_t>& lp,
                                      const simplex::simplex_solver_settings_t<i_t, f_t>& settings,
                                      const std::vector<simplex::variable_type_t>& var_types,
                                      const std::vector<f_t>& edge_norms,
                                      f_t root_relax_work_estimate,
                                      f_t start_time,
                                      std::vector<i_t>& basic_list,
                                      std::vector<i_t>& nonbasic_list,
                                      std::vector<simplex::variable_status_t>& vstatus,
                                      simplex::lp_solution_t<i_t, f_t>& soln,
                                      simplex::basis_update_mpf_t<i_t, f_t>& basis_update,
                                      i_t& num_fractional,
                                      std::vector<i_t>& fractional);

}  // namespace cuopt::mathematical_optimization::mip
