/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2025 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved. SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#pragma once

#include <linear_programming/pdhg.hpp>
#include <linear_programming/termination_strategy/convergence_information.hpp>
#include <linear_programming/termination_strategy/infeasibility_information.hpp>

#include <cuopt/linear_programming/pdlp/pdlp_warm_start_data.hpp>
#include <cuopt/linear_programming/pdlp/solver_settings.hpp>
#include <cuopt/linear_programming/pdlp/solver_solution.hpp>
#include <mip/problem/problem.cuh>

#include <raft/core/handle.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_scalar.hpp>
#include <rmm/device_uvector.hpp>

namespace cuopt::linear_programming::detail {
template <typename i_t, typename f_t>
class pdlp_termination_strategy_t {
 public:
  pdlp_termination_strategy_t(raft::handle_t const* handle_ptr,
                              problem_t<i_t, f_t>& op_problem,
                              cusparse_view_t<i_t, f_t>& cusparse_view,
                              const i_t primal_size,
                              const i_t dual_size,
                              const pdlp_solver_settings_t<i_t, f_t>& settings);

  pdlp_termination_status_t evaluate_termination_criteria(
    pdhg_solver_t<i_t, f_t>& current_pdhg_solver,
    rmm::device_uvector<f_t>& primal_iterate,
    rmm::device_uvector<f_t>& dual_iterate,
    [[maybe_unused]] const rmm::device_uvector<f_t>& dual_slack,
    const rmm::device_uvector<f_t>& combined_bounds,  // Only useful if per_constraint_residual
    const rmm::device_uvector<f_t>&
      objective_coefficients  // Only useful if per_constraint_residual
  );

  void print_termination_criteria(i_t iteration, f_t elapsed);

  void set_relative_dual_tolerance_factor(f_t dual_tolerance_factor);
  void set_relative_primal_tolerance_factor(f_t primal_tolerance_factor);
  f_t get_relative_dual_tolerance_factor() const;
  f_t get_relative_primal_tolerance_factor() const;

  const convergence_information_t<i_t, f_t>& get_convergence_information() const;

  // Deep copy is used when save best primal so far is toggled
  optimization_problem_solution_t<i_t, f_t> fill_return_problem_solution(
    i_t number_of_iterations,
    pdhg_solver_t<i_t, f_t>& current_pdhg_solver,
    rmm::device_uvector<f_t>& primal_iterate,
    rmm::device_uvector<f_t>& dual_iterate,
    pdlp_warm_start_data_t<i_t, f_t> warm_start_data,
    pdlp_termination_status_t termination_status,
    bool deep_copy = false);

  // Call the above with an empty pdlp_warm_start_data
  optimization_problem_solution_t<i_t, f_t> fill_return_problem_solution(
    i_t number_of_iterations,
    pdhg_solver_t<i_t, f_t>& current_pdhg_solver,
    rmm::device_uvector<f_t>& primal_iterate,
    rmm::device_uvector<f_t>& dual_iterate,
    pdlp_termination_status_t termination_status,
    bool deep_copy = false);

 private:
  void check_termination_criteria();

  raft::handle_t const* handle_ptr_{nullptr};
  rmm::cuda_stream_view stream_view_;

  problem_t<i_t, f_t>* problem_ptr;

  convergence_information_t<i_t, f_t> convergence_information_;
  infeasibility_information_t<i_t, f_t> infeasibility_information_;

  rmm::device_scalar<i_t> termination_status_;
  const pdlp_solver_settings_t<i_t, f_t>& settings_;
};
}  // namespace cuopt::linear_programming::detail
