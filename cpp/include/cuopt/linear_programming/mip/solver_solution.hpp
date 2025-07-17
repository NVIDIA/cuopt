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

#include <cuopt/linear_programming/constants.h>
#include <cuopt/error.hpp>
#include <cuopt/linear_programming/mip/solver_stats.hpp>
#include <cuopt/linear_programming/utilities/internals.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_uvector.hpp>

#include <raft/core/handle.hpp>

#include <fstream>
#include <string>
#include <vector>

namespace cuopt::linear_programming {

enum class mip_termination_status_t : int8_t {
  NoTermination = CUOPT_TERIMINATION_STATUS_NO_TERMINATION,
  Optimal       = CUOPT_TERIMINATION_STATUS_OPTIMAL,
  FeasibleFound = CUOPT_TERIMINATION_STATUS_FEASIBLE_FOUND,
  Infeasible    = CUOPT_TERIMINATION_STATUS_INFEASIBLE,
  Unbounded     = CUOPT_TERIMINATION_STATUS_UNBOUNDED,
  TimeLimit     = CUOPT_TERIMINATION_STATUS_TIME_LIMIT,
};

template <typename i_t, typename f_t>
class mip_solution_t : public base_solution_t {
 public:
  mip_solution_t(rmm::device_uvector<f_t> solution,
                 std::vector<std::string> var_names,
                 f_t objective,
                 f_t mip_gap,
                 mip_termination_status_t termination_status,
                 f_t max_constraint_violation,
                 f_t max_int_violation,
                 f_t max_variable_bound_violation,
                 solver_stats_t<i_t, f_t> stats,
                 std::vector<rmm::device_uvector<f_t>> solution_pool = {});

  mip_solution_t(mip_termination_status_t termination_status,
                 solver_stats_t<i_t, f_t> stats,
                 rmm::cuda_stream_view stream_view);
  mip_solution_t(const cuopt::logic_error& error_status, rmm::cuda_stream_view stream_view);

  bool is_mip() const override { return true; }
  const rmm::device_uvector<f_t>& get_solution() const;
  rmm::device_uvector<f_t>& get_solution();
  f_t get_objective_value() const;
  f_t get_mip_gap() const;
  f_t get_solution_bound() const;
  double get_total_solve_time() const;
  double get_presolve_time() const;
  mip_termination_status_t get_termination_status() const;
  static std::string get_termination_status_string(mip_termination_status_t termination_status);
  std::string get_termination_status_string() const;
  const cuopt::logic_error& get_error_status() const;
  f_t get_max_constraint_violation() const;
  f_t get_max_int_violation() const;
  f_t get_max_variable_bound_violation() const;
  i_t get_num_nodes() const;
  i_t get_num_simplex_iterations() const;
  const std::vector<std::string>& get_variable_names() const;
  const std::vector<rmm::device_uvector<f_t>>& get_solution_pool() const;
  void write_to_sol_file(std::string_view filename, rmm::cuda_stream_view stream_view) const;

 private:
  rmm::device_uvector<f_t> solution_;
  std::vector<std::string> var_names_;
  f_t objective_;
  f_t mip_gap_;
  mip_termination_status_t termination_status_;
  cuopt::logic_error error_status_;
  f_t max_constraint_violation_;
  f_t max_int_violation_;
  f_t max_variable_bound_violation_;
  solver_stats_t<i_t, f_t> stats_;
  std::vector<rmm::device_uvector<f_t>> solution_pool_;
};

}  // namespace cuopt::linear_programming
