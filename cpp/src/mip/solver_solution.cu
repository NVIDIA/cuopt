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

#include <cuopt/linear_programming/mip/solver_solution.hpp>
#include <cuopt/logger.hpp>
#include <mip/mip_constants.hpp>

#include <limits>
#include <math_optimization/solution_writer.hpp>
#include <raft/common/nvtx.hpp>
#include <raft/util/cudart_utils.hpp>
#include <vector>

namespace cuopt::linear_programming {

template <typename i_t, typename f_t>
mip_solution_t<i_t, f_t>::mip_solution_t(rmm::device_uvector<f_t> solution,
                                         std::vector<std::string> var_names,
                                         f_t objective,
                                         f_t mip_gap,
                                         mip_termination_status_t termination_status,
                                         f_t max_constraint_violation,
                                         f_t max_int_violation,
                                         f_t max_variable_bound_violation,
                                         solver_stats_t<i_t, f_t> stats,
                                         std::vector<rmm::device_uvector<f_t>> solution_pool)
  : solution_(std::move(solution)),
    var_names_(std::move(var_names)),
    objective_(objective),
    mip_gap_(mip_gap),
    termination_status_(termination_status),
    max_constraint_violation_(max_constraint_violation),
    max_int_violation_(max_int_violation),
    max_variable_bound_violation_(max_variable_bound_violation),
    stats_(stats),
    solution_pool_(std::move(solution_pool)),
    error_status_(cuopt::logic_error("", cuopt::error_type_t::Success))
{
}

template <typename i_t, typename f_t>
mip_solution_t<i_t, f_t>::mip_solution_t(mip_termination_status_t termination_status,
                                         solver_stats_t<i_t, f_t> stats,
                                         rmm::cuda_stream_view stream_view)
  : solution_(0, stream_view),
    objective_(0),
    mip_gap_(0),
    termination_status_(termination_status),
    max_constraint_violation_(0),
    max_int_violation_(0),
    max_variable_bound_violation_(0),
    stats_(stats),
    error_status_(cuopt::logic_error("", cuopt::error_type_t::Success))
{
}

template <typename i_t, typename f_t>
mip_solution_t<i_t, f_t>::mip_solution_t(const cuopt::logic_error& error_status,
                                         rmm::cuda_stream_view stream_view)
  : solution_(0, stream_view),
    objective_(0),
    mip_gap_(0),
    termination_status_(mip_termination_status_t::NoTermination),
    max_constraint_violation_(0),
    max_int_violation_(0),
    max_variable_bound_violation_(0),
    error_status_(error_status)
{
}

template <typename i_t, typename f_t>
const cuopt::logic_error& mip_solution_t<i_t, f_t>::get_error_status() const
{
  return error_status_;
}

template <typename i_t, typename f_t>
const rmm::device_uvector<f_t>& mip_solution_t<i_t, f_t>::get_solution() const
{
  return solution_;
}

template <typename i_t, typename f_t>
rmm::device_uvector<f_t>& mip_solution_t<i_t, f_t>::get_solution()
{
  return solution_;
}

template <typename i_t, typename f_t>
f_t mip_solution_t<i_t, f_t>::get_objective_value() const
{
  return objective_;
}

template <typename i_t, typename f_t>
f_t mip_solution_t<i_t, f_t>::get_mip_gap() const
{
  return mip_gap_;
}

template <typename i_t, typename f_t>
f_t mip_solution_t<i_t, f_t>::get_solution_bound() const
{
  return stats_.solution_bound;
}

template <typename i_t, typename f_t>
double mip_solution_t<i_t, f_t>::get_total_solve_time() const
{
  return stats_.total_solve_time;
}

template <typename i_t, typename f_t>
double mip_solution_t<i_t, f_t>::get_presolve_time() const
{
  return stats_.presolve_time;
}

template <typename i_t, typename f_t>
mip_termination_status_t mip_solution_t<i_t, f_t>::get_termination_status() const
{
  return termination_status_;
}

template <typename i_t, typename f_t>
std::string mip_solution_t<i_t, f_t>::get_termination_status_string(
  mip_termination_status_t termination_status)
{
  switch (termination_status) {
    case mip_termination_status_t::NoTermination: return "NoTermination";
    case mip_termination_status_t::Optimal: return "Optimal";
    case mip_termination_status_t::FeasibleFound: return "FeasibleFound";
    case mip_termination_status_t::Infeasible: return "Infeasible";
    case mip_termination_status_t::TimeLimit: return "TimeLimit";
    case mip_termination_status_t::Unbounded:
      return "Unbounded";
      // Do not implement default case to trigger compile time error if new enum is added
  }
  return std::string();
}

template <typename i_t, typename f_t>
std::string mip_solution_t<i_t, f_t>::get_termination_status_string() const
{
  return get_termination_status_string(termination_status_);
}

template <typename i_t, typename f_t>
f_t mip_solution_t<i_t, f_t>::get_max_constraint_violation() const
{
  return max_constraint_violation_;
}

template <typename i_t, typename f_t>
f_t mip_solution_t<i_t, f_t>::get_max_int_violation() const
{
  return max_int_violation_;
}

template <typename i_t, typename f_t>
f_t mip_solution_t<i_t, f_t>::get_max_variable_bound_violation() const
{
  return max_variable_bound_violation_;
}

template <typename i_t, typename f_t>
solver_stats_t<i_t, f_t> mip_solution_t<i_t, f_t>::get_stats() const
{
  return stats_;
}

template <typename i_t, typename f_t>
i_t mip_solution_t<i_t, f_t>::get_num_nodes() const
{
  return stats_.num_nodes;
}

template <typename i_t, typename f_t>
i_t mip_solution_t<i_t, f_t>::get_num_simplex_iterations() const
{
  return stats_.num_simplex_iterations;
}

template <typename i_t, typename f_t>
const std::vector<std::string>& mip_solution_t<i_t, f_t>::get_variable_names() const
{
  return var_names_;
}

template <typename i_t, typename f_t>
const std::vector<rmm::device_uvector<f_t>>& mip_solution_t<i_t, f_t>::get_solution_pool() const
{
  return solution_pool_;
}

template <typename i_t, typename f_t>
void mip_solution_t<i_t, f_t>::write_to_sol_file(std::string_view filename,
                                                 rmm::cuda_stream_view stream_view) const
{
  std::string status = get_termination_status_string();
  // Override for no termination
  if (termination_status_ == mip_termination_status_t::NoTermination ||
      termination_status_ == mip_termination_status_t::Infeasible) {
    status = "Infeasible";
  }

  double objective_value = get_objective_value();
  auto& var_names        = get_variable_names();
  std::vector<f_t> solution;
  solution.resize(solution_.size());
  raft::copy(solution.data(), solution_.data(), solution_.size(), stream_view.value());
  RAFT_CUDA_TRY(cudaStreamSynchronize(stream_view.value()));

  solution_writer_t::write_solution_to_sol_file(
    std::string(filename), status, objective_value, var_names, solution);
}

template <typename i_t, typename f_t>
void mip_solution_t<i_t, f_t>::log_summary() const
{
  CUOPT_LOG_INFO("Termination Status: {}", get_termination_status_string());
  CUOPT_LOG_INFO("Objective Value: %f", get_objective_value());
  CUOPT_LOG_INFO("Max constraint violation: %f", get_max_constraint_violation());
  CUOPT_LOG_INFO("Max integer violation: %f", get_max_int_violation());
  CUOPT_LOG_INFO("Max variable bound violation: %f", get_max_variable_bound_violation());
  CUOPT_LOG_INFO("MIP Gap: %f", get_mip_gap());
  CUOPT_LOG_INFO("Solution Bound: %f", get_solution_bound());
  CUOPT_LOG_INFO("Presolve Time: %f", get_presolve_time());
  CUOPT_LOG_INFO("Total Solve Time: %f", get_total_solve_time());
}

#if MIP_INSTANTIATE_FLOAT
template class mip_solution_t<int, float>;
#endif

#if MIP_INSTANTIATE_DOUBLE
template class mip_solution_t<int, double>;
#endif
}  // namespace cuopt::linear_programming
