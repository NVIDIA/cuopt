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

#include <cuopt/linear_programming/pdlp/solver_solution.hpp>
#include <cuopt/logger.hpp>
#include <math_optimization/solution_writer.hpp>
#include <mip/mip_constants.hpp>

#include <raft/common/nvtx.hpp>
#include <raft/util/cudart_utils.hpp>

#include <limits>
#include <vector>

namespace cuopt::linear_programming {

template <typename i_t, typename f_t>
optimization_problem_solution_t<i_t, f_t>::optimization_problem_solution_t(
  pdlp_termination_status_t termination_status, rmm::cuda_stream_view stream_view)
  : primal_solution_{0, stream_view},
    dual_solution_{0, stream_view},
    reduced_cost_{0, stream_view},
    termination_status_(termination_status),
    error_status_(cuopt::logic_error("", cuopt::error_type_t::Success))
{
}

template <typename i_t, typename f_t>
optimization_problem_solution_t<i_t, f_t>::optimization_problem_solution_t(
  cuopt::logic_error error_status_, rmm::cuda_stream_view stream_view)
  : primal_solution_{0, stream_view},
    dual_solution_{0, stream_view},
    reduced_cost_{0, stream_view},
    termination_status_(pdlp_termination_status_t::NoTermination),
    error_status_(error_status_)
{
}

template <typename i_t, typename f_t>
optimization_problem_solution_t<i_t, f_t>::optimization_problem_solution_t(
  rmm::device_uvector<f_t>& final_primal_solution,
  rmm::device_uvector<f_t>& final_dual_solution,
  rmm::device_uvector<f_t>& final_reduced_cost,
  pdlp_warm_start_data_t<i_t, f_t>& warm_start_data,
  const std::string objective_name,
  const std::vector<std::string>& var_names,
  const std::vector<std::string>& row_names,
  additional_termination_information_t& termination_stats,
  pdlp_termination_status_t termination_status)
  : primal_solution_(std::move(final_primal_solution)),
    dual_solution_(std::move(final_dual_solution)),
    reduced_cost_(std::move(final_reduced_cost)),
    pdlp_warm_start_data_(std::move(warm_start_data)),
    objective_name_(objective_name),
    var_names_(std::move(var_names)),
    row_names_(std::move(row_names)),
    termination_stats_(std::move(termination_stats)),
    termination_status_(termination_status),
    error_status_(cuopt::logic_error("", cuopt::error_type_t::Success))
{
}

template <typename i_t, typename f_t>
optimization_problem_solution_t<i_t, f_t>::optimization_problem_solution_t(
  rmm::device_uvector<f_t>& final_primal_solution,
  rmm::device_uvector<f_t>& final_dual_solution,
  rmm::device_uvector<f_t>& final_reduced_cost,
  const std::string objective_name,
  const std::vector<std::string>& var_names,
  const std::vector<std::string>& row_names,
  additional_termination_information_t& termination_stats,
  pdlp_termination_status_t termination_status)
  : primal_solution_(std::move(final_primal_solution)),
    dual_solution_(std::move(final_dual_solution)),
    reduced_cost_(std::move(final_reduced_cost)),
    objective_name_(objective_name),
    var_names_(std::move(var_names)),
    row_names_(std::move(row_names)),
    termination_stats_(std::move(termination_stats)),
    termination_status_(termination_status),
    error_status_(cuopt::logic_error("", cuopt::error_type_t::Success))
{
}

template <typename i_t, typename f_t>
optimization_problem_solution_t<i_t, f_t>::optimization_problem_solution_t(
  rmm::device_uvector<f_t>& final_primal_solution,
  rmm::device_uvector<f_t>& final_dual_solution,
  rmm::device_uvector<f_t>& final_reduced_cost,
  const std::string objective_name,
  const std::vector<std::string>& var_names,
  const std::vector<std::string>& row_names,
  additional_termination_information_t& termination_stats,
  pdlp_termination_status_t termination_status,
  const raft::handle_t* handler_ptr,
  [[maybe_unused]] bool deep_copy)
  : primal_solution_(final_primal_solution, handler_ptr->get_stream()),
    dual_solution_(final_dual_solution, handler_ptr->get_stream()),
    reduced_cost_(final_reduced_cost, handler_ptr->get_stream()),
    objective_name_(objective_name),
    var_names_(var_names),
    row_names_(row_names),
    termination_stats_(termination_stats),
    termination_status_(termination_status),
    error_status_(cuopt::logic_error("", cuopt::error_type_t::Success))
{
}

template <typename i_t, typename f_t>
void optimization_problem_solution_t<i_t, f_t>::copy_from(
  const raft::handle_t* handle_ptr, const optimization_problem_solution_t<i_t, f_t>& other)
{
  raft::copy(primal_solution_.data(),
             other.primal_solution_.data(),
             primal_solution_.size(),
             handle_ptr->get_stream());
  raft::copy(dual_solution_.data(),
             other.dual_solution_.data(),
             dual_solution_.size(),
             handle_ptr->get_stream());
  raft::copy(reduced_cost_.data(),
             other.reduced_cost_.data(),
             reduced_cost_.size(),
             handle_ptr->get_stream());
  termination_stats_  = other.termination_stats_;
  termination_status_ = other.termination_status_;
  objective_name_     = other.objective_name_;
  var_names_          = other.var_names_;
  row_names_          = other.row_names_;
  // We do not copy the warm start info. As it is not needed for this purpose.
  handle_ptr->sync_stream();
}

template <typename i_t, typename f_t>
void optimization_problem_solution_t<i_t, f_t>::write_additional_termination_statistics_to_file(
  std::ofstream& myfile)
{
  myfile << "\t\"Additional termination information\" : { " << std::endl;
  myfile << "\t\"Number of steps taken\" : " << termination_stats_.number_of_steps_taken << ","
         << std::endl;
  if (termination_stats_.solved_by_pdlp) {
    myfile << "\t\"Total number of attempted steps\" : "
           << termination_stats_.total_number_of_attempted_steps << "," << std::endl;
  }
  myfile << "\t\"Total solve time\" : " << termination_stats_.solve_time;
  if (termination_stats_.solved_by_pdlp) {
    myfile << "," << std::endl;
    myfile << "\t\t\"Convergence measures\" : { " << std::endl;
    myfile << "\t\t\t\"Absolute primal residual\" : " << termination_stats_.l2_primal_residual
           << "," << std::endl;
    myfile << "\t\t\t\"Relative primal residual\" : "
           << termination_stats_.l2_relative_primal_residual << "," << std::endl;
    myfile << "\t\t\t\"Absolute dual residual\" : " << termination_stats_.l2_dual_residual << ","
           << std::endl;
    myfile << "\t\t\t\"Relative dual residual\" : " << termination_stats_.l2_relative_dual_residual
           << "," << std::endl;
    myfile << "\t\t\t\"Primal objective value\" : " << termination_stats_.primal_objective << ","
           << std::endl;
    myfile << "\t\t\t\"Dual objective value\" : " << termination_stats_.dual_objective << ","
           << std::endl;
    myfile << "\t\t\t\"Gap\" : " << termination_stats_.gap << "," << std::endl;
    myfile << "\t\t\t\"Relative gap\" : " << termination_stats_.relative_gap << std::endl;
    myfile << "\t\t}, " << std::endl;
    myfile << "\t\t\"Infeasibility measures\" : {" << std::endl;
    myfile << "\t\t\t\"Maximum error for the linear constraints and sign constraints\" : "
           << termination_stats_.max_primal_ray_infeasibility << "," << std::endl;
    myfile << "\t\t\t\"Objective value for the extreme primal ray\" : "
           << termination_stats_.primal_ray_linear_objective << "," << std::endl;
    myfile << "\t\t\t\"Maximum constraint error\" : "
           << termination_stats_.max_dual_ray_infeasibility << "," << std::endl;
    myfile << "\t\t\t\"Objective value for the extreme dual ray\" : "
           << termination_stats_.dual_ray_linear_objective << std::endl;
    myfile << "\t\t} " << std::endl;
  } else
    myfile << std::endl;

  myfile << "\t} " << std::endl;
}

template <typename i_t, typename f_t>
void optimization_problem_solution_t<i_t, f_t>::write_to_file(std::string_view filename,
                                                              rmm::cuda_stream_view stream_view,
                                                              bool generate_variable_values)
{
  raft::common::nvtx::range fun_scope("write final solution to file");

  std::ofstream myfile(filename.data());
  myfile.precision(std::numeric_limits<f_t>::digits10 + 1);

  if (termination_status_ == pdlp_termination_status_t::NumericalError) {
    myfile << "{ " << std::endl;
    myfile << "\t\"Termination reason\" : \"" << get_termination_status_string() << "\"}"
           << std::endl;
    return;
  }
  std::vector<f_t> primal_solution;
  std::vector<f_t> dual_solution;
  std::vector<f_t> reduced_cost;
  primal_solution.resize(primal_solution_.size());
  dual_solution.resize(dual_solution_.size());
  reduced_cost.resize(reduced_cost_.size());
  raft::copy(
    primal_solution.data(), primal_solution_.data(), primal_solution_.size(), stream_view.value());
  raft::copy(
    dual_solution.data(), dual_solution_.data(), dual_solution_.size(), stream_view.value());
  raft::copy(reduced_cost.data(), reduced_cost_.data(), reduced_cost_.size(), stream_view.value());
  RAFT_CUDA_TRY(cudaStreamSynchronize(stream_view.value()));

  myfile << "{ " << std::endl;
  myfile << "\t\"Termination reason\" : \"" << get_termination_status_string() << "\","
         << std::endl;
  myfile << "\t\"Objective value for " << objective_name_ << "\" : " << get_objective_value() << ","
         << std::endl;
  if (!var_names_.empty() && generate_variable_values) {
    myfile << "\t\"Primal variables\" : {" << std::endl;
    for (size_t i = 0; i < primal_solution.size() - 1; i++) {
      myfile << "\t\t\"" << var_names_[i] << "\" : " << primal_solution[i] << "," << std::endl;
    }
    myfile << "\t\t\"" << var_names_[primal_solution.size() - 1]
           << "\" : " << primal_solution[primal_solution.size() - 1] << std::endl;
    myfile << "}, " << std::endl;
    myfile << "\t\"Dual variables\" : {" << std::endl;
    for (size_t i = 0; i < dual_solution.size() - 1; i++) {
      myfile << "\t\t\"" << row_names_[i] << "\" : " << dual_solution[i] << "," << std::endl;
    }
    myfile << "\t\t\"" << row_names_[dual_solution.size() - 1]
           << "\" : " << dual_solution[dual_solution.size() - 1] << std::endl;
    myfile << "\t}, " << std::endl;
    myfile << "\t\"Reduced costs\" : {" << std::endl;
    for (size_t i = 0; i < reduced_cost.size() - 1; i++) {
      myfile << "\t\t\"" << i << "\" : " << reduced_cost[i] << "," << std::endl;
    }
    myfile << "\t\t\"" << reduced_cost.size() - 1
           << "\" : " << reduced_cost[reduced_cost.size() - 1] << std::endl;
    myfile << "\t}, " << std::endl;
  }

  write_additional_termination_statistics_to_file(myfile);
  myfile << "} " << std::endl;

  myfile.close();
}

template <typename i_t, typename f_t>
void optimization_problem_solution_t<i_t, f_t>::set_solve_time(double ms)
{
  termination_stats_.solve_time = ms;
}

template <typename i_t, typename f_t>
void optimization_problem_solution_t<i_t, f_t>::set_termination_status(
  pdlp_termination_status_t termination_status)
{
  termination_status_ = termination_status;
}

template <typename i_t, typename f_t>
double optimization_problem_solution_t<i_t, f_t>::get_solve_time() const
{
  return termination_stats_.solve_time;
}

template <typename i_t, typename f_t>
std::string optimization_problem_solution_t<i_t, f_t>::get_termination_status_string(
  pdlp_termination_status_t termination_status)
{
  switch (termination_status) {
    case pdlp_termination_status_t::Optimal: return "Optimal";
    case pdlp_termination_status_t::PrimalInfeasible: return "Primal Infeasible";
    case pdlp_termination_status_t::DualInfeasible: return "Dual Infeasible";
    case pdlp_termination_status_t::IterationLimit: return "Iteration Limit";
    case pdlp_termination_status_t::TimeLimit: return "Time Limit";
    case pdlp_termination_status_t::NumericalError: return "A numerical error was encountered.";
    case pdlp_termination_status_t::PrimalFeasible: return "Primal Feasible";
    case pdlp_termination_status_t::ConcurrentLimit: return "Concurrent Limit";
    default: return "Unknown cuOpt status";
  }
}

template <typename i_t, typename f_t>
std::string optimization_problem_solution_t<i_t, f_t>::get_termination_status_string() const
{
  return get_termination_status_string(termination_status_);
}

template <typename i_t, typename f_t>
f_t optimization_problem_solution_t<i_t, f_t>::get_objective_value() const
{
  return termination_stats_.primal_objective;
}

template <typename i_t, typename f_t>
rmm::device_uvector<f_t>& optimization_problem_solution_t<i_t, f_t>::get_primal_solution()
{
  return primal_solution_;
}

template <typename i_t, typename f_t>
const rmm::device_uvector<f_t>& optimization_problem_solution_t<i_t, f_t>::get_primal_solution()
  const
{
  return primal_solution_;
}

template <typename i_t, typename f_t>
rmm::device_uvector<f_t>& optimization_problem_solution_t<i_t, f_t>::get_dual_solution()
{
  return dual_solution_;
}

template <typename i_t, typename f_t>
const rmm::device_uvector<f_t>& optimization_problem_solution_t<i_t, f_t>::get_dual_solution() const
{
  return dual_solution_;
}

template <typename i_t, typename f_t>
rmm::device_uvector<f_t>& optimization_problem_solution_t<i_t, f_t>::get_reduced_cost()
{
  return reduced_cost_;
}

template <typename i_t, typename f_t>
pdlp_termination_status_t optimization_problem_solution_t<i_t, f_t>::get_termination_status() const
{
  return termination_status_;
}

template <typename i_t, typename f_t>
cuopt::logic_error optimization_problem_solution_t<i_t, f_t>::get_error_status() const
{
  return error_status_;
}

template <typename i_t, typename f_t>
optimization_problem_solution_t<i_t, f_t>::additional_termination_information_t
optimization_problem_solution_t<i_t, f_t>::get_additional_termination_information() const
{
  return termination_stats_;
}

template <typename i_t, typename f_t>
pdlp_warm_start_data_t<i_t, f_t>&
optimization_problem_solution_t<i_t, f_t>::get_pdlp_warm_start_data()
{
  return pdlp_warm_start_data_;
}

template <typename i_t, typename f_t>
void optimization_problem_solution_t<i_t, f_t>::write_to_sol_file(
  std::string_view filename, rmm::cuda_stream_view stream_view) const
{
  auto status = get_termination_status_string();
  if (termination_status_ != pdlp_termination_status_t::Optimal &&
      termination_status_ != pdlp_termination_status_t::PrimalFeasible) {
    status = "Infeasible";
  }

  auto objective_value = get_objective_value();
  std::vector<f_t> solution;
  solution.resize(primal_solution_.size());
  raft::copy(
    solution.data(), primal_solution_.data(), primal_solution_.size(), stream_view.value());
  RAFT_CUDA_TRY(cudaStreamSynchronize(stream_view.value()));
  solution_writer_t::write_solution_to_sol_file(
    std::string(filename), status, objective_value, var_names_, solution);
}

#if MIP_INSTANTIATE_FLOAT
template class optimization_problem_solution_t<int, float>;
#endif

#if MIP_INSTANTIATE_DOUBLE
template class optimization_problem_solution_t<int, double>;
#endif
}  // namespace cuopt::linear_programming
