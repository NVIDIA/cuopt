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
#include <cuopt/linear_programming/pdlp/pdlp_warm_start_data.hpp>
#include <cuopt/linear_programming/utilities/internals.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_uvector.hpp>

#include <raft/core/handle.hpp>

#include <fstream>
#include <string>
#include <vector>

namespace cuopt::linear_programming {

// Possible reasons for terminating
enum class pdlp_termination_status_t : int8_t {
  NoTermination    = CUOPT_TERIMINATION_STATUS_NO_TERMINATION,
  NumericalError   = CUOPT_TERIMINATION_STATUS_NUMERICAL_ERROR,
  Optimal          = CUOPT_TERIMINATION_STATUS_OPTIMAL,
  PrimalInfeasible = CUOPT_TERIMINATION_STATUS_INFEASIBLE,
  DualInfeasible   = CUOPT_TERIMINATION_STATUS_UNBOUNDED,
  IterationLimit   = CUOPT_TERIMINATION_STATUS_ITERATION_LIMIT,
  TimeLimit        = CUOPT_TERIMINATION_STATUS_TIME_LIMIT,
  PrimalFeasible   = CUOPT_TERIMINATION_STATUS_PRIMAL_FEASIBLE,
  ConcurrentLimit  = CUOPT_TERIMINATION_STATUS_CONCURRENT_LIMIT
};

/**
 * @brief A container of PDLP solver output
 * @tparam i_t Integer type. Currently only int is supported.
 * @tparam f_t Floating point type. Currently only float (32bit) and double (64bit) are supported.
 */
template <typename i_t, typename f_t>
class optimization_problem_solution_t : public base_solution_t {
 public:
  bool is_mip() const override { return false; }
  /**
   * @brief Structure containing additional termination information such as the number of steps,
   * objective and residual values
   *
   */
  struct additional_termination_information_t {
    /** Number of pdlp steps taken before termination */
    i_t number_of_steps_taken;
    /** Number of pdhg steps taken before termination */
    i_t total_number_of_attempted_steps;
    /** L2 norm of the primal residual (absolute primal residual) */
    f_t l2_primal_residual;
    /** L2 norm of the primal residual divided by the L2 norm of the right hand side (b) */
    f_t l2_relative_primal_residual;
    /** L2 norm of the dual residual */
    f_t l2_dual_residual;
    /** L2 norm of the dual residual divided by the L2 norm of the objective coefficient (c) */
    f_t l2_relative_dual_residual;

    /** Primal Objective */
    f_t primal_objective;
    /** Dual Objective */
    f_t dual_objective;

    /** Gap between primal and dual objective value */

    f_t gap;
    /** Gap divided by the absolute sum of the primal and dual objective values */
    f_t relative_gap;

    /** Maximum error for the linear constraints and sign constraints */
    f_t max_primal_ray_infeasibility;
    /** Objective value for the extreme primal ray */
    f_t primal_ray_linear_objective;

    /** Maximum constraint error */
    f_t max_dual_ray_infeasibility;
    /** Objective value for the extreme dual ray */
    f_t dual_ray_linear_objective;

    /** Solve time in seconds */
    double solve_time;
  };

  /**
   * @brief Construct an optimization problem solution that serves as PDLP solver output
   * Used when has not converged or to build the best so far object
   *
   * @param[in] termination_status_ Reason for termination. Possible values are : 'NumericalError'
   * 'Optimal', 'PrimalInfeasible', 'DualInfeasible', 'TimeLimit'
   * @param[in] stream_view An rmm view to a stream. All computations will go through this stream
   */
  optimization_problem_solution_t(pdlp_termination_status_t termination_status_,
                                  rmm::cuda_stream_view stream_view);

  /**
   * @brief Construct an optimization problem solution that serves as PDLP solver output
   * Used when an internal error has occurred
   *
   * @param[in] error_status_ The error object, containing info about what went wrong
   * 'Optimal', 'PrimalInfeasible', 'DualInfeasible', 'TimeLimit'
   * @param[in] stream_view An rmm view to a stream. All computations will go through this stream
   */
  optimization_problem_solution_t(cuopt::logic_error error_status_,
                                  rmm::cuda_stream_view stream_view);
  /**
   * @brief Construct an optimization problem solution that serves as PDLP solver output
   *
   * @param[in] final_primal_solution The final primal solution
   * @param[in] final_dual_solution The final dual solution
   * @param[in] final_reduced_cost The final reduced cost
   * @param[in] objective_name The objective name
   * @param[in] var_names The variables names
   * @param[in] row_names The rows name
   * @param[in] termination_stats The termination statistics
   * @param[in] termination_status_ The termination reason
   */
  optimization_problem_solution_t(rmm::device_uvector<f_t>& final_primal_solution,
                                  rmm::device_uvector<f_t>& final_dual_solution,
                                  rmm::device_uvector<f_t>& final_reduced_cost,
                                  pdlp_warm_start_data_t<i_t, f_t>& warm_start_data,
                                  const std::string objective_name,
                                  const std::vector<std::string>& var_names,
                                  const std::vector<std::string>& row_names,
                                  additional_termination_information_t& termination_stats,
                                  pdlp_termination_status_t termination_status_);

  optimization_problem_solution_t(rmm::device_uvector<f_t>& final_primal_solution,
                                  rmm::device_uvector<f_t>& final_dual_solution,
                                  rmm::device_uvector<f_t>& final_reduced_cost,
                                  const std::string objective_name,
                                  const std::vector<std::string>& var_names,
                                  const std::vector<std::string>& row_names,
                                  additional_termination_information_t& termination_stats,
                                  pdlp_termination_status_t termination_status_);

  /**
   * @brief Construct variant used in best_primal_so_far to do a deep copy instead of move since we
   * need to keep the results in the solver
   *
   * @param[in] final_primal_solution The final primal solution
   * @param[in] final_dual_solution The final dual solution
   * @param[in] final_reduced_cost The final reduced cost
   * @param[in] objective_name The objective name
   * @param[in] var_names The variables names
   * @param[in] row_names The rows name
   * @param[in] termination_stats The termination statistics
   * @param[in] termination_status_ The termination reason
   */
  optimization_problem_solution_t(rmm::device_uvector<f_t>& final_primal_solution,
                                  rmm::device_uvector<f_t>& final_dual_solution,
                                  rmm::device_uvector<f_t>& final_reduced_cost,
                                  const std::string objective_name,
                                  const std::vector<std::string>& var_names,
                                  const std::vector<std::string>& row_names,
                                  additional_termination_information_t& termination_stats,
                                  pdlp_termination_status_t termination_status,
                                  const raft::handle_t* handler_ptr,
                                  bool deep_copy);

  /**
   * @brief Set the solve time in seconds
   *
   * @param ms Time in ms
   */
  void set_solve_time(double ms);

  /**
   * @brief Set the termination reason
   *
   * @param termination_status termination reason
   */
  void set_termination_status(pdlp_termination_status_t termination_status);

  /**
   * @brief Get the solve time in seconds
   *
   * @return Time in seconds
   */
  double get_solve_time() const;

  /**
   * @brief Returns the final status as a human readable string
   * @return The human readable solver status string
   */
  std::string get_termination_status_string() const;
  static std::string get_termination_status_string(pdlp_termination_status_t termination_status);

  /**
   * @brief Returns the objective value of the solution as a `f_t`. The objective value is
   * calculated based on the user provided objective function and the primal solution found by the
   * solver.
   * @return Best objective value
   */
  f_t get_objective_value() const;

  /**
   * @brief Returns the solution for the values of the primal variables as a vector of `f_t`.
   *
   * @return rmm::device_uvector<i_t> The device memory container for the primal solution.
   */
  rmm::device_uvector<f_t>& get_primal_solution();
  const rmm::device_uvector<f_t>& get_primal_solution() const;

  /**
   * @brief Returns the solution for the values of the dual variables as a vector of `f_t`.
   *
   * @return rmm::device_uvector<i_t> The device memory container for the dual solution.
   */
  rmm::device_uvector<f_t>& get_dual_solution();
  const rmm::device_uvector<f_t>& get_dual_solution() const;

  /**
   * @brief Returns the reduced cost as a vector of `f_t`. The reduced cost contains the dual
   * multipliers for the linear constraints.
   *
   * @return rmm::device_uvector<i_t> The device memory container for the reduced cost.
   */
  rmm::device_uvector<f_t>& get_reduced_cost();

  /**
   * @brief Get termination reason
   * @return Termination reason
   */
  pdlp_termination_status_t get_termination_status() const;

  /**
   * @brief Get the error status
   * @return The error status
   */
  cuopt::logic_error get_error_status() const;

  /**
   * @brief Get the additional_termination_information_t object which contains various measures and
   * statistics regarding the solution and solver state at the end of solving.
   * @return Additional termination information
   */
  additional_termination_information_t get_additional_termination_information() const;

  pdlp_warm_start_data_t<i_t, f_t>& get_pdlp_warm_start_data();

  /**
   * @brief Writes the solver_solution object as a JSON object to the 'filename' file using
   * 'stream_view' to transfer the data from device to host before it is written to the file.
   * @param filename Name of the output file
   * @param stream_view Non-owning stream view object
   */
  void write_to_file(std::string_view filename,
                     rmm::cuda_stream_view stream_view,
                     bool generate_variable_values = true);

  /**
   * @brief Writes the solver_solution object as a '.sol' file as supported by other solvers and
   * used in MIPLIB using 'stream_view' to transfer the data from device to host before it is
   * written to the file.
   * @param filename Name of the output file
   * @param stream_view Non-owning stream view object
   */
  void write_to_sol_file(std::string_view filename, rmm::cuda_stream_view stream_view) const;

  /**
   * @brief Copy solution from another solution object
   * @param handle_ptr The handle pointer
   * @param other The other solution object
   */
  void copy_from(const raft::handle_t* handle_ptr,
                 const optimization_problem_solution_t<i_t, f_t>& other);

 private:
  void write_additional_termination_statistics_to_file(std::ofstream& myfile);

  rmm::device_uvector<f_t> primal_solution_;
  rmm::device_uvector<f_t> dual_solution_;
  rmm::device_uvector<f_t> reduced_cost_;
  pdlp_warm_start_data_t<i_t, f_t> pdlp_warm_start_data_;

  pdlp_termination_status_t termination_status_;

  additional_termination_information_t termination_stats_;

  /** name of the objective (only a single objective is currently allowed) */
  std::string objective_name_;
  /** names of each of the variables in the OP */
  std::vector<std::string> var_names_{};
  /** names of each of the rows in the OP */
  std::vector<std::string> row_names_{};
  /** error struct */
  cuopt::logic_error error_status_;
};
}  // namespace cuopt::linear_programming
