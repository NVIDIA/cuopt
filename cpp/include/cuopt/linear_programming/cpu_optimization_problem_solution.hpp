/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#pragma once

#include <cuopt/linear_programming/cpu_pdlp_warm_start_data.hpp>
#include <cuopt/linear_programming/mip/solver_solution.hpp>
#include <cuopt/linear_programming/mip/solver_stats.hpp>
#include <cuopt/linear_programming/optimization_problem_solution_interface.hpp>
#include <cuopt/linear_programming/pdlp/solver_solution.hpp>
#include <cuopt/linear_programming/utilities/cython_types.hpp>

#include <raft/core/copy.hpp>

#include <vector>

namespace cuopt::linear_programming {

/**
 * @brief CPU-backed LP solution (uses std::vector instead of rmm::device_uvector)
 *
 * This class stores solution data in host memory (std::vector).
 * Used for remote execution where GPU memory is not available.
 */
template <typename i_t, typename f_t>
class cpu_lp_solution_t : public lp_solution_interface_t<i_t, f_t> {
 public:
  // Bring base class overloads into scope to avoid hiding warnings
  using lp_solution_interface_t<i_t, f_t>::get_objective_value;
  using lp_solution_interface_t<i_t, f_t>::get_dual_objective_value;

  /**
   * @brief Construct an empty CPU LP solution (for errors)
   */
  cpu_lp_solution_t(pdlp_termination_status_t termination_status, cuopt::logic_error error_status)
    : termination_status_(termination_status),
      error_status_(error_status),
      solve_time_(0.0),
      primal_objective_(std::numeric_limits<f_t>::signaling_NaN()),
      dual_objective_(std::numeric_limits<f_t>::signaling_NaN()),
      l2_primal_residual_(std::numeric_limits<f_t>::signaling_NaN()),
      l2_dual_residual_(std::numeric_limits<f_t>::signaling_NaN()),
      gap_(std::numeric_limits<f_t>::signaling_NaN()),
      num_iterations_(0),
      solved_by_pdlp_(false)
  {
  }

  /**
   * @brief Construct CPU LP solution with complete termination info
   * Used for real remote execution when we have all the data
   */
  cpu_lp_solution_t(std::vector<f_t>&& primal_solution,
                    std::vector<f_t>&& dual_solution,
                    std::vector<f_t>&& reduced_cost,
                    pdlp_termination_status_t termination_status,
                    f_t primal_objective,
                    f_t dual_objective,
                    double solve_time,
                    f_t l2_primal_residual,
                    f_t l2_dual_residual,
                    f_t gap,
                    i_t num_iterations,
                    bool solved_by_pdlp)
    : primal_solution_(std::move(primal_solution)),
      dual_solution_(std::move(dual_solution)),
      reduced_cost_(std::move(reduced_cost)),
      termination_status_(termination_status),
      error_status_("", cuopt::error_type_t::Success),
      solve_time_(solve_time),
      primal_objective_(primal_objective),
      dual_objective_(dual_objective),
      l2_primal_residual_(l2_primal_residual),
      l2_dual_residual_(l2_dual_residual),
      gap_(gap),
      num_iterations_(num_iterations),
      solved_by_pdlp_(solved_by_pdlp)
  {
  }

  /**
   * @brief Construct CPU LP solution with complete data including warm start
   * Used for remote execution with warmstart support
   */
  cpu_lp_solution_t(std::vector<f_t>&& primal_solution,
                    std::vector<f_t>&& dual_solution,
                    std::vector<f_t>&& reduced_cost,
                    pdlp_termination_status_t termination_status,
                    f_t primal_objective,
                    f_t dual_objective,
                    double solve_time,
                    f_t l2_primal_residual,
                    f_t l2_dual_residual,
                    f_t gap,
                    i_t num_iterations,
                    bool solved_by_pdlp,
                    cpu_pdlp_warm_start_data_t<i_t, f_t>&& warmstart_data)
    : primal_solution_(std::move(primal_solution)),
      dual_solution_(std::move(dual_solution)),
      reduced_cost_(std::move(reduced_cost)),
      termination_status_(termination_status),
      error_status_("", cuopt::error_type_t::Success),
      solve_time_(solve_time),
      primal_objective_(primal_objective),
      dual_objective_(dual_objective),
      l2_primal_residual_(l2_primal_residual),
      l2_dual_residual_(l2_dual_residual),
      gap_(gap),
      num_iterations_(num_iterations),
      solved_by_pdlp_(solved_by_pdlp),
      pdlp_warm_start_data_(std::move(warmstart_data))
  {
  }

  // Host memory accessors (interface implementations)
  std::vector<f_t> get_primal_solution_host() const override { return primal_solution_; }
  std::vector<f_t> get_dual_solution_host() const override { return dual_solution_; }
  std::vector<f_t> get_reduced_cost_host() const override { return reduced_cost_; }

  // Interface implementations
  cuopt::logic_error get_error_status() const override { return error_status_; }

  f_t get_solve_time() const override { return solve_time_; }

  i_t get_primal_solution_size() const override { return primal_solution_.size(); }

  i_t get_dual_solution_size() const override { return dual_solution_.size(); }

  i_t get_reduced_cost_size() const override { return reduced_cost_.size(); }

  f_t get_objective_value(i_t = 0) const override { return primal_objective_; }

  f_t get_dual_objective_value(i_t = 0) const override { return dual_objective_; }

  pdlp_termination_status_t get_termination_status(i_t = 0) const override
  {
    return termination_status_;
  }

  f_t get_l2_primal_residual(i_t = 0) const override { return l2_primal_residual_; }

  f_t get_l2_dual_residual(i_t = 0) const override { return l2_dual_residual_; }

  f_t get_gap(i_t = 0) const override { return gap_; }

  i_t get_num_iterations(i_t = 0) const override { return num_iterations_; }

  bool is_solved_by_pdlp(i_t = 0) const override { return solved_by_pdlp_; }

  const pdlp_warm_start_data_t<i_t, f_t>& get_pdlp_warm_start_data() const override
  {
    throw cuopt::logic_error(
      "PDLP warm start data not available for CPU solutions (use individual accessors)",
      cuopt::error_type_t::RuntimeError);
  }

  bool has_warm_start_data() const override { return pdlp_warm_start_data_.is_populated(); }

  // Warmstart data accessor - returns the CPU warmstart struct
  const cpu_pdlp_warm_start_data_t<i_t, f_t>& get_cpu_pdlp_warm_start_data() const
  {
    return pdlp_warm_start_data_;
  }

  cpu_pdlp_warm_start_data_t<i_t, f_t>& get_cpu_pdlp_warm_start_data()
  {
    return pdlp_warm_start_data_;
  }

  // Individual warm start data accessors (return stored host vectors)
  std::vector<f_t> get_current_primal_solution_host() const override
  {
    return pdlp_warm_start_data_.current_primal_solution_;
  }
  std::vector<f_t> get_current_dual_solution_host() const override
  {
    return pdlp_warm_start_data_.current_dual_solution_;
  }
  std::vector<f_t> get_initial_primal_average_host() const override
  {
    return pdlp_warm_start_data_.initial_primal_average_;
  }
  std::vector<f_t> get_initial_dual_average_host() const override
  {
    return pdlp_warm_start_data_.initial_dual_average_;
  }
  std::vector<f_t> get_current_ATY_host() const override
  {
    return pdlp_warm_start_data_.current_ATY_;
  }
  std::vector<f_t> get_sum_primal_solutions_host() const override
  {
    return pdlp_warm_start_data_.sum_primal_solutions_;
  }
  std::vector<f_t> get_sum_dual_solutions_host() const override
  {
    return pdlp_warm_start_data_.sum_dual_solutions_;
  }
  std::vector<f_t> get_last_restart_duality_gap_primal_solution_host() const override
  {
    return pdlp_warm_start_data_.last_restart_duality_gap_primal_solution_;
  }
  std::vector<f_t> get_last_restart_duality_gap_dual_solution_host() const override
  {
    return pdlp_warm_start_data_.last_restart_duality_gap_dual_solution_;
  }
  f_t get_initial_primal_weight() const override
  {
    return pdlp_warm_start_data_.initial_primal_weight_;
  }
  f_t get_initial_step_size() const override { return pdlp_warm_start_data_.initial_step_size_; }
  i_t get_total_pdlp_iterations() const override
  {
    return pdlp_warm_start_data_.total_pdlp_iterations_;
  }
  i_t get_total_pdhg_iterations() const override
  {
    return pdlp_warm_start_data_.total_pdhg_iterations_;
  }
  f_t get_last_candidate_kkt_score() const override
  {
    return pdlp_warm_start_data_.last_candidate_kkt_score_;
  }
  f_t get_last_restart_kkt_score() const override
  {
    return pdlp_warm_start_data_.last_restart_kkt_score_;
  }
  f_t get_sum_solution_weight() const override
  {
    return pdlp_warm_start_data_.sum_solution_weight_;
  }
  i_t get_iterations_since_last_restart() const override
  {
    return pdlp_warm_start_data_.iterations_since_last_restart_;
  }

  /**
   * @brief Convert CPU solution to GPU solution
   * Copies data from host (std::vector) to device (rmm::device_uvector)
   */
  optimization_problem_solution_t<i_t, f_t> to_gpu_solution(
    rmm::cuda_stream_view stream_view) override
  {
    // Create device vectors and copy data from host
    rmm::device_uvector<f_t> primal_device(primal_solution_.size(), stream_view);
    rmm::device_uvector<f_t> dual_device(dual_solution_.size(), stream_view);
    rmm::device_uvector<f_t> reduced_cost_device(reduced_cost_.size(), stream_view);

    if (!primal_solution_.empty()) {
      raft::copy(
        primal_device.data(), primal_solution_.data(), primal_solution_.size(), stream_view);
    }
    if (!dual_solution_.empty()) {
      raft::copy(dual_device.data(), dual_solution_.data(), dual_solution_.size(), stream_view);
    }
    if (!reduced_cost_.empty()) {
      raft::copy(
        reduced_cost_device.data(), reduced_cost_.data(), reduced_cost_.size(), stream_view);
    }

    using additional_info_t =
      typename optimization_problem_solution_t<i_t, f_t>::additional_termination_information_t;
    std::vector<additional_info_t> termination_stats(1);
    termination_stats[0].primal_objective      = primal_objective_;
    termination_stats[0].dual_objective        = dual_objective_;
    termination_stats[0].solve_time            = solve_time_;
    termination_stats[0].l2_primal_residual    = l2_primal_residual_;
    termination_stats[0].l2_dual_residual      = l2_dual_residual_;
    termination_stats[0].gap                   = gap_;
    termination_stats[0].number_of_steps_taken = num_iterations_;
    termination_stats[0].solved_by_pdlp        = solved_by_pdlp_;

    std::vector<pdlp_termination_status_t> termination_status_vec = {termination_status_};

    // Convert CPU warmstart to GPU warmstart
    auto gpu_warmstart = convert_to_gpu_warmstart(pdlp_warm_start_data_, stream_view);

    // Create GPU solution
    return optimization_problem_solution_t<i_t, f_t>(primal_device,
                                                     dual_device,
                                                     reduced_cost_device,
                                                     std::move(gpu_warmstart),
                                                     "",  // objective_name
                                                     {},  // var_names
                                                     {},  // row_names
                                                     std::move(termination_stats),
                                                     std::move(termination_status_vec));
  }

  /**
   * @brief Convert to CPU-backed linear_programming_ret_t struct for Python/Cython
   * Populates the cpu_solutions_t variant.  Moves std::vector data with zero-copy.
   */
  cuopt::cython::linear_programming_ret_t to_cpu_linear_programming_ret_t();

  /**
   * @brief Polymorphic conversion to Python return type (interface override)
   * Populates the cpu_solutions_t variant inside linear_programming_ret_t.
   */
  cuopt::cython::linear_programming_ret_t to_python_lp_ret() override
  {
    return to_cpu_linear_programming_ret_t();
  }

 private:
  std::vector<f_t> primal_solution_;
  std::vector<f_t> dual_solution_;
  std::vector<f_t> reduced_cost_;
  pdlp_termination_status_t termination_status_;
  cuopt::logic_error error_status_;
  double solve_time_;
  f_t primal_objective_;
  f_t dual_objective_;
  f_t l2_primal_residual_;
  f_t l2_dual_residual_;
  f_t gap_;
  i_t num_iterations_;
  bool solved_by_pdlp_;

  // PDLP warm start data (embedded struct, CPU-backed using std::vector)
  cpu_pdlp_warm_start_data_t<i_t, f_t> pdlp_warm_start_data_;
};

/**
 * @brief CPU-backed MIP solution (uses std::vector instead of rmm::device_uvector)
 *
 * This class stores solution data in host memory (std::vector).
 * Used for remote execution where GPU memory is not available.
 */
template <typename i_t, typename f_t>
class cpu_mip_solution_t : public mip_solution_interface_t<i_t, f_t> {
 public:
  /**
   * @brief Construct an empty CPU MIP solution (for errors)
   */
  cpu_mip_solution_t(mip_termination_status_t termination_status, cuopt::logic_error error_status)
    : termination_status_(termination_status),
      error_status_(error_status),
      objective_(std::numeric_limits<f_t>::signaling_NaN()),
      mip_gap_(std::numeric_limits<f_t>::signaling_NaN()),
      solution_bound_(std::numeric_limits<f_t>::signaling_NaN()),
      total_solve_time_(0.0),
      presolve_time_(0.0),
      max_constraint_violation_(std::numeric_limits<f_t>::signaling_NaN()),
      max_int_violation_(std::numeric_limits<f_t>::signaling_NaN()),
      max_variable_bound_violation_(std::numeric_limits<f_t>::signaling_NaN()),
      num_nodes_(0),
      num_simplex_iterations_(0)
  {
  }

  /**
   * @brief Construct CPU MIP solution with data
   */
  cpu_mip_solution_t(std::vector<f_t>&& solution,
                     mip_termination_status_t termination_status,
                     f_t objective,
                     f_t mip_gap,
                     f_t solution_bound,
                     double total_solve_time,
                     double presolve_time,
                     f_t max_constraint_violation,
                     f_t max_int_violation,
                     f_t max_variable_bound_violation,
                     i_t num_nodes,
                     i_t num_simplex_iterations)
    : solution_(std::move(solution)),
      termination_status_(termination_status),
      error_status_("", cuopt::error_type_t::Success),
      objective_(objective),
      mip_gap_(mip_gap),
      solution_bound_(solution_bound),
      total_solve_time_(total_solve_time),
      presolve_time_(presolve_time),
      max_constraint_violation_(max_constraint_violation),
      max_int_violation_(max_int_violation),
      max_variable_bound_violation_(max_variable_bound_violation),
      num_nodes_(num_nodes),
      num_simplex_iterations_(num_simplex_iterations)
  {
  }

  // Host memory accessor (interface implementation)
  std::vector<f_t> get_solution_host() const override { return solution_; }

  // Interface implementations
  cuopt::logic_error get_error_status() const override { return error_status_; }

  f_t get_solve_time() const override { return total_solve_time_; }

  i_t get_solution_size() const override { return solution_.size(); }

  f_t get_objective_value() const override { return objective_; }

  f_t get_mip_gap() const override { return mip_gap_; }

  f_t get_solution_bound() const override { return solution_bound_; }

  mip_termination_status_t get_termination_status() const override { return termination_status_; }

  f_t get_presolve_time() const override { return presolve_time_; }

  f_t get_max_constraint_violation() const override { return max_constraint_violation_; }

  f_t get_max_int_violation() const override { return max_int_violation_; }

  f_t get_max_variable_bound_violation() const override { return max_variable_bound_violation_; }

  i_t get_num_nodes() const override { return num_nodes_; }

  i_t get_num_simplex_iterations() const override { return num_simplex_iterations_; }

  /**
   * @brief Convert CPU solution to GPU solution
   * Copies data from host (std::vector) to device (rmm::device_uvector)
   */
  mip_solution_t<i_t, f_t> to_gpu_solution(rmm::cuda_stream_view stream_view) override
  {
    // Create device vector and copy data from host
    rmm::device_uvector<f_t> solution_device(solution_.size(), stream_view);

    if (!solution_.empty()) {
      raft::copy(solution_device.data(), solution_.data(), solution_.size(), stream_view);
    }

    // Create solver stats
    solver_stats_t<i_t, f_t> stats;
    stats.total_solve_time       = total_solve_time_;
    stats.presolve_time          = presolve_time_;
    stats.solution_bound         = solution_bound_;
    stats.num_nodes              = num_nodes_;
    stats.num_simplex_iterations = num_simplex_iterations_;

    // Create GPU solution
    return mip_solution_t<i_t, f_t>(std::move(solution_device),
                                    {},  // var_names
                                    objective_,
                                    mip_gap_,
                                    termination_status_,
                                    max_constraint_violation_,
                                    max_int_violation_,
                                    max_variable_bound_violation_,
                                    stats);
  }

  /**
   * @brief Convert to CPU-backed mip_ret_t struct for Python/Cython
   * Populates the cpu_buffer variant.  Moves std::vector data with zero-copy.
   */
  cuopt::cython::mip_ret_t to_cpu_mip_ret_t();

  /**
   * @brief Polymorphic conversion to Python return type (interface override)
   * Populates the cpu_buffer variant inside mip_ret_t.
   */
  cuopt::cython::mip_ret_t to_python_mip_ret() override { return to_cpu_mip_ret_t(); }

 private:
  std::vector<f_t> solution_;
  mip_termination_status_t termination_status_;
  cuopt::logic_error error_status_;
  f_t objective_;
  f_t mip_gap_;
  f_t solution_bound_;
  double total_solve_time_;
  double presolve_time_;
  f_t max_constraint_violation_;
  f_t max_int_violation_;
  f_t max_variable_bound_violation_;
  i_t num_nodes_;
  i_t num_simplex_iterations_;
};

}  // namespace cuopt::linear_programming
