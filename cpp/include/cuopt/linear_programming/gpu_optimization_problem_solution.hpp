/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#pragma once

#include <cuopt/linear_programming/cpu_optimization_problem_solution.hpp>
#include <cuopt/linear_programming/mip/solver_solution.hpp>
#include <cuopt/linear_programming/optimization_problem_solution_interface.hpp>
#include <cuopt/linear_programming/pdlp/solver_solution.hpp>
#include <cuopt/linear_programming/utilities/cython_types.hpp>

#include <raft/core/copy.hpp>
#include <rmm/cuda_stream_view.hpp>

namespace cuopt::linear_programming {

/**
 * @brief GPU-backed LP solution (wraps optimization_problem_solution_t)
 *
 * This class wraps the existing optimization_problem_solution_t which uses GPU memory.
 * It implements the interface to allow polymorphism with CPU solutions.
 */
template <typename i_t, typename f_t>
class gpu_lp_solution_t : public lp_solution_interface_t<i_t, f_t> {
 public:
  // Bring base class overloads into scope to avoid hiding warnings
  using lp_solution_interface_t<i_t, f_t>::get_objective_value;
  using lp_solution_interface_t<i_t, f_t>::get_dual_objective_value;

  /**
   * @brief Construct from existing optimization_problem_solution_t (move)
   */
  explicit gpu_lp_solution_t(optimization_problem_solution_t<i_t, f_t>&& solution)
    : solution_(std::move(solution))
  {
  }

  // Interface implementations
  cuopt::logic_error get_error_status() const override { return solution_.get_error_status(); }

  f_t get_solve_time() const override { return static_cast<f_t>(solution_.get_solve_time()); }

  i_t get_primal_solution_size() const override { return solution_.get_primal_solution().size(); }

  i_t get_dual_solution_size() const override { return solution_.get_dual_solution().size(); }

  i_t get_reduced_cost_size() const override
  {
    return const_cast<optimization_problem_solution_t<i_t, f_t>&>(solution_)
      .get_reduced_cost()
      .size();
  }

  std::vector<f_t> get_primal_solution_host() const override
  {
    auto stream = solution_.get_primal_solution().stream();
    std::vector<f_t> result(solution_.get_primal_solution().size());
    raft::copy(result.data(),
               solution_.get_primal_solution().data(),
               solution_.get_primal_solution().size(),
               stream);
    stream.synchronize();
    return result;
  }

  std::vector<f_t> get_dual_solution_host() const override
  {
    auto stream = solution_.get_dual_solution().stream();
    std::vector<f_t> result(solution_.get_dual_solution().size());
    raft::copy(result.data(),
               solution_.get_dual_solution().data(),
               solution_.get_dual_solution().size(),
               stream);
    stream.synchronize();
    return result;
  }

  std::vector<f_t> get_reduced_cost_host() const override
  {
    auto& reduced_cost =
      const_cast<optimization_problem_solution_t<i_t, f_t>&>(solution_).get_reduced_cost();
    auto stream = reduced_cost.stream();
    std::vector<f_t> result(reduced_cost.size());
    raft::copy(result.data(), reduced_cost.data(), reduced_cost.size(), stream);
    stream.synchronize();
    return result;
  }

  f_t get_objective_value(i_t id = 0) const override { return solution_.get_objective_value(id); }

  f_t get_dual_objective_value(i_t id = 0) const override
  {
    return solution_.get_dual_objective_value(id);
  }

  pdlp_termination_status_t get_termination_status(i_t id = 0) const override
  {
    return solution_.get_termination_status(id);
  }

  f_t get_l2_primal_residual(i_t id = 0) const override
  {
    return solution_.get_additional_termination_information(id).l2_primal_residual;
  }

  f_t get_l2_dual_residual(i_t id = 0) const override
  {
    return solution_.get_additional_termination_information(id).l2_dual_residual;
  }

  f_t get_gap(i_t id = 0) const override
  {
    return solution_.get_additional_termination_information(id).gap;
  }

  i_t get_num_iterations(i_t id = 0) const override
  {
    return solution_.get_additional_termination_information(id).number_of_steps_taken;
  }

  bool is_solved_by_pdlp(i_t id = 0) const override
  {
    return solution_.get_additional_termination_information(id).solved_by_pdlp;
  }

  const pdlp_warm_start_data_t<i_t, f_t>& get_pdlp_warm_start_data() const override
  {
    return const_cast<optimization_problem_solution_t<i_t, f_t>&>(solution_)
      .get_pdlp_warm_start_data();
  }

  bool has_warm_start_data() const override
  {
    return const_cast<optimization_problem_solution_t<i_t, f_t>&>(solution_)
             .get_pdlp_warm_start_data()
             .current_primal_solution_.size() > 0;
  }

  // Individual warm start data accessors (copy from device to host)
  std::vector<f_t> get_current_primal_solution_host() const override
  {
    auto& ws =
      const_cast<optimization_problem_solution_t<i_t, f_t>&>(solution_).get_pdlp_warm_start_data();
    if (ws.current_primal_solution_.size() == 0) return {};
    auto stream = ws.current_primal_solution_.stream();
    std::vector<f_t> result(ws.current_primal_solution_.size());
    raft::copy(result.data(),
               ws.current_primal_solution_.data(),
               ws.current_primal_solution_.size(),
               stream);
    stream.synchronize();
    return result;
  }

  std::vector<f_t> get_current_dual_solution_host() const override
  {
    auto& ws =
      const_cast<optimization_problem_solution_t<i_t, f_t>&>(solution_).get_pdlp_warm_start_data();
    if (ws.current_dual_solution_.size() == 0) return {};
    auto stream = ws.current_dual_solution_.stream();
    std::vector<f_t> result(ws.current_dual_solution_.size());
    raft::copy(
      result.data(), ws.current_dual_solution_.data(), ws.current_dual_solution_.size(), stream);
    stream.synchronize();
    return result;
  }

  std::vector<f_t> get_initial_primal_average_host() const override
  {
    auto& ws =
      const_cast<optimization_problem_solution_t<i_t, f_t>&>(solution_).get_pdlp_warm_start_data();
    if (ws.initial_primal_average_.size() == 0) return {};
    auto stream = ws.initial_primal_average_.stream();
    std::vector<f_t> result(ws.initial_primal_average_.size());
    raft::copy(
      result.data(), ws.initial_primal_average_.data(), ws.initial_primal_average_.size(), stream);
    stream.synchronize();
    return result;
  }

  std::vector<f_t> get_initial_dual_average_host() const override
  {
    auto& ws =
      const_cast<optimization_problem_solution_t<i_t, f_t>&>(solution_).get_pdlp_warm_start_data();
    if (ws.initial_dual_average_.size() == 0) return {};
    auto stream = ws.initial_dual_average_.stream();
    std::vector<f_t> result(ws.initial_dual_average_.size());
    raft::copy(
      result.data(), ws.initial_dual_average_.data(), ws.initial_dual_average_.size(), stream);
    stream.synchronize();
    return result;
  }

  std::vector<f_t> get_current_ATY_host() const override
  {
    auto& ws =
      const_cast<optimization_problem_solution_t<i_t, f_t>&>(solution_).get_pdlp_warm_start_data();
    if (ws.current_ATY_.size() == 0) return {};
    auto stream = ws.current_ATY_.stream();
    std::vector<f_t> result(ws.current_ATY_.size());
    raft::copy(result.data(), ws.current_ATY_.data(), ws.current_ATY_.size(), stream);
    stream.synchronize();
    return result;
  }

  std::vector<f_t> get_sum_primal_solutions_host() const override
  {
    auto& ws =
      const_cast<optimization_problem_solution_t<i_t, f_t>&>(solution_).get_pdlp_warm_start_data();
    if (ws.sum_primal_solutions_.size() == 0) return {};
    auto stream = ws.sum_primal_solutions_.stream();
    std::vector<f_t> result(ws.sum_primal_solutions_.size());
    raft::copy(
      result.data(), ws.sum_primal_solutions_.data(), ws.sum_primal_solutions_.size(), stream);
    stream.synchronize();
    return result;
  }

  std::vector<f_t> get_sum_dual_solutions_host() const override
  {
    auto& ws =
      const_cast<optimization_problem_solution_t<i_t, f_t>&>(solution_).get_pdlp_warm_start_data();
    if (ws.sum_dual_solutions_.size() == 0) return {};
    auto stream = ws.sum_dual_solutions_.stream();
    std::vector<f_t> result(ws.sum_dual_solutions_.size());
    raft::copy(result.data(), ws.sum_dual_solutions_.data(), ws.sum_dual_solutions_.size(), stream);
    stream.synchronize();
    return result;
  }

  std::vector<f_t> get_last_restart_duality_gap_primal_solution_host() const override
  {
    auto& ws =
      const_cast<optimization_problem_solution_t<i_t, f_t>&>(solution_).get_pdlp_warm_start_data();
    if (ws.last_restart_duality_gap_primal_solution_.size() == 0) return {};
    auto stream = ws.last_restart_duality_gap_primal_solution_.stream();
    std::vector<f_t> result(ws.last_restart_duality_gap_primal_solution_.size());
    raft::copy(result.data(),
               ws.last_restart_duality_gap_primal_solution_.data(),
               ws.last_restart_duality_gap_primal_solution_.size(),
               stream);
    stream.synchronize();
    return result;
  }

  std::vector<f_t> get_last_restart_duality_gap_dual_solution_host() const override
  {
    auto& ws =
      const_cast<optimization_problem_solution_t<i_t, f_t>&>(solution_).get_pdlp_warm_start_data();
    if (!ws.is_populated()) return {};
    auto stream = ws.last_restart_duality_gap_dual_solution_.stream();
    std::vector<f_t> result(ws.last_restart_duality_gap_dual_solution_.size());
    raft::copy(result.data(),
               ws.last_restart_duality_gap_dual_solution_.data(),
               ws.last_restart_duality_gap_dual_solution_.size(),
               stream);
    stream.synchronize();
    return result;
  }

  f_t get_initial_primal_weight() const override
  {
    return const_cast<optimization_problem_solution_t<i_t, f_t>&>(solution_)
      .get_pdlp_warm_start_data()
      .initial_primal_weight_;
  }
  f_t get_initial_step_size() const override
  {
    return const_cast<optimization_problem_solution_t<i_t, f_t>&>(solution_)
      .get_pdlp_warm_start_data()
      .initial_step_size_;
  }
  i_t get_total_pdlp_iterations() const override
  {
    return const_cast<optimization_problem_solution_t<i_t, f_t>&>(solution_)
      .get_pdlp_warm_start_data()
      .total_pdlp_iterations_;
  }
  i_t get_total_pdhg_iterations() const override
  {
    return const_cast<optimization_problem_solution_t<i_t, f_t>&>(solution_)
      .get_pdlp_warm_start_data()
      .total_pdhg_iterations_;
  }
  f_t get_last_candidate_kkt_score() const override
  {
    return const_cast<optimization_problem_solution_t<i_t, f_t>&>(solution_)
      .get_pdlp_warm_start_data()
      .last_candidate_kkt_score_;
  }
  f_t get_last_restart_kkt_score() const override
  {
    return const_cast<optimization_problem_solution_t<i_t, f_t>&>(solution_)
      .get_pdlp_warm_start_data()
      .last_restart_kkt_score_;
  }
  f_t get_sum_solution_weight() const override
  {
    return const_cast<optimization_problem_solution_t<i_t, f_t>&>(solution_)
      .get_pdlp_warm_start_data()
      .sum_solution_weight_;
  }
  i_t get_iterations_since_last_restart() const override
  {
    return const_cast<optimization_problem_solution_t<i_t, f_t>&>(solution_)
      .get_pdlp_warm_start_data()
      .iterations_since_last_restart_;
  }

  optimization_problem_solution_t<i_t, f_t> to_gpu_solution(rmm::cuda_stream_view) override
  {
    throw cuopt::logic_error("to_gpu_solution called on already GPU-backed solution",
                             error_type_t::ValidationError);
  }

  /**
   * @brief Convert GPU solution to CPU solution
   * Copies data from device to host for test mode or CPU-only environments.
   * @return A new cpu_lp_solution_t with all data copied to host vectors
   */
  std::unique_ptr<cpu_lp_solution_t<i_t, f_t>> to_cpu_solution() const
  {
    auto primal_host  = get_primal_solution_host();
    auto dual_host    = get_dual_solution_host();
    auto reduced_host = get_reduced_cost_host();

    if (has_warm_start_data()) {
      auto& gpu_ws = const_cast<optimization_problem_solution_t<i_t, f_t>&>(solution_)
                       .get_pdlp_warm_start_data();
      auto cpu_ws = convert_to_cpu_warmstart(gpu_ws, gpu_ws.current_primal_solution_.stream());

      return std::make_unique<cpu_lp_solution_t<i_t, f_t>>(std::move(primal_host),
                                                           std::move(dual_host),
                                                           std::move(reduced_host),
                                                           get_termination_status(),
                                                           get_objective_value(),
                                                           get_dual_objective_value(),
                                                           get_solve_time(),
                                                           get_l2_primal_residual(),
                                                           get_l2_dual_residual(),
                                                           get_gap(),
                                                           get_num_iterations(),
                                                           is_solved_by_pdlp(),
                                                           std::move(cpu_ws));
    }

    return std::make_unique<cpu_lp_solution_t<i_t, f_t>>(std::move(primal_host),
                                                         std::move(dual_host),
                                                         std::move(reduced_host),
                                                         get_termination_status(),
                                                         get_objective_value(),
                                                         get_dual_objective_value(),
                                                         get_solve_time(),
                                                         get_l2_primal_residual(),
                                                         get_l2_dual_residual(),
                                                         get_gap(),
                                                         get_num_iterations(),
                                                         is_solved_by_pdlp());
  }

  /**
   * @brief Convert to GPU-backed linear_programming_ret_t struct for Python/Cython
   * Moves device_uvector data into device_buffer wrappers with zero-copy.
   */
  cuopt::cython::linear_programming_ret_t to_linear_programming_ret_t();

  /**
   * @brief Polymorphic conversion to Python return type (interface override)
   * Populates the gpu_solutions_t variant inside linear_programming_ret_t.
   */
  cuopt::cython::linear_programming_ret_t to_python_lp_ret() override
  {
    return to_linear_programming_ret_t();
  }

 private:
  optimization_problem_solution_t<i_t, f_t> solution_;
};

/**
 * @brief GPU-backed MIP solution (wraps mip_solution_t)
 *
 * This class wraps the existing mip_solution_t which uses GPU memory.
 * It implements the interface to allow polymorphism with CPU solutions.
 */
template <typename i_t, typename f_t>
class gpu_mip_solution_t : public mip_solution_interface_t<i_t, f_t> {
 public:
  /**
   * @brief Construct from existing mip_solution_t (move)
   */
  explicit gpu_mip_solution_t(mip_solution_t<i_t, f_t>&& solution) : solution_(std::move(solution))
  {
  }

  // Interface implementations
  cuopt::logic_error get_error_status() const override { return solution_.get_error_status(); }

  f_t get_solve_time() const override { return static_cast<f_t>(solution_.get_total_solve_time()); }

  i_t get_solution_size() const override { return solution_.get_solution().size(); }

  std::vector<f_t> get_solution_host() const override
  {
    auto stream = solution_.get_solution().stream();
    std::vector<f_t> result(solution_.get_solution().size());
    raft::copy(
      result.data(), solution_.get_solution().data(), solution_.get_solution().size(), stream);
    stream.synchronize();
    return result;
  }

  f_t get_objective_value() const override { return solution_.get_objective_value(); }

  f_t get_mip_gap() const override { return solution_.get_mip_gap(); }

  f_t get_solution_bound() const override { return solution_.get_solution_bound(); }

  mip_termination_status_t get_termination_status() const override
  {
    return solution_.get_termination_status();
  }

  f_t get_presolve_time() const override { return solution_.get_presolve_time(); }

  f_t get_max_constraint_violation() const override
  {
    return solution_.get_max_constraint_violation();
  }

  f_t get_max_int_violation() const override { return solution_.get_max_int_violation(); }

  f_t get_max_variable_bound_violation() const override
  {
    return solution_.get_max_variable_bound_violation();
  }

  i_t get_num_nodes() const override { return solution_.get_num_nodes(); }

  i_t get_num_simplex_iterations() const override { return solution_.get_num_simplex_iterations(); }

  mip_solution_t<i_t, f_t> to_gpu_solution(rmm::cuda_stream_view) override
  {
    throw cuopt::logic_error("to_gpu_solution called on already GPU-backed MIP solution",
                             error_type_t::ValidationError);
  }

  /**
   * @brief Convert GPU MIP solution to CPU MIP solution
   * Copies data from device to host for test mode or CPU-only environments.
   * @return A new cpu_mip_solution_t with all data copied to host vectors
   */
  std::unique_ptr<cpu_mip_solution_t<i_t, f_t>> to_cpu_solution() const
  {
    auto solution_host = get_solution_host();

    return std::make_unique<cpu_mip_solution_t<i_t, f_t>>(std::move(solution_host),
                                                          get_termination_status(),
                                                          get_objective_value(),
                                                          get_mip_gap(),
                                                          get_solution_bound(),
                                                          get_solve_time(),
                                                          get_presolve_time(),
                                                          get_max_constraint_violation(),
                                                          get_max_int_violation(),
                                                          get_max_variable_bound_violation(),
                                                          get_num_nodes(),
                                                          get_num_simplex_iterations());
  }

  /**
   * @brief Convert to GPU-backed mip_ret_t struct for Python/Cython
   * Moves device_uvector data into device_buffer wrappers with zero-copy.
   */
  cuopt::cython::mip_ret_t to_mip_ret_t();

  /**
   * @brief Polymorphic conversion to Python return type (interface override)
   * Populates the gpu_buffer variant inside mip_ret_t.
   */
  cuopt::cython::mip_ret_t to_python_mip_ret() override { return to_mip_ret_t(); }

 private:
  mip_solution_t<i_t, f_t> solution_;
};

}  // namespace cuopt::linear_programming
