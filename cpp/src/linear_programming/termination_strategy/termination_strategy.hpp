/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */
#pragma once

#include <linear_programming/pdhg.hpp>
#include <linear_programming/pdlp_climber_strategy.hpp>
#include <linear_programming/swap_and_resize_helper.cuh>
#include <linear_programming/termination_strategy/convergence_information.hpp>
#include <linear_programming/termination_strategy/infeasibility_information.hpp>

#include <cuopt/linear_programming/pdlp/pdlp_warm_start_data.hpp>
#include <cuopt/linear_programming/pdlp/solver_settings.hpp>
#include <cuopt/linear_programming/pdlp/solver_solution.hpp>
#include <mip/problem/problem.cuh>

#include <utilities/unique_pinned_ptr.hpp>

#include <raft/core/handle.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_scalar.hpp>
#include <rmm/device_uvector.hpp>

#include <thrust/universal_vector.h>

namespace cuopt::linear_programming::detail {
template <typename i_t, typename f_t>
class pdlp_termination_strategy_t {
 public:
  pdlp_termination_strategy_t(raft::handle_t const* handle_ptr,
                              problem_t<i_t, f_t>& op_problem,
                              const problem_t<i_t, f_t>& scaled_op_problem,
                              cusparse_view_t<i_t, f_t>& cusparse_view,
                              const cusparse_view_t<i_t, f_t>& scaled_cusparse_view,
                              const i_t primal_size,
                              const i_t dual_size,
                              const pdlp_initial_scaling_strategy_t<i_t, f_t>&
                                scaling_strategy,  // Only used for cuPDLPx infeaislbity detection
                              const pdlp_solver_settings_t<i_t, f_t>& settings,
                              const std::vector<pdlp_climber_strategy_t>& climber_strategies);

  void evaluate_termination_criteria(
    pdhg_solver_t<i_t, f_t>& current_pdhg_solver,
    rmm::device_uvector<f_t>& primal_iterate,
    rmm::device_uvector<f_t>& dual_iterate,
    const rmm::device_uvector<f_t>& dual_slack,      // // Only useful in cuPDLPx restart mode
    rmm::device_uvector<f_t>& delta_primal_iterate,  // Only useful for infeasiblity detection
    rmm::device_uvector<f_t>& delta_dual_iterate,    // Only useful for infeasiblity detection
    i_t total_pdlp_iterations,
    const rmm::device_uvector<f_t>& combined_bounds,  // Only useful if per_constraint_residual
    const rmm::device_uvector<f_t>&
      objective_coefficients  // Only useful if per_constraint_residual
  );

  // Only useful in batch mode to store information of removed climber faster
  struct gpu_batch_additional_termination_information_t {
    gpu_batch_additional_termination_information_t(size_t batch_size)
      : number_of_steps_taken(batch_size),
        total_number_of_attempted_steps(batch_size),
        l2_primal_residual(batch_size),
        l2_relative_primal_residual(batch_size),
        l2_dual_residual(batch_size),
        l2_relative_dual_residual(batch_size),
        primal_objective(batch_size),
        dual_objective(batch_size),
        gap(batch_size),
        relative_gap(batch_size)
    {
    }

    struct view_t {
      raft::device_span<i_t> number_of_steps_taken;
      raft::device_span<i_t> total_number_of_attempted_steps;
      raft::device_span<f_t> l2_primal_residual;
      raft::device_span<f_t> l2_relative_primal_residual;
      raft::device_span<f_t> l2_dual_residual;
      raft::device_span<f_t> l2_relative_dual_residual;
      raft::device_span<f_t> primal_objective;
      raft::device_span<f_t> dual_objective;
      raft::device_span<f_t> gap;
      raft::device_span<f_t> relative_gap;
    };

    view_t view()
    {
      return view_t{
        make_span(number_of_steps_taken),
        make_span(total_number_of_attempted_steps),
        make_span(l2_primal_residual),
        make_span(l2_relative_primal_residual),
        make_span(l2_dual_residual),
        make_span(l2_relative_dual_residual),
        make_span(primal_objective),
        make_span(dual_objective),
        make_span(gap),
        make_span(relative_gap),
      };
    }
    /** Number of pdlp steps taken before termination */
    thrust::universal_host_pinned_vector<i_t> number_of_steps_taken;
    /** Number of pdhg steps taken before termination */
    thrust::universal_host_pinned_vector<i_t> total_number_of_attempted_steps;
    /** L2 norm of the primal residual (absolute primal residual) */
    thrust::universal_host_pinned_vector<f_t> l2_primal_residual;
    /** L2 norm of the primal residual divided by the L2 norm of the right hand side (b) */
    thrust::universal_host_pinned_vector<f_t> l2_relative_primal_residual;
    /** L2 norm of the dual residual */
    thrust::universal_host_pinned_vector<f_t> l2_dual_residual;
    /** L2 norm of the dual residual divided by the L2 norm of the objective coefficient (c) */
    thrust::universal_host_pinned_vector<f_t> l2_relative_dual_residual;
    /** Primal Objective */
    thrust::universal_host_pinned_vector<f_t> primal_objective;
    /** Dual Objective */
    thrust::universal_host_pinned_vector<f_t> dual_objective;

    /** Gap between primal and dual objective value */
    thrust::universal_host_pinned_vector<f_t> gap;
    /** Gap divided by the absolute sum of the primal and dual objective values */
    thrust::universal_host_pinned_vector<f_t> relative_gap;
  };

  void print_termination_criteria(i_t iteration, f_t elapsed, i_t best_id = 0) const;

  void swap_context(const thrust::universal_host_pinned_vector<swap_pair_t<i_t>>& swap_pairs);
  void resize_context(i_t new_size);

  void fill_gpu_terms_stats(i_t number_of_iterations);
  void convert_gpu_terms_stats_to_host(
    std::vector<
      typename optimization_problem_solution_t<i_t, f_t>::additional_termination_information_t>&
      additional_termination_informations);

  void set_relative_dual_tolerance_factor(f_t dual_tolerance_factor);
  void set_relative_primal_tolerance_factor(f_t primal_tolerance_factor);
  f_t get_relative_dual_tolerance_factor() const;
  f_t get_relative_primal_tolerance_factor() const;

  pdlp_termination_status_t get_termination_status(i_t id) const;
  std::vector<pdlp_termination_status_t> get_terminations_status();
  bool all_optimal_status() const;
  bool all_done() const;
  static __host__ __device__ bool is_done(pdlp_termination_status_t term);
  bool has_optimal_status() const;
  i_t nb_optimal_solutions() const;
  i_t get_optimal_solution_id() const;

  const convergence_information_t<i_t, f_t>& get_convergence_information() const;
  const infeasibility_information_t<i_t, f_t>& get_infeasibility_information() const;

  // Deep copy is used when save best primal so far is toggled
  optimization_problem_solution_t<i_t, f_t> fill_return_problem_solution(
    i_t number_of_iterations,
    pdhg_solver_t<i_t, f_t>& current_pdhg_solver,
    rmm::device_uvector<f_t>& primal_iterate,
    rmm::device_uvector<f_t>& dual_iterate,
    pdlp_warm_start_data_t<i_t, f_t>&& warm_start_data,
    std::vector<pdlp_termination_status_t>&& termination_status,
    bool deep_copy = false);

  // This verions simply calls the above with an empty pdlp_warm_start_data
  // It is used when we return without an optimal solution (infeasible, time limit...)
  optimization_problem_solution_t<i_t, f_t> fill_return_problem_solution(
    i_t number_of_iterations,
    pdhg_solver_t<i_t, f_t>& current_pdhg_solver,
    rmm::device_uvector<f_t>& primal_iterate,
    rmm::device_uvector<f_t>& dual_iterate,
    std::vector<pdlp_termination_status_t>&& termination_status,
    bool deep_copy = false);

 private:
  void check_termination_criteria();

  raft::handle_t const* handle_ptr_{nullptr};
  rmm::cuda_stream_view stream_view_;

  problem_t<i_t, f_t>* problem_ptr;

  convergence_information_t<i_t, f_t> convergence_information_;
  infeasibility_information_t<i_t, f_t> infeasibility_information_;

  thrust::universal_host_pinned_vector<i_t> termination_status_;
  const pdlp_solver_settings_t<i_t, f_t>& settings_;

  gpu_batch_additional_termination_information_t gpu_batch_additional_termination_information_;
  thrust::universal_host_pinned_vector<i_t> original_index_;

  const std::vector<pdlp_climber_strategy_t>& climber_strategies_;
};
}  // namespace cuopt::linear_programming::detail
