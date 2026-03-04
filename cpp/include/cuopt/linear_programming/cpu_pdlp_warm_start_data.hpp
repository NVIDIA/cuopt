/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#pragma once

#include <cuopt/linear_programming/pdlp/pdlp_warm_start_data.hpp>
#include <vector>

namespace cuopt::linear_programming {

// CPU version of pdlp_warm_start_data_t using std::vector for remote execution
template <typename i_t, typename f_t>
struct cpu_pdlp_warm_start_data_t {
  std::vector<f_t> current_primal_solution_;
  std::vector<f_t> current_dual_solution_;
  std::vector<f_t> initial_primal_average_;
  std::vector<f_t> initial_dual_average_;
  std::vector<f_t> current_ATY_;
  std::vector<f_t> sum_primal_solutions_;
  std::vector<f_t> sum_dual_solutions_;
  std::vector<f_t> last_restart_duality_gap_primal_solution_;
  std::vector<f_t> last_restart_duality_gap_dual_solution_;
  f_t initial_primal_weight_{-1};
  f_t initial_step_size_{-1};
  i_t total_pdlp_iterations_{-1};
  i_t total_pdhg_iterations_{-1};
  f_t last_candidate_kkt_score_{-1};
  f_t last_restart_kkt_score_{-1};
  f_t sum_solution_weight_{-1};
  i_t iterations_since_last_restart_{-1};

  // Default constructor
  cpu_pdlp_warm_start_data_t() = default;

  /**
   * @brief Construct from a warm-start data view by copying span data to host vectors.
   * @note The input view's spans (e.g., current_primal_solution_, current_dual_solution_)
   *       must reference host (CPU) memory, since std::vector::assign is used internally.
   *       To convert from GPU/device warm-start data, use convert_to_cpu_warmstart() instead.
   */
  cpu_pdlp_warm_start_data_t(const pdlp_warm_start_data_view_t<i_t, f_t>& view)
    : initial_primal_weight_(view.initial_primal_weight_),
      initial_step_size_(view.initial_step_size_),
      total_pdlp_iterations_(view.total_pdlp_iterations_),
      total_pdhg_iterations_(view.total_pdhg_iterations_),
      last_candidate_kkt_score_(view.last_candidate_kkt_score_),
      last_restart_kkt_score_(view.last_restart_kkt_score_),
      sum_solution_weight_(view.sum_solution_weight_),
      iterations_since_last_restart_(view.iterations_since_last_restart_)
  {
    // Copy vector data from spans
    if (view.current_primal_solution_.data() != nullptr) {
      current_primal_solution_.assign(
        view.current_primal_solution_.data(),
        view.current_primal_solution_.data() + view.current_primal_solution_.size());
    }
    if (view.current_dual_solution_.data() != nullptr) {
      current_dual_solution_.assign(
        view.current_dual_solution_.data(),
        view.current_dual_solution_.data() + view.current_dual_solution_.size());
    }
    if (view.initial_primal_average_.data() != nullptr) {
      initial_primal_average_.assign(
        view.initial_primal_average_.data(),
        view.initial_primal_average_.data() + view.initial_primal_average_.size());
    }
    if (view.initial_dual_average_.data() != nullptr) {
      initial_dual_average_.assign(
        view.initial_dual_average_.data(),
        view.initial_dual_average_.data() + view.initial_dual_average_.size());
    }
    if (view.current_ATY_.data() != nullptr) {
      current_ATY_.assign(view.current_ATY_.data(),
                          view.current_ATY_.data() + view.current_ATY_.size());
    }
    if (view.sum_primal_solutions_.data() != nullptr) {
      sum_primal_solutions_.assign(
        view.sum_primal_solutions_.data(),
        view.sum_primal_solutions_.data() + view.sum_primal_solutions_.size());
    }
    if (view.sum_dual_solutions_.data() != nullptr) {
      sum_dual_solutions_.assign(view.sum_dual_solutions_.data(),
                                 view.sum_dual_solutions_.data() + view.sum_dual_solutions_.size());
    }
    if (view.last_restart_duality_gap_primal_solution_.data() != nullptr) {
      last_restart_duality_gap_primal_solution_.assign(
        view.last_restart_duality_gap_primal_solution_.data(),
        view.last_restart_duality_gap_primal_solution_.data() +
          view.last_restart_duality_gap_primal_solution_.size());
    }
    if (view.last_restart_duality_gap_dual_solution_.data() != nullptr) {
      last_restart_duality_gap_dual_solution_.assign(
        view.last_restart_duality_gap_dual_solution_.data(),
        view.last_restart_duality_gap_dual_solution_.data() +
          view.last_restart_duality_gap_dual_solution_.size());
    }
  }

  // Check if warmstart data is populated (same sentinel check as release/26.02)
  bool is_populated() const { return !last_restart_duality_gap_dual_solution_.empty(); }
};

// Forward declare GPU type for conversion functions
template <typename i_t, typename f_t>
struct pdlp_warm_start_data_t;

// Convert GPU → CPU warmstart (D2H copy)
template <typename i_t, typename f_t>
cpu_pdlp_warm_start_data_t<i_t, f_t> convert_to_cpu_warmstart(
  const pdlp_warm_start_data_t<i_t, f_t>& gpu_data, rmm::cuda_stream_view stream);

// Convert CPU → GPU warmstart (H2D copy)
template <typename i_t, typename f_t>
pdlp_warm_start_data_t<i_t, f_t> convert_to_gpu_warmstart(
  const cpu_pdlp_warm_start_data_t<i_t, f_t>& cpu_data, rmm::cuda_stream_view stream);

}  // namespace cuopt::linear_programming
