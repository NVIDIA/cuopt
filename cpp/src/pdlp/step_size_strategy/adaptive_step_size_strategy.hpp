/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */
#pragma once

#include <cuopt/linear_programming/pdlp/pdlp_hyper_params.cuh>
#include <utilities/event_handler.cuh>
#include <utilities/unique_pinned_ptr.hpp>

#include <pdlp/cusparse_view.hpp>
#include <pdlp/pdhg.hpp>
#include <pdlp/pdlp_climber_strategy.hpp>
#include <pdlp/saddle_point.hpp>
#include <pdlp/swap_and_resize_helper.cuh>

#include <raft/core/handle.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_scalar.hpp>
#include <rmm/device_uvector.hpp>

#include <thrust/universal_vector.h>

namespace cuopt::linear_programming::detail {
template <typename i_t, typename f_t>
class adaptive_step_size_strategy_t {
 public:
  /**
   * @brief A device-side view of the `adaptive_step_size_strategy_t` structure with the RAII stuffs
   *        stripped out, to make it easy to work inside kernels
   *
   * @note It is assumed that the pointers are NOT owned by this class, but rather
   *       by the encompassing `adaptive_step_size_strategy_t` class via RAII abstractions like
   *       `rmm::device_uvector`
   */
  struct view_t {
    raft::device_span<f_t> primal_weight;
    raft::device_span<f_t> step_size;
    i_t* valid_step_size;

    f_t* interaction;
    f_t* movement;

    f_t* norm_squared_delta_primal;
    f_t* norm_squared_delta_dual;

    pdlp_hyper_params::pdlp_hyper_params_t hyper_params;
  };

  adaptive_step_size_strategy_t(raft::handle_t const* handle_ptr,
                                rmm::device_uvector<f_t>* primal_weight,
                                rmm::device_uvector<f_t>* step_size,
                                bool is_legacy_batch_mode,
                                i_t primal_size,
                                i_t dual_size,
                                const std::vector<pdlp_climber_strategy_t>& climber_strategies,
                                const pdlp_hyper_params::pdlp_hyper_params_t& hyper_params);

  void compute_step_sizes(pdhg_solver_t<i_t, f_t>& pdhg_solver,
                          rmm::device_uvector<f_t>& primal_step_size,
                          rmm::device_uvector<f_t>& dual_step_size,
                          i_t total_pdlp_iterations);

  void get_primal_and_dual_stepsizes(rmm::device_uvector<f_t>& primal_step_size,
                                     rmm::device_uvector<f_t>& dual_step_size);
  /**
   * @brief Gets the device-side view (with raw pointers), for ease of access
   *        inside cuda kernels
   */
  view_t view();

  i_t get_valid_step_size() const;
  void set_valid_step_size(i_t);
  f_t get_interaction(i_t) const;
  f_t get_norm_squared_delta_primal(i_t) const;
  f_t get_norm_squared_delta_dual(i_t) const;
  const rmm::device_uvector<f_t>& get_interaction() const;
  const rmm::device_uvector<f_t>& get_norm_squared_delta_primal() const;
  const rmm::device_uvector<f_t>& get_norm_squared_delta_dual() const;

  void compute_interaction_and_movement(rmm::device_uvector<f_t>& tmp_primal,
                                        cusparse_view_t<i_t, f_t>& cusparse_view,
                                        saddle_point_state_t<i_t, f_t>& current_saddle_point_state);

  void swap_context(const thrust::universal_host_pinned_vector<swap_pair_t<i_t>>& swap_pairs);
  void resize_context(i_t new_size);

 private:
  const bool batch_mode_;

  raft::handle_t const* handle_ptr_{nullptr};
  rmm::cuda_stream_view stream_view_;

  i_t primal_size_;
  i_t dual_size_;

  rmm::device_uvector<f_t>* primal_weight_;
  rmm::device_uvector<f_t>* step_size_;
  // Host pinned memory scalar written in kernel
  // Combines both numerical_issue and valid_step size and save the device/host memcpy
  // -1: Error ; 0: Invalid step size ; 1: Valid step size
  thrust::universal_host_pinned_vector<i_t> valid_step_size_;

  rmm::device_uvector<f_t> interaction_;

  rmm::device_uvector<f_t> norm_squared_delta_primal_;
  rmm::device_uvector<f_t> norm_squared_delta_dual_;

  const rmm::device_scalar<f_t> reusable_device_scalar_value_1_;
  const rmm::device_scalar<f_t> reusable_device_scalar_value_0_;

  rmm::device_buffer dot_product_storage;
  size_t dot_product_bytes{0};

  ping_pong_graph_t<i_t> graph;

  const std::vector<pdlp_climber_strategy_t>& climber_strategies_;
  const pdlp_hyper_params::pdlp_hyper_params_t& hyper_params_;
};
}  // namespace cuopt::linear_programming::detail
