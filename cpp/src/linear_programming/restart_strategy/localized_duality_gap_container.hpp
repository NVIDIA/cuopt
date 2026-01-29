/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */
#pragma once

#include <cuopt/linear_programming/pdlp/pdlp_hyper_params.cuh>
#include <linear_programming/pdlp_climber_strategy.hpp>
#include <linear_programming/swap_and_resize_helper.cuh>

#include <raft/core/device_span.hpp>
#include <raft/core/handle.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_scalar.hpp>
#include <rmm/device_uvector.hpp>

namespace cuopt::linear_programming::detail {
template <typename i_t, typename f_t>
struct localized_duality_gap_container_t {
 public:
  localized_duality_gap_container_t(raft::handle_t const* handle_ptr,
                                    i_t primal_size,
                                    i_t dual_size,
                                    const std::vector<pdlp_climber_strategy_t>& climber_strategies,
                                    const pdlp_hyper_params::pdlp_hyper_params_t& hyper_params);

  struct view_t {
    /** size of primal problem */
    i_t primal_size;
    /** size of dual problem */
    i_t dual_size;

    f_t* lagrangian_value;
    f_t* lower_bound_value;
    f_t* upper_bound_value;
    f_t* distance_traveled;
    raft::device_span<f_t> primal_distance_traveled;
    raft::device_span<f_t> dual_distance_traveled;
    f_t* normalized_gap;

    f_t* primal_solution;
    f_t* dual_solution;
    f_t* primal_gradient;
    f_t* dual_gradient;
    f_t* primal_solution_tr;
    f_t* dual_solution_tr;

    pdlp_hyper_params::pdlp_hyper_params_t hyper_params;
  };

  void swap_context(const thrust::universal_host_pinned_vector<swap_pair_t<i_t>>& swap_pairs);
  void resize_context(i_t new_size);

  /**
   * @brief Gets the device-side view (with raw pointers), for ease of access
   *        inside cuda kernels
   */
  view_t view();

  i_t primal_size_h_;
  i_t dual_size_h_;
  rmm::device_scalar<f_t> lagrangian_value_;
  rmm::device_scalar<f_t> lower_bound_value_;
  rmm::device_scalar<f_t> upper_bound_value_;
  rmm::device_scalar<f_t> distance_traveled_;
  rmm::device_uvector<f_t> primal_distance_traveled_;
  rmm::device_uvector<f_t> dual_distance_traveled_;
  rmm::device_scalar<f_t> normalized_gap_;

  rmm::device_uvector<f_t> primal_solution_;
  rmm::device_uvector<f_t> dual_solution_;
  rmm::device_uvector<f_t> primal_gradient_;
  rmm::device_uvector<f_t> dual_gradient_;
  rmm::device_uvector<f_t> primal_solution_tr_;
  rmm::device_uvector<f_t> dual_solution_tr_;
};
}  // namespace cuopt::linear_programming::detail
