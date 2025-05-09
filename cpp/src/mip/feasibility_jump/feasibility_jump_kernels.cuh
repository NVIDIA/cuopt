/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights
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

#include "feasibility_jump.cuh"

#include <mip/logger.cuh>
#include <utilities/device_utils.cuh>

#include <raft/random/rng.cuh>

#include <cub/cub.cuh>

namespace cuopt::linear_programming::detail {

enum class weight_strategy_t { Increment, Multiply };

template <typename i_t, typename f_t>
__global__ void compute_iteration_related_variables_kernel(
  typename fj_t<i_t, f_t>::climber_data_t::view_t fj);
template <typename i_t, typename f_t>
__global__ void load_balancing_prepare_iteration(
  const __grid_constant__ typename fj_t<i_t, f_t>::climber_data_t::view_t fj);
template <typename i_t, typename f_t>
__global__ void load_balancing_sanity_checks(const __grid_constant__
                                             typename fj_t<i_t, f_t>::climber_data_t::view_t fj);
template <typename i_t, typename f_t>
__global__ void load_balancing_prepare_iteration(
  const __grid_constant__ typename fj_t<i_t, f_t>::climber_data_t::view_t fj);
template <typename i_t, typename f_t>
__global__ void load_balancing_compute_workid_mappings(
  typename fj_t<i_t, f_t>::climber_data_t::view_t fj,
  raft::device_span<i_t> row_size_prefix_sum,
  raft::device_span<i_t> var_indices,
  raft::device_span<fj_load_balancing_workid_mapping_t> work_id_to_var_idx);
template <typename i_t, typename f_t>
__global__ void load_balancing_init_cstr_bounds_csr(
  typename fj_t<i_t, f_t>::climber_data_t::view_t fj,
  raft::device_span<i_t> row_size_prefix_sum,
  raft::device_span<fj_load_balancing_workid_mapping_t> work_id_to_var_idx);
template <typename i_t, typename f_t>
__global__ void load_balancing_compute_scores_binary(
  const __grid_constant__ typename fj_t<i_t, f_t>::climber_data_t::view_t fj);
template <typename i_t, typename f_t>
__global__ void load_balancing_mtm_compute_candidates(
  const __grid_constant__ typename fj_t<i_t, f_t>::climber_data_t::view_t fj);
template <typename i_t, typename f_t>
__global__ void load_balancing_mtm_compute_scores(
  const __grid_constant__ typename fj_t<i_t, f_t>::climber_data_t::view_t fj);

template <typename i_t, typename f_t>
__global__ void init_lhs_and_violation(typename fj_t<i_t, f_t>::climber_data_t::view_t fj);

// Update the jump move tables after the best jump value has been computed for a "heavy" variable
template <typename i_t, typename f_t>
__global__ void heavy_jump_table_update_kernel(typename fj_t<i_t, f_t>::climber_data_t::view_t fj,
                                               i_t idx);

template <typename i_t, typename f_t>
__global__ void update_heavy_constraints_score(
  typename fj_t<i_t, f_t>::climber_data_t::view_t view);

// when we reach the bottom of a greedy descent, increase the weight of the violated constraints
// to escape the local minimum (as outlined in the paper)
template <typename i_t, typename f_t>
__global__ void handle_local_minimum_kernel(typename fj_t<i_t, f_t>::climber_data_t::view_t fj);

template <typename i_t, typename f_t>
__global__ void update_lift_moves_kernel(typename fj_t<i_t, f_t>::climber_data_t::view_t fj);

template <typename i_t, typename f_t>
__global__ void update_breakthrough_moves_kernel(
  typename fj_t<i_t, f_t>::climber_data_t::view_t fj);

template <typename i_t, typename f_t>
__global__ void update_assignment_kernel(typename fj_t<i_t, f_t>::climber_data_t::view_t fj,
                                         bool IgnoreLoadBalancing = false);

template <typename i_t, typename f_t>
__global__ void update_changed_constraints_kernel(
  typename fj_t<i_t, f_t>::climber_data_t::view_t fj);

template <typename i_t, typename f_t>
__global__ void update_best_solution_kernel(typename fj_t<i_t, f_t>::climber_data_t::view_t fj);

enum MTMMoveType { FJ_MTM_VIOLATED, FJ_MTM_SATISFIED };

template <typename i_t,
          typename f_t,
          MTMMoveType move_type = FJ_MTM_VIOLATED,
          bool is_binary_pb     = false>
__global__ void compute_mtm_moves_kernel(typename fj_t<i_t, f_t>::climber_data_t::view_t fj,
                                         bool ForceRefresh = false);

template <typename i_t, typename f_t>
__global__ void select_variable_kernel(typename fj_t<i_t, f_t>::climber_data_t::view_t fj);

}  // namespace cuopt::linear_programming::detail
