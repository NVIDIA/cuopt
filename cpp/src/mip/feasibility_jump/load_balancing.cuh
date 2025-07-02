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

#include <cuda_runtime_api.h>

#include "feasibility_jump_kernels.cuh"

#include <cub/block/block_merge_sort.cuh>
#include <utilities/cuda_helpers.cuh>

DI uint32_t get_unique_warp_id()
{
  uint32_t block_warp_id   = threadIdx.x / raft::WarpSize;
  uint32_t warps_per_block = blockDim.x / raft::WarpSize;
  return blockIdx.x * warps_per_block + block_warp_id;
}

DI uint32_t get_warp_id_stride()
{
  uint32_t warps_per_block = blockDim.x / raft::WarpSize;
  return gridDim.x * warps_per_block;
}

template <typename T>
static DI const T* bsearch_lower_bound(const T* ptr, const T value, size_t len)
{
#pragma unroll 4
  while (len > 1) {
    size_t half = len / 2;
    ptr += (ptr[half - 1] < value) * half;
    len -= half;
  }
  return ptr;
}

template <typename i_t, typename f_t>
DI bool needs_full_refresh(const typename fj_t<i_t, f_t>::climber_data_t::view_t& fj)
{
  bool full_refresh = false;
  if (*fj.iterations == 0) full_refresh = true;
  if (*fj.full_refresh_iteration == *fj.iterations) full_refresh = true;
  if (*fj.selected_var == std::numeric_limits<i_t>::max()) full_refresh = true;

  return full_refresh;
}

template <typename i_t, typename f_t>
struct csr_load_balancing_iterator;

template <typename i_t, typename f_t>
struct csr_load_balancing_sentinel {
  DI friend bool operator!=(const csr_load_balancing_iterator<i_t, f_t>& other,
                            csr_load_balancing_sentinel rhs)
  {
    return other.worker_id < other.base.max_worker_id;
  }
};

template <typename i_t, typename f_t>
struct csr_load_balancer {
  DI csr_load_balancer(const typename fj_t<i_t, f_t>::climber_data_t::view_t& in_fj,
                       raft::device_span<i_t> row_size_prefix_sum,
                       raft::device_span<fj_load_balancing_workid_mapping_t> in_work_id_to_var_idx)
    : fj(in_fj), work_id_to_var_idx(in_work_id_to_var_idx)
  {
    max_worker_id = row_size_prefix_sum.back();
    cuopt_assert(max_worker_id > 0, "kernel called with an empty work list");

    full_refresh = needs_full_refresh<i_t, f_t>(fj);
  }

  DI csr_load_balancing_iterator<i_t, f_t> begin()
  {
    return csr_load_balancing_iterator<i_t, f_t>(*this);
  }
  DI csr_load_balancing_sentinel<i_t, f_t> end() { return csr_load_balancing_sentinel<i_t, f_t>{}; }

  const typename fj_t<i_t, f_t>::climber_data_t::view_t& fj;
  i_t max_worker_id;
  raft::device_span<fj_load_balancing_workid_mapping_t> work_id_to_var_idx;
  bool full_refresh;
};

template <typename i_t, typename f_t>
struct csr_load_balancing_iterator {
  DI csr_load_balancing_iterator(csr_load_balancer<i_t, f_t>& in_base) : base(in_base) {}

  DI thrust::tuple<i_t, i_t, i_t, i_t, i_t, i_t> operator*()
  {
    auto [var_idx, subworkid, offset_begin, offset_end] = base.work_id_to_var_idx[worker_id];
    cuopt_assert(var_idx < base.fj.pb.n_variables, "invalid var_idx");

    bool skip = false;
    if (!base.full_refresh && !base.fj.iteration_related_variables.contains(var_idx)) skip = true;

    i_t csr_offset = lane_id + subworkid * raft::WarpSize + offset_begin;
    return {var_idx, subworkid, offset_begin, offset_end, csr_offset, skip};
  }

  DI auto& operator++()
  {
    worker_id += stride;
    return *this;
  }

  csr_load_balancer<i_t, f_t>& base;
  i_t worker_id    = get_unique_warp_id();
  i_t lane_id      = threadIdx.x % raft::WarpSize;
  const i_t stride = get_warp_id_stride();
};

// Initialize the moves and scores that will be changed during this LB iteration
// optionally fallback to the old codepath if the overhead of LB would likely be too great
template <typename i_t, typename f_t>
__global__ void load_balancing_prepare_iteration(const __grid_constant__
                                                 typename fj_t<i_t, f_t>::climber_data_t::view_t fj)
{
  bool full_refresh = needs_full_refresh<i_t, f_t>(fj);

  // alternate codepath in the case of a small related_var/total_var ratio
  if (!full_refresh && fj.pb.related_variables.size() > 0 &&
      fj.pb.n_variables / fj.work_ids_for_related_vars[*fj.selected_var] >=
        fj.settings->parameters.old_codepath_total_var_to_relvar_ratio_threshold) {
    auto range = fj.pb.range_for_related_vars(*fj.selected_var);

    for (i_t i = blockIdx.x + range.first; i < range.second; i += gridDim.x) {
      i_t var_idx = fj.pb.related_variables[i];
      update_jump_value<i_t, f_t, FJ_MTM_VIOLATED, false>(fj, var_idx);
    }

    if (FIRST_THREAD) *fj.load_balancing_skip = true;
    return;
  }

  // no fallback, prepare the LB codepath
  if (FIRST_THREAD) *fj.load_balancing_skip = false;

  // not a full refresh and the related variable table was precomputed
  if (!full_refresh && fj.pb.related_variables.size() > 0) {
    auto range = fj.pb.range_for_related_vars(*fj.selected_var);
    for (i_t i = TH_ID_X + range.first; i < range.second; i += GRID_STRIDE) {
      i_t var_idx = fj.pb.related_variables[i];
      fj.iteration_related_variables.set(var_idx);

      fj.jump_locks[var_idx]           = 0;
      fj.jump_candidate_count[var_idx] = 0;

      // the score of binary variables is computed by repeated atomicAdds, initalize it
      // to the neutral element
      // continuous/integer variable moves do not require this
      if (fj.pb.is_binary_variable[var_idx]) {
        fj.jump_move_scores[var_idx] = fj_t<i_t, f_t>::move_score_t::zero();
        fj.jump_move_delta[var_idx]  = NAN;
      } else {
        fj.jump_move_scores[var_idx] = fj_t<i_t, f_t>::move_score_t::invalid();
        fj.jump_move_delta[var_idx]  = NAN;
      }
    }
    return;
  }

  // no luck, can't use precomputations; compute the related_variables bitmap from scratch
  if (!full_refresh) {
    compute_iteration_related_variables<i_t, f_t>(fj);
    cg::this_grid().sync();
  }

  for (i_t var_idx = TH_ID_X; var_idx < fj.pb.n_variables; var_idx += GRID_STRIDE) {
    // if this variable is among those that will be updated this iteration
    if (full_refresh || fj.iteration_related_variables.contains(var_idx)) {
      fj.jump_locks[var_idx]           = 0;
      fj.jump_candidate_count[var_idx] = 0;

      if (fj.pb.is_binary_variable[var_idx]) {
        fj.jump_move_scores[var_idx] = fj_t<i_t, f_t>::move_score_t::zero();
        fj.jump_move_delta[var_idx]  = NAN;
      } else {
        fj.jump_move_scores[var_idx] = fj_t<i_t, f_t>::move_score_t::invalid();
        fj.jump_move_delta[var_idx]  = NAN;
      }
    }
  }
}

// Load balancing works by assigning a warp to each groupin of up to 32 nnzs.
// Initialize the schedule structure by looking up in the prefix sum table
// which workid maps to which variable

// TODO preload in shmem? run only once per solve, likely unecessary
template <typename i_t, typename f_t>
__global__ void load_balancing_compute_workid_mappings(
  typename fj_t<i_t, f_t>::climber_data_t::view_t fj,
  raft::device_span<i_t> row_size_prefix_sum,
  raft::device_span<i_t> var_indices,
  raft::device_span<fj_load_balancing_workid_mapping_t> work_id_to_var_idx)
{
  i_t max_work_id = row_size_prefix_sum.back();

  for (i_t workid = TH_ID_X; workid < max_work_id; workid += GRID_STRIDE) {
    auto ptr =
      bsearch_lower_bound<i_t>(row_size_prefix_sum.data(), workid + 1, row_size_prefix_sum.size());
    i_t idx = ptr - row_size_prefix_sum.data();
    cuopt_assert(idx >= 0 && idx < var_indices.size(), "invalid index");
    i_t var_idx = var_indices[idx];
    cuopt_assert(var_idx >= 0 && var_idx < fj.pb.n_variables, "invalid var_idx");
    uint32_t subworkid = *ptr - (workid + 1);

    auto [offset_begin, offset_end] = fj.pb.reverse_range_for_var(var_idx);

    // save the data needed by this workid (var_idx, slice of the nnzs to process)
    work_id_to_var_idx[workid] = {
      (uint32_t)var_idx, subworkid, (uint32_t)offset_begin, (uint32_t)offset_end};
  }
}

// Initialize some precomputed data structures needed by the LB algorithms
// for better performance
template <typename i_t, typename f_t>
__global__ void load_balancing_init_cstr_bounds_csr(
  typename fj_t<i_t, f_t>::climber_data_t::view_t fj,
  raft::device_span<i_t> row_size_prefix_sum,
  raft::device_span<fj_load_balancing_workid_mapping_t> work_id_to_var_idx)
{
  i_t max_work_id = row_size_prefix_sum.back();
  i_t lane_id     = threadIdx.x % raft::WarpSize;

  cuopt_assert(max_work_id > 0, "kernel called with an empty work list");

  const i_t stride = get_warp_id_stride();
  for (i_t workid = get_unique_warp_id(); workid < max_work_id; workid += stride) {
    auto [var_idx, subworkid, offset_begin, offset_end] = work_id_to_var_idx[workid];
    cuopt_assert(var_idx < fj.pb.n_variables, "invalid var_idx");

    i_t csr_offset = lane_id + subworkid * raft::WarpSize + offset_begin;

    if (csr_offset < offset_end) {
      auto cstr_idx   = fj.pb.reverse_constraints[csr_offset];
      auto cstr_coeff = fj.pb.reverse_coefficients[csr_offset];

      cuopt_assert(csr_offset >= 0, "");
      cuopt_assert(csr_offset < fj.constraint_lower_bounds_csr.size(), "");
      cuopt_assert(csr_offset < fj.constraint_upper_bounds_csr.size(), "");
      cuopt_assert(csr_offset < fj.cstr_coeff_reciprocal.size(), "");

      fj.constraint_lower_bounds_csr[csr_offset] = fj.pb.constraint_lower_bounds[cstr_idx];
      fj.constraint_upper_bounds_csr[csr_offset] = fj.pb.constraint_upper_bounds[cstr_idx];

      // sanity check, no thread should have written anything to this address yet
      cuopt_assert(fj.cstr_coeff_reciprocal[csr_offset] == 0, "double write detected");

      fj.cstr_coeff_reciprocal[csr_offset] = 1 / cstr_coeff;
      cuopt_assert(
        fj.cstr_coeff_reciprocal[csr_offset] != 0 && isfinite(fj.cstr_coeff_reciprocal[csr_offset]),
        "");
    }
  }
}

template <typename i_t, typename f_t>
static DI void warp_update_move_score(const typename fj_t<i_t, f_t>::climber_data_t::view_t& fj,
                                      typename fj_t<i_t, f_t>::move_score_t& score_info,
                                      i_t var_idx,
                                      f_t delta,
                                      f_t base_feas,
                                      f_t bonus_robust,
                                      i_t subworkid,
                                      i_t single_warp)
{
  f_t base  = base_feas;
  f_t bonus = bonus_robust;

  if (subworkid == 0) {
    auto [base_obj, bonus_breakthrough] = move_objective_score<i_t, f_t>(fj, var_idx, delta);
    base += base_obj;
    bonus += bonus_breakthrough;
  }

  if (single_warp) {
    score_info.base  = base;
    score_info.bonus = bonus;
  } else {
    atomicAdd(&score_info.base, base);
    atomicAdd(&score_info.bonus, bonus);
  }
}

// Load balancing pass for binary variables.
template <typename i_t, typename f_t>
__global__ void load_balancing_compute_scores_binary(
  const __grid_constant__ typename fj_t<i_t, f_t>::climber_data_t::view_t fj)
{
  if (*fj.load_balancing_skip) return;

  i_t lane_id = threadIdx.x % raft::WarpSize;

  for (auto [var_idx, subworkid, offset_begin, offset_end, csr_offset, skip] :
       csr_load_balancer<i_t, f_t>{fj, fj.row_size_bin_prefix_sum, fj.work_id_to_bin_var_idx}) {
    cuopt_assert(fj.pb.is_binary_variable[var_idx], "variable is not binary");

    if (skip) continue;

    bool single_warp = (offset_end - offset_begin) <= raft::WarpSize;

    f_t delta    = 1.0 - 2 * fj.incumbent_assignment[var_idx];
    f_t obj_diff = fj.pb.objective_coefficients[var_idx] * delta;

    // sanity checks
    if (threadIdx.x == 0) {
      cuopt_assert(fj.incumbent_assignment[var_idx] == 0 || fj.incumbent_assignment[var_idx] == 1,
                   "Current assignment is not binary!");
      cuopt_assert(
        fj.pb.variable_lower_bounds[var_idx] == 0 && fj.pb.variable_upper_bounds[var_idx] == 1, "");
      cuopt_assert(
        fj.pb.check_variable_within_bounds(var_idx, fj.incumbent_assignment[var_idx] + delta),
        "Var not within bounds!");
      cuopt_assert(delta != 0, "invalid candidate move");
      cuopt_assert(isfinite(delta), "invalid candidate move");
    }

    f_t base_feas    = 0;
    f_t bonus_robust = 0;

    // Go over each constraint that this variable would affect.
    // each thread in the warp computes its portion of the total score
    if (csr_offset < offset_end) {
      auto cstr_idx   = fj.pb.reverse_constraints[csr_offset];
      auto cstr_coeff = fj.pb.reverse_coefficients[csr_offset];
      // coalesced, avoids an indirect likely-uncoalesced memory load
      auto c_lb = fj.constraint_lower_bounds_csr[csr_offset];
      auto c_ub = fj.constraint_upper_bounds_csr[csr_offset];

      auto [cstr_base_feas, cstr_bonus_robust] = feas_score_constraint<i_t, f_t>(
        fj, var_idx, delta, cstr_idx, cstr_coeff, c_lb, c_ub, fj.incumbent_lhs[cstr_idx]);

      base_feas += cstr_base_feas;
      bonus_robust += cstr_bonus_robust;
    }

    // reduce the score components in-warp
    // usually the best option as constraints are usually small (<32)
    base_feas    = raft::warpReduce(base_feas);
    bonus_robust = raft::warpReduce(bonus_robust);

    // the first thread in the wapr takes care of computing the objective component
    // and of saving the computed move
    if (lane_id == 0) {
      warp_update_move_score<i_t, f_t>(fj,
                                       fj.jump_move_scores[var_idx],
                                       var_idx,
                                       delta,
                                       base_feas,
                                       bonus_robust,
                                       subworkid,
                                       single_warp);
      if (subworkid == 0) fj.jump_move_delta[var_idx] = delta;
    }
  }
}

// Load-balancing pass computing the continuous/integer move candidates for each variable.
// Each thread in a wapr computes a potential candidate move.
template <typename i_t, typename f_t>
__global__ void load_balancing_mtm_compute_candidates(
  const __grid_constant__ typename fj_t<i_t, f_t>::climber_data_t::view_t fj)
{
  if (*fj.load_balancing_skip) return;

  i_t lane_id = threadIdx.x % raft::WarpSize;

  const i_t stride = get_warp_id_stride();
  for (auto [var_idx, subworkid, offset_begin, offset_end, _, skip] : csr_load_balancer<i_t, f_t>{
         fj, fj.row_size_nonbin_prefix_sum, fj.work_id_to_nonbin_var_idx}) {
    cuopt_assert(!fj.pb.is_binary_variable[var_idx], "variable is binary");

    if (skip) continue;

    i_t csr_offset = lane_id + subworkid * raft::WarpSize + offset_begin;
    if (csr_offset >= offset_end) continue;

    auto cstr_idx = fj.pb.reverse_constraints[csr_offset];
    // only consider violated constraints
    if (!fj.violated_constraints.contains(cstr_idx)) continue;

    // we're not really compute bound BUT FP64 division compiles into an additional codepath
    // which incurs additional register pressure, so multiply by the precomputed reciprocal instead
    auto rcp_cstr_coeff = fj.cstr_coeff_reciprocal[csr_offset];
    f_t c_lb            = fj.constraint_lower_bounds_csr[csr_offset];
    f_t c_ub            = fj.constraint_upper_bounds_csr[csr_offset];
    f_t v_lb            = fj.pb.variable_lower_bounds[var_idx];
    f_t v_ub            = fj.pb.variable_upper_bounds[var_idx];

    cuopt_assert(c_lb == fj.pb.constraint_lower_bounds[cstr_idx], "");
    cuopt_assert(c_ub == fj.pb.constraint_upper_bounds[cstr_idx], "");

    cuopt_assert(cstr_idx >= 0 && cstr_idx < fj.pb.n_constraints, "");
    cuopt_assert(isfinite(fj.incumbent_lhs[cstr_idx]), "");
    cuopt_assert(rcp_cstr_coeff == 1 / fj.pb.reverse_coefficients[csr_offset], "");
    f_t cstr_coeff = fj.pb.reverse_coefficients[csr_offset];

    f_t cstr_tolerance = fj.get_corrected_tolerance(cstr_idx);

    f_t old_val   = fj.incumbent_assignment[var_idx];
    f_t obj_coeff = fj.pb.objective_coefficients[var_idx];
    f_t delta     = 0;

    f_t bound = c_lb;
    i_t sign  = -1;
    // if this bound is not violated (or is infinite), switch to the other bound
    if (fj.incumbent_lhs[cstr_idx] + cstr_tolerance >= c_lb) {
      bound = c_ub;
      sign  = 1;
    }

    // factor to correct the lhs/rhs to turn a lb <= lhs <= ub constraint into
    // two virtual constraints lhs <= ub and -lhs <= -lb
    f_t lhs   = fj.incumbent_lhs[cstr_idx] * sign;
    f_t rhs   = bound * sign;
    f_t slack = rhs - lhs;

    f_t new_val = old_val;

    f_t delta_ij = slack * rcp_cstr_coeff * sign;
    new_val      = old_val + delta_ij;
    if (fj.pb.is_integer_var(var_idx)) {
      new_val = rcp_cstr_coeff * sign > 0 ? floor(new_val + fj.pb.tolerances.integrality_tolerance)
                                          : ceil(new_val - fj.pb.tolerances.integrality_tolerance);
    }

    new_val = max(min(new_val, v_ub), v_lb);

    cuopt_assert(isfinite(new_val), "");
    cuopt_assert(fj.pb.check_variable_within_bounds(var_idx, new_val), "");
    cuopt_assert(isfinite(old_val), "");
    cuopt_assert(isfinite(new_val - old_val), "Jump move delta is not finite!");

    delta = isfinite(new_val) ? new_val - old_val : 0;

    bool is_duplicate = false;
    // check across the warp to opportunistically eliminate duplicate candidate moves
    uint32_t mask = __match_any_sync(__activemask(), delta);
    if (__popc(mask) > 1) {
      auto mask_ffs = __ffs(mask) - 1;
      is_duplicate  = lane_id != mask_ffs;
    }

    // only store actual candidates (delta non-zero)
    if (!fj.pb.integer_equal(delta, (f_t)0) && !is_duplicate) {
      i_t count = atomicAdd(&fj.jump_candidate_count[var_idx], 1);
      i_t idx   = offset_begin + count;
      cuopt_assert(idx >= offset_begin && idx < offset_end, "overrun");
      fj.jump_candidates[idx].delta     = delta;
      fj.jump_candidates[idx].score     = fj_t<i_t, f_t>::move_score_t::zero();
      fj.candidate_arrived_workids[idx] = 0;
      cuopt_assert(
        fj.pb.check_variable_within_bounds(var_idx, fj.incumbent_assignment[var_idx] + delta),
        "assignment not within bounds");
    }
  }
}

// Compute the scores of each candidate move of each variable
// maximize occupancy for better results, we can afford the extra L1TEX traffic
template <typename i_t, typename f_t>
__launch_bounds__(TPB_loadbalance, 16) __global__
  void load_balancing_mtm_compute_scores(const __grid_constant__
                                         typename fj_t<i_t, f_t>::climber_data_t::view_t fj)
{
  if (*fj.load_balancing_skip) return;

  i_t lane_id = threadIdx.x % raft::WarpSize;

  const i_t stride = get_warp_id_stride();
  for (auto [var_idx, subworkid, offset_begin, offset_end, _, skip] : csr_load_balancer<i_t, f_t>{
         fj, fj.row_size_nonbin_prefix_sum, fj.work_id_to_nonbin_var_idx}) {
    cuopt_assert(!fj.pb.is_binary_variable[var_idx], "variable is binary");

    if (skip) continue;

    i_t csr_offset   = lane_id + subworkid * raft::WarpSize + offset_begin;
    i_t warp_count   = (offset_end - offset_begin - 1) / raft::WarpSize + 1;
    bool single_warp = warp_count == 1;

    i_t cstr_idx   = -1;
    f_t cstr_coeff = NAN;
    f_t c_lb       = NAN;
    f_t c_ub       = NAN;

    if (csr_offset < offset_end) {
      cstr_idx   = fj.pb.reverse_constraints[csr_offset];
      cstr_coeff = fj.pb.reverse_coefficients[csr_offset];
      // coalesced, avoids an indirect likely-uncoalesced memory load
      c_lb = fj.constraint_lower_bounds_csr[csr_offset];
      c_ub = fj.constraint_upper_bounds_csr[csr_offset];

      cuopt_assert(c_lb == fj.pb.constraint_lower_bounds[cstr_idx], "");
      cuopt_assert(c_ub == fj.pb.constraint_upper_bounds[cstr_idx], "");

      cuopt_assert(cstr_idx >= 0 && cstr_idx < fj.pb.n_constraints, "");
    }

    f_t v_lb = fj.pb.variable_lower_bounds[var_idx];
    f_t v_ub = fj.pb.variable_upper_bounds[var_idx];

    // candidate counts is usually very small (<4) thanks to early duplicate deletion in the
    // previous kernel rarely limits the thoroughput nor leads to noticeable imbalance
    const i_t candidate_count = fj.jump_candidate_count[var_idx];
    for (i_t i = offset_begin; i < offset_begin + candidate_count; ++i) {
      auto& candidate = fj.jump_candidates[i];
      f_t delta       = candidate.delta;

      if (lane_id == 0) {
        cuopt_assert(delta != 0, "invalid candidate move");
        cuopt_assert(isfinite(delta), "invalid candidate move");
        cuopt_assert(isfinite(fj.incumbent_lhs[cstr_idx]), "");
        cuopt_assert(
          fj.pb.check_variable_within_bounds(var_idx, fj.incumbent_assignment[var_idx] + delta),
          "assignment not within bounds");
      }

      auto& score_info = candidate.score;

      f_t base_feas    = 0;
      f_t bonus_robust = 0;

      // same as for the binary var kernel, compute each score compoenent per thread
      // and merge then via a wapr reduce
      if (csr_offset < offset_end) {
        cuopt_assert(c_lb == fj.pb.constraint_lower_bounds[cstr_idx], "bound sanity check failed");
        cuopt_assert(c_ub == fj.pb.constraint_upper_bounds[cstr_idx], "bound sanity check failed");

        auto [cstr_base_feas, cstr_bonus_robust] = feas_score_constraint<i_t, f_t>(
          fj, var_idx, delta, cstr_idx, cstr_coeff, c_lb, c_ub, fj.incumbent_lhs[cstr_idx]);

        base_feas += cstr_base_feas;
        bonus_robust += cstr_bonus_robust;
      }

      base_feas    = raft::warpReduce(base_feas);
      bonus_robust = raft::warpReduce(bonus_robust);

      if (lane_id == 0) {
        warp_update_move_score<i_t, f_t>(
          fj, score_info, var_idx, delta, base_feas, bonus_robust, subworkid, single_warp);

        // if this was the last warp to arrive on this jump candidate, then its score is final.
        // update the current best candidate accordingly
        i_t arrived_count = 1 + atomicAdd(&fj.candidate_arrived_workids[i], 1);
        cuopt_assert(arrived_count > 0 && arrived_count <= warp_count, "invalid arrival count");
        if (arrived_count == warp_count) {
          // check if this move candidate would be better than the current assigned one for this
          // variable.

          cuopt_assert(isfinite(candidate.score.base), "invalid score");
          cuopt_assert(isfinite(candidate.score.bonus), "invalid score");

          // early exit, atomic load
          cuda::atomic_ref<typename fj_t<i_t, f_t>::move_score_t, cuda::thread_scope_device>
            best_score_ref{fj.jump_move_scores[var_idx]};
          auto best_score = best_score_ref.load(cuda::memory_order_relaxed);

          if (best_score < candidate.score ||
              (best_score == candidate.score && candidate.delta < fj.jump_move_delta[var_idx])) {
            // update the best move delta
            acquire_lock(&fj.jump_locks[var_idx]);

            // reject this move if it would increase the target variable to a numerically unstable
            // value
            if (!fj.move_numerically_stable(fj.incumbent_assignment[var_idx],
                                            fj.incumbent_assignment[var_idx] + delta,
                                            base_feas)) {
              fj.jump_move_scores[var_idx] = fj_t<i_t, f_t>::move_score_t::invalid();
            } else if (fj.jump_move_scores[var_idx] < candidate.score
                       // determinism for ease of debugging
                       || (fj.jump_move_scores[var_idx] == candidate.score &&
                           candidate.delta < fj.jump_move_delta[var_idx])) {
              fj.jump_move_delta[var_idx]  = candidate.delta;
              fj.jump_move_scores[var_idx] = candidate.score;
            }
            release_lock(&fj.jump_locks[var_idx]);
          }
        }
      }
    }
  }
}

template <typename i_t, typename f_t>
__global__ void load_balancing_sanity_checks(const __grid_constant__
                                             typename fj_t<i_t, f_t>::climber_data_t::view_t v)
{
  // check that all warps have arrived
  for (i_t var_idx = blockIdx.x; var_idx < v.pb.n_variables; var_idx += gridDim.x) {
    auto [offset_begin, offset_end] = v.pb.reverse_range_for_var(var_idx);
    i_t warp_count                  = (offset_end - offset_begin - 1) / raft::WarpSize + 1;

    if (v.pb.is_binary_variable[var_idx]) continue;
    i_t candidate_count = v.jump_candidate_count[var_idx];
    for (i_t i = offset_begin + threadIdx.x; i < offset_begin + candidate_count; i += blockDim.x) {
      if (!(v.candidate_arrived_workids[i] == warp_count)) {
        printf("(iter %d) [%d]: %d vs %d\n",
               *v.iterations,
               var_idx,
               v.candidate_arrived_workids[i],
               warp_count);
        __trap();
      }

      if (!isfinite(v.jump_move_delta_check[var_idx])) {
        printf(
          "--- (iter %d) [%d]: delta %g\n", *v.iterations, var_idx, v.jump_candidates[i].delta);
        cuopt_assert(isnan(v.jump_candidates[i].delta), "invalid delta");
        __trap();
      }
    }
  }

  cg::this_grid().sync();

  for (i_t var_idx = TH_ID_X; var_idx < v.pb.n_variables; var_idx += GRID_STRIDE) {
    f_t delta_1  = v.jump_move_delta[var_idx];
    f_t delta_2  = v.jump_move_delta_check[var_idx];
    auto score_1 = v.jump_move_scores[var_idx];
    auto score_2 = v.jump_move_score_check[var_idx];

    f_t rel_error = abs((delta_1 - delta_2) / delta_1);
    if (rel_error > 1e-3) {
      printf("(iter %d) [%d]: was %g, is %g, error %g\n",
             *v.iterations,
             var_idx,
             delta_1,
             delta_2,
             rel_error);
      __trap();
    }

    if (!(score_1 == score_1.invalid() && score_2 == score_2.invalid()) &&
        !(v.pb.integer_equal(score_1.base, score_2.base) &&
          v.pb.integer_equal(score_1.bonus, score_2.bonus))) {
      printf("(iter %d) [%d, int:%d]: delta %g/%g was %f/%f, is %f/%f\n",
             *v.iterations,
             var_idx,
             v.pb.is_integer_var(var_idx),
             delta_1,
             delta_2,
             score_1.base,
             score_1.bonus,
             score_2.base,
             score_2.bonus);
      __trap();
    }
  }
}