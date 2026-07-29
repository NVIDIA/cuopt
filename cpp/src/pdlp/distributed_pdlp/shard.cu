/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <pdlp/distributed_pdlp/shard.hpp>
#include <pdlp/pdlp.cuh>
#include <pdlp/utils.cuh>

#include <utilities/copy_helpers.hpp>

#include <raft/core/copy.hpp>
#include <raft/core/device_setter.hpp>

#include <cassert>
#include <limits>

namespace cuopt::mathematical_optimization::pdlp {

// This must be done in .cu file because the pdlp_solver_t is not already complete in the hpp file
// This is caused by the problematic cyclic include of pdlp_solver_t
template <typename i_t, typename f_t>
pdlp_shard_t<i_t, f_t>::~pdlp_shard_t() = default;

template <typename i_t, typename f_t>
pdlp_shard_t<i_t, f_t>::pdlp_shard_t(int device_id,
                                     rank_data_t<i_t, f_t>&& rd,
                                     nccl_comm_unique_ptr_t&& comm,
                                     io::mps_data_model_t<i_t, f_t> const& mps,
                                     pdlp_solver_settings_t<i_t, f_t> const& settings)
  : device_id(device_id),
    stream(),
    handle(stream.view()),
    comm(std::move(comm)),
    rank_data(std::move(rd)),
    opt_problem(&handle),
    sub_problem(std::nullopt),
    sub_pdlp(nullptr)
{
  assert(raft::device_setter::get_current_device() == device_id &&
         "Right device must be set before building the shard");

  // ---- 0. Problem-level scalars, taken straight from the global mps. ----
  // Objective coefficients / offset / scaling factor are passed through
  // unchanged; the max -> min conversion (negating all three) happens once,
  // in mip::problem_t's constructor via convert_to_maximization_problem
  const bool maximize                = mps.get_sense();
  const f_t objective_offset         = mps.get_objective_offset();
  const f_t objective_scaling_factor = mps.get_objective_scaling_factor();

  // Global (unpartitioned) host arrays, indexed by global var / cstr id.
  const std::vector<f_t>& g_obj        = mps.get_objective_coefficients();
  const std::vector<f_t>& g_var_lower  = mps.get_variable_lower_bounds();
  const std::vector<f_t>& g_var_upper  = mps.get_variable_upper_bounds();
  const std::vector<f_t>& g_cstr_lower = mps.get_constraint_lower_bounds();
  const std::vector<f_t>& g_cstr_upper = mps.get_constraint_upper_bounds();

  // ---- 1. Gather per-shard host slices using rank_data's index maps. ----
  // All vectors are sized to TOTAL (owned + halo). Owned slots get real
  // values; halo slots keep defaults because they should not be accessed before getting filled.
  std::vector<f_t> h_obj(rank_data.total_var_size, f_t{0});
  std::vector<f_t> h_var_lower(rank_data.total_var_size, -std::numeric_limits<f_t>::infinity());
  std::vector<f_t> h_var_upper(rank_data.total_var_size, std::numeric_limits<f_t>::infinity());
  std::vector<f_t> h_cstr_lower(rank_data.total_cstr_size, -std::numeric_limits<f_t>::infinity());
  std::vector<f_t> h_cstr_upper(rank_data.total_cstr_size, std::numeric_limits<f_t>::infinity());

  for (i_t i = 0; i < rank_data.owned_var_size; ++i) {
    const auto g   = rank_data.local_to_global_var[i];
    h_obj[i]       = g_obj[g];
    h_var_lower[i] = g_var_lower[g];
    h_var_upper[i] = g_var_upper[g];
  }
  for (i_t i = 0; i < rank_data.owned_cstr_size; ++i) {
    const auto g    = rank_data.local_to_global_cstr[i];
    h_cstr_lower[i] = g_cstr_lower[g];
    h_cstr_upper[i] = g_cstr_upper[g];
  }

  // ---- 2. Populate opt_problem (constructed in init list) on this shard's device. ----
  opt_problem.set_csr_constraint_matrix(rank_data.h_A_values.data(),
                                        static_cast<i_t>(rank_data.h_A_values.size()),
                                        rank_data.h_A_col_indices.data(),
                                        static_cast<i_t>(rank_data.h_A_col_indices.size()),
                                        rank_data.h_A_row_offsets.data(),
                                        static_cast<i_t>(rank_data.h_A_row_offsets.size()));

  // Primal axis: TOTAL (owned + halo)
  opt_problem.set_objective_coefficients(h_obj.data(), rank_data.total_var_size);
  opt_problem.set_variable_lower_bounds(h_var_lower.data(), rank_data.total_var_size);
  opt_problem.set_variable_upper_bounds(h_var_upper.data(), rank_data.total_var_size);

  // Dual axis: TOTAL (owned + halo)
  opt_problem.set_constraint_lower_bounds(h_cstr_lower.data(), rank_data.total_cstr_size);
  opt_problem.set_constraint_upper_bounds(h_cstr_upper.data(), rank_data.total_cstr_size);

  opt_problem.set_maximize(maximize);
  opt_problem.set_objective_offset(objective_offset);
  opt_problem.set_objective_scaling_factor(objective_scaling_factor);
  opt_problem.set_problem_category(problem_category_t::LP);

  // ---- 3. Build problem_t from opt_problem (UNSCALED). ----
  sub_problem.emplace(opt_problem);

  // ---- 4. Override reverse_* with the real local A_T from rank_data. ----
  // problem_t's ctor computes the transpose of the LOCAL A, which is wrong
  // in multi-GPU: A_local is owned_cstr x total_var, and A_t_local is the
  // pre-sliced owned_var x total_cstr matrix we built during partitioning.
  auto stream_view = handle.get_stream();
  sub_problem->reverse_offsets.resize(rank_data.h_A_t_row_offsets.size(), stream_view);
  sub_problem->reverse_constraints.resize(rank_data.h_A_t_col_indices.size(), stream_view);
  sub_problem->reverse_coefficients.resize(rank_data.h_A_t_values.size(), stream_view);
  raft::copy(sub_problem->reverse_offsets.data(),
             rank_data.h_A_t_row_offsets.data(),
             rank_data.h_A_t_row_offsets.size(),
             stream_view);
  raft::copy(sub_problem->reverse_constraints.data(),
             rank_data.h_A_t_col_indices.data(),
             rank_data.h_A_t_col_indices.size(),
             stream_view);
  raft::copy(sub_problem->reverse_coefficients.data(),
             rank_data.h_A_t_values.data(),
             rank_data.h_A_t_values.size(),
             stream_view);
  handle.sync_stream(stream_view);

  // ---- 5. Build sub_pdlp (single-GPU mode). ----
  // is_distributed_sub_pdlp=true has two effects in pdlp_solver_t's ctor:
  //   * skip the CSR/CSC transpose validity check -- A and A_T here are two
  //     independent local slices, not transposes (A has all owned rows and
  //     A_T has all owned columns).
  //   * skip local Ruiz / Pock-Chambolle inside initial_scaling_strategy_'s
  //     ctor -- distributed scaling (multi_gpu_engine_t::distributed_scaling)
  //     runs a cross-shard-coherent scaling later. Local per-shard scaling
  //     would be incoherent across shards.
  sub_pdlp = std::make_unique<pdlp_solver_t<i_t, f_t>>(
    *sub_problem, settings, /*is_legacy_batch_mode=*/false, /*is_distributed_sub_pdlp=*/true);

  // ---- 6. Build per-peer halo-exchange plans ----
  // For each peer p, we precompute:
  //   send_indices_d[p] : local indices to gather (uploaded from host send plan)
  //   send_buf_d[p]     : f_t staging buffer sized to match
  // Self-peer slot is present but empty (size 0).
  auto build_send_plan = [&](std::vector<std::vector<i_t>> const& send_per_peer,
                             std::vector<rmm::device_uvector<i_t>>& indices_d,
                             std::vector<rmm::device_uvector<f_t>>& buf_d) {
    const std::size_t n_peers = send_per_peer.size();
    indices_d.reserve(n_peers);
    buf_d.reserve(n_peers);
    for (auto const& send_to_peer : send_per_peer) {
      indices_d.emplace_back(send_to_peer.size(), stream_view);
      buf_d.emplace_back(send_to_peer.size(), stream_view);
      if (!send_to_peer.empty()) {
        raft::copy(indices_d.back().data(), send_to_peer.data(), send_to_peer.size(), stream_view);
      }
    }
  };
  build_send_plan(rank_data.var_send_per_peer, var_send_indices_d, var_send_buf_d);
  build_send_plan(rank_data.cstr_send_per_peer, cstr_send_indices_d, cstr_send_buf_d);

  handle.sync_stream(stream_view);
}

template struct pdlp_shard_t<int, double>;
template struct pdlp_shard_t<int, float>;

}  // namespace cuopt::mathematical_optimization::pdlp
