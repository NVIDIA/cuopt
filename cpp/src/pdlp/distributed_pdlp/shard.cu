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

namespace cuopt::linear_programming::detail {

// This must be done in .cu file because the pdlp_solver_t is not already complete in the hpp file
template <typename i_t, typename f_t>
pdlp_shard_t<i_t, f_t>::~pdlp_shard_t() = default;

template <typename i_t, typename f_t>
pdlp_shard_t<i_t, f_t>::pdlp_shard_t(int device_id,
                                     rank_data_t<i_t, f_t>&& rd,
                                     ncclComm_t raw_comm,
                                     std::vector<f_t> const& h_global_obj,
                                     std::vector<f_t> const& h_global_var_lower,
                                     std::vector<f_t> const& h_global_var_upper,
                                     std::vector<f_t> const& h_global_cstr_lower,
                                     std::vector<f_t> const& h_global_cstr_upper,
                                     std::vector<f_t> const& h_global_obj_scaled,
                                     std::vector<f_t> const& h_global_var_lower_scaled,
                                     std::vector<f_t> const& h_global_var_upper_scaled,
                                     std::vector<f_t> const& h_global_cstr_lower_scaled,
                                     std::vector<f_t> const& h_global_cstr_upper_scaled,
                                     std::vector<f_t> const& h_global_cummulative_cstr_scaling,
                                     std::vector<f_t> const& h_global_cummulative_var_scaling,
                                     f_t h_bound_rescaling,
                                     f_t h_objective_rescaling,
                                     bool maximize,
                                     f_t objective_offset,
                                     f_t objective_scaling_factor,
                                     pdlp_solver_settings_t<i_t, f_t> const& settings)
  : device_id(device_id),
    stream(),
    handle(stream.view()),
    comm(raw_comm, nccl_comm_deleter_t{device_id}),
    rank_data(std::move(rd)),
    opt_problem(std::nullopt),
    sub_problem(std::nullopt),
    sub_pdlp(nullptr)
{
  assert(raft::device_setter::get_current_device() == device_id &&
         "Right device must be set before building the shard");

  // ---- 1. Gather per-shard host slices using rank_data's index maps. ----
  // All vectors are sized to TOTAL (owned + halo). Owned slots get real
  // values; halo slots keep defaults because they should not be accessed
  std::vector<f_t> h_obj(rank_data.total_var_size, f_t{0});
  std::vector<f_t> h_var_lower(rank_data.total_var_size, -std::numeric_limits<f_t>::infinity());
  std::vector<f_t> h_var_upper(rank_data.total_var_size, std::numeric_limits<f_t>::infinity());
  std::vector<f_t> h_cstr_lower(rank_data.total_cstr_size, -std::numeric_limits<f_t>::infinity());
  std::vector<f_t> h_cstr_upper(rank_data.total_cstr_size, std::numeric_limits<f_t>::infinity());

  std::vector<f_t> h_obj_scaled(rank_data.total_var_size, f_t{0});
  std::vector<f_t> h_var_lower_scaled(rank_data.total_var_size,
                                      -std::numeric_limits<f_t>::infinity());
  std::vector<f_t> h_var_upper_scaled(rank_data.total_var_size,
                                      std::numeric_limits<f_t>::infinity());
  std::vector<f_t> h_cstr_lower_scaled(rank_data.total_cstr_size,
                                       -std::numeric_limits<f_t>::infinity());
  std::vector<f_t> h_cstr_upper_scaled(rank_data.total_cstr_size,
                                       std::numeric_limits<f_t>::infinity());

  for (i_t i = 0; i < rank_data.owned_var_size; ++i) {
    const auto g          = rank_data.local_to_global_var[i];
    h_obj[i]              = h_global_obj[g];
    h_var_lower[i]        = h_global_var_lower[g];
    h_var_upper[i]        = h_global_var_upper[g];
    h_obj_scaled[i]       = h_global_obj_scaled[g];
    h_var_lower_scaled[i] = h_global_var_lower_scaled[g];
    h_var_upper_scaled[i] = h_global_var_upper_scaled[g];
  }
  for (i_t i = 0; i < rank_data.owned_cstr_size; ++i) {
    const auto g           = rank_data.local_to_global_cstr[i];
    h_cstr_lower[i]        = h_global_cstr_lower[g];
    h_cstr_upper[i]        = h_global_cstr_upper[g];
    h_cstr_lower_scaled[i] = h_global_cstr_lower_scaled[g];
    h_cstr_upper_scaled[i] = h_global_cstr_upper_scaled[g];
  }

  // Get local scaling factors
  std::vector<f_t> h_cstr_scaling_local(rank_data.total_cstr_size, f_t{1});
  std::vector<f_t> h_var_scaling_local(rank_data.total_var_size, f_t{1});
  for (i_t i = 0; i < rank_data.owned_cstr_size; ++i) {
    h_cstr_scaling_local[i] = h_global_cummulative_cstr_scaling[rank_data.local_to_global_cstr[i]];
  }
  for (i_t i = 0; i < rank_data.owned_var_size; ++i) {
    h_var_scaling_local[i] = h_global_cummulative_var_scaling[rank_data.local_to_global_var[i]];
  }

  // ---- 2. Build optimization_problem_t on this shard's device (UNSCALED). ----
  opt_problem.emplace(&handle);
  opt_problem->set_csr_constraint_matrix(rank_data.h_A_values.data(),
                                         static_cast<i_t>(rank_data.h_A_values.size()),
                                         rank_data.h_A_col_indices.data(),
                                         static_cast<i_t>(rank_data.h_A_col_indices.size()),
                                         rank_data.h_A_row_offsets.data(),
                                         static_cast<i_t>(rank_data.h_A_row_offsets.size()));

  // Primal axis: TOTAL (owned + halo). Halo slots have neutral defaults.
  opt_problem->set_objective_coefficients(h_obj.data(), rank_data.total_var_size);
  opt_problem->set_variable_lower_bounds(h_var_lower.data(), rank_data.total_var_size);
  opt_problem->set_variable_upper_bounds(h_var_upper.data(), rank_data.total_var_size);

  // Dual axis: TOTAL (owned + halo). Halo slots have ±inf so trivially satisfied.
  opt_problem->set_constraint_lower_bounds(h_cstr_lower.data(), rank_data.total_cstr_size);
  opt_problem->set_constraint_upper_bounds(h_cstr_upper.data(), rank_data.total_cstr_size);

  opt_problem->set_maximize(maximize);
  opt_problem->set_objective_offset(objective_offset);
  opt_problem->set_objective_scaling_factor(objective_scaling_factor);
  opt_problem->set_problem_category(problem_category_t::LP);

  // ---- 3. Build problem_t from opt_problem (still UNSCALED). ----
  sub_problem.emplace(*opt_problem);

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

  // ---- 5. Build sub_pdlp (single-GPU mode; multi_gpu flags cleared by caller). ----
  //         At this point sub_pdlp.op_problem_scaled_ is an unscaled copy
  //         of sub_problem and sub_pdlp.initial_scaling_strategy_ has
  //         unit cumulative factors (sub-settings disable Ruiz / PC iters).
  // NOTE: pass is_legacy_batch_mode=true to disable CUDA-graph capture inside
  // sub_pdlp while debugging fake-mGPU divergence. The flag is a pure
  // graph-capture toggle (ping_pong_graph_t / manual_cuda_graph_t) and does
  // not change any algorithm semantics. Restore to false once the path is
  // confirmed correct.
  sub_pdlp = std::make_unique<pdlp_solver_t<i_t, f_t>>(
    *sub_problem, settings, /*is_legacy_batch_mode=*/true);

  sub_pdlp->pdhg_solver_.set_is_multi_gpu(true);

  // Inject master-scaled buffers inside sub_pdlp
  auto& scaled = sub_pdlp->get_op_problem_scaled();
  raft::copy(scaled.coefficients.data(),
             rank_data.h_A_values_scaled.data(),
             rank_data.h_A_values_scaled.size(),
             stream_view);
  raft::copy(scaled.reverse_coefficients.data(),
             rank_data.h_A_t_values_scaled.data(),
             rank_data.h_A_t_values_scaled.size(),
             stream_view);
  raft::copy(
    scaled.objective_coefficients.data(), h_obj_scaled.data(), h_obj_scaled.size(), stream_view);
  raft::copy(scaled.constraint_lower_bounds.data(),
             h_cstr_lower_scaled.data(),
             h_cstr_lower_scaled.size(),
             stream_view);
  raft::copy(scaled.constraint_upper_bounds.data(),
             h_cstr_upper_scaled.data(),
             h_cstr_upper_scaled.size(),
             stream_view);

  using f_t2 = typename type_2<f_t>::type;
  std::vector<f_t2> h_var_bounds_scaled_packed(rank_data.total_var_size);
  for (i_t i = 0; i < rank_data.total_var_size; ++i) {
    h_var_bounds_scaled_packed[i].x = h_var_lower_scaled[i];
    h_var_bounds_scaled_packed[i].y = h_var_upper_scaled[i];
  }
  raft::copy(scaled.variable_bounds.data(),
             h_var_bounds_scaled_packed.data(),
             h_var_bounds_scaled_packed.size(),
             stream_view);

  combine_constraint_bounds<i_t, f_t>(scaled, scaled.combined_bounds);

  // Inject master-scaled buffers inside sub_pdlp.initil_strategy
  auto& scaling = sub_pdlp->get_initial_scaling_strategy();
  scaling.set_cummulative_scaling(h_cstr_scaling_local, h_var_scaling_local);
  scaling.set_h_bound_rescaling(h_bound_rescaling);
  scaling.set_h_objective_rescaling(h_objective_rescaling);

  sub_pdlp->pdhg_solver_.get_cusparse_view().create_spmv_op_plans(
    /* is_reflected */ true);
  // ---- 6. Build per-peer halo-exchange plans (ported from metis_tests). ----
  // For each peer p, we precompute:
  //   send_indices_d[p] : local indices to gather (uploaded from host send plan)
  //   send_buf_d[p]     : f_t staging buffer sized to match
  // Self-peer slot is present but empty (size 0). Used in engine halo exchange.
  auto build_send_plan = [&](auto const& send_per_peer, auto& indices_d, auto& buf_d) {
    const std::size_t n_peers = send_per_peer.size();
    indices_d.reserve(n_peers);
    buf_d.reserve(n_peers);
    for (auto const& send_to_peer : send_per_peer) {
      rmm::device_uvector<i_t> idx(send_to_peer.size(), stream_view);
      rmm::device_uvector<f_t> buf(send_to_peer.size(), stream_view);
      if (!send_to_peer.empty()) {
        raft::copy(idx.data(), send_to_peer.data(), send_to_peer.size(), stream_view);
      }
      indices_d.emplace_back(std::move(idx));
      buf_d.emplace_back(std::move(buf));
    }
  };
  build_send_plan(rank_data.var_send_per_peer, var_send_indices_d, var_send_buf_d);
  build_send_plan(rank_data.cstr_send_per_peer, cstr_send_indices_d, cstr_send_buf_d);

  handle.sync_stream(stream_view);
}

template struct pdlp_shard_t<int, double>;
template struct pdlp_shard_t<int, float>;

}  // namespace cuopt::linear_programming::detail
