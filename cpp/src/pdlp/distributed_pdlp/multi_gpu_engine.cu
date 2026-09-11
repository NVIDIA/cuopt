/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <pdlp/distributed_pdlp/multi_gpu_engine.hpp>
#include <pdlp/pdlp.cuh>

#include <cuopt/error.hpp>

#include <raft/core/device_setter.hpp>

#include <nccl.h>

#include <numeric>

#include <utilities/logger.hpp>

namespace cuopt::mathematical_optimization::pdlp {

template <typename i_t, typename f_t>
multi_gpu_engine_t<i_t, f_t>::multi_gpu_engine_t(
  std::vector<rank_data_t<i_t, f_t>>&& rank_data,
  io::mps_data_model_t<i_t, f_t> const& mps,
  pdlp_solver_settings_t<i_t, f_t> const& sub_solver_settings)
  : stream()
{
  const int nb_parts = static_cast<int>(rank_data.size());
  cuopt_expects(
    nb_parts > 0, error_type_t::ValidationError, "multi_gpu_engine_t: rank_data must be non-empty");

  shards.reserve(nb_parts);
  std::vector<int> devices(nb_parts);
  std::iota(devices.begin(), devices.end(), 0);

  // Create NCCL Comms, then immediately wrap each in a RAII owner so they are
  // destroyed on any exception (e.g. a shard ctor throwing) before being
  // handed off to a shard.
  std::vector<nccl_comm_unique_ptr_t> comms;
  comms.reserve(nb_parts);
  std::vector<ncclComm_t> raw_comms(nb_parts, nullptr);
  cuopt_expects(ncclCommInitAll(raw_comms.data(), nb_parts, devices.data()) == ncclSuccess,
                error_type_t::RuntimeError,
                "ncclCommInitAll failed");

  for (int r = 0; r < nb_parts; ++r) {
    comms.emplace_back(raw_comms[r], nccl_comm_deleter_t{devices[r]});
  }

  // 3. Construct one shard per rank, pinned to its device. Ownership of each
  //    communicator moves into its shard.
  for (int r = 0; r < nb_parts; ++r) {
    raft::device_setter guard(devices[r]);  // shard ctor needs device set
    shards.emplace_back(std::make_unique<pdlp_shard_t<i_t, f_t>>(
      devices[r], std::move(rank_data[r]), std::move(comms[r]), mps, sub_solver_settings));
  }

  // Two different events
  // capture_*_event_ are used inside graph capture
  // ext_*_event_ are used when sync is needed outside of graph
  graph_master_ready_event_ = std::make_unique<cuopt::event_handler_t>();
  sync_master_ready_event_  = std::make_unique<cuopt::event_handler_t>();
  graph_shard_ready_events_.reserve(nb_parts);
  sync_shard_ready_events_.reserve(nb_parts);
  for_each_shard([&](auto&) {
    graph_shard_ready_events_.emplace_back(std::make_unique<cuopt::event_handler_t>());
    sync_shard_ready_events_.emplace_back(std::make_unique<cuopt::event_handler_t>());
  });

  // Cache per-shard partition metadata for gather_owned_*_to_master_bufs.
  local_to_global_vars_.reserve(nb_parts);
  local_to_global_cstrs_.reserve(nb_parts);
  for (int r = 0; r < nb_parts; ++r) {
    auto const& rd = shards[r]->rank_data;
    local_to_global_vars_.push_back(rd.local_to_global_var);
    local_to_global_cstrs_.push_back(rd.local_to_global_cstr);
  }
}

template <typename i_t, typename f_t>
void multi_gpu_engine_t<i_t, f_t>::set_master(pdlp_solver_t<i_t, f_t>* m)
{
  cuopt_assert(m != nullptr, "set_master: master pointer must not be null");
  master_pdlp_ = m;
}

// -------- High-level: A @ x and A_T @ y -----------------------------------
template <typename i_t, typename f_t>
void multi_gpu_engine_t<i_t, f_t>::distributed_compute_A_x()
{
  halo_exchange_var([](pdlp_solver_t<i_t, f_t>& p) -> rmm::device_uvector<f_t>& {
    return p.pdhg_solver_.get_reflected_primal();
  });
  for_each_shard([](auto& shard) { shard.sub_pdlp->pdhg_solver_.spmvop_A_x(); });
}

template <typename i_t, typename f_t>
void multi_gpu_engine_t<i_t, f_t>::distributed_compute_At_y()
{
  halo_exchange_cstr([](pdlp_solver_t<i_t, f_t>& p) -> rmm::device_uvector<f_t>& {
    return p.pdhg_solver_.get_dual_solution();
  });
  for_each_shard([](auto& shard) { shard.sub_pdlp->pdhg_solver_.spmvop_At_y(); });
}

// -------- Cross-stream fork / join / sync ---------------------------------
template <typename i_t, typename f_t>
void multi_gpu_engine_t<i_t, f_t>::synchronize_shards()
{
  for_each_shard([](auto& s) { s.stream.synchronize(); });
}

template <typename i_t, typename f_t>
void multi_gpu_engine_t<i_t, f_t>::graph_capture_fork_to_shards(rmm::cuda_stream_view master_stream)
{
  graph_master_ready_event_->record(master_stream);
  for_each_shard([&](auto& s) { graph_master_ready_event_->stream_wait(s.stream.view()); });
}

template <typename i_t, typename f_t>
void multi_gpu_engine_t<i_t, f_t>::graph_capture_join_from_shards(
  rmm::cuda_stream_view master_stream)
{
  for_each_shard([&](auto& s, int r) { graph_shard_ready_events_[r]->record(s.stream.view()); });
  for (auto& e : graph_shard_ready_events_) {
    e->stream_wait(master_stream);
  }
}

template <typename i_t, typename f_t>
void multi_gpu_engine_t<i_t, f_t>::sync_await_master(rmm::cuda_stream_view master_stream)
{
  sync_master_ready_event_->record(master_stream);
  for_each_shard([&](auto& s) { sync_master_ready_event_->stream_wait(s.stream.view()); });
}

template <typename i_t, typename f_t>
void multi_gpu_engine_t<i_t, f_t>::sync_await_shards(rmm::cuda_stream_view master_stream)
{
  for_each_shard([&](auto& s, int r) { sync_shard_ready_events_[r]->record(s.stream.view()); });
  for (auto& e : sync_shard_ready_events_) {
    e->stream_wait(master_stream);
  }
}

// -------- Halo exchange ------------
// typename pdlp_shard_t<i_t, f_t>::halo_axis_t is the unified view of the halo exchange metadata
// for both variables and constraints. It contains the send and receive buffers for each peer, the
// owned size, and the receive offsets and counts for each peer. There is one for variables and one
// for constraints, on each shard. This allows to avoid duplicating the logic for variables and
// constraints.
template <typename i_t, typename f_t>
void multi_gpu_engine_t<i_t, f_t>::halo_exchange_bufs_impl(
  std::vector<raft::device_span<f_t>> const& bufs,
  std::vector<typename pdlp_shard_t<i_t, f_t>::halo_axis_t> const& axes)
{
  const int nb = static_cast<int>(shards.size());
  cuopt_expects(static_cast<int>(bufs.size()) == nb && static_cast<int>(axes.size()) == nb,
                error_type_t::RuntimeError,
                "halo_exchange_bufs_impl: bufs / axes must have size == shards.size()");

  // Step 1: gather owned values that each peer needs into per-peer staging.
  for_each_shard([&](auto& s, int r) {
    auto const& ax = axes[r];
    auto x         = bufs[r];
    for (int peer = 0; peer < nb; ++peer) {
      if (peer == r) continue;
      if (ax.send_indices[peer].size() == 0) continue;
      thrust::gather(rmm::exec_policy_nosync(s.stream.view()),
                     ax.send_indices[peer].begin(),
                     ax.send_indices[peer].end(),
                     x.data(),
                     ax.send_buf[peer].begin());
    }
  });

  // Step 2: matched send / recv across the whole topology in one NCCL group.
  CUOPT_NCCL_TRY(ncclGroupStart());
  for_each_shard([&](auto& s, int r) {
    auto const& ax = axes[r];
    for (int peer = 0; peer < nb; ++peer) {
      if (peer == r) continue;
      CUOPT_NCCL_TRY(ncclSend(ax.send_buf[peer].data(),
                              ax.send_buf[peer].size(),
                              nccl_data_type<f_t>(),
                              peer,
                              s.comm.get(),
                              s.stream.view().get()));
    }
  });
  for_each_shard([&](auto& s, int r) {
    auto const& ax = axes[r];
    auto x         = bufs[r];
    for (int peer = 0; peer < nb; ++peer) {
      if (peer == r) continue;
      f_t* recv_ptr = x.data() + ax.owned_size + ax.recv_offsets[peer];
      CUOPT_NCCL_TRY(ncclRecv(recv_ptr,
                              static_cast<size_t>(ax.recv_counts[peer]),
                              nccl_data_type<f_t>(),
                              peer,
                              s.comm.get(),
                              s.stream.view().get()));
    }
  });
  CUOPT_NCCL_TRY(ncclGroupEnd());
}

template <typename i_t, typename f_t>
void multi_gpu_engine_t<i_t, f_t>::halo_exchange_var_bufs(
  std::vector<raft::device_span<f_t>> const& bufs)
{
  std::vector<typename pdlp_shard_t<i_t, f_t>::halo_axis_t> axes;
  axes.reserve(shards.size());
  for (auto& s : shards)
    axes.push_back(s->var_halo_axis());
  halo_exchange_bufs_impl(bufs, axes);
}

template <typename i_t, typename f_t>
void multi_gpu_engine_t<i_t, f_t>::halo_exchange_var_bufs(
  std::vector<rmm::device_uvector<f_t>>& bufs)
{
  std::vector<raft::device_span<f_t>> spans;
  spans.reserve(bufs.size());
  for (auto& b : bufs)
    spans.emplace_back(b.data(), b.size());
  halo_exchange_var_bufs(spans);
}

template <typename i_t, typename f_t>
void multi_gpu_engine_t<i_t, f_t>::halo_exchange_cstr_bufs(
  std::vector<raft::device_span<f_t>> const& bufs)
{
  std::vector<typename pdlp_shard_t<i_t, f_t>::halo_axis_t> axes;
  axes.reserve(shards.size());
  for (auto& s : shards)
    axes.push_back(s->cstr_halo_axis());
  halo_exchange_bufs_impl(bufs, axes);
}

template <typename i_t, typename f_t>
void multi_gpu_engine_t<i_t, f_t>::halo_exchange_cstr_bufs(
  std::vector<rmm::device_uvector<f_t>>& bufs)
{
  std::vector<raft::device_span<f_t>> spans;
  spans.reserve(bufs.size());
  for (auto& b : bufs)
    spans.emplace_back(b.data(), b.size());
  halo_exchange_cstr_bufs(spans);
}

// -------- Gather owned slices to master -------------------------------------
template <typename i_t, typename f_t>
void multi_gpu_engine_t<i_t, f_t>::gather_owned_to_master_bufs_impl(
  std::vector<raft::device_span<f_t const>> const& shard_owned,
  raft::device_span<f_t> master_buf,
  std::vector<std::vector<i_t>> const& local_to_globals)
{
  cuopt_assert(master_pdlp_ != nullptr,
               "gather_owned_to_master_bufs_impl requires set_master(...)");
  const int nb = static_cast<int>(shards.size());
  cuopt_expects(
    static_cast<int>(shard_owned.size()) == nb && static_cast<int>(local_to_globals.size()) == nb,
    error_type_t::RuntimeError,
    "gather_owned_to_master_bufs_impl: shard_owned / local_to_globals "
    "must have size == shards.size()");

  // Assemble on host in global-index order.
  std::vector<f_t> h_master(master_buf.size());
  for_each_shard([&](auto& s, int r) {
    const std::size_t n_owned = shard_owned[r].size();
    if (n_owned == 0) return;
    std::vector<f_t> tmp(n_owned);
    raft::copy(tmp.data(), shard_owned[r].data(), n_owned, s.stream.view());
    // Sync so tmp is populated before the host scatter (and stays valid).
    s.stream.synchronize();
    thrust::scatter(
      thrust::host, tmp.begin(), tmp.end(), local_to_globals[r].begin(), h_master.begin());
  });

  // Single H->D onto master's device (`stream` lives on the master device).
  raft::copy(master_buf.data(), h_master.data(), master_buf.size(), stream.view());
  stream.synchronize();
}

template <typename i_t, typename f_t>
void multi_gpu_engine_t<i_t, f_t>::gather_owned_var_to_master_bufs(
  std::vector<raft::device_span<f_t const>> const& shard_owned, raft::device_span<f_t> master_buf)
{
  gather_owned_to_master_bufs_impl(shard_owned, master_buf, local_to_global_vars_);
}

template <typename i_t, typename f_t>
void multi_gpu_engine_t<i_t, f_t>::gather_owned_cstr_to_master_bufs(
  std::vector<raft::device_span<f_t const>> const& shard_owned, raft::device_span<f_t> master_buf)
{
  gather_owned_to_master_bufs_impl(shard_owned, master_buf, local_to_global_cstrs_);
}

// -------- NCCL allreduce (sum, in place) ------------------------------------
template <typename i_t, typename f_t>
void multi_gpu_engine_t<i_t, f_t>::allreduce_sum_inplace_bufs(
  std::vector<raft::device_scalar_view<f_t>> const& scalars)
{
  const int nb = static_cast<int>(shards.size());
  cuopt_expects(static_cast<int>(scalars.size()) == nb,
                error_type_t::RuntimeError,
                "allreduce_sum_inplace_bufs: scalars.size() must equal shards.size()");
  if (nb == 0) return;

  CUOPT_NCCL_TRY(ncclGroupStart());
  for_each_shard([&](auto& s, int r) {
    f_t* p = scalars[r].data_handle();
    CUOPT_NCCL_TRY(ncclAllReduce(p,
                                 p,
                                 /*count=*/1,
                                 nccl_data_type<f_t>(),
                                 ncclSum,
                                 s.comm.get(),
                                 s.stream.view().get()));
  });
  CUOPT_NCCL_TRY(ncclGroupEnd());
}

template <typename i_t, typename f_t>
void multi_gpu_engine_t<i_t, f_t>::allreduce_sum_inplace_to_master_buf(
  std::vector<raft::device_scalar_view<f_t>> const& shard_scalars,
  raft::device_scalar_view<f_t> master_dst)
{
  cuopt_assert(master_pdlp_ != nullptr,
               "allreduce_sum_inplace_to_master_buf requires set_master(...)");
  allreduce_sum_inplace_bufs(shard_scalars);
  if (shards.empty()) return;
  auto master_stream = master_pdlp_->get_handle_ptr()->get_stream();
  sync_await_shards(master_stream);
  auto& s0 = *shards[0];
  raft::device_setter guard(s0.device_id);
  raft::copy(master_dst.data_handle(), shard_scalars[0].data_handle(), 1, master_stream);
}

// -------- Distributed dot / L2 norm -----------------------------------------
template <typename i_t, typename f_t>
void multi_gpu_engine_t<i_t, f_t>::distributed_dot_bufs(
  std::vector<raft::device_span<f_t>> const& a_bufs,
  std::vector<raft::device_span<f_t>> const& b_bufs,
  std::vector<raft::device_scalar_view<f_t>> const& out_scalars)
{
  const int nb = static_cast<int>(shards.size());
  cuopt_expects(static_cast<int>(a_bufs.size()) == nb && static_cast<int>(b_bufs.size()) == nb &&
                  static_cast<int>(out_scalars.size()) == nb,
                error_type_t::RuntimeError,
                "distributed_dot_bufs: a_bufs / b_bufs / out_scalars must "
                "all have size == shards.size()");

  for_each_shard([&](auto& s, int r) {
    cuopt_expects(a_bufs[r].size() == b_bufs[r].size(),
                  error_type_t::RuntimeError,
                  "distributed_dot_bufs: a_bufs[r] and b_bufs[r] must have equal size");
    RAFT_CUBLAS_TRY(raft::linalg::detail::cublasdot(s.handle.get_cublas_handle(),
                                                    static_cast<int>(a_bufs[r].size()),
                                                    a_bufs[r].data(),
                                                    1,
                                                    b_bufs[r].data(),
                                                    1,
                                                    out_scalars[r].data_handle(),
                                                    s.stream.view().get()));
  });

  allreduce_sum_inplace_bufs(out_scalars);
}

template <typename i_t, typename f_t>
void multi_gpu_engine_t<i_t, f_t>::distributed_dot_bufs(
  std::vector<raft::device_span<f_t>> const& a_bufs,
  std::vector<raft::device_span<f_t>> const& b_bufs,
  std::vector<rmm::device_scalar<f_t>>& out_scalars)
{
  std::vector<raft::device_scalar_view<f_t>> views;
  views.reserve(out_scalars.size());
  for (auto& s : out_scalars)
    views.emplace_back(raft::make_device_scalar_view<f_t>(s.data()));
  distributed_dot_bufs(a_bufs, b_bufs, views);
}

template <typename i_t, typename f_t>
void multi_gpu_engine_t<i_t, f_t>::distributed_l2_norm_bufs(
  std::vector<raft::device_span<f_t>> const& in_bufs,
  std::vector<raft::device_scalar_view<f_t>> const& out_scalars)
{
  distributed_dot_bufs(in_bufs, in_bufs, out_scalars);
  for_each_shard([&](pdlp_shard_t<i_t, f_t>& s, int r) {
    cub::DeviceTransform::Transform(
      out_scalars[r].data_handle(),
      out_scalars[r].data_handle(),
      1,
      [] __device__(f_t x) { return cuda::std::sqrt(x); },
      s.stream.view().get());
  });
}

template <typename i_t, typename f_t>
void multi_gpu_engine_t<i_t, f_t>::distributed_l2_norm_bufs(
  std::vector<raft::device_span<f_t>> const& in_bufs,
  std::vector<rmm::device_scalar<f_t>>& out_scalars)
{
  std::vector<raft::device_scalar_view<f_t>> views;
  views.reserve(out_scalars.size());
  for (auto& s : out_scalars)
    views.emplace_back(raft::make_device_scalar_view<f_t>(s.data()));
  distributed_l2_norm_bufs(in_bufs, views);
}

template <typename i_t, typename f_t>
void multi_gpu_engine_t<i_t, f_t>::distributed_l2_norm_to_master_buf(
  std::vector<raft::device_span<f_t>> const& in_bufs,
  std::vector<raft::device_scalar_view<f_t>> const& shard_out,
  raft::device_scalar_view<f_t> master_dst)
{
  cuopt_assert(master_pdlp_ != nullptr,
               "distributed_l2_norm_to_master_buf requires set_master(...)");
  distributed_l2_norm_bufs(in_bufs, shard_out);
  auto master_stream = master_pdlp_->get_handle_ptr()->get_stream();
  sync_await_shards(master_stream);
  auto& s0 = *shards[0];
  raft::device_setter guard(s0.device_id);
  raft::copy(master_dst.data_handle(), shard_out[0].data_handle(), 1, master_stream);
}

// -------- Fused halo-exchange + SpMV ----------------------------------------
template <typename i_t, typename f_t>
void multi_gpu_engine_t<i_t, f_t>::distributed_spmv_At(
  std::vector<rmm::device_uvector<f_t>>& in_bufs,
  std::vector<cusparse_dn_vec_uptr>& in_descs,
  std::vector<cusparse_dn_vec_uptr>& out_descs)
{
  halo_exchange_cstr_bufs(in_bufs);
  for_each_shard([&](auto& s, int r) {
    s.sub_pdlp->pdhg_solver_.spmv_At_into(in_descs[r].get(), out_descs[r].get());
  });
}

template <typename i_t, typename f_t>
void multi_gpu_engine_t<i_t, f_t>::distributed_spmv_A(
  std::vector<rmm::device_uvector<f_t>>& in_bufs,
  std::vector<cusparse_dn_vec_uptr>& in_descs,
  std::vector<cusparse_dn_vec_uptr>& out_descs)
{
  halo_exchange_var_bufs(in_bufs);
  for_each_shard([&](auto& s, int r) {
    s.sub_pdlp->pdhg_solver_.spmv_A_into(in_descs[r].get(), out_descs[r].get());
  });
}

template struct multi_gpu_engine_t<int, double>;
template struct multi_gpu_engine_t<int, float>;

}  // namespace cuopt::mathematical_optimization::pdlp
