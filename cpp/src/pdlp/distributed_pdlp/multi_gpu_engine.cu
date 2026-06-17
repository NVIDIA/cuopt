/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <pdlp/distributed_pdlp/multi_gpu_engine.hpp>
#include <pdlp/pdlp.cuh>

#include <cuopt/error.hpp>

#include <raft/core/device_setter.hpp>

#include <nccl.h>

#include <chrono>
#include <numeric>

#include <utilities/logger.hpp>

namespace cuopt::linear_programming::detail {

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
  CUOPT_LOG_INFO("distributed_pdlp: building %d shard solver(s) ...", nb_parts);
  auto shard_build_t0 = std::chrono::high_resolution_clock::now();
  for (int r = 0; r < nb_parts; ++r) {
    raft::device_setter guard(devices[r]);  // shard ctor needs device set
    shards.emplace_back(std::make_unique<pdlp_shard_t<i_t, f_t>>(
      devices[r], std::move(rank_data[r]), std::move(comms[r]), mps, sub_solver_settings));
  }
  auto shard_build_t1 = std::chrono::high_resolution_clock::now();
  CUOPT_LOG_INFO("distributed_pdlp: shard build done in %.3f s",
                 std::chrono::duration<double>(shard_build_t1 - shard_build_t0).count());

  // Two different events
  // capture_*_event_ are used inside graph capture
  // ext_*_event_ are used when sync is needed outside of graph
  graph_master_ready_event_ = std::make_unique<cuopt::event_handler_t>();
  sync_master_ready_event_  = std::make_unique<cuopt::event_handler_t>();
  graph_shard_ready_events_.reserve(nb_parts);
  sync_shard_ready_events_.reserve(nb_parts);
  for (int r = 0; r < nb_parts; ++r) {
    raft::device_setter guard(devices[r]);
    graph_shard_ready_events_.emplace_back(std::make_unique<cuopt::event_handler_t>());
    sync_shard_ready_events_.emplace_back(std::make_unique<cuopt::event_handler_t>());
  }
}

// -------- High-level: A @ x and A_T @ y -----------------------------------
template <typename i_t, typename f_t>
void multi_gpu_engine_t<i_t, f_t>::distributed_compute_A_x()
{
  halo_exchange_var(
    [](auto& pdhg) -> rmm::device_uvector<f_t>& { return pdhg.get_reflected_primal(); });
  for_each_shard([](auto& shard) { shard.sub_pdlp->pdhg_solver_.spmvop_A_x(); });
}

template <typename i_t, typename f_t>
void multi_gpu_engine_t<i_t, f_t>::distributed_compute_At_y()
{
  halo_exchange_cstr(
    [](auto& pdhg) -> rmm::device_uvector<f_t>& { return pdhg.get_dual_solution(); });
  for_each_shard([](auto& shard) { shard.sub_pdlp->pdhg_solver_.spmvop_At_y(); });
}

// -------- Cross-stream fork / join / sync ---------------------------------
template <typename i_t, typename f_t>
void multi_gpu_engine_t<i_t, f_t>::graph_capture_fork_to_shards(
  rmm::cuda_stream_view master_stream)
{
  graph_master_ready_event_->record(master_stream);
  for (auto& s : shards) {
    raft::device_setter guard(s->device_id);
    graph_master_ready_event_->stream_wait(s->stream.view());
  }
}

template <typename i_t, typename f_t>
void multi_gpu_engine_t<i_t, f_t>::graph_capture_join_from_shards(
  rmm::cuda_stream_view master_stream)
{
  const int nb = static_cast<int>(shards.size());
  for (int r = 0; r < nb; ++r) {
    raft::device_setter guard(shards[r]->device_id);
    graph_shard_ready_events_[r]->record(shards[r]->stream.view());
  }
  for (auto& e : graph_shard_ready_events_) {
    e->stream_wait(master_stream);
  }
}

template <typename i_t, typename f_t>
void multi_gpu_engine_t<i_t, f_t>::sync_await_master(rmm::cuda_stream_view master_stream)
{
  sync_master_ready_event_->record(master_stream);
  for (auto& s : shards) {
    raft::device_setter guard(s->device_id);
    sync_master_ready_event_->stream_wait(s->stream.view());
  }
}

template <typename i_t, typename f_t>
void multi_gpu_engine_t<i_t, f_t>::sync_await_shards(rmm::cuda_stream_view master_stream)
{
  const int nb = static_cast<int>(shards.size());
  for (int r = 0; r < nb; ++r) {
    raft::device_setter guard(shards[r]->device_id);
    sync_shard_ready_events_[r]->record(shards[r]->stream.view());
  }
  for (auto& e : sync_shard_ready_events_) {
    e->stream_wait(master_stream);
  }
}

template struct multi_gpu_engine_t<int, double>;
template struct multi_gpu_engine_t<int, float>;

}  // namespace cuopt::linear_programming::detail
