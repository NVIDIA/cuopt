/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <pdlp/distributed_pdlp/rank_data.hpp>
#include <pdlp/distributed_pdlp/shard.hpp>

#include <cuopt/linear_programming/pdlp/solver_settings.hpp>

#include <rmm/cuda_stream.hpp>

#include <memory>
#include <vector>

namespace cuopt::linear_programming::detail {

template <typename i_t, typename f_t>
struct multi_gpu_engine_t {
  // Constructs shards from rank_data
  multi_gpu_engine_t(std::vector<rank_data_t<i_t, f_t>>&& rank_data,
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
                     pdlp_solver_settings_t<i_t, f_t> const& sub_solver_settings);

  multi_gpu_engine_t(const multi_gpu_engine_t&)            = delete;
  multi_gpu_engine_t& operator=(const multi_gpu_engine_t&) = delete;



  template <typename Fn>
  void for_each_shard(Fn&& fn)
  {
    for (auto& s : shards) {
      raft::device_setter guard(s->device_id);   
      fn(*s);                                     
    }
  }

  template <typename... InAccess,
          typename OutAccess,
          typename SizeAccess,
          typename Op>
  void distributed_transform(std::tuple<InAccess...> in_accessors,
                            OutAccess                out,
                            SizeAccess               sz,
                            Op                       op)
  {
    for_each_shard([&](auto& shard) {
      auto& sub = *shard.sub_pdlp;
      // turns the Tuple of lambdas into a tuple of rmm::device_uvector
      auto cub_inputs = std::apply(
        [&sub](auto&... acc) { return cuda::std::make_tuple(acc(sub)...); },
        in_accessors);

      cub::DeviceTransform::Transform(cub_inputs,
                                      out(sub),
                                      sz(sub),
                                      op,
                                      shard.stream.view());
    });
  }
  // --- 2) convenience: single input accessor (delegates) ---
  template <typename InAccess,
  typename OutAccess,
  typename SizeAccess,
  typename Op>
  void distributed_transform(InAccess   in,
                  OutAccess  out,
                  SizeAccess sz,
                  Op         op)
  {
  distributed_transform(std::make_tuple(in), out, sz, op);
  }

  // Engine-level stream for fork/join orchestration (master side).
  rmm::cuda_stream stream;

  // Shards stored by unique_ptr because pdlp_shard_t is immovable
  // (owns device-affine resources: handle, NCCL comm, RMM buffers).
  std::vector<std::unique_ptr<pdlp_shard_t<i_t, f_t>>> shards;
};

}  // namespace cuopt::linear_programming::detail
