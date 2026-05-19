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
  multi_gpu_engine_t(
    std::vector<rank_data_t<i_t, f_t>>&&      rank_data,
    std::vector<f_t> const&                   h_global_obj,
    std::vector<f_t> const&                   h_global_var_lower,
    std::vector<f_t> const&                   h_global_var_upper,
    std::vector<f_t> const&                   h_global_cstr_lower,
    std::vector<f_t> const&                   h_global_cstr_upper,
    std::vector<f_t> const&                   h_global_obj_scaled,
    std::vector<f_t> const&                   h_global_var_lower_scaled,
    std::vector<f_t> const&                   h_global_var_upper_scaled,
    std::vector<f_t> const&                   h_global_cstr_lower_scaled,
    std::vector<f_t> const&                   h_global_cstr_upper_scaled,
    std::vector<f_t> const&                   h_global_cummulative_cstr_scaling,
    std::vector<f_t> const&                   h_global_cummulative_var_scaling,
    f_t                                       h_bound_rescaling,
    f_t                                       h_objective_rescaling,
    bool                                      maximize,
    f_t                                       objective_offset,
    f_t                                       objective_scaling_factor,
    pdlp_solver_settings_t<i_t, f_t> const&   sub_solver_settings);
 
   multi_gpu_engine_t(const multi_gpu_engine_t&)            = delete;
   multi_gpu_engine_t& operator=(const multi_gpu_engine_t&) = delete;
 
   // Engine-level stream for fork/join orchestration (master side).
   rmm::cuda_stream stream;
 
   // Shards stored by unique_ptr because pdlp_shard_t is immovable
   // (owns device-affine resources: handle, NCCL comm, RMM buffers).
   std::vector<std::unique_ptr<pdlp_shard_t<i_t, f_t>>> shards;
 };
 
 }  // namespace cuopt::linear_programming::detail