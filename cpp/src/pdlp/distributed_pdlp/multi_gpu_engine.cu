/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

 #include <pdlp/distributed_pdlp/multi_gpu_engine.hpp>

 #include <cuopt/error.hpp>
 
 #include <raft/core/device_setter.hpp>
 
 #include <nccl.h>
 
 #include <numeric>
 
 namespace cuopt::linear_programming::detail {
 
 template <typename i_t, typename f_t>
 multi_gpu_engine_t<i_t, f_t>::multi_gpu_engine_t(
   std::vector<rank_data_t<i_t, f_t>>&&      rank_data,
   std::vector<f_t> const&                   h_global_obj,
   std::vector<f_t> const&                   h_global_var_lower,
   std::vector<f_t> const&                   h_global_var_upper,
   std::vector<f_t> const&                   h_global_cstr_lower,
   std::vector<f_t> const&                   h_global_cstr_upper,
   bool                                      maximize,
   f_t                                       objective_offset,
   f_t                                       objective_scaling_factor,
   pdlp_solver_settings_t<i_t, f_t> const&   sub_solver_settings)
   : stream()
 {
   const int nb_parts = static_cast<int>(rank_data.size());
   cuopt_expects(nb_parts > 0,
                 error_type_t::ValidationError,
                 "multi_gpu_engine_t: rank_data must be non-empty");
 
   shards.reserve(nb_parts);
 
   // 1:1 rank -> device mapping. (Matches metis_tests; refine later if needed.)
   std::vector<int> devices(nb_parts);
   std::iota(devices.begin(), devices.end(), 0);
 
   // 2. Collectively bootstrap NCCL communicators across all devices.
   //    Must be done together; each comm is then handed to one shard,
   //    which wraps it in a unique_ptr with the device-aware deleter.
   std::vector<ncclComm_t> raw_comms(nb_parts);
   cuopt_expects(ncclCommInitAll(raw_comms.data(), nb_parts, devices.data()) == ncclSuccess,
                 error_type_t::RuntimeError,
                 "ncclCommInitAll failed");
 
   // 3. Construct one shard per rank, pinned to its device.
   for (int r = 0; r < nb_parts; ++r) {
     raft::device_setter guard(devices[r]);  // shard ctor asserts current device
     shards.emplace_back(std::make_unique<pdlp_shard_t<i_t, f_t>>(
       devices[r],
       std::move(rank_data[r]),
       raw_comms[r],
       h_global_obj,
       h_global_var_lower,
       h_global_var_upper,
       h_global_cstr_lower,
       h_global_cstr_upper,
       maximize,
       objective_offset,
       objective_scaling_factor,
       sub_solver_settings));
   }
 }
 
 template struct multi_gpu_engine_t<int, double>;
 // template struct multi_gpu_engine_t<int, float>;
 
 }  // namespace cuopt::linear_programming::detail