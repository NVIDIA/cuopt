/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#include <cuopt/linear_programming/cpu_pdlp_warm_start_data.hpp>
#include <cuopt/linear_programming/pdlp/pdlp_warm_start_data.hpp>
#include <mip_heuristics/mip_constants.hpp>
#include <raft/core/copy.hpp>

namespace cuopt::linear_programming {

// Helper to copy device_uvector to std::vector (D2H)
template <typename T>
std::vector<T> device_to_host_vector(const rmm::device_uvector<T>& device_vec,
                                     rmm::cuda_stream_view stream)
{
  if (device_vec.size() == 0) return std::vector<T>();

  std::vector<T> host_vec(device_vec.size());
  raft::copy(host_vec.data(), device_vec.data(), device_vec.size(), stream);
  stream.synchronize();
  return host_vec;
}

// Helper to copy std::vector to device_uvector (H2D)
template <typename T>
rmm::device_uvector<T> host_to_device_vector(const std::vector<T>& host_vec,
                                             rmm::cuda_stream_view stream)
{
  if (host_vec.empty()) return rmm::device_uvector<T>(0, stream);

  rmm::device_uvector<T> device_vec(host_vec.size(), stream);
  raft::copy(device_vec.data(), host_vec.data(), host_vec.size(), stream);
  stream.synchronize();
  return device_vec;
}

// Convert GPU → CPU warmstart (D2H copy)
template <typename i_t, typename f_t>
cpu_pdlp_warm_start_data_t<i_t, f_t> convert_to_cpu_warmstart(
  const pdlp_warm_start_data_t<i_t, f_t>& gpu_data, rmm::cuda_stream_view stream)
{
  cpu_pdlp_warm_start_data_t<i_t, f_t> cpu_data;

  // Copy all vector fields from GPU to CPU
  cpu_data.current_primal_solution_ =
    device_to_host_vector(gpu_data.current_primal_solution_, stream);
  cpu_data.current_dual_solution_ = device_to_host_vector(gpu_data.current_dual_solution_, stream);
  cpu_data.initial_primal_average_ =
    device_to_host_vector(gpu_data.initial_primal_average_, stream);
  cpu_data.initial_dual_average_ = device_to_host_vector(gpu_data.initial_dual_average_, stream);
  cpu_data.current_ATY_          = device_to_host_vector(gpu_data.current_ATY_, stream);
  cpu_data.sum_primal_solutions_ = device_to_host_vector(gpu_data.sum_primal_solutions_, stream);
  cpu_data.sum_dual_solutions_   = device_to_host_vector(gpu_data.sum_dual_solutions_, stream);
  cpu_data.last_restart_duality_gap_primal_solution_ =
    device_to_host_vector(gpu_data.last_restart_duality_gap_primal_solution_, stream);
  cpu_data.last_restart_duality_gap_dual_solution_ =
    device_to_host_vector(gpu_data.last_restart_duality_gap_dual_solution_, stream);

  // Copy scalar fields
  cpu_data.initial_primal_weight_         = gpu_data.initial_primal_weight_;
  cpu_data.initial_step_size_             = gpu_data.initial_step_size_;
  cpu_data.total_pdlp_iterations_         = gpu_data.total_pdlp_iterations_;
  cpu_data.total_pdhg_iterations_         = gpu_data.total_pdhg_iterations_;
  cpu_data.last_candidate_kkt_score_      = gpu_data.last_candidate_kkt_score_;
  cpu_data.last_restart_kkt_score_        = gpu_data.last_restart_kkt_score_;
  cpu_data.sum_solution_weight_           = gpu_data.sum_solution_weight_;
  cpu_data.iterations_since_last_restart_ = gpu_data.iterations_since_last_restart_;

  return cpu_data;
}

// Convert CPU → GPU warmstart (H2D copy)
template <typename i_t, typename f_t>
pdlp_warm_start_data_t<i_t, f_t> convert_to_gpu_warmstart(
  const cpu_pdlp_warm_start_data_t<i_t, f_t>& cpu_data, rmm::cuda_stream_view stream)
{
  pdlp_warm_start_data_t<i_t, f_t> gpu_data;

  // Copy all vector fields from CPU to GPU
  gpu_data.current_primal_solution_ =
    host_to_device_vector(cpu_data.current_primal_solution_, stream);
  gpu_data.current_dual_solution_ = host_to_device_vector(cpu_data.current_dual_solution_, stream);
  gpu_data.initial_primal_average_ =
    host_to_device_vector(cpu_data.initial_primal_average_, stream);
  gpu_data.initial_dual_average_ = host_to_device_vector(cpu_data.initial_dual_average_, stream);
  gpu_data.current_ATY_          = host_to_device_vector(cpu_data.current_ATY_, stream);
  gpu_data.sum_primal_solutions_ = host_to_device_vector(cpu_data.sum_primal_solutions_, stream);
  gpu_data.sum_dual_solutions_   = host_to_device_vector(cpu_data.sum_dual_solutions_, stream);
  gpu_data.last_restart_duality_gap_primal_solution_ =
    host_to_device_vector(cpu_data.last_restart_duality_gap_primal_solution_, stream);
  gpu_data.last_restart_duality_gap_dual_solution_ =
    host_to_device_vector(cpu_data.last_restart_duality_gap_dual_solution_, stream);

  // Copy scalar fields
  gpu_data.initial_primal_weight_         = cpu_data.initial_primal_weight_;
  gpu_data.initial_step_size_             = cpu_data.initial_step_size_;
  gpu_data.total_pdlp_iterations_         = cpu_data.total_pdlp_iterations_;
  gpu_data.total_pdhg_iterations_         = cpu_data.total_pdhg_iterations_;
  gpu_data.last_candidate_kkt_score_      = cpu_data.last_candidate_kkt_score_;
  gpu_data.last_restart_kkt_score_        = cpu_data.last_restart_kkt_score_;
  gpu_data.sum_solution_weight_           = cpu_data.sum_solution_weight_;
  gpu_data.iterations_since_last_restart_ = cpu_data.iterations_since_last_restart_;

  return gpu_data;
}

// Explicit template instantiations
template cpu_pdlp_warm_start_data_t<int, double> convert_to_cpu_warmstart(
  const pdlp_warm_start_data_t<int, double>&, rmm::cuda_stream_view);

template pdlp_warm_start_data_t<int, double> convert_to_gpu_warmstart(
  const cpu_pdlp_warm_start_data_t<int, double>&, rmm::cuda_stream_view);

#if MIP_INSTANTIATE_FLOAT
template cpu_pdlp_warm_start_data_t<int, float> convert_to_cpu_warmstart(
  const pdlp_warm_start_data_t<int, float>&, rmm::cuda_stream_view);

template pdlp_warm_start_data_t<int, float> convert_to_gpu_warmstart(
  const cpu_pdlp_warm_start_data_t<int, float>&, rmm::cuda_stream_view);
#endif

}  // namespace cuopt::linear_programming
