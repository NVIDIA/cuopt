/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#include <cuopt/linear_programming/optimization_problem_conversions.hpp>

#include <mip/mip_constants.hpp>

#include <raft/core/copy.hpp>
#include <raft/core/error.hpp>

#include <algorithm>
#include <cstring>
#include <vector>

namespace cuopt {
namespace linear_programming {

template <typename i_t, typename f_t>
optimization_problem_t<i_t, f_t> data_model_view_to_optimization_problem(
  raft::handle_t const* handle_ptr, const cuopt::mps_parser::data_model_view_t<i_t, f_t>& view)
{
  optimization_problem_t<i_t, f_t> op_problem(handle_ptr);
  op_problem.set_maximize(view.get_sense());

  // Set constraint matrix if offsets are present (includes empty problems with offsets=[0])
  if (view.get_constraint_matrix_offsets().size() > 0) {
    op_problem.set_csr_constraint_matrix(view.get_constraint_matrix_values().data(),
                                         view.get_constraint_matrix_values().size(),
                                         view.get_constraint_matrix_indices().data(),
                                         view.get_constraint_matrix_indices().size(),
                                         view.get_constraint_matrix_offsets().data(),
                                         view.get_constraint_matrix_offsets().size());
  }

  if (view.get_constraint_bounds().size() != 0) {
    op_problem.set_constraint_bounds(view.get_constraint_bounds().data(),
                                     view.get_constraint_bounds().size());
  }
  if (view.get_objective_coefficients().size() != 0) {
    op_problem.set_objective_coefficients(view.get_objective_coefficients().data(),
                                          view.get_objective_coefficients().size());
  }
  op_problem.set_objective_scaling_factor(view.get_objective_scaling_factor());
  op_problem.set_objective_offset(view.get_objective_offset());
  if (view.get_variable_lower_bounds().size() != 0) {
    op_problem.set_variable_lower_bounds(view.get_variable_lower_bounds().data(),
                                         view.get_variable_lower_bounds().size());
  }
  if (view.get_variable_upper_bounds().size() != 0) {
    op_problem.set_variable_upper_bounds(view.get_variable_upper_bounds().data(),
                                         view.get_variable_upper_bounds().size());
  }
  if (view.get_variable_types().size() != 0) {
    auto var_types = view.get_variable_types();

    // Check if the pointer is on host or device
    cudaPointerAttributes attrs;
    cudaError_t err = cudaPointerGetAttributes(&attrs, var_types.data());

    std::vector<char> host_var_types(var_types.size());
    if (err == cudaSuccess && attrs.type == cudaMemoryTypeDevice) {
      // Source is on GPU - copy to host
      RAFT_CUDA_TRY(cudaMemcpy(host_var_types.data(),
                               var_types.data(),
                               var_types.size() * sizeof(char),
                               cudaMemcpyDeviceToHost));
    } else {
      // Source is on host (or unregistered) - direct copy
      if (err != cudaSuccess) { cudaGetLastError(); }  // Clear cudaPointerGetAttributes error
      if (err != cudaSuccess && err != cudaErrorInvalidValue) { RAFT_CUDA_TRY(err); }
      std::memcpy(host_var_types.data(), var_types.data(), var_types.size() * sizeof(char));
    }

    std::vector<var_t> enum_variable_types(var_types.size());
    for (std::size_t i = 0; i < var_types.size(); ++i) {
      enum_variable_types[i] = host_var_types[i] == 'I' ? var_t::INTEGER : var_t::CONTINUOUS;
    }
    op_problem.set_variable_types(enum_variable_types.data(), enum_variable_types.size());
  }

  if (view.get_row_types().size() != 0) {
    op_problem.set_row_types(view.get_row_types().data(), view.get_row_types().size());
  }
  if (view.get_constraint_lower_bounds().size() != 0) {
    op_problem.set_constraint_lower_bounds(view.get_constraint_lower_bounds().data(),
                                           view.get_constraint_lower_bounds().size());
  }
  if (view.get_constraint_upper_bounds().size() != 0) {
    op_problem.set_constraint_upper_bounds(view.get_constraint_upper_bounds().data(),
                                           view.get_constraint_upper_bounds().size());
  }

  if (view.get_objective_name().size() != 0) {
    op_problem.set_objective_name(view.get_objective_name());
  }
  if (view.get_problem_name().size() != 0) {
    op_problem.set_problem_name(view.get_problem_name().data());
  }
  if (view.get_variable_names().size() != 0) {
    op_problem.set_variable_names(view.get_variable_names());
  }
  if (view.get_row_names().size() != 0) { op_problem.set_row_names(view.get_row_names()); }

  if (view.has_quadratic_objective()) {
    // Copy quadratic objective from view to vectors first since we need host data
    std::vector<f_t> Q_values(view.get_quadratic_objective_values().size());
    std::vector<i_t> Q_indices(view.get_quadratic_objective_indices().size());
    std::vector<i_t> Q_offsets(view.get_quadratic_objective_offsets().size());

    // Check if the pointer is on host or device
    cudaPointerAttributes attrs;
    cudaError_t err =
      cudaPointerGetAttributes(&attrs, view.get_quadratic_objective_values().data());

    if (err == cudaSuccess && attrs.type == cudaMemoryTypeDevice) {
      // Source is on GPU - copy to host
      RAFT_CUDA_TRY(cudaMemcpy(Q_values.data(),
                               view.get_quadratic_objective_values().data(),
                               Q_values.size() * sizeof(f_t),
                               cudaMemcpyDeviceToHost));
      RAFT_CUDA_TRY(cudaMemcpy(Q_indices.data(),
                               view.get_quadratic_objective_indices().data(),
                               Q_indices.size() * sizeof(i_t),
                               cudaMemcpyDeviceToHost));
      RAFT_CUDA_TRY(cudaMemcpy(Q_offsets.data(),
                               view.get_quadratic_objective_offsets().data(),
                               Q_offsets.size() * sizeof(i_t),
                               cudaMemcpyDeviceToHost));
    } else {
      // Source is on host - direct copy
      if (err != cudaSuccess) { cudaGetLastError(); }  // Clear cudaPointerGetAttributes error
      if (err != cudaSuccess && err != cudaErrorInvalidValue) { RAFT_CUDA_TRY(err); }
      std::memcpy(Q_values.data(),
                  view.get_quadratic_objective_values().data(),
                  Q_values.size() * sizeof(f_t));
      std::memcpy(Q_indices.data(),
                  view.get_quadratic_objective_indices().data(),
                  Q_indices.size() * sizeof(i_t));
      std::memcpy(Q_offsets.data(),
                  view.get_quadratic_objective_offsets().data(),
                  Q_offsets.size() * sizeof(i_t));
    }

    op_problem.set_quadratic_objective_matrix(Q_values.data(),
                                              Q_values.size(),
                                              Q_indices.data(),
                                              Q_indices.size(),
                                              Q_offsets.data(),
                                              Q_offsets.size());
  }

  return op_problem;
}

template <typename i_t, typename f_t>
optimization_problem_t<i_t, f_t> mps_data_model_to_optimization_problem(
  raft::handle_t const* handle_ptr, const cuopt::mps_parser::mps_data_model_t<i_t, f_t>& data_model)
{
  optimization_problem_t<i_t, f_t> op_problem(handle_ptr);
  op_problem.set_maximize(data_model.get_sense());

  op_problem.set_csr_constraint_matrix(data_model.get_constraint_matrix_values().data(),
                                       data_model.get_constraint_matrix_values().size(),
                                       data_model.get_constraint_matrix_indices().data(),
                                       data_model.get_constraint_matrix_indices().size(),
                                       data_model.get_constraint_matrix_offsets().data(),
                                       data_model.get_constraint_matrix_offsets().size());

  if (data_model.get_constraint_bounds().size() != 0) {
    op_problem.set_constraint_bounds(data_model.get_constraint_bounds().data(),
                                     data_model.get_constraint_bounds().size());
  }
  if (data_model.get_objective_coefficients().size() != 0) {
    op_problem.set_objective_coefficients(data_model.get_objective_coefficients().data(),
                                          data_model.get_objective_coefficients().size());
  }
  op_problem.set_objective_scaling_factor(data_model.get_objective_scaling_factor());
  op_problem.set_objective_offset(data_model.get_objective_offset());
  if (data_model.get_variable_lower_bounds().size() != 0) {
    op_problem.set_variable_lower_bounds(data_model.get_variable_lower_bounds().data(),
                                         data_model.get_variable_lower_bounds().size());
  }
  if (data_model.get_variable_upper_bounds().size() != 0) {
    op_problem.set_variable_upper_bounds(data_model.get_variable_upper_bounds().data(),
                                         data_model.get_variable_upper_bounds().size());
  }
  if (data_model.get_variable_types().size() != 0) {
    std::vector<var_t> enum_variable_types(data_model.get_variable_types().size());
    std::transform(
      data_model.get_variable_types().cbegin(),
      data_model.get_variable_types().cend(),
      enum_variable_types.begin(),
      [](const auto val) -> var_t { return val == 'I' ? var_t::INTEGER : var_t::CONTINUOUS; });
    op_problem.set_variable_types(enum_variable_types.data(), enum_variable_types.size());
  }

  if (data_model.get_row_types().size() != 0) {
    op_problem.set_row_types(data_model.get_row_types().data(), data_model.get_row_types().size());
  }
  if (data_model.get_constraint_lower_bounds().size() != 0) {
    op_problem.set_constraint_lower_bounds(data_model.get_constraint_lower_bounds().data(),
                                           data_model.get_constraint_lower_bounds().size());
  }
  if (data_model.get_constraint_upper_bounds().size() != 0) {
    op_problem.set_constraint_upper_bounds(data_model.get_constraint_upper_bounds().data(),
                                           data_model.get_constraint_upper_bounds().size());
  }

  if (data_model.get_objective_name().size() != 0) {
    op_problem.set_objective_name(data_model.get_objective_name());
  }
  if (data_model.get_problem_name().size() != 0) {
    op_problem.set_problem_name(data_model.get_problem_name().data());
  }
  if (data_model.get_variable_names().size() != 0) {
    op_problem.set_variable_names(data_model.get_variable_names());
  }
  if (data_model.get_row_names().size() != 0) {
    op_problem.set_row_names(data_model.get_row_names());
  }

  if (data_model.get_quadratic_objective_values().size() != 0) {
    const std::vector<f_t> Q_values  = data_model.get_quadratic_objective_values();
    const std::vector<i_t> Q_indices = data_model.get_quadratic_objective_indices();
    const std::vector<i_t> Q_offsets = data_model.get_quadratic_objective_offsets();
    op_problem.set_quadratic_objective_matrix(Q_values.data(),
                                              Q_values.size(),
                                              Q_indices.data(),
                                              Q_indices.size(),
                                              Q_offsets.data(),
                                              Q_offsets.size());
  }

  return op_problem;
}

template <typename i_t, typename f_t>
cuopt::mps_parser::data_model_view_t<i_t, f_t> create_view_from_mps_data_model(
  const cuopt::mps_parser::mps_data_model_t<i_t, f_t>& mps_data_model)
{
  cuopt::mps_parser::data_model_view_t<i_t, f_t> view;

  view.set_maximize(mps_data_model.get_sense());

  // Always set constraint matrix if offsets exist (even for empty problems with 0 constraints)
  // Validation requires at least offsets=[0] to be set
  if (!mps_data_model.get_constraint_matrix_offsets().empty()) {
    view.set_csr_constraint_matrix(mps_data_model.get_constraint_matrix_values().data(),
                                   mps_data_model.get_constraint_matrix_values().size(),
                                   mps_data_model.get_constraint_matrix_indices().data(),
                                   mps_data_model.get_constraint_matrix_indices().size(),
                                   mps_data_model.get_constraint_matrix_offsets().data(),
                                   mps_data_model.get_constraint_matrix_offsets().size());
  }

  if (!mps_data_model.get_constraint_bounds().empty()) {
    view.set_constraint_bounds(mps_data_model.get_constraint_bounds().data(),
                               mps_data_model.get_constraint_bounds().size());
  }

  if (!mps_data_model.get_objective_coefficients().empty()) {
    view.set_objective_coefficients(mps_data_model.get_objective_coefficients().data(),
                                    mps_data_model.get_objective_coefficients().size());
  }

  if (mps_data_model.has_quadratic_objective()) {
    view.set_quadratic_objective_matrix(mps_data_model.get_quadratic_objective_values().data(),
                                        mps_data_model.get_quadratic_objective_values().size(),
                                        mps_data_model.get_quadratic_objective_indices().data(),
                                        mps_data_model.get_quadratic_objective_indices().size(),
                                        mps_data_model.get_quadratic_objective_offsets().data(),
                                        mps_data_model.get_quadratic_objective_offsets().size());
  }

  view.set_objective_scaling_factor(mps_data_model.get_objective_scaling_factor());
  view.set_objective_offset(mps_data_model.get_objective_offset());

  if (!mps_data_model.get_variable_lower_bounds().empty()) {
    view.set_variable_lower_bounds(mps_data_model.get_variable_lower_bounds().data(),
                                   mps_data_model.get_variable_lower_bounds().size());
  }

  if (!mps_data_model.get_variable_upper_bounds().empty()) {
    view.set_variable_upper_bounds(mps_data_model.get_variable_upper_bounds().data(),
                                   mps_data_model.get_variable_upper_bounds().size());
  }

  if (!mps_data_model.get_variable_types().empty()) {
    view.set_variable_types(mps_data_model.get_variable_types().data(),
                            mps_data_model.get_variable_types().size());
  }

  if (!mps_data_model.get_row_types().empty()) {
    view.set_row_types(mps_data_model.get_row_types().data(),
                       mps_data_model.get_row_types().size());
  }

  if (!mps_data_model.get_constraint_lower_bounds().empty()) {
    view.set_constraint_lower_bounds(mps_data_model.get_constraint_lower_bounds().data(),
                                     mps_data_model.get_constraint_lower_bounds().size());
  }

  if (!mps_data_model.get_constraint_upper_bounds().empty()) {
    view.set_constraint_upper_bounds(mps_data_model.get_constraint_upper_bounds().data(),
                                     mps_data_model.get_constraint_upper_bounds().size());
  }

  view.set_objective_name(mps_data_model.get_objective_name());
  view.set_problem_name(mps_data_model.get_problem_name());

  if (!mps_data_model.get_variable_names().empty()) {
    view.set_variable_names(mps_data_model.get_variable_names());
  }

  if (!mps_data_model.get_row_names().empty()) {
    view.set_row_names(mps_data_model.get_row_names());
  }

  if (!mps_data_model.get_initial_primal_solution().empty()) {
    view.set_initial_primal_solution(mps_data_model.get_initial_primal_solution().data(),
                                     mps_data_model.get_initial_primal_solution().size());
  }

  if (!mps_data_model.get_initial_dual_solution().empty()) {
    view.set_initial_dual_solution(mps_data_model.get_initial_dual_solution().data(),
                                   mps_data_model.get_initial_dual_solution().size());
  }

  view.set_is_device_memory(false);  // MPS data is always in CPU memory
  return view;
}

template <typename i_t, typename f_t>
cuopt::mps_parser::data_model_view_t<i_t, f_t> cpu_problem_data_t<i_t, f_t>::create_view() const
{
  cuopt::mps_parser::data_model_view_t<i_t, f_t> v;
  v.set_maximize(maximize);
  v.set_objective_scaling_factor(objective_scaling_factor);
  v.set_objective_offset(objective_offset);

  if (!A_values.empty()) {
    v.set_csr_constraint_matrix(A_values.data(),
                                A_values.size(),
                                A_indices.data(),
                                A_indices.size(),
                                A_offsets.data(),
                                A_offsets.size());
  }
  if (!constraint_bounds.empty()) {
    v.set_constraint_bounds(constraint_bounds.data(), constraint_bounds.size());
  }
  if (!constraint_lower_bounds.empty()) {
    v.set_constraint_lower_bounds(constraint_lower_bounds.data(), constraint_lower_bounds.size());
  }
  if (!constraint_upper_bounds.empty()) {
    v.set_constraint_upper_bounds(constraint_upper_bounds.data(), constraint_upper_bounds.size());
  }
  if (!objective_coefficients.empty()) {
    v.set_objective_coefficients(objective_coefficients.data(), objective_coefficients.size());
  }
  if (!variable_lower_bounds.empty()) {
    v.set_variable_lower_bounds(variable_lower_bounds.data(), variable_lower_bounds.size());
  }
  if (!variable_upper_bounds.empty()) {
    v.set_variable_upper_bounds(variable_upper_bounds.data(), variable_upper_bounds.size());
  }
  if (!variable_types.empty()) {
    v.set_variable_types(variable_types.data(), variable_types.size());
  }
  if (!quadratic_objective_values.empty()) {
    v.set_quadratic_objective_matrix(quadratic_objective_values.data(),
                                     quadratic_objective_values.size(),
                                     quadratic_objective_indices.data(),
                                     quadratic_objective_indices.size(),
                                     quadratic_objective_offsets.data(),
                                     quadratic_objective_offsets.size());
  }
  v.set_is_device_memory(false);
  return v;
}

template <typename i_t, typename f_t>
cpu_problem_data_t<i_t, f_t> copy_view_to_cpu(
  raft::handle_t const* handle_ptr,
  const cuopt::mps_parser::data_model_view_t<i_t, f_t>& gpu_view)
{
  cpu_problem_data_t<i_t, f_t> cpu_data;
  auto stream = handle_ptr->get_stream();

  cpu_data.maximize                 = gpu_view.get_sense();
  cpu_data.objective_scaling_factor = gpu_view.get_objective_scaling_factor();
  cpu_data.objective_offset         = gpu_view.get_objective_offset();

  auto copy_to_host = [stream](auto& dst_vec, auto src_span) {
    if (src_span.size() > 0) {
      dst_vec.resize(src_span.size());
      raft::copy(dst_vec.data(), src_span.data(), src_span.size(), stream);
    }
  };

  copy_to_host(cpu_data.A_values, gpu_view.get_constraint_matrix_values());
  copy_to_host(cpu_data.A_indices, gpu_view.get_constraint_matrix_indices());
  copy_to_host(cpu_data.A_offsets, gpu_view.get_constraint_matrix_offsets());
  copy_to_host(cpu_data.constraint_bounds, gpu_view.get_constraint_bounds());
  copy_to_host(cpu_data.constraint_lower_bounds, gpu_view.get_constraint_lower_bounds());
  copy_to_host(cpu_data.constraint_upper_bounds, gpu_view.get_constraint_upper_bounds());
  copy_to_host(cpu_data.objective_coefficients, gpu_view.get_objective_coefficients());
  copy_to_host(cpu_data.variable_lower_bounds, gpu_view.get_variable_lower_bounds());
  copy_to_host(cpu_data.variable_upper_bounds, gpu_view.get_variable_upper_bounds());
  copy_to_host(cpu_data.quadratic_objective_values, gpu_view.get_quadratic_objective_values());
  copy_to_host(cpu_data.quadratic_objective_indices, gpu_view.get_quadratic_objective_indices());
  copy_to_host(cpu_data.quadratic_objective_offsets, gpu_view.get_quadratic_objective_offsets());

  // Variable types need special handling (char array)
  auto var_types_span = gpu_view.get_variable_types();
  if (var_types_span.size() > 0) {
    cpu_data.variable_types.resize(var_types_span.size());

    // Check if variable_types is host-backed or device-backed
    cudaPointerAttributes attrs;
    cudaError_t err = cudaPointerGetAttributes(&attrs, var_types_span.data());

    if (err == cudaSuccess && attrs.type == cudaMemoryTypeDevice) {
      // Device memory - use async copy
      RAFT_CUDA_TRY(cudaMemcpyAsync(cpu_data.variable_types.data(),
                                    var_types_span.data(),
                                    var_types_span.size() * sizeof(char),
                                    cudaMemcpyDeviceToHost,
                                    stream));
    } else {
      // Host memory or unregistered - use direct copy
      if (err != cudaSuccess && err != cudaErrorInvalidValue) { RAFT_CUDA_TRY(err); }
      std::memcpy(cpu_data.variable_types.data(),
                  var_types_span.data(),
                  var_types_span.size() * sizeof(char));
    }
  }

  // Synchronize to ensure all copies are complete
  RAFT_CUDA_TRY(cudaStreamSynchronize(stream));

  return cpu_data;
}

// Explicit template instantiations
#define INSTANTIATE(F_TYPE)                                                                \
  template optimization_problem_t<int, F_TYPE> data_model_view_to_optimization_problem(    \
    raft::handle_t const* handle_ptr,                                                      \
    const cuopt::mps_parser::data_model_view_t<int, F_TYPE>& view);                        \
                                                                                            \
  template optimization_problem_t<int, F_TYPE> mps_data_model_to_optimization_problem(     \
    raft::handle_t const* handle_ptr,                                                      \
    const cuopt::mps_parser::mps_data_model_t<int, F_TYPE>& data_model);                   \
                                                                                            \
  template cuopt::mps_parser::data_model_view_t<int, F_TYPE> create_view_from_mps_data_model( \
    const cuopt::mps_parser::mps_data_model_t<int, F_TYPE>& mps_data_model);               \
                                                                                            \
  template struct cpu_problem_data_t<int, F_TYPE>;                                         \
                                                                                            \
  template cpu_problem_data_t<int, F_TYPE> copy_view_to_cpu(                               \
    raft::handle_t const* handle_ptr,                                                      \
    const cuopt::mps_parser::data_model_view_t<int, F_TYPE>& gpu_view);

#if MIP_INSTANTIATE_FLOAT
INSTANTIATE(float)
#endif

#if MIP_INSTANTIATE_DOUBLE
INSTANTIATE(double)
#endif

#undef INSTANTIATE

}  // namespace linear_programming
}  // namespace cuopt
