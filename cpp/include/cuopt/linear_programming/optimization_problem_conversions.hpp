/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#pragma once

#include <cuopt/linear_programming/optimization_problem.hpp>
#include <mps_parser/data_model_view.hpp>
#include <mps_parser/mps_data_model.hpp>

#include <raft/core/handle.hpp>

namespace cuopt {
namespace linear_programming {

/**
 * @brief Convert a data_model_view_t to an optimization_problem_t.
 * 
 * This function creates a GPU-resident optimization problem from a view that can
 * point to either CPU or GPU memory. The view's pointers are checked at runtime to
 * determine their memory location, and appropriate copy operations are performed.
 * 
 * @tparam i_t Integer type (typically int)
 * @tparam f_t Floating-point type (float or double)
 * @param handle_ptr RAFT handle for GPU operations. Must not be null.
 * @param view Non-owning view pointing to problem data (CPU or GPU memory)
 * @return optimization_problem_t<i_t, f_t> GPU-resident optimization problem
 * 
 * @note This function handles the conversion from both CPU and GPU source data.
 *       Variable types and quadratic objective data are always copied to CPU first
 *       before being set on the optimization_problem_t (as required by the API).
 */
template <typename i_t, typename f_t>
optimization_problem_t<i_t, f_t> data_model_view_to_optimization_problem(
  raft::handle_t const* handle_ptr,
  const cuopt::mps_parser::data_model_view_t<i_t, f_t>& view);

/**
 * @brief Convert an mps_data_model_t to an optimization_problem_t.
 * 
 * This function creates a GPU-resident optimization problem from MPS data that
 * resides in CPU memory. All data is copied from CPU to GPU.
 * 
 * @tparam i_t Integer type (typically int)
 * @tparam f_t Floating-point type (float or double)
 * @param handle_ptr RAFT handle for GPU operations. Must not be null.
 * @param data_model MPS data model with problem data in CPU memory
 * @return optimization_problem_t<i_t, f_t> GPU-resident optimization problem
 * 
 * @note All data in mps_data_model_t is in CPU memory (std::vector).
 *       This function performs CPU → GPU copies for all problem data.
 */
template <typename i_t, typename f_t>
optimization_problem_t<i_t, f_t> mps_data_model_to_optimization_problem(
  raft::handle_t const* handle_ptr,
  const cuopt::mps_parser::mps_data_model_t<i_t, f_t>& data_model);

/**
 * @brief Create a data_model_view_t from an mps_data_model_t.
 * 
 * This helper function creates a non-owning view pointing to the CPU memory
 * in an mps_data_model_t. The view can be used for remote solves or for
 * creating an optimization_problem_t.
 * 
 * @tparam i_t Integer type (typically int)
 * @tparam f_t Floating-point type (float or double)
 * @param mps_data_model MPS data model with problem data in CPU memory
 * @return data_model_view_t<i_t, f_t> Non-owning view with is_device_memory_=false
 * 
 * @note The returned view points to memory owned by mps_data_model.
 *       The mps_data_model must remain alive while the view is in use.
 */
template <typename i_t, typename f_t>
cuopt::mps_parser::data_model_view_t<i_t, f_t> create_view_from_mps_data_model(
  const cuopt::mps_parser::mps_data_model_t<i_t, f_t>& mps_data_model);

/**
 * @brief Helper struct to hold CPU copies of GPU problem data.
 * 
 * This struct is used when GPU data needs to be copied to CPU for remote solve.
 * It provides a create_view() method to create a data_model_view_t pointing to
 * its CPU memory.
 * 
 * @tparam i_t Integer type (typically int)
 * @tparam f_t Floating-point type (float or double)
 */
template <typename i_t, typename f_t>
struct cpu_problem_data_t {
  std::vector<f_t> A_values;
  std::vector<i_t> A_indices;
  std::vector<i_t> A_offsets;
  std::vector<f_t> constraint_bounds;
  std::vector<f_t> constraint_lower_bounds;
  std::vector<f_t> constraint_upper_bounds;
  std::vector<f_t> objective_coefficients;
  std::vector<f_t> variable_lower_bounds;
  std::vector<f_t> variable_upper_bounds;
  std::vector<char> variable_types;
  std::vector<f_t> quadratic_objective_values;
  std::vector<i_t> quadratic_objective_indices;
  std::vector<i_t> quadratic_objective_offsets;
  bool maximize;
  f_t objective_scaling_factor;
  f_t objective_offset;

  /**
   * @brief Create a data_model_view_t pointing to this CPU data.
   * @return data_model_view_t<i_t, f_t> Non-owning view with is_device_memory_=false
   */
  cuopt::mps_parser::data_model_view_t<i_t, f_t> create_view() const;
};

/**
 * @brief Copy GPU view data to CPU memory.
 * 
 * This function is used when we have a GPU-resident view but need CPU data
 * (e.g., for remote solve). All data is copied from GPU to CPU synchronously.
 * 
 * @tparam i_t Integer type (typically int)
 * @tparam f_t Floating-point type (float or double)
 * @param handle_ptr RAFT handle for GPU operations. Must not be null.
 * @param gpu_view View pointing to GPU memory
 * @return cpu_problem_data_t<i_t, f_t> CPU copy of all problem data
 * 
 * @note This function synchronizes the CUDA stream to ensure all copies complete.
 */
template <typename i_t, typename f_t>
cpu_problem_data_t<i_t, f_t> copy_view_to_cpu(
  raft::handle_t const* handle_ptr,
  const cuopt::mps_parser::data_model_view_t<i_t, f_t>& gpu_view);

}  // namespace linear_programming
}  // namespace cuopt
