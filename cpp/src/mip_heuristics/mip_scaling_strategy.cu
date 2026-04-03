/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#include <mip_heuristics/mip_constants.hpp>
#include <mip_heuristics/mip_scaling_strategy.cuh>
#include <pdlp/utils.cuh>
#include <utilities/logger.hpp>

#include <raft/util/cudart_utils.hpp>

#include <cub/cub.cuh>

#include <thrust/binary_search.h>
#include <thrust/count.h>
#include <thrust/fill.h>
#include <thrust/functional.h>
#include <thrust/inner_product.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/permutation_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/sort.h>
#include <thrust/transform.h>
#include <thrust/transform_reduce.h>
#include <thrust/tuple.h>

#include <rmm/device_uvector.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>

namespace cuopt::linear_programming::detail {

constexpr int row_scaling_max_iterations             = 8;
constexpr double row_scaling_min_initial_log2_spread = 12.0;
constexpr int row_scaling_factor_exponent            = 5;
constexpr int row_scaling_big_m_soft_factor_exponent = 4;
constexpr double row_scaling_min_factor =
  1.0 / static_cast<double>(std::uint64_t{1} << row_scaling_factor_exponent);
constexpr double row_scaling_max_factor =
  static_cast<double>(std::uint64_t{1} << row_scaling_factor_exponent);
constexpr double row_scaling_big_m_soft_min_factor =
  1.0 / static_cast<double>(std::uint64_t{1} << row_scaling_big_m_soft_factor_exponent);
constexpr double row_scaling_big_m_soft_max_factor       = 1.0;
constexpr double row_scaling_spread_rel_tol              = 1.0e-2;
constexpr double integer_coefficient_rel_tol             = 1.0e-6;
constexpr double integer_multiplier_rounding_tolerance   = 1.0e-6;
constexpr double min_abs_objective_coefficient_threshold = 1.0e-2;
constexpr double max_obj_scaling_coefficient             = 1.0e3;

constexpr int cumulative_row_scaling_exponent = 8;
constexpr double cumulative_row_scaling_min =
  1.0 / static_cast<double>(std::uint64_t{1} << cumulative_row_scaling_exponent);
constexpr double cumulative_row_scaling_max =
  static_cast<double>(std::uint64_t{1} << cumulative_row_scaling_exponent);

constexpr double post_scaling_max_ratio_warn = 1.0e15;

constexpr double big_m_abs_threshold   = 1.0e4;
constexpr double big_m_ratio_threshold = 1.0e4;

template <typename f_t>
struct abs_value_transform_t {
  __device__ f_t operator()(f_t value) const { return raft::abs(value); }
};

template <typename f_t>
struct nonzero_abs_or_inf_transform_t {
  __device__ f_t operator()(f_t value) const
  {
    const f_t abs_value = raft::abs(value);
    return abs_value > f_t(0) ? abs_value : std::numeric_limits<f_t>::infinity();
  }
};

template <typename i_t, typename f_t>
struct nonzero_count_transform_t {
  __device__ i_t operator()(f_t value) const { return raft::abs(value) > f_t(0) ? i_t(1) : i_t(0); }
};

template <typename item_t>
struct max_op_t {
  __host__ __device__ item_t operator()(const item_t& lhs, const item_t& rhs) const
  {
    return lhs > rhs ? lhs : rhs;
  }
};

template <typename item_t>
struct min_op_t {
  __host__ __device__ item_t operator()(const item_t& lhs, const item_t& rhs) const
  {
    return lhs < rhs ? lhs : rhs;
  }
};

struct gcd_op_t {
  __host__ __device__ std::int64_t operator()(std::int64_t lhs, std::int64_t rhs) const
  {
    lhs = lhs < 0 ? -lhs : lhs;
    rhs = rhs < 0 ? -rhs : rhs;
    if (lhs == 0) { return rhs; }
    if (rhs == 0) { return lhs; }
    while (rhs != 0) {
      const std::int64_t remainder = lhs % rhs;
      lhs                          = rhs;
      rhs                          = remainder;
    }
    return lhs;
  }
};

template <typename f_t>
struct integer_coeff_for_integer_var_transform_t {
  __device__ std::int64_t operator()(thrust::tuple<f_t, var_t> coeff_with_type) const
  {
    const f_t coefficient = thrust::get<0>(coeff_with_type);
    const var_t var_type  = thrust::get<1>(coeff_with_type);
    if (var_type != var_t::INTEGER) { return std::int64_t{0}; }

    const f_t abs_coefficient = raft::abs(coefficient);
    if (!isfinite(abs_coefficient) || abs_coefficient <= f_t(0)) { return std::int64_t{0}; }

    const f_t rounded_abs_coefficient = round(abs_coefficient);
    const f_t tolerance_scale         = abs_coefficient > f_t(1) ? abs_coefficient : f_t(1);
    const f_t integrality_tolerance =
      static_cast<f_t>(integer_coefficient_rel_tol) * tolerance_scale;
    if (raft::abs(abs_coefficient - rounded_abs_coefficient) > integrality_tolerance) {
      return std::int64_t{0};
    }
    if (rounded_abs_coefficient <= f_t(0) ||
        rounded_abs_coefficient > static_cast<f_t>(std::numeric_limits<std::int64_t>::max())) {
      return std::int64_t{0};
    }
    return static_cast<std::int64_t>(rounded_abs_coefficient);
  }
};

template <typename i_t, typename f_t>
void compute_row_inf_norm(
  const cuopt::linear_programming::optimization_problem_t<i_t, f_t>& op_problem,
  rmm::device_uvector<std::uint8_t>& temp_storage,
  size_t temp_storage_bytes,
  rmm::device_uvector<f_t>& row_inf_norm,
  rmm::cuda_stream_view stream_view)
{
  const auto& matrix_values  = op_problem.get_constraint_matrix_values();
  const auto& matrix_offsets = op_problem.get_constraint_matrix_offsets();
  auto coeff_abs_iter =
    thrust::make_transform_iterator(matrix_values.data(), abs_value_transform_t<f_t>{});
  size_t current_bytes = temp_storage_bytes;
  RAFT_CUDA_TRY(cub::DeviceSegmentedReduce::Reduce(temp_storage.data(),
                                                   current_bytes,
                                                   coeff_abs_iter,
                                                   row_inf_norm.data(),
                                                   op_problem.get_n_constraints(),
                                                   matrix_offsets.data(),
                                                   matrix_offsets.data() + 1,
                                                   max_op_t<f_t>{},
                                                   f_t(0),
                                                   stream_view));
}

template <typename i_t, typename f_t>
void compute_row_integer_gcd(
  const cuopt::linear_programming::optimization_problem_t<i_t, f_t>& op_problem,
  rmm::device_uvector<std::uint8_t>& temp_storage,
  size_t temp_storage_bytes,
  rmm::device_uvector<std::int64_t>& row_integer_gcd,
  rmm::cuda_stream_view stream_view)
{
  const auto& matrix_values  = op_problem.get_constraint_matrix_values();
  const auto& matrix_indices = op_problem.get_constraint_matrix_indices();
  const auto& matrix_offsets = op_problem.get_constraint_matrix_offsets();
  const auto& variable_types = op_problem.get_variable_types();
  if (variable_types.size() != static_cast<size_t>(op_problem.get_n_variables())) {
    thrust::fill(op_problem.get_handle_ptr()->get_thrust_policy(),
                 row_integer_gcd.begin(),
                 row_integer_gcd.end(),
                 std::int64_t{0});
    return;
  }
  auto variable_type_per_nnz =
    thrust::make_permutation_iterator(variable_types.data(), matrix_indices.data());
  auto coeff_and_type_iter =
    thrust::make_zip_iterator(thrust::make_tuple(matrix_values.data(), variable_type_per_nnz));
  auto integer_coeff_iter = thrust::make_transform_iterator(
    coeff_and_type_iter, integer_coeff_for_integer_var_transform_t<f_t>{});
  size_t current_bytes = temp_storage_bytes;
  RAFT_CUDA_TRY(cub::DeviceSegmentedReduce::Reduce(temp_storage.data(),
                                                   current_bytes,
                                                   integer_coeff_iter,
                                                   row_integer_gcd.data(),
                                                   op_problem.get_n_constraints(),
                                                   matrix_offsets.data(),
                                                   matrix_offsets.data() + 1,
                                                   gcd_op_t{},
                                                   std::int64_t{0},
                                                   stream_view));
}

template <typename i_t, typename f_t>
void compute_big_m_skip_rows(
  const cuopt::linear_programming::optimization_problem_t<i_t, f_t>& op_problem,
  rmm::device_uvector<std::uint8_t>& temp_storage,
  size_t temp_storage_bytes,
  rmm::device_uvector<f_t>& row_inf_norm,
  rmm::device_uvector<f_t>& row_min_nonzero,
  rmm::device_uvector<i_t>& row_nonzero_count,
  rmm::device_uvector<i_t>& row_skip_scaling)
{
  const auto& matrix_values  = op_problem.get_constraint_matrix_values();
  const auto& matrix_offsets = op_problem.get_constraint_matrix_offsets();
  const auto stream_view     = op_problem.get_handle_ptr()->get_stream();
  auto coeff_abs_iter =
    thrust::make_transform_iterator(matrix_values.data(), abs_value_transform_t<f_t>{});
  auto coeff_nonzero_min_iter =
    thrust::make_transform_iterator(matrix_values.data(), nonzero_abs_or_inf_transform_t<f_t>{});
  auto coeff_nonzero_count_iter =
    thrust::make_transform_iterator(matrix_values.data(), nonzero_count_transform_t<i_t, f_t>{});

  size_t max_bytes = temp_storage_bytes;
  RAFT_CUDA_TRY(cub::DeviceSegmentedReduce::Reduce(temp_storage.data(),
                                                   max_bytes,
                                                   coeff_abs_iter,
                                                   row_inf_norm.data(),
                                                   op_problem.get_n_constraints(),
                                                   matrix_offsets.data(),
                                                   matrix_offsets.data() + 1,
                                                   max_op_t<f_t>{},
                                                   f_t(0),
                                                   stream_view));
  size_t min_bytes = temp_storage_bytes;
  RAFT_CUDA_TRY(cub::DeviceSegmentedReduce::Reduce(temp_storage.data(),
                                                   min_bytes,
                                                   coeff_nonzero_min_iter,
                                                   row_min_nonzero.data(),
                                                   op_problem.get_n_constraints(),
                                                   matrix_offsets.data(),
                                                   matrix_offsets.data() + 1,
                                                   min_op_t<f_t>{},
                                                   std::numeric_limits<f_t>::infinity(),
                                                   stream_view));
  size_t count_bytes = temp_storage_bytes;
  RAFT_CUDA_TRY(cub::DeviceSegmentedReduce::Reduce(temp_storage.data(),
                                                   count_bytes,
                                                   coeff_nonzero_count_iter,
                                                   row_nonzero_count.data(),
                                                   op_problem.get_n_constraints(),
                                                   matrix_offsets.data(),
                                                   matrix_offsets.data() + 1,
                                                   thrust::plus<i_t>{},
                                                   i_t(0),
                                                   stream_view));

  auto row_begin = thrust::make_zip_iterator(
    thrust::make_tuple(row_inf_norm.begin(), row_min_nonzero.begin(), row_nonzero_count.begin()));
  auto row_end = thrust::make_zip_iterator(
    thrust::make_tuple(row_inf_norm.end(), row_min_nonzero.end(), row_nonzero_count.end()));
  thrust::transform(
    op_problem.get_handle_ptr()->get_thrust_policy(),
    row_begin,
    row_end,
    row_skip_scaling.begin(),
    [] __device__(auto row_info) -> i_t {
      const f_t row_norm          = thrust::get<0>(row_info);
      const f_t row_min_non_zero  = thrust::get<1>(row_info);
      const i_t row_non_zero_size = thrust::get<2>(row_info);
      if (row_non_zero_size < i_t(2) || row_min_non_zero >= std::numeric_limits<f_t>::infinity()) {
        return i_t(0);
      }

      const f_t row_ratio = row_norm / row_min_non_zero;
      return row_norm >= static_cast<f_t>(big_m_abs_threshold) &&
                 row_ratio >= static_cast<f_t>(big_m_ratio_threshold)
               ? i_t(1)
               : i_t(0);
    });
}

template <typename i_t, typename f_t>
void scale_objective(cuopt::linear_programming::optimization_problem_t<i_t, f_t>& op_problem)
{
  auto& obj_coefficients = op_problem.get_objective_coefficients();
  const i_t n_cols       = op_problem.get_n_variables();
  if (n_cols == 0) { return; }

  const auto* handle_ptr = op_problem.get_handle_ptr();

  f_t min_abs_obj = thrust::transform_reduce(handle_ptr->get_thrust_policy(),
                                             obj_coefficients.begin(),
                                             obj_coefficients.end(),
                                             nonzero_abs_or_inf_transform_t<f_t>{},
                                             std::numeric_limits<f_t>::infinity(),
                                             min_op_t<f_t>{});

  f_t max_abs_obj = thrust::transform_reduce(handle_ptr->get_thrust_policy(),
                                             obj_coefficients.begin(),
                                             obj_coefficients.end(),
                                             abs_value_transform_t<f_t>{},
                                             f_t(0),
                                             max_op_t<f_t>{});

  if (!std::isfinite(static_cast<double>(min_abs_obj)) || min_abs_obj <= f_t(0) ||
      max_abs_obj <= f_t(0)) {
    CUOPT_LOG_INFO("MIP_OBJ_SCALING skipped: no finite nonzero objective coefficients");
    return;
  }

  if (static_cast<double>(min_abs_obj) >= min_abs_objective_coefficient_threshold) {
    CUOPT_LOG_INFO("MIP_OBJ_SCALING skipped: min_abs_coeff=%g already above threshold=%g",
                   static_cast<double>(min_abs_obj),
                   min_abs_objective_coefficient_threshold);
    return;
  }

  double raw_scale = min_abs_objective_coefficient_threshold / static_cast<double>(min_abs_obj);
  double scale     = std::min(raw_scale, max_obj_scaling_coefficient);

  double post_max = static_cast<double>(max_abs_obj) * scale;
  if (post_max > 1.0e6) {
    CUOPT_LOG_INFO("MIP_OBJ_SCALING skipped: would push max_coeff from %g to %g (limit 1e6)",
                   static_cast<double>(max_abs_obj),
                   post_max);
    return;
  }

  f_t scale_f = static_cast<f_t>(scale);
  thrust::transform(handle_ptr->get_thrust_policy(),
                    obj_coefficients.begin(),
                    obj_coefficients.end(),
                    obj_coefficients.begin(),
                    [scale_f] __device__(f_t c) -> f_t { return c * scale_f; });

  f_t old_sf  = op_problem.get_objective_scaling_factor();
  f_t old_off = op_problem.get_objective_offset();
  op_problem.set_objective_scaling_factor(old_sf / scale_f);
  op_problem.set_objective_offset(old_off * scale_f);

  CUOPT_LOG_INFO(
    "MIP_OBJ_SCALING applied: min_abs_coeff=%g max_abs_coeff=%g scale=%g new_scaling_factor=%g",
    static_cast<double>(min_abs_obj),
    static_cast<double>(max_abs_obj),
    scale,
    static_cast<double>(old_sf / scale_f));
}

template <typename i_t, typename f_t>
rmm::device_uvector<std::int64_t> capture_pre_scaling_integer_gcd(
  const cuopt::linear_programming::optimization_problem_t<i_t, f_t>& op_problem,
  rmm::device_uvector<std::uint8_t>& temp_storage,
  size_t temp_storage_bytes,
  rmm::cuda_stream_view stream_view)
{
  const i_t n_rows = op_problem.get_n_constraints();
  rmm::device_uvector<std::int64_t> gcd(static_cast<size_t>(n_rows), stream_view);
  compute_row_integer_gcd(op_problem, temp_storage, temp_storage_bytes, gcd, stream_view);
  return gcd;
}

template <typename i_t, typename f_t>
void assert_integer_coefficient_integrality(
  const cuopt::linear_programming::optimization_problem_t<i_t, f_t>& op_problem,
  rmm::device_uvector<std::uint8_t>& temp_storage,
  size_t temp_storage_bytes,
  const rmm::device_uvector<std::int64_t>& pre_scaling_gcd,
  rmm::cuda_stream_view stream_view)
{
  const auto* handle_ptr = op_problem.get_handle_ptr();
  const i_t n_rows       = op_problem.get_n_constraints();
  rmm::device_uvector<std::int64_t> post_scaling_gcd(static_cast<size_t>(n_rows), stream_view);
  compute_row_integer_gcd(
    op_problem, temp_storage, temp_storage_bytes, post_scaling_gcd, stream_view);

  i_t broken_rows = thrust::inner_product(
    handle_ptr->get_thrust_policy(),
    pre_scaling_gcd.begin(),
    pre_scaling_gcd.end(),
    post_scaling_gcd.begin(),
    i_t(0),
    thrust::plus<i_t>{},
    [] __device__(std::int64_t pre_gcd, std::int64_t post_gcd) -> i_t {
      return (pre_gcd > std::int64_t{0} && post_gcd == std::int64_t{0}) ? i_t(1) : i_t(0);
    });

  if (broken_rows > 0) {
    CUOPT_LOG_WARN("MIP row scaling: %d rows lost integer coefficient integrality after scaling",
                   broken_rows);
  }
  cuopt_assert(broken_rows == 0,
               "MIP scaling must preserve integer coefficients for integer variables");
}

template <typename i_t, typename f_t>
mip_scaling_strategy_t<i_t, f_t>::mip_scaling_strategy_t(
  typename mip_scaling_strategy_t<i_t, f_t>::optimization_problem_type_t& op_problem_scaled)
  : handle_ptr_(op_problem_scaled.get_handle_ptr()),
    stream_view_(handle_ptr_->get_stream()),
    op_problem_scaled_(op_problem_scaled)
{
}

template <typename i_t, typename f_t>
size_t dry_run_cub(const cuopt::linear_programming::optimization_problem_t<i_t, f_t>& op_problem,
                   i_t n_rows,
                   rmm::device_uvector<f_t>& row_inf_norm,
                   rmm::device_uvector<f_t>& row_min_nonzero,
                   rmm::device_uvector<i_t>& row_nonzero_count,
                   rmm::device_uvector<std::int64_t>& row_integer_gcd,
                   rmm::cuda_stream_view stream_view)
{
  const auto& matrix_values     = op_problem.get_constraint_matrix_values();
  const auto& matrix_indices    = op_problem.get_constraint_matrix_indices();
  const auto& matrix_offsets    = op_problem.get_constraint_matrix_offsets();
  const auto& variable_types    = op_problem.get_variable_types();
  size_t temp_storage_bytes     = 0;
  size_t current_required_bytes = 0;

  auto coeff_abs_iter =
    thrust::make_transform_iterator(matrix_values.data(), abs_value_transform_t<f_t>{});
  RAFT_CUDA_TRY(cub::DeviceSegmentedReduce::Reduce(nullptr,
                                                   current_required_bytes,
                                                   coeff_abs_iter,
                                                   row_inf_norm.data(),
                                                   n_rows,
                                                   matrix_offsets.data(),
                                                   matrix_offsets.data() + 1,
                                                   max_op_t<f_t>{},
                                                   f_t(0),
                                                   stream_view));
  temp_storage_bytes = std::max(temp_storage_bytes, current_required_bytes);

  auto coeff_nonzero_min_iter =
    thrust::make_transform_iterator(matrix_values.data(), nonzero_abs_or_inf_transform_t<f_t>{});
  RAFT_CUDA_TRY(cub::DeviceSegmentedReduce::Reduce(nullptr,
                                                   current_required_bytes,
                                                   coeff_nonzero_min_iter,
                                                   row_min_nonzero.data(),
                                                   n_rows,
                                                   matrix_offsets.data(),
                                                   matrix_offsets.data() + 1,
                                                   min_op_t<f_t>{},
                                                   std::numeric_limits<f_t>::infinity(),
                                                   stream_view));
  temp_storage_bytes = std::max(temp_storage_bytes, current_required_bytes);

  auto coeff_nonzero_count_iter =
    thrust::make_transform_iterator(matrix_values.data(), nonzero_count_transform_t<i_t, f_t>{});
  RAFT_CUDA_TRY(cub::DeviceSegmentedReduce::Reduce(nullptr,
                                                   current_required_bytes,
                                                   coeff_nonzero_count_iter,
                                                   row_nonzero_count.data(),
                                                   n_rows,
                                                   matrix_offsets.data(),
                                                   matrix_offsets.data() + 1,
                                                   thrust::plus<i_t>{},
                                                   i_t(0),
                                                   stream_view));
  temp_storage_bytes = std::max(temp_storage_bytes, current_required_bytes);

  if (variable_types.size() == static_cast<size_t>(op_problem.get_n_variables())) {
    auto variable_type_per_nnz =
      thrust::make_permutation_iterator(variable_types.data(), matrix_indices.data());
    auto coeff_and_type_iter =
      thrust::make_zip_iterator(thrust::make_tuple(matrix_values.data(), variable_type_per_nnz));
    auto integer_coeff_iter = thrust::make_transform_iterator(
      coeff_and_type_iter, integer_coeff_for_integer_var_transform_t<f_t>{});
    RAFT_CUDA_TRY(cub::DeviceSegmentedReduce::Reduce(nullptr,
                                                     current_required_bytes,
                                                     integer_coeff_iter,
                                                     row_integer_gcd.data(),
                                                     n_rows,
                                                     matrix_offsets.data(),
                                                     matrix_offsets.data() + 1,
                                                     gcd_op_t{},
                                                     std::int64_t{0},
                                                     stream_view));
    temp_storage_bytes = std::max(temp_storage_bytes, current_required_bytes);
  }

  return temp_storage_bytes;
}

template <typename i_t, typename f_t>
void mip_scaling_strategy_t<i_t, f_t>::scale_problem(bool do_objective_scaling)
{
  raft::common::nvtx::range fun_scope("mip_scale_problem");

  auto& matrix_values           = op_problem_scaled_.get_constraint_matrix_values();
  auto& matrix_offsets          = op_problem_scaled_.get_constraint_matrix_offsets();
  auto& constraint_bounds       = op_problem_scaled_.get_constraint_bounds();
  auto& constraint_lower_bounds = op_problem_scaled_.get_constraint_lower_bounds();
  auto& constraint_upper_bounds = op_problem_scaled_.get_constraint_upper_bounds();
  const i_t n_rows              = op_problem_scaled_.get_n_constraints();
  const i_t n_cols              = op_problem_scaled_.get_n_variables();
  const i_t nnz                 = op_problem_scaled_.get_nnz();

  if (do_objective_scaling) {
    scale_objective(op_problem_scaled_);
  } else {
    CUOPT_LOG_INFO("MIP_OBJ_SCALING skipped: disabled by user setting");
  }

  if (n_rows == 0 || nnz <= 0) { return; }
  cuopt_assert(constraint_bounds.size() == size_t{0} ||
                 constraint_bounds.size() == static_cast<size_t>(n_rows),
               "constraint_bounds must be empty or have one value per constraint");

  rmm::device_uvector<f_t> row_inf_norm(static_cast<size_t>(n_rows), stream_view_);
  rmm::device_uvector<f_t> row_min_nonzero(static_cast<size_t>(n_rows), stream_view_);
  rmm::device_uvector<i_t> row_nonzero_count(static_cast<size_t>(n_rows), stream_view_);
  rmm::device_uvector<std::int64_t> row_integer_gcd(static_cast<size_t>(n_rows), stream_view_);
  rmm::device_uvector<f_t> row_rhs_magnitude(static_cast<size_t>(n_rows), stream_view_);
  rmm::device_uvector<i_t> row_skip_scaling(static_cast<size_t>(n_rows), stream_view_);
  thrust::fill(
    handle_ptr_->get_thrust_policy(), row_skip_scaling.begin(), row_skip_scaling.end(), i_t(0));
  rmm::device_uvector<f_t> iteration_scaling(static_cast<size_t>(n_rows), stream_view_);
  rmm::device_uvector<f_t> cumulative_scaling(static_cast<size_t>(n_rows), stream_view_);
  thrust::fill(
    handle_ptr_->get_thrust_policy(), cumulative_scaling.begin(), cumulative_scaling.end(), f_t(1));
  rmm::device_uvector<i_t> coefficient_row_index(static_cast<size_t>(nnz), stream_view_);
  rmm::device_uvector<double> ref_log2_values(static_cast<size_t>(n_rows), stream_view_);

  thrust::upper_bound(handle_ptr_->get_thrust_policy(),
                      matrix_offsets.begin(),
                      matrix_offsets.end(),
                      thrust::make_counting_iterator<i_t>(0),
                      thrust::make_counting_iterator<i_t>(nnz),
                      coefficient_row_index.begin());
  thrust::transform(
    handle_ptr_->get_thrust_policy(),
    coefficient_row_index.begin(),
    coefficient_row_index.end(),
    coefficient_row_index.begin(),
    [] __device__(i_t row_upper_bound_idx) -> i_t { return row_upper_bound_idx - 1; });

  size_t temp_storage_bytes = dry_run_cub(op_problem_scaled_,
                                          n_rows,
                                          row_inf_norm,
                                          row_min_nonzero,
                                          row_nonzero_count,
                                          row_integer_gcd,
                                          stream_view_);

  rmm::device_uvector<std::uint8_t> temp_storage(temp_storage_bytes, stream_view_);

  cuopt_func_call(auto pre_scaling_gcd = capture_pre_scaling_integer_gcd(
                    op_problem_scaled_, temp_storage, temp_storage_bytes, stream_view_));

  compute_big_m_skip_rows(op_problem_scaled_,
                          temp_storage,
                          temp_storage_bytes,
                          row_inf_norm,
                          row_min_nonzero,
                          row_nonzero_count,
                          row_skip_scaling);

  i_t big_m_rows = thrust::count(
    handle_ptr_->get_thrust_policy(), row_skip_scaling.begin(), row_skip_scaling.end(), i_t(1));

  CUOPT_LOG_INFO("MIP row scaling start: rows=%d cols=%d max_iterations=%d soft_big_m_rows=%d",
                 n_rows,
                 n_cols,
                 row_scaling_max_iterations,
                 big_m_rows);

  f_t original_max_coeff = thrust::transform_reduce(handle_ptr_->get_thrust_policy(),
                                                    matrix_values.begin(),
                                                    matrix_values.end(),
                                                    abs_value_transform_t<f_t>{},
                                                    f_t(0),
                                                    max_op_t<f_t>{});

  double previous_row_log2_spread = std::numeric_limits<double>::infinity();
  for (int iteration = 0; iteration < row_scaling_max_iterations; ++iteration) {
    compute_row_inf_norm(
      op_problem_scaled_, temp_storage, temp_storage_bytes, row_inf_norm, stream_view_);
    compute_row_integer_gcd(
      op_problem_scaled_, temp_storage, temp_storage_bytes, row_integer_gcd, stream_view_);

    using row_stats_t        = thrust::tuple<double, double, double, double>;
    auto row_norm_log2_stats = thrust::transform_reduce(
      handle_ptr_->get_thrust_policy(),
      row_inf_norm.begin(),
      row_inf_norm.end(),
      [] __device__(f_t row_norm) -> row_stats_t {
        if (row_norm == f_t(0)) {
          return {0.0,
                  0.0,
                  std::numeric_limits<double>::infinity(),
                  -std::numeric_limits<double>::infinity()};
        }
        const double row_log2 = log2(static_cast<double>(row_norm));
        return {row_log2, 1.0, row_log2, row_log2};
      },
      row_stats_t{0.0,
                  0.0,
                  std::numeric_limits<double>::infinity(),
                  -std::numeric_limits<double>::infinity()},
      [] __device__(row_stats_t a, row_stats_t b) -> row_stats_t {
        return {thrust::get<0>(a) + thrust::get<0>(b),
                thrust::get<1>(a) + thrust::get<1>(b),
                min_op_t<double>{}(thrust::get<2>(a), thrust::get<2>(b)),
                max_op_t<double>{}(thrust::get<3>(a), thrust::get<3>(b))};
      });
    const i_t active_row_count = static_cast<i_t>(thrust::get<1>(row_norm_log2_stats));
    if (active_row_count == 0) { break; }
    const double row_log2_spread =
      thrust::get<3>(row_norm_log2_stats) - thrust::get<2>(row_norm_log2_stats);
    if (iteration == 0 && row_log2_spread <= row_scaling_min_initial_log2_spread) {
      CUOPT_LOG_INFO("MIP row scaling skipped: initial_log2_spread=%g threshold=%g",
                     row_log2_spread,
                     row_scaling_min_initial_log2_spread);
      break;
    }
    if (std::isfinite(previous_row_log2_spread)) {
      const double spread_improvement = previous_row_log2_spread - row_log2_spread;
      if (spread_improvement <=
          row_scaling_spread_rel_tol * std::max(1.0, previous_row_log2_spread)) {
        break;
      }
    }
    previous_row_log2_spread = row_log2_spread;

    thrust::transform(handle_ptr_->get_thrust_policy(),
                      thrust::make_zip_iterator(thrust::make_tuple(
                        constraint_lower_bounds.begin(), constraint_upper_bounds.begin())),
                      thrust::make_zip_iterator(thrust::make_tuple(constraint_lower_bounds.end(),
                                                                   constraint_upper_bounds.end())),
                      row_rhs_magnitude.begin(),
                      [] __device__(auto row_bounds) -> f_t {
                        const f_t lower_bound = thrust::get<0>(row_bounds);
                        const f_t upper_bound = thrust::get<1>(row_bounds);
                        f_t rhs_norm          = f_t(0);
                        if (isfinite(lower_bound)) { rhs_norm = raft::abs(lower_bound); }
                        if (isfinite(upper_bound)) {
                          const f_t upper_abs = raft::abs(upper_bound);
                          rhs_norm            = upper_abs > rhs_norm ? upper_abs : rhs_norm;
                        }
                        return rhs_norm;
                      });

    constexpr double neg_inf_sentinel = -1.0e300;
    thrust::transform(handle_ptr_->get_thrust_policy(),
                      thrust::make_zip_iterator(thrust::make_tuple(
                        row_inf_norm.begin(), row_rhs_magnitude.begin(), row_skip_scaling.begin())),
                      thrust::make_zip_iterator(thrust::make_tuple(
                        row_inf_norm.end(), row_rhs_magnitude.end(), row_skip_scaling.end())),
                      ref_log2_values.begin(),
                      [] __device__(auto row_info) -> double {
                        const f_t row_norm = thrust::get<0>(row_info);
                        const f_t rhs_norm = thrust::get<1>(row_info);
                        const i_t is_big_m = thrust::get<2>(row_info);
                        if (is_big_m) { return -std::numeric_limits<double>::infinity(); }
                        if (rhs_norm == f_t(0)) { return -std::numeric_limits<double>::infinity(); }
                        if (row_norm <= f_t(0)) { return -std::numeric_limits<double>::infinity(); }
                        return log2(static_cast<double>(row_norm));
                      });
    thrust::sort(handle_ptr_->get_thrust_policy(), ref_log2_values.begin(), ref_log2_values.end());
    auto valid_begin_iter = thrust::lower_bound(handle_ptr_->get_thrust_policy(),
                                                ref_log2_values.begin(),
                                                ref_log2_values.end(),
                                                neg_inf_sentinel);
    i_t n_invalid         = static_cast<i_t>(valid_begin_iter - ref_log2_values.begin());
    i_t valid_count       = n_rows - n_invalid;
    if (valid_count == 0) { break; }
    i_t median_idx = n_invalid + valid_count / 2;
    double h_median_log2;
    RAFT_CUDA_TRY(cudaMemcpyAsync(&h_median_log2,
                                  ref_log2_values.data() + median_idx,
                                  sizeof(double),
                                  cudaMemcpyDeviceToHost,
                                  stream_view_));
    handle_ptr_->sync_stream();
    f_t target_norm = static_cast<f_t>(exp2(h_median_log2));
    cuopt_assert(std::isfinite(static_cast<double>(target_norm)), "target_norm must be finite");
    cuopt_assert(target_norm > f_t(0), "target_norm must be positive");

    thrust::transform(
      handle_ptr_->get_thrust_policy(),
      thrust::make_zip_iterator(thrust::make_tuple(row_inf_norm.begin(),
                                                   row_skip_scaling.begin(),
                                                   row_integer_gcd.begin(),
                                                   cumulative_scaling.begin(),
                                                   row_rhs_magnitude.begin())),
      thrust::make_zip_iterator(thrust::make_tuple(row_inf_norm.end(),
                                                   row_skip_scaling.end(),
                                                   row_integer_gcd.end(),
                                                   cumulative_scaling.end(),
                                                   row_rhs_magnitude.end())),
      iteration_scaling.begin(),
      [target_norm] __device__(auto row_info) -> f_t {
        const f_t row_norm               = thrust::get<0>(row_info);
        const i_t is_big_m               = thrust::get<1>(row_info);
        const std::int64_t row_coeff_gcd = thrust::get<2>(row_info);
        const f_t cum_scale              = thrust::get<3>(row_info);
        const f_t rhs_norm               = thrust::get<4>(row_info);
        if (row_norm == f_t(0)) { return f_t(1); }
        if (rhs_norm == f_t(0)) { return f_t(1); }

        const f_t desired_scaling = target_norm / row_norm;
        if (!isfinite(desired_scaling) || desired_scaling <= f_t(0)) { return f_t(1); }

        f_t min_scaling = is_big_m ? static_cast<f_t>(row_scaling_big_m_soft_min_factor)
                                   : static_cast<f_t>(row_scaling_min_factor);
        f_t max_scaling = is_big_m ? static_cast<f_t>(row_scaling_big_m_soft_max_factor)
                                   : static_cast<f_t>(row_scaling_max_factor);

        if (!is_big_m && row_norm >= static_cast<f_t>(big_m_abs_threshold)) {
          if (max_scaling > f_t(1)) { max_scaling = f_t(1); }
        }

        const f_t cum_lower = static_cast<f_t>(cumulative_row_scaling_min) / cum_scale;
        const f_t cum_upper = static_cast<f_t>(cumulative_row_scaling_max) / cum_scale;
        if (cum_lower > min_scaling) { min_scaling = cum_lower; }
        if (cum_upper < max_scaling) { max_scaling = cum_upper; }
        if (min_scaling > max_scaling) { return f_t(1); }

        f_t row_scaling = desired_scaling;
        if (row_scaling < min_scaling) { row_scaling = min_scaling; }
        if (row_scaling > max_scaling) { row_scaling = max_scaling; }

        // Fix E: prefer power-of-two scaling for integer rows (exact in IEEE 754)
        if (row_coeff_gcd > std::int64_t{0}) {
          const f_t gcd_value = static_cast<f_t>(row_coeff_gcd);
          if (isfinite(gcd_value) && gcd_value > f_t(0)) {
            const double log2_scaling = log2(static_cast<double>(row_scaling));
            int k_candidates[3]       = {static_cast<int>(round(log2_scaling)),
                                         static_cast<int>(floor(log2_scaling)),
                                         static_cast<int>(ceil(log2_scaling))};
            bool found_pow2           = false;
            for (int ci = 0; ci < 3 && !found_pow2; ++ci) {
              int k    = k_candidates[ci];
              f_t pow2 = static_cast<f_t>(exp2(static_cast<double>(k)));
              if (pow2 < min_scaling || pow2 > max_scaling) { continue; }
              bool preserves =
                (k >= 0) || (-k < 63 && (row_coeff_gcd % (std::int64_t{1} << (-k))) == 0);
              if (preserves) {
                row_scaling = pow2;
                found_pow2  = true;
              }
            }
            if (!found_pow2) {
              std::int64_t min_mult = static_cast<std::int64_t>(
                ceil(static_cast<double>(min_scaling * gcd_value -
                                         static_cast<f_t>(integer_multiplier_rounding_tolerance))));
              std::int64_t max_mult = static_cast<std::int64_t>(floor(
                static_cast<double>(max_scaling * gcd_value +
                                    static_cast<f_t>(integer_multiplier_rounding_tolerance))));
              if (min_mult < std::int64_t{1}) { min_mult = std::int64_t{1}; }
              if (max_mult < min_mult) { max_mult = min_mult; }
              std::int64_t proj_mult = static_cast<std::int64_t>(round(row_scaling * gcd_value));
              if (proj_mult < min_mult) { proj_mult = min_mult; }
              if (proj_mult > max_mult) { proj_mult = max_mult; }
              row_scaling = static_cast<f_t>(proj_mult) / gcd_value;
            }
          }
        }
        return row_scaling;
      });

    i_t scaled_rows =
      thrust::count_if(handle_ptr_->get_thrust_policy(),
                       iteration_scaling.begin(),
                       iteration_scaling.end(),
                       [] __device__(f_t row_scale) -> bool { return row_scale != f_t(1); });
    CUOPT_LOG_INFO(
      "MIP_SCALING_METRICS iteration=%d log2_spread=%g target_norm=%g scaled_rows=%d "
      "valid_rows=%d",
      iteration,
      row_log2_spread,
      static_cast<double>(target_norm),
      scaled_rows,
      valid_count);
    if (scaled_rows == 0) { break; }

    f_t predicted_max = thrust::inner_product(handle_ptr_->get_thrust_policy(),
                                              row_inf_norm.begin(),
                                              row_inf_norm.end(),
                                              iteration_scaling.begin(),
                                              f_t(0),
                                              max_op_t<f_t>{},
                                              thrust::multiplies<f_t>{});
    if (predicted_max > original_max_coeff) {
      CUOPT_LOG_INFO("MIP_SCALING magnitude guard: predicted_max=%g > original_max=%g, stopping",
                     static_cast<double>(predicted_max),
                     static_cast<double>(original_max_coeff));
      break;
    }

    thrust::transform(
      handle_ptr_->get_thrust_policy(),
      matrix_values.begin(),
      matrix_values.end(),
      thrust::make_permutation_iterator(iteration_scaling.begin(), coefficient_row_index.begin()),
      matrix_values.begin(),
      thrust::multiplies<f_t>{});

    thrust::transform(handle_ptr_->get_thrust_policy(),
                      cumulative_scaling.begin(),
                      cumulative_scaling.end(),
                      iteration_scaling.begin(),
                      cumulative_scaling.begin(),
                      thrust::multiplies<f_t>{});

    thrust::transform(handle_ptr_->get_thrust_policy(),
                      constraint_lower_bounds.begin(),
                      constraint_lower_bounds.end(),
                      iteration_scaling.begin(),
                      constraint_lower_bounds.begin(),
                      thrust::multiplies<f_t>{});
    thrust::transform(handle_ptr_->get_thrust_policy(),
                      constraint_upper_bounds.begin(),
                      constraint_upper_bounds.end(),
                      iteration_scaling.begin(),
                      constraint_upper_bounds.begin(),
                      thrust::multiplies<f_t>{});
    if (constraint_bounds.size() == static_cast<size_t>(n_rows)) {
      thrust::transform(handle_ptr_->get_thrust_policy(),
                        constraint_bounds.begin(),
                        constraint_bounds.end(),
                        iteration_scaling.begin(),
                        constraint_bounds.begin(),
                        thrust::multiplies<f_t>{});
    }
  }

  CUOPT_LOG_INFO("MIP_SCALING_SUMMARY rows=%d bigm_rows=%d final_spread=%g",
                 n_rows,
                 big_m_rows,
                 previous_row_log2_spread);

  cuopt_func_call(assert_integer_coefficient_integrality(
    op_problem_scaled_, temp_storage, temp_storage_bytes, pre_scaling_gcd, stream_view_));

  const f_t post_max_coeff         = thrust::transform_reduce(handle_ptr_->get_thrust_policy(),
                                                      matrix_values.begin(),
                                                      matrix_values.end(),
                                                      abs_value_transform_t<f_t>{},
                                                      f_t(0),
                                                      max_op_t<f_t>{});
  const f_t post_min_nonzero_coeff = thrust::transform_reduce(handle_ptr_->get_thrust_policy(),
                                                              matrix_values.begin(),
                                                              matrix_values.end(),
                                                              nonzero_abs_or_inf_transform_t<f_t>{},
                                                              std::numeric_limits<f_t>::infinity(),
                                                              min_op_t<f_t>{});
  if (std::isfinite(static_cast<double>(post_max_coeff)) &&
      std::isfinite(static_cast<double>(post_min_nonzero_coeff)) &&
      post_min_nonzero_coeff > f_t(0)) {
    const double post_ratio =
      static_cast<double>(post_max_coeff) / static_cast<double>(post_min_nonzero_coeff);
    if (post_ratio > post_scaling_max_ratio_warn) {
      CUOPT_LOG_WARN(
        "MIP row scaling: extreme coefficient ratio after scaling: max=%g min_nz=%g ratio=%g",
        static_cast<double>(post_max_coeff),
        static_cast<double>(post_min_nonzero_coeff),
        post_ratio);
    }
  }

  CUOPT_LOG_INFO("MIP row scaling completed");
  op_problem_scaled_.print_scaling_information();
}

#define INSTANTIATE(F_TYPE) template class mip_scaling_strategy_t<int, F_TYPE>;

#if MIP_INSTANTIATE_FLOAT
INSTANTIATE(float)
#endif

#if MIP_INSTANTIATE_DOUBLE
INSTANTIATE(double)
#endif

}  // namespace cuopt::linear_programming::detail
