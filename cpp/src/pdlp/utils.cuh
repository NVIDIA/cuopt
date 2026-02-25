/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#pragma once

#include <optional>
#include <pdlp/pdlp_constants.hpp>
#include <pdlp/restart_strategy/pdlp_restart_strategy.cuh>
#include <utilities/macros.cuh>

#include <raft/core/device_span.hpp>
#include <raft/linalg/binary_op.cuh>
#include <raft/linalg/detail/cublas_wrappers.hpp>
#include <raft/linalg/norm.cuh>
#include <raft/util/cuda_utils.cuh>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_scalar.hpp>
#include <rmm/device_uvector.hpp>

#include <thrust/execution_policy.h>
#include <thrust/functional.h>
#include <thrust/transform_reduce.h>
#include <thrust/tuple.h>

namespace cuopt::linear_programming::detail {

template <typename f_t, int BLOCK_SIZE>
DI f_t deterministic_block_reduce(raft::device_span<f_t> shared, f_t val)
{
  cuopt_assert(shared.size() >= BLOCK_SIZE / raft::WarpSize,
               "Not enough shared to do a warp reduce");

  const int lane = threadIdx.x % raft::WarpSize;
  const int wid  = threadIdx.x / raft::WarpSize;

  val = raft::warpReduce(val);  // Each warp performs partial reduction

  if (lane == 0) shared[wid] = val;  // Write reduced value to shared memory

  __syncthreads();  // Wait for all partial reductions

  // read from shared memory only if that warp existed
  val = (threadIdx.x < BLOCK_SIZE / raft::WarpSize) ? shared[lane] : f_t(0);

  if (wid == 0) val = raft::warpReduce(val);  // Final reduce within first warp

  return val;
}

template <typename f_t>
struct max_abs_value {
  __device__ __forceinline__ f_t operator()(f_t a, f_t b)
  {
    return raft::abs(a) < raft::abs(b) ? raft::abs(b) : raft::abs(a);
  }
};

template <typename i_t>
i_t conditional_major(uint64_t total_pdlp_iterations)
{
  uint64_t step                       = 10;
  uint64_t threshold                  = 1000;
  [[maybe_unused]] uint64_t iteration = 0;

  [[maybe_unused]] constexpr uint64_t max_u64 = std::numeric_limits<uint64_t>::max();

  while (total_pdlp_iterations >= threshold) {
    ++iteration;

    cuopt_assert(step <= max_u64 / 10, "Overflow risk in step during conditional_major");
    cuopt_assert(threshold <= max_u64 / 10, "Overflow risk in threshold during conditional_major");

    step *= 10;
    threshold *= 10;
  }
  return step;
}

template <typename f_t>
struct a_times_scalar {
  a_times_scalar(const f_t scalar) : scalar_{scalar} {}
  HDI f_t operator()(f_t a) { return a * scalar_; }

  const f_t scalar_;
};

template <typename f_t>
struct batch_safe_div {
  HDI f_t operator()(f_t a, f_t b)
  {
    cuopt_assert(b != f_t(0), "Division by zero");
    return b != f_t(0) ? a / b : a;
  }
};

template <typename f_t>
struct a_sub_scalar_times_b {
  a_sub_scalar_times_b(const f_t* scalar) : scalar_{scalar} {}
  __device__ __forceinline__ f_t operator()(f_t a, f_t b) { return a - *scalar_ * b; }

  const f_t* scalar_;
};

template <typename f_t, typename f_t2>
struct primal_projection {
  primal_projection(const f_t* step_size) : step_size_(step_size) {}

  __device__ __forceinline__ thrust::tuple<f_t, f_t, f_t> operator()(f_t primal,
                                                                     f_t obj_coeff,
                                                                     f_t AtY,
                                                                     f_t2 bounds)
  {
    f_t lower    = get_lower(bounds);
    f_t upper    = get_upper(bounds);
    f_t gradient = obj_coeff - AtY;
    f_t next     = primal - (*step_size_ * gradient);
    next         = raft::max<f_t>(raft::min<f_t>(next, upper), lower);
    return thrust::make_tuple(next, next - primal, next - primal + next);
  }

  const f_t* step_size_;
  const f_t* scalar_;
};

template <typename f_t>
struct dual_projection {
  dual_projection(const f_t* scalar) : scalar_{scalar} {}
  __device__ __forceinline__ thrust::tuple<f_t, f_t> operator()(f_t dual,
                                                                f_t gradient,
                                                                f_t lower,
                                                                f_t upper)
  {
    f_t next = dual - (*scalar_ * gradient);
    f_t low  = next + *scalar_ * lower;
    f_t up   = next + *scalar_ * upper;
    next     = raft::max<f_t>(low, raft::min<f_t>(up, f_t(0)));
    return thrust::make_tuple(next, next - dual);
  }
  const f_t* scalar_;
};

template <typename f_t>
struct a_add_scalar_times_b {
  a_add_scalar_times_b(const f_t* scalar) : scalar_{scalar} {}
  __device__ __forceinline__ f_t operator()(f_t a, f_t b) { return a + *scalar_ * b; }

  const f_t* scalar_;
};

template <typename f_t>
struct a_divides_sqrt_b_bounded {
  // if b is larger than zero return a / sqrt(b) and otherwise return a
  __device__ __forceinline__ f_t operator()(f_t a, f_t b)
  {
    return b > f_t(0) ? a / raft::sqrt(b) : a;
  }
};

template <typename f_t>
struct constraint_clamp {
  __device__ f_t operator()(f_t value, f_t lower, f_t upper)
  {
    return raft::min<f_t>(raft::max<f_t>(value, lower), upper);
  }
};

template <typename f_t, typename f_t2>
struct clamp {
  __device__ f_t operator()(f_t value, f_t2 bounds)
  {
    return raft::min<f_t>(raft::max<f_t>(value, get_lower(bounds)), get_upper(bounds));
  }
};

template <typename f_t>
struct combine_finite_abs_bounds {
  __device__ __host__ f_t operator()(f_t lower, f_t upper)
  {
    f_t val = f_t(0);
    if (isfinite(upper)) { val = raft::max<f_t>(val, raft::abs(upper)); }
    if (isfinite(lower)) { val = raft::max<f_t>(val, raft::abs(lower)); }
    return val;
  }
};

// Used to wrap the problem input around a problem inside the batch
// This is used to iterate over (for example) the objective coefficient when they are the same for
// all climbers Every variable with the same index across problems in the batch should have the same
// bounds This also work if the bound are actually batch wide since the size will be bigger
template <typename f_t>
struct problem_wrapped_iterator {
  problem_wrapped_iterator(const f_t* problem_input, int problem_size)
    : problem_input_(problem_input), problem_size_(problem_size)
  {
  }
  HDI f_t operator()(int id) { return problem_input_[id % problem_size_]; }

  const f_t* problem_input_;
  // TODO use i_t
  int problem_size_;
};

//
template <typename f_t>
static inline auto problem_wrap_container(const rmm::device_uvector<f_t>& in)
{
  return thrust::make_transform_iterator(thrust::make_counting_iterator(0),
                                         problem_wrapped_iterator<f_t>(in.data(), in.size()));
}

template <typename f_t>
struct power_two_func_t {
  HDI f_t operator()(f_t val) { return val * val; }
};

template <typename i_t, typename f_t>
void inline combine_constraint_bounds(const problem_t<i_t, f_t>& op_problem,
                                      rmm::device_uvector<f_t>& combined_bounds,
                                      bool batch_mode = false)
{
  if (!batch_mode) {
    combined_bounds.resize(op_problem.n_constraints, op_problem.handle_ptr->get_stream());
    if (combined_bounds.size() > 0) {
      raft::linalg::binaryOp(combined_bounds.data(),
                             op_problem.constraint_lower_bounds.data(),
                             op_problem.constraint_upper_bounds.data(),
                             op_problem.n_constraints,
                             combine_finite_abs_bounds<f_t>(),
                             op_problem.handle_ptr->get_stream());
    }
  } else {
    // In batch mode we use combined_constraint_bounds in convergeance_information to fill the
    // primal residual which will be bigger
    cuopt_assert(combined_bounds.size() % op_problem.n_constraints == 0,
                 "combined_bounds size must be a multiple of op_problem.n_constraints");
    // TODO later batch mode: different constraint bounds
    cub::DeviceTransform::Transform(
      cuda::std::make_tuple(problem_wrap_container(op_problem.constraint_lower_bounds),
                            problem_wrap_container(op_problem.constraint_upper_bounds)),
      combined_bounds.data(),
      combined_bounds.size(),
      combine_finite_abs_bounds<f_t>(),
      op_problem.handle_ptr->get_stream());
  }
}

template <typename f_t>
void inline compute_sum_bounds(const rmm::device_uvector<f_t>& constraint_lower_bounds,
                               const rmm::device_uvector<f_t>& constraint_upper_bounds,
                               rmm::device_scalar<f_t>& out,
                               rmm::cuda_stream_view stream_view)
{
  rmm::device_buffer d_temp_storage;
  size_t bytes = 0;
  auto main_op = [] HD(const thrust::tuple<f_t, f_t> t) {
    const f_t lower = thrust::get<0>(t);
    const f_t upper = thrust::get<1>(t);
    f_t sum         = f_t(0);
    if (isfinite(lower) && (lower != upper)) sum += lower * lower;
    if (isfinite(upper)) sum += upper * upper;
    return sum;
  };
  cub::DeviceReduce::TransformReduce(
    nullptr,
    bytes,
    thrust::make_zip_iterator(constraint_lower_bounds.data(), constraint_upper_bounds.data()),
    out.data(),
    constraint_lower_bounds.size(),
    cuda::std::plus<>{},
    main_op,
    f_t(0),
    stream_view);

  d_temp_storage.resize(bytes, stream_view);

  cub::DeviceReduce::TransformReduce(
    d_temp_storage.data(),
    bytes,
    thrust::make_zip_iterator(constraint_lower_bounds.data(), constraint_upper_bounds.data()),
    out.data(),
    constraint_lower_bounds.size(),
    cuda::std::plus<>{},
    main_op,
    f_t(0),
    stream_view);

  const f_t res = std::sqrt(out.value(stream_view));
  out.set_value_async(res, stream_view);

  // Sync since we are using local variable
  RAFT_CUDA_TRY(cudaStreamSynchronize(stream_view));
}

template <typename f_t>
struct violation {
  violation() {}
  violation(f_t* _scalar) {}
  __device__ __host__ f_t operator()(f_t value, f_t lower, f_t upper)
  {
    if (value < lower) {
      return lower - value;
    } else if (value > upper) {
      return value - upper;
    }
    return f_t(0);
  }
};

template <typename f_t, typename f_t2>
struct max_violation {
  max_violation() {}
  __device__ f_t operator()(const thrust::tuple<f_t, f_t2>& t) const
  {
    const f_t value   = thrust::get<0>(t);
    const f_t2 bounds = thrust::get<1>(t);
    const f_t lower   = get_lower(bounds);
    const f_t upper   = get_upper(bounds);
    f_t local_max     = f_t(0.0);
    if (isfinite(lower)) { local_max = raft::max(local_max, -value); }
    if (isfinite(upper)) { local_max = raft::max(local_max, value); }
    return local_max;
  }
};

template <typename f_t, typename f_t2>
struct divide_check_zero {
  __device__ f_t2 operator()(f_t2 bounds, f_t value)
  {
    if (value == f_t{0}) {
      return f_t2{0, 0};
    } else {
      return f_t2{get_lower(bounds) / value, get_upper(bounds) / value};
    }
  }
};

// Necessary until we have access to cuda::zip_transform_iterator (CTK 13.2)
template <typename f_t>
struct tuple_multiplies {
  HDI f_t operator()(thrust::tuple<f_t, f_t> value)
  {
    return thrust::get<0>(value) * thrust::get<1>(value);
  }
};

template <typename f_t, typename f_t2>
struct bound_value_gradient {
  __device__ f_t operator()(f_t value, f_t2 bounds)
  {
    f_t lower = get_lower(bounds);
    f_t upper = get_upper(bounds);
    if (value > f_t(0) && value < f_t(0)) { return 0; }
    return value > f_t(0) ? lower : upper;
  }
};

template <typename f_t>
struct constraint_bound_value_reduced_cost_product {
  __device__ f_t operator()(f_t value, f_t lower, f_t upper)
  {
    f_t bound_value = f_t(0);
    if (value > f_t(0)) {
      // A positive reduced cost is associated with a binding lower bound.
      bound_value = lower;
    } else if (value < f_t(0)) {
      // A negative reduced cost is associated with a binding upper bound.
      bound_value = upper;
    }
    f_t val = isfinite(bound_value) ? value * bound_value : f_t(0);
    return val;
  }
};

template <typename f_t, typename f_t2>
struct bound_value_reduced_cost_product {
  __device__ f_t operator()(f_t value, f_t2 variable_bounds)
  {
    f_t lower       = get_lower(variable_bounds);
    f_t upper       = get_upper(variable_bounds);
    f_t bound_value = f_t(0);
    if (value > f_t(0)) {
      // A positive reduced cost is associated with a binding lower bound.
      bound_value = lower;
    } else if (value < f_t(0)) {
      // A negative reduced cost is associated with a binding upper bound.
      bound_value = upper;
    }
    f_t val = isfinite(bound_value) ? value * bound_value : f_t(0);
    return val;
  }
};

template <typename f_t>
struct copy_gradient_if_should_be_reduced_cost {
  __device__ f_t operator()(f_t value, f_t bound, f_t gradient)
  {
    if (gradient == f_t(0)) { return gradient; }
    if (raft::abs(value - bound) <= raft::abs(value)) { return gradient; }
    return f_t(0);
  }
};

template <typename f_t>
struct copy_gradient_if_finite_bounds {
  __device__ f_t operator()(f_t bound, f_t gradient)
  {
    if (gradient == f_t(0)) { return gradient; }
    if (isfinite(bound)) { return gradient; }
    return f_t(0);
  }
};

template <typename f_t>
struct transform_constraint_lower_bounds {
  __device__ f_t operator()(f_t lower, f_t upper)
  {
    return isfinite(upper) ? -raft::myInf<f_t>() : 0;
  }
};

template <typename f_t>
struct transform_constraint_upper_bounds {
  __device__ f_t operator()(f_t lower, f_t upper)
  {
    return isfinite(lower) ? raft::myInf<f_t>() : 0;
  }
};

template <typename f_t>
struct zero_if_is_finite {
  __device__ f_t operator()(f_t value)
  {
    if (isfinite(value)) { return 0; }
    return value;
  }
};

template <typename f_t>
struct negate_t {
  __device__ f_t operator()(f_t value) { return -value; }
};

template <typename i_t, typename f_t>
struct minus {
  __device__ minus(raft::device_span<f_t> a, raft::device_span<f_t> b) : a_(a), b_(b) {}

  DI f_t operator()(i_t index) { return a_[index] - b_[index]; }

  raft::device_span<f_t> a_;
  raft::device_span<f_t> b_;
};

template <typename i_t, typename f_t>
struct identity {
  __device__ identity(raft::device_span<f_t> a) : a_(a) {}

  DI f_t operator()(i_t index) { return a_[index]; }

  raft::device_span<f_t> a_;
};

template <typename i_t, typename f_t>
struct compute_direction_and_threshold {
  compute_direction_and_threshold(
    typename pdlp_restart_strategy_t<i_t, f_t>::view_t restart_strategy_view)
    : view(restart_strategy_view)
  {
  }

  __device__ void operator()(i_t idx)
  {
    if (view.center_point[idx] >= view.upper_bound[idx] && view.objective_vector[idx] <= f_t(0))
      return;
    if (view.center_point[idx] <= view.lower_bound[idx] && view.objective_vector[idx] >= f_t(0))
      return;

    if (view.objective_vector[idx] == f_t(0.0)) {
      view.threshold[idx] = std::numeric_limits<f_t>::infinity();
      return;
    }

    view.direction_full[idx] = -view.objective_vector[idx] / view.weights[idx];

    if (view.direction_full[idx] > f_t(0))
      view.threshold[idx] =
        (view.upper_bound[idx] - view.center_point[idx]) / view.direction_full[idx];
    else if (view.direction_full[idx] < f_t(0))
      view.threshold[idx] =
        (view.lower_bound[idx] - view.center_point[idx]) / view.direction_full[idx];
  }

 private:
  typename pdlp_restart_strategy_t<i_t, f_t>::view_t view;
};

template <typename i_t, typename f_t>
struct weighted_l2_if_infinite {
  weighted_l2_if_infinite(typename pdlp_restart_strategy_t<i_t, f_t>::view_t restart_strategy_view)
    : view(restart_strategy_view)
  {
  }

  __device__ f_t operator()(i_t idx)
  {
    // If this threshold value is inf, squared norm of direction (if not 0 to not participate)
    return (isinf(view.threshold[idx]))
             ? view.direction_full[idx] * view.direction_full[idx] * view.weights[idx]
             : f_t(0);
  }

 private:
  typename pdlp_restart_strategy_t<i_t, f_t>::view_t view;
};

template <typename f_t>
f_t device_to_host_value(f_t* iter)
{
  f_t host_value;
  cudaMemcpy(&host_value, iter, sizeof(f_t), cudaMemcpyDeviceToHost);
  return host_value;
}

template <typename i_t, typename f_t>
void inline my_l2_norm(const f_t* in, f_t* out, size_t size, raft::handle_t const* handle_ptr)
{
  constexpr int stride = 1;
  RAFT_CUBLAS_TRY(raft::linalg::detail::cublasnrm2(
    handle_ptr->get_cublas_handle(), size, in, stride, out, handle_ptr->get_stream()));
}

template <typename i_t, typename f_t>
void inline my_l2_norm(const rmm::device_uvector<f_t>& input_vector,
                       rmm::device_scalar<f_t>& result,
                       raft::handle_t const* handle_ptr)
{
  my_l2_norm<i_t, f_t>(input_vector.data(), result.data(), input_vector.size(), handle_ptr);
}

template <typename i_t, typename f_t>
void inline my_l2_norm(const rmm::device_uvector<f_t>& input_vector,
                       rmm::device_uvector<f_t>& result,
                       raft::handle_t const* handle_ptr)
{
  my_l2_norm<i_t, f_t>(input_vector.data(), result.data(), input_vector.size(), handle_ptr);
}

template <typename i_t, typename f_t>
void inline my_l2_weighted_norm(const rmm::device_uvector<f_t>& input_vector,
                                f_t weight,
                                rmm::device_scalar<f_t>& result,
                                rmm::cuda_stream_view stream)
{
  auto fin_op  = [] __device__(f_t in) { return raft::sqrt(in); };
  auto main_op = [weight] __device__(f_t in, i_t _) { return in * in * weight; };
  raft::linalg::reduce<true, true, f_t, f_t, i_t>(result.data(),
                                                  input_vector.data(),
                                                  (i_t)input_vector.size(),
                                                  1,
                                                  f_t(0.0),
                                                  stream,
                                                  false,
                                                  main_op,
                                                  raft::Sum<f_t>(),
                                                  fin_op);
}

template <typename f_t>
struct is_nan_or_inf {
  __device__ bool operator()(const f_t x) { return isnan(x) || isinf(x); }
};

// Used to compute the linf of (residual_i - rel * b/c_i)
template <typename i_t, typename f_t>
struct relative_residual_t {
  __device__ f_t operator()(const thrust::tuple<f_t, f_t>& t) const
  {
    const f_t residual = thrust::get<0>(t);
    // Rhs for either primal (b) and dual (c)
    const f_t rhs = thrust::get<1>(t);

    // Used for best primal so far, count how many constraints are violated
    if (abs_.has_value() && nb_violated_constraints_.has_value()) {
      if (residual >= *abs_ + rel_ * rhs) atomicAdd(*nb_violated_constraints_, 1);
    }
    return residual - rel_ * rhs;
  }

  const f_t rel_;
  std::optional<const f_t> abs_{std::nullopt};
  std::optional<i_t*> nb_violated_constraints_{std::nullopt};
};

template <typename f_t>
struct abs_t {
  __device__ f_t operator()(const f_t in) const { return raft::abs(in); }
};

template <typename f_t>
void inline my_inf_norm(const rmm::device_uvector<f_t>& input_vector,
                        f_t* result,
                        raft::handle_t const* handle_ptr)
{
  auto stream   = handle_ptr->get_stream();
  auto abs_iter = thrust::make_transform_iterator(input_vector.data(), abs_t<f_t>{});
  auto n        = input_vector.size();

  void* d_temp      = nullptr;
  size_t temp_bytes = 0;
  cub::DeviceReduce::Max(d_temp, temp_bytes, abs_iter, result, n, stream);
  rmm::device_buffer temp_buf(temp_bytes, stream);
  cub::DeviceReduce::Max(temp_buf.data(), temp_bytes, abs_iter, result, n, stream);
}

template <typename f_t>
void inline my_inf_norm(const rmm::device_uvector<f_t>& input_vector,
                        rmm::device_scalar<f_t>& result,
                        raft::handle_t const* handle_ptr)
{
  my_inf_norm(input_vector, result.data(), handle_ptr);
}

template <typename f_t>
void inline my_inf_norm(const rmm::device_uvector<f_t>& input_vector,
                        rmm::device_uvector<f_t>& result,
                        raft::handle_t const* handle_ptr)
{
  my_inf_norm(input_vector, result.data(), handle_ptr);
}

}  // namespace cuopt::linear_programming::detail
