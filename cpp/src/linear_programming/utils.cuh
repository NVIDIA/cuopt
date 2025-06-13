/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved. SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#include <linear_programming/pdlp_constants.hpp>
#include <linear_programming/restart_strategy/pdlp_restart_strategy.cuh>
#include <optional>
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

template <typename f_t>
struct a_sub_scalar_times_b {
  a_sub_scalar_times_b(const f_t* scalar) : scalar_{scalar} {}
  __device__ __forceinline__ f_t operator()(f_t a, f_t b) { return a - *scalar_ * b; }

  const f_t* scalar_;
};

template <typename f_t>
struct primal_projection {
  primal_projection(const f_t* step_size) : step_size_(step_size) {}

  __device__ __forceinline__ thrust::tuple<f_t, f_t, f_t> operator()(
    f_t primal, f_t obj_coeff, f_t AtY, f_t lower, f_t upper)
  {
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
struct clamp {
  __device__ f_t operator()(f_t value, f_t lower, f_t upper)
  {
    return raft::min<f_t>(raft::max<f_t>(value, lower), upper);
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

template <typename i_t, typename f_t>
void inline combine_constraint_bounds(const problem_t<i_t, f_t>& op_problem,
                                      rmm::device_uvector<f_t>& combined_bounds)
{
  combined_bounds.resize(op_problem.n_constraints, op_problem.handle_ptr->get_stream());
  if (combined_bounds.size() > 0) {
    raft::linalg::binaryOp(combined_bounds.data(),
                           op_problem.constraint_lower_bounds.data(),
                           op_problem.constraint_upper_bounds.data(),
                           op_problem.n_constraints,
                           combine_finite_abs_bounds<f_t>(),
                           op_problem.handle_ptr->get_stream());
  }
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

template <typename f_t>
struct max_violation {
  max_violation() {}
  __device__ f_t operator()(const thrust::tuple<f_t, f_t, f_t>& t) const
  {
    const f_t value = thrust::get<0>(t);
    const f_t lower = thrust::get<1>(t);
    const f_t upper = thrust::get<2>(t);
    f_t local_max   = f_t(0.0);
    if (isfinite(lower)) { local_max = raft::max(local_max, -value); }
    if (isfinite(upper)) { local_max = raft::max(local_max, value); }
    return local_max;
  }
};

template <typename f_t>
struct bound_value_gradient {
  __device__ f_t operator()(f_t value, f_t lower, f_t upper)
  {
    if (value > f_t(0) && value < f_t(0)) { return 0; }
    return value > f_t(0) ? lower : upper;
  }
};

template <typename f_t>
struct bound_value_reduced_cost_product {
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
void inline my_l2_norm(const rmm::device_uvector<f_t>& input_vector,
                       rmm::device_scalar<f_t>& result,
                       raft::handle_t const* handle_ptr)
{
  constexpr int stride = 1;
  RAFT_CUBLAS_TRY(raft::linalg::detail::cublasnrm2(handle_ptr->get_cublas_handle(),
                                                   input_vector.size(),
                                                   input_vector.data(),
                                                   stride,
                                                   result.data(),
                                                   handle_ptr->get_stream()));
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
      if (residual >= abs_.value() + rel_ * rhs) atomicAdd(nb_violated_constraints_.value(), 1);
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
                        rmm::device_scalar<f_t>& result,
                        raft::handle_t const* handle_ptr)
{
  const f_t neutral = f_t(0.0);
  thrust::device_ptr<f_t> result_ptr(result.data());

  *result_ptr = thrust::transform_reduce(handle_ptr->get_thrust_policy(),
                                         input_vector.data(),
                                         input_vector.data() + input_vector.size(),
                                         abs_t<f_t>{},
                                         neutral,
                                         thrust::maximum<f_t>());
}

}  // namespace cuopt::linear_programming::detail
