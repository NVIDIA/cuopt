/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2025, NVIDIA CORPORATION & AFFILIATES. All rights
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

#include <cuopt/routing/routing_structures.hpp>
#include <utilities/cuda_helpers.cuh>

namespace cuopt {
namespace routing {
namespace detail {

constexpr double EPSILON = 0.0001;

enum class dim_t {
  DIST = 0,
  // time
  TIME,
  // pdp capacity ( positive/negative supply )
  CAP,
  PRIZE,
  TASKS,
  SERVICE_TIME,
  MISMATCH,
  BREAK,
  VEHICLE_FIXED_COST,
  SIZE
};

constexpr double get_nominal_diff(double c_1, double c_2)
{
  double diff = c_1 - c_2;
  if (EPSILON < diff && diff < 0.) { return 0.; }
  return diff;
}

// This is currently a different cost structure than the legacy solver
// As we merge two solvers, reconsolidate these two cost structures
template <class enum_t>
struct static_vec_t {
 public:
  static constexpr size_t N = (size_t)enum_t::SIZE;
  HDI static_vec_t(const double init_cost[N])
  {
    for (size_t i = 0; i < N; ++i) {
      cost[i] = init_cost[i];
    }
  }

  static_vec_t(const static_vec_t& other) = default;

  HDI static_vec_t() { zero_initialize(); }

  HDI void zero_initialize()
  {
    for (size_t i = 0; i < N; ++i) {
      cost[i] = 0.;
    }
  }

  HDI void copy_from(const static_vec_t& other)
  {
    for (size_t i = 0; i < N; ++i) {
      cost[i] = other[i];
    }
  }

  HDI double& operator[](size_t idx)
  {
    cuopt_assert(idx < N, "Too big index for cost dimension");
    return cost[idx];
  }

  HDI double& operator[](enum_t idx) { return cost[(size_t)idx]; }

  HDI double operator[](size_t idx) const
  {
    cuopt_assert(idx < N, "Too big index for cost dimension");
    return cost[idx];
  }

  HDI double operator[](enum_t idx) const { return cost[(size_t)idx]; }

  DI void atomic_add(const static_vec_t& b)
  {
    for (size_t i = 0; i < N; ++i) {
      atomicAdd(cost + i, b[i]);
    }
  }

  HDI void operator+=(const static_vec_t& b)
  {
    for (size_t i = 0; i < N; ++i) {
      cost[i] += b[i];
    }
  }

  HDI void operator-=(const static_vec_t& b)
  {
    for (size_t i = 0; i < N; ++i) {
      cost[i] -= b[i];
    }
  }

  HDI static_vec_t operator+(const static_vec_t& b) const
  {
    static_vec_t return_cost(*this);
    return_cost += b;
    return return_cost;
  }

  HDI static_vec_t operator*(const static_vec_t& b) const
  {
    static_vec_t return_cost(*this);
    for (size_t i = 0; i < N; ++i) {
      return_cost[i] = return_cost[i] * b[i];
    }
    return return_cost;
  }

  HDI static_vec_t operator-(const static_vec_t& b) const
  {
    static_vec_t return_cost(*this);
    return_cost -= b;
    return return_cost;
  }

  HDI double sum() const
  {
    double total = 0.;
    for (size_t i = 0; i < N; ++i) {
      total += cost[i];
    }
    return total;
  }

  HDI void print() const
  {
    printf("cost: ");
    for (size_t i = 0; i < N; ++i) {
      printf("%f\t", cost[i]);
    }
    printf("\n");
  }

  std::vector<double> to_vec() const
  {
    std::vector<double> cost_vec(N, 0);
    for (size_t i = 0; i < N; ++i) {
      cost_vec[i] = cost[i];
    }
    return cost_vec;
  }

  static HDI double dot(const static_vec_t& a, const static_vec_t& b)
  {
    double total = 0.;
    for (size_t i = 0; i < N; ++i) {
      total += a.cost[i] * b.cost[i];
    }
    return total;
  }

  static HDI static_vec_t nominal_diff(static_vec_t c_1, static_vec_t c_2)
  {
    static_vec_t diff;
    for (size_t i = 0; i < N; ++i) {
      diff[i] = get_nominal_diff(c_1[i], c_2[i]);
    }
    return diff;
  }

 private:
  double cost[N];
};

using infeasible_cost_t = static_vec_t<dim_t>;
using objective_cost_t  = static_vec_t<objective_t>;

struct cost_dimension_info_t {
  bool has_max_constraint = false;
  HDI bool has_constraints() const { return has_max_constraint; }
};

struct time_dimension_info_t {
  HDI bool should_compute_travel_time() const { return has_max_constraint || has_travel_time_obj; }
  bool has_max_constraint  = false;
  bool has_travel_time_obj = false;
  HDI constexpr bool has_constraints() const { return true; }
};

struct capacity_dimension_info_t {
  uint8_t n_capacity_dimensions = 0;
  HDI bool has_constraints() const { return n_capacity_dimensions > 0; }
};

struct prize_dimension_info_t {
  // prize collection does not have dimension
  constexpr bool has_constraints() const { return false; }
};

struct tasks_dimension_info_t {
  double mean_tasks = 0.;
  constexpr bool has_constraints() const { return false; }
};

struct service_time_dimension_info_t {
  double mean_service_time = 0.;
  constexpr bool has_constraints() const { return false; }
};

struct mismatch_dimension_info_t {
  bool has_vehicle_order_match = false;
  constexpr bool has_constraints() const { return has_vehicle_order_match; }
};

struct break_dimension_info_t {
  bool has_breaks = false;
  constexpr bool has_constraints() const { return has_breaks; };
};

struct vehicle_fixed_cost_dimension_info_t {
  constexpr bool has_constraints() const { return false; };
};

/**
 * @brief Get const reference to specified dimension of an object. This assumes that the object
 * being passed has all the dimensions and they are named in a specific way
 *
 * @tparam I
 * @tparam T
 * @param obj
 * @return HDI&
 */
template <dim_t I, class T>
static HDI const auto& get_dimension_of(const T& obj) noexcept
{
  if constexpr (I == dim_t::TIME) {
    return obj.time_dim;
  } else if constexpr (I == dim_t::DIST) {
    return obj.distance_dim;
  } else if constexpr (I == dim_t::CAP) {
    return obj.capacity_dim;
  } else if constexpr (I == dim_t::PRIZE) {
    return obj.prize_dim;
  } else if constexpr (I == dim_t::TASKS) {
    return obj.tasks_dim;
  } else if constexpr (I == dim_t::SERVICE_TIME) {
    return obj.service_time_dim;
  } else if constexpr (I == dim_t::MISMATCH) {
    return obj.mismatch_dim;
  } else if constexpr (I == dim_t::BREAK) {
    return obj.break_dim;
  } else if constexpr (I == dim_t::VEHICLE_FIXED_COST) {
    return obj.vehicle_fixed_cost_dim;
  }
}

template <size_t I, class T>
static HDI const auto& get_dimension_of(const T& obj) noexcept
{
  static_assert(I < (size_t)dim_t::SIZE, "template parameter I cannot exceed (size_t)dim_t::SIZE");
  return get_dimension_of<(dim_t)I>(obj);
}

/**
 * @brief Get the string from dim_t enum
 *
 * @param dim Dimension to get string of
 * @return String of dimension
 */
template <int I>
constexpr auto dim_to_string() noexcept
{
  if constexpr (I == (int)dim_t::TIME) {
    return "Time dimension";
  } else if constexpr (I == (int)dim_t::DIST) {
    return "Distance dimension";
  } else if constexpr (I == (int)dim_t::CAP) {
    return "Capacity dimension";
  } else if constexpr (I == (int)dim_t::PRIZE) {
    return "Prize dimension";
  } else if constexpr (I == (int)dim_t::TASKS) {
    return "Tasks dimension";
  } else if constexpr (I == (int)dim_t::SERVICE_TIME) {
    return "Service time dimension";
  } else if constexpr (I == (int)dim_t::MISMATCH) {
    return "Vehicle order match dimension";
  } else if constexpr (I == (int)dim_t::BREAK) {
    return "Break dimension";
  } else if constexpr (I == (int)dim_t::VEHICLE_FIXED_COST) {
    return "Vehicle cost dimension";
  }
}

/**
 * @brief Get non const reference to specified dimension of an object
 *
 * @tparam I
 * @tparam T
 * @param obj
 * @return HDI&
 */
template <dim_t I, class T>
static HDI auto& get_dimension_of(T& obj) noexcept
{
  const auto& const_obj = *(const_cast<const T*>(&obj));
  auto& const_dim       = get_dimension_of<I>(const_obj);
  return const_cast<std::decay_t<decltype(const_dim)>&>(const_dim);
}

template <size_t I, class T>
static HDI auto& get_dimension_of(T& obj) noexcept
{
  static_assert(I < (size_t)dim_t::SIZE, "template parameter I cannot exceed (size_t)dim_t::SIZE");
  return get_dimension_of<(dim_t)I>(obj);
}

/**
 * @brief Utility class/struct to enacapsulate all the
 * enabled dimensions in a given problem
 *
 */
class enabled_dimensions_t {
 public:
  enabled_dimensions_t() = default;

  /**
   * @brief Enable a specified dimension
   *
   * @param dim
   */
  void enable_dimension(dim_t dim) { hash |= (1 << (int)dim); }

  /**
   * @brief Enable a specified objective
   *
   * @param obj
   */
  void enable_objective(objective_t obj, double weight)
  {
    obj_hash |= (1 << (int)obj);
    objective_weights[obj] = weight;
  }

  /**
   * @brief Enable all dimensions
   *
   */
  void enable_all_dimensions()
  {
    for (size_t i = 0; i < (size_t)dim_t::SIZE; ++i) {
      enable_dimension((dim_t)i);
    }
  }

  /**
   * @returns true if a given dimension is enabled
   *
   * @param dim
   * @return true
   * @return false
   */
  HDI bool has_dimension(dim_t dim) const { return hash & (1 << (int)dim); }

  HDI bool has_objective(objective_t obj) const { return obj_hash & (1 << (int)obj); }

  template <size_t I>
  HDI auto& get_dimension() const noexcept
  {
    using my_type = std::decay_t<decltype((*this))>;
    return detail::get_dimension_of<I, my_type>(*this);
  }

  template <size_t I>
  HDI auto& get_dimension() noexcept
  {
    using my_type = std::decay_t<decltype((*this))>;
    return detail::get_dimension_of<I, my_type>(*this);
  }

  template <dim_t dim>
  HDI auto& get_dimension() const noexcept
  {
    return get_dimension<(size_t)dim>();
  }

  template <dim_t dim>
  HDI auto& get_dimension() noexcept
  {
    return get_dimension<(size_t)dim>();
  }

  cost_dimension_info_t distance_dim;
  time_dimension_info_t time_dim;
  capacity_dimension_info_t capacity_dim;
  prize_dimension_info_t prize_dim;
  tasks_dimension_info_t tasks_dim;
  service_time_dimension_info_t service_time_dim;
  mismatch_dimension_info_t mismatch_dim;
  break_dimension_info_t break_dim;
  vehicle_fixed_cost_dimension_info_t vehicle_fixed_cost_dim;

  objective_cost_t objective_weights;
  bool is_tsp{false};

 private:
  // 32 bits are sufficient to encode all dimensions
  // If we need to encode more information, we may reevaluate this
  static_assert((size_t)dim_t::SIZE < 32u, "Use higher precision integer!");
  uint32_t hash = 0;

  static_assert((size_t)objective_t::SIZE < 8u, "Use higher precision integer!");
  uint8_t obj_hash = 0;
};

}  // namespace detail
}  // namespace routing
}  // namespace cuopt
