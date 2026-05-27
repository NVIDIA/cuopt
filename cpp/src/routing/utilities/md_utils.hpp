/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#pragma once

#include <cuopt/error.hpp>
#include <cuopt/routing/data_model_view.hpp>

#include <thrust/functional.h>
#include <thrust/logical.h>
#include <cuda/stream>
#include <rmm/device_uvector.hpp>
#include <utilities/copy_helpers.hpp>
#include <utilities/macros.cuh>
#include <vector>

namespace cuopt {
namespace routing {

template <typename f_t, size_t NCON_DIMS = 4>
struct mdarray_view_t {
  constexpr auto get_vehicle_type_matrices(uint8_t vehicle_type) const
  {
    return buffer_ptr + vehicle_type * (extent[3] * extent[2] * extent[1]);
  }

  constexpr auto get_cost_matrix(uint8_t vehicle_type, uint8_t matrix_type) const
  {
    auto vehicle_type_matrices = get_vehicle_type_matrices(vehicle_type);
    return vehicle_type_matrices + (extent[3] * extent[2] * matrix_type);
  }

  constexpr auto get_cost_matrix(uint8_t vehicle_type) const
  {
    return get_cost_matrix(vehicle_type, cost_matrix_index);
  }

  constexpr auto get_distance_matrix(uint8_t vehicle_type) const
  {
    return get_cost_matrix(vehicle_type, distance_matrix_index);
  }

  constexpr auto get_time_matrix(uint8_t vehicle_type) const
  {
    if (time_matrix_index != cost_matrix_index) {
      return get_cost_matrix(vehicle_type, time_matrix_index);
    }
    if (extent[1] == 2 && distance_matrix_index == cost_matrix_index) {
      return get_cost_matrix(vehicle_type, static_cast<uint8_t>(extent[1] - 1));
    }
    return get_cost_matrix(vehicle_type, cost_matrix_index);
  }

  f_t const* buffer_ptr{nullptr};
  // dim4(n_vehicle_types, n_matrix_types, n_loc, n_loc)
  size_t extent[NCON_DIMS];
  uint8_t cost_matrix_index{0};
  uint8_t distance_matrix_index{0};
  uint8_t time_matrix_index{0};
};

template <typename f_t, size_t NCON_DIMS = 4>
struct h_mdarray_t {
  h_mdarray_t() {}
  h_mdarray_t(std::vector<size_t> const& extent_)
  {
    cuopt_assert(extent_.size() == NCON_DIMS, "Wrong dimensions");
    size_t size = 1;
    for (size_t i = 0; i < NCON_DIMS; ++i) {
      size *= extent_[i];
      extent[i] = extent_[i];
    }
    buffer.resize(size);
  }

  constexpr auto get_vehicle_type_matrices(uint8_t vehicle_type)
  {
    return buffer.data() + vehicle_type * (extent[3] * extent[2] * extent[1]);
  }

  constexpr auto get_cost_matrix(uint8_t vehicle_type, uint8_t matrix_type)
  {
    auto vehicle_type_matrices = get_vehicle_type_matrices(vehicle_type);
    return vehicle_type_matrices + (extent[3] * extent[2] * matrix_type);
  }

  constexpr auto get_cost_matrix(uint8_t vehicle_type)
  {
    return get_cost_matrix(vehicle_type, cost_matrix_index);
  }

  constexpr auto get_distance_matrix(uint8_t vehicle_type)
  {
    return get_cost_matrix(vehicle_type, distance_matrix_index);
  }

  constexpr auto get_time_matrix(uint8_t vehicle_type)
  {
    return get_cost_matrix(vehicle_type, time_matrix_index);
  }

  auto view() const
  {
    mdarray_view_t<f_t> view;
    view.buffer_ptr             = buffer.data();
    view.cost_matrix_index      = cost_matrix_index;
    view.distance_matrix_index  = distance_matrix_index;
    view.time_matrix_index      = time_matrix_index;
    for (size_t i = 0; i < NCON_DIMS; ++i) {
      view.extent[i] = extent[i];
    }
    return view;
  }
  size_t extent[NCON_DIMS];
  std::vector<f_t> buffer;
  uint8_t cost_matrix_index{0};
  uint8_t distance_matrix_index{0};
  uint8_t time_matrix_index{0};
};

template <typename f_t, size_t NCON_DIMS = 4>
struct d_mdarray_t {
  d_mdarray_t(cuda::stream_ref stream_) : buffer(0, stream_), stream(stream_) {}
  d_mdarray_t(std::vector<size_t> const& extent_, cuda::stream_ref stream_)
    : buffer(0, stream_), stream(stream_)
  {
    cuopt_assert(extent_.size() == NCON_DIMS, "Wrong dimensions");
    size_t size = 1;
    for (size_t i = 0; i < NCON_DIMS; ++i) {
      size *= extent_[i];
      extent[i] = extent_[i];
    }
    buffer.resize(size, stream_);
  }

  constexpr auto get_vehicle_type_matrices(uint8_t vehicle_type)
  {
    return buffer.data() + vehicle_type * (extent[3] * extent[2] * extent[1]);
  }

  constexpr auto get_cost_matrix(uint8_t vehicle_type, uint8_t matrix_type)
  {
    auto vehicle_type_matrices = get_vehicle_type_matrices(vehicle_type);
    return vehicle_type_matrices + (extent[3] * extent[2] * matrix_type);
  }

  constexpr auto get_cost_matrix(uint8_t vehicle_type)
  {
    return get_cost_matrix(vehicle_type, cost_matrix_index);
  }

  constexpr auto get_distance_matrix(uint8_t vehicle_type)
  {
    return get_cost_matrix(vehicle_type, distance_matrix_index);
  }

  constexpr auto get_time_matrix(uint8_t vehicle_type)
  {
    return get_cost_matrix(vehicle_type, time_matrix_index);
  }

  auto view() const
  {
    mdarray_view_t<f_t> view;
    view.buffer_ptr             = buffer.data();
    view.cost_matrix_index      = cost_matrix_index;
    view.distance_matrix_index  = distance_matrix_index;
    view.time_matrix_index      = time_matrix_index;
    for (size_t i = 0; i < NCON_DIMS; ++i) {
      view.extent[i] = extent[i];
    }
    return view;
  }

  size_t extent[NCON_DIMS];
  rmm::device_uvector<f_t> buffer;
  cuda::stream_ref stream;
  uint8_t cost_matrix_index{0};
  uint8_t distance_matrix_index{0};
  uint8_t time_matrix_index{0};
};

namespace detail {

template <typename i_t, typename f_t>
bool limit_matrix_entries(f_t* matrix, i_t width, raft::handle_t const* handle_ptr)
{
  i_t mat_size  = width * width;
  f_t max_value = 1.0e+30;

  bool exceeds_max =
    thrust::any_of(handle_ptr->get_thrust_policy(),
                   matrix,
                   matrix + mat_size,
                   [max_value] __device__(f_t x) -> bool { return x > max_value; });

  if (exceeds_max) {
    thrust::transform(
      handle_ptr->get_thrust_policy(),
      matrix,
      matrix + mat_size,
      matrix,
      [max_value] __device__(f_t x) -> f_t { return x > max_value ? max_value : x; });
  }
  return exceeds_max;
}

template <typename i_t, typename f_t>
void fill_data_model_matrices(data_model_view_t<i_t, f_t>& data_model, d_mdarray_t<f_t>& matrices)

{
  i_t n_vehicle_types = matrices.extent[0];
  for (auto vehicle_type = 0; vehicle_type < n_vehicle_types; ++vehicle_type) {
    data_model.add_cost_matrix(matrices.get_cost_matrix(vehicle_type), vehicle_type);
    if (matrices.distance_matrix_index != matrices.cost_matrix_index) {
      data_model.add_distance_matrix(matrices.get_distance_matrix(vehicle_type), vehicle_type);
    }
    if (matrices.time_matrix_index != matrices.cost_matrix_index) {
      data_model.add_transit_time_matrix(matrices.get_time_matrix(vehicle_type), vehicle_type);
    }
  }
}

template <typename f_t>
auto create_host_mdarray(size_t nlocations, uint8_t n_vehicle_types, uint8_t n_matrix_types)
{
  std::vector<size_t> full_matrix_extent{n_vehicle_types, n_matrix_types, nlocations, nlocations};
  h_mdarray_t<f_t> matrices{full_matrix_extent};
  return matrices;
}

template <typename f_t>
auto create_device_mdarray(size_t nlocations,
                           uint8_t n_vehicle_types,
                           uint8_t n_matrix_types,
                           cuda::stream_ref stream)
{
  std::vector<size_t> full_matrix_extent{n_vehicle_types, n_matrix_types, nlocations, nlocations};
  d_mdarray_t<f_t> matrices{full_matrix_extent, stream};
  return matrices;
}

inline auto get_unique_vehicle_types(const raft::device_span<uint8_t const>& vehicle_types,
                                     cuda::stream_ref stream)
{
  auto h_vehicle_types = cuopt::host_copy(vehicle_types, stream);

  std::map<uint8_t, uint8_t> vehicle_types_map;

  if (h_vehicle_types.empty()) {
    vehicle_types_map[0] = 0;
  } else {
    for (auto& v : h_vehicle_types) {
      if (!vehicle_types_map.count(v)) { vehicle_types_map[v] = vehicle_types_map.size(); }
    }
  }

  return vehicle_types_map;
}

template <typename i_t, typename f_t>
bool has_distance_matrix(data_model_view_t<i_t, f_t> const& data_model)
{
  auto vehicle_types_map = get_unique_vehicle_types(data_model.get_vehicle_types(),
                                                    data_model.get_handle_ptr()->get_stream());
  for (auto& [old_type, new_type] : vehicle_types_map) {
    if (data_model.get_distance_matrix(old_type)) { return true; }
  }
  return false;
}

template <typename i_t, typename f_t>
bool has_transit_time_matrix(data_model_view_t<i_t, f_t> const& data_model)
{
  auto vehicle_types_map = get_unique_vehicle_types(data_model.get_vehicle_types(),
                                                    data_model.get_handle_ptr()->get_stream());
  for (auto& [old_type, new_type] : vehicle_types_map) {
    if (data_model.get_transit_time_matrix(old_type)) { return true; }
  }
  return false;
}

template <typename i_t, typename f_t>
auto get_cost_matrix_type_dim(data_model_view_t<i_t, f_t> const& data_model)
{
  auto n_matrix_types = 2;
  if (has_transit_time_matrix<i_t, f_t>(data_model)) { ++n_matrix_types; }
  return n_matrix_types;
}

template <typename i_t, typename f_t>
std::tuple<float const*, float const*, float const*> get_vehicle_matrices(
  data_model_view_t<i_t, f_t> const& data_model, uint8_t vehicle_type)
{
  auto cost_matrix = data_model.get_cost_matrix(vehicle_type);
  if (!cost_matrix)
    cuopt_expects(
      false, error_type_t::ValidationError, "Set vehicle types when using multiple matrices");
  auto distance_matrix = data_model.get_distance_matrix(vehicle_type);
  auto time_matrix = data_model.get_transit_time_matrix(vehicle_type);
  if (!time_matrix) time_matrix = cost_matrix;
  return std::make_tuple(cost_matrix, distance_matrix, time_matrix);
}

template <typename i_t, typename f_t>
void fill_mdarray_from_data_model(d_mdarray_t<f_t>& matrices,
                                  data_model_view_t<i_t, f_t> const& data_model)
{
  auto stream            = data_model.get_handle_ptr()->get_stream();
  auto vehicle_types     = data_model.get_vehicle_types();
  auto nlocations        = data_model.get_num_locations();
  auto vehicle_types_map = get_unique_vehicle_types(vehicle_types, stream);

  matrices.cost_matrix_index     = 0;
  matrices.distance_matrix_index = 1;
  matrices.time_matrix_index     = 0;
  {
    uint8_t next_index = 2;
    if (has_transit_time_matrix<i_t, f_t>(data_model)) {
      matrices.time_matrix_index = next_index++;
    }
  }

  for (auto& [old_type, new_type] : vehicle_types_map) {
    auto [cost_matrix, distance_matrix, time_matrix] =
      get_vehicle_matrices<i_t, f_t>(data_model, old_type);
    auto cost_matrix_span = matrices.get_cost_matrix(new_type, matrices.cost_matrix_index);
    raft::copy(cost_matrix_span, cost_matrix, nlocations * nlocations, stream);

    if (limit_matrix_entries(cost_matrix_span, nlocations, data_model.get_handle_ptr())) {
      std::cout << "\nMax cost matrix value overriden to 1.0e+30";
    }

    auto distance_matrix_span = matrices.get_cost_matrix(new_type, matrices.distance_matrix_index);
    if (has_distance_matrix<i_t, f_t>(data_model)) {
      raft::copy(distance_matrix_span, distance_matrix, nlocations * nlocations, stream);
      if (limit_matrix_entries(distance_matrix_span, nlocations, data_model.get_handle_ptr())) {
        std::cout << "\nMax distance matrix value overriden to 1.0e+30";
      }
    } else {
      thrust::fill(rmm::exec_policy(stream),
                   distance_matrix_span,
                   distance_matrix_span + (nlocations * nlocations),
                   f_t{0});
    }

    if (matrices.time_matrix_index != matrices.cost_matrix_index) {
      auto time_matrix_span = matrices.get_cost_matrix(new_type, matrices.time_matrix_index);
      raft::copy(time_matrix_span, time_matrix, nlocations * nlocations, stream);
      if (limit_matrix_entries(time_matrix_span, nlocations, data_model.get_handle_ptr())) {
        std::cout << "\nMax time matrix value overriden to 1.0e+30";
      }
    }
  }
}

}  // namespace detail
}  // namespace routing
}  // namespace cuopt
