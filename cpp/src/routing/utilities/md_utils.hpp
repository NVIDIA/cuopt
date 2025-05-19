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

#include <cuopt/error.hpp>
#include <cuopt/routing/data_model_view.hpp>

#include <thrust/functional.h>
#include <thrust/logical.h>
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
    return get_cost_matrix(vehicle_type, 0);
  }

  constexpr auto get_time_matrix(uint8_t vehicle_type) const
  {
    return get_cost_matrix(vehicle_type, extent[1] - 1);
  }

  f_t const* buffer_ptr{nullptr};
  // dim4(n_vehicle_types, n_matrix_types, n_loc, n_loc)
  size_t extent[NCON_DIMS];
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

  constexpr auto get_cost_matrix(uint8_t vehicle_type) { return get_cost_matrix(vehicle_type, 0); }

  constexpr auto get_time_matrix(uint8_t vehicle_type)
  {
    return get_cost_matrix(vehicle_type, extent[1] - 1);
  }

  auto view() const
  {
    mdarray_view_t<f_t> view;
    view.buffer_ptr = buffer.data();
    for (size_t i = 0; i < NCON_DIMS; ++i) {
      view.extent[i] = extent[i];
    }
    return view;
  }
  size_t extent[NCON_DIMS];
  std::vector<f_t> buffer;
};

template <typename f_t, size_t NCON_DIMS = 4>
struct d_mdarray_t {
  d_mdarray_t(rmm::cuda_stream_view stream_) : buffer(0, stream_), stream(stream_) {}
  d_mdarray_t(std::vector<size_t> const& extent_, rmm::cuda_stream_view stream_)
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

  constexpr auto get_cost_matrix(uint8_t vehicle_type) { return get_cost_matrix(vehicle_type, 0); }

  constexpr auto get_time_matrix(uint8_t vehicle_type)
  {
    return get_cost_matrix(vehicle_type, extent[1] - 1);
  }

  auto view() const
  {
    mdarray_view_t<f_t> view;
    view.buffer_ptr = buffer.data();
    for (size_t i = 0; i < NCON_DIMS; ++i) {
      view.extent[i] = extent[i];
    }
    return view;
  }

  size_t extent[NCON_DIMS];
  rmm::device_uvector<f_t> buffer;
  rmm::cuda_stream_view stream;
};

namespace detail {

template <typename i_t, typename f_t>
bool limit_matrix_entries(f_t* matrix, i_t width, raft::handle_t const* handle_ptr)
{
  i_t mat_size  = width * width;
  f_t max_value = 1.0e+30;

  bool exceeds_max = thrust::any_of(
    handle_ptr->get_thrust_policy(), matrix, matrix + mat_size, [max_value] __device__(auto& x) {
      return x > max_value;
    });

  if (exceeds_max) {
    thrust::transform(handle_ptr->get_thrust_policy(),
                      matrix,
                      matrix + mat_size,
                      matrix,
                      [max_value] __device__(auto& x) { return x > max_value ? max_value : x; });
  }
  return exceeds_max;
}

template <typename i_t, typename f_t>
void fill_data_model_matrices(data_model_view_t<i_t, f_t>& data_model, d_mdarray_t<f_t>& matrices)

{
  auto stream         = data_model.get_handle_ptr()->get_stream();
  i_t n_vehicle_types = matrices.extent[0];
  i_t n_matrix_types  = matrices.extent[1];
  for (auto vehicle_type = 0; vehicle_type < n_vehicle_types; ++vehicle_type) {
    for (auto matrix_type = 0; matrix_type < n_matrix_types; ++matrix_type) {
      auto const matrix = matrices.get_cost_matrix(vehicle_type, matrix_type);
      if (matrix_type == 0)
        data_model.add_cost_matrix(matrix, vehicle_type);
      else
        data_model.add_transit_time_matrix(matrix, vehicle_type);
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
                           rmm::cuda_stream_view stream)
{
  std::vector<size_t> full_matrix_extent{n_vehicle_types, n_matrix_types, nlocations, nlocations};
  d_mdarray_t<f_t> matrices{full_matrix_extent, stream};
  return matrices;
}

inline auto get_unique_vehicle_types(const raft::device_span<uint8_t const>& vehicle_types,
                                     rmm::cuda_stream_view stream)
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
auto get_cost_matrix_type_dim(data_model_view_t<i_t, f_t> const& data_model)
{
  auto n_matrix_types    = 1;
  auto vehicle_types_map = get_unique_vehicle_types(data_model.get_vehicle_types(),
                                                    data_model.get_handle_ptr()->get_stream());
  for (auto& [old_type, new_type] : vehicle_types_map) {
    if (data_model.get_transit_time_matrix(old_type)) {
      ++n_matrix_types;
      break;
    }
  }
  return n_matrix_types;
}

template <typename i_t, typename f_t>
std::tuple<float const*, float const*> get_vehicle_matrices(
  data_model_view_t<i_t, f_t> const& data_model, uint8_t vehicle_type)
{
  auto cost_matrix = data_model.get_cost_matrix(vehicle_type);
  if (!cost_matrix)
    cuopt_expects(
      false, error_type_t::ValidationError, "Set vehicle types when using multiple matrices");
  auto time_matrix = data_model.get_transit_time_matrix(vehicle_type);
  if (!time_matrix) time_matrix = cost_matrix;
  return std::make_tuple(cost_matrix, time_matrix);
}

template <typename i_t, typename f_t>
void fill_mdarray_from_data_model(d_mdarray_t<f_t>& matrices,
                                  data_model_view_t<i_t, f_t> const& data_model)
{
  auto stream            = data_model.get_handle_ptr()->get_stream();
  auto vehicle_types     = data_model.get_vehicle_types();
  auto nlocations        = data_model.get_num_locations();
  auto vehicle_types_map = get_unique_vehicle_types(vehicle_types, stream);

  for (auto& [old_type, new_type] : vehicle_types_map) {
    auto [cost_matrix, time_matrix] = get_vehicle_matrices<i_t, f_t>(data_model, old_type);
    auto cost_matrix_span           = matrices.get_cost_matrix(new_type);
    auto time_matrix_span           = matrices.get_time_matrix(new_type);
    raft::copy(cost_matrix_span, cost_matrix, nlocations * nlocations, stream);

    if (limit_matrix_entries(cost_matrix_span, nlocations, data_model.get_handle_ptr())) {
      std::cout << "\nMax cost matrix value overriden to 1.0e+30";
    }

    raft::copy(time_matrix_span, time_matrix, nlocations * nlocations, stream);
    if (limit_matrix_entries(time_matrix_span, nlocations, data_model.get_handle_ptr())) {
      std::cout << "\nMax time matrix value overriden to 1.0e+30";
    }
  }
}

}  // namespace detail
}  // namespace routing
}  // namespace cuopt
