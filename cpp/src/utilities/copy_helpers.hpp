/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2025 NVIDIA CORPORATION &
 * AFFILIATES. All rights reserved. SPDX-License-Identifier: Apache-2.0
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

#include <raft/core/device_span.hpp>
#include <raft/core/handle.hpp>
#include <raft/util/cudart_utils.hpp>

#include <rmm/device_uvector.hpp>
#include <rmm/exec_policy.hpp>

#include <thrust/transform.h>
#include <cuda/std/functional>

namespace cuopt {

template <typename T>
struct type_2 {
  using type = void;
};

template <>
struct type_2<int> {
  using type = int2;
};

template <>
struct type_2<float> {
  using type = float2;
};

template <>
struct type_2<double> {
  using type = double2;
};

template <typename T>
struct scalar_type {
  using type = void;
};

template <>
struct scalar_type<int2> {
  using type = int;
};

template <>
struct scalar_type<float2> {
  using type = float;
};

template <>
struct scalar_type<double2> {
  using type = double;
};

template <>
struct scalar_type<const int2> {
  using type = const int;
};

template <>
struct scalar_type<const float2> {
  using type = const float;
};

template <>
struct scalar_type<const double2> {
  using type = const double;
};

template <typename T>
raft::device_span<typename type_2<T>::type> make_span_2(rmm::device_uvector<T>& container)
{
  using T2 = typename type_2<T>::type;
  static_assert(sizeof(T2) == 2 * sizeof(T));
  return raft::device_span<T2>(reinterpret_cast<T2*>(container.data()),
                               sizeof(T) * container.size() / sizeof(T2));
}

template <typename T>
raft::device_span<const typename type_2<T>::type> make_span_2(
  rmm::device_uvector<T> const& container)
{
  using T2 = typename type_2<T>::type;
  static_assert(sizeof(T2) == 2 * sizeof(T));
  return raft::device_span<const T2>(reinterpret_cast<const T2*>(container.data()),
                                     sizeof(T) * container.size() / sizeof(T2));
}

template <typename f_t2>
__host__ __device__ inline typename scalar_type<f_t2>::type& get_lower(f_t2& val)
{
  return val.x;
}

template <typename f_t2>
__host__ __device__ inline typename scalar_type<f_t2>::type& get_upper(f_t2& val)
{
  return val.y;
}

/**
 * @brief Simple utility function to copy device ptr to host
 *
 * @tparam T
 * @param device_ptr
 * @param size
 * @param stream_view
 * @return auto
 */
template <typename T>
auto host_copy(T const* device_ptr, size_t size, rmm::cuda_stream_view stream_view)
{
  if (!device_ptr) return std::vector<T>{};
  std::vector<T> host_vec(size);
  raft::copy(host_vec.data(), device_ptr, size, stream_view);
  stream_view.synchronize();
  return host_vec;
}

/**
 * @brief Simple utility function to copy bool device ptr to host
 *
 * @tparam T
 * @param[in] device_ptr
 * @param[in] size
 * @param[in] stream_view
 * @return auto
 */
inline auto host_copy(bool const* device_ptr, size_t size, rmm::cuda_stream_view stream_view)
{
  if (!device_ptr) { return std::vector<bool>(0); }
  rmm::device_uvector<int> d_int_vec(size, stream_view);
  d_int_vec.resize(size, stream_view);
  cuda::std::identity id;
  thrust::transform(
    rmm::exec_policy(stream_view), device_ptr, device_ptr + size, d_int_vec.begin(), id);
  auto h_int_vec = host_copy(d_int_vec.data(), d_int_vec.size(), stream_view);
  std::vector<bool> h_bool_vec(h_int_vec.size());
  for (size_t i = 0; i < h_int_vec.size(); ++i) {
    h_bool_vec[i] = static_cast<bool>(h_int_vec[i]);
  }
  stream_view.synchronize();
  return h_bool_vec;
}

/**
 * @brief Simple utility function to copy device_uvector to host
 *
 * @tparam T
 * @param device_vec
 * @param stream_view
 * @return auto
 */
template <typename T>
auto host_copy(rmm::device_uvector<T> const& device_vec)
{
  return host_copy(device_vec.data(), device_vec.size(), device_vec.stream());
}

/**
 * @brief Simple utility function to copy device_uvector to host
 *
 * @tparam T
 * @param device_vec
 * @param stream_view
 * @return auto
 */
template <typename T, typename Allocator>
auto host_copy(rmm::device_uvector<T> const& device_vec)
{
  std::vector<T, Allocator> host_vec(device_vec.size());
  raft::copy(host_vec.data(), device_vec.data(), device_vec.size(), device_vec.stream());
  device_vec.stream().synchronize();
  return host_vec;
}

/**
 * @brief Simple utility function to copy device span to host
 *
 * @tparam T
 * @param device_vec
 * @param stream_view
 * @return auto
 */
template <typename T>
auto host_copy(raft::device_span<T> const& device_vec, rmm::cuda_stream_view stream_view)
{
  return host_copy(device_vec.data(), device_vec.size(), stream_view);
}

/**
 * @brief Simple utility function to copy device vector to host
 *
 * @tparam T
 * @param device_vec
 * @param stream_view
 * @return auto
 */
template <typename T>
auto host_copy(rmm::device_uvector<T> const& device_vec, rmm::cuda_stream_view stream_view)
{
  return host_copy(device_vec.data(), device_vec.size(), stream_view);
}

/**
 * @brief Simple utility function to copy std::vector to device
 *
 * @tparam T
 * @param[in] device_vec
 * @param[in] stream_view
 * @return device_vec
 */
template <typename T>
inline rmm::device_uvector<T> device_copy(rmm::device_uvector<T> const& device_vec,
                                          rmm::cuda_stream_view stream_view)
{
  rmm::device_uvector<T> device_vec_copy(device_vec.size(), stream_view);
  raft::copy(device_vec_copy.data(), device_vec.data(), device_vec.size(), stream_view);
  return device_vec_copy;
}

/**
 * @brief Simple utility function to copy std::vector to device
 *
 * @tparam T
 * @param[in] host_vec
 * @param[in] stream_view
 * @return device_vec
 */
template <typename T>
inline void device_copy(rmm::device_uvector<T>& device_vec,
                        std::vector<T> const& host_vec,
                        rmm::cuda_stream_view stream_view)
{
  device_vec.resize(host_vec.size(), stream_view);
  raft::copy(device_vec.data(), host_vec.data(), host_vec.size(), stream_view);
}

/**
 * @brief Simple utility function to copy std::vector to device
 *
 * @tparam T
 * @param[in] host_vec
 * @param[in] stream_view
 * @return device_vec
 */
template <typename T, typename Allocator>
inline auto device_copy(std::vector<T, Allocator> const& host_vec,
                        rmm::cuda_stream_view stream_view)
{
  rmm::device_uvector<T> device_vec(host_vec.size(), stream_view);
  raft::copy(device_vec.data(), host_vec.data(), host_vec.size(), stream_view);
  return device_vec;
}

/**
 * @brief template specialization for boolean vector
 *
 * @param[in] host_vec
 * @param[in] stream_view
 * @return device_vec
 */
inline auto device_copy(std::vector<bool> const& host_vec, rmm::cuda_stream_view stream_view)
{
  std::vector<uint8_t> host_vec_int(host_vec.size());
  for (size_t i = 0; i < host_vec.size(); ++i) {
    host_vec_int[i] = host_vec[i];
  }
  auto device_vec_int = device_copy(host_vec_int, stream_view);

  rmm::device_uvector<bool> device_vec(host_vec.size(), stream_view);

  thrust::transform(rmm::exec_policy(stream_view),
                    device_vec_int.begin(),
                    device_vec_int.end(),
                    device_vec.begin(),
                    cuda::std::identity());

  return device_vec;
}

template <typename T>
void print(std::string_view const name, std::vector<T> const& container)
{
  std::cout << name << "=[";
  for (auto const& item : container) {
    std::cout << item << ",";
  }
  std::cout << "]\n";
}

template <typename T>
void print(std::string_view const name, rmm::device_uvector<T> const& container)
{
  raft::print_device_vector(name.data(), container.data(), container.size(), std::cout);
}

template <typename T>
raft::device_span<T> make_span(rmm::device_uvector<T>& container,
                               typename rmm::device_uvector<T>::size_type beg,
                               typename rmm::device_uvector<T>::size_type end)
{
  return raft::device_span<T>(container.data() + beg, end - beg);
}

template <typename T>
raft::device_span<const T> make_span(rmm::device_uvector<T> const& container,
                                     typename rmm::device_uvector<T>::size_type beg,
                                     typename rmm::device_uvector<T>::size_type end)
{
  return raft::device_span<const T>(container.data() + beg, end - beg);
}

template <typename T>
raft::device_span<T> make_span(rmm::device_uvector<T>& container)
{
  return raft::device_span<T>(container.data(), container.size());
}

template <typename T>
raft::device_span<const T> make_span(rmm::device_uvector<T> const& container)
{
  return raft::device_span<const T>(container.data(), container.size());
}

// resizes the device vector if it the std vector is larger
template <typename T>
inline void expand_device_copy(rmm::device_uvector<T>& device_vec,
                               std::vector<T> const& host_vec,
                               rmm::cuda_stream_view stream_view)
{
  if (host_vec.size() > device_vec.size()) { device_vec.resize(host_vec.size(), stream_view); }
  raft::copy(device_vec.data(), host_vec.data(), host_vec.size(), stream_view);
}

template <typename T>
inline void expand_device_copy(rmm::device_uvector<T>& dst_vec,
                               rmm::device_uvector<T> const& src_vec,
                               rmm::cuda_stream_view stream_view)
{
  if (src_vec.size() > dst_vec.size()) { dst_vec.resize(src_vec.size(), stream_view); }
  raft::copy(dst_vec.data(), src_vec.data(), src_vec.size(), stream_view);
}

template <typename f_t, typename f_t2>
std::tuple<std::vector<f_t>, std::vector<f_t>> extract_host_bounds(
  const rmm::device_uvector<f_t2>& variable_bounds, const raft::handle_t* handle_ptr)
{
  rmm::device_uvector<f_t> var_lb(variable_bounds.size(), handle_ptr->get_stream());
  rmm::device_uvector<f_t> var_ub(variable_bounds.size(), handle_ptr->get_stream());
  thrust::transform(
    handle_ptr->get_thrust_policy(),
    variable_bounds.begin(),
    variable_bounds.end(),
    thrust::make_zip_iterator(thrust::make_tuple(var_lb.begin(), var_ub.begin())),
    [] __device__(auto i) { return thrust::make_tuple(get_lower(i), get_upper(i)); });
  handle_ptr->sync_stream();
  auto h_var_lb = cuopt::host_copy(var_lb);
  auto h_var_ub = cuopt::host_copy(var_ub);
  return std::make_tuple(h_var_lb, h_var_ub);
}

}  // namespace cuopt
