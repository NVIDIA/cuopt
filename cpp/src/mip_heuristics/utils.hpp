/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#pragma once

#include <thrust/iterator/permutation_iterator.h>
#include <utilities/macros.cuh>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <type_traits>
#include <vector>

#ifdef __CUDACC__
#define CUOPT_MIP_HOST_DEVICE inline __host__ __device__
#else
#define CUOPT_MIP_HOST_DEVICE inline
#endif

namespace cuopt::mathematical_optimization::mip {

// checks if a given float value can be exactly represented as an integer of type int_t.
template <typename int_t, typename f_t>
inline bool is_exactly_representable(f_t value)
{
  static_assert(std::is_integral_v<int_t>);
  static_assert(std::is_floating_point_v<f_t>);

  if constexpr (std::numeric_limits<f_t>::digits < std::numeric_limits<int_t>::digits) {
    return false;
  } else {
    return std::isfinite(value) && std::trunc(value) == value &&
           value >= (f_t)std::numeric_limits<int_t>::lowest() &&
           value <= (f_t)std::numeric_limits<int_t>::max();
  }
}

// Ogita-Rump-Oishi Dot2. TwoProduct recovers the rounding of each coefficient-value product, which
// a compensated summation over already-multiplied terms cannot see.
// https://epubs.siam.org/doi/abs/10.1137/030601818
template <typename UIt, typename VIt>
__attribute__((optimize("no-fast-math"))) CUOPT_MIP_HOST_DEVICE auto compensated_dot2(
  UIt u_it, VIt v_it, std::size_t n, decltype(u_it[0] * v_it[0]) init = 0)
{
  using f_t = decltype(u_it[0] * v_it[0]);
  f_t p     = init;
  f_t s     = 0;
  for (std::size_t k = 0; k < n; ++k) {
    const f_t u = u_it[k];
    const f_t v = v_it[k];
    const f_t h = u * v;
    const f_t t = p + h;
    const f_t z = t - p;
    s += ((p - (t - z)) + (h - z)) + fma(u, v, -h);
    p = t;
  }
  return p + s;
}

template <typename CoeffIt, typename IndexIt, typename ValueIt>
CUOPT_MIP_HOST_DEVICE auto compensated_dot2_csr(CoeffIt coefficients,
                                                IndexIt columns,
                                                ValueIt values,
                                                size_t nnz,
                                                decltype(coefficients[0] * values[0]) init = 0)
{
  return compensated_dot2(
    coefficients, thrust::make_permutation_iterator(values, columns), nnz, init);
}

template <typename OffsetIt, typename IndexIt, typename CoeffIt, typename ValueIt, typename i_t>
CUOPT_MIP_HOST_DEVICE auto compensated_dot2_csr(OffsetIt offsets,
                                                IndexIt columns,
                                                CoeffIt coefficients,
                                                ValueIt values,
                                                i_t row,
                                                decltype(coefficients[0] * values[0]) init = 0)
{
  const auto begin = offsets[row];
  const auto end   = offsets[row + 1];
  return compensated_dot2_csr(coefficients + begin, columns + begin, values, end - begin, init);
}

template <typename CsrLike, typename Values, typename i_t>
inline auto compensated_dot2_csr(const CsrLike& csr, const Values& values, i_t row)
{
  return compensated_dot2_csr(
    csr.offsets.data(), csr.variables.data(), csr.coefficients.data(), values.data(), row);
}

template <typename i_t>
struct host_contiguous_set_t {
  void resize(i_t max_size)
  {
    cuopt_assert(max_size >= 0, "invalid max size");
    contents.clear();
    contents.reserve(max_size);
    index_map.assign(max_size, -1);
    is_member.assign(max_size, 0);
  }

  void clear()
  {
    for (i_t val : contents) {
      index_map[val] = -1;
      is_member[val] = 0;
    }
    contents.clear();
  }

  void insert(i_t val)
  {
    cuopt_assert(val >= 0 && val < max_size(), "Value is out of bounds");
    cuopt_assert(!contains(val), "Value already exists");
    index_map[val] = contents.size();
    is_member[val] = 1;
    contents.push_back(val);
  }

  void remove(i_t val)
  {
    cuopt_assert(val >= 0 && val < max_size(), "Value is out of bounds");
    cuopt_assert(contains(val), "Value not found");
    const i_t idx       = index_map[val];
    const i_t last_val  = contents.back();
    contents[idx]       = last_val;
    index_map[last_val] = idx;
    contents.pop_back();
    index_map[val] = -1;
    is_member[val] = 0;
  }

  bool contains(i_t val) const
  {
    cuopt_assert(val >= 0 && val < max_size(), "Value is out of bounds");
    return is_member[val] != 0;
  }

  auto begin() const { return contents.begin(); }
  auto end() const { return contents.end(); }
  i_t size() const { return contents.size(); }
  i_t max_size() const { return index_map.size(); }
  bool empty() const { return contents.empty(); }

  std::vector<i_t> contents;
  std::vector<i_t> index_map;
  std::vector<uint8_t> is_member;
};

}  // namespace cuopt::mathematical_optimization::mip

#undef CUOPT_MIP_HOST_DEVICE
