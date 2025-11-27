/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#pragma once

#include <cstring>

namespace cuopt {
namespace routing {
namespace detail {

using word_t                = uint64_t;
constexpr size_t word_bytes = sizeof(word_t);
constexpr size_t word_bits  = word_bytes * 8;

// Makes bitset constexpr. Remove this when C++23 is enabled.
template <size_t size>
struct device_bitset_t {
  HDI device_bitset_t() : array{0} {}
  HDI device_bitset_t<size>& set(size_t pos, bool val = true)
  {
    if (val) {
      array[pos / word_bits] |= ((static_cast<word_t>(1)) << (pos % word_bits));
    } else {
      array[pos / word_bits] &= ~((static_cast<word_t>(1)) << (pos % word_bits));
    }
    return *this;
  }

  __host__ std::string to_string() const
  {
    auto word_bits = sizeof(array) * 8;
    std::string out(word_bits, '0');
    for (size_t i = word_bits; i > 0; --i) {
      if (test(i - 1)) { out[word_bits - i] = '1'; }
    }
    return out;
  }

  HDI void print() const
  {
    auto word_bits = sizeof(array) * 8;
    for (size_t i = word_bits; i > 0; --i) {
      if (test(i - 1)) {
        printf("1");
      } else {
        printf("0");
      }
    }
    printf("\n");
  }

  HDI bool test(size_t pos) const
  {
    return ((array[pos / word_bits] & (static_cast<word_t>(1) << (pos % word_bits))) !=
            static_cast<word_t>(0));
  }

  HDI device_bitset_t<size>& reset()
  {
    for (size_t i = 0; i < sizeof(array) / word_bytes; ++i)
      array[i] = static_cast<word_t>(0);
    return *this;
  }

  HDI bool operator[](size_t pos) const { return this->test(pos); }

  HDI device_bitset_t<size>& operator|=(device_bitset_t<size> const& rhs) noexcept
  {
    for (size_t i = 0; i < sizeof(array) / word_bytes; ++i)
      array[i] |= rhs.array[i];
    return *this;
  }

  HDI bool operator==(device_bitset_t<size> const& rhs) const noexcept
  {
    for (size_t i = 0; i < sizeof(array) / word_bytes; ++i)
      if (array[i] != rhs.array[i]) return false;
    return true;
  }

  HDI device_bitset_t<size> operator&(device_bitset_t<size> const& rhs) const noexcept
  {
    device_bitset_t<size> return_val = *this;
    for (size_t i = 0; i < sizeof(rhs.array) / word_bytes; ++i)
      return_val.array[i] = return_val.array[i] & rhs.array[i];
    return return_val;
  }

  HDI device_bitset_t<size> operator|(device_bitset_t<size> const& rhs) const noexcept
  {
    device_bitset_t<size> return_val = *this;
    for (size_t i = 0; i < sizeof(rhs.array) / word_bytes; ++i)
      return_val.array[i] = return_val.array[i] | rhs.array[i];
    return return_val;
  }

  HDI device_bitset_t<size> operator^(device_bitset_t<size> const& rhs) const noexcept
  {
    device_bitset_t<size> return_val = *this;
    for (size_t i = 0; i < sizeof(rhs.array) / word_bytes; ++i)
      return_val.array[i] = return_val.array[i] ^ rhs.array[i];
    return return_val;
  }

  word_t array[size / word_bits];
};

template <size_t size>
HDI device_bitset_t<size> operator~(device_bitset_t<size> const& first) noexcept
{
  device_bitset_t<size> return_val = first;
  for (size_t i = 0; i < sizeof(return_val.array) / word_bytes; ++i)
    return_val.array[i] = ~return_val.array[i];
  return return_val;
}

}  // namespace detail
}  // namespace routing
}  // namespace cuopt
