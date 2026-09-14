/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#pragma once
#include <bitset>
#include <cstdint>

namespace cuopt {

class splitmix64_t {
 public:
  static constexpr uint64_t default_seed   = 0xBAD0FF1CED15EA5EUL;
  static constexpr uint64_t default_stream = 0x9E3779B97F4A7C15UL;

  /// Creates a new instance of the SplitMix64 generator.
  ///
  /// @param[in]    seed    generator seed
  /// @param[in]    stream    generator increment
  splitmix64_t(uint64_t seed = default_seed, uint64_t stream = default_stream)
  {
    set_seed(seed, stream);
  }

  /// Creates a new instance of the SplitMix64 generator
  /// from an `other` generator.
  ///
  /// @param[in]    other    SplitMix64 generator to use as the data source
  splitmix64_t(const splitmix64_t& other) : state_(other.state_), stream_(other.stream_) {}

  /// Default destructor.
  ~splitmix64_t() = default;

  /// Reseeds the generator.
  ///
  /// @param[in]    seed    generator seed (optional)
  /// @param[in]    stream    generator increment (optional)
  constexpr void set_seed(uint64_t seed = default_seed, uint64_t stream = default_stream)
  {
    state_  = seed;
    stream_ = stream;
    next_state();
  }

  /// @returns the next uniformly distributed 32-bit unsigned integer.
  constexpr uint32_t next_u32() { return next_u64() >> 32; }

  /// @returns the next uniformly distributed non-negative 32-bit signed integer in [0,
  /// INT32_MAX].
  constexpr int32_t next_i32()
  {
    int32_t ret;
    uint32_t val;
    val = next_u32();
    ret = int32_t(val & 0x7fffffff);
    return ret;
  }

  ///@returns the next uniformly distributed 64-bit unsigned integer.
  constexpr uint64_t next_u64() { return mix64(next_state()); }

  /// @returns the next uniformly distributed non-negative 64-bit signed integer in [0,
  /// INT64_MAX].
  constexpr int64_t next_i64()
  {
    int64_t ret;
    uint64_t val;
    val = next_u64();
    ret = int64_t(val & 0x7fffffffffffffff);
    return ret;
  }

  /// @returns a uniformly distributed float in [0, 1).
  float next_float() { return (next_u32() >> 8) * 0x1.0p-24; }

  /// @returns a uniformly distributed double in [0, 1).
  double next_double() { return (next_u64() >> 11) * 0x1.0p-53; }

  /// "Splits" the generator, creating a new instance of SplitMix64 in the process.
  uint64_t generate_stream()
  {
    next_state();
    uint64_t new_stream = mix_stream(next_state());
    return new_stream;
  }

 private:
  uint64_t state_;   ///< internal state
  uint64_t stream_;  ///< increment

  /// "Mixes" (i.e., scramble and blend) the bits of a number `z`
  /// to generate a new random stream.
  ///
  /// @param[in]    z    any integer to use as the source
  /// @returns the random stream to use in another instance of SplitMix64
  static uint64_t mix_stream(uint64_t z)
  {
    z = (z ^ (z >> 33)) * 0xFF51AFD7ED558CCDUL;
    z = (z ^ (z >> 33)) * 0xC4CEB9FE1A85EC53UL;
    z ^= (z >> 33);
    z |= 1;
    std::bitset<64> b(z ^ (z >> 1));
    return b.count() < 24 ? z ^ 0xAAAAAAAAAAAAAAAA : z;
  }

  /// "Mixes" (i.e., scramble and blend) the bits of a number `z`
  /// to generate the next pseudorandom number in the sequence.
  ///
  /// @param[in]    z    any integer to use as the source
  /// @returns a pseudorandom number
  static constexpr uint64_t mix64(uint64_t z)
  {
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9UL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBUL;
    return z ^ (z >> 31);
  }

  /// Update the internal state and returns the old state.
  constexpr uint64_t next_state()
  {
    uint64_t oldstate = state_;
    state_ += stream_;
    return oldstate;
  }
};

}  // namespace cuopt
