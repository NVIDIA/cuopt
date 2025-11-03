/*
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights
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

#include <stdint.h>

// Copied from raft/PCGenerator (rng_device.cuh).
// It is based on the PCG code (https://www.pcg-random.org/).
namespace cuopt {
class PCG {
 public:
  static constexpr uint64_t default_seed   = 0x853c49e6748fea9bULL;
  static constexpr uint64_t default_stream = 0xda3e39cb94b95bdbULL;

  /**
   * @brief ctor. Initializes the PCG
   * @param rng_state is the generator state used for initializing the generator
   * @param subsequence specifies the subsequence to be generated out of 2^64 possible subsequences
   * In a parallel setting, like threads of a CUDA kernel, each thread is required to generate a
   * unique set of random numbers. This can be achieved by initializing the generator with same
   * rng_state for all the threads and diststreamt values for subsequence.
   */
  PCG(const uint64_t seed = default_seed, const uint64_t subsequence = 0)
  {
    _init_pcg(seed, default_stream + subsequence, subsequence);
  }

  /**
   * @brief ctor. This is lower level constructor for PCG
   * This code is derived from PCG basic code
   * @param seed A 64-bit seed for the generator
   * @param subsequence The id of subsequence that should be generated [0, 2^64-1]
   * @param offset Initial `offset` number of items are skipped from the subsequence
   */
  PCG(uint64_t seed, uint64_t subsequence, uint64_t offset)
  {
    _init_pcg(seed, subsequence, offset);
  }

  // Based on "Random Number Generation with Arbitrary Strides" F. B. Brown
  // Link https://mcnp.lanl.gov/pdf_files/anl-rn-arb-stride.pdf
  void skipahead(uint64_t offset)
  {
    uint64_t G = 1;
    uint64_t h = 6364136223846793005ULL;
    uint64_t C = 0;
    uint64_t f = stream;
    while (offset) {
      if (offset & 1) {
        G = G * h;
        C = C * h + f;
      }
      f = f * (h + 1);
      h = h * h;
      offset >>= 1;
    }
    state = state * G + C;
  }

  /**
   * @defgroup NextRand Generate the next random number
   * @brief This code is derived from PCG basic code
   * @{
   */
  uint32_t next_u32()
  {
    uint32_t ret;
    uint64_t oldstate   = state;
    state               = oldstate * 6364136223846793005ULL + stream;
    uint32_t xorshifted = ((oldstate >> 18u) ^ oldstate) >> 27u;
    uint32_t rot        = oldstate >> 59u;
    ret                 = (xorshifted >> rot) | (xorshifted << ((-rot) & 31));
    return ret;
  }

  uint64_t next_u64()
  {
    uint64_t ret;
    uint32_t a, b;
    a   = next_u32();
    b   = next_u32();
    ret = uint64_t(a) | (uint64_t(b) << 32);
    return ret;
  }

  int32_t next_i32()
  {
    int32_t ret;
    uint32_t val;
    val = next_u32();
    ret = int32_t(val & 0x7fffffff);
    return ret;
  }

  int64_t next_i64()
  {
    int64_t ret;
    uint64_t val;
    val = next_u64();
    ret = int64_t(val & 0x7fffffffffffffff);
    return ret;
  }

  float next_float() { return static_cast<float>((next_u32() >> 8) * 0x1.0p-24); }

  double next_double() { return static_cast<double>((next_u64() >> 11) * 0x1.0p-53); }

  template <typename T>
  T next()
  {
    T val;
    next(val);
    return val;
  }

  void next(uint32_t& ret) { ret = next_u32(); }
  void next(uint64_t& ret) { ret = next_u64(); }
  void next(int32_t& ret) { ret = next_i32(); }
  void next(int64_t& ret) { ret = next_i64(); }
  void next(float& ret) { ret = next_float(); }
  void next(double& ret) { ret = next_double(); }

  // Generate a random integer uniformly distributed in [low, high].
  // FIXME: When C++20 is enabled, switch to `std::integer`.
  template <typename i_t>
  i_t uniform(i_t low, i_t high)
  {
    // Fractional scaling may exhibit slightly bias, but should be
    // fine for our use case.
    double val = next_double();
    i_t dist   = high - low;
    return low + static_cast<i_t>(val * dist);
  }

 private:
  uint64_t state;
  uint64_t stream;

  void _init_pcg(uint64_t seed, uint64_t subsequence, uint64_t offset)
  {
    state  = uint64_t(0);
    stream = (subsequence << 1u) | 1u;
    uint32_t discard;
    next(discard);
    state += seed;
    next(discard);
    skipahead(offset);
  }
};
}  // namespace cuopt
