/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#pragma once

#include <stdint.h>
#include <utility>
#include <vector>

// Copied from raft/PCGenerator (rng_device.cuh). We cannot use raft implementation
// on the CPU (.cpp file) since the raft header includes CUDA code.
// The original code is from https://www.pcg-random.org/.
namespace cuopt {
class pcgenerator_t {
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
  pcgenerator_t(const uint64_t seed        = default_seed,
                const uint64_t subsequence = default_stream,
                uint64_t offset            = 0)
  {
    set_seed(seed, subsequence, offset);
  }

  // Set the seed, subsequence and offset of the PCG
  void set_seed(uint64_t seed, const uint64_t subsequence = default_stream, uint64_t offset = 0)
  {
    state  = uint64_t(0);
    stream = (subsequence << 1u) | 1u;
    uint32_t discard;
    next(discard);
    state += seed;
    next(discard);
    skipahead(offset);
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

  /// Draws a sample from a uniform distribution. The samples are uniformly distributed over
  /// the semi-closed interval `[low, high)`. This routine may have a **slight bias** toward
  /// some numbers in the range (scaling by floating-point).
  template <typename T>
  T uniform(T low, T high)
  {
    double val = next_double();
    T range    = high - low;
    return low + (val * range);
  }

  // Shuffles the contents of a sequence using the Fisher–Yates algorithm.
  template <typename T>
  void shuffle(std::vector<T>& seq)
  {
    if (seq.empty()) { return; }
    for (size_t i = 0; i < seq.size() - 1; ++i) {
      size_t j = uniform(i, seq.size());
      if (j != i) std::swap(seq[i], seq[j]);
    }
  }

 private:
  uint64_t state;
  uint64_t stream;
};
}  // namespace cuopt
