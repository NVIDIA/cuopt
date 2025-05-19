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
#include "macros.hpp"

#include <cstdint>
#include <memory>
#include <vector>

/* This is xoshiro256++ 1.0, one of our all-purpose, rock-solid generators.
   It has excellent (sub-ns) speed, a state (256 bits) that is large
   enough for any parallel application, and it passes all tests we are
   aware of.
   http://prng.di.unimi.it/xoshiro256plusplus.c
   For generating just floating-point numbers, xoshiro256+ is even faster.

   The state must be seeded so that it is not everywhere zero. If you have
   a 64-bit seed, we suggest to seed a splitmix64 generator and use its
   output to fill s. */

constexpr bool is_constraint(size_t index)
{
  if (index == DIST) return false;
  return true;
}

/*! Initializing xoshiro */
static thread_local uint64_t s_xoshiro[4]{
  0x9E3779B97F4A7C15LL, 0xBF58476D1CE4E5B9LL, 0x94D049BB133111EBLL, 0x16B6F9BB132A53FFLL};

/*! \brief { Generate rotations} */
static inline uint64_t rotl(const uint64_t x, int k) { return (x << k) | (x >> (64 - k)); }

/*! Get next random number */
uint64_t inline next_random(void)
{
  const uint64_t result = rotl(s_xoshiro[0] + s_xoshiro[3], 23) + s_xoshiro[0];

  const uint64_t t = s_xoshiro[1] << 17;

  s_xoshiro[2] ^= s_xoshiro[0];
  s_xoshiro[3] ^= s_xoshiro[1];
  s_xoshiro[1] ^= s_xoshiro[2];
  s_xoshiro[0] ^= s_xoshiro[3];

  s_xoshiro[2] ^= t;

  s_xoshiro[3] = rotl(s_xoshiro[3], 45);

  return result;
}

struct next_random_object {
  using result_type = uint64_t;
  static constexpr uint64_t min() { return 0; }
  static constexpr uint64_t max() { return UINT64_MAX; }
  uint64_t operator()() { return next_random(); }
};

// RAII method of restoring a buffer
struct file_buffer_t {
  FILE* file_ptr;
  file_buffer_t(std::string file_name)
  {
    if (file_name != "") {
      file_ptr = fopen(file_name.c_str(), "w");
    } else {
      file_ptr = stdout;
    }
  }

  ~file_buffer_t()
  {
    if (file_ptr != stdout) { fclose(file_ptr); }
  }
};

template <class T>
static inline bool find_and_pop(std::vector<T>& input, T element) noexcept
{
  auto it = std::find(input.begin(), input.end(), element);
  if (it == input.end()) return false;
  int rem_index    = it - input.begin();
  input[rem_index] = input.back();
  input.pop_back();
  return true;
}

template <class T>
static inline T pop_random(std::vector<T>& input) noexcept
{
  cuopt_assert(!input.empty(), "input can't be empty!");
  cuopt_expects(!input.empty(), cuopt::error_type_t::RuntimeError, "A runtime error occurred!");
  int index    = next_random() % input.size();
  T item       = input[index];
  input[index] = input.back();
  input.pop_back();
  return item;
}

template <class Solution>
bool check_if_routes_empty(const Solution& a)
{
  for (const auto& route : a.routes) {
    if (route.is_empty()) return true;
    cuopt_assert(!route.is_empty(), "route can't be empty");
  }
  return false;
}
