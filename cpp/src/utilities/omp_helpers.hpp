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

#ifdef _OPENMP

#include <omp.h>
#include <type_traits>

namespace cuopt {

// Wrapper of omp_lock_t. Optionally, you can provide a hint as defined in
// https://www.openmp.org/spec-html/5.1/openmpse39.html#x224-2570003.9
class omp_mutex_t {
 public:
  omp_mutex_t() { omp_init_lock(&mutex); }
  virtual ~omp_mutex_t() { omp_destroy_lock(&mutex); }
  void lock() { omp_set_lock(&mutex); }
  void unlock() { omp_unset_lock(&mutex); }
  bool try_lock() { return omp_test_lock(&mutex); }

 private:
  omp_lock_t mutex;
};

// Wrapper for omp atomic operations. See
// https://www.openmp.org/spec-html/5.1/openmpsu105.html.
template <typename T>
class omp_atomic_t {
 public:
  omp_atomic_t() = default;
  omp_atomic_t(T val) : val(val) {}

  T operator=(T new_val)
  {
    store(new_val);
    return new_val;
  }

  operator T() { return load(); }
  T operator+=(T inc) { return fetch_add(inc) + inc; }
  T operator-=(T inc) { return fetch_sub(inc) - inc; }

  // In theory, this should be enabled only for integers,
  // but it works for any numerical types.
  T operator++() { return fetch_add(T(1)) + 1; }
  T operator++(int) { return fetch_add(T(1)); }
  T operator--() { return fetch_sub(T(1)) - 1; }
  T operator--(int) { return fetch_sub(T(1)); }

  T load()
  {
    T res;
#pragma omp atomic read
    res = val;
    return res;
  }

  void store(T new_val)
  {
#pragma omp atomic write
    val = new_val;
  }

  T exchange(T other)
  {
    T old;
#pragma omp atomic capture
    {
      old = val;
      val = other;
    }
    return old;
  }

  T fetch_add(T inc)
  {
    T old;
#pragma omp atomic capture
    {
      old = val;
      val += inc;
    }
    return old;
  }

  T fetch_sub(T inc) { return fetch_add(-inc); }

// Atomic CAS are only supported in OpenMP v5.1
// (gcc 12+ or clang 14+), however, nvcc (or the host compiler) cannot
// parse it correctly yet
#ifndef __NVCC__

  T fetch_min(T other)
  {
    T old;
#pragma omp atomic compare capture
    {
      old = val;
      val = other < val ? other : val;
    }
    return old;
  }

  T fetch_max(T other)
  {
    T old;
#pragma omp atomic compare capture
    {
      old = val;
      val = other > val ? other : val;
    }
    return old;
  }
#endif

 private:
  T val;
};

#endif

}  // namespace cuopt
