/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#pragma once

#ifdef _OPENMP

#include <omp.h>
#include <memory>
#include <utility>

namespace cuopt {

// Wrapper of omp_lock_t. Optionally, you can provide a hint as defined in
// https://www.openmp.org/spec-html/5.1/openmpse39.html#x224-2570003.9
class omp_mutex_t {
 public:
  omp_mutex_t() : mutex(new omp_lock_t) { omp_init_lock(mutex.get()); }

  omp_mutex_t(const omp_mutex_t&) = delete;

  omp_mutex_t(omp_mutex_t&& other) { *this = std::move(other); }

  omp_mutex_t& operator=(const omp_mutex_t&) = delete;

  omp_mutex_t& operator=(omp_mutex_t&& other)
  {
    if (&other != this) {
      if (mutex) { omp_destroy_lock(mutex.get()); }
      mutex = std::move(other.mutex);
    }
    return *this;
  }

  virtual ~omp_mutex_t()
  {
    if (mutex) {
      omp_destroy_lock(mutex.get());
      mutex.reset();
    }
  }

  void lock() { omp_set_lock(mutex.get()); }

  void unlock() { omp_unset_lock(mutex.get()); }

  bool try_lock() { return omp_test_lock(mutex.get()); }

 private:
  std::unique_ptr<omp_lock_t> mutex;
};

// Empty class with the same methods as `omp_mutex_t`. This is mainly used for cleanly disabling
// the `omp_mutex_t` via type alias (`lock` and `unlock` are replaced by NOOPs).
class fake_omp_mutex_t {
 public:
  static void lock() {}
  static void unlock() {}
  static bool try_lock() { return true; }
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

  operator T() const { return load(); }
  T operator+=(T inc) { return fetch_add(inc) + inc; }
  T operator-=(T inc) { return fetch_sub(inc) - inc; }

  // In theory, this should be enabled only for integers,
  // but it works for any numerical types.
  T operator++() { return fetch_add(T(1)) + 1; }
  T operator++(int) { return fetch_add(T(1)); }
  T operator--() { return fetch_sub(T(1)) - 1; }
  T operator--(int) { return fetch_sub(T(1)); }

  // Possible values for memory order: relaxed, acquire, seq_cst
  T load(std::memory_order memory_order = std::memory_order::seq_cst) const
  {
    T res;
    if (memory_order == std::memory_order::relaxed) {
#pragma omp atomic read relaxed
      res = val;
    } else if (memory_order == std::memory_order::acquire) {
#pragma omp atomic read acquire
      res = val;
    } else {
#pragma omp atomic read
      res = val;
    }
    return res;
  }

  // Possible values for memory order: relaxed, release, seq_cst
  void store(T new_val, std::memory_order memory_order = std::memory_order::seq_cst)
  {
    if (memory_order == std::memory_order::relaxed) {
#pragma omp atomic write relaxed
      val = new_val;
    } else if (memory_order == std::memory_order::release) {
#pragma omp atomic write release
      val = new_val;
    } else {
#pragma omp atomic write
      val = new_val;
    }
  }

  T exchange(T other, std::memory_order memory_order = std::memory_order::seq_cst)
  {
    T old;
    if (memory_order == std::memory_order::relaxed) {
#pragma omp atomic capture relaxed
      {
        old = val;
        val = other;
      }
    } else if (memory_order == std::memory_order::acquire) {
#pragma omp atomic capture acquire
      {
        old = val;
        val = other;
      }
    } else if (memory_order == std::memory_order::release) {
#pragma omp atomic capture release
      {
        old = val;
        val = other;
      }
    } else if (memory_order == std::memory_order::acq_rel) {
#pragma omp atomic capture acq_rel
      {
        old = val;
        val = other;
      }
    } else {
#pragma omp atomic capture
      {
        old = val;
        val = other;
      }
    }
    return old;
  }

  T fetch_add(T inc, std::memory_order memory_order = std::memory_order::seq_cst)
  {
    T old;
    if (memory_order == std::memory_order::relaxed) {
#pragma omp atomic capture relaxed
      {
        old = val;
        val += inc;
      }
    } else if (memory_order == std::memory_order::acquire) {
#pragma omp atomic capture acquire
      {
        old = val;
        val += inc;
      }
    } else if (memory_order == std::memory_order::release) {
#pragma omp atomic capture release
      {
        old = val;
        val += inc;
      }
    } else if (memory_order == std::memory_order::acq_rel) {
#pragma omp atomic capture acq_rel
      {
        old = val;
        val += inc;
      }
    } else {
#pragma omp atomic capture
      {
        old = val;
        val += inc;
      }
    }
    return old;
  }

  T fetch_sub(T inc) { return fetch_add(-inc); }

  // Get the underlying value without atomics
  T& underlying() { return val; }
  T underlying() const { return val; }

 private:
  T val;

#ifndef __NVCC__
  friend double fetch_min(omp_atomic_t<double>& atomic_var, double other);
  friend double fetch_max(omp_atomic_t<double>& atomic_var, double other);
#endif
};

// Atomic CAS are only supported in OpenMP v5.1
// (gcc 12+ or clang 14+), however, nvcc (or the host compiler) cannot
// parse it correctly yet
#ifndef __NVCC__

// Free non-template functions are necessary because of a clang 20 bug
// when omp atomic compare is used within a templated context.
// see https://github.com/llvm/llvm-project/issues/127466
inline double fetch_min(omp_atomic_t<double>& atomic_var, double other)
{
  double old;
#pragma omp atomic compare capture
  {
    old = atomic_var.val;
    if (other < atomic_var.val) { atomic_var.val = other; }
  }
  return old;
}

inline double fetch_max(omp_atomic_t<double>& atomic_var, double other)
{
  double old;
#pragma omp atomic compare capture
  {
    old = atomic_var.val;
    if (other > atomic_var.val) { atomic_var.val = other; }
  }
  return old;
}
#endif

#endif

}  // namespace cuopt
