/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#pragma once

#include <cmath>

namespace cuopt {

// Split `total` entries across `n` blocks and return the [start, end) range for `k`-th block.
// Useful for splitting work across OpenMP tasks/threads/workers.
template <typename i_t>
std::pair<i_t, i_t> calculate_index_range(i_t k, double total, double n)
{
  i_t start = std::floor(k * total / n);
  i_t end   = std::floor((k + 1) * total / n);
  return {start, end};
}

}  // namespace cuopt

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
    store(new_val, std::memory_order::relaxed);
    return new_val;
  }

  operator T() const { return load(std::memory_order::relaxed); }
  T operator+=(T inc) { return fetch_add(inc, std::memory_order::relaxed) + inc; }
  T operator-=(T inc) { return fetch_sub(inc, std::memory_order::relaxed) - inc; }

  // In theory, this should be enabled only for integers,
  // but it works for any numerical types.
  T operator++() { return fetch_add(T(1), std::memory_order::relaxed) + 1; }
  T operator++(int) { return fetch_add(T(1), std::memory_order::relaxed); }
  T operator--() { return fetch_sub(T(1), std::memory_order::relaxed) - 1; }
  T operator--(int) { return fetch_sub(T(1), std::memory_order::relaxed); }

  // Possible values for memory order: relaxed, acquire, seq_cst
  T load(std::memory_order memory_order) const
  {
    T res;
    if (memory_order == std::memory_order::seq_cst) {
#pragma omp atomic read seq_cst
      res = val;
    } else if (memory_order == std::memory_order::acquire) {
#pragma omp atomic read acquire
      res = val;
    } else {
#pragma omp atomic read relaxed
      res = val;
    }
    return res;
  }

  // Possible values for memory order: relaxed, release, seq_cst
  void store(T new_val, std::memory_order memory_order)
  {
    if (memory_order == std::memory_order::seq_cst) {
#pragma omp atomic write seq_cst
      val = new_val;
    } else if (memory_order == std::memory_order::release) {
#pragma omp atomic write release
      val = new_val;
    } else {
#pragma omp atomic write relaxed
      val = new_val;
    }
  }

  T exchange(T other, std::memory_order memory_order)
  {
    T old;
    if (memory_order == std::memory_order::seq_cst) {
#pragma omp atomic capture seq_cst
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
#pragma omp atomic capture relaxed
      {
        old = val;
        val = other;
      }
    }
    return old;
  }

  T fetch_add(T inc, std::memory_order memory_order)
  {
    T old;
    if (memory_order == std::memory_order::seq_cst) {
#pragma omp atomic capture seq_cst
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
#pragma omp atomic capture relaxed
      {
        old = val;
        val += inc;
      }
    }
    return old;
  }

  T fetch_sub(T inc, std::memory_order memory_order) { return fetch_add(-inc, memory_order); }

  T fetch_min(T other, std::memory_order memory_order)
  {
    T old;

    if (memory_order == std::memory_order::seq_cst) {
#pragma omp atomic compare capture seq_cst
      {
        old = val;
        if (other < val) { val = other; }
      }
    } else if (memory_order == std::memory_order::acquire) {
#pragma omp atomic compare capture acquire
      {
        old = val;
        if (other < val) { val = other; }
      }
    } else if (memory_order == std::memory_order::release) {
#pragma omp atomic compare capture release
      {
        old = val;
        if (other < val) { val = other; }
      }
    } else if (memory_order == std::memory_order::acq_rel) {
#pragma omp atomic compare capture acq_rel
      {
        old = val;
        if (other < val) { val = other; }
      }
    } else {
#pragma omp atomic compare capture relaxed
      {
        old = val;
        if (other < val) { val = other; }
      }
    }

    return old;
  }

  T fetch_max(T other, std::memory_order memory_order)
  {
    T old;
    if (memory_order == std::memory_order::seq_cst) {
#pragma omp atomic compare capture seq_cst
      {
        old = val;
        if (other > val) { val = other; }
      }
    } else if (memory_order == std::memory_order::acquire) {
#pragma omp atomic compare capture acquire
      {
        old = val;
        if (other > val) { val = other; }
      }
    } else if (memory_order == std::memory_order::release) {
#pragma omp atomic compare capture release
      {
        old = val;
        if (other > val) { val = other; }
      }
    } else if (memory_order == std::memory_order::acq_rel) {
#pragma omp atomic compare capture acq_rel
      {
        old = val;
        if (other > val) { val = other; }
      }
    } else {
#pragma omp atomic compare capture relaxed
      {
        old = val;
        if (other > val) { val = other; }
      }
    }
    return old;
  }

  // Get the underlying value without atomics
  T& underlying() { return val; }
  T underlying() const { return val; }

 private:
  T val;
};

}  // namespace cuopt

#endif
