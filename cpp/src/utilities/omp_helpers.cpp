/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#include <utilities/omp_helpers.hpp>

#ifdef _OPENMP

#include <utility>

namespace cuopt {

// All operations on the underlying `omp_lock_t` are defined out-of-line so
// that `new omp_lock_t` and the matching (sized) `delete` invoked through
// `std::unique_ptr<omp_lock_t>` exist in exactly one translation unit. This
// avoids ODR-induced `new-delete-type-mismatch` errors when other TUs (most
// notably NVCC host passes) end up with a differently sized `omp_lock_t`.

omp_mutex_t::omp_mutex_t() : mutex(new omp_lock_t) { omp_init_lock(mutex.get()); }

omp_mutex_t::omp_mutex_t(omp_mutex_t&& other) noexcept { *this = std::move(other); }

omp_mutex_t& omp_mutex_t::operator=(omp_mutex_t&& other) noexcept
{
  if (&other != this) {
    if (mutex) { omp_destroy_lock(mutex.get()); }
    mutex = std::move(other.mutex);
  }
  return *this;
}

omp_mutex_t::~omp_mutex_t()
{
  if (mutex) {
    omp_destroy_lock(mutex.get());
    mutex.reset();
  }
}

}  // namespace cuopt

#endif  // _OPENMP
