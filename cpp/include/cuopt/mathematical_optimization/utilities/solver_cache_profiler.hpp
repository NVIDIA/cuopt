/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#pragma once

#include <utilities/logger.hpp>

#include <array>
#include <chrono>
#include <cstdarg>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

namespace cuopt::linear_programming::cache_profile {

enum class cache_id : int {
  C01 = 0,  // raft::handle_t + stream
  C02,      // cuBLAS / cuSparse warmup (init_handler)
  C03,      // problem fingerprint (structural hash)
  C04,      // augmented vs ADAT choice
  C05,      // ADAT / augmented sparsity pattern
  C06,      // cuDSS handle + config
  C07,      // cuDSS symbolic factorization
  C08,      // dense-column / SOC layout metadata
  C09,      // device buffer allocation (iteration_data setup)
  COUNT
};

inline constexpr int num_cache_ids = static_cast<int>(cache_id::COUNT);

inline const char* cache_id_label(cache_id id)
{
  switch (id) {
    case cache_id::C01: return "C01 raft handle+stream";
    case cache_id::C02: return "C02 cuBLAS/cuSparse warmup";
    case cache_id::C03: return "C03 problem fingerprint";
    case cache_id::C04: return "C04 augmented vs ADAT choice";
    case cache_id::C05: return "C05 KKT sparsity pattern";
    case cache_id::C06: return "C06 cuDSS handle+config";
    case cache_id::C07: return "C07 cuDSS symbolic factorization";
    case cache_id::C08: return "C08 dense-column/SOC layout";
    case cache_id::C09: return "C09 device buffer allocation";
    default: return "C?? unknown";
  }
}

class profiler_t {
 public:
  static profiler_t& instance()
  {
    static profiler_t prof;
    return prof;
  }

  bool enabled() const { return enabled_; }

  void reset()
  {
    times_.fill(0.0);
  }

  void add(cache_id id, double seconds)
  {
    if (!enabled_) { return; }
    times_[static_cast<int>(id)] += seconds;
  }

  double get(cache_id id) const { return times_[static_cast<int>(id)]; }

  double total_measured() const
  {
    double sum = 0.0;
    for (double t : times_) {
      sum += t;
    }
    return sum;
  }

  void log_summary() const
  {
    if (!enabled_) { return; }
    auto emit = [](const char* fmt, ...) {
      va_list args;
      va_start(args, fmt);
      char buf[512];
      vsnprintf(buf, sizeof(buf), fmt, args);
      va_end(args);
      CUOPT_LOG_INFO("%s", buf);
      fprintf(stderr, "%s\n", buf);
    };
    emit("=== Solver cache profile (ms) ===");
    for (int i = 0; i < num_cache_ids; ++i) {
      const double ms = times_[i] * 1000.0;
      emit("Cache profile: %s %.3f", cache_id_label(static_cast<cache_id>(i)), ms);
    }
    emit("Cache profile: TOTAL measured %.3f", total_measured() * 1000.0);
    emit("=== End solver cache profile ===");
  }

 private:
  profiler_t()
  {
    const char* env = std::getenv("CUOPT_CACHE_PROFILE");
    enabled_        = env != nullptr && env[0] != '\0' && std::strcmp(env, "0") != 0;
  }

  bool enabled_{false};
  std::array<double, num_cache_ids> times_{};
};

inline bool enabled() { return profiler_t::instance().enabled(); }

inline void reset() { profiler_t::instance().reset(); }

inline void add(cache_id id, double seconds) { profiler_t::instance().add(id, seconds); }

inline void log_summary() { profiler_t::instance().log_summary(); }

class scoped_timer_t {
 public:
  explicit scoped_timer_t(cache_id id) : id_(id), start_(clock_::now()), active_(enabled()) {}

  ~scoped_timer_t()
  {
    if (!active_) { return; }
    const double elapsed =
      std::chrono::duration<double>(clock_::now() - start_).count();
    add(id_, elapsed);
  }

 private:
  using clock_ = std::chrono::steady_clock;
  cache_id id_;
  clock_::time_point start_;
  bool active_;
};

}  // namespace cuopt::linear_programming::cache_profile

#define CUOPT_CACHE_PROFILE_SCOPE(id) \
  ::cuopt::linear_programming::cache_profile::scoped_timer_t CUOPT_CACHE_PROFILE_CONCAT( \
    _cuopt_cache_scope_, __LINE__)(id)

#define CUOPT_CACHE_PROFILE_CONCAT(a, b) CUOPT_CACHE_PROFILE_CONCAT_IMPL(a, b)
#define CUOPT_CACHE_PROFILE_CONCAT_IMPL(a, b) a##b
