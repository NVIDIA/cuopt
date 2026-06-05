// SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights
// reserved. SPDX-License-Identifier: Apache-2.0

#pragma once

#include <linux/perf_event.h>
#include <sys/ioctl.h>
#include <sys/syscall.h>
#include <unistd.h>

#include <array>
#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <vector>

namespace mps_fast {

struct perf_counter_spec_t {
  const char* name;
  uint32_t type;
  uint64_t config;
};

static constexpr uint64_t perf_cache_config(uint64_t cache, uint64_t op, uint64_t result)
{
  return cache | (op << 8) | (result << 16);
}

// Small scoped Linux perf_event_open wrapper for coarse phase diagnostics.
//
// Important limitations:
// - Counters are per-thread: construct one instance inside each worker whose
//   work should be measured, then aggregate snapshots.
// - These are generic perf events; exact mappings vary by CPU. Some events may
//   be unavailable or unhelpful, e.g. store-side DTLB misses on this node.
// - This deliberately does not use event groups or time_enabled/time_running
//   scaling, so counts are approximate if the kernel multiplexes counters.
static constexpr std::array<perf_counter_spec_t, 8> PERF_COUNTER_SPECS = {{
  {"cycles", PERF_TYPE_HARDWARE, PERF_COUNT_HW_CPU_CYCLES},
  {"instructions", PERF_TYPE_HARDWARE, PERF_COUNT_HW_INSTRUCTIONS},
  {"cache_refs", PERF_TYPE_HARDWARE, PERF_COUNT_HW_CACHE_REFERENCES},
  {"cache_misses", PERF_TYPE_HARDWARE, PERF_COUNT_HW_CACHE_MISSES},
  {"branch_misses", PERF_TYPE_HARDWARE, PERF_COUNT_HW_BRANCH_MISSES},
  {"backend_stall_cycles", PERF_TYPE_HARDWARE, PERF_COUNT_HW_STALLED_CYCLES_BACKEND},
  {"dtlb_load_misses",
   PERF_TYPE_HW_CACHE,
   perf_cache_config(
     PERF_COUNT_HW_CACHE_DTLB, PERF_COUNT_HW_CACHE_OP_READ, PERF_COUNT_HW_CACHE_RESULT_MISS)},
  {"dtlb_store_misses",
   PERF_TYPE_HW_CACHE,
   perf_cache_config(
     PERF_COUNT_HW_CACHE_DTLB, PERF_COUNT_HW_CACHE_OP_WRITE, PERF_COUNT_HW_CACHE_RESULT_MISS)},
}};

struct perf_counter_snapshot_t {
  bool active                                            = false;
  int open_errno                                         = 0;
  std::array<uint64_t, PERF_COUNTER_SPECS.size()> values = {};
};

class thread_perf_counters_t {
 public:
  thread_perf_counters_t()
  {
    fds_.fill(-1);
    for (size_t i = 0; i < PERF_COUNTER_SPECS.size(); ++i) {
      perf_event_attr attr = {};
      attr.type            = PERF_COUNTER_SPECS[i].type;
      attr.size            = sizeof(attr);
      attr.config          = PERF_COUNTER_SPECS[i].config;
      attr.disabled        = 1;
      attr.exclude_kernel  = 1;
      attr.exclude_hv      = 1;

      int fd = (int)syscall(__NR_perf_event_open, &attr, 0, -1, -1, 0);
      if (fd < 0) {
        if (first_errno_ == 0) { first_errno_ = errno; }
        continue;
      }
      fds_[i] = fd;
      active_ = true;
    }

    if (active_) {
      for (int fd : fds_) {
        if (fd >= 0) {
          ioctl(fd, PERF_EVENT_IOC_RESET, 0);
          ioctl(fd, PERF_EVENT_IOC_ENABLE, 0);
        }
      }
    }
  }

  thread_perf_counters_t(const thread_perf_counters_t&)            = delete;
  thread_perf_counters_t& operator=(const thread_perf_counters_t&) = delete;

  ~thread_perf_counters_t() { close_all(); }

  perf_counter_snapshot_t stop()
  {
    perf_counter_snapshot_t snapshot;
    snapshot.active     = active_;
    snapshot.open_errno = first_errno_;

    for (size_t i = 0; i < fds_.size(); ++i) {
      int fd = fds_[i];
      if (fd < 0) continue;
      ioctl(fd, PERF_EVENT_IOC_DISABLE, 0);
      uint64_t value = 0;
      if (read(fd, &value, sizeof(value)) == (ssize_t)sizeof(value)) { snapshot.values[i] = value; }
    }
    close_all();
    active_ = false;
    return snapshot;
  }

 private:
  void close_all()
  {
    for (int& fd : fds_) {
      if (fd >= 0) {
        close(fd);
        fd = -1;
      }
    }
  }

  bool active_     = false;
  int first_errno_ = 0;
  std::array<int, PERF_COUNTER_SPECS.size()> fds_;
};

static inline void print_perf_totals(const char* label,
                                     const std::vector<perf_counter_snapshot_t>& snapshots)
{
  std::array<unsigned long long, PERF_COUNTER_SPECS.size()> totals = {};
  bool any_active                                                  = false;
  int first_errno                                                  = 0;
  for (const auto& snapshot : snapshots) {
    if (snapshot.open_errno != 0 && first_errno == 0) { first_errno = snapshot.open_errno; }
    if (!snapshot.active) continue;
    any_active = true;
    for (size_t i = 0; i < PERF_COUNTER_SPECS.size(); ++i) {
      totals[i] += snapshot.values[i];
    }
  }

  if (!any_active) {
    std::fprintf(stderr, "[PERF] %s unavailable errno=%d\n", label, first_errno);
    return;
  }

  double ipc       = totals[0] == 0 ? 0.0 : (double)totals[1] / (double)totals[0];
  double miss_rate = totals[2] == 0 ? 0.0 : (double)totals[3] / (double)totals[2];
  std::fprintf(stderr, "[PERF] %s", label);
  for (size_t i = 0; i < PERF_COUNTER_SPECS.size(); ++i) {
    std::fprintf(stderr, " %s=%llu", PERF_COUNTER_SPECS[i].name, totals[i]);
  }
  std::fprintf(stderr, " ipc=%.3f cache_miss_rate=%.6f\n", ipc, miss_rate);
}

}  // namespace mps_fast
