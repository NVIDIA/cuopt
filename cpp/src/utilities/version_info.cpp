/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */
#include "version_info.hpp"

#include <cuda_runtime.h>

#include <cuopt/version_config.hpp>
#include <utilities/build_info.hpp>
#include <utilities/logger.hpp>

#include <hwy/per_target.h>
#include <hwy/targets.h>

#include <fcntl.h>
#include <sched.h>
#include <unistd.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace cuopt {

// Reads up to buf_size-1 bytes, NUL-terminates, strips trailing whitespace/NULs.
// Returns bytes kept (excluding the terminator), or -1 on failure.
static ssize_t read_file_buf(const char* path, char* buf, size_t buf_size)
{
  if (buf_size == 0) return -1;
  const int fd = open(path, O_RDONLY);
  if (fd < 0) return -1;
  const ssize_t n = read(fd, buf, buf_size - 1);
  close(fd);
  if (n < 0) return -1;
  buf[n] = '\0';

  // Device-tree properties are often NUL-terminated without a trailing newline.
  size_t len = 0;
  while (len < (size_t)n && buf[len] != '\0') {
    ++len;
  }
  buf[len] = '\0';
  while (len > 0 &&
         (buf[len - 1] == '\n' || buf[len - 1] == '\r' || buf[len - 1] == ' ' ||
          buf[len - 1] == '\t')) {
    buf[--len] = '\0';
  }
  return (ssize_t)len;
}

// Parses a kernel CPU list ("0-3,8,10-11") into cpus[0..max_cpus). Returns count written.
static int parse_cpu_list(const char* list, int* cpus, int max_cpus)
{
  int count     = 0;
  const char* p = list;
  while (*p && count < max_cpus) {
    while (*p == ',' || *p == ' ' || *p == '\t' || *p == '\n' || *p == '\r') {
      ++p;
    }
    if (*p == '\0') break;

    char* end     = nullptr;
    const long lo = std::strtol(p, &end, 10);
    if (end == p) break;
    p = end;

    if (*p == '-') {
      ++p;
      const long hi = std::strtol(p, &end, 10);
      if (end == p) break;
      p = end;
      for (long cpu = lo; cpu <= hi && count < max_cpus; ++cpu) {
        cpus[count++] = (int)cpu;
      }
    } else {
      cpus[count++] = (int)lo;
    }
  }
  return count;
}

static void mark_cpus_from_list(const char* list, char visited[CPU_SETSIZE])
{
  const char* p = list;
  while (*p) {
    while (*p == ',' || *p == ' ' || *p == '\t' || *p == '\n' || *p == '\r') {
      ++p;
    }
    if (*p == '\0') break;

    char* end     = nullptr;
    const long lo = std::strtol(p, &end, 10);
    if (end == p) break;
    p = end;

    if (*p == '-') {
      ++p;
      const long hi = std::strtol(p, &end, 10);
      if (end == p) break;
      p = end;
      for (long cpu = lo; cpu <= hi; ++cpu) {
        if (cpu >= 0 && cpu < CPU_SETSIZE) { visited[cpu] = 1; }
      }
    } else if (lo >= 0 && lo < CPU_SETSIZE) {
      visited[lo] = 1;
    }
  }
}

// CPUs this process may run on (respects Slurm/cgroup cpusets, taskset, etc.).
static int get_allowed_cpus(int* cpus, int max_cpus)
{
  cpu_set_t set;
  CPU_ZERO(&set);
  int count = 0;
  if (sched_getaffinity(0, sizeof(set), &set) == 0) {
    for (int cpu = 0; cpu < CPU_SETSIZE && count < max_cpus; ++cpu) {
      if (CPU_ISSET(cpu, &set)) { cpus[count++] = cpu; }
    }
  }
  if (count > 0) return count;

  char buf[256];
  if (read_file_buf("/sys/devices/system/cpu/online", buf, sizeof(buf)) < 0) return 0;
  return parse_cpu_list(buf, cpus, max_cpus);
}

static int get_physical_cores(const int* allowed_cpus, int allowed_count)
{
  if (allowed_count <= 0) return 0;

  char visited[CPU_SETSIZE];
  std::memset(visited, 0, sizeof(visited));
  int cores = 0;

  for (int i = 0; i < allowed_count; ++i) {
    const int cpu = allowed_cpus[i];
    if (cpu < 0 || cpu >= CPU_SETSIZE || visited[cpu]) continue;

    char path[128];
    char buf[256];
    snprintf(path,
             sizeof(path),
             "/sys/devices/system/cpu/cpu%d/topology/core_cpus_list",
             cpu);
    ssize_t n = read_file_buf(path, buf, sizeof(buf));
    if (n < 0) {
      snprintf(path,
               sizeof(path),
               "/sys/devices/system/cpu/cpu%d/topology/thread_siblings_list",
               cpu);
      n = read_file_buf(path, buf, sizeof(buf));
    }

    if (n >= 0) {
      mark_cpus_from_list(buf, visited);
    }
    visited[cpu] = 1;
    ++cores;
  }

  return cores > 0 ? cores : allowed_count;
}

static bool copy_stripped(char* dst, size_t dst_size, const char* src)
{
  if (dst_size == 0) return false;
  size_t len = std::strlen(src);
  while (len > 0 && (src[len - 1] == '\n' || src[len - 1] == '\r' || src[len - 1] == ' ')) {
    --len;
  }
  if (len >= dst_size) len = dst_size - 1;
  std::memcpy(dst, src, len);
  dst[len] = '\0';
  return len > 0;
}

static bool get_cpu_model_from_proc(char* out, size_t out_size)
{
  FILE* cpuinfo = fopen("/proc/cpuinfo", "r");
  if (cpuinfo == nullptr) return false;

  char line[512];
  while (fgets(line, sizeof(line), cpuinfo) != nullptr) {
    const char* field = std::strstr(line, "model name");
    if (field == nullptr) field = std::strstr(line, "Processor");
    if (field == nullptr) continue;

    const char* colon = std::strchr(field, ':');
    if (colon == nullptr) continue;
    ++colon;
    while (*colon == ' ' || *colon == '\t') {
      ++colon;
    }
    const bool ok = copy_stripped(out, out_size, colon);
    fclose(cpuinfo);
    return ok;
  }
  fclose(cpuinfo);
  return false;
}

static void get_cpu_model(char* out, size_t out_size)
{
  if (get_cpu_model_from_proc(out, out_size)) return;

  char buf[256];
  if (read_file_buf("/sys/firmware/devicetree/base/model", buf, sizeof(buf)) >= 0 ||
      read_file_buf("/proc/device-tree/model", buf, sizeof(buf)) >= 0) {
    if (copy_stripped(out, out_size, buf)) return;
  }
  if (read_file_buf("/sys/devices/virtual/dmi/id/product_name", buf, sizeof(buf)) >= 0) {
    if (copy_stripped(out, out_size, buf)) return;
  }
  std::snprintf(out, out_size, "Unknown");
}

static const char* get_simd_target()
{
  const int64_t target = hwy::DispatchedTarget();
  switch (target) {
    case HWY_AVX3:
    case HWY_AVX3_DL:
    case HWY_AVX3_ZEN4:
    case HWY_AVX3_SPR:
    case HWY_AVX10_2: return "AVX-512";
    default: return hwy::TargetName(target);
  }
}

struct host_memory_info_t {
  double total_gb{};
  double available_gb{};
};

static host_memory_info_t get_host_memory_info()
{
  FILE* meminfo = fopen("/proc/meminfo", "r");
  if (meminfo == nullptr) return {};

  char line[256];
  long total_kb     = 0;
  long available_kb = 0;
  long free_kb      = 0;
  int found         = 0;
  while (found < 3 && fgets(line, sizeof(line), meminfo) != nullptr) {
    long value_kb = 0;
    if (std::sscanf(line, "MemTotal: %ld", &value_kb) == 1) {
      total_kb = value_kb;
      ++found;
    } else if (std::sscanf(line, "MemAvailable: %ld", &value_kb) == 1) {
      available_kb = value_kb;
      ++found;
    } else if (std::sscanf(line, "MemFree: %ld", &value_kb) == 1) {
      free_kb = value_kb;
      ++found;
    }
  }
  fclose(meminfo);

  if (available_kb == 0) { available_kb = free_kb; }
  constexpr double kb_per_gib = 1024.0 * 1024.0;
  return {total_kb / kb_per_gib, available_kb / kb_per_gib};
}

void print_version_info(int num_devices)
{
  int version = 0;
  if (cudaRuntimeGetVersion(&version) != cudaSuccess) {
    CUOPT_LOG_WARN("Failed to query CUDA runtime version");
    version = 0;
  }
  int major = version / 1000;
  int minor = (version % 1000) / 10;

  CUOPT_LOG_INFO("cuOpt version: %d.%d.%d, git hash: %s, host arch: %s, device archs: %s",
                 CUOPT_VERSION_MAJOR,
                 CUOPT_VERSION_MINOR,
                 CUOPT_VERSION_PATCH,
                 CUOPT_GIT_COMMIT_HASH,
                 CUOPT_CPU_ARCHITECTURE,
                 CUOPT_CUDA_ARCHITECTURES);

  const auto memory = get_host_memory_info();
  int allowed_cpus[CPU_SETSIZE];
  const int allowed_count = get_allowed_cpus(allowed_cpus, CPU_SETSIZE);
  char cpu_model[256];
  get_cpu_model(cpu_model, sizeof(cpu_model));
  CUOPT_LOG_INFO("CPU: %s, threads: %dC/%dT, RAM usage: %.2f/%.2fGiB",
                 cpu_model,
                 get_physical_cores(allowed_cpus, allowed_count),
                 allowed_count,
                 std::max(0.0, memory.total_gb - memory.available_gb),
                 memory.total_gb);
  CUOPT_LOG_INFO("CPU SIMD target: %s", get_simd_target());

  for (int device_id = 0; device_id < num_devices; ++device_id) {
    cudaDeviceProp device_prop{};
    if (cudaGetDeviceProperties(&device_prop, device_id) != cudaSuccess) {
      CUOPT_LOG_WARN("Failed to query CUDA device properties for device ID %d", device_id);
      continue;
    }

    const cudaUUID_t uuid = device_prop.uuid;
    char uuid_str[37]     = {0};
    snprintf(uuid_str,
             sizeof(uuid_str),
             "%02x%02x%02x%02x-%02x%02x-%02x%02x-%02x%02x-%02x%02x%02x%02x%02x%02x",
             (unsigned char)uuid.bytes[0],
             (unsigned char)uuid.bytes[1],
             (unsigned char)uuid.bytes[2],
             (unsigned char)uuid.bytes[3],
             (unsigned char)uuid.bytes[4],
             (unsigned char)uuid.bytes[5],
             (unsigned char)uuid.bytes[6],
             (unsigned char)uuid.bytes[7],
             (unsigned char)uuid.bytes[8],
             (unsigned char)uuid.bytes[9],
             (unsigned char)uuid.bytes[10],
             (unsigned char)uuid.bytes[11],
             (unsigned char)uuid.bytes[12],
             (unsigned char)uuid.bytes[13],
             (unsigned char)uuid.bytes[14],
             (unsigned char)uuid.bytes[15]);

    CUOPT_LOG_INFO("CUDA %d.%d, device: %s (ID %d), VRAM: %.2f GiB",
                   major,
                   minor,
                   device_prop.name,
                   device_id,
                   (double)device_prop.totalGlobalMem / (1024.0 * 1024.0 * 1024.0));
    CUOPT_LOG_INFO("CUDA device UUID: %s", uuid_str);
  }
}

}  // namespace cuopt
