/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights
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
#include "version_info.hpp"

#include <cuda_runtime.h>

#include <cuopt/logger.hpp>
#include <cuopt/version_config.hpp>

#include <fstream>
#include <iomanip>
#include <set>
#include <sstream>
#include <string>
#include <thread>

namespace cuopt {

static int get_physical_cores()
{
  std::ifstream cpuinfo("/proc/cpuinfo");
  if (!cpuinfo.is_open()) return 0;

  std::string line;
  int physical_id = -1, core_id = -1;
  std::set<std::pair<int, int>> cores;

  while (std::getline(cpuinfo, line)) {
    if (line.find("physical id") != std::string::npos) {
      physical_id = std::stoi(line.substr(line.find(":") + 1));
    } else if (line.find("core id") != std::string::npos) {
      core_id = std::stoi(line.substr(line.find(":") + 1));
    }

    if (physical_id != -1 && core_id != -1) {
      cores.insert({physical_id, core_id});
      physical_id = -1;
      core_id     = -1;
    }
  }

  if (cores.empty()) {
    cpuinfo.clear();
    cpuinfo.seekg(0);
    while (std::getline(cpuinfo, line)) {
      if (line.find("cpu cores") != std::string::npos) {
        return std::stoi(line.substr(line.find(":") + 1));
      }
    }
    return 1;
  }
  return cores.size();
}

static std::string get_cpu_model_from_proc()
{
  std::ifstream cpuinfo("/proc/cpuinfo");
  if (!cpuinfo.is_open()) return "";

  std::string line;
  while (std::getline(cpuinfo, line)) {
    std::size_t pos = line.find("model name");
    if (pos == std::string::npos) pos = line.find("Processor");
    if (pos != std::string::npos) {
      std::size_t colon = line.find(':', pos);
      if (colon != std::string::npos) return line.substr(colon + 2);  // Skip ": "
    }
  }
  return "";
}

// From https://gcc.gnu.org/onlinedocs/gcc/x86-Built-in-Functions.html
// Also supported by clang
static std::string get_cpu_model_builtin()
{
#if (defined(__x86_64__) || defined(__i386__)) && (defined(__GNUC__) || defined(__clang__))
  __builtin_cpu_init();
  return __builtin_cpu_is("amd")               ? "AMD CPU"
         : __builtin_cpu_is("intel")           ? "Intel CPU"
         : __builtin_cpu_is("atom")            ? "Intel Atom CPU"
         : __builtin_cpu_is("slm")             ? "Intel Silvermont CPU"
         : __builtin_cpu_is("core2")           ? "Intel Core 2 CPU"
         : __builtin_cpu_is("corei7")          ? "Intel Core i7 CPU"
         : __builtin_cpu_is("nehalem")         ? "Intel Core i7 Nehalem CPU"
         : __builtin_cpu_is("westmere")        ? "Intel Core i7 Westmere CPU"
         : __builtin_cpu_is("sandybridge")     ? "Intel Core i7 Sandy Bridge CPU"
         : __builtin_cpu_is("ivybridge")       ? "Intel Core i7 Ivy Bridge CPU"
         : __builtin_cpu_is("haswell")         ? "Intel Core i7 Haswell CPU"
         : __builtin_cpu_is("broadwell")       ? "Intel Core i7 Broadwell CPU"
         : __builtin_cpu_is("skylake")         ? "Intel Core i7 Skylake CPU"
         : __builtin_cpu_is("skylake-avx512")  ? "Intel Core i7 Skylake AVX512 CPU"
         : __builtin_cpu_is("cannonlake")      ? "Intel Core i7 Cannon Lake CPU"
         : __builtin_cpu_is("icelake-client")  ? "Intel Core i7 Ice Lake Client CPU"
         : __builtin_cpu_is("icelake-server")  ? "Intel Core i7 Ice Lake Server CPU"
         : __builtin_cpu_is("cascadelake")     ? "Intel Core i7 Cascadelake CPU"
         : __builtin_cpu_is("tigerlake")       ? "Intel Core i7 Tigerlake CPU"
         : __builtin_cpu_is("cooperlake")      ? "Intel Core i7 Cooperlake CPU"
         : __builtin_cpu_is("sapphirerapids")  ? "Intel Core i7 sapphirerapids CPU"
         : __builtin_cpu_is("alderlake")       ? "Intel Core i7 Alderlake CPU"
         : __builtin_cpu_is("rocketlake")      ? "Intel Core i7 Rocketlake CPU"
         : __builtin_cpu_is("graniterapids")   ? "Intel Core i7 graniterapids CPU"
         : __builtin_cpu_is("graniterapids-d") ? "Intel Core i7 graniterapids D CPU"
         : __builtin_cpu_is("bonnell")         ? "Intel Atom Bonnell CPU"
         : __builtin_cpu_is("silvermont")      ? "Intel Atom Silvermont CPU"
         : __builtin_cpu_is("goldmont")        ? "Intel Atom Goldmont CPU"
         : __builtin_cpu_is("goldmont-plus")   ? "Intel Atom Goldmont Plus CPU"
         : __builtin_cpu_is("tremont")         ? "Intel Atom Tremont CPU"
         : __builtin_cpu_is("sierraforest")    ? "Intel Atom Sierra Forest CPU"
         : __builtin_cpu_is("grandridge")      ? "Intel Atom Grand Ridge CPU"
         : __builtin_cpu_is("amdfam10h")       ? "AMD Family 10h CPU"
         : __builtin_cpu_is("barcelona")       ? "AMD Family 10h Barcelona CPU"
         : __builtin_cpu_is("shanghai")        ? "AMD Family 10h Shanghai CPU"
         : __builtin_cpu_is("istanbul")        ? "AMD Family 10h Istanbul CPU"
         : __builtin_cpu_is("btver1")          ? "AMD Family 14h CPU"
         : __builtin_cpu_is("amdfam15h")       ? "AMD Family 15h CPU"
         : __builtin_cpu_is("bdver1")          ? "AMD Family 15h Bulldozer version 1"
         : __builtin_cpu_is("bdver2")          ? "AMD Family 15h Bulldozer version 2"
         : __builtin_cpu_is("bdver3")          ? "AMD Family 15h Bulldozer version 3"
         : __builtin_cpu_is("bdver4")          ? "AMD Family 15h Bulldozer version 4"
         : __builtin_cpu_is("btver2")          ? "AMD Family 16h CPU"
         : __builtin_cpu_is("amdfam17h")       ? "AMD Family 17h CPU"
         : __builtin_cpu_is("znver1")          ? "AMD Family 17h Zen version 1"
         : __builtin_cpu_is("znver2")          ? "AMD Family 17h Zen version 2"
         : __builtin_cpu_is("amdfam19h")       ? "AMD Family 19h CPU"
                                               : "Unknown";
#else
  return "Unknown";
#endif
}

static std::string get_cpu_model()
{
  if (auto model_from_proc = get_cpu_model_from_proc(); !model_from_proc.empty()) {
    return model_from_proc;
  } else if (auto model_from_builtin = get_cpu_model_builtin(); !model_from_builtin.empty()) {
    return model_from_builtin;
  }
  return "Unknown";
}

static double get_available_memory_gb()
{
  std::ifstream meminfo("/proc/meminfo");
  if (!meminfo.is_open()) return 0.0;

  std::string line;
  long kb = 0;
  while (std::getline(meminfo, line)) {
    if (line.find("MemAvailable:") == 0 || line.find("MemFree:") == 0) {
      std::size_t pos = line.find_first_of("0123456789");
      if (pos != std::string::npos) {
        kb = std::stol(line.substr(pos));
        break;
      }
    }
  }

  return kb / (1024.0 * 1024.0);  // Convert KB to GB
}

void print_version_info()
{
  int device_id = 0;
  cudaGetDevice(&device_id);
  cudaDeviceProp device_prop;
  cudaGetDeviceProperties(&device_prop, device_id);
  cudaUUID_t uuid   = device_prop.uuid;
  char uuid_str[37] = {0};
  snprintf(uuid_str,
           sizeof(uuid_str),
           "%02x%02x%02x%02x-%02x%02x-%02x%02x-%02x%02x-%02x%02x%02x%02x%02x%02x",
           uuid.bytes[0],
           uuid.bytes[1],
           uuid.bytes[2],
           uuid.bytes[3],
           uuid.bytes[4],
           uuid.bytes[5],
           uuid.bytes[6],
           uuid.bytes[7],
           uuid.bytes[8],
           uuid.bytes[9],
           uuid.bytes[10],
           uuid.bytes[11],
           uuid.bytes[12],
           uuid.bytes[13],
           uuid.bytes[14],
           uuid.bytes[15]);
  int version = 0;
  cudaRuntimeGetVersion(&version);
  int major = version / 1000;
  int minor = (version % 1000) / 10;
  CUOPT_LOG_INFO("cuOpt version: %d.%d.%d, git hash: %s, host arch: %s, device archs: %s",
                 CUOPT_VERSION_MAJOR,
                 CUOPT_VERSION_MINOR,
                 CUOPT_VERSION_PATCH,
                 CUOPT_GIT_COMMIT_HASH,
                 CUOPT_CPU_ARCHITECTURE,
                 CUOPT_CUDA_ARCHITECTURES);
  CUOPT_LOG_INFO("CPU: %s, threads (physical/logical): %d/%d, RAM: %.2f GiB",
                 get_cpu_model().c_str(),
                 get_physical_cores(),
                 std::thread::hardware_concurrency(),
                 get_available_memory_gb());
  CUOPT_LOG_INFO("CUDA %d.%d, device: %s (ID %d), VRAM: %.2f GiB",
                 major,
                 minor,
                 device_prop.name,
                 device_id,
                 (double)device_prop.totalGlobalMem / (1024.0 * 1024.0 * 1024.0));
  CUOPT_LOG_INFO("CUDA device UUID: %s\n", uuid_str);
}

}  // namespace cuopt
