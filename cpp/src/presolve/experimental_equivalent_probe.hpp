/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// Experimental scoped permutation transport; no cuOpt object layout changes.
#pragma once

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace lpfeas_equivalent_probe {

inline bool enabled()
{
  const char* reorder  = std::getenv("LPFEAS_REORDER");
  const bool reordered = reorder != nullptr && std::strcmp(reorder, "rcm") == 0;
  const char* value    = std::getenv("LPFEAS_EQUIVALENT_PROBE");
  if (value == nullptr) { return reordered; }
  if (std::strcmp(value, "1") == 0) { return true; }
  if (std::strcmp(value, "0") == 0 && !reordered) { return false; }
  throw std::runtime_error(
    "LPFEAS_EQUIVALENT_PROBE must be unset or 1 with RCM; 0 is allowed only without RCM");
}

inline bool diagnostics_enabled()
{
  const char* value = std::getenv("LPFEAS_INIT_DIAGNOSTICS");
  return (value != nullptr && std::strcmp(value, "1") == 0) ||
         std::getenv("LPFEAS_EQUIVALENT_PROBE_DUMP_DIR") != nullptr;
}

inline bool scaled_dump_enabled()
{
  const char* value = std::getenv("LPFEAS_EQUIVALENT_PROBE_DUMP_SCALED");
  if (value == nullptr || std::strcmp(value, "0") == 0) { return false; }
  if (std::strcmp(value, "1") == 0) { return true; }
  throw std::runtime_error("LPFEAS_EQUIVALENT_PROBE_DUMP_SCALED must be unset, 0, or 1");
}

inline std::uint64_t byte_digest(const void* data, std::size_t bytes)
{
  auto* p            = static_cast<const unsigned char*>(data);
  std::uint64_t hash = UINT64_C(14695981039346656037);
  for (std::size_t i = 0; i < bytes; ++i) {
    hash = (hash ^ p[i]) * UINT64_C(1099511628211);
  }
  return hash;
}

template <class T>
inline std::uint64_t byte_digest(const std::vector<T>& values)
{
  return byte_digest(values.data(), values.size() * sizeof(T));
}

inline void validate_permutation(const std::vector<int>& map, std::size_t size)
{
  if (map.size() != size) { throw std::runtime_error("Equivalent probe map shape mismatch"); }
  std::vector<unsigned char> seen(size, 0);
  for (int old : map) {
    if (old < 0 || static_cast<std::size_t>(old) >= size || seen[old] != 0) {
      throw std::runtime_error("Equivalent probe map is not a bijection");
    }
    seen[old] = 1;
  }
}

struct context {
  const void* presolver_identity;
  std::size_t rows, columns;
  bool reordered;
  std::vector<int> row_new_to_old, column_new_to_old;
  std::string dump_directory;
};

// The registry definitions live in solve.cu only, shared by all three patched TUs.
class scoped_context {
 public:
  explicit scoped_context(const void* presolver_identity);
  ~scoped_context();
  scoped_context(const scoped_context&)            = delete;
  scoped_context& operator=(const scoped_context&) = delete;
  void bind_problem(const void* problem_identity, std::size_t rows, std::size_t columns);

 private:
  const void* presolver_identity_;
  const void* problem_identity_{nullptr};
};

void publish(const void* presolver_identity,
             std::size_t rows,
             std::size_t columns,
             bool reordered,
             const std::vector<int>& row_new_to_old,
             const std::vector<int>& column_new_to_old);

std::shared_ptr<const context> require_context(const void* problem_identity);

template <class T>
inline void write_vector(const context& state, const char* name, const std::vector<T>& values)
{
  if (state.dump_directory.empty()) { return; }
  const auto path = std::filesystem::path(state.dump_directory) / name;
  // The run-owned directory is new; exclusive creation also rejects repeated setup/dump calls.
  std::FILE* file = std::fopen(path.string().c_str(), "wbx");
  if (file == nullptr) {
    throw std::runtime_error("Cannot create exclusive probe dump: " + path.string());
  }
  const auto count       = std::fwrite(values.data(), sizeof(T), values.size(), file);
  const int close_status = std::fclose(file);
  if (count != values.size() || close_status != 0) {
    throw std::runtime_error("Cannot complete probe dump: " + path.string());
  }
}

template <class T>
inline void permute_probe(const context& state, std::vector<T>& probe)
{
  if (probe.size() != state.rows) {
    throw std::runtime_error("Equivalent probe row count changed");
  }
  if (!state.reordered) { return; }
  if (state.row_new_to_old.size() != probe.size()) {
    throw std::runtime_error("Equivalent probe is missing the published row map");
  }
  std::vector<T> permuted(probe.size());
  for (std::size_t row = 0; row < probe.size(); ++row) {
    // Copy scalar bytes: no arithmetic or RNG regeneration in the reordered coordinates.
    std::memcpy(&permuted[row], &probe[state.row_new_to_old[row]], sizeof(T));
  }
  probe.swap(permuted);
}

}  // namespace lpfeas_equivalent_probe
