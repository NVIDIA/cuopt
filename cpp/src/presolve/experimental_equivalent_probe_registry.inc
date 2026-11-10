/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// Included exactly once by solve.cu; the CPU fixture uses the same implementation.
#include <presolve/experimental_equivalent_probe.hpp>

#include <mutex>
#include <unordered_map>

namespace lpfeas_equivalent_probe {
namespace {
struct registry {
  std::mutex lock;
  std::unordered_map<const void*, std::shared_ptr<const context>> owners;
  std::unordered_map<const void*, std::shared_ptr<const context>> problems;
};

registry& storage()
{
  static registry value;
  return value;
}
}  // namespace

scoped_context::scoped_context(const void* presolver_identity)
  : presolver_identity_(presolver_identity)
{
  if (presolver_identity == nullptr) { throw std::runtime_error("Null probe presolver identity"); }
  auto& state = storage();
  std::lock_guard<std::mutex> guard(state.lock);
  if (!state.owners.emplace(presolver_identity, nullptr).second) {
    throw std::runtime_error("Duplicate live probe presolver identity");
  }
}

scoped_context::~scoped_context()
{
  auto& state = storage();
  std::lock_guard<std::mutex> guard(state.lock);
  if (problem_identity_ != nullptr) { state.problems.erase(problem_identity_); }
  state.owners.erase(presolver_identity_);
}

void scoped_context::bind_problem(const void* problem_identity,
                                  std::size_t rows,
                                  std::size_t columns)
{
  auto& state = storage();
  std::lock_guard<std::mutex> guard(state.lock);
  const auto owner = state.owners.find(presolver_identity_);
  if (problem_identity == nullptr || problem_identity_ != nullptr || owner == state.owners.end() ||
      !owner->second || owner->second->rows != rows || owner->second->columns != columns) {
    throw std::runtime_error("Missing, duplicate, or mismatched equivalent-probe publication");
  }
  if (!state.problems.emplace(problem_identity, owner->second).second) {
    throw std::runtime_error("Duplicate live probe problem identity");
  }
  problem_identity_ = problem_identity;
}

void publish(const void* presolver_identity,
             std::size_t rows,
             std::size_t columns,
             bool reordered,
             const std::vector<int>& row_new_to_old,
             const std::vector<int>& column_new_to_old)
{
  if (reordered) {
    validate_permutation(row_new_to_old, rows);
    validate_permutation(column_new_to_old, columns);
  } else if (!row_new_to_old.empty() || !column_new_to_old.empty()) {
    throw std::runtime_error("Inactive RCM unexpectedly retained a map");
  }
  std::string dump_directory;
  if (const char* value = std::getenv("LPFEAS_EQUIVALENT_PROBE_DUMP_DIR")) {
    dump_directory = value;
    if (dump_directory.empty() || !std::filesystem::path(dump_directory).is_absolute()) {
      throw std::runtime_error("Equivalent probe dump directory must be a new absolute path");
    }
  }
  if (scaled_dump_enabled() && dump_directory.empty()) {
    throw std::runtime_error("Scaled CSR dumping requires LPFEAS_EQUIVALENT_PROBE_DUMP_DIR");
  }
  auto publication = std::make_shared<const context>(context{
    presolver_identity, rows, columns, reordered, row_new_to_old, column_new_to_old, dump_directory});
  auto& state = storage();
  std::lock_guard<std::mutex> guard(state.lock);
  const auto owner = state.owners.find(presolver_identity);
  if (owner == state.owners.end() || owner->second) {
    throw std::runtime_error("Equivalent probe publication has no unique registered owner");
  }
  if (!dump_directory.empty()) {
    if (!std::filesystem::create_directory(dump_directory)) {
      throw std::runtime_error("Equivalent probe dump directory already exists");
    }
    write_vector(*publication, "row_new_to_old.i32.bin", row_new_to_old);
    write_vector(*publication, "column_new_to_old.i32.bin", column_new_to_old);
  }
  owner->second = std::move(publication);
}

std::shared_ptr<const context> require_context(const void* problem_identity)
{
  auto& state = storage();
  std::lock_guard<std::mutex> guard(state.lock);
  const auto found = state.problems.find(problem_identity);
  if (found == state.problems.end()) {
    throw std::runtime_error("Equivalent probe has no context for this exact problem identity");
  }
  return found->second;
}
}  // namespace lpfeas_equivalent_probe
