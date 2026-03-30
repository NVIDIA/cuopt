/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#pragma once

#include <atomic>
#include <cassert>
#include <span>
#include <vector>

namespace cuopt::linear_programming::dual_simplex {

template <typename i_t, typename f_t>
struct shared_strong_branching_context_t {
  std::vector<std::atomic<int>> solved;

  explicit shared_strong_branching_context_t(size_t num_subproblems) : solved(num_subproblems)
  {
    for (auto& s : solved)
      s.store(0);
  }
};

template <typename i_t, typename f_t>
struct shared_strong_branching_context_view_t {
  std::span<std::atomic<int>> solved;

  shared_strong_branching_context_view_t() = default;

  shared_strong_branching_context_view_t(std::span<std::atomic<int>> s) : solved(s) {}

  bool is_valid() const { return !solved.empty(); }

  bool is_solved(i_t local_idx) const
  {
    assert(local_idx >= 0 && static_cast<size_t>(local_idx) < solved.size() &&
           "local_idx out of bounds");
    return solved[local_idx].load() != 0;
  }

  void mark_solved(i_t local_idx) const
  {
    assert(local_idx >= 0 && static_cast<size_t>(local_idx) < solved.size() &&
           "local_idx out of bounds");
    solved[local_idx].store(1);
  }

  shared_strong_branching_context_view_t subview(i_t offset, i_t count) const
  {
    assert(offset >= 0 && count >= 0 && static_cast<size_t>(offset + count) <= solved.size() &&
           "subview out of bounds");
    return {solved.subspan(offset, count)};
  }
};

}  // namespace cuopt::linear_programming::dual_simplex
