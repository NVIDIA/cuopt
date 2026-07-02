/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#pragma once

#include <memory>

#include <raft/core/handle.hpp>
#include <rmm/cuda_stream.hpp>

namespace cuopt::linear_programming::dual_simplex {
template <typename i_t, typename f_t>
class iteration_data_t;
template <typename i_t, typename f_t>
struct barrier_symbolic_cache_t;

template <typename i_t, typename f_t>
void barrier_store_symbolic_cache_from_iteration_data(iteration_data_t<i_t, f_t>& data,
                                                      barrier_symbolic_cache_t<i_t, f_t>& cache);
}  // namespace cuopt::linear_programming::dual_simplex

namespace cuopt::cython {

/**
 * @brief Lean GPU solve session: owns RAFT handle + stream and optional barrier symbolic cache.
 *
 * Created on first solve when session_enabled; reused on subsequent solves with the same capsule.
 * Per-solve state (optimization_problem_t, presolve, barrier_lp) remains stack-local.
 */
class lp_solve_session_t {
 public:
  static std::unique_ptr<lp_solve_session_t> create(unsigned stream_flags);

  lp_solve_session_t(lp_solve_session_t&&) noexcept;
  lp_solve_session_t& operator=(lp_solve_session_t&&) noexcept;
  ~lp_solve_session_t();

  [[nodiscard]] raft::handle_t* handle_ptr();
  [[nodiscard]] raft::handle_t const* handle_ptr() const;
  [[nodiscard]] rmm::cuda_stream_view stream_view() const;

  /**
   * @brief Returns cached symbolic state when valid and @p handle matches the stored handle.
   */
  [[nodiscard]] linear_programming::dual_simplex::barrier_symbolic_cache_t<int, double>*
  symbolic_cache_for_reuse(raft::handle_t const* handle);

  void clear_symbolic_cache();

  void store_symbolic_cache(
    linear_programming::dual_simplex::iteration_data_t<int, double>& data);

 private:
  lp_solve_session_t(std::unique_ptr<rmm::cuda_stream> stream,
                     std::unique_ptr<raft::handle_t> handle);

  struct impl;
  std::unique_ptr<impl> impl_;
};

}  // namespace cuopt::cython
