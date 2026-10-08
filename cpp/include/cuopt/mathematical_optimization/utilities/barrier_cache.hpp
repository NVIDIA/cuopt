/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#pragma once

#include <cuopt/export.hpp>

#include <memory>

#include <raft/core/handle.hpp>
#include <rmm/cuda_stream.hpp>

namespace cuopt::mathematical_optimization::barrier {
template <typename i_t, typename f_t>
class iteration_data_t;

void destroy_iteration_data(iteration_data_t<int, double>* data);

void apply_barrier_linear_objective(iteration_data_t<int, double>& data,
                                    double const* barrier_c,
                                    int n);

void apply_barrier_rhs(iteration_data_t<int, double>& data, double const* barrier_b, int m);
}  // namespace cuopt::mathematical_optimization::barrier

namespace cuopt {
namespace CUOPT_EXPORT mathematical_optimization {

template <typename i_t, typename f_t>
struct barrier_transform_t;

/**
 * @brief GPU solve cache owned by DataModel when CUOPT_SEQUENCE_SOLVE is enabled.
 *
 * After an Optimal full solve, holds iteration_data_t and the user-to-presolve transform.
 * The update APIs crush new user data into that workspace and marks the cache dirty so the
 * next solve reuses it (skip convert/presolve/scaling).
 */
template <typename i_t, typename f_t>
class barrier_cache {
 public:
  static std::unique_ptr<barrier_cache> create(unsigned stream_flags);

  barrier_cache(barrier_cache&&) noexcept;
  barrier_cache& operator=(barrier_cache&&) noexcept;
  ~barrier_cache();

  [[nodiscard]] raft::handle_t* handle_ptr();
  [[nodiscard]] raft::handle_t const* handle_ptr() const;

  /** Drop cached iteration workspace and transform (handle/stream stay). */
  void clear();

  /**
   * @brief Take ownership of barrier iteration workspace. @p data may be null (clears).
   */
  void store_iteration_data(barrier::iteration_data_t<i_t, f_t>* data);

  /**
   * @brief Release ownership of cached iteration workspace; caller must delete or wrap it.
   */
  barrier::iteration_data_t<i_t, f_t>* release_iteration_data();

  void store_transform(std::unique_ptr<barrier_transform_t<i_t, f_t>> transform);
  [[nodiscard]] barrier_transform_t<i_t, f_t>* transform();
  [[nodiscard]] barrier_transform_t<i_t, f_t> const* transform() const;
  /** True when an update API has staged new data that the next solve should reuse. */
  [[nodiscard]] bool dirty() const;
  void mark_clean();

  /** True when the last update_rhs made a row presolve dropped as empty infeasible. */
  [[nodiscard]] bool rhs_infeasible() const;

  /**
   * Crush the input linear objective into cached iteration_data_t.c / d_c_ and mark dirty.
   * Requires a stored transform and iteration_data from an Optimal solve.
   */
  void update_linear_objective(f_t const* c, i_t n);

  /**
   * Crush the input constraint RHS into cached iteration_data_t.b / d_b_ and mark dirty.
   * Requires a stored transform and iteration_data from an Optimal solve.
   */
  void update_rhs(f_t const* b, i_t m);

 private:
  barrier_cache(std::unique_ptr<rmm::cuda_stream> stream, std::unique_ptr<raft::handle_t> handle);

  struct impl;
  std::unique_ptr<impl> impl_;
};

// Sequence-solve cache used by the solver and the Python capsule. One instantiation.
using barrier_cache_t = barrier_cache<int, double>;

}  // namespace CUOPT_EXPORT mathematical_optimization
}  // namespace cuopt
