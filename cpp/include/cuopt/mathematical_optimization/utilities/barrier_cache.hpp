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
template <typename i_t, typename f_t>
struct barrier_symbolic_cache_t;

template <typename i_t, typename f_t>
void barrier_store_symbolic_cache_from_iteration_data(iteration_data_t<i_t, f_t>& data,
                                                      barrier_symbolic_cache_t<i_t, f_t>& cache);

void destroy_iteration_data(iteration_data_t<int, double>* data);

void apply_barrier_linear_objective(iteration_data_t<int, double>& data,
                                    double const* barrier_c,
                                    int n);
}  // namespace cuopt::mathematical_optimization::barrier

namespace cuopt {
namespace CUOPT_EXPORT cython {

struct barrier_front_end_cache_t;

/**
 * @brief Lean GPU solve session: owns RAFT handle + stream, optional barrier symbolic cache,
 * and optional barrier iteration_data_t (GPU IPM workspace) after an Optimal solve.
 *
 * Created on first solve when sequence_solve; reused on subsequent solves with the same capsule.
 * Per-solve convert/presolve/scaling remain stack-local until the continue path (D); A keeps
 * iteration_data_t, B keeps front-end maps + c_dirty.
 */
class barrier_cache_t {
 public:
  static std::unique_ptr<barrier_cache_t> create(unsigned stream_flags);

  barrier_cache_t(barrier_cache_t&&) noexcept;
  barrier_cache_t& operator=(barrier_cache_t&&) noexcept;
  ~barrier_cache_t();

  [[nodiscard]] raft::handle_t* handle_ptr();
  [[nodiscard]] raft::handle_t const* handle_ptr() const;
  [[nodiscard]] rmm::cuda_stream_view stream_view() const;

  /**
   * @brief Returns cached symbolic state when valid and @p handle matches the stored handle.
   */
  [[nodiscard]] mathematical_optimization::barrier::barrier_symbolic_cache_t<int, double>*
  symbolic_cache_for_reuse(raft::handle_t const* handle);

  void clear_symbolic_cache();

  void store_symbolic_cache(
    mathematical_optimization::barrier::iteration_data_t<int, double>& data);

  /**
   * @brief Take ownership of barrier iteration workspace. @p data may be null (clears).
   */
  void store_iteration_data(
    mathematical_optimization::barrier::iteration_data_t<int, double>* data);

  /**
   * @brief Release ownership of cached iteration workspace; caller must delete or wrap it.
   */
  mathematical_optimization::barrier::iteration_data_t<int, double>* release_iteration_data();

  [[nodiscard]] mathematical_optimization::barrier::iteration_data_t<int, double>*
  iteration_data();

  void clear_iteration_data();

  void store_front_end_cache(std::unique_ptr<barrier_front_end_cache_t> cache);
  [[nodiscard]] barrier_front_end_cache_t* front_end_cache();
  [[nodiscard]] barrier_front_end_cache_t const* front_end_cache() const;
  void clear_front_end_cache();
  void set_c_dirty(bool dirty);
  [[nodiscard]] bool c_dirty() const;
  [[nodiscard]] bool has_front_end_cache() const;

  /**
   * Crush user-space linear objective into cached iteration_data_t.c / d_c_ and set c_dirty.
   * Requires a stored front-end cache and iteration_data from an Optimal solve.
   */
  void update_linear_objective(double const* c, int n);

 private:
  barrier_cache_t(std::unique_ptr<rmm::cuda_stream> stream,
                     std::unique_ptr<raft::handle_t> handle);

  struct impl;
  std::unique_ptr<impl> impl_;
};

}  // namespace CUOPT_EXPORT cython
}  // namespace cuopt
