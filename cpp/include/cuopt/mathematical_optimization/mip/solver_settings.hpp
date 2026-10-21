/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#pragma once

#include <optional>
#include <vector>

#include <cuopt/mathematical_optimization/constants.h>
#include <cuopt/export.hpp>
#include <cuopt/mathematical_optimization/mip/diving_hyper_params.hpp>
#include <cuopt/mathematical_optimization/mip/heuristics_hyper_params.hpp>
#include <cuopt/mathematical_optimization/mip/mip_parameters.hpp>
#include <cuopt/mathematical_optimization/mip/submip_hyper_params.hpp>
#include <cuopt/mathematical_optimization/pdlp/pdlp_hyper_params.cuh>
#include <cuopt/mathematical_optimization/pdlp/solver_settings.hpp>
#include <cuopt/mathematical_optimization/utilities/internals.hpp>

#include <cuda/stream>
#include <raft/core/device_span.hpp>
#include <rmm/device_uvector.hpp>

#include <vector>

namespace cuopt {
namespace CUOPT_EXPORT mathematical_optimization {

// Forward declare solver_settings_t for friend class
template <typename i_t, typename f_t>
class solver_settings_t;

template <typename i_t, typename f_t>
class mip_solver_settings_t;

template <typename i_t, typename f_t>
struct mip_solver_settings_accessor;

template <typename i_t, typename f_t>
class mip_solver_settings_t : public mip_parameters_t<i_t, f_t> {
 public:
  using typename mip_parameters_t<i_t, f_t>::tolerances_t;

  mip_solver_settings_t() = default;

  /**
   * @brief Set the callback for the user solution
   *
   * @param[in] callback - Callback handler for user solutions.
   * @param[in] user_data - Pointer to user-defined data forwarded to the callback.
   */
  void set_mip_callback(internals::base_solution_callback_t* callback = nullptr,
                        void* user_data                               = nullptr);

  /**
   * @brief Add an primal solution.
   *
   * @note This function can be called multiple times to add more solutions.
   *
   * @param[in] initial_solution Device or host memory pointer to a floating
   * point array of size size. cuOpt copies this data. Copy happens on the
   * stream of the raft:handler passed to the problem.
   * @param size Size of the initial_solution array.
   */
  void add_initial_solution(const f_t* initial_solution,
                            i_t size,
                            cuda::stream_ref stream = cuda::stream_ref{
                              cudaStream_t{cudaStreamDefault}});

  /**
   * @brief Get the callback for the user solution
   *
   * @return callback pointer
   */
  const std::vector<internals::base_solution_callback_t*> get_mip_callbacks() const;

  template <typename U, typename V>
  friend class problem_checking_t;
  /** Initial primal solutions */
  std::vector<std::shared_ptr<rmm::device_uvector<f_t>>> initial_solutions;

 private:
  std::vector<internals::base_solution_callback_t*> mip_callbacks_;
  std::optional<i_t> semi_continuous_original_num_variables_;
  std::vector<i_t> semi_continuous_binary_to_original_indices_;

  friend class solver_settings_t<i_t, f_t>;
  friend struct mip_solver_settings_accessor<i_t, f_t>;
};

template <typename i_t, typename f_t>
struct mip_solver_settings_accessor {
  static void clear_mip_callbacks(mip_solver_settings_t<i_t, f_t>& settings)
  {
    settings.mip_callbacks_.clear();
  }

  static void set_semi_continuous_callback_translation(mip_solver_settings_t<i_t, f_t>& settings,
                                                       i_t original_num_variables,
                                                       std::vector<i_t> binary_to_original_indices)
  {
    settings.semi_continuous_original_num_variables_     = original_num_variables;
    settings.semi_continuous_binary_to_original_indices_ = std::move(binary_to_original_indices);
  }

  static bool has_semi_continuous_callback_translation(
    const mip_solver_settings_t<i_t, f_t>& settings)
  {
    return settings.semi_continuous_original_num_variables_.has_value();
  }

  static i_t get_semi_continuous_original_num_variables(
    const mip_solver_settings_t<i_t, f_t>& settings)
  {
    return settings.semi_continuous_original_num_variables_.value_or(0);
  }

  static const std::vector<i_t>& get_semi_continuous_binary_to_original_indices(
    const mip_solver_settings_t<i_t, f_t>& settings)
  {
    return settings.semi_continuous_binary_to_original_indices_;
  }
};

}  // namespace CUOPT_EXPORT mathematical_optimization
}  // namespace cuopt
