/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#pragma once

#include <cuopt/routing/routing_structures.hpp>
#include <fstream>
#include <limits>
#include <ostream>

namespace cuopt {
namespace routing {

template <typename i_t, typename f_t>
class solver_settings_t {
 public:
  solver_settings_t() = default;

  /**
   * @brief Set a fixed solving time in seconds, the timer starts when `solve`
   * is called.
   *
   * @note Accuracy may be impacted. Problem under 100 locations may be solved
   * with reasonable accuracy under a second. Larger problems may need a few
   * minutes. A generous upper bond is to set the number of seconds to
   * num_locations_. By default it is set to num_locations_/5
   * by considering n_climbers vs run-time tradeoff.
   * If increased accuracy is desired, this needs to set to higher numbers.
   *
   * @param[in] seconds The number of seconds
   */
  void set_time_limit(f_t seconds);

  /**
   * @brief This is an experimental developer feature that allows displaying
   * internal information on the terminal during the solver execution.
   * @note Execution time may be impacted
   *
   * @param[in] verbose True to enable display
   */
  void set_verbose_mode(bool verbose);

  /**
   * @brief This is an experimental developer feature that allows displaying
   * constraint error information on the terminal incase of infeasible solve.
   * @note Execution time may be impacted
   *
   * @param[in] logging True to enable display
   */
  void set_error_logging_mode(bool logging);

  /**
   * @brief This is an experimental developer feature that allows displaying
   * internal best results to a given file in a csv format.
   * @note Quality of the solution might be impacted.
   *
   * @param[in] file_path Absolute path of output file.
   * @param[in] interval Dumping interval as seconds.
   */
  void dump_best_results(const std::string& file_path, i_t interval);

  /**
   * @brief Return set solving time
   * @return Solving time set in seconds
   */
  f_t get_time_limit() const noexcept;

  /**
   * @brief Return true if verbose mode is enabled
   */
  bool get_verbose_mode() const noexcept;

  /**
   * @brief Return true if error logging is enabled
   */
  bool get_error_logging_mode() const noexcept;

  /**
   * @brief Get the dump best results information
   *
   * @return std::tuple<i_t, bool, std::string>
   */
  std::tuple<i_t, bool, std::string> get_dump_best_results() const noexcept;

  bool enable_verbose_mode_{false};
  bool log_errors_{false};
  f_t time_limit_{std::numeric_limits<f_t>::max()};
  i_t dump_interval_{std::numeric_limits<i_t>::max()};
  bool dump_best_results_{false};
  std::string best_result_file_name_;
};

}  // namespace routing
}  // namespace cuopt
