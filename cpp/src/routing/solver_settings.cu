/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights
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

#include <cuopt/error.hpp>
#include <cuopt/routing/solver_settings.hpp>

namespace cuopt {
namespace routing {

template <typename i_t, typename f_t>
void solver_settings_t<i_t, f_t>::set_time_limit(f_t seconds)
{
  time_limit_ = seconds;
}

template <typename i_t, typename f_t>
void solver_settings_t<i_t, f_t>::set_verbose_mode(bool verbose)
{
  enable_verbose_mode_ = verbose;
}

template <typename i_t, typename f_t>
void solver_settings_t<i_t, f_t>::set_error_logging_mode(bool logging)
{
  log_errors_ = logging;
}

template <typename i_t, typename f_t>
void solver_settings_t<i_t, f_t>::dump_best_results(const std::string& file_path, i_t interval)
{
  dump_interval_         = interval;
  dump_best_results_     = true;
  best_result_file_name_ = file_path;
}

template <typename i_t, typename f_t>
f_t solver_settings_t<i_t, f_t>::get_time_limit() const noexcept
{
  return time_limit_;
}

template <typename i_t, typename f_t>
bool solver_settings_t<i_t, f_t>::get_verbose_mode() const noexcept
{
  return enable_verbose_mode_;
}

template <typename i_t, typename f_t>
bool solver_settings_t<i_t, f_t>::get_error_logging_mode() const noexcept
{
  return log_errors_;
}

template <typename i_t, typename f_t>
std::tuple<i_t, bool, std::string> solver_settings_t<i_t, f_t>::get_dump_best_results()
  const noexcept
{
  return std::make_tuple(dump_interval_, dump_best_results_, best_result_file_name_);
}

template class solver_settings_t<int, float>;
}  // namespace routing
}  // namespace cuopt
