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

#include <cuopt/logger.hpp>
#include <cuopt/routing/solve.hpp>
#include <routing/solver.hpp>

namespace cuopt {
namespace routing {
template <typename i_t, typename f_t>
assignment_t<i_t> solve(data_model_view_t<i_t, f_t> const& data_model,
                        solver_settings_t<i_t, f_t> const& settings)
{
  try {
    cuopt::routing::solver_t<i_t, f_t> solver(data_model, settings);
    return solver.solve();
  } catch (const cuopt::logic_error& e) {
    CUOPT_LOG_ERROR("Error in solve: %s", e.what());
    return assignment_t<i_t>(e, data_model.get_handle_ptr()->get_stream());
  } catch (const std::bad_alloc& e) {
    CUOPT_LOG_ERROR("Error in solve: %s", e.what());
    return assignment_t<i_t>(
      cuopt::logic_error("Memory allocation failed", cuopt::error_type_t::RuntimeError),
      data_model.get_handle_ptr()->get_stream());
  }
}

template assignment_t<int> solve(data_model_view_t<int, float> const& data_model,
                                 solver_settings_t<int, float> const& settings);
}  // namespace routing
}  // namespace cuopt
