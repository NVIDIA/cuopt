/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
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

#pragma once

#include <cuopt/linear_programming/optimization_problem.hpp>

namespace cuopt::linear_programming::detail {

template <typename i_t, typename f_t>
class third_party_presolve_t {
 public:
  third_party_presolve_t() = default;

  std::pair<optimization_problem_t<i_t, f_t>, bool> apply(
    optimization_problem_t<i_t, f_t> const& op_problem,
    problem_category_t category,
    bool dual_postsolve,
    f_t absolute_tolerance,
    f_t relative_tolerance,
    double time_limit,
    i_t num_cpu_threads = 0);

  void undo(rmm::device_uvector<f_t>& primal_solution,
            rmm::device_uvector<f_t>& dual_solution,
            rmm::device_uvector<f_t>& reduced_costs,
            problem_category_t category,
            bool status_to_skip,
            rmm::cuda_stream_view stream_view);
};

}  // namespace cuopt::linear_programming::detail
