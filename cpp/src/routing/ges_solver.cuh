/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#pragma once

#include "node/pdp_node.cuh"
#include "problem/problem.cuh"
#include "solution/pool_allocator.cuh"

#include <cuopt/routing/assignment.hpp>
#include <cuopt/routing/data_model_view.hpp>
#include <cuopt/routing/solver_settings.hpp>
#include <utilities/timer.hpp>

namespace cuopt {
namespace routing {

/**
 * @brief GES solver class. Currently this is not intended to be exposed to the public API
 * The diversity management can be added as another class or can be managed by this class
 *
 * @tparam i_t integer type
 * @tparam f_t float type
 */
template <typename i_t, typename f_t, request_t REQUEST>
class ges_solver_t {
 public:
  ges_solver_t(const data_model_view_t<i_t, f_t>& data_model,
               const solver_settings_t<i_t, f_t>& solver_settings,
               double time_limit,
               i_t expected_route_count         = -1,
               std::ofstream* intermediate_file = nullptr);
  // by default route count is 0, so the ges will work until the time limit is reached
  assignment_t<i_t> compute_ges_solution(std::string diversity_manager_file = "");
  assignment_t<i_t> get_ges_assignment(detail::solution_t<i_t, f_t, REQUEST>& sol,
                                       std::vector<i_t> const& accepted = {});

  // instantiate timer above anything
  timer_t timer;
  const detail::problem_t<i_t, f_t> problem;
  detail::
    pool_allocator_t<i_t, f_t, detail::solution_t<i_t, f_t, REQUEST>, detail::problem_t<i_t, f_t>>
      pool_allocator;
  i_t expected_route_count;
  // currently the interval dump time interval is ignored, adjust when we get settings struct
  // keep in mind that currently this is not thread safe, we can adjust it later
  std::ofstream* intermediate_file;
};

}  // namespace routing
}  // namespace cuopt
