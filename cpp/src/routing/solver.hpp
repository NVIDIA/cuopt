/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#pragma once

#include <cuopt/routing/assignment.hpp>
#include <cuopt/routing/data_model_view.hpp>
#include <cuopt/routing/solver_settings.hpp>

#include <routing/fleet_info.hpp>
#include <routing/hyper_params.hpp>
#include <routing/order_info.hpp>
#include <routing/routing_details.hpp>
#include <routing/structures.hpp>

#include <utilities/high_res_timer.hpp>

#include <rmm/device_scalar.hpp>

#include <limits>
#include <ostream>
namespace cuopt {
namespace routing {
/**
 * @brief A composable vehicle routing solver.
 * @tparam i_t Integer type. int (32bit) is expected at the moment. Please
 * open an issue if other type are needed.
 * @tparam f_t Floating point type. float (32bit) is expected at the moment.
 * Please open an issue if other type are needed.
 */
template <typename i_t, typename f_t>
class solver_t {
 public:
  /**
   * @brief Construct a solver_t object from the model defined in
   `data_model`. It is an error to call this method on an uninitialized
   data_model_view_t object.
   *
   * @note Use the resources of `data_model`.
   *
   * @throws cuopt::logic_error when an error occurs.
   *
   * @param[in] data_model The `data_model_view_t` to use
   */
  solver_t(data_model_view_t<i_t, f_t> const& data_model,
           solver_settings_t<i_t, f_t> const& settings);

  /**
   * @brief Solves the routing problem.
   * @note Calling solve twice is an undefined behaviour
   * @return assignment_t owning container for the solver output.
   */
  assignment_t<i_t> solve();

 protected:
  template <request_t REQUEST>
  assignment_t<i_t> run_ges_solver(i_t target_vehicles);

  raft::handle_t const* handle_ptr_{nullptr};

  solver_settings_t<i_t, f_t> settings_{};
  std::ofstream best_result_file_;

  detail::hyper_params_t hyper_params{};
  data_model_view_t<i_t, f_t> const* data_view_ptr_;
  solver_settings_t<i_t, f_t> const* solver_settings_ptr_;
};

}  // namespace routing
}  // namespace cuopt
