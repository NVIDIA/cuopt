/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2025 NVIDIA CORPORATION & AFFILIATES. All rights
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

#include <utilities/vector_helpers.cuh>
#include "routing/fleet_order_info.hpp"
#include "routing/ges_solver.cuh"
#include "routing/solver.hpp"
#include "routing/utilities/cuopt_utils.cuh"
#include "routing/utilities/env_utils.hpp"

#include <cuopt/error.hpp>
#include <routing/structures.hpp>
#include <routing/utilities/check_input.hpp>
#include <utilities/copy_helpers.hpp>
#include <utilities/high_res_timer.hpp>
#include <utilities/vector_helpers.cuh>

#include <raft/util/cudart_utils.hpp>

#include <thrust/count.h>
#include <thrust/fill.h>
#include <thrust/find.h>
#include <thrust/iterator/discard_iterator.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/scatter.h>
#include <thrust/sort.h>
#include <thrust/transform.h>
#include <thrust/tuple.h>
#include <thrust/unique.h>
#include <chrono>
#include <limits>
#include <numeric>

namespace cuopt {
namespace routing {

template <typename i_t, typename f_t>
solver_t<i_t, f_t>::solver_t(data_model_view_t<i_t, f_t> const& data_model,
                             solver_settings_t<i_t, f_t> const& settings)
  : handle_ptr_(data_model.get_handle_ptr()), settings_(settings)
{
  auto n_matrix_types = detail::get_cost_matrix_type_dim<i_t, f_t>(data_model);
  if (n_matrix_types == 1 && !data_model.get_vehicle_max_times().empty()) {
    cuopt_expects(false,
                  error_type_t::ValidationError,
                  "Time matrix should be set in order to use vehicle max time constraints");
  }

  data_view_ptr_       = &data_model;
  solver_settings_ptr_ = &settings;
}

template <typename i_t, typename f_t>
assignment_t<i_t> solver_t<i_t, f_t>::solve()
{
  if (settings_.dump_best_results_) { best_result_file_.open(settings_.best_result_file_name_); }
  if (settings_.time_limit_ == std::numeric_limits<f_t>::max()) {
    // order_info_ is populated in ges_solver_t constructor, so use data_model here
    settings_.time_limit_ = data_view_ptr_->get_num_orders() / 5;
  }
  // TODO accept a settings object once we have full feature in ges solver
  // We only set target vehicles and use fixed route loop in the below case. The other paths will
  // run regular fixed route loop.
  auto target_vehicles = -1;
  if (data_view_ptr_->get_fleet_size() == data_view_ptr_->get_min_vehicles()) {
    target_vehicles = data_view_ptr_->get_min_vehicles();
  }

  const bool is_pdp = data_view_ptr_->get_pickup_delivery_pair().first != nullptr;

  if (is_pdp) { return run_ges_solver<request_t::PDP>(target_vehicles); }
  return run_ges_solver<request_t::VRP>(target_vehicles);
}

template <typename i_t, typename f_t>
template <request_t REQUEST>
assignment_t<i_t> solver_t<i_t, f_t>::run_ges_solver(i_t target_vehicles)
{
  ges_solver_t<i_t, f_t, REQUEST> s{*data_view_ptr_,
                                    *solver_settings_ptr_,
                                    (double)settings_.time_limit_,
                                    target_vehicles,
                                    &best_result_file_};
  auto a = s.compute_ges_solution(settings_.best_result_file_name_);
  if (settings_.dump_best_results_) { best_result_file_.close(); }
  return a;
}

template class solver_t<int, float>;

}  // namespace routing
}  // namespace cuopt
