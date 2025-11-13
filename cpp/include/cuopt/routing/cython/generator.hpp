/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#pragma once

#include <cuopt/routing/data_model_view.hpp>
#include <cuopt/routing/routing_structures.hpp>

#include <raft/core/handle.hpp>

#include <rmm/device_uvector.hpp>

namespace cuopt {
namespace routing {
namespace generator {

enum class dataset_distribution_t { CLUSTERED = 0, RANDOM, RANDOM_CLUSTERED };

/**
 * @brief Container for dataset parameters.
 * @note Current generator provides a number of orders equal to
 * the number of locations so only n_locations is specified.
 * @tparam i_t Integer type. Needs to be int (32bit) at the moment. Please open
 * an issue if other type are needed.
 * @tparam f_t Floating point type. Needs to be float (32bit) at the moment.
 */
template <typename i_t, typename f_t>
struct dataset_params_t {
  i_t n_locations{100};
  bool asymmetric{false};
  i_t dim{0};
  demand_i_t const* min_demand{};
  demand_i_t const* max_demand{};
  cap_i_t const* min_capacities{};
  cap_i_t const* max_capacities{};
  i_t min_service_time{0};
  i_t max_service_time{0};
  f_t tw_tightness{0.};
  f_t drop_return_trips{0.};
  i_t n_shifts{1};
  i_t n_vehicle_types{1};
  i_t n_matrix_types{1};
  i_t break_dim{0};
  dataset_distribution_t distrib{dataset_distribution_t::RANDOM};
  f_t center_box_min{};
  f_t center_box_max{n_locations / 2.f};
  i_t seed{};
};

}  // namespace generator
}  // namespace routing
}  // namespace cuopt
