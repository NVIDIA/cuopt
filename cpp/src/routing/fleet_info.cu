/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */
#include <routing/fleet_info.hpp>
#include <utilities/copy_helpers.hpp>
#include <utilities/vector_helpers.cuh>

#include <algorithm>
#include <cmath>

namespace cuopt {
namespace routing {
namespace detail {

template <typename i_t, typename f_t>
void populate_matrices(data_model_view_t<i_t, f_t> const& data_model, d_mdarray_t<f_t>& matrices_)
{
  auto handle_ptr_     = data_model.get_handle_ptr();
  auto stream_view_    = handle_ptr_->get_stream();
  auto nlocations      = data_model.get_num_locations();
  auto n_vehicle_types = data_model.get_num_vehicle_types();
  if (n_vehicle_types > 1 && data_model.get_vehicle_types().empty()) {
    cuopt_expects(
      false, error_type_t::ValidationError, "Set vehicle types when using multiple matrices");
  }

  // Check for consistency of cost matrices
  const auto& cost_matrices         = data_model.get_cost_matrices();
  const auto& distance_matrices     = data_model.get_distance_matrices();
  const auto& transit_time_matrices = data_model.get_transit_time_matrices();
  const auto total_tiers            = std::get<4>(data_model.get_vehicle_distance_tiers());
  const bool requires_distance = !data_model.get_vehicle_max_distances().empty() || total_tiers > 0;

  if (cost_matrices.empty()) { EXE_CUOPT_FAIL("Cost matrix (or matrices) must be specified!"); }
  cuopt_expects(!requires_distance || !distance_matrices.empty(),
                error_type_t::ValidationError,
                "A distance matrix must be set when using vehicle distance tiers or maximum "
                "distances");

  for (auto& [vtype, time_matrix] : transit_time_matrices) {
    if (!cost_matrices.count(vtype)) {
      auto msg = std::string("Cost matrix for vehicle type ") + std::to_string(vtype) +
                 std::string(" is not specified");
      execute_cuopt_fail(msg);
    }
  }

  if (n_vehicle_types > 1) {
    const auto& vtypes = data_model.get_vehicle_types();
    auto vtypes_h      = cuopt::host_copy(vtypes, stream_view_);
    for (auto& vtype : vtypes_h) {
      if (!cost_matrices.count(vtype)) {
        auto msg = std::string("Cost matrix for vehicle type ") + std::to_string(vtype) +
                   std::string(" is not specified");
        execute_cuopt_fail(msg);
      }
    }
  }

  if (!distance_matrices.empty()) {
    auto vehicle_types_map = get_unique_vehicle_types(data_model.get_vehicle_types(), stream_view_);
    for (auto const& vehicle_type_mapping : vehicle_types_map) {
      cuopt_expects(distance_matrices.count(vehicle_type_mapping.first) > 0,
                    error_type_t::ValidationError,
                    "All vehicle distance matrices should be set");
    }

    const size_t matrix_size = static_cast<size_t>(nlocations) * static_cast<size_t>(nlocations);
    for (auto const& distance_matrix_entry : distance_matrices) {
      const bool valid =
        thrust::all_of(handle_ptr_->get_thrust_policy(),
                       distance_matrix_entry.second,
                       distance_matrix_entry.second + matrix_size,
                       [] __device__(f_t value) { return value == value && value >= f_t{0}; });
      cuopt_expects(valid,
                    error_type_t::ValidationError,
                    "Distance matrix values must be non-negative and not NaN");
    }
  }

  auto n_matrix_types = detail::get_cost_matrix_type_dim<i_t, f_t>(data_model);
  matrices_ =
    detail::create_device_mdarray<f_t>(nlocations, n_vehicle_types, n_matrix_types, stream_view_);
  detail::fill_mdarray_from_data_model(matrices_, data_model);
}

template <typename i_t, typename f_t>
void populate_fleet_order_constraints(data_model_view_t<i_t, f_t> const& data_model,
                                      fleet_order_constraints_t<i_t>& fleet_order_constraints_,
                                      bool& is_homogenous)
{
  auto handle_ptr_                = data_model.get_handle_ptr();
  auto stream_view_               = handle_ptr_->get_stream();
  const i_t fleet_size            = data_model.get_fleet_size();
  const i_t n_orders              = data_model.get_num_orders();
  const auto& order_service_times = data_model.get_order_service_times();

  raft::device_span<i_t const> default_service_times{};
  raft::device_span<i_t const> service_times{};

  bool valid = true;
  if (!order_service_times.empty()) {
    valid = static_cast<int>(order_service_times.size()) == fleet_size ||
            order_service_times.find(-1) != order_service_times.end();
  }
  cuopt_expects(
    valid,
    error_type_t::ValidationError,
    "If no default service times are provided, all vehicle service times need to be provided");

  fleet_order_constraints_.resize(fleet_size, n_orders);

  if (order_service_times.find(-1) != order_service_times.end()) {
    default_service_times = order_service_times.at(-1);
  }

  if (default_service_times.empty() && order_service_times.empty()) {
    fleet_order_constraints_.fill(0);
  } else {
    for (int truck_id = 0; truck_id < fleet_size; ++truck_id) {
      if (order_service_times.find(truck_id) != order_service_times.end()) {
        service_times = order_service_times.at(truck_id);
      } else {
        service_times = default_service_times;
      }
      const auto& ref_to_check = order_service_times.find(0) != order_service_times.end()
                                   ? order_service_times.at(0)
                                   : default_service_times;
      is_homogenous            = is_homogenous && thrust::equal(handle_ptr_->get_thrust_policy(),
                                                     ref_to_check.data(),
                                                     ref_to_check.data() + service_times.size(),
                                                     service_times.data());
      raft::copy(fleet_order_constraints_.order_service_times.data() + truck_id * n_orders,
                 service_times.data(),
                 service_times.size(),
                 stream_view_);
    }
  }
}

template <typename i_t, typename f_t>
void populate_fleet_info(data_model_view_t<i_t, f_t> const& data_model,
                         detail::fleet_info_t<i_t, f_t>& fleet_info_)
{
  auto handle_ptr_   = data_model.get_handle_ptr();
  auto stream_view   = handle_ptr_->get_stream();
  auto fleet_size    = data_model.get_fleet_size();
  auto nlocations    = data_model.get_num_locations();
  bool is_homogenous = true;
  fleet_info_.resize(fleet_size, stream_view);

  if (data_model.get_vehicle_time_windows().first != nullptr) {
    raft::copy(fleet_info_.v_earliest_time_.data(),
               data_model.get_vehicle_time_windows().first,
               fleet_size,
               stream_view);

    raft::copy(fleet_info_.v_latest_time_.data(),
               data_model.get_vehicle_time_windows().second,
               fleet_size,
               stream_view);
    is_homogenous =
      is_homogenous &&
      all_entries_are_equal(handle_ptr_, fleet_info_.v_earliest_time_.data(), fleet_size);
    is_homogenous = is_homogenous && all_entries_are_equal(
                                       handle_ptr_, fleet_info_.v_latest_time_.data(), fleet_size);
  } else {
    // subtract -1 to ensure that we can set max values for service times
    // in vehicle order match
    int32_t max_time = std::numeric_limits<int32_t>::max() - 1;
    thrust::uninitialized_fill(handle_ptr_->get_thrust_policy(),
                               fleet_info_.v_earliest_time_.begin(),
                               fleet_info_.v_earliest_time_.end(),
                               0);
    thrust::uninitialized_fill(handle_ptr_->get_thrust_policy(),
                               fleet_info_.v_latest_time_.begin(),
                               fleet_info_.v_latest_time_.end(),
                               max_time);
  }

  if (auto [start_locations, return_locations] = data_model.get_vehicle_locations();
      start_locations != nullptr) {
    raft::copy(
      fleet_info_.v_start_locations_.data(), start_locations, fleet_size, stream_view.get());
    raft::copy(
      fleet_info_.v_return_locations_.data(), return_locations, fleet_size, stream_view.get());
    is_homogenous =
      is_homogenous &&
      all_entries_are_equal(handle_ptr_, fleet_info_.v_start_locations_.data(), fleet_size);
    is_homogenous =
      is_homogenous &&
      all_entries_are_equal(handle_ptr_, fleet_info_.v_return_locations_.data(), fleet_size);
  } else {
    thrust::uninitialized_fill(handle_ptr_->get_thrust_policy(),
                               fleet_info_.v_start_locations_.begin(),
                               fleet_info_.v_start_locations_.end(),
                               0);

    thrust::uninitialized_fill(handle_ptr_->get_thrust_policy(),
                               fleet_info_.v_return_locations_.begin(),
                               fleet_info_.v_return_locations_.end(),
                               0);
  }

  if (auto drop_return_trip = data_model.get_drop_return_trips(); drop_return_trip) {
    raft::copy(
      fleet_info_.v_drop_return_trip_.data(), drop_return_trip, fleet_size, stream_view.get());
    is_homogenous =
      is_homogenous &&
      all_entries_are_equal(handle_ptr_, fleet_info_.v_drop_return_trip_.data(), fleet_size);
  } else {
    thrust::uninitialized_fill(handle_ptr_->get_thrust_policy(),
                               fleet_info_.v_drop_return_trip_.begin(),
                               fleet_info_.v_drop_return_trip_.end(),
                               false);
  }

  if (auto skip_first_trip = data_model.get_skip_first_trips(); skip_first_trip) {
    raft::copy(
      fleet_info_.v_skip_first_trip_.data(), skip_first_trip, fleet_size, stream_view.get());
    is_homogenous =
      is_homogenous &&
      all_entries_are_equal(handle_ptr_, fleet_info_.v_skip_first_trip_.data(), fleet_size);
  } else {
    thrust::uninitialized_fill(handle_ptr_->get_thrust_policy(),
                               fleet_info_.v_skip_first_trip_.begin(),
                               fleet_info_.v_skip_first_trip_.end(),
                               false);
  }

  auto vehicle_types = data_model.get_vehicle_types();
  if (!vehicle_types.empty()) {
    std::vector<uint8_t> h_vehicle_types(fleet_size);
    raft::copy(h_vehicle_types.data(), vehicle_types.begin(), vehicle_types.size(), stream_view);

    auto vehicle_types_map = get_unique_vehicle_types(vehicle_types, stream_view);

    std::vector<uint8_t> renumbered_vehicle_types;
    for (auto type : h_vehicle_types) {
      cuopt_expects(data_model.get_cost_matrix(type) != nullptr,
                    error_type_t::ValidationError,
                    "All vehicle cost matrices should be set");
      renumbered_vehicle_types.push_back(vehicle_types_map.at(type));
    }

    raft::copy(
      fleet_info_.v_types_.data(), renumbered_vehicle_types.data(), fleet_size, stream_view);
    is_homogenous =
      is_homogenous && all_entries_are_equal(handle_ptr_, fleet_info_.v_types_.data(), fleet_size);
  } else {
    thrust::uninitialized_fill(handle_ptr_->get_thrust_policy(),
                               fleet_info_.v_types_.begin(),
                               fleet_info_.v_types_.end(),
                               0);
  }
  populate_matrices(data_model, fleet_info_.matrices_);
  populate_fleet_order_constraints(data_model, fleet_info_.fleet_order_constraints_, is_homogenous);
  const bool has_separate_distance_matrix = detail::has_distance_matrix<i_t, f_t>(data_model);

  // max constraints
  if (auto vehicle_max_distances = data_model.get_vehicle_max_distances();
      !vehicle_max_distances.empty()) {
    cuopt_expects(has_separate_distance_matrix,
                  error_type_t::ValidationError,
                  "vehicle_max_distances requires add_distance_matrix() to be set");
    auto host_max_distances = cuopt::host_copy(vehicle_max_distances, stream_view);
    const bool valid =
      std::all_of(host_max_distances.begin(), host_max_distances.end(), [](f_t value) {
        return std::isfinite(value) && value >= f_t{0};
      });
    cuopt_expects(valid,
                  error_type_t::ValidationError,
                  "Vehicle maximum distances must be finite and non-negative");
    fleet_info_.v_max_distances_.resize(fleet_size, stream_view);
    raft::copy(
      fleet_info_.v_max_distances_.data(), vehicle_max_distances.data(), fleet_size, stream_view);
    is_homogenous =
      is_homogenous &&
      all_entries_are_equal(handle_ptr_, fleet_info_.v_max_distances_.data(), fleet_size);
  }

  if (auto vehicle_max_costs = data_model.get_vehicle_max_costs(); !vehicle_max_costs.empty()) {
    auto host_max_costs = cuopt::host_copy(vehicle_max_costs, stream_view);
    const bool valid    = std::all_of(host_max_costs.begin(), host_max_costs.end(), [](f_t value) {
      return std::isfinite(value) && value >= f_t{0};
    });
    cuopt_expects(valid,
                  error_type_t::ValidationError,
                  "Vehicle maximum costs must be finite and non-negative");
    fleet_info_.v_max_costs_.resize(fleet_size, stream_view);
    raft::copy(fleet_info_.v_max_costs_.data(), vehicle_max_costs.data(), fleet_size, stream_view);
    is_homogenous = is_homogenous &&
                    all_entries_are_equal(handle_ptr_, fleet_info_.v_max_costs_.data(), fleet_size);
  }

  if (auto vehicle_max_times = data_model.get_vehicle_max_times(); !vehicle_max_times.empty()) {
    fleet_info_.v_max_times_.resize(fleet_size, stream_view);
    raft::copy(fleet_info_.v_max_times_.data(), vehicle_max_times.data(), fleet_size, stream_view);
    is_homogenous = is_homogenous &&
                    all_entries_are_equal(handle_ptr_, fleet_info_.v_max_times_.data(), fleet_size);
  }

  if (auto vehicle_fixed_costs = data_model.get_vehicle_fixed_costs();
      !vehicle_fixed_costs.empty()) {
    raft::copy(
      fleet_info_.v_fixed_costs_.data(), vehicle_fixed_costs.data(), fleet_size, stream_view);
    is_homogenous = is_homogenous && all_entries_are_equal(
                                       handle_ptr_, fleet_info_.v_fixed_costs_.data(), fleet_size);
  } else {
    thrust::uninitialized_fill(handle_ptr_->get_thrust_policy(),
                               fleet_info_.v_fixed_costs_.begin(),
                               fleet_info_.v_fixed_costs_.end(),
                               -1.f);
  }

  // Copy distance tiers from data_model to fleet_info
  auto [thresholds, fixed_costs, costs_per_unit, tier_offsets, total_tiers] =
    data_model.get_vehicle_distance_tiers();

  if (thresholds != nullptr && total_tiers > 0) {
    cuopt_expects(has_separate_distance_matrix,
                  error_type_t::ValidationError,
                  "vehicle_distance_tiers requires add_distance_matrix() to be set");
    // Resize and copy the flattened tiers data
    fleet_info_.v_distance_tiers_.resize(total_tiers, stream_view);
    fleet_info_.v_tier_offsets_.resize(fleet_size + 1, stream_view);

    std::vector<i_t> h_tier_offsets(fleet_size + 1);

    // Copy tier offsets
    raft::copy(fleet_info_.v_tier_offsets_.data(), tier_offsets, fleet_size + 1, stream_view);
    raft::copy(h_tier_offsets.data(), tier_offsets, fleet_size + 1, stream_view);

    // Copy tier data (thresholds, fixed_costs, costs_per_unit) into distance_tier_t structs
    std::vector<distance_tier_t<f_t>> h_tiers(total_tiers);
    std::vector<f_t> h_thresholds(total_tiers);
    std::vector<f_t> h_fixed_costs(total_tiers);
    std::vector<f_t> h_costs_per_unit(total_tiers);

    raft::copy(h_thresholds.data(), thresholds, total_tiers, stream_view);
    raft::copy(h_fixed_costs.data(), fixed_costs, total_tiers, stream_view);
    raft::copy(h_costs_per_unit.data(), costs_per_unit, total_tiers, stream_view);
    handle_ptr_->sync_stream();

    cuopt_expects(h_tier_offsets.front() == 0 && h_tier_offsets.back() == total_tiers &&
                    std::is_sorted(h_tier_offsets.begin(), h_tier_offsets.end()),
                  error_type_t::ValidationError,
                  "Invalid distance tier offsets");
    for (i_t vehicle_id = 0; vehicle_id < fleet_size; ++vehicle_id) {
      const auto tier_begin = h_tier_offsets[vehicle_id];
      const auto tier_end   = h_tier_offsets[vehicle_id + 1];
      cuopt_expects(tier_begin < tier_end,
                    error_type_t::ValidationError,
                    "Each vehicle must have at least one distance tier");
      for (i_t tier = tier_begin; tier < tier_end; ++tier) {
        cuopt_expects(std::isfinite(h_thresholds[tier]) && h_thresholds[tier] >= 0.f &&
                        std::isfinite(h_fixed_costs[tier]) && h_fixed_costs[tier] >= 0.f &&
                        std::isfinite(h_costs_per_unit[tier]) && h_costs_per_unit[tier] >= 0.f,
                      error_type_t::ValidationError,
                      "Distance tier values must be finite and non-negative");
        if (tier > tier_begin) {
          cuopt_expects(h_thresholds[tier - 1] < h_thresholds[tier],
                        error_type_t::ValidationError,
                        "Distance tier thresholds must be strictly increasing");
        }
      }
      cuopt_expects(h_thresholds[tier_end - 1] == std::numeric_limits<f_t>::max(),
                    error_type_t::ValidationError,
                    "The last distance tier threshold for each vehicle must be float32 max");
    }

    // Pack into distance_tier_t structs
    for (i_t i = 0; i < total_tiers; ++i) {
      h_tiers[i].threshold     = h_thresholds[i];
      h_tiers[i].fixed_cost    = h_fixed_costs[i];
      h_tiers[i].cost_per_unit = h_costs_per_unit[i];
    }

    for (i_t vehicle_id = 1; vehicle_id < fleet_size && is_homogenous; ++vehicle_id) {
      const auto first_begin = h_tier_offsets[0];
      const auto first_end   = h_tier_offsets[1];
      const auto tier_begin  = h_tier_offsets[vehicle_id];
      const auto tier_end    = h_tier_offsets[vehicle_id + 1];
      is_homogenous =
        first_end - first_begin == tier_end - tier_begin &&
        std::equal(
          h_tiers.begin() + first_begin, h_tiers.begin() + first_end, h_tiers.begin() + tier_begin);
    }

    raft::copy(fleet_info_.v_distance_tiers_.data(), h_tiers.data(), total_tiers, stream_view);
  }

  fleet_info_.is_homogenous_ = is_homogenous;
}

template <typename i_t, typename f_t>
void populate_vehicle_infos(data_model_view_t<i_t, f_t> const& data_model,
                            detail::fleet_info_t<i_t, f_t>& fleet_info)
{
  auto handle_ptr_ = data_model.get_handle_ptr();
  auto stream_view = handle_ptr_->get_stream();
  auto fleet_size  = data_model.get_fleet_size();
  // populate vehicle info
  std::vector<VehicleInfo<f_t>> h_vehicle_infos(fleet_size);
  for (i_t id = 0; id < fleet_size; ++id) {
    h_vehicle_infos[id] = fleet_info.create_vehicle_info(id);
  }
  raft::copy(fleet_info.v_vehicle_infos_.data(), h_vehicle_infos.data(), fleet_size, stream_view);
  handle_ptr_->sync_stream();
}

template void populate_fleet_info(data_model_view_t<int, float> const& data_model,
                                  detail::fleet_info_t<int, float>& fleet_info);

template void populate_vehicle_infos(data_model_view_t<int, float> const& data_model,
                                     detail::fleet_info_t<int, float>& fleet_info);
}  // namespace detail
}  // namespace routing
}  // namespace cuopt
