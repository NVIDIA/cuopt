/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */
#include <routing/fleet_info.hpp>
#include <utilities/copy_helpers.hpp>
#include <utilities/vector_helpers.cuh>

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
  const auto& transit_time_matrices = data_model.get_transit_time_matrices();

  if (cost_matrices.empty()) { EXE_CUOPT_FAIL("Cost matrix (or matrices) must be specified!"); }

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
      fleet_info_.v_start_locations_.data(), start_locations, fleet_size, stream_view.value());
    raft::copy(
      fleet_info_.v_return_locations_.data(), return_locations, fleet_size, stream_view.value());
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
      fleet_info_.v_drop_return_trip_.data(), drop_return_trip, fleet_size, stream_view.value());
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
      fleet_info_.v_skip_first_trip_.data(), skip_first_trip, fleet_size, stream_view.value());
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

  // max constraints
  if (auto vehicle_max_costs = data_model.get_vehicle_max_costs(); !vehicle_max_costs.empty()) {
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
