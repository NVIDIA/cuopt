/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#pragma once

#include <routing/routing_details.hpp>
#include <routing/utilities/md_utils.hpp>
#include "fleet_order_constraints.hpp"
#include "utilities/strided_span.hpp"

#include "vehicle_info.hpp"

namespace cuopt {
namespace routing {
namespace detail {

template <typename i_t, typename f_t>
class fleet_info_t {
 public:
  fleet_info_t(raft::handle_t const* handle_ptr, i_t num_vehicles)
    : handle_ptr_(handle_ptr),
      v_break_offset_(0, handle_ptr_->get_stream()),
      v_break_duration_(0, handle_ptr_->get_stream()),
      v_break_earliest_(0, handle_ptr_->get_stream()),
      v_break_latest_(0, handle_ptr_->get_stream()),
      v_earliest_time_(num_vehicles, handle_ptr_->get_stream()),
      v_latest_time_(num_vehicles, handle_ptr_->get_stream()),
      v_types_(num_vehicles, handle_ptr_->get_stream()),
      v_start_locations_(num_vehicles, handle_ptr_->get_stream()),
      v_return_locations_(num_vehicles, handle_ptr_->get_stream()),
      v_drop_return_trip_(num_vehicles, handle_ptr_->get_stream()),
      v_skip_first_trip_(num_vehicles, handle_ptr_->get_stream()),
      v_capacities_(num_vehicles, handle_ptr_->get_stream()),
      v_vehicle_infos_(num_vehicles, handle_ptr_->get_stream()),
      matrices_(handle_ptr_->get_stream()),
      fleet_order_constraints_(handle_ptr, 0, 0),
      v_max_costs_(0, handle_ptr_->get_stream()),
      v_max_times_(0, handle_ptr_->get_stream()),
      v_fixed_costs_(0, handle_ptr_->get_stream()),
      v_buckets_(0, handle_ptr_->get_stream()),
      v_vehicle_availability_(0, handle_ptr_->get_stream()),
      is_homogenous_(true)
  {
  }

  i_t fleet_size() const { return (i_t)v_earliest_time_.size(); }

  auto constexpr get_num_vehicles() const { return v_earliest_time_.size(); }

  constexpr bool has_time_matrix() const { return matrices_.extent[1] > 1; }

  constexpr bool is_homogenous() const { return is_homogenous_; }

  void resize(i_t size, rmm::cuda_stream_view stream)
  {
    v_earliest_time_.resize(size, stream);
    v_latest_time_.resize(size, stream);
    v_types_.resize(size, stream);
    v_drop_return_trip_.resize(size, stream);
    v_skip_first_trip_.resize(size, stream);
    v_start_locations_.resize(size, stream);
    v_return_locations_.resize(size, stream);
    v_capacities_.resize(size, stream);
    v_vehicle_infos_.resize(size, stream);
    v_fixed_costs_.resize(size, stream);
    v_buckets_.resize(size, stream);
  }

  auto to_host()
  {
    host_t h;
    h.break_offset            = host_copy(v_break_offset_);
    h.break_durations         = host_copy(v_break_duration_);
    h.break_earliest          = host_copy(v_break_earliest_);
    h.break_latest            = host_copy(v_break_latest_);
    h.earliest_time           = host_copy(v_earliest_time_);
    h.latest_time             = host_copy(v_latest_time_);
    h.start_locations         = host_copy(v_start_locations_);
    h.return_locations        = host_copy(v_return_locations_);
    h.drop_return_trip        = host_copy(v_drop_return_trip_);
    h.skip_first_trip         = host_copy(v_skip_first_trip_);
    h.capacities              = host_copy(v_capacities_);
    h.max_costs               = host_copy(v_max_costs_);
    h.max_times               = host_copy(v_max_times_);
    h.fixed_costs             = host_copy(v_fixed_costs_);
    h.fleet_order_constraints = fleet_order_constraints_.to_host();
    h.types                   = host_copy(v_types_);
    h.buckets                 = host_copy(v_buckets_);
    h.matrices                = detail::create_host_mdarray<f_t>(
      matrices_.extent[2], matrices_.extent[0], matrices_.extent[1]);
    raft::copy(h.matrices.buffer.data(),
               matrices_.buffer.data(),
               matrices_.buffer.size(),
               matrices_.buffer.stream());
    return h;
  }

  struct host_t {
    static constexpr bool is_device = false;

    constexpr raft::span<i_t const, is_device> get_break_vector(i_t truck_id,
                                                                const std::vector<i_t>& vec) const
    {
      if (!vec.empty()) {
        i_t offset    = break_offset[truck_id];
        i_t break_dim = break_offset[truck_id + 1] - offset;
        return raft::span<i_t const, is_device>(vec.data() + offset, break_dim);
      } else {
        return raft::span<i_t const, is_device>();
      }
    }

    constexpr VehicleInfo<f_t, is_device> get_vehicle_info(const i_t vehicle_id) const
    {
      size_t num_vehicles = earliest_time.size();
      cuopt_assert(vehicle_id < (i_t)num_vehicles && vehicle_id >= 0,
                   "Vehicle id should be in the range!");

      VehicleInfo<f_t, is_device> info;
      info.drop_return_trip = drop_return_trip[vehicle_id];
      info.skip_first_trip  = skip_first_trip[vehicle_id];
      info.type             = types[vehicle_id];

      if (!max_costs.empty()) { info.max_cost = max_costs[vehicle_id]; }

      if (!max_times.empty()) { info.max_time = max_times[vehicle_id]; }
      info.fixed_cost          = fixed_costs[vehicle_id];
      info.matrices            = matrices.view();
      info.order_service_times = fleet_order_constraints.get_order_service_times(vehicle_id);
      info.order_match         = fleet_order_constraints.get_order_match(vehicle_id);

      size_t stride = num_vehicles;
      i_t n_cap_dim = capacities.size() / num_vehicles;
      info.capacities =
        cuopt::strided_span<const cap_i_t>(capacities.data() + vehicle_id, stride, n_cap_dim);
      info.break_durations = get_break_vector(vehicle_id, break_durations);
      info.break_earliest  = get_break_vector(vehicle_id, break_earliest);
      info.break_latest    = get_break_vector(vehicle_id, break_latest);
      info.earliest        = earliest_time[vehicle_id];
      info.latest          = latest_time[vehicle_id];
      info.start           = start_locations[vehicle_id];
      info.end             = return_locations[vehicle_id];
      return info;
    }

    std::vector<i_t> break_offset;
    std::vector<i_t> break_durations;
    std::vector<i_t> break_earliest;
    std::vector<i_t> break_latest;
    std::vector<i_t> earliest_time;
    std::vector<i_t> latest_time;
    std::vector<cap_i_t> capacities;
    std::vector<i_t> start_locations;
    std::vector<i_t> return_locations;
    std::vector<uint8_t> types;
    std::vector<i_t> buckets;
    typename fleet_order_constraints_t<i_t>::host_t fleet_order_constraints;
    std::vector<bool> drop_return_trip;
    std::vector<bool> skip_first_trip;
    std::vector<f_t> max_costs;
    std::vector<f_t> max_times;
    std::vector<f_t> fixed_costs;
    std::vector<i_t> vehicle_availability;
    h_mdarray_t<f_t> matrices;
  };

  struct view_t {
    constexpr VehicleInfo<f_t> get_vehicle_info(const i_t vehicle_id) const
    {
      cuopt_assert(vehicle_id < num_vehicles && vehicle_id >= 0,
                   "Vehicle id should be in the range!");
      return vehicle_infos[vehicle_id];
    }

    constexpr i_t get_num_vehicles() const { return num_vehicles; }

    constexpr i_t is_homogenous_fleet() const { return is_homogenous; }

    constexpr bool has_time_matrix() const { return matrices.extent[1] > 1; }
    i_t num_vehicles = 0;
    mdarray_view_t<f_t> matrices{};
    const i_t* break_offset{nullptr};
    const i_t* break_durations{nullptr};
    const i_t* break_earliest{nullptr};
    const i_t* break_latest{nullptr};
    const i_t* earliest_time{nullptr};
    const i_t* latest_time{nullptr};
    const uint8_t* types{nullptr};
    const i_t* start_locations{nullptr};
    const i_t* return_locations{nullptr};
    raft::device_span<const cap_i_t> capacities{};
    raft::device_span<const VehicleInfo<f_t>> vehicle_infos{};
    typename fleet_order_constraints_t<i_t>::view_t fleet_order_constraints;

    const bool* drop_return_trip{nullptr};
    const bool* skip_first_trip{nullptr};

    raft::device_span<const f_t> max_costs{};
    raft::device_span<const f_t> max_times{};
    raft::device_span<const f_t> fixed_costs{};
    raft::device_span<const i_t> buckets{};
    raft::device_span<const i_t> vehicle_availability{};
    bool is_homogenous{true};
  };

  view_t view() const
  {
    view_t v;
    v.num_vehicles     = v_earliest_time_.size();
    v.matrices         = matrices_.view();
    v.break_offset     = v_break_offset_.data();
    v.break_durations  = v_break_duration_.data();
    v.break_earliest   = v_break_earliest_.data();
    v.break_latest     = v_break_latest_.data();
    v.earliest_time    = v_earliest_time_.data();
    v.latest_time      = v_latest_time_.data();
    v.types            = v_types_.data();
    v.start_locations  = v_start_locations_.data();
    v.return_locations = v_return_locations_.data();
    v.drop_return_trip = v_drop_return_trip_.data();
    v.skip_first_trip  = v_skip_first_trip_.data();
    v.capacities = raft::device_span<const cap_i_t>(v_capacities_.data(), v_capacities_.size());
    v.vehicle_infos =
      raft::device_span<const VehicleInfo<f_t>>(v_vehicle_infos_.data(), v_vehicle_infos_.size());
    v.fleet_order_constraints = fleet_order_constraints_.view();

    v.max_costs   = raft::device_span<const f_t>(v_max_costs_.data(), v_max_costs_.size());
    v.max_times   = raft::device_span<const f_t>(v_max_times_.data(), v_max_times_.size());
    v.fixed_costs = raft::device_span<const f_t>(v_fixed_costs_.data(), v_fixed_costs_.size());
    v.buckets     = raft::device_span<const i_t>(v_buckets_.data(), v_buckets_.size());
    v.vehicle_availability =
      raft::device_span<const i_t>(v_vehicle_availability_.data(), v_vehicle_availability_.size());
    v.is_homogenous = is_homogenous_;
    return v;
  }

  constexpr raft::device_span<i_t const> get_break_vector(i_t truck_id,
                                                          const rmm::device_uvector<i_t>& vec,
                                                          rmm::cuda_stream_view stream) const
  {
    if (!vec.is_empty()) {
      i_t offset    = v_break_offset_.element(truck_id, stream);
      i_t break_dim = v_break_offset_.element(truck_id + 1, stream) - offset;
      return raft::device_span<i_t const>(vec.data() + offset, break_dim);
    } else {
      return raft::device_span<i_t const>();
    }
  }

  constexpr VehicleInfo<f_t> get_vehicle_info(const i_t vehicle_id,
                                              rmm::cuda_stream_view stream) const
  {
    return v_vehicle_infos_.element(vehicle_id, stream);
  }

  constexpr VehicleInfo<f_t> create_vehicle_info(const i_t vehicle_id) const
  {
    size_t num_vehicles = v_earliest_time_.size();
    cuopt_assert(vehicle_id < (i_t)num_vehicles && vehicle_id >= 0,
                 "Vehicle id should be in the range!");

    VehicleInfo<f_t> info;
    info.drop_return_trip = v_drop_return_trip_.element(vehicle_id, handle_ptr_->get_stream());
    info.skip_first_trip  = v_skip_first_trip_.element(vehicle_id, handle_ptr_->get_stream());
    info.type             = v_types_.element(vehicle_id, handle_ptr_->get_stream());

    if (!v_max_costs_.is_empty()) {
      info.max_cost = v_max_costs_.element(vehicle_id, handle_ptr_->get_stream());
    }

    if (!v_max_times_.is_empty()) {
      info.max_time = v_max_times_.element(vehicle_id, handle_ptr_->get_stream());
    }
    info.fixed_cost          = v_fixed_costs_.element(vehicle_id, handle_ptr_->get_stream());
    info.matrices            = matrices_.view();
    info.order_service_times = fleet_order_constraints_.get_order_service_times(vehicle_id);
    info.order_match         = fleet_order_constraints_.get_order_match(vehicle_id);

    size_t stride = num_vehicles;
    i_t n_cap_dim = v_capacities_.size() / num_vehicles;
    info.capacities =
      cuopt::strided_span<const cap_i_t>(v_capacities_.data() + vehicle_id, stride, n_cap_dim);
    info.break_durations =
      get_break_vector(vehicle_id, v_break_duration_, handle_ptr_->get_stream());
    info.break_earliest =
      get_break_vector(vehicle_id, v_break_earliest_, handle_ptr_->get_stream());
    info.break_latest = get_break_vector(vehicle_id, v_break_latest_, handle_ptr_->get_stream());
    info.earliest     = v_earliest_time_.element(vehicle_id, handle_ptr_->get_stream());
    info.latest       = v_latest_time_.element(vehicle_id, handle_ptr_->get_stream());
    info.start        = v_start_locations_.element(vehicle_id, handle_ptr_->get_stream());
    info.end          = v_return_locations_.element(vehicle_id, handle_ptr_->get_stream());
    return info;
  }

  raft::handle_t const* handle_ptr_{nullptr};
  d_mdarray_t<f_t> matrices_;
  rmm::device_uvector<i_t> v_break_offset_;
  rmm::device_uvector<i_t> v_break_duration_;
  rmm::device_uvector<i_t> v_break_earliest_;
  rmm::device_uvector<i_t> v_break_latest_;
  rmm::device_uvector<i_t> v_earliest_time_;
  rmm::device_uvector<i_t> v_latest_time_;
  rmm::device_uvector<cap_i_t> v_capacities_;
  rmm::device_uvector<VehicleInfo<f_t>> v_vehicle_infos_;
  rmm::device_uvector<uint8_t> v_types_;
  rmm::device_uvector<i_t> v_start_locations_;
  rmm::device_uvector<i_t> v_return_locations_;
  rmm::device_uvector<bool> v_drop_return_trip_;
  rmm::device_uvector<bool> v_skip_first_trip_;
  fleet_order_constraints_t<i_t> fleet_order_constraints_;
  rmm::device_uvector<f_t> v_max_costs_;
  rmm::device_uvector<f_t> v_max_times_;
  rmm::device_uvector<f_t> v_fixed_costs_;
  rmm::device_uvector<i_t> v_buckets_;
  rmm::device_uvector<i_t> v_vehicle_availability_;
  bool is_homogenous_;
};

/**
 * @brief helper function to construct the fleet_info object from data model
 *
 * @tparam i_t
 * @tparam f_t
 * @param data_model
 * @param fleet_info
 */
template <typename i_t, typename f_t>
void populate_fleet_info(data_model_view_t<i_t, f_t> const& data_model,
                         detail::fleet_info_t<i_t, f_t>& fleet_info);

/**
 * @brief helper function to construct the vehicle_infos of fleet_info object
 *
 * @tparam i_t
 * @tparam f_t
 * @param data_model
 * @param fleet_info
 */
template <typename i_t, typename f_t>
void populate_vehicle_infos(data_model_view_t<i_t, f_t> const& data_model,
                            detail::fleet_info_t<i_t, f_t>& fleet_info);
}  // namespace detail
}  // namespace routing
}  // namespace cuopt
