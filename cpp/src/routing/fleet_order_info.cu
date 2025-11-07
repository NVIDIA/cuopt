/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#include <routing/fleet_order_info.hpp>
#include <routing/utilities/check_input.hpp>

#include <cuda/std/functional>
#include <utilities/vector_helpers.cuh>

namespace cuopt {
namespace routing {
namespace detail {

template <typename i_t, typename f_t>
void populate_demand_container(data_model_view_t<i_t, f_t> const& data_model,
                               fleet_info_t<i_t, f_t>& fleet_info_,
                               order_info_t<i_t, f_t>& order_info_)
{
  auto handle_ptr_ = data_model.get_handle_ptr();
  auto stream_view = handle_ptr_->get_stream();
  i_t fleet_size   = data_model.get_fleet_size();
  auto& capacities = data_model.get_capacity_dimensions();
  // If there are no capacities provided, create one capacity with dummy entries
  if (capacities.empty()) {
    order_info_.v_demand_.resize(order_info_.get_num_orders(), handle_ptr_->get_stream());
    fleet_info_.v_capacities_.resize(fleet_size, handle_ptr_->get_stream());

    thrust::uninitialized_fill(handle_ptr_->get_thrust_policy(),
                               order_info_.v_demand_.begin(),
                               order_info_.v_demand_.end(),
                               0);

    thrust::uninitialized_fill(handle_ptr_->get_thrust_policy(),
                               fleet_info_.v_capacities_.begin(),
                               fleet_info_.v_capacities_.end(),
                               1);
  } else {
    const i_t n_capacity_dimensions = capacities.size();
    // resize the containers for having contiguous flattened arrays
    order_info_.v_demand_.resize(order_info_.get_num_orders() * n_capacity_dimensions,
                                 handle_ptr_->get_stream());
    fleet_info_.v_capacities_.resize(fleet_size * n_capacity_dimensions, handle_ptr_->get_stream());

    // identity functor for assigning i_t to uint16_t
    cuda::std::identity id;
    // populate each dimension
    for (i_t dim = 0; dim < n_capacity_dimensions; dim++) {
      // check if pick-up and delivery demands are exact negation of each other
      auto [pickup_indices, delivery_indices] = data_model.get_pickup_delivery_pair();
      if (delivery_indices != nullptr) {
        bool valid_demand = detail::check_pickup_demands(pickup_indices,
                                                         delivery_indices,
                                                         capacities[dim].get_demands(),
                                                         order_info_.get_num_requests(),
                                                         handle_ptr_->get_stream());
        cuopt_expects(valid_demand,
                      error_type_t::ValidationError,
                      "Demands of pickup and delivery pairs must be exact negation of each other");
      }
      thrust::transform(handle_ptr_->get_thrust_policy(),
                        capacities[dim].get_demands(),
                        capacities[dim].get_demands() + order_info_.get_num_orders(),
                        order_info_.v_demand_.data() + dim * order_info_.get_num_orders(),
                        id);

      thrust::transform(handle_ptr_->get_thrust_policy(),
                        capacities[dim].get_vehicle_capacities(),
                        capacities[dim].get_vehicle_capacities() + fleet_size,
                        fleet_info_.v_capacities_.data() + dim * fleet_size,
                        id);
      fleet_info_.is_homogenous_ =
        fleet_info_.is_homogenous_ &&
        all_entries_are_equal(handle_ptr_, capacities[dim].get_vehicle_capacities(), fleet_size);

      fleet_info_.is_homogenous_ =
        fleet_info_.is_homogenous_ &&
        all_entries_are_equal(
          handle_ptr_, fleet_info_.v_capacities_.data() + dim * fleet_size, fleet_size);
    }
  }
}

template void populate_demand_container(data_model_view_t<int, float> const& data_model,
                                        fleet_info_t<int, float>& fleet_info,
                                        order_info_t<int, float>& order_info);
}  // namespace detail
}  // namespace routing
}  // namespace cuopt
