/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#include <routing/order_info.hpp>
#include <routing/utilities/check_input.hpp>

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

namespace cuopt {
namespace routing {
namespace detail {

template <typename i_t, typename f_t>
void populate_time_windows(data_model_view_t<i_t, f_t> const& data_model,
                           detail::order_info_t<i_t, f_t>& order_info_)
{
  auto handle_ptr_ = data_model.get_handle_ptr();
  auto stream_view = handle_ptr_->get_stream();
  order_info_.v_earliest_time_.resize(order_info_.get_num_orders(), stream_view);
  order_info_.v_latest_time_.resize(order_info_.get_num_orders(), stream_view);
  auto [earliest, latest] = data_model.get_order_time_windows();
  if (earliest) {
    raft::copy(order_info_.v_earliest_time_.data(),
               earliest,
               order_info_.get_num_orders(),
               stream_view.value());
    raft::copy(
      order_info_.v_latest_time_.data(), latest, order_info_.get_num_orders(), stream_view.value());
  } else {
    // subtract -1 to ensure that we can set max values for service times
    // in vehicle order match
    int32_t max_time = std::numeric_limits<int32_t>::max() - 1;
    thrust::uninitialized_fill(handle_ptr_->get_thrust_policy(),
                               order_info_.v_earliest_time_.begin(),
                               order_info_.v_earliest_time_.end(),
                               0);
    thrust::uninitialized_fill(handle_ptr_->get_thrust_policy(),
                               order_info_.v_latest_time_.begin(),
                               order_info_.v_latest_time_.end(),
                               max_time);
  }
  // check if latest time of deliveries is earlier than earliest time of pickups
  auto [pickup_indices, delivery_indices] = data_model.get_pickup_delivery_pair();
  if (delivery_indices != nullptr) {
    bool valid_tw = detail::check_pickup_tw(pickup_indices,
                                            delivery_indices,
                                            order_info_.v_earliest_time_.data(),
                                            order_info_.v_latest_time_.data(),
                                            order_info_.get_num_requests(),
                                            stream_view);
    cuopt_expects(valid_tw,
                  error_type_t::ValidationError,
                  "Latest time of a delivery cannot be earlier than earliest time of the pickup");
  }
}

template <typename i_t, typename f_t>
void check_pickup_delivery_pairs(data_model_view_t<i_t, f_t> const& data_model)
{
  auto [pickup_indices, delivery_indices] = data_model.get_pickup_delivery_pair();
  if (!pickup_indices) return;

  auto handle_ptr      = data_model.get_handle_ptr();
  auto order_locations = data_model.get_order_locations();
  auto n_orders        = data_model.get_num_orders();

  bool depot_included = order_locations == nullptr;

  cuopt_expects(n_orders % 2 == (int)depot_included,
                error_type_t::ValidationError,
                "There should be even number of destination nodes for pickup and deliveries");
  auto n_requests = (n_orders - (int)depot_included) / 2;
  cuopt_expects(detail::check_min_max_values(pickup_indices,
                                             n_requests,
                                             std::numeric_limits<uint16_t>::min(),
                                             std::numeric_limits<uint16_t>::max(),
                                             handle_ptr->get_stream()),
                error_type_t::ValidationError,
                "Pickup indices must be between 0 and max uint16_t!");
  cuopt_expects(detail::check_min_max_values(delivery_indices,
                                             n_requests,
                                             std::numeric_limits<uint16_t>::min(),
                                             std::numeric_limits<uint16_t>::max(),
                                             handle_ptr->get_stream()),
                error_type_t::ValidationError,
                "Delivery indices must be between 0 and max uint16_t!");
}

template <typename i_t, typename f_t>
void check_depot_times(data_model_view_t<i_t, f_t> const& data_model)
{
  auto order_locations = data_model.get_order_locations();
  if (order_locations) return;

  auto [earliest, latest] = data_model.get_order_time_windows();
  if (!earliest) return;

  auto handle_ptr = data_model.get_handle_ptr();
  auto n_orders   = data_model.get_num_orders();
  i_t depot_earliest, depot_latest;
  raft::copy(&depot_earliest, earliest, 1, handle_ptr->get_stream());
  raft::copy(&depot_latest, latest, 1, handle_ptr->get_stream());
  RAFT_CUDA_TRY(cudaStreamSynchronize(handle_ptr->get_stream()));

  rmm::device_uvector<i_t> v_latest_time(n_orders, handle_ptr->get_stream());
  rmm::device_uvector<i_t> v_earliest_time(n_orders, handle_ptr->get_stream());

  raft::copy(v_latest_time.data(), latest, n_orders, handle_ptr->get_stream());
  raft::copy(v_earliest_time.data(), earliest, n_orders, handle_ptr->get_stream());

  cuopt_expects(
    detail::check_min_latest_with_depot(v_latest_time, depot_earliest, handle_ptr->get_stream()),
    error_type_t::ValidationError,
    "Minimum of latest arrival times should be bigger than "
    "depot's earliest arrival time!");
  cuopt_expects(
    detail::check_max_earliest_with_depot(v_earliest_time, depot_latest, handle_ptr->get_stream()),
    error_type_t::ValidationError,
    "Maximum of earliest arrival times should be smaller than "
    "depot's latest arrival time!");
  cuopt_expects(
    detail::check_earliest_with_latest(v_earliest_time, v_latest_time, handle_ptr->get_stream()),
    error_type_t::ValidationError,
    "Earliest time should be smaller than "
    "latest time!");
}

template <typename i_t, typename f_t>
void populate_order_info(data_model_view_t<i_t, f_t> const& data_model,
                         detail::order_info_t<i_t, f_t>& order_info_)
{
  check_pickup_delivery_pairs(data_model);
  check_depot_times(data_model);

  auto order_locations         = data_model.get_order_locations();
  auto nlocations              = data_model.get_num_locations();
  auto norders                 = data_model.get_num_orders();
  auto pickup_delivery_indices = data_model.get_pickup_delivery_pair();
  auto handle_ptr_             = data_model.get_handle_ptr();
  auto stream                  = handle_ptr_->get_stream();
  auto stream_view             = handle_ptr_->get_stream();

  const bool is_pdp           = pickup_delivery_indices.first != nullptr;
  order_info_.depot_included_ = (order_locations == nullptr);
  order_info_.resize(norders, is_pdp, stream_view);
  if (is_pdp) {
    auto pickup_indices   = pickup_delivery_indices.first;
    auto delivery_indices = pickup_delivery_indices.second;
    order_info_.v_pair_indices_.set_element_to_zero_async(0, handle_ptr_->get_stream());

    thrust::uninitialized_fill(handle_ptr_->get_thrust_policy(),
                               order_info_.v_is_pickup_index_.begin(),
                               order_info_.v_is_pickup_index_.end(),
                               false);

    auto zip_it =
      thrust::make_permutation_iterator(order_info_.v_is_pickup_index_.data(), pickup_indices);
    i_t n_requests = order_info_.get_num_requests();
    thrust::fill(handle_ptr_->get_thrust_policy(), zip_it, zip_it + n_requests, true);

    thrust::scatter(handle_ptr_->get_thrust_policy(),
                    pickup_indices,
                    pickup_indices + n_requests,
                    delivery_indices,
                    order_info_.v_pair_indices_.begin());

    thrust::scatter(handle_ptr_->get_thrust_policy(),
                    delivery_indices,
                    delivery_indices + n_requests,
                    pickup_indices,
                    order_info_.v_pair_indices_.begin());
    rmm::device_uvector<i_t> temp_abs(norders, stream);
    raft::copy(temp_abs.begin(), order_info_.v_pair_indices_.begin(), norders, stream);

    detail::transform_absolute(temp_abs, handle_ptr_->get_stream());
    auto end = thrust::unique(handle_ptr_->get_thrust_policy(), temp_abs.begin(), temp_abs.end());
    i_t unique_items = end - temp_abs.begin();
    cuopt_expects(norders == unique_items,
                  error_type_t::ValidationError,
                  "All indices should be included in pickup delivery pairs!");
    i_t* max_element_ptr =
      thrust::max_element(handle_ptr_->get_thrust_policy(), temp_abs.begin(), temp_abs.end());
    i_t h_max_element;
    raft::copy(&h_max_element, max_element_ptr, 1, stream);
    RAFT_CUDA_TRY(cudaStreamSynchronize(stream));
    cuopt_expects(norders - 1 == h_max_element,
                  error_type_t::ValidationError,
                  "Index given is too big or an index in the delivery pickup pairs is missing!");
    cuopt_expects(n_requests < 16384,
                  error_type_t::ValidationError,
                  "Number of pd requests should be inferior to 16384.");
  }

  if (order_locations != nullptr) {
    order_info_.v_order_locations_.resize(norders, stream_view);
    raft::copy(order_info_.v_order_locations_.data(), order_locations, norders, stream);
  }

  order_info_.v_prizes_.resize(norders, stream_view);
  if (auto prizes = data_model.get_order_prizes(); prizes.size() > 0) {
    raft::copy(order_info_.v_prizes_.data(), prizes.data(), norders, stream);

    // check if prize of pickup is different from delivery
    if (is_pdp) {
      bool valid_prize = detail::check_pdp_values<i_t, f_t>(pickup_delivery_indices.first,
                                                            pickup_delivery_indices.second,
                                                            order_info_.v_prizes_.data(),
                                                            order_info_.get_num_requests(),
                                                            handle_ptr_->get_stream());
      cuopt_expects(valid_prize,
                    error_type_t::ValidationError,
                    "Prizes of pickup and delivery pairs must be equal");
    }

  } else {
    thrust::uninitialized_fill(handle_ptr_->get_thrust_policy(),
                               order_info_.v_prizes_.begin(),
                               order_info_.v_prizes_.end(),
                               std::numeric_limits<f_t>::max());
  }

  populate_time_windows(data_model, order_info_);
}

template void populate_order_info(data_model_view_t<int, float> const& data_model,
                                  detail::order_info_t<int, float>& order_info);
}  // namespace detail
}  // namespace routing
}  // namespace cuopt
