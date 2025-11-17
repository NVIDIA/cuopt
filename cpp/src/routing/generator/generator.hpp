/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#pragma once

#include <cuopt/routing/cython/generator.hpp>
#include <routing/fleet_info.hpp>
#include <routing/order_info.hpp>
#include <routing/utilities/md_utils.hpp>

namespace cuopt {
namespace routing {
namespace generator {

template <typename f_t>
using coordinates_t = std::tuple<rmm::device_uvector<f_t>, rmm::device_uvector<f_t>>;

template <typename i_t>
using time_window_t =
  std::tuple<rmm::device_uvector<i_t>, rmm::device_uvector<i_t>, rmm::device_uvector<i_t>>;

template <typename i_t>
using break_dimension_t =
  std::tuple<rmm::device_uvector<i_t>, rmm::device_uvector<i_t>, rmm::device_uvector<i_t>>;

template <typename i_t>
using vehicle_time_window_t = std::tuple<rmm::device_uvector<i_t>, rmm::device_uvector<i_t>>;
/**
 * @brief Container for allocated dataset.
 * @tparam i_t Integer type. Needs to be int (32bit) at the moment. Please open
 * an issue if other type are needed.
 * @tparam f_t Floating point type. Needs to be float (32bit) at the moment.
 */
template <typename i_t, typename f_t>
class dataset_t {
 public:
  dataset_t(rmm::device_uvector<f_t>& x_pos,
            rmm::device_uvector<f_t>& y_pos,
            detail::order_info_t<i_t, f_t>& order_info,
            detail::fleet_info_t<i_t, f_t>& fleet_info,
            std::vector<break_dimension_t<i_t>>& break_container);

  std::tuple<rmm::device_uvector<f_t>&, rmm::device_uvector<f_t>&> get_coordinates();
  detail::order_info_t<i_t, f_t>& get_order_info();
  detail::fleet_info_t<i_t, f_t>& get_fleet_info();
  std::vector<break_dimension_t<i_t>>& get_vehicle_breaks();

 private:
  rmm::device_uvector<f_t> v_x_pos_;
  rmm::device_uvector<f_t> v_y_pos_;
  detail::order_info_t<i_t, f_t> order_info_;
  detail::fleet_info_t<i_t, f_t> fleet_info_;
  std::vector<break_dimension_t<i_t>> break_container_;
};

template <typename i_t, typename f_t>
detail::order_info_t<i_t, f_t> generate_order_info(raft::handle_t const& handle,
                                                   dataset_params_t<i_t, f_t> const& params);

template <typename i_t, typename f_t>
detail::fleet_order_constraints_t<i_t> generate_fleet_order_constraints(
  raft::handle_t const& handle, dataset_params_t<i_t, f_t> const& params);

template <typename i_t, typename f_t>
detail::fleet_info_t<i_t, f_t> generate_fleet_info(
  raft::handle_t const& handle,
  dataset_params_t<i_t, f_t> const& params,
  detail::order_info_t<i_t, f_t> const& order_info);

template <typename i_t, typename f_t>
coordinates_t<f_t> generate_coordinates(raft::handle_t const&, dataset_params_t<i_t, f_t> const&);

template <typename i_t, typename f_t>
d_mdarray_t<f_t> generate_matrices(raft::handle_t const&,
                                   dataset_params_t<i_t, f_t> const&,
                                   coordinates_t<f_t> const&);

template <typename i_t, typename f_t>
rmm::device_uvector<demand_i_t> generate_demands(raft::handle_t const& handle,
                                                 dataset_params_t<i_t, f_t> const& params);

template <typename i_t, typename f_t>
rmm::device_uvector<cap_i_t> generate_vehicle_capacities(raft::handle_t const& handle,
                                                         dataset_params_t<i_t, f_t> const& params,
                                                         size_t fleet_size);

template <typename i_t, typename f_t>
time_window_t<i_t> generate_time_windows(raft::handle_t const& handle,
                                         dataset_params_t<i_t, f_t> const& params);

template <typename i_t, typename f_t>
rmm::device_uvector<bool> generate_drop_return_trips(raft::handle_t const& handle,
                                                     dataset_params_t<i_t, f_t> const& params,
                                                     size_t fleet_size);

template <typename i_t, typename f_t>
rmm::device_uvector<bool> generate_skip_first_trips(raft::handle_t const& handle,
                                                    dataset_params_t<i_t, f_t> const& params,
                                                    size_t fleet_size);

template <typename i_t, typename f_t>
vehicle_time_window_t<i_t> generate_vehicle_time_windows(
  raft::handle_t const& handle,
  dataset_params_t<i_t, f_t> const& params,
  detail::order_info_t<i_t, f_t> const& order_info,
  detail::fleet_order_constraints_t<i_t> const& fleet_order_constraints,
  size_t fleet_size);

template <typename i_t, typename f_t>
std::vector<break_dimension_t<i_t>> generate_vehicle_breaks(
  raft::handle_t const& handle,
  dataset_params_t<i_t, f_t> const& params,
  detail::order_info_t<i_t, f_t> const& order_info,
  size_t fleet_size);

template <typename i_t, typename f_t>
rmm::device_uvector<uint8_t> generate_vehicle_types(raft::handle_t const& handle,
                                                    dataset_params_t<i_t, f_t> const& params,
                                                    size_t fleet_size);

/**
 * @brief Constructs a dataset_t object from the parameters defined in
 *`params`.
 * @param[in] handle Library handle (RAFT) containing hardware resources
 * information. A default handle is valid.
 * @param[in] params The `dataset_params_t` to use
 * @return dataset_t holding matrices, orders and vehicle info.
 */
template <typename i_t, typename f_t>
dataset_t<i_t, f_t> generate_dataset(raft::handle_t const& handle,
                                     dataset_params_t<i_t, f_t> const& params);
}  // namespace generator
}  // namespace routing
}  // namespace cuopt
