/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#pragma once

#include <utilities/cuda_helpers.cuh>

#include <cuopt/routing/data_model_view.hpp>
#include <cuopt/routing/solver_settings.hpp>
#include <routing/fleet_info.hpp>
#include <routing/fleet_order_info.hpp>
#include <routing/order_info.hpp>
#include <routing/problem/special_nodes.cuh>
#include <routing/routing_helpers.cuh>
#include <routing/structures.hpp>
#include <routing/utilities/check_input.hpp>
#include <utilities/seed_generator.cuh>

#include <raft/core/handle.hpp>

#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/scatter.h>
#include <thrust/sort.h>

#include <rmm/device_uvector.hpp>

#include <optional>

namespace cuopt {
namespace routing {
namespace detail {

template <typename i_t, typename f_t>
struct viables_t {
  viables_t(const data_model_view_t<i_t, f_t>& data_model_view_)
    : compatibility_matrix(0, data_model_view_.get_handle_ptr()->get_stream()),
      viable_to_pickups(0, data_model_view_.get_handle_ptr()->get_stream()),
      n_viable_to_pickups(0, data_model_view_.get_handle_ptr()->get_stream()),
      viable_from_pickups(0, data_model_view_.get_handle_ptr()->get_stream()),
      n_viable_from_pickups(0, data_model_view_.get_handle_ptr()->get_stream()),
      viable_to_deliveries(0, data_model_view_.get_handle_ptr()->get_stream()),
      n_viable_to_deliveries(0, data_model_view_.get_handle_ptr()->get_stream()),
      viable_from_deliveries(0, data_model_view_.get_handle_ptr()->get_stream()),
      n_viable_from_deliveries(0, data_model_view_.get_handle_ptr()->get_stream())
  {
  }

  struct view_t {
    DI raft::device_span<const i_t> get_viable_from_pickups(i_t node,
                                                            i_t n_requests,
                                                            i_t max_n_neighbors,
                                                            bool exclude_self_in_neighbors,
                                                            i_t batch_num = 0) const
    {
      i_t n_viable     = n_viable_from_pickups[node] - i_t(exclude_self_in_neighbors);
      i_t batch_offset = batch_num * max_n_neighbors;
      // filter the tailing ones
      n_viable         = n_viable - batch_offset;
      i_t n_considered = max(0, min(n_viable, max_n_neighbors));
      return raft::device_span<const i_t>{
        &viable_from_pickups[node * n_requests] + i_t(exclude_self_in_neighbors) + batch_offset,
        (size_t)n_considered};
    }
    DI raft::device_span<const i_t> get_viable_to_pickups(i_t node,
                                                          i_t n_requests,
                                                          i_t max_n_neighbors,
                                                          bool exclude_self_in_neighbors,
                                                          i_t batch_num = 0) const
    {
      i_t n_viable     = n_viable_to_pickups[node] - i_t(exclude_self_in_neighbors);
      i_t batch_offset = batch_num * max_n_neighbors;
      // filter the tailing ones
      n_viable         = n_viable - batch_offset;
      i_t n_considered = max(0, min(n_viable, max_n_neighbors));
      return raft::device_span<const i_t>{
        &viable_to_pickups[node * n_requests] + i_t(exclude_self_in_neighbors) + batch_offset,
        (size_t)n_considered};
    }
    DI raft::device_span<const i_t> get_viable_from_deliveries(i_t node,
                                                               i_t n_requests,
                                                               i_t max_n_neighbors,
                                                               bool exclude_self_in_neighbors,
                                                               i_t batch_num = 0) const
    {
      i_t n_viable     = n_viable_from_deliveries[node] - i_t(exclude_self_in_neighbors);
      i_t batch_offset = batch_num * max_n_neighbors;
      // filter the tailing ones
      n_viable         = n_viable - batch_offset;
      i_t n_considered = max(0, min(n_viable, max_n_neighbors));
      return raft::device_span<const i_t>{
        &viable_from_deliveries[node * n_requests] + i_t(exclude_self_in_neighbors) + batch_offset,
        (size_t)n_considered};
    }
    DI raft::device_span<const i_t> get_viable_to_deliveries(i_t node,
                                                             i_t n_requests,
                                                             i_t max_n_neighbors,
                                                             bool exclude_self_in_neighbors,
                                                             i_t batch_num = 0) const
    {
      i_t n_viable     = n_viable_to_deliveries[node] - i_t(exclude_self_in_neighbors);
      i_t batch_offset = batch_num * max_n_neighbors;
      // filter the tailing ones
      n_viable         = n_viable - batch_offset;
      i_t n_considered = max(0, min(n_viable, max_n_neighbors));
      return raft::device_span<const i_t>{
        &viable_to_deliveries[node * n_requests] + i_t(exclude_self_in_neighbors) + batch_offset,
        (size_t)n_considered};
    }
    raft::device_span<const uint8_t> compatibility_matrix;
    raft::device_span<const i_t> viable_to_pickups;
    raft::device_span<const i_t> n_viable_to_pickups;
    raft::device_span<const i_t> viable_from_pickups;
    raft::device_span<const i_t> n_viable_from_pickups;
    raft::device_span<const i_t> viable_to_deliveries;
    raft::device_span<const i_t> n_viable_to_deliveries;
    raft::device_span<const i_t> viable_from_deliveries;
    raft::device_span<const i_t> n_viable_from_deliveries;
    i_t max_viable_row_size;
  };

  view_t view() const
  {
    view_t v;
    v.compatibility_matrix =
      raft::device_span<const uint8_t>{compatibility_matrix.data(), compatibility_matrix.size()};
    v.viable_to_pickups =
      raft::device_span<const i_t>{viable_to_pickups.data(), viable_to_pickups.size()};
    v.n_viable_to_pickups =
      raft::device_span<const i_t>{n_viable_to_pickups.data(), n_viable_to_pickups.size()};
    v.viable_from_pickups =
      raft::device_span<const i_t>{viable_from_pickups.data(), viable_from_pickups.size()};
    v.n_viable_from_pickups =
      raft::device_span<const i_t>{n_viable_from_pickups.data(), n_viable_from_pickups.size()};
    v.viable_to_deliveries =
      raft::device_span<const i_t>{viable_to_deliveries.data(), viable_to_deliveries.size()};
    v.n_viable_to_deliveries =
      raft::device_span<const i_t>{n_viable_to_deliveries.data(), n_viable_to_deliveries.size()};
    v.viable_from_deliveries =
      raft::device_span<const i_t>{viable_from_deliveries.data(), viable_from_deliveries.size()};
    v.n_viable_from_deliveries = raft::device_span<const i_t>{n_viable_from_deliveries.data(),
                                                              n_viable_from_deliveries.size()};
    v.max_viable_row_size      = max_viable_row_size;
    return v;
  }

  rmm::device_uvector<uint8_t> compatibility_matrix;
  rmm::device_uvector<i_t> viable_to_pickups;
  rmm::device_uvector<i_t> n_viable_to_pickups;
  rmm::device_uvector<i_t> viable_from_pickups;
  rmm::device_uvector<i_t> n_viable_from_pickups;
  rmm::device_uvector<i_t> viable_to_deliveries;
  rmm::device_uvector<i_t> n_viable_to_deliveries;
  rmm::device_uvector<i_t> viable_from_deliveries;
  rmm::device_uvector<i_t> n_viable_from_deliveries;
  i_t max_viable_row_size = 0;
};

// Possibly temporary class
// Try to merge this with data_model class or create a common problem representation class that
// can be used by both solvers
template <typename i_t, typename f_t>
class problem_t {
 public:
  problem_t()            = delete;
  problem_t(problem_t&&) = default;
  problem_t(const data_model_view_t<i_t, f_t>& data_model_view_,
            solver_settings_t<i_t, f_t> const& solver_settings_);

  void populate_dimensions_info();
  void sort_viable_matrix(rmm::device_uvector<i_t>& viable_from_matrix,
                          rmm::device_uvector<i_t>& viable_to_matrix);

  // This is temporary
  void populate_host_arrays();

  void initialize_depot_info();

  VehicleInfo<f_t, false> get_vehicle_info(i_t vehicle_id) const;

  std::vector<std::vector<i_t>> get_vehicle_buckets() const;
  void populate_vehicle_buckets();

  bool is_pickup(i_t node_id) const;

  bool is_pdp() const;
  bool is_cvrp() const;
  bool is_cvrp_intra() const;

  // FIXME:: This is not scalable as we add more features. We should be able to use the method
  // that we use in kernels
  double distance_between(const NodeInfo<>& node_1,
                          const NodeInfo<>& node_2,
                          const int& vehicle_id) const;

  struct view_t {
    DI NodeInfo<> get_start_depot_node_info(const i_t vehicle_id) const
    {
      return start_depot_node_infos[vehicle_id];
    }

    DI NodeInfo<> get_return_depot_node_info(const i_t vehicle_id) const
    {
      return return_depot_node_infos[vehicle_id];
    }

    DI i_t get_max_break_dimensions() const { return special_nodes.num_max_break_dimensions; }

    DI i_t get_break_dimensions(const i_t vehicle_id) const
    {
      return special_nodes.get_break_dimensions(vehicle_id);
    }

    DI bool has_special_nodes() const { return !special_nodes.empty(); }
    constexpr i_t get_num_buckets() const { return fleet_info.vehicle_availability.size(); }
    DI bool has_non_uniform_breaks() const { return non_uniform_breaks; }
    DI bool is_cvrp_intra() const
    {
      return !order_info.is_pdp() && !dimensions_info.has_dimension(dim_t::TIME) &&
             !dimensions_info.has_dimension(dim_t::BREAK);
    }
    DI bool is_cvrp() const { return is_cvrp_; }

    typename fleet_info_t<i_t, f_t>::view_t fleet_info;
    typename order_info_t<i_t, f_t>::view_t order_info;
    raft::device_span<const i_t> pickup_indices;
    raft::device_span<const i_t> delivery_indices;
    enabled_dimensions_t dimensions_info;
    typename viables_t<i_t, f_t>::view_t viables;
    raft::device_span<const NodeInfo<>> start_depot_node_infos;
    raft::device_span<const NodeInfo<>> return_depot_node_infos;
    raft::device_span<const i_t> bucket_to_vehicle_id;
    typename special_nodes_t<i_t>::view_t special_nodes;
    bool non_uniform_breaks{false};
    bool is_cvrp_{false};
  };

  view_t view() const
  {
    view_t v;
    v.fleet_info        = fleet_info.view();
    v.order_info        = order_info.view();
    auto [p_idx, d_idx] = data_view_ptr->get_pickup_delivery_pair();
    if (p_idx) {
      v.pickup_indices   = raft::device_span<const i_t>{p_idx, (size_t)get_num_requests()};
      v.delivery_indices = raft::device_span<const i_t>{d_idx, (size_t)get_num_requests()};
    }
    v.dimensions_info = dimensions_info;
    v.viables         = viables.view();

    v.start_depot_node_infos = raft::device_span<const NodeInfo<>>(start_depot_node_infos.data(),
                                                                   start_depot_node_infos.size());

    v.return_depot_node_infos = raft::device_span<const NodeInfo<>>(return_depot_node_infos.data(),
                                                                    return_depot_node_infos.size());
    v.bucket_to_vehicle_id    = cuopt::make_span(bucket_to_vehicle_id);
    v.special_nodes           = special_nodes.view();
    v.non_uniform_breaks      = has_non_uniform_breaks();
    v.is_cvrp_                = is_cvrp();
    return v;
  }

  const data_model_view_t<i_t, f_t>* data_view_ptr;
  const solver_settings_t<i_t, f_t>* solver_settings_ptr;

  i_t get_num_orders() const;

  i_t get_num_requests() const;

  i_t get_num_buckets() const;

  detail::NodeInfo<> get_node_info_of_node(const int node) const;

  std::optional<NodeInfo<>> get_single_depot() const;

  NodeInfo<> get_start_depot_node_info(const i_t vehicle_id) const;

  detail::NodeInfo<> get_brother_node_info(const NodeInfo<>& node) const;

  void populate_special_nodes();

  i_t get_fleet_size() const;

  i_t get_max_break_dimensions() const;

  bool has_vehicle_breaks() const;

  bool has_non_uniform_breaks() const;

  bool has_prize_collection() const;

  bool has_vehicle_fixed_costs() const;

  std::vector<i_t> get_preferred_order_of_vehicles() const;

  // handle
  raft::handle_t const* handle_ptr{nullptr};

  fleet_info_t<i_t, f_t> fleet_info;
  typename fleet_info_t<i_t, f_t>::host_t fleet_info_h;
  order_info_t<i_t, f_t> order_info;
  typename order_info_t<i_t, f_t>::host_t order_info_h;
  enabled_dimensions_t dimensions_info;

  // FIXME:: host copies of vectors needed for diversity manager
  // we should not need to have copies here, instead we should implement
  // appropriate host functions in order_info_, fleet_info_ classes and call
  // them directly
  std::map<i_t, std::vector<f_t>> distance_matrices_h;
  std::vector<i_t> pair_indices_h;
  std::vector<bool> is_pickup_h;
  std::vector<i_t> order_locations_h;
  std::vector<uint8_t> vehicle_types_h;
  std::vector<bool> drop_return_trip_h;
  std::vector<bool> skip_first_trip_h;
  std::vector<std::vector<i_t>> vehicle_buckets_h;
  std::vector<i_t> bucket_to_vehicle_id_h;
  viables_t<i_t, f_t> viables;

  std::optional<NodeInfo<>> single_depot_node;
  std::vector<NodeInfo<>> start_depot_node_infos_h, return_depot_node_infos_h;
  rmm::device_uvector<NodeInfo<>> start_depot_node_infos, return_depot_node_infos;
  rmm::device_uvector<i_t> bucket_to_vehicle_id;

  special_nodes_t<i_t> special_nodes;
  bool is_tsp{false};
  bool is_cvrp_{false};
  bool non_uniform_breaks_{false};
};

}  // namespace detail
}  // namespace routing
}  // namespace cuopt
