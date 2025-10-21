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

#pragma once

#include "dimensions_route.cuh"

#include <routing/fleet_info.hpp>

namespace cuopt {
namespace routing {
namespace detail {

template <typename i_t, typename f_t, request_t REQUEST>
class route_t {
 public:
  route_t() = delete;
  route_t(solution_handle_t<i_t, f_t> const* sol_handle_,
          i_t route_id_,
          i_t vehicle_id_,
          const fleet_info_t<i_t, f_t>* fleet_info_ptr_,
          enabled_dimensions_t dimensions_info_)
    : sol_handle(sol_handle_),
      dimensions(sol_handle_, dimensions_info_),
      route_id(route_id_, sol_handle_->get_stream()),
      vehicle_id(vehicle_id_, sol_handle_->get_stream()),
      infeasibility_cost(sol_handle_->get_stream()),
      objective_cost(sol_handle_->get_stream()),
      n_nodes(sol_handle_->get_stream()),
      fleet_info_ptr(fleet_info_ptr_)
  {
    raft::common::nvtx::range fun_scope("zero route_t copy_ctr");
    infeasible_cost_t zero_inf;
    objective_cost_t zero_obj;
    infeasibility_cost.set_value_async(zero_inf, sol_handle->get_stream());
    objective_cost.set_value_async(zero_obj, sol_handle->get_stream());
  }

  void print() const
  {
    std::cout << "( " << route_id.value(route_id.stream()) << ", "
              << vehicle_id.value(vehicle_id.stream()) << ")";
    dimensions.requests.print(n_nodes.value(n_nodes.stream()));
  }

  route_t(const route_t& route)
    : sol_handle(route.sol_handle),
      dimensions(route.dimensions),
      route_id(route.route_id, route.sol_handle->get_stream()),
      vehicle_id(route.vehicle_id, route.sol_handle->get_stream()),
      n_nodes(route.n_nodes, route.sol_handle->get_stream()),
      infeasibility_cost(route.infeasibility_cost, route.sol_handle->get_stream()),
      objective_cost(route.objective_cost, route.sol_handle->get_stream()),
      fleet_info_ptr(route.fleet_info_ptr)
  {
    raft::common::nvtx::range fun_scope("route copy_ctr");
  }

  route_t& operator=(route_t&& route) = default;

  // Make the all buffers factor bigger
  void resize(i_t new_size) { dimensions.resize(new_size); }

  // extend for other things later
  i_t max_nodes_per_route() const noexcept { return dimensions.requests.node_info.size(); }

  infeasible_cost_t get_infeasibility_cost() const noexcept
  {
    return infeasibility_cost.value(sol_handle->get_stream());
  }

  i_t get_vehicle_id() const { return vehicle_id.value(sol_handle->get_stream()); }

  template <dim_t dim>
  auto& get_dim()
  {
    return get_dimension_of<dim>(dimensions);
  }
  template <dim_t dim>
  auto& get_dim() const
  {
    return get_dimension_of<dim>(dimensions);
  }

  auto& dimensions_info() const { return dimensions.dimensions_info; }
  auto& dimensions_info() { return dimensions.dimensions_info; }

  i_t get_num_breaks() const
  {
    if (dimensions_info().has_dimension(dim_t::BREAK)) {
      i_t num_nodes = n_nodes.value(sol_handle->get_stream());
      return get_dim<dim_t::BREAK>().breaks_forward.element(num_nodes, sol_handle->get_stream());
    }

    return 0;
  }

  i_t get_num_service_nodes() const
  {
    i_t num_nodes = n_nodes.value(sol_handle->get_stream());
    if (dimensions_info().has_dimension(dim_t::BREAK)) {
      i_t num_breaks =
        get_dim<dim_t::BREAK>().breaks_forward.element(num_nodes, sol_handle->get_stream());
      return num_nodes - num_breaks - 1;
    }

    return num_nodes - 1;
  }

  bool is_empty() const { return get_num_service_nodes() == 0; }

  // this will be used in device code
  struct view_t {
    static view_t create_view(i_t* num_nodes_,
                              i_t* route_id_,
                              i_t* vehicle_id_,
                              infeasible_cost_t* infeasibility_cost_,
                              objective_cost_t* objective_cost_,
                              typename fleet_info_t<i_t, f_t>::view_t fleet_info_)
    {
      view_t v;
      v.n_nodes            = num_nodes_;
      v.route_id           = route_id_;
      v.vehicle_id         = vehicle_id_;
      v.infeasibility_cost = infeasibility_cost_;
      v.objective_cost     = objective_cost_;
      v.fleet_info         = fleet_info_;
      return v;
    }
    DI auto& requests() const { return dimensions.requests; }
    DI auto& requests() { return dimensions.requests; }

    template <dim_t dim>
    DI auto& get_dim()
    {
      return get_dimension_of<dim>(dimensions);
    }
    template <dim_t dim>
    DI auto& get_dim() const
    {
      return get_dimension_of<dim>(dimensions);
    }
    DI auto& dimensions_info() const { return dimensions.dimensions_info; }
    DI auto& dimensions_info() { return dimensions.dimensions_info; }

    DI int get_num_breaks() const
    {
      if (dimensions_info().has_dimension(dim_t::BREAK)) {
        return get_dim<dim_t::BREAK>().breaks_forward[*n_nodes];
      }

      return 0;
    }

    DI int get_num_service_nodes() const { return *n_nodes - 1 - get_num_breaks(); }

    template <request_t r_t = REQUEST, std::enable_if_t<r_t == request_t::VRP, bool> = true>
    DI bool is_valid() const
    {
      return *n_nodes > 0;
    }

    template <request_t r_t = REQUEST, std::enable_if_t<r_t == request_t::PDP, bool> = true>
    DI bool is_valid() const
    {
      return *n_nodes > 0 && (*n_nodes - get_num_breaks()) % 2 == 1;
    }

    template <request_t r_t = REQUEST, std::enable_if_t<r_t == request_t::VRP, bool> = true>
    DI auto get_request_node(typename route_node_map_t<i_t>::view_t const& route_node_map,
                             request_id_t<r_t> const& request) const
    {
      return dimensions.get_request_node(route_node_map, request);
    }

    template <request_t r_t = REQUEST, std::enable_if_t<r_t == request_t::PDP, bool> = true>
    DI auto get_request_node(typename route_node_map_t<i_t>::view_t const& route_node_map,
                             request_id_t<r_t> const& request) const
    {
      return dimensions.get_request_node(route_node_map, request);
    }

    DI const VehicleInfo<f_t> vehicle_info() const
    {
      return fleet_info.get_vehicle_info(*vehicle_id);
    }

    /**
     * @brief Get the node object from the index
     *
     * @param idx Intra route index
     * @return node_t
     */
    DI node_t<i_t, f_t, REQUEST> get_node(i_t idx) const { return dimensions.get_node(idx); }

    DI void set_node(i_t idx, const node_t<i_t, f_t, REQUEST>& node)
    {
      dimensions.set_node(idx, node);
    }

    DI i_t node_id(i_t idx) const { return dimensions.node_id(idx); }
    DI i_t brother_id(i_t idx) const { return dimensions.brother_id(idx); }
    DI NodeInfo<i_t> node_info(i_t idx) const { return dimensions.node_info(idx); };

    DI void set_forward_data(i_t idx, const node_t<i_t, f_t, REQUEST>& node)
    {
      loop_over_dimensions(dimensions_info(), [&](auto I) {
        auto& node_dim = get_dimension_of<I>(node);
        get_dimension_of<I>(dimensions).set_forward_data(idx, node_dim);
      });
    }

    DI void set_backward_data(i_t idx, const node_t<i_t, f_t, REQUEST>& node)
    {
      loop_over_dimensions(dimensions_info(), [&](auto I) {
        auto& node_dim = get_dimension_of<I>(node);
        get_dimension_of<I>(dimensions).set_backward_data(idx, node_dim);
      });
    }

    DI bool is_feasible() const
    {
      // check last node and see whether it is feasible
      return get_node(*n_nodes).forward_feasible(this->vehicle_info());
    }

    DI bool is_time_feasible() const
    {
      if (!dimensions_info().has_dimension(dim_t::TIME)) { return true; }
      // check last node and see whether it is feasible
      return get_node(*n_nodes).time_dim.forward_feasible(this->vehicle_info());
    }

    DI void copy_to_tsp_route()
    {
      dimensions.requests.tsp_requests.start = get_node(0).node_info();
      dimensions.requests.tsp_requests.end   = get_node(*n_nodes).node_info();

      for (i_t tid = threadIdx.x; tid < *n_nodes; tid += blockDim.x) {
        if (get_node(tid).node_info().is_depot()) { continue; }
        dimensions.requests.tsp_requests.pred[get_node(tid).node_info().node()] =
          get_node(tid - 1).node_info();
        dimensions.requests.tsp_requests.succ[get_node(tid).node_info().node()] =
          get_node(tid + 1).node_info();
      }

      dimensions.requests.tsp_requests.pred[dimensions.requests.tsp_requests.end.node()] =
        get_node(*n_nodes - 1).node_info();
      dimensions.requests.tsp_requests.succ[dimensions.requests.tsp_requests.start.node()] =
        get_node(1).node_info();
    }

    // insert a single node to the route
    // note that this will produce an infeasible route(in terms of PD clustering) for PDP problems
    // we choose whether we want to update the global vars, since this might be used to create temp
    // routes
    // nb_nodes allow to delete more than 1 consecutive nodes
    DI void eject_node(i_t ejection_idx,
                       typename route_node_map_t<i_t>::view_t& route_node_map,
                       bool update_global_arrs = true,
                       int nb_nodes            = 1)
    {
      cuopt_assert(__popc(__activemask()) == 1, "eject_node should be single threaded");
      cuopt_assert(ejection_idx >= 0, "ejection_idx should be greated than 0");
      cuopt_assert(ejection_idx + nb_nodes - 1 < *n_nodes,
                   "ejection_idx should be smaller than n_nodes");
      cuopt_assert(*n_nodes >= 1, "Size cannot be smaller than 1");
      if (update_global_arrs) {
        for (i_t i = 0; i < nb_nodes; ++i) {
          // Update route_id_per_node
          const auto ejected_node_info = node_info(ejection_idx + i);
          route_node_map.reset_node(ejected_node_info);
        }
      }
      // Left shift to delete nodes
      for (i_t i = ejection_idx; i <= *n_nodes - nb_nodes; ++i) {
        const auto node = get_node(i + nb_nodes);
        set_node(i, node);
        // Update intra_route_idx_per_node
        if (update_global_arrs) { route_node_map.set_intra_route_idx(node.node_info(), i); }
      }
      // Update size
      *n_nodes -= nb_nodes;
    }

    DI void parallel_eject_node(typename route_t<i_t, f_t, REQUEST>::view_t& route_eject_buffer,
                                i_t ejection_idx,
                                typename route_node_map_t<i_t>::view_t& route_node_map,
                                bool update_global_arrs = true,
                                int nb_nodes            = 1)
    {
      if (nb_nodes == 0) return;
      cuopt_assert(ejection_idx > 0, "ejection_idx should be greated than 0");
      cuopt_assert(ejection_idx + nb_nodes - 1 < *n_nodes,
                   "ejection_idx should be smaller than n_nodes");
      cuopt_assert(*n_nodes >= 1, "Size cannot be smaller than 1");
      if (update_global_arrs) {
        for (i_t i = threadIdx.x; i < nb_nodes; i += blockDim.x) {
          // Update route_id_per_node
          const auto ejected_node_info = node_info(ejection_idx + i);
          route_node_map.reset_node(ejected_node_info);
        }
      }
      const i_t n_nodes_w_depot = *n_nodes + 1;
      // copy nodes to be shifted
      route_eject_buffer.copy_from(*this, ejection_idx + nb_nodes, n_nodes_w_depot, 0);
      __syncthreads();
      this->copy_from(
        route_eject_buffer, 0, n_nodes_w_depot - (ejection_idx + nb_nodes), ejection_idx);
      __syncthreads();
      // Left shift to delete nodes
      for (i_t i = ejection_idx + threadIdx.x; i < n_nodes_w_depot - nb_nodes; i += blockDim.x) {
        const auto& request = requests().get_node(i);
        // Update intra_route_idx_per_node
        if (update_global_arrs) { route_node_map.set_intra_route_idx(request.info, i); }
      }
      // Update size
      if (threadIdx.x == 0) { *n_nodes -= nb_nodes; }
      __syncthreads();
    }

    /**
     * @brief Takes an array of nodes insertion_nodes, inserts them to this route in parallel.
     * The insertion is done by shifting the succeding part of the route by nb_nodes in parallel.
     * The parallel shift is done by two parallel copies, one to temporary buffer, and one back from
     * temporary buffer. The insertion is also in parallel, by copying the insertion_nodes array
     * into the opened gap
     */
    DI void parallel_insert_node(
      typename route_t<i_t, f_t, REQUEST>::view_t& route_insert_buffer,
      i_t insertion_idx,
      const typename dimensions_route_t<i_t, f_t, REQUEST>::view_t& insertion_nodes,
      typename route_node_map_t<i_t>::view_t& route_node_map,
      bool update_global_arrs = true,
      i_t nb_nodes            = 1)
    {
      if (nb_nodes == 0) return;
      cuopt_assert(insertion_idx >= 0, "insertion_idx should be greated than 0");
      // copy to open up space
      const i_t n_nodes_w_depot = *n_nodes + 1;
      // copy nodes to be shifted
      route_insert_buffer.copy_from(*this, insertion_idx + 1, n_nodes_w_depot, 0);
      __syncthreads();
      this->copy_from(route_insert_buffer,
                      0,
                      n_nodes_w_depot - (insertion_idx + 1),
                      insertion_idx + nb_nodes + 1);
      __syncthreads();
      if (threadIdx.x == 0) {
        // Update size
        *n_nodes += nb_nodes;
      }
      for (i_t i = threadIdx.x; i < nb_nodes; i += blockDim.x) {
        const auto& node = insertion_nodes.get_node(i);
        set_node(i + insertion_idx + 1, node);
        if (update_global_arrs) { route_node_map.set_route_id(node.node_info(), *route_id); }
      }
      __syncthreads();
      if (update_global_arrs) {
        for (i_t i = insertion_idx + threadIdx.x + 1; i <= *n_nodes; i += blockDim.x) {
          const auto& request = requests().get_node(i);

          route_node_map.set_route_id_and_intra_idx(request.info, *route_id, i);
        }
      }
      __syncthreads();
    }

    // insert a single node to the route AFTER insertion_idx
    // note that this will produce an infeasible route(in terms of PD clustering) for PDP problems
    // we choose whether we want to update the global vars, since this might be used to create temp
    // routes
    // nb_nodes allow to insert more than 1 consecutive nodes
    DI void insert_node(i_t insertion_idx,
                        const node_t<i_t, f_t, REQUEST>* insertion_nodes,
                        typename route_node_map_t<i_t>::view_t& route_node_map,
                        bool update_global_arrs = true,
                        i_t nb_nodes            = 1)
    {
      cuopt_assert(__popc(__activemask()) == 1, "Insert node should be single threaded");
      cuopt_assert(insertion_idx >= 0, "insertion_idx should be greated than 0");
      cuopt_assert(insertion_idx <= *n_nodes, "insertion_idx should be less than n_nodes");
      cuopt_assert(*n_nodes + nb_nodes + 1 <= max_nodes_per_route(),
                   "Can not insert more nodes than the max nodes per route");
      i_t i = *n_nodes;
      cuopt_assert(i >= 1, "Size cannot be smaller than 1");
      // Right shift to leave room
      for (; i != insertion_idx; --i) {
        const auto& node = get_node(i);
        set_node(i + nb_nodes, node);
        // Update intra_route_idx_per_node
        if (update_global_arrs) {
          // no need to update route id because it is staying the same
          route_node_map.set_intra_route_idx(node.node_info(), i + nb_nodes);
        }
      }
      cuopt_assert(i >= 0, "I should be strictly positive");
      // Insert nodes
      for (i_t j = 0; j < nb_nodes; ++j) {
        const auto& insertion_node = insertion_nodes[j];
        set_node(i + 1 + j, insertion_node);
        if (update_global_arrs) {
          route_node_map.set_route_id_and_intra_idx(
            insertion_node.node_info(), *route_id, i + 1 + j);
        }
      }
      // Update size
      *n_nodes += nb_nodes;
    }

    DI void insert_node(i_t insertion_idx,
                        const node_t<i_t, f_t, REQUEST>& insertion_node,
                        typename route_node_map_t<i_t>::view_t& route_node_map,
                        bool update_global_arrs = true)
    {
      insert_node(insertion_idx, &insertion_node, route_node_map, update_global_arrs, 1);
    }

    template <request_t r_t = REQUEST, std::enable_if_t<r_t == request_t::VRP, bool> = true>
    DI void insert_request(request_id_t<REQUEST> const& request_location,
                           const request_node_t<i_t, f_t, REQUEST>& request_node,
                           typename route_node_map_t<i_t>::view_t& route_node_map,
                           bool update_global_arrs = true)
    {
      auto node = request_node.node();
      cuopt_assert(node.request.info.node_type() == node_type_t::DELIVERY,
                   "VRP should insert only delivery nodes");
      insert_node(request_location.id(), node, route_node_map, update_global_arrs);
    }

    // code reuse of insert_node can be implemented inside, but it makes it harder to read
    // Insert pickup and delivery by shifting nodes inside the route and update size
    // Suppose the route is big enough, single threaded for now
    // TODO: handle route not big enough
    template <request_t r_t = REQUEST, std::enable_if_t<r_t == request_t::PDP, bool> = true>
    DI void insert_request(request_id_t<REQUEST> const& request_location,
                           const request_node_t<i_t, f_t, REQUEST>& request_node,
                           typename route_node_map_t<i_t>::view_t& route_node_map,
                           bool update_global_arrs = true)
    {
      auto pickup_location   = request_location.pickup;
      auto delivery_location = request_location.delivery;
      auto pickup            = request_node.pickup;
      auto delivery          = request_node.delivery;
      cuopt_assert(__popc(__activemask()) == 1, "Insert pickup delivery should be single threaded");
      // TODO: fetch node for shared mem

      // Shift forward by two (to handle pickup) until delivery is reached
      // Start at curr_route_size to shift end depot
      i_t i = *n_nodes;

      cuopt_assert(i >= 1, "Size cannot be smaller than 1");
      cuopt_assert(i > delivery_location,
                   "Delivery insertion position should be smaller than route size");
      cuopt_assert(i > pickup_location,
                   "Pickup insertion position should be smaller than route size");
      cuopt_assert(pickup_location >= 0, "Pickup insertion position should be strictly positive");
      cuopt_assert(delivery_location >= 0,
                   "Delivery insertion position should be strictly positive");
      cuopt_assert(pickup_location <= delivery_location,
                   "Pickup insertion location should be smaller or equal to delivery location");
      for (; i != delivery_location; --i) {
        const auto& node = get_node(i);
        set_node(i + 2, node);
        // Update intra_route_idx_per_node
        if (update_global_arrs) { route_node_map.set_intra_route_idx(node.node_info(), i + 2); }
      }
      cuopt_assert(i >= 0, "I should be strictly positive");
      // Insert delivery
      set_node(i + 2, delivery);
      // Update intra_route_idx_per_node
      if (update_global_arrs) {
        route_node_map.set_route_id_and_intra_idx(delivery.node_info(), *route_id, i + 2);
      }

      // Shift forward by one until pickup is reached
      for (; i != pickup_location; --i) {
        const auto& node = get_node(i);
        set_node(i + 1, node);
        // Update intra_route_idx_per_node
        if (update_global_arrs) { route_node_map.set_intra_route_idx(node.node_info(), i + 1); }
      }
      cuopt_assert(i >= 0, "I should be strictly positive");
      // Insert pickup
      set_node(i + 1, pickup);
      if (update_global_arrs) {
        route_node_map.set_route_id_and_intra_idx(pickup.node_info(), *route_id, i + 1);
      }
      // Update size
      *n_nodes += 2;
    }

    // this should be called after an ejection or insertion is executed
    DI void compute_intra_indices(typename route_node_map_t<i_t>::view_t& route_node_map)
    {
      for (i_t i = threadIdx.x; i < *n_nodes; i += blockDim.x) {
        auto& node_info = requests().node_info[i];
        route_node_map.set_intra_route_idx(node_info, i);
      }
    }

    // copies forward and backward data from another route for given interval
    // TODO load the data that is required for computations
    DI void copy_forward_data(const view_t& orig_route, i_t start_idx, i_t end_idx, i_t write_start)
    {
      cuopt_assert(end_idx >= start_idx, "End index should be greater than to start index");
      loop_over_dimensions(dimensions_info(), [&] __device__(auto I) {
        get_dimension_of<I>(dimensions)
          .copy_forward_data(
            get_dimension_of<I>(orig_route.dimensions), start_idx, end_idx, write_start);
      });
    }

    DI void copy_backward_data(const view_t& orig_route,
                               i_t start_idx,
                               i_t end_idx,
                               i_t write_start)
    {
      cuopt_assert(end_idx >= start_idx, "End index should be greater than to start index");
      loop_over_dimensions(dimensions_info(), [&] __device__(auto I) {
        get_dimension_of<I>(dimensions)
          .copy_backward_data(
            get_dimension_of<I>(orig_route.dimensions), start_idx, end_idx, write_start);
      });
    }

    // Copies data from start_idx to end_idx (not included) in orig_route
    // To current route + write_start
    DI void copy_fixed_route_data(const view_t& orig_route,
                                  i_t start_idx,
                                  i_t end_idx,
                                  i_t write_start)
    {
      cuopt_assert(start_idx <= end_idx, "From index should be smaller than to index");
      requests().copy_fixed_route_data(orig_route.requests(), start_idx, end_idx, write_start);
      loop_over_dimensions(dimensions_info(), [&] __device__(auto I) {
        get_dimension_of<I>(dimensions)
          .copy_fixed_route_data(
            get_dimension_of<I>(orig_route.dimensions), start_idx, end_idx, write_start);
      });
    }

    DI void copy_from(const view_t& orig_route, i_t start_idx, i_t end_idx, i_t write_start)
    {
      cuopt_assert(start_idx <= end_idx, "From index should be smaller than to index");
      copy_fixed_route_data(orig_route, start_idx, end_idx, write_start);
      copy_forward_data(orig_route, start_idx, end_idx, write_start);
      copy_backward_data(orig_route, start_idx, end_idx, write_start);
    }

    DI void copy_from(const view_t& orig_route)
    {
      copy_from(orig_route, 0, *orig_route.n_nodes + 1, 0);
      block_copy(infeasibility_cost, orig_route.infeasibility_cost, 1);
      block_copy(objective_cost, orig_route.objective_cost, 1);
      block_copy(n_nodes, orig_route.n_nodes, 1);
      block_copy(route_id, orig_route.route_id, 1);
      block_copy(vehicle_id, orig_route.vehicle_id, 1);
    }

    template <request_t r_t, std::enable_if_t<r_t == request_t::VRP, bool> = true>
    DI void copy_route_data_after_ejection(const view_t& orig_route,
                                           i_t intra_idx,
                                           bool copy_forward = false)
    {
      if (copy_forward) {
        copy_forward_data(orig_route, 0, intra_idx, 0);
        copy_forward_data(orig_route, intra_idx + 1, *orig_route.n_nodes + 1, intra_idx);
      }
      copy_fixed_route_data(orig_route, 0, intra_idx, 0);
      copy_fixed_route_data(orig_route, intra_idx + 1, *orig_route.n_nodes + 1, intra_idx);

      __syncthreads();
    }

    template <request_t r_t, std::enable_if_t<r_t == request_t::PDP, bool> = true>
    DI void copy_route_data_after_ejection(const view_t& orig_route,
                                           i_t intra_idx,
                                           i_t delivery_intra_idx,
                                           bool copy_forward = false)
    {
      if (copy_forward) {
        copy_forward_data(orig_route, 0, intra_idx, 0);
        copy_forward_data(orig_route, intra_idx + 1, delivery_intra_idx, intra_idx);
        copy_forward_data(
          orig_route, delivery_intra_idx + 1, *orig_route.n_nodes + 1, delivery_intra_idx - 1);
      }
      copy_fixed_route_data(orig_route, 0, intra_idx, 0);
      cuopt_assert(intra_idx < delivery_intra_idx,
                   "Intra pickup and delivery indices should be in stricly increasing order");
      copy_fixed_route_data(orig_route, intra_idx + 1, delivery_intra_idx, intra_idx);
      // + 1 to include end depot
      // Start index : intra_idx + (delivery_intra_idx - intra_idx - 1)
      copy_fixed_route_data(
        orig_route, delivery_intra_idx + 1, *orig_route.n_nodes + 1, delivery_intra_idx - 1);

      __syncthreads();
    }

    // Copies in current route data from orig_route
    // Nodes contained in intra_indices are ejected and thus not copied
    DI void copy_route_data_after_ejections(const view_t& orig_route,
                                            const i_t* intra_ejection_indices,
                                            i_t n_ejections)
    {
      cuopt_assert(intra_ejection_indices != nullptr, "Intra indices array should not be nullptr");
      cuopt_assert((n_ejections >= request_info_t<i_t, REQUEST>::size()),
                   "Number of ejection should be at least two");
      cuopt_assert((n_ejections % request_info_t<i_t, REQUEST>::size()) == 0,
                   "Number of ejection should be even");
      cuopt_assert(0 < intra_ejection_indices[0],
                   "Intra indices should be in stricly increasing order");

      const auto route_length = *orig_route.n_nodes;

      if (threadIdx.x == 0) { *n_nodes = *orig_route.n_nodes - n_ejections; }

      // Just need to initilize forward data of beginning
      copy_forward_data(orig_route, 0, intra_ejection_indices[0], 0);
      copy_fixed_route_data(orig_route, 0, intra_ejection_indices[0], 0);

      i_t curr_size = intra_ejection_indices[0];

      for (int i = 0; i < n_ejections - 1; ++i) {
        const i_t curr = intra_ejection_indices[i];
        const i_t next = intra_ejection_indices[i + 1];
        cuopt_assert(curr < next, "Intra indices should be in stricly increasing order");
        copy_fixed_route_data(orig_route, curr + 1, next, curr_size);
        curr_size += next - curr - 1;
      }

      // + 1 to go up to end depot
      cuopt_assert(intra_ejection_indices[n_ejections - 1] < route_length + 1,
                   "Intra indices should be in stricly increasing order");
      copy_fixed_route_data(
        orig_route, intra_ejection_indices[n_ejections - 1] + 1, route_length + 1, curr_size);

      // Just need to initilize backward data of end
      copy_backward_data(
        orig_route, intra_ejection_indices[n_ejections - 1] + 1, route_length + 1, curr_size);
    }

    // extend for other things later
    DI thrust::tuple<objective_cost_t, infeasible_cost_t> compute_cost(
      bool check_single_threaded = true)
    {
      cuopt_assert(!check_single_threaded || __popc(__activemask()) == 1,
                   "Compute cost should be single threaded");

      // zero-out the cost. Since some routes are stored and shared memory and not explicitly
      // zeroed-out, we have to do it here.
      objective_cost[0].zero_initialize();
      infeasibility_cost[0].zero_initialize();

      loop_over_dimensions(dimensions_info(), [&](auto I) {
        get_dimension_of<I>(dimensions)
          .compute_cost(this->vehicle_info(), *n_nodes, objective_cost[0], infeasibility_cost[0]);
      });

      return thrust::make_tuple(objective_cost[0], infeasibility_cost[0]);
    }

    DI double get_weighted_excess(const infeasible_cost_t weights) const
    {
      return infeasible_cost_t::dot(weights, *infeasibility_cost);
    }

    DI double get_cost(const bool include_objective, const infeasible_cost_t weights) const
    {
      double total_cost = infeasible_cost_t::dot(weights, infeasibility_cost[0]);
      if (include_objective) {
        double obj_cost =
          objective_cost_t::dot(dimensions_info().objective_weights, objective_cost[0]);
        total_cost += obj_cost;
      }
      return total_cost;
    }

    DI infeasible_cost_t get_infeasibility_cost() const { return *infeasibility_cost; }
    DI objective_cost_t get_objective_cost() const { return *objective_cost; }

    // extend for other things later
    DI i_t max_nodes_per_route() const noexcept { return requests().node_info.size(); }

    DI f_t get_time_between(NodeInfo<i_t> prev, NodeInfo<i_t> next) const
    {
      return get_transit_time(prev, next, vehicle_info(), true);
    }

    /**
     * @brief Creates a shared route just with required computation data
     *
     * @param shmem The shared memory region that will be aliased
     * @param orig_route Original route from which the meta data will be read
     * @param n_nodes Number of nodes to be allocated in the temporary route
     * @param new_route_id Temporary route id, don't read it from original route as it might be
     * different
     * @return route view
     */
    static DI view_t create_shared_route(i_t* shmem,
                                         const view_t orig_route,
                                         i_t n_nodes_route,
                                         bool is_tsp = false)
    {
      view_t v;
      v.infeasibility_cost = (infeasible_cost_t*)shmem;
      v.objective_cost     = (objective_cost_t*)&v.infeasibility_cost[1];
      i_t* sh_ptr          = (i_t*)&v.objective_cost[1];

      thrust::tie(v.dimensions, sh_ptr) =
        dimensions_route_t<i_t, f_t, REQUEST>::view_t::create_shared_route(
          sh_ptr, orig_route.dimensions_info(), n_nodes_route, is_tsp);

      v.n_nodes    = (i_t*)sh_ptr;
      v.route_id   = (i_t*)&v.n_nodes[1];
      v.vehicle_id = (i_t*)&v.route_id[1];

      // vehicle info will still be in global memory
      v.fleet_info = orig_route.fleet_info;
      if (threadIdx.x == 0) {
        *v.n_nodes    = n_nodes_route;
        *v.route_id   = *orig_route.route_id;
        *v.vehicle_id = *orig_route.vehicle_id;
      }
      return v;
    }

    DI unsigned long shared_end_address()
    {
      // address of last item
      return reinterpret_cast<unsigned long>(&vehicle_id[1]);
    }

    static DI void compute_forward_in_between(view_t& curr_route, i_t start, i_t end)
    {
      cuopt_assert(start >= 0, "Start has to be positive.");
      cuopt_assert(end > 0, "End has to be stricly positive.");
      cuopt_assert(end >= start, "End should be bigger than start.");
      cuopt_assert(start <= curr_route.get_num_nodes(), "Start should be smaller than n_nodes+1.");
      auto curr_node = curr_route.get_node(start);
      while (start < end) {
        auto next_node = curr_route.get_node(start + 1);
        curr_node.calculate_forward_all(next_node, curr_route.vehicle_info());
        curr_route.set_forward_data(start + 1, next_node);
        curr_node = next_node;
        ++start;
      }
    }

    static DI void compute_forward(view_t& curr_route, i_t start = 0)
    {
      compute_forward_in_between(curr_route, start, curr_route.get_num_nodes());
    }

    static DI void compute_backward_in_between(view_t& curr_route, i_t start, i_t end)
    {
      cuopt_assert(start >= 0, "Start has to be positive.");
      cuopt_assert(end > 0, "End has to be stricly positive.");
      cuopt_assert(end >= start, "End should be bigger than start.");
      cuopt_assert(end <= curr_route.get_num_nodes(), "End should be smaller than n_nodes+1.");
      auto curr_node = curr_route.get_node(end);
      while (end > start) {
        auto prev_node = curr_route.get_node(end - 1);
        curr_node.calculate_backward_all(prev_node, curr_route.vehicle_info());
        curr_route.set_backward_data(end - 1, prev_node);
        curr_node = prev_node;
        --end;
      }
    }

    static DI void compute_backward(view_t& curr_route, i_t start = 0)
    {
      compute_backward_in_between(curr_route, start, curr_route.get_num_nodes());
    }

    DI i_t get_num_nodes() const { return n_nodes[0]; }

    DI i_t get_id() const { return route_id[0]; }

    DI void set_num_nodes(i_t num_nodes) { n_nodes[0] = num_nodes; }

    DI void set_id(i_t id_) { route_id[0] = id_; }

    DI i_t get_vehicle_id() const { return vehicle_id[0]; }

    DI void set_vehicle_id(i_t id_) { vehicle_id[0] = id_; }

    DI const i_t* get_num_nodes_ptr() const { return n_nodes; }

    DI void reset()
    {
      set_id(-1);
      set_vehicle_id(-1);
      set_num_nodes(-1);
    }

    static DI void compute_forward_backward_cost(view_t& curr_route)
    {
      // first thread does forward and cost
      if (threadIdx.x == 0) {
        route_t<i_t, f_t, REQUEST>::view_t::compute_forward(curr_route);
        curr_route.compute_cost();
      }
      // last thread does backward
      if (threadIdx.x == blockDim.x - 1) {
        route_t<i_t, f_t, REQUEST>::view_t::compute_backward(curr_route);
      }
    }

    DI void compute_actual_arrival_time()
    {
      cuopt_assert(__popc(__activemask()) == 1,
                   "compute_actual_arrival_time should be single threaded");
      // start time is always greater than vehicle earliest
      // if there is an excess in the begining add it
      double time_stamp =
        dimensions.time_dim.departure_forward[0] + dimensions.time_dim.excess_forward[0];

      if (dimensions_info().time_dim.should_compute_travel_time()) {
        time_stamp = dimensions.time_dim.earliest_arrival_backward[0];
      }

      // printf("latest arrival backward = %e\n", );
      for (i_t i = 0; i <= *n_nodes; ++i) {
        time_stamp = max(time_stamp, dimensions.time_dim.window_start[i]);
        dimensions.time_dim.actual_arrival[i] = time_stamp;
        if (i != *n_nodes) {
          time_stamp += get_transit_time(dimensions.requests.node_info[i],
                                         dimensions.requests.node_info[i + 1],
                                         vehicle_info(),
                                         true);
        }
      }
    }

    typename dimensions_route_t<i_t, f_t, REQUEST>::view_t dimensions;

   private:
    i_t* n_nodes{nullptr};
    i_t* route_id{nullptr};
    i_t* vehicle_id{nullptr};
    infeasible_cost_t* infeasibility_cost{nullptr};
    objective_cost_t* objective_cost{nullptr};
    typename fleet_info_t<i_t, f_t>::view_t fleet_info;
  };

  view_t view()
  {
    view_t v = view_t::create_view(n_nodes.data(),
                                   route_id.data(),
                                   vehicle_id.data(),
                                   infeasibility_cost.data(),
                                   objective_cost.data(),
                                   fleet_info_ptr->view());

    v.dimensions = dimensions.view();
    return v;
  }

  /**
   * @brief Get the shared memory size required to store a route of a given size
   *
   * @param route_size
   * @return size_t
   */
  HDI size_t static get_shared_size(i_t route_size, enabled_dimensions_t dimensions_info)
  {
    // everything that is stored in rmm::device_scalar should be stored in shared
    size_t sz = 3 * sizeof(i_t)  // route_id, vehicle_id, n_nodes
                + sizeof(infeasible_cost_t) +
                sizeof(objective_cost_t);  // infeasibility cost, objective cost
    sz += dimensions_route_t<i_t, f_t, REQUEST>::get_shared_size(route_size, dimensions_info);
    return sz;
  }

  // solution handle
  solution_handle_t<i_t, f_t> const* sol_handle;

  // route structure that holds all dimensions data
  dimensions_route_t<i_t, f_t, REQUEST> dimensions;

  // route id
  rmm::device_scalar<i_t> route_id;

  // vehicle id to handle heterogenous properties
  rmm::device_scalar<i_t> vehicle_id;

  // number of nodes
  rmm::device_scalar<i_t> n_nodes;
  // device cost info, this cost contains excess cost as well
  rmm::device_scalar<infeasible_cost_t> infeasibility_cost;

  rmm::device_scalar<objective_cost_t> objective_cost;

  // fleet info
  const fleet_info_t<i_t, f_t>* fleet_info_ptr;
};

}  // namespace detail
}  // namespace routing
}  // namespace cuopt
