/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#pragma once

#include <raft/core/handle.hpp>
#include <utilities/cuda_helpers.cuh>
#include "../routing_helpers.cuh"
#include "routing/fleet_info.hpp"

#include <rmm/device_uvector.hpp>

namespace cuopt {
namespace routing {
namespace detail {

// static size of the arrays in capacity_node_t
constexpr int default_max_capacity_dim = 3;

template <typename i_t, typename f_t, int max_capacity_dim_ = default_max_capacity_dim>
class capacity_node_t {
 public:
  HDI capacity_node_t(const capacity_dimension_info_t& dimension_info)
    : n_capacity_dimensions(dimension_info.n_capacity_dimensions)
  {
  }

  HDI capacity_node_t() = delete;

  static constexpr int max_capacity_dim = max_capacity_dim_;
  //! Data copied from problem
  i_t demand[max_capacity_dim] = {0};
  //! Total commodity gathered from begining up to considered node (incl. the node) - forward
  //! calculation
  i_t gathered[max_capacity_dim] = {0};
  //! Max load of the vehicle before the node (incl. immediatly after the node) - forward
  //! calculation
  i_t max_to_node[max_capacity_dim] = {0};
  //! Max load after the node (considering only the final fragment) - backward calculation
  i_t max_after[max_capacity_dim] = {0};
  //! Dimensions of capacity
  uint8_t n_capacity_dimensions;

  /*! \brief { Calculate next node forward capacity data based on actual node
   *  \param[in] actual { Actual node data }
   *  \param[in][out] next { Has to contain the supply loaded/unloaded(negative load). Total
   * gathered and max_to_node will be filled.} */
  void HDI calculate_forward(capacity_node_t& next, [[maybe_unused]] f_t dummy = 0) const noexcept
  {
    constexpr_for<max_capacity_dim>([&](auto i) {
      if (i < n_capacity_dimensions) {
        next.gathered[i]    = gathered[i] + next.demand[i];
        next.max_to_node[i] = max(next.gathered[i], max_to_node[i]);
      }
    });
  }

  /*! \brief { Calculate prev node time backward data based on actual node}
   *           \param[in][out] prev { Has to contain the supply loaded/unloaded(negative load).
   * max_after will be filled.} */
  void HDI calculate_backward(capacity_node_t& prev, [[maybe_unused]] f_t dummy = 0) const noexcept
  {
    constexpr_for<max_capacity_dim>([&](auto i) {
      if (i < n_capacity_dimensions) {
        prev.max_after[i] = max(0, max(prev.demand[i], prev.demand[i] + max_after[i]));
      }
    });
  }

  /*! \brief  { Combine information from the begining and ending fragments.}
      \return { Capacity excess of route represented by nodes prev and next }*/
  static i_t HDI combine(const capacity_node_t& prev,
                         const capacity_node_t& next,
                         const VehicleInfo<f_t>& vehicle_info,
                         const f_t dummy = 0.) noexcept
  {
    i_t excess_of_route = 0;
    constexpr_for<max_capacity_dim>([&](auto i) {
      if (i < prev.n_capacity_dimensions) {
        excess_of_route += max(0,
                               max(prev.max_to_node[i], prev.gathered[i] + next.max_after[i]) -
                                 vehicle_info.capacities[i]);
      }
    });
    return excess_of_route;
  }

  HDI double forward_excess(const VehicleInfo<f_t>& vehicle_info) const noexcept
  {
    double excess = 0;
    constexpr_for<max_capacity_dim>([&](auto i) {
      if (i < n_capacity_dimensions) {
        excess += max(0, max_to_node[i] - vehicle_info.capacities[i]);
      }
    });
    return excess;
  }

  HDI double backward_excess(const VehicleInfo<f_t>& vehicle_info) const noexcept
  {
    double excess = 0;
    constexpr_for<max_capacity_dim>([&](auto i) {
      if (i < n_capacity_dimensions) {
        excess += max(0, max_after[i] - vehicle_info.capacities[i]);
      }
    });
    return excess;
  }

  HDI bool forward_feasible(const VehicleInfo<f_t>& vehicle_info,
                            const double weight    = 1.,
                            const f_t excess_limit = 0.) const noexcept
  {
    return forward_excess(vehicle_info) * weight <= excess_limit;
  }

  HDI bool backward_feasible(const VehicleInfo<f_t>& vehicle_info,
                             const double weight    = 1.,
                             const f_t excess_limit = 0.) const noexcept
  {
    return backward_excess(vehicle_info) * weight <= excess_limit;
  }

  template <bool is_device = true>
  HDI void get_cost([[maybe_unused]] const capacity_node_t& prev,
                    const VehicleInfo<f_t, is_device>& vehicle_info,
                    const capacity_dimension_info_t& dim_info,
                    [[maybe_unused]] objective_cost_t& obj_cost,
                    infeasible_cost_t& inf_cost) const noexcept
  {
    double infeasibility_cost = 0.;
    constexpr_for<max_capacity_dim>([&](auto i) {
      if (i < dim_info.n_capacity_dimensions) {
        i_t prev_gathered = gathered[i] - demand[i];
        infeasibility_cost +=
          max(0, max(max_to_node[i], max_after[i] + prev_gathered) - vehicle_info.capacities[i]);
      }
    });
    inf_cost[dim_t::CAP] = infeasibility_cost;
  }
};

}  // namespace detail
}  // namespace routing
}  // namespace cuopt
