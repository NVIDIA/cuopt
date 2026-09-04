/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#pragma once
#include <math.h>
#include <algorithm>

namespace cuopt {
namespace routing {
namespace detail {

template <typename i_t, typename f_t>
class mismatch_node_t {
 public:
  //! Infeasible vehicle-order assignment count accumulated forward
  i_t mismatch_forward = 0;
  //! Infeasible vehicle-order assignment count accumulated backward
  i_t mismatch_backward = 0;
  //! Finite vehicle-order assignment costs accumulated forward
  double cost_forward = 0.;
  //! Finite vehicle-order assignment costs accumulated backward
  double cost_backward = 0.;

  /*! \brief { Calculate next node forward state based on arc from this->next.
               If arc is infinite, increment mismatch; otherwise accumulate cost. } */
  template <bool is_device = true>
  void HDI calculate_forward(mismatch_node_t& next, double arc) const noexcept
  {
    if (isinf(arc)) {
      next.mismatch_forward = mismatch_forward + 1;
      next.cost_forward     = cost_forward;
    } else {
      next.mismatch_forward = mismatch_forward;
      next.cost_forward     = cost_forward + arc;
    }
  }

  /*! \brief { Calculate prev node backward state based on arc from prev->this.
               If arc is infinite, increment mismatch; otherwise accumulate cost. } */
  template <bool is_device = true>
  void HDI calculate_backward(mismatch_node_t& prev, double arc) const noexcept
  {
    if (isinf(arc)) {
      prev.mismatch_backward = mismatch_backward + 1;
      prev.cost_backward     = cost_backward;
    } else {
      prev.mismatch_backward = mismatch_backward;
      prev.cost_backward     = cost_backward + arc;
    }
  }

  HDI double forward_excess([[maybe_unused]] const VehicleInfo<f_t>& vehicle_info) const noexcept
  {
    return mismatch_forward;
  }

  HDI double backward_excess([[maybe_unused]] const VehicleInfo<f_t>& vehicle_info) const noexcept
  {
    return mismatch_backward;
  }

  HDI bool forward_feasible([[maybe_unused]] const VehicleInfo<f_t>& vehicle_info,
                            const double weight       = 1.,
                            const double excess_limit = 0.) const noexcept
  {
    return mismatch_forward * weight <= excess_limit;
  }

  /*! \brief  { Combine infeasibility from prefix (prev) and suffix (next) with arc at the join.
                Infeasible arc contributes 1 to mismatch; finite arcs are already in fwd/bwd costs.
     }
      \return { Infeasibility at the split point } */
  static HDI double combine(const mismatch_node_t& prev,
                            const mismatch_node_t& next,
                            [[maybe_unused]] const VehicleInfo<f_t>& vehicle_info,
                            double arc) noexcept
  {
    return prev.mismatch_forward + next.mismatch_backward + (isinf(arc) ? 1 : 0);
  }

  HDI bool backward_feasible([[maybe_unused]] const VehicleInfo<f_t>& vehicle_info,
                             const double weight       = 1.,
                             const double excess_limit = 0.) const noexcept
  {
    return mismatch_backward * weight <= excess_limit;
  }

  template <bool is_device = true>
  HDI void get_cost([[maybe_unused]] const mismatch_node_t& prev_node,
                    [[maybe_unused]] const VehicleInfo<f_t, is_device>& vehicle_info,
                    const mismatch_dimension_info_t& dim_info,
                    objective_cost_t& obj_cost,
                    infeasible_cost_t& inf_cost) const noexcept
  {
    inf_cost[dim_t::MISMATCH] = ((double)mismatch_forward + (double)mismatch_backward);
    if (dim_info.has_vehicle_order_cost) {
      obj_cost[objective_t::VEHICLE_ORDER_COST] = cost_forward + cost_backward;
    }
  }
};

}  // namespace detail
}  // namespace routing
}  // namespace cuopt
