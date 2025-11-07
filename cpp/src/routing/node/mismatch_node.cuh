/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#pragma once
#include <algorithm>

namespace cuopt {
namespace routing {
namespace detail {

template <typename i_t, typename f_t>
class mismatch_node_t {
 public:
  //! Mismatch gathered to node
  i_t mismatch_forward = 0;
  //! Mismatch gathered after node
  i_t mismatch_backward = 0;

  /*! \brief { Calculate next node forward gathered distance data based on actual node} */
  void HDI calculate_forward(mismatch_node_t& next, f_t mismatch_between) const noexcept
  {
    next.mismatch_forward = mismatch_forward + mismatch_between;
  }

  /*! \brief { Calculate prev node gathered distance backward data based on actual node} */
  void HDI calculate_backward(mismatch_node_t& prev, f_t mismatch_between) const noexcept
  {
    prev.mismatch_backward = mismatch_backward + mismatch_between;
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

  /*! \brief  { Combine information from begining and ending fragments.}
      \return { Distance excess of route represented by nodes prev and next }*/
  static HDI double combine(const mismatch_node_t& prev,
                            const mismatch_node_t& next,
                            [[maybe_unused]] const VehicleInfo<f_t>& vehicle_info,
                            double mismatch_between) noexcept
  {
    return prev.mismatch_forward + next.mismatch_backward + mismatch_between;
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
                    [[maybe_unused]] const mismatch_dimension_info_t& dim_info,
                    [[maybe_unused]] objective_cost_t& obj_cost,
                    infeasible_cost_t& inf_cost) const noexcept
  {
    inf_cost[dim_t::MISMATCH] = ((double)mismatch_forward + (double)mismatch_backward);
  }
};

}  // namespace detail
}  // namespace routing
}  // namespace cuopt
