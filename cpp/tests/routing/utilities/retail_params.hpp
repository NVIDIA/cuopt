/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#pragma once

struct retail_params_t {
  retail_params_t& set_drop_return_trip()
  {
    drop_return_trip = true;
    return *this;
  }

  retail_params_t& set_multi_capacity()
  {
    multi_capacity = true;
    return *this;
  }

  retail_params_t& set_vehicle_tw()
  {
    vehicle_tw = true;
    return *this;
  }

  retail_params_t& set_vehicle_lower_bound(int val)
  {
    vehicle_lower_bound = val;
    return *this;
  }

  retail_params_t& set_pickup()
  {
    pickup = true;
    return *this;
  }

  retail_params_t& set_vehicle_breaks()
  {
    vehicle_breaks = true;
    return *this;
  }

  retail_params_t& set_vehicle_max_costs()
  {
    vehicle_max_costs = true;
    return *this;
  }

  retail_params_t& set_vehicle_max_times()
  {
    vehicle_max_times = true;
    return *this;
  }

  retail_params_t& set_vehicle_fixed_costs()
  {
    vehicle_fixed_costs = true;
    return *this;
  }

  bool drop_return_trip{false};
  bool multi_capacity{false};
  bool vehicle_tw{false};
  int vehicle_lower_bound{0};
  bool pickup{false};
  bool vehicle_breaks{false};
  bool vehicle_max_costs{false};
  bool vehicle_max_times{false};
  bool vehicle_fixed_costs{false};
};
