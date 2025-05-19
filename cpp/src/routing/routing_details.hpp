/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2025, NVIDIA CORPORATION & AFFILIATES. All rights
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

#include <cuopt/error.hpp>
#include <cuopt/routing/routing_structures.hpp>

#include <raft/core/handle.hpp>
#include <rmm/device_uvector.hpp>

#include <iostream>
#include <ostream>
#include <string>
#include <vector>

namespace cuopt {
namespace routing {

namespace detail {

template <typename i_t, typename f_t>
class demand_container_t {
 public:
  demand_i_t* demand_arr;
  cap_i_t* capacities;
  bool* vehicle_order_match{nullptr};
  i_t n_dimensions;
  i_t fleet_size;
  i_t order_nodes;
  demand_container_t() {};
  ~demand_container_t() {};
};

template <typename i_t, typename f_t>
class precedence_container_t {
 public:
  precedence_container_t() = default;
  precedence_container_t(const i_t* _precedence,
                         const uint8_t* _precedence_bitmap,
                         const i_t* _n_precedence,
                         const i_t* _reverse_precedence,
                         const i_t* _n_reverse__precedence,
                         f_t* _service_end_times,
                         i_t _n_nodes,
                         i_t _n_orders,
                         f_t* _min_dependent_begin_times,
                         f_t* _tmp_service_end_times,
                         f_t* _tmp_min_dependent_begin_times,
                         i_t* _node_to_path_map)
  {
    if (_service_end_times != nullptr) {
      precedence                    = _precedence;
      precedence_bitmap             = _precedence_bitmap;
      n_precedence                  = _n_precedence;
      reverse_precedence            = _reverse_precedence;
      n_reverse_precedence          = _n_reverse__precedence;
      service_end_times             = _service_end_times;
      n_nodes                       = _n_nodes;
      n_orders                      = _n_orders;
      min_dependent_begin_times     = _min_dependent_begin_times;
      tmp_service_end_times         = _tmp_service_end_times;
      tmp_min_dependent_begin_times = _tmp_min_dependent_begin_times;
      node_to_path_map              = _node_to_path_map;
    }
  }

  bool is_empty() const noexcept { return precedence == nullptr; }

  f_t* service_end_times{nullptr};
  // this is the minimum of service begin times for a dependent node(a node that has precedence
  // constraint)
  f_t* min_dependent_begin_times{nullptr};
  f_t* tmp_service_end_times{nullptr};
  f_t* tmp_min_dependent_begin_times{nullptr};
  const i_t* n_precedence{nullptr};
  const i_t* precedence{nullptr};
  const i_t* n_reverse_precedence{nullptr};
  const i_t* reverse_precedence{nullptr};
  const uint8_t* precedence_bitmap{nullptr};
  i_t* node_to_path_map{nullptr};
  i_t n_nodes;
  i_t n_orders;
};

}  // namespace detail
}  // namespace routing
}  // namespace cuopt
