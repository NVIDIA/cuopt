/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2025, NVIDIA CORPORATION & AFFILIATES. All rights
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

namespace cuopt {
namespace routing {
namespace detail {

enum class search_type_t { IMPROVE = 0, RANDOM, CROSS };

union cost_counter_t {
  uint64_t counter;
  double cost;
};

struct __align__(16) cand_t {
  HDI cand_t(uint pair_1_, uint pair_2_, uint64_t counter_) : pair_1(pair_1_), pair_2(pair_2_)
  {
    cost_counter.counter = counter_;
  }

  HDI cand_t(uint pair_1_, uint pair_2_, double cost_) : pair_1(pair_1_), pair_2(pair_2_)
  {
    cost_counter.cost = cost_;
  }

  HDI cand_t() : pair_1(0), pair_2(0) { cost_counter.cost = std::numeric_limits<double>::max(); }

  HDI bool is_valid(search_type_t search_type) const
  {
    return search_type == search_type_t::RANDOM
             ? cost_counter.counter > 1ULL
             : cost_counter.cost < std::numeric_limits<double>::max();
  }

  DI bool operator==(cand_t const& cand) const noexcept
  {
    return pair_1 == cand.pair_1 && pair_2 == cand.pair_2 &&
           __double_as_longlong(cost_counter.cost) == __double_as_longlong(cand.cost_counter.cost);
  }

  DI bool operator!=(cand_t const& cand) const noexcept { return !(*this == cand); }

  template <search_type_t search_type>
  static HDI cand_t create()
  {
    cand_t candidate;
    if constexpr (search_type == search_type_t::RANDOM) { candidate.cost_counter.counter = 1ULL; }
    return candidate;
  }

  uint pair_1;
  uint pair_2;
  cost_counter_t cost_counter;
};

struct cross_cand_t : public cand_t {
  HDI cross_cand_t(uint pair_1_, uint pair_2_, uint64_t counter_, int id_1_, int id_2_)
    : cand_t(pair_1_, pair_2_, counter_), id_1(id_1_), id_2(id_2_)
  {
  }

  HDI cross_cand_t(uint pair_1_, uint pair_2_, double cost_, int id_1_, int id_2_)
    : cand_t(pair_1_, pair_2_, cost_), id_1(id_1_), id_2(id_2_)
  {
  }

  HDI cross_cand_t() : cand_t(), id_1(0), id_2(0) {}

  HDI cross_cand_t(const cand_t& cand, int id_1_, int id_2_)
    : cand_t(cand), id_1(id_1_), id_2(id_2_)
  {
  }

  int id_1;
  int id_2;
};

HDI bool operator<(const cand_t& c1, const cand_t& c2)
{
  return c1.cost_counter.cost < c2.cost_counter.cost;
}

}  // namespace detail
}  // namespace routing
}  // namespace cuopt
