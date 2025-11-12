/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

namespace cuopt::routing {

enum class config_t : int { DEFAULT, CVRP, TSP };

template <typename i_t>
struct diversity_config_t {
  template <config_t CONFIG>
  static constexpr i_t min_island_size()
  {
    if constexpr (CONFIG == config_t::DEFAULT) {
      return 3;
    } else if constexpr (CONFIG == config_t::TSP) {
      return 10;
    } else {
      return population_size<CONFIG>() / 2;
    }
  }

  template <config_t CONFIG>
  static constexpr i_t population_size()
  {
    return 16;
  }

  template <config_t CONFIG, std::enable_if_t<CONFIG == config_t::CVRP, bool> = true>
  static constexpr i_t island_size()
  {
    return 5;
  }
};

}  // namespace cuopt::routing
