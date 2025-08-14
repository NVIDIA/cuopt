/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights
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

#include <cuopt/routing/distance_engine/waypoint_matrix.hpp>
#include <vector>
#include "utilities/data_model.hpp"

namespace cuopt {
namespace distance_engine {
namespace test {
template <typename i_t, typename f_t>
class base_test_t {
 public:
  raft::handle_t handle;
  waypoint_matrix_t<i_t, f_t> waypoint_matrix;

  std::vector<i_t> target_locations{};
};

template <typename T>
std::vector<T> parse_vector(std::vector<std::string> const& vect)
{
  std::vector<T> out(vect.size());

  for (size_t i = 0; i != vect.size(); ++i) {
    if constexpr (std::is_same_v<T, int>)
      out[i] = std::stoi(vect[i]);
    else if constexpr (std::is_same_v<T, float>)
      out[i] = std::stof(vect[i]);
  }

  return out;
}

template <typename offsets_t, typename indices_t, typename weights_t>
std::tuple<std::vector<offsets_t>, std::vector<indices_t>, std::vector<weights_t>>
parse_waypoint_matrix_file(
  std::tuple<std::vector<std::string>, std::vector<std::string>, std::vector<std::string>> const&
    waypoint_matrix_info)
{
  return {parse_vector<offsets_t>(std::get<0>(waypoint_matrix_info)),
          parse_vector<indices_t>(std::get<1>(waypoint_matrix_info)),
          parse_vector<weights_t>(std::get<2>(waypoint_matrix_info))};
}

template <typename test_param_t, typename i_t, typename f_t>
struct waypoint_matrix_params_t : public test_param_t {
  std::vector<i_t> offsets;
  std::vector<i_t> indices;
  std::vector<f_t> weights;
  std::vector<i_t> target_locations;
};

template <typename f_t>
struct cost_matrix_params_t {
  std::vector<f_t> cost_matrix;
};

template <typename i_t, typename f_t>
struct waypoint_sequence_params_t : public cost_matrix_params_t<f_t> {
  std::vector<i_t> locations;
  std::vector<i_t> full_path;
  std::vector<i_t> sequence_offsets;
};

template <typename f_t>
struct shortest_path_cost_params_t : public cost_matrix_params_t<f_t> {
  std::vector<f_t> custom_weights;
};

template <typename i_t, typename f_t>
waypoint_matrix_params_t<cost_matrix_params_t<f_t>, i_t, f_t> parse_tests(
  std::tuple<std::vector<i_t>, std::vector<i_t>, std::vector<f_t>> const& waypoint_matrix_info,
  std::vector<i_t> const& target_locations,
  std::vector<f_t> const& cost_matrix)
{
  return {std::move(cost_matrix),
          std::move(std::get<0>(waypoint_matrix_info)),
          std::move(std::get<1>(waypoint_matrix_info)),
          std::move(std::get<2>(waypoint_matrix_info)),
          std::move(target_locations)};
}

template <typename data_model_view_t, typename i_t = int, typename f_t = float>
static waypoint_matrix_params_t<waypoint_sequence_params_t<i_t, f_t>, i_t, f_t> parse_data_model(
  data_model_view_t&& data_model)
{
  return {
    data_model.expected_cost_matrix,
    data_model.locations,
    data_model.expected_full_path,
    data_model.expected_sequence_offsets,
    data_model.offsets,
    data_model.indices,
    data_model.weights,
    data_model.target_locations,
  };
}

template <typename data_model_view_t, typename i_t = int, typename f_t = float>
static waypoint_matrix_params_t<shortest_path_cost_params_t<f_t>, i_t, f_t>
parse_data_model_custom_weight(data_model_view_t&& data_model)
{
  return {
    data_model.expected_custom_matrix,
    data_model.custom_weights,
    data_model.offsets,
    data_model.indices,
    data_model.weights,
    data_model.target_locations,
  };
}

template <typename... Args>
std::vector<waypoint_matrix_params_t<waypoint_sequence_params_t<int, float>, int, float>>
parse_data_models(Args... args)
{
  return {parse_data_model(args)...};
}

template <typename... Args>
std::vector<waypoint_matrix_params_t<shortest_path_cost_params_t<float>, int, float>>
parse_data_models_custom_weight(Args... args)
{
  return {parse_data_model_custom_weight(args)...};
}

}  // namespace test
}  // namespace distance_engine
}  // namespace cuopt
