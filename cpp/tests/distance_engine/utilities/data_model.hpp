/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */
#pragma once

#include <vector>
namespace cuopt {
namespace distance_engine {
namespace test {
template <typename i_t, typename f_t>
struct first_test_data_model_t {
  std::vector<i_t> offsets{0, 2, 3, 4, 6, 8, 9, 10};
  std::vector<i_t> indices{1, 6, 4, 3, 2, 4, 2, 6, 4, 0};
  std::vector<f_t> weights{2, 10, 3, 2, 2, 5, 1, 1, 2, 10};
  std::vector<f_t> custom_weights{
    1, 10000000, 10, 1000, 1000, 10000, 100, 100000, 1000000, 10000000};
  std::vector<f_t> expected_custom_matrix{
    0, 1111, 100011, 10110000, 0, 110000, 10000000, 10001111, 0};

  std::vector<i_t> target_locations{0, 3, 6};
  std::vector<i_t> locations{0, 1, 2};

  std::vector<f_t> expected_cost_matrix{0, 8, 6, 16, 0, 6, 10, 18, 0};
  std::vector<i_t> expected_full_path{0, 1, 4, 2, 3, 3, 4, 6};
  std::vector<i_t> expected_sequence_offsets{0, 5, static_cast<i_t>(expected_full_path.size())};
};
static first_test_data_model_t<int, float> first_input_;

template <typename i_t, typename f_t>
struct second_test_data_model_t {
  std::vector<i_t> offsets{0, 3, 5, 7, 8, 9};
  std::vector<i_t> indices{1, 2, 3, 0, 2, 0, 3, 4, 0};
  std::vector<f_t> weights{1, 2, 3, 4, 5, 6, 7, 8, 9};

  std::vector<i_t> target_locations{0, 1, 2, 4};
  std::vector<i_t> locations{0, 2, 3, 0, 0, 1, 0};

  std::vector<f_t> expected_cost_matrix{
    0.0, 1.0, 2.0, 11.0, 4.0, 0.0, 5.0, 15.0, 6.0, 7.0, 0.0, 15.0, 9.0, 10.0, 11.0, 0.0};
  std::vector<i_t> expected_full_path{0, 2, 2, 3, 4, 4, 0, 0, 0, 1, 1, 0};
  std::vector<i_t> expected_sequence_offsets{0, 2, 5, 7, 8, 10, 12};
};

static second_test_data_model_t<int, float> second_input_;

}  // namespace test
}  // namespace distance_engine
}  // namespace cuopt
