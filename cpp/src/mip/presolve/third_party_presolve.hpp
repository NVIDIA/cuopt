/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#pragma once

#include <optional>
#include <vector>

#include <cuopt/linear_programming/optimization_problem.hpp>

namespace cuopt::linear_programming::detail {

template <typename i_t, typename f_t>
struct third_party_presolve_result_t {
  optimization_problem_t<i_t, f_t> reduced_problem;
  std::vector<i_t> implied_integer_indices;
  std::vector<i_t> reduced_to_original_map;
  std::vector<i_t> original_to_reduced_map;
  // clique info, etc...
};

template <typename i_t, typename f_t>
class third_party_presolve_t {
 public:
  third_party_presolve_t() = default;

  std::optional<third_party_presolve_result_t<i_t, f_t>> apply(
    optimization_problem_t<i_t, f_t> const& op_problem,
    problem_category_t category,
    bool dual_postsolve,
    f_t absolute_tolerance,
    f_t relative_tolerance,
    double time_limit,
    i_t num_cpu_threads = 0);

  void undo(rmm::device_uvector<f_t>& primal_solution,
            rmm::device_uvector<f_t>& dual_solution,
            rmm::device_uvector<f_t>& reduced_costs,
            problem_category_t category,
            bool status_to_skip,
            bool dual_postsolve,
            rmm::cuda_stream_view stream_view);

  void uncrush_primal_solution(const std::vector<f_t>& reduced_primal,
                               std::vector<f_t>& full_primal) const;
  const std::vector<i_t>& get_reduced_to_original_map() const { return reduced_to_original_map_; }
  const std::vector<i_t>& get_original_to_reduced_map() const { return original_to_reduced_map_; }

 private:
  std::vector<i_t> reduced_to_original_map_{};
  std::vector<i_t> original_to_reduced_map_{};
};

}  // namespace cuopt::linear_programming::detail
