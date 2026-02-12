/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#pragma once

#include <cuopt/linear_programming/optimization_problem.hpp>

#include <thrust/sequence.h>
#include <thrust/uninitialized_fill.h>
#include <rmm/device_uvector.hpp>

namespace cuopt {
namespace linear_programming::detail {

template <typename i_t, typename f_t>
class problem_t;

template <typename i_t, typename f_t>
class solution_t;

template <typename i_t, typename f_t>
class third_party_presolve_t;

template <typename i_t, typename f_t>
struct substitution_t {
  f_t timestamp;
  i_t substituting_var;
  i_t substituted_var;
  f_t offset;
  f_t coefficient;
};

template <typename i_t, typename f_t>
class presolve_data_t {
 public:
  presolve_data_t(const optimization_problem_t<i_t, f_t>& problem, rmm::cuda_stream_view stream)
    : variable_offsets(problem.get_n_variables(), 0),
      additional_var_used(problem.get_n_variables(), false),
      additional_var_id_per_var(problem.get_n_variables(), -1),
      objective_offset(problem.get_objective_offset()),
      objective_scaling_factor(problem.get_objective_scaling_factor()),
      variable_mapping(0, stream),
      fixed_var_assignment(0, stream),
      var_flags(0, stream)
  {
  }

  presolve_data_t(const presolve_data_t& other, rmm::cuda_stream_view stream)
    : variable_offsets(other.variable_offsets),
      additional_var_used(other.additional_var_used),
      additional_var_id_per_var(other.additional_var_id_per_var),
      objective_offset(other.objective_offset),
      objective_scaling_factor(other.objective_scaling_factor),
      variable_mapping(other.variable_mapping, stream),
      fixed_var_assignment(other.fixed_var_assignment, stream),
      var_flags(other.var_flags, stream),
      papilo_presolve_ptr(other.papilo_presolve_ptr),
      papilo_reduced_to_original_map(other.papilo_reduced_to_original_map),
      papilo_original_to_reduced_map(other.papilo_original_to_reduced_map),
      papilo_original_num_variables(other.papilo_original_num_variables),
      variable_substitutions(other.variable_substitutions)
  {
  }

  void initialize_var_mapping(const problem_t<i_t, f_t>& problem, const raft::handle_t* handle_ptr)
  {
    variable_mapping.resize(problem.n_variables, handle_ptr->get_stream());
    thrust::sequence(
      handle_ptr->get_thrust_policy(), variable_mapping.begin(), variable_mapping.end());
    fixed_var_assignment.resize(problem.n_variables, handle_ptr->get_stream());
    thrust::uninitialized_fill(handle_ptr->get_thrust_policy(),
                               fixed_var_assignment.begin(),
                               fixed_var_assignment.end(),
                               0.);
    variable_substitutions.clear();
  }

  void reset_additional_vars(const problem_t<i_t, f_t>& problem, const raft::handle_t* handle_ptr)
  {
    variable_offsets.assign(problem.n_variables, 0);
    additional_var_used.assign(problem.n_variables, false);
    additional_var_id_per_var.assign(problem.n_variables, -1);
  }

  bool pre_process_assignment(problem_t<i_t, f_t>& problem, rmm::device_uvector<f_t>& assignment);
  void post_process_assignment(problem_t<i_t, f_t>& problem,
                               rmm::device_uvector<f_t>& current_assignment,
                               bool resize_to_original_problem = true);
  void post_process_solution(problem_t<i_t, f_t>& problem, solution_t<i_t, f_t>& solution);

  void set_papilo_presolve_data(const third_party_presolve_t<i_t, f_t>* presolver_ptr,
                                std::vector<i_t> reduced_to_original,
                                std::vector<i_t> original_to_reduced,
                                i_t original_num_variables);
  bool has_papilo_presolve_data() const { return papilo_presolve_ptr != nullptr; }
  i_t get_papilo_original_num_variables() const { return papilo_original_num_variables; }
  void papilo_uncrush_assignment(problem_t<i_t, f_t>& problem,
                                 rmm::device_uvector<f_t>& assignment) const;

  presolve_data_t(presolve_data_t&&)                 = default;
  presolve_data_t& operator=(presolve_data_t&&)      = default;
  presolve_data_t& operator=(const presolve_data_t&) = delete;

  // offsets of variables
  std::vector<f_t> variable_offsets;
  std::vector<bool> additional_var_used;
  std::vector<i_t> additional_var_id_per_var;
  f_t objective_offset;
  f_t objective_scaling_factor;

  rmm::device_uvector<i_t> variable_mapping;
  rmm::device_uvector<f_t> fixed_var_assignment;
  rmm::device_uvector<i_t> var_flags;

  const third_party_presolve_t<i_t, f_t>* papilo_presolve_ptr{nullptr};
  std::vector<i_t> papilo_reduced_to_original_map{};
  std::vector<i_t> papilo_original_to_reduced_map{};
  i_t papilo_original_num_variables{0};
  // Variable substitutions from probing: x_substituted = offset + coefficient * x_substituting
  // Applied in post_process_assignment to recover substituted variable values
  std::vector<substitution_t<i_t, f_t>> variable_substitutions;
};

}  // namespace linear_programming::detail
}  // namespace cuopt
