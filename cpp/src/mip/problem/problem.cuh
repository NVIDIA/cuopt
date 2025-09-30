/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights
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

// THIS IS LIKELY THE INNER-MOST INCLUDE
// FOR COMPILE TIME, WE SHOULD KEEP THE INCLUDES ON THIS HEADER MINIMAL

#include "host_helper.cuh"
#include "presolve_data.cuh"

#include <mip/logger.hpp>
#include <mip/relaxed_lp/lp_state.cuh>

#include <cuopt/linear_programming/mip/solver_settings.hpp>
#include <cuopt/linear_programming/optimization_problem.hpp>
#include <cuopt/linear_programming/utilities/internals.hpp>
#include "host_helper.cuh"
#include "problem_fixing.cuh"

#include <utilities/macros.cuh>

#include <raft/core/nvtx.hpp>
#include <raft/random/rng_device.cuh>
#include <raft/util/cuda_dev_essentials.cuh>
#include <raft/util/cudart_utils.hpp>
#include <rmm/device_scalar.hpp>
#include <rmm/device_uvector.hpp>

#include <dual_simplex/user_problem.hpp>

namespace cuopt {

namespace linear_programming::detail {

template <typename i_t, typename f_t>
class solution_t;

constexpr double OBJECTIVE_EPSILON = 1e-7;
constexpr double MACHINE_EPSILON   = 1e-7;
constexpr bool USE_REL_TOLERANCE   = true;

template <typename i_t, typename f_t>
class problem_t {
 public:
  problem_t(const optimization_problem_t<i_t, f_t>& problem,
            const typename mip_solver_settings_t<i_t, f_t>::tolerances_t tolerances_ = {});
  problem_t() = delete;
  // copy constructor
  problem_t(const problem_t<i_t, f_t>& problem);
  problem_t(const problem_t<i_t, f_t>& problem, bool no_deep_copy);
  problem_t(problem_t<i_t, f_t>&& problem) = default;
  problem_t& operator=(problem_t&&)        = default;
  void op_problem_cstr_body(const optimization_problem_t<i_t, f_t>& problem_);

  problem_t<i_t, f_t> get_problem_after_fixing_vars(
    rmm::device_uvector<f_t>& assignment,
    const rmm::device_uvector<i_t>& variables_to_fix,
    rmm::device_uvector<i_t>& variable_map,
    const raft::handle_t* handle_ptr);
  void remove_given_variables(problem_t<i_t, f_t>& original_problem,
                              rmm::device_uvector<f_t>& assignment,
                              rmm::device_uvector<i_t>& variable_map,
                              const raft::handle_t* handle_ptr);

  i_t get_n_binary_variables();
  void check_problem_representation(bool check_transposed       = false,
                                    bool check_mip_related_data = true);
  void recompute_auxilliary_data(bool check_representation = true);
  void compute_n_integer_vars();
  void compute_binary_var_table();
  void compute_related_variables(double time_limit);
  void fix_given_variables(problem_t<i_t, f_t>& original_problem,
                           rmm::device_uvector<f_t>& assignment,
                           const rmm::device_uvector<i_t>& variables_to_fix,
                           const raft::handle_t* handle_ptr);

  void insert_variables(variables_delta_t<i_t, f_t>& h_vars);
  void insert_constraints(constraints_delta_t<i_t, f_t>& h_constraints);
  void resize_variables(size_t size);
  void resize_constraints(size_t matrix_size, size_t constraint_size, size_t var_size);
  void preprocess_problem();
  bool pre_process_assignment(rmm::device_uvector<f_t>& assignment);
  void post_process_assignment(rmm::device_uvector<f_t>& current_assignment,
                               bool resize_to_original_problem = true);
  void post_process_solution(solution_t<i_t, f_t>& solution);
  void compute_transpose_of_problem();
  f_t get_user_obj_from_solver_obj(f_t solver_obj) const;
  void compute_integer_fixed_problem();
  void fill_integer_fixed_problem(rmm::device_uvector<f_t>& assignment,
                                  const raft::handle_t* handle_ptr);
  void copy_rhs_from_problem(const raft::handle_t* handle_ptr);
  rmm::device_uvector<f_t> get_fixed_assignment_from_integer_fixed_problem(
    const rmm::device_uvector<f_t>& assignment);
  bool is_integer(f_t val) const;
  bool integer_equal(f_t val1, f_t val2) const;

  void get_host_user_problem(
    cuopt::linear_programming::dual_simplex::user_problem_t<i_t, f_t>& user_problem) const;

  void add_cutting_plane_at_objective(f_t objective);
  void compute_vars_with_objective_coeffs();

  struct view_t {
    HDI std::pair<i_t, i_t> reverse_range_for_var(i_t v) const
    {
      cuopt_assert(v >= 0 && v < n_variables, "Variable should be within the range");
      return std::make_pair(reverse_offsets[v], reverse_offsets[v + 1]);
    }

    HDI std::pair<i_t, i_t> range_for_constraint(i_t c) const
    {
      return std::make_pair(offsets[c], offsets[c + 1]);
    }

    HDI std::pair<i_t, i_t> range_for_related_vars(i_t v) const
    {
      return std::make_pair(related_variables_offsets[v], related_variables_offsets[v + 1]);
    }

    HDI bool check_variable_within_bounds(i_t v, f_t val) const
    {
      const f_t int_tol = tolerances.integrality_tolerance;
      auto bounds       = variable_bounds[v];
      bool within_bounds =
        val <= (get_upper(bounds) + int_tol) && val >= (get_lower(bounds) - int_tol);
      return within_bounds;
    }

    HDI bool is_integer_var(i_t v) const { return var_t::INTEGER == variable_types[v]; }

    // check if the variable is integer according to the tolerances
    // specified for this problem
    HDI bool is_integer(f_t val) const
    {
      return raft::abs(round(val) - (val)) <= tolerances.integrality_tolerance;
    }
    HDI bool integer_equal(f_t val1, f_t val2) const
    {
      return raft::abs(val1 - val2) <= tolerances.integrality_tolerance;
    }

    HDI f_t get_random_for_var(i_t v, raft::random::PCGenerator& rng) const
    {
      cuopt_assert(var_t::INTEGER != variable_types[v],
                   "Random value can only be called on continuous values");
      auto bounds = variable_bounds[v];

      f_t val;
      if (isfinite(get_lower(bounds)) && isfinite(get_upper(bounds))) {
        f_t diff = get_upper(bounds) - get_lower(bounds);
        val      = diff * rng.next_float() + get_lower(bounds);
      } else {
        auto finite_bound = isfinite(get_lower(bounds)) ? get_lower(bounds) : get_upper(bounds);
        val               = finite_bound;
      }
      cuopt_assert(isfinite(get_lower(bounds)), "Value must be finite");
      return val;
    }

    using f_t2 = typename type_2<f_t>::type;
    typename mip_solver_settings_t<i_t, f_t>::tolerances_t tolerances;
    i_t n_variables;
    i_t n_integer_vars;
    i_t n_constraints;
    i_t nnz;

    raft::device_span<f_t> reverse_coefficients;
    raft::device_span<i_t> reverse_constraints;
    raft::device_span<i_t> reverse_offsets;

    raft::device_span<f_t> coefficients;
    raft::device_span<i_t> variables;
    raft::device_span<i_t> offsets;
    raft::device_span<f_t> objective_coefficients;
    raft::device_span<f_t2> variable_bounds;
    raft::device_span<f_t> constraint_lower_bounds;
    raft::device_span<f_t> constraint_upper_bounds;
    raft::device_span<var_t> variable_types;
    raft::device_span<i_t> is_binary_variable;
    raft::device_span<i_t> integer_indices;
    raft::device_span<i_t> binary_indices;
    raft::device_span<i_t> nonbinary_indices;
    raft::device_span<i_t> related_variables;
    raft::device_span<i_t> related_variables_offsets;
    f_t objective_offset;
    f_t objective_scaling_factor;
  };

  view_t view();

  const optimization_problem_t<i_t, f_t>* original_problem_ptr;
  const raft::handle_t* handle_ptr;
  std::shared_ptr<problem_t<i_t, f_t>> integer_fixed_problem = nullptr;
  rmm::device_uvector<i_t> integer_fixed_variable_map;

  std::function<void(const std::vector<f_t>&)> branch_and_bound_callback;

  typename mip_solver_settings_t<i_t, f_t>::tolerances_t tolerances{};
  i_t n_variables{0};
  i_t n_constraints{0};
  i_t nnz{0};
  i_t n_binary_vars{0};
  i_t n_integer_vars{0};
  bool maximize{false};
  bool is_binary_pb{false};
  bool empty{false};

  presolve_data_t<i_t, f_t> presolve_data;

  // original variable ids
  // this vector refers to the problem after any presolve or preprocessing
  // it is to have correct access to the parent problem when we fix some variables
  std::vector<i_t> original_ids;
  // reverse original ids
  std::vector<i_t> reverse_original_ids;

  // reverse CSR matrix
  rmm::device_uvector<f_t> reverse_coefficients;
  rmm::device_uvector<i_t> reverse_constraints;
  rmm::device_uvector<i_t> reverse_offsets;

  // original CSR matrix
  rmm::device_uvector<f_t> coefficients;
  rmm::device_uvector<i_t> variables;
  rmm::device_uvector<i_t> offsets;

  /** weights in the objective function */
  rmm::device_uvector<f_t> objective_coefficients;
  using f_t2 = typename type_2<f_t>::type;
  rmm::device_uvector<f_t2> variable_bounds;
  rmm::device_uvector<f_t> constraint_lower_bounds;
  rmm::device_uvector<f_t> constraint_upper_bounds;
  /* biggest between cstr lower and upper */
  rmm::device_uvector<f_t> combined_bounds;
  /** Type of each variable */
  rmm::device_uvector<var_t> variable_types;
  /** The indices of the integer variables */
  rmm::device_uvector<i_t> integer_indices;
  rmm::device_uvector<i_t> binary_indices;
  rmm::device_uvector<i_t> nonbinary_indices;
  /** table to quickly test wheter or not a variable is binary */
  rmm::device_uvector<i_t> is_binary_variable;
  /** for a given variable var_idx, all other variables
   *  which are involved in constraints that contain var_idx */
  rmm::device_uvector<i_t> related_variables;
  rmm::device_uvector<i_t> related_variables_offsets;
  /** names of each of the variables in the OP */
  std::vector<std::string> var_names{};
  /** names of each of the rows (aka constraints or objective) in the OP */
  std::vector<std::string> row_names{};
  /** name of the objective (only a single objective is currently allowed) */
  std::string objective_name;
  f_t objective_offset;
  bool is_scaled_{false};
  bool preprocess_called{false};
  // this LP state keeps the warm start data of some solution of
  // 1. Original problem: it is unchanged and part of it is used
  // to warm start slightly modified problems.
  // 2. Integer fixed problem: this is useful as the problem structure
  // is always the same and only the RHS changes. Using this helps in warm start.
  lp_state_t<i_t, f_t> lp_state;
  problem_fixing_helpers_t<i_t, f_t> fixing_helpers;
  bool cutting_plane_added{false};
  std::pair<std::vector<i_t>, std::vector<f_t>> vars_with_objective_coeffs;
};

}  // namespace linear_programming::detail
}  // namespace cuopt
