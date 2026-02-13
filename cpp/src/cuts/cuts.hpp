/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */
#pragma once

#include <dual_simplex/basis_updates.hpp>
#include <dual_simplex/presolve.hpp>
#include <dual_simplex/simplex_solver_settings.hpp>
#include <dual_simplex/sparse_vector.hpp>
#include <dual_simplex/types.hpp>
#include <dual_simplex/user_problem.hpp>

#include <algorithm>
#include <array>
#include <numeric>
#include <string>
#include <vector>

#include <cmath>
#include <cstdint>

namespace cuopt::linear_programming::dual_simplex {

enum cut_type_t : int8_t {
  MIXED_INTEGER_GOMORY   = 0,
  MIXED_INTEGER_ROUNDING = 1,
  KNAPSACK               = 2,
  CHVATAL_GOMORY         = 3,
  MAX_CUT_TYPE           = 4
};

template <typename i_t, typename f_t>
struct cut_info_t {
  bool has_cuts() const
  {
    i_t total_cuts = 0;
    for (i_t i = 0; i < MAX_CUT_TYPE; i++) {
      total_cuts += num_cuts[i];
    }
    return total_cuts > 0;
  }
  void record_cut_types(const std::vector<cut_type_t>& cut_types)
  {
    for (cut_type_t cut_type : cut_types) {
      num_cuts[static_cast<int>(cut_type)]++;
    }
  }
  const char* cut_type_names[MAX_CUT_TYPE] = {"Gomory   ", "MIR      ", "Knapsack ", "Strong CG"};
  std::array<i_t, MAX_CUT_TYPE> num_cuts   = {0};
};

template <typename i_t, typename f_t>
void print_cut_info(const simplex_solver_settings_t<i_t, f_t>& settings,
                    const cut_info_t<i_t, f_t>& cut_info)
{
  if (cut_info.has_cuts()) {
    for (i_t i = 0; i < MAX_CUT_TYPE; i++) {
      settings.log.printf("%s cuts : %d\n", cut_info.cut_type_names[i], cut_info.num_cuts[i]);
    }
  }
}

template <typename i_t, typename f_t>
void print_cut_types(const std::string& prefix,
                     const std::vector<cut_type_t>& cut_types,
                     const simplex_solver_settings_t<i_t, f_t>& settings)
{
  cut_info_t<i_t, f_t> cut_info;
  cut_info.record_cut_types(cut_types);
  settings.log.printf("%s: ", prefix.c_str());
  for (i_t i = 0; i < MAX_CUT_TYPE; i++) {
    settings.log.printf("%s cuts: %d ", cut_info.cut_type_names[i], cut_info.num_cuts[i]);
    if (i < MAX_CUT_TYPE - 1) { settings.log.printf(", "); }
  }
  settings.log.printf("\n");
}

template <typename f_t>
f_t fractional_part(f_t a)
{
  return a - std::floor(a);
}

// Computes a permutation of a score vector that puts the highest scores first
template <typename i_t, typename f_t>
void best_score_first_permutation(std::vector<f_t>& scores, std::vector<i_t>& permutation)
{
  if (permutation.size() != scores.size()) { permutation.resize(scores.size()); }
  std::iota(permutation.begin(), permutation.end(), 0);
  std::sort(
    permutation.begin(), permutation.end(), [&](i_t a, i_t b) { return scores[a] > scores[b]; });
}

// Computes a permutation of a score vector that puts the highest score last
template <typename i_t, typename f_t>
void best_score_last_permutation(std::vector<f_t>& scores, std::vector<i_t>& permutation)
{
  if (permutation.size() != scores.size()) { permutation.resize(scores.size()); }
  std::iota(permutation.begin(), permutation.end(), 0);
  std::sort(
    permutation.begin(), permutation.end(), [&](i_t a, i_t b) { return scores[a] < scores[b]; });
}

// Routines for verifying cuts against a saved solution
template <typename i_t, typename f_t>
void read_saved_solution_for_cut_verification(const lp_problem_t<i_t, f_t>& lp,
                                              const simplex_solver_settings_t<i_t, f_t>& settings,
                                              std::vector<f_t>& saved_solution);

template <typename i_t, typename f_t>
void write_solution_for_cut_verification(const lp_problem_t<i_t, f_t>& lp,
                                         const std::vector<f_t>& solution);

template <typename i_t, typename f_t>
void verify_cuts_against_saved_solution(const csr_matrix_t<i_t, f_t>& cuts,
                                        const std::vector<f_t>& cut_rhs,
                                        const std::vector<f_t>& saved_solution);

template <typename i_t, typename f_t>
class cut_pool_t {
 public:
  cut_pool_t(i_t original_vars, const simplex_solver_settings_t<i_t, f_t>& settings)
    : original_vars_(original_vars),
      settings_(settings),
      cut_storage_(0, original_vars, 0),
      rhs_storage_(0),
      cut_age_(0),
      cut_type_(0),
      scored_cuts_(0)
  {
  }

  // Add a cut in the form: cut'*x >= rhs.
  // We expect that the cut is violated by the current relaxation xstar
  // cut'*xstart < rhs
  void add_cut(cut_type_t cut_type, const sparse_vector_t<i_t, f_t>& cut, f_t rhs);

  void score_cuts(std::vector<f_t>& x_relax);

  // We return the cuts in the form best_cuts*x <= best_rhs
  i_t get_best_cuts(csr_matrix_t<i_t, f_t>& best_cuts,
                    std::vector<f_t>& best_rhs,
                    std::vector<cut_type_t>& best_cut_types);

  void age_cuts();

  void drop_cuts();

  i_t pool_size() const { return cut_storage_.m; }

  void print_cutpool_types() { print_cut_types("In cut pool", cut_type_, settings_); }

 private:
  f_t cut_distance(i_t row, const std::vector<f_t>& x, f_t& cut_violation, f_t& cut_norm);
  f_t cut_density(i_t row);
  f_t cut_orthogonality(i_t i, i_t j);

  i_t original_vars_;
  const simplex_solver_settings_t<i_t, f_t>& settings_;

  csr_matrix_t<i_t, f_t> cut_storage_;
  std::vector<f_t> rhs_storage_;
  std::vector<i_t> cut_age_;
  std::vector<cut_type_t> cut_type_;

  i_t scored_cuts_;
  std::vector<f_t> cut_distances_;
  std::vector<f_t> cut_norms_;
  std::vector<f_t> cut_orthogonality_;
  std::vector<f_t> cut_scores_;
  std::vector<i_t> best_cuts_;
};

template <typename i_t, typename f_t>
class knapsack_generation_t {
 public:
  knapsack_generation_t(const lp_problem_t<i_t, f_t>& lp,
                        const simplex_solver_settings_t<i_t, f_t>& settings,
                        csr_matrix_t<i_t, f_t>& Arow,
                        const std::vector<i_t>& new_slacks,
                        const std::vector<variable_type_t>& var_types);

  i_t generate_knapsack_cuts(const lp_problem_t<i_t, f_t>& lp,
                             const simplex_solver_settings_t<i_t, f_t>& settings,
                             csr_matrix_t<i_t, f_t>& Arow,
                             const std::vector<i_t>& new_slacks,
                             const std::vector<variable_type_t>& var_types,
                             const std::vector<f_t>& xstar,
                             i_t knapsack_row,
                             sparse_vector_t<i_t, f_t>& cut,
                             f_t& cut_rhs);

  i_t num_knapsack_constraints() const { return knapsack_constraints_.size(); }
  const std::vector<i_t>& get_knapsack_constraints() const { return knapsack_constraints_; }

 private:
  // Generate a heuristic solution to the 0-1 knapsack problem
  f_t greedy_knapsack_problem(const std::vector<f_t>& values,
                              const std::vector<f_t>& weights,
                              f_t rhs,
                              std::vector<f_t>& solution);

  // Solve a 0-1 knapsack problem using dynamic programming
  f_t solve_knapsack_problem(const std::vector<f_t>& values,
                             const std::vector<f_t>& weights,
                             f_t rhs,
                             std::vector<f_t>& solution);

  std::vector<i_t> is_slack_;
  std::vector<i_t> knapsack_constraints_;
  const simplex_solver_settings_t<i_t, f_t>& settings_;
};

// Forward declaration
template <typename i_t, typename f_t>
class mixed_integer_rounding_cut_t;

template <typename i_t, typename f_t>
class cut_generation_t {
 public:
  cut_generation_t(cut_pool_t<i_t, f_t>& cut_pool,
                   const lp_problem_t<i_t, f_t>& lp,
                   const simplex_solver_settings_t<i_t, f_t>& settings,
                   csr_matrix_t<i_t, f_t>& Arow,
                   const std::vector<i_t>& new_slacks,
                   const std::vector<variable_type_t>& var_types)
    : cut_pool_(cut_pool), knapsack_generation_(lp, settings, Arow, new_slacks, var_types)
  {
  }

  void generate_cuts(const lp_problem_t<i_t, f_t>& lp,
                     const simplex_solver_settings_t<i_t, f_t>& settings,
                     csr_matrix_t<i_t, f_t>& Arow,
                     const std::vector<i_t>& new_slacks,
                     const std::vector<variable_type_t>& var_types,
                     basis_update_mpf_t<i_t, f_t>& basis_update,
                     const std::vector<f_t>& xstar,
                     const std::vector<i_t>& basic_list,
                     const std::vector<i_t>& nonbasic_list);

 private:
  // Generate all mixed integer gomory cuts
  void generate_gomory_cuts(const lp_problem_t<i_t, f_t>& lp,
                            const simplex_solver_settings_t<i_t, f_t>& settings,
                            csr_matrix_t<i_t, f_t>& Arow,
                            const std::vector<i_t>& new_slacks,
                            const std::vector<variable_type_t>& var_types,
                            basis_update_mpf_t<i_t, f_t>& basis_update,
                            const std::vector<f_t>& xstar,
                            const std::vector<i_t>& basic_list,
                            const std::vector<i_t>& nonbasic_list);

  // Generate all mixed integer rounding cuts
  void generate_mir_cuts(const lp_problem_t<i_t, f_t>& lp,
                         const simplex_solver_settings_t<i_t, f_t>& settings,
                         csr_matrix_t<i_t, f_t>& Arow,
                         const std::vector<i_t>& new_slacks,
                         const std::vector<variable_type_t>& var_types,
                         const std::vector<f_t>& xstar);

  // Generate all knapsack cuts
  void generate_knapsack_cuts(const lp_problem_t<i_t, f_t>& lp,
                              const simplex_solver_settings_t<i_t, f_t>& settings,
                              csr_matrix_t<i_t, f_t>& Arow,
                              const std::vector<i_t>& new_slacks,
                              const std::vector<variable_type_t>& var_types,
                              const std::vector<f_t>& xstar);

  cut_pool_t<i_t, f_t>& cut_pool_;
  knapsack_generation_t<i_t, f_t> knapsack_generation_;
};

template <typename i_t, typename f_t>
class tableau_equality_t {
 public:
  tableau_equality_t(const lp_problem_t<i_t, f_t>& lp,
                     basis_update_mpf_t<i_t, f_t>& basis_update,
                     const std::vector<i_t>& nonbasic_list)
    : b_bar_(lp.num_rows, 0.0),
      nonbasic_mark_(lp.num_cols, 0),
      x_workspace_(lp.num_cols, 0.0),
      x_mark_(lp.num_cols, 0),
      c_workspace_(lp.num_cols, 0.0)
  {
    basis_update.b_solve(lp.rhs, b_bar_);
    for (i_t j : nonbasic_list) {
      nonbasic_mark_[j] = 1;
    }
  }

  // Generates the base inequalities: C*x == d that will be turned into cuts
  i_t generate_base_equality(const lp_problem_t<i_t, f_t>& lp,
                             const simplex_solver_settings_t<i_t, f_t>& settings,
                             csr_matrix_t<i_t, f_t>& Arow,
                             const std::vector<variable_type_t>& var_types,
                             basis_update_mpf_t<i_t, f_t>& basis_update,
                             const std::vector<f_t>& xstar,
                             const std::vector<i_t>& basic_list,
                             const std::vector<i_t>& nonbasic_list,
                             i_t i,
                             sparse_vector_t<i_t, f_t>& inequality,
                             f_t& inequality_rhs);

 private:
  std::vector<f_t> b_bar_;
  std::vector<i_t> nonbasic_mark_;
  std::vector<f_t> x_workspace_;
  std::vector<i_t> x_mark_;
  std::vector<f_t> c_workspace_;
};

template <typename i_t, typename f_t>
class mixed_integer_rounding_cut_t {
 public:
  mixed_integer_rounding_cut_t(const lp_problem_t<i_t, f_t>& lp,
                               const simplex_solver_settings_t<i_t, f_t>& settings,
                               const std::vector<i_t>& new_slacks,
                               const std::vector<f_t>& xstar);

  // Convert an inequality of the form: sum_j a_j x_j >= beta
  // with l_j <= x_j <= u_j into the form:
  // sum_{j not in L union U} d_j x_j + sum_{j in L} d_j v_j
  // + sum_{j in U} d_j w_j >= delta,
  // where v_j = x_j - l_j for j in L
  // and   w_j = u_j - x_j for j in Us
  void to_nonnegative(const lp_problem_t<i_t, f_t>& lp,
                      sparse_vector_t<i_t, f_t>& inequality,
                      f_t& rhs);

  void relaxation_to_nonnegative(const lp_problem_t<i_t, f_t>& lp,
                                 const std::vector<f_t>& xstar,
                                 std::vector<f_t>& xstar_nonnegative);

  // Convert an inequality of the form:
  // sum_{j not in L union U} d_j x_j + sum_{j in L} d_j v_j
  // + sum_{j in U} d_j w_j >= delta
  // where v_j = x_j - l_j for j in L
  // and   w_j = u_j - x_j for j in U
  // back to an inequality on the original variables
  // sum_j a_j x_j >= beta
  void to_original(const lp_problem_t<i_t, f_t>& lp,
                   sparse_vector_t<i_t, f_t>& inequality,
                   f_t& rhs);

  // Given a cut of the form sum_j d_j x_j >= beta
  // with l_j <= x_j <= u_j, try to remove coefficients d_j
  // with | d_j | < epsilon
  void remove_small_coefficients(const std::vector<f_t>& lower_bounds,
                                 const std::vector<f_t>& upper_bounds,
                                 sparse_vector_t<i_t, f_t>& cut,
                                 f_t& cut_rhs);

  // Given an inequality sum_j a_j x_j >= beta, x_j >= 0, x_j in Z, j in I
  // generate an MIR cut of the form sum_j d_j x_j >= delta
  i_t generate_cut_nonnegative(const sparse_vector_t<i_t, f_t>& a,
                               f_t beta,
                               const std::vector<variable_type_t>& var_types,
                               sparse_vector_t<i_t, f_t>& cut,
                               f_t& cut_rhs);

  f_t compute_violation(const sparse_vector_t<i_t, f_t>& cut,
                        f_t cut_rhs,
                        const std::vector<f_t>& xstar);

  i_t generate_cut(const sparse_vector_t<i_t, f_t>& a,
                   f_t beta,
                   const std::vector<f_t>& upper_bounds,
                   const std::vector<f_t>& lower_bounds,
                   const std::vector<variable_type_t>& var_types,
                   sparse_vector_t<i_t, f_t>& cut,
                   f_t& cut_rhs);

  void substitute_slacks(const lp_problem_t<i_t, f_t>& lp,
                         csr_matrix_t<i_t, f_t>& Arow,
                         sparse_vector_t<i_t, f_t>& cut,
                         f_t& cut_rhs);

  // Combine the pivot row with the inequality to eliminate the variable j
  // The new inequality is returned in inequality and inequality_rhs
  void combine_rows(const lp_problem_t<i_t, f_t>& lp,
                    csr_matrix_t<i_t, f_t>& Arow,
                    i_t j,
                    const sparse_vector_t<i_t, f_t>& pivot_row,
                    f_t pivot_row_rhs,
                    sparse_vector_t<i_t, f_t>& inequality,
                    f_t& inequality_rhs);

 private:
  i_t num_vars_;
  const simplex_solver_settings_t<i_t, f_t>& settings_;
  std::vector<f_t> x_workspace_;
  std::vector<i_t> x_mark_;
  std::vector<i_t> has_lower_;
  std::vector<i_t> has_upper_;
  std::vector<i_t> is_slack_;
  std::vector<i_t> slack_rows_;
  std::vector<i_t> indices_;
  std::vector<i_t> bound_info_;
  bool needs_complement_;
};

template <typename i_t, typename f_t>
class strong_cg_cut_t {
 public:
  strong_cg_cut_t(const lp_problem_t<i_t, f_t>& lp,
                  const std::vector<variable_type_t>& var_types,
                  const std::vector<f_t>& xstar);

  i_t generate_strong_cg_cut(const lp_problem_t<i_t, f_t>& lp,
                             const simplex_solver_settings_t<i_t, f_t>& settings,
                             const std::vector<variable_type_t>& var_types,
                             const sparse_vector_t<i_t, f_t>& inequality,
                             const f_t inequality_rhs,
                             const std::vector<f_t>& xstar,
                             sparse_vector_t<i_t, f_t>& cut,
                             f_t& cut_rhs);

  i_t remove_continuous_variables_integers_nonnegative(
    const lp_problem_t<i_t, f_t>& lp,
    const simplex_solver_settings_t<i_t, f_t>& settings,
    const std::vector<variable_type_t>& var_types,
    sparse_vector_t<i_t, f_t>& inequality,
    f_t& inequality_rhs);

  void to_original_integer_variables(const lp_problem_t<i_t, f_t>& lp,
                                     sparse_vector_t<i_t, f_t>& cut,
                                     f_t& cut_rhs);

  i_t generate_strong_cg_cut_integer_only(const simplex_solver_settings_t<i_t, f_t>& settings,
                                          const std::vector<variable_type_t>& var_types,
                                          const sparse_vector_t<i_t, f_t>& inequality,
                                          f_t inequality_rhs,
                                          sparse_vector_t<i_t, f_t>& cut,
                                          f_t& cut_rhs);

 private:
  i_t generate_strong_cg_cut_helper(const std::vector<i_t>& indicies,
                                    const std::vector<f_t>& coefficients,
                                    f_t rhs,
                                    const std::vector<variable_type_t>& var_types,
                                    sparse_vector_t<i_t, f_t>& cut,
                                    f_t& cut_rhs);

  std::vector<i_t> transformed_variables_;
};

template <typename i_t, typename f_t>
i_t add_cuts(const simplex_solver_settings_t<i_t, f_t>& settings,
             const csr_matrix_t<i_t, f_t>& cuts,
             const std::vector<f_t>& cut_rhs,
             lp_problem_t<i_t, f_t>& lp,
             std::vector<i_t>& new_slacks,
             lp_solution_t<i_t, f_t>& solution,
             basis_update_mpf_t<i_t, f_t>& basis_update,
             std::vector<i_t>& basic_list,
             std::vector<i_t>& nonbasic_list,
             std::vector<variable_status_t>& vstatus,
             std::vector<f_t>& edge_norms);

template <typename i_t, typename f_t>
void remove_cuts(lp_problem_t<i_t, f_t>& lp,
                 const simplex_solver_settings_t<i_t, f_t>& settings,
                 csr_matrix_t<i_t, f_t>& Arow,
                 std::vector<i_t>& new_slacks,
                 i_t original_rows,
                 std::vector<variable_type_t>& var_types,
                 std::vector<variable_status_t>& vstatus,
                 std::vector<f_t>& edge_norms,
                 std::vector<f_t>& x,
                 std::vector<f_t>& y,
                 std::vector<f_t>& z,
                 std::vector<i_t>& basic_list,
                 std::vector<i_t>& nonbasic_list,
                 basis_update_mpf_t<i_t, f_t>& basis_update);

}  // namespace cuopt::linear_programming::dual_simplex
