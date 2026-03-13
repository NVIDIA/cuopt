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
#include <atomic>
#include <future>
#include <memory>
#include <numeric>
#include <string>
#include <vector>

#include <cmath>
#include <cstdint>

namespace cuopt::linear_programming::detail {
template <typename i_t, typename f_t>
struct clique_table_t;
}

namespace cuopt::linear_programming::dual_simplex {

enum cut_type_t : int8_t {
  MIXED_INTEGER_GOMORY   = 0,
  MIXED_INTEGER_ROUNDING = 1,
  KNAPSACK               = 2,
  CHVATAL_GOMORY         = 3,
  CLIQUE                 = 4,
  MAX_CUT_TYPE           = 5
};

template <typename f_t>
struct cut_gap_closure_t {
  f_t initial_gap{0.0};
  f_t final_gap{0.0};
  f_t gap_closed{0.0};
  f_t gap_closed_ratio{0.0};
};

template <typename f_t>
cut_gap_closure_t<f_t> compute_cut_gap_closure(f_t objective_reference,
                                               f_t objective_before_cuts,
                                               f_t objective_after_cuts)
{
  const f_t initial_gap      = std::abs(objective_reference - objective_before_cuts);
  const f_t final_gap        = std::abs(objective_reference - objective_after_cuts);
  const f_t gap_closed       = initial_gap - final_gap;
  constexpr f_t eps          = static_cast<f_t>(1e-12);
  const f_t gap_closed_ratio = initial_gap > eps ? gap_closed / initial_gap : static_cast<f_t>(0.0);
  return {initial_gap, final_gap, gap_closed, gap_closed_ratio};
}

template <typename i_t, typename f_t>
struct inequality_t {
  inequality_t() : vector(), rhs(0.0) {}
  inequality_t(i_t num_cols) : vector(num_cols, 0), rhs(0.0) {}
  inequality_t(csr_matrix_t<i_t, f_t>& A, i_t row, f_t rhs_value) : vector(A, row), rhs(rhs_value)
  {
  }
  sparse_vector_t<i_t, f_t> vector;
  f_t rhs;

  void push_back(i_t j, f_t x)
  {
    vector.i.push_back(j);
    vector.x.push_back(x);
  }
  void clear()
  {
    vector.i.clear();
    vector.x.clear();
  }
  void reserve(size_t n)
  {
    vector.i.reserve(n);
    vector.x.reserve(n);
  }
  size_t size() const { return vector.i.size(); }
  i_t index(i_t k) const { return vector.i[k]; }
  f_t coeff(i_t k) const { return vector.x[k]; }
  void negate()
  {
    vector.negate();
    rhs *= -1.0;
  }
  void sort() { vector.sort(); }
  void squeeze(inequality_t<i_t, f_t>& out) const
  {
    vector.squeeze(out.vector);
    out.rhs = rhs;
  }
  void scale(f_t factor)
  {
    vector.scale(factor);
    rhs *= factor;
  }
  void print() const
  {
    for (i_t k = 0; k < size(); k++) {
      printf("%g x%d ", coeff(k), index(k));
    }
    printf("\nrhs %g\n", rhs);
  }
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
  const char* cut_type_names[MAX_CUT_TYPE] = {
    "Gomory   ", "MIR      ", "Knapsack ", "Strong CG", "Clique   "};
  std::array<i_t, MAX_CUT_TYPE> num_cuts = {0};
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
    settings.log.printf("%s cuts: %d\n", cut_info.cut_type_names[i], cut_info.num_cuts[i]);
  }
}

template <typename f_t>
f_t fractional_part(f_t a)
{
  return a - std::floor(a);
}

template <typename f_t>
bool add_work_estimate(f_t accesses,
                       f_t* work_estimate,
                       f_t max_work_estimate,
                       bool* work_limit_reached = nullptr)
{
  if (work_estimate == nullptr) { return false; }
  *work_estimate += accesses;
  const bool over_work_limit = *work_estimate > max_work_estimate;
  if (over_work_limit && work_limit_reached != nullptr) { *work_limit_reached = true; }
  return over_work_limit;
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

// Test-only helper to run the production maximal-clique algorithm used by clique cuts.
// adjacency_list must contain local vertex indices in [0, n_vertices).
std::vector<std::vector<int>> find_maximal_cliques_for_test(
  const std::vector<std::vector<int>>& adjacency_list,
  const std::vector<double>& weights,
  double min_weight,
  int max_calls,
  double time_limit);

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
  void add_cut(cut_type_t cut_type, const inequality_t<i_t, f_t>& cut);

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
  const f_t min_cut_distance_{1e-4};
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
                             inequality_t<i_t, f_t>& cut);

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

// Forward declarations
template <typename i_t, typename f_t>
class mixed_integer_rounding_cut_t;

template <typename i_t, typename f_t>
class variable_bounds_t;

template <typename i_t, typename f_t>
class cut_generation_t {
 public:
  cut_generation_t(
    cut_pool_t<i_t, f_t>& cut_pool,
    const lp_problem_t<i_t, f_t>& lp,
    const simplex_solver_settings_t<i_t, f_t>& settings,
    csr_matrix_t<i_t, f_t>& Arow,
    const std::vector<i_t>& new_slacks,
    const std::vector<variable_type_t>& var_types,
    const user_problem_t<i_t, f_t>& user_problem,
    std::shared_ptr<detail::clique_table_t<i_t, f_t>> clique_table                      = nullptr,
    std::future<std::shared_ptr<detail::clique_table_t<i_t, f_t>>>* clique_table_future = nullptr,
    std::atomic<bool>* signal_extend                                                    = nullptr)
    : cut_pool_(cut_pool),
      knapsack_generation_(lp, settings, Arow, new_slacks, var_types),
      user_problem_(user_problem),
      clique_table_(std::move(clique_table)),
      clique_table_future_(clique_table_future),
      signal_extend_(signal_extend)
  {
  }

  bool generate_cuts(const lp_problem_t<i_t, f_t>& lp,
                     const simplex_solver_settings_t<i_t, f_t>& settings,
                     csr_matrix_t<i_t, f_t>& Arow,
                     const std::vector<i_t>& new_slacks,
                     const std::vector<variable_type_t>& var_types,
                     basis_update_mpf_t<i_t, f_t>& basis_update,
                     const std::vector<f_t>& xstar,
                     const std::vector<f_t>& ystar,
                     const std::vector<f_t>& zstar,
                     const std::vector<i_t>& basic_list,
                     const std::vector<i_t>& nonbasic_list,
                     variable_bounds_t<i_t, f_t>& variable_bounds,
                     f_t start_time);

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
                         const std::vector<f_t>& xstar,
                         const std::vector<f_t>& ystar,
                         variable_bounds_t<i_t, f_t>& variable_bounds);

  // Generate all knapsack cuts
  void generate_knapsack_cuts(const lp_problem_t<i_t, f_t>& lp,
                              const simplex_solver_settings_t<i_t, f_t>& settings,
                              csr_matrix_t<i_t, f_t>& Arow,
                              const std::vector<i_t>& new_slacks,
                              const std::vector<variable_type_t>& var_types,
                              const std::vector<f_t>& xstar);

  // Generate clique cuts from conflict graph cliques
  bool generate_clique_cuts(const lp_problem_t<i_t, f_t>& lp,
                            const simplex_solver_settings_t<i_t, f_t>& settings,
                            const std::vector<variable_type_t>& var_types,
                            const std::vector<f_t>& xstar,
                            const std::vector<f_t>& reduced_costs,
                            f_t start_time);

  cut_pool_t<i_t, f_t>& cut_pool_;
  knapsack_generation_t<i_t, f_t> knapsack_generation_;
  const user_problem_t<i_t, f_t>& user_problem_;
  std::shared_ptr<detail::clique_table_t<i_t, f_t>> clique_table_;
  std::future<std::shared_ptr<detail::clique_table_t<i_t, f_t>>>* clique_table_future_{nullptr};
  std::atomic<bool>* signal_extend_{nullptr};
};

template <typename i_t, typename f_t>
class scratch_pad_t {
 public:
  scratch_pad_t(i_t num_vars) : workspace_(num_vars, 0.0), mark_(num_vars, 0)
  {
    indices_.reserve(num_vars);
  }

  // O(1) to add a value to the pad
  void add_to_pad(i_t j, f_t value)
  {
    workspace_[j] += value;
    if (!mark_[j]) {
      mark_[j] = 1;
      indices_.push_back(j);
    }
  }

  // O(nz) to clear the pad
  void clear_pad()
  {
    for (i_t j : indices_) {
      workspace_[j] = 0.0;
      mark_[j]      = 0;
    }
    indices_.clear();
  }

  // O(nz) to get the pad
  void get_pad(std::vector<i_t>& indices, std::vector<f_t>& values)
  {
    indices.reserve(indices_.size());
    values.reserve(indices_.size());
    indices.clear();
    values.clear();
    const i_t nz = indices_.size();
    for (i_t k = 0; k < nz; k++) {
      const i_t j   = indices_[k];
      const f_t val = workspace_[j];
      if (val != 0.0) {
        indices.push_back(j);
        values.push_back(val);
      }
    }
  }

 private:
  std::vector<f_t> workspace_;
  std::vector<i_t> mark_;
  std::vector<i_t> indices_;
};

template <typename i_t, typename f_t>
class mixed_integer_gomory_cut_t {
 public:
  mixed_integer_gomory_cut_t() {}

  bool rational_coefficients(const std::vector<variable_type_t>& var_types,
                             const inequality_t<i_t, f_t>& input_inequality,
                             inequality_t<i_t, f_t>& rational_inequality);

 private:
  bool rational_approximation(f_t x,
                              int64_t max_denominator,
                              int64_t& numerator,
                              int64_t& denominator);

  int64_t gcd(const std::vector<int64_t>& integers);
  int64_t lcm(const std::vector<int64_t>& integers);
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
                             inequality_t<i_t, f_t>& inequality);

 private:
  std::vector<f_t> b_bar_;
  std::vector<i_t> nonbasic_mark_;
  std::vector<f_t> x_workspace_;
  std::vector<i_t> x_mark_;
  std::vector<f_t> c_workspace_;
};

template <typename i_t, typename f_t>
class variable_bounds_t {
 public:
  variable_bounds_t(const lp_problem_t<i_t, f_t>& lp,
                    const simplex_solver_settings_t<i_t, f_t>& settings,
                    const std::vector<variable_type_t>& var_types,
                    const csr_matrix_t<i_t, f_t>& Arow,
                    const std::vector<i_t>& new_slacks);

  std::vector<i_t> upper_offsets;
  std::vector<i_t> upper_variables;
  std::vector<f_t> upper_weights;
  std::vector<f_t> upper_biases;

  std::vector<i_t> lower_offsets;
  std::vector<i_t> lower_variables;
  std::vector<f_t> lower_weights;
  std::vector<f_t> lower_biases;

  void resize(i_t new_num_cols)
  {
    const i_t current_upper_nz = upper_offsets.back();
    upper_offsets.resize(new_num_cols + 1, current_upper_nz);
    const i_t current_lower_nz = lower_offsets.back();
    lower_offsets.resize(new_num_cols + 1, current_lower_nz);
  }

 private:
  f_t lower_activity(f_t lower_bound, f_t upper_bound, f_t coefficient)
  {
    return (coefficient > 0.0 ? lower_bound : upper_bound) * coefficient;
  }

  f_t upper_activity(f_t lower_bound, f_t upper_bound, f_t coefficient)
  {
    return (coefficient > 0.0 ? upper_bound : lower_bound) * coefficient;
  }

  // Returns the lower activity adjusted for the number of lower inf variables
  // adjusted_lower_activity = { activity - lower_activity_i - lower_activity_j, if num_lower_inf =
  // 0
  //                           { activity - lower_activity_i                   , if num_lower_inf =
  //                           1, lower_activity_j = -inf { activity - lower_activity_j , if
  //                           num_lower_inf = 1, lower_activity_i != -inf { activity , if
  //                           num_lower_inf = 2, lower_activity_i = lower_activity_j = -inf { -inf
  //                           , if num_lower_inf > 2
  f_t adjusted_lower_activity(f_t activity,
                              i_t num_lower_inf,
                              f_t lower_activity_i,
                              f_t lower_activity_j)
  {
    if (num_lower_inf == 0) {
      return activity - lower_activity_i - lower_activity_j;
    } else if (num_lower_inf == 1 && lower_activity_j == -inf) {
      return activity - lower_activity_i;
    } else if (num_lower_inf == 1 && lower_activity_i == -inf) {
      return activity - lower_activity_j;
    } else if (num_lower_inf == 2 && lower_activity_i == -inf && lower_activity_j == -inf) {
      return activity;
    } else {
      return -inf;
    }
  }

  // Returns the upper activity adjusted for the number of upper inf variables
  // adjusted_upper_activity = { activity - upper_activity_i - upper_activity_j, if num_upper_inf =
  // 0
  //                           { activity - upper_activity_i                   , if num_upper_inf =
  //                           1, upper_activity_j = inf { activity - upper_activity_j , if
  //                           num_upper_inf = 1, upper_activity_i != inf { activity , if
  //                           num_upper_inf = 2, upper_activity_i = upper_activity_j = inf { inf ,
  //                           if num_upper_inf > 2
  f_t adjusted_upper_activity(f_t activity,
                              i_t num_upper_inf,
                              f_t upper_activity_i,
                              f_t upper_activity_j)
  {
    if (num_upper_inf == 0) {
      return activity - upper_activity_i - upper_activity_j;
    } else if (num_upper_inf == 1 && upper_activity_j == inf) {
      return activity - upper_activity_i;
    } else if (num_upper_inf == 1 && upper_activity_i == inf) {
      return activity - upper_activity_j;
    } else if (num_upper_inf == 2 && upper_activity_i == inf && upper_activity_j == inf) {
      return activity;
    } else {
      return inf;
    }
  }

  std::vector<f_t> upper_activities_;
  std::vector<i_t> num_pos_inf_;
  std::vector<f_t> lower_activities_;
  std::vector<i_t> num_neg_inf_;

  std::vector<i_t> slack_map_;
};

template <typename i_t, typename f_t>
class complemented_mixed_integer_rounding_cut_t {
 public:
  complemented_mixed_integer_rounding_cut_t(const lp_problem_t<i_t, f_t>& lp,
                                            const simplex_solver_settings_t<i_t, f_t>& settings,
                                            const std::vector<i_t>& new_slacks);

  void compute_initial_scores_for_rows(const lp_problem_t<i_t, f_t>& lp,
                                       const simplex_solver_settings_t<i_t, f_t>& settings,
                                       const csr_matrix_t<i_t, f_t>& Arow,
                                       const std::vector<f_t>& xstar,
                                       const std::vector<f_t>& ystar,
                                       std::vector<f_t>& score);

  // Perform bound substitution for the continuous variables using simple bounds
  // and variable bounds. And bound substitution for the integer variables
  // using simple bounds.
  void bound_substitution(const lp_problem_t<i_t, f_t>& lp,
                          const variable_bounds_t<i_t, f_t>& variable_bounds,
                          const std::vector<variable_type_t>& var_types,
                          const std::vector<f_t>& xstar,
                          std::vector<f_t>& transformed_xstar);

  // Converts an inequality of the form: sum_j a_j x_j >= beta
  // with l_j <= x_j <= u_j into the form:
  // sum_{j not in L union U} d_j x_j + sum_{j in L} d_j v_j
  // + sum_{j in U} d_j w_j >= delta,
  // where v_j = x_j - l_j for j in L
  // and   w_j = u_j - x_j for j in U
  void transform_inequality(const variable_bounds_t<i_t, f_t>& variable_bounds,
                            const std::vector<variable_type_t>& var_type,
                            inequality_t<i_t, f_t>& inequality);

  // Converts an inequality of the form:
  // sum_{j not in L union U} d_j x_j + sum_{j in L} d_j v_j
  // + sum_{j in U} d_j w_j >= delta,
  // where v_j = x_j - l_j for j in L
  // and   w_j = u_j - x_j for j in U
  // back to the form: sum_j a_j x_j >= beta
  // with l_j <= x_j <= u_j
  void untransform_inequality(const variable_bounds_t<i_t, f_t>& variable_bounds,
                              const std::vector<variable_type_t>& var_type,
                              inequality_t<i_t, f_t>& inequality);

  bool cut_generation_heuristic(const inequality_t<i_t, f_t>& transformed_inequality,
                                const std::vector<variable_type_t>& var_types,
                                const std::vector<f_t>& transformed_xstar,
                                inequality_t<i_t, f_t>& transformed_cut,
                                f_t& work_estimate);

  bool scale_uncomplement_and_generate_cut(const std::vector<variable_type_t>& var_types,
                                           const std::vector<f_t>& transformed_xstar,
                                           const std::vector<i_t>& complemented_indices,
                                           const inequality_t<i_t, f_t>& complemented_inequality,
                                           f_t delta,
                                           inequality_t<i_t, f_t>& cut_delta,
                                           f_t& work_estimate);

  // This routine takes an inequality and generates the MIR cut
  bool generate_cut_nonnegative_maintain_indicies(const inequality_t<i_t, f_t>& inequality,
                                                  const std::vector<variable_type_t>& var_types,
                                                  inequality_t<i_t, f_t>& cut);

  f_t compute_violation(const inequality_t<i_t, f_t>& cut, const std::vector<f_t>& xstar);

  f_t new_upper(i_t j) const { return transformed_upper_[j]; }

  // Given a cut of the form sum_j d_j x_j >= beta
  // with l_j <= x_j <= u_j, try to remove coefficients d_j
  // with | d_j | < epsilon
  void remove_small_coefficients(const std::vector<f_t>& lower_bounds,
                                 const std::vector<f_t>& upper_bounds,
                                 inequality_t<i_t, f_t>& cut);

  void substitute_slacks(const lp_problem_t<i_t, f_t>& lp,
                         csr_matrix_t<i_t, f_t>& Arow,
                         inequality_t<i_t, f_t>& cut);

  // Combine the pivot row with the inequality to eliminate the variable j
  // The new inequality is returned in inequality and inequality_rhs
  // The multiplier for the pivot row is returned
  f_t combine_rows(const lp_problem_t<i_t, f_t>& lp,
                   csr_matrix_t<i_t, f_t>& Arow,
                   i_t j,
                   const inequality_t<i_t, f_t>& pivot_row,
                   inequality_t<i_t, f_t>& inequality);

  const f_t get_lb_star(i_t j) const { return lb_star_[j]; }
  const f_t get_ub_star(i_t j) const { return ub_star_[j]; }

  const i_t slack_rows(i_t j) const { return slack_rows_[j]; }
  const i_t slack_cols(i_t i) const { return slack_cols_[i]; }

  bool scale_and_generate_mir_cut(const std::vector<variable_type_t>& var_types,
                                  const std::vector<f_t>& transformed_xstar,
                                  const inequality_t<i_t, f_t>& inequality,
                                  f_t divisor,
                                  std::vector<inequality_t<i_t, f_t>>& cuts,
                                  std::vector<f_t>& violations,
                                  std::vector<f_t>& deltas);

  bool check_violation_and_add_cut(const inequality_t<i_t, f_t>& inequality,
                                   const std::vector<f_t>& xstar,
                                   f_t divisor,
                                   std::vector<inequality_t<i_t, f_t>>& cuts,
                                   std::vector<f_t>& violations,
                                   std::vector<f_t>& deltas);

 private:
  std::vector<i_t> is_slack_;
  std::vector<i_t>
    slack_rows_;  // slack_rows_[j] = i, if variable j is slack for row i, -1 is sentinal value
  std::vector<i_t>
    slack_cols_;  // slack_cols_[i] = j, if variable j is slack for row i  -1 is sentinal value

  std::vector<i_t> lb_variable_;
  std::vector<f_t> lb_star_;
  std::vector<i_t> ub_variable_;
  std::vector<f_t> ub_star_;

  std::vector<i_t> bound_changed_;
  std::vector<f_t> transformed_upper_;

  scratch_pad_t<i_t, f_t> scratch_pad_;
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
                             const inequality_t<i_t, f_t>& inequality,
                             const std::vector<f_t>& xstar,
                             inequality_t<i_t, f_t>& cut);

  i_t remove_continuous_variables_integers_nonnegative(
    const lp_problem_t<i_t, f_t>& lp,
    const simplex_solver_settings_t<i_t, f_t>& settings,
    const std::vector<variable_type_t>& var_types,
    inequality_t<i_t, f_t>& inequality);

  void to_original_integer_variables(const lp_problem_t<i_t, f_t>& lp, inequality_t<i_t, f_t>& cut);

  i_t generate_strong_cg_cut_integer_only(const simplex_solver_settings_t<i_t, f_t>& settings,
                                          const std::vector<variable_type_t>& var_types,
                                          const inequality_t<i_t, f_t>& inequality,
                                          inequality_t<i_t, f_t>& cut);

 private:
  i_t generate_strong_cg_cut_helper(const std::vector<i_t>& indicies,
                                    const std::vector<f_t>& coefficients,
                                    f_t rhs,
                                    const std::vector<variable_type_t>& var_types,
                                    inequality_t<i_t, f_t>& cut);

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
i_t remove_cuts(lp_problem_t<i_t, f_t>& lp,
                const simplex_solver_settings_t<i_t, f_t>& settings,
                f_t start_time,
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
