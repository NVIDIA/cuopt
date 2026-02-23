/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#include "../linear_programming/utilities/pdlp_test_utilities.cuh"
#include "mip_utils.cuh"

#include <cuopt/linear_programming/pdlp/solver_settings.hpp>
#include <cuopt/linear_programming/pdlp/solver_solution.hpp>
#include <cuopt/linear_programming/solve.hpp>
#include <cuts/cuts.hpp>
#include <mip_heuristics/presolve/conflict_graph/clique_table.cuh>
#include <mip_heuristics/problem/problem.cuh>
#include <mps_parser/parser.hpp>
#include <utilities/common_utils.hpp>
#include <utilities/copy_helpers.hpp>
#include <utilities/error.hpp>
#include <utilities/timer.hpp>

#include <raft/core/handle.hpp>
#include <raft/util/cudart_utils.hpp>

#include <gtest/gtest.h>

#include <algorithm>
#include <cstdint>
#include <filesystem>
#include <limits>
#include <optional>
#include <sstream>
#include <string>
#include <unordered_set>
#include <vector>

namespace cuopt::linear_programming::test {

namespace {

constexpr double kCliqueTestTol = 1e-6;

mps_parser::mps_data_model_t<int, double> create_pairwise_triangle_set_packing_problem()
{
  // Maximize x0 + x1 + x2 via minimizing -x0 - x1 - x2.
  // Pairwise conflicts:
  //   x0 + x1 <= 1
  //   x1 + x2 <= 1
  //   x0 + x2 <= 1
  mps_parser::mps_data_model_t<int, double> problem;
  std::vector<int> offsets         = {0, 2, 4, 6};
  std::vector<int> indices         = {0, 1, 1, 2, 0, 2};
  std::vector<double> coefficients = {1.0, 1.0, 1.0, 1.0, 1.0, 1.0};
  problem.set_csr_constraint_matrix(coefficients.data(),
                                    coefficients.size(),
                                    indices.data(),
                                    indices.size(),
                                    offsets.data(),
                                    offsets.size());
  std::vector<double> lower_bounds = {-std::numeric_limits<double>::infinity(),
                                      -std::numeric_limits<double>::infinity(),
                                      -std::numeric_limits<double>::infinity()};
  std::vector<double> upper_bounds = {1.0, 1.0, 1.0};
  problem.set_constraint_lower_bounds(lower_bounds.data(), lower_bounds.size());
  problem.set_constraint_upper_bounds(upper_bounds.data(), upper_bounds.size());
  std::vector<double> var_lower_bounds = {0.0, 0.0, 0.0};
  std::vector<double> var_upper_bounds = {1.0, 1.0, 1.0};
  problem.set_variable_lower_bounds(var_lower_bounds.data(), var_lower_bounds.size());
  problem.set_variable_upper_bounds(var_upper_bounds.data(), var_upper_bounds.size());
  std::vector<double> objective_coefficients = {-1.0, -1.0, -1.0};
  problem.set_objective_coefficients(objective_coefficients.data(), objective_coefficients.size());
  std::vector<char> variable_types = {'I', 'I', 'I'};
  problem.set_variable_types(variable_types);
  problem.set_maximize(false);
  return problem;
}

mps_parser::mps_data_model_t<int, double> create_pairwise_triangle_with_isolated_variable_problem()
{
  // Same triangle conflicts as create_pairwise_triangle_set_packing_problem(),
  // plus an isolated binary variable x3 with no conflict rows.
  mps_parser::mps_data_model_t<int, double> problem;
  std::vector<int> offsets         = {0, 2, 4, 6};
  std::vector<int> indices         = {0, 1, 1, 2, 0, 2};
  std::vector<double> coefficients = {1.0, 1.0, 1.0, 1.0, 1.0, 1.0};
  problem.set_csr_constraint_matrix(coefficients.data(),
                                    coefficients.size(),
                                    indices.data(),
                                    indices.size(),
                                    offsets.data(),
                                    offsets.size());
  std::vector<double> lower_bounds = {-std::numeric_limits<double>::infinity(),
                                      -std::numeric_limits<double>::infinity(),
                                      -std::numeric_limits<double>::infinity()};
  std::vector<double> upper_bounds = {1.0, 1.0, 1.0};
  problem.set_constraint_lower_bounds(lower_bounds.data(), lower_bounds.size());
  problem.set_constraint_upper_bounds(upper_bounds.data(), upper_bounds.size());
  std::vector<double> var_lower_bounds = {0.0, 0.0, 0.0, 0.0};
  std::vector<double> var_upper_bounds = {1.0, 1.0, 1.0, 1.0};
  problem.set_variable_lower_bounds(var_lower_bounds.data(), var_lower_bounds.size());
  problem.set_variable_upper_bounds(var_upper_bounds.data(), var_upper_bounds.size());
  std::vector<double> objective_coefficients = {-1.0, -1.0, -1.0, 0.0};
  problem.set_objective_coefficients(objective_coefficients.data(), objective_coefficients.size());
  std::vector<char> variable_types = {'I', 'I', 'I', 'I'};
  problem.set_variable_types(variable_types);
  problem.set_maximize(false);
  return problem;
}

mps_parser::mps_data_model_t<int, double> create_binary_continuous_mixed_conflict_problem()
{
  // x0 + y1 <= 1  (must be ignored for clique graph because y1 is continuous)
  // x0 + x2 <= 1  (must generate a conflict edge)
  mps_parser::mps_data_model_t<int, double> problem;
  std::vector<int> offsets         = {0, 2, 4};
  std::vector<int> indices         = {0, 1, 0, 2};
  std::vector<double> coefficients = {1.0, 1.0, 1.0, 1.0};
  problem.set_csr_constraint_matrix(coefficients.data(),
                                    coefficients.size(),
                                    indices.data(),
                                    indices.size(),
                                    offsets.data(),
                                    offsets.size());
  std::vector<double> lower_bounds = {-std::numeric_limits<double>::infinity(),
                                      -std::numeric_limits<double>::infinity()};
  std::vector<double> upper_bounds = {1.0, 1.0};
  problem.set_constraint_lower_bounds(lower_bounds.data(), lower_bounds.size());
  problem.set_constraint_upper_bounds(upper_bounds.data(), upper_bounds.size());
  std::vector<double> var_lower_bounds = {0.0, 0.0, 0.0};
  std::vector<double> var_upper_bounds = {1.0, 1.0, 1.0};
  problem.set_variable_lower_bounds(var_lower_bounds.data(), var_lower_bounds.size());
  problem.set_variable_upper_bounds(var_upper_bounds.data(), var_upper_bounds.size());
  std::vector<double> objective_coefficients = {0.0, 0.0, 0.0};
  problem.set_objective_coefficients(objective_coefficients.data(), objective_coefficients.size());
  std::vector<char> variable_types = {'I', 'C', 'I'};
  problem.set_variable_types(variable_types);
  problem.set_maximize(false);
  return problem;
}

mps_parser::mps_data_model_t<int, double> create_near_binary_bound_conflict_problem()
{
  // x0 + x1 <= 1 but x1 has upper bound 0.9999999, so this row should not be
  // treated as a binary conflict row.
  mps_parser::mps_data_model_t<int, double> problem;
  std::vector<int> offsets         = {0, 2};
  std::vector<int> indices         = {0, 1};
  std::vector<double> coefficients = {1.0, 1.0};
  problem.set_csr_constraint_matrix(coefficients.data(),
                                    coefficients.size(),
                                    indices.data(),
                                    indices.size(),
                                    offsets.data(),
                                    offsets.size());
  std::vector<double> lower_bounds = {-std::numeric_limits<double>::infinity()};
  std::vector<double> upper_bounds = {1.0};
  problem.set_constraint_lower_bounds(lower_bounds.data(), lower_bounds.size());
  problem.set_constraint_upper_bounds(upper_bounds.data(), upper_bounds.size());
  std::vector<double> var_lower_bounds = {0.0, 0.0};
  std::vector<double> var_upper_bounds = {1.0, 0.9999999};
  problem.set_variable_lower_bounds(var_lower_bounds.data(), var_lower_bounds.size());
  problem.set_variable_upper_bounds(var_upper_bounds.data(), var_upper_bounds.size());
  std::vector<double> objective_coefficients = {0.0, 0.0};
  problem.set_objective_coefficients(objective_coefficients.data(), objective_coefficients.size());
  std::vector<char> variable_types = {'I', 'I'};
  problem.set_variable_types(variable_types);
  problem.set_maximize(false);
  return problem;
}

mps_parser::mps_data_model_t<int, double> create_weighted_addtl_conflict_problem()
{
  // One weighted binary knapsack row:
  //   1*x0 + 2*x1 + 3*x2 + 4*x3 <= 5
  // This creates base clique {x2, x3} and additional clique inducing conflict {x1, x3}.
  mps_parser::mps_data_model_t<int, double> problem;
  std::vector<int> offsets         = {0, 4};
  std::vector<int> indices         = {0, 1, 2, 3};
  std::vector<double> coefficients = {1.0, 2.0, 3.0, 4.0};
  problem.set_csr_constraint_matrix(coefficients.data(),
                                    coefficients.size(),
                                    indices.data(),
                                    indices.size(),
                                    offsets.data(),
                                    offsets.size());
  std::vector<double> lower_bounds = {-std::numeric_limits<double>::infinity()};
  std::vector<double> upper_bounds = {5.0};
  problem.set_constraint_lower_bounds(lower_bounds.data(), lower_bounds.size());
  problem.set_constraint_upper_bounds(upper_bounds.data(), upper_bounds.size());
  std::vector<double> var_lower_bounds = {0.0, 0.0, 0.0, 0.0};
  std::vector<double> var_upper_bounds = {1.0, 1.0, 1.0, 1.0};
  problem.set_variable_lower_bounds(var_lower_bounds.data(), var_lower_bounds.size());
  problem.set_variable_upper_bounds(var_upper_bounds.data(), var_upper_bounds.size());
  std::vector<double> objective_coefficients = {0.0, 0.0, 0.0, 0.0};
  problem.set_objective_coefficients(objective_coefficients.data(), objective_coefficients.size());
  std::vector<char> variable_types = {'I', 'I', 'I', 'I'};
  problem.set_variable_types(variable_types);
  problem.set_maximize(false);
  return problem;
}

detail::clique_table_t<int, double> build_clique_table_for_model_with_min_size(
  const raft::handle_t& handle,
  const mps_parser::mps_data_model_t<int, double>& model,
  int min_clique_size)
{
  auto op_problem = mps_data_model_to_optimization_problem(&handle, model);
  detail::problem_t<int, double> mip_problem(op_problem);
  dual_simplex::user_problem_t<int, double> host_problem(op_problem.get_handle_ptr());
  mip_problem.get_host_user_problem(host_problem);

  detail::clique_config_t clique_config;
  clique_config.min_clique_size = min_clique_size;
  detail::clique_table_t<int, double> clique_table(2 * host_problem.num_cols,
                                                   clique_config.min_clique_size,
                                                   clique_config.max_clique_size_for_extension);

  mip_solver_settings_t<int, double> settings;
  cuopt::timer_t timer(std::numeric_limits<double>::infinity());
  detail::build_clique_table(host_problem, clique_table, settings.tolerances, true, true, timer);
  return clique_table;
}

detail::clique_table_t<int, double> build_clique_table_for_model(
  const raft::handle_t& handle, const mps_parser::mps_data_model_t<int, double>& model)
{
  return build_clique_table_for_model_with_min_size(handle, model, 1);
}

std::vector<std::vector<char>> build_original_adjacency_matrix(
  detail::clique_table_t<int, double>& clique_table, int num_vars)
{
  std::vector<std::vector<char>> adj(num_vars, std::vector<char>(num_vars, 0));
  for (int i = 0; i < num_vars; ++i) {
    for (int j = i + 1; j < num_vars; ++j) {
      if (clique_table.check_adjacency(i, j)) {
        adj[i][j] = 1;
        adj[j][i] = 1;
      }
    }
  }
  return adj;
}

std::vector<std::vector<int>> maximal_cliques_bruteforce(const std::vector<std::vector<char>>& adj)
{
  const int n = static_cast<int>(adj.size());
  if (n <= 0 || n > 20) { return {}; }
  const uint64_t total_masks = (uint64_t{1} << n);
  std::vector<std::vector<int>> maximal_cliques;

  auto is_mask_clique = [&](uint64_t mask) {
    for (int i = 0; i < n; ++i) {
      if ((mask & (uint64_t{1} << i)) == 0) { continue; }
      for (int j = i + 1; j < n; ++j) {
        if ((mask & (uint64_t{1} << j)) == 0) { continue; }
        if (!adj[i][j]) { return false; }
      }
    }
    return true;
  };

  for (uint64_t mask = 1; mask < total_masks; ++mask) {
    if (!is_mask_clique(mask)) { continue; }
    bool is_maximal = true;
    for (int v = 0; v < n && is_maximal; ++v) {
      if (mask & (uint64_t{1} << v)) { continue; }
      bool can_extend = true;
      for (int u = 0; u < n; ++u) {
        if ((mask & (uint64_t{1} << u)) == 0) { continue; }
        if (!adj[v][u]) {
          can_extend = false;
          break;
        }
      }
      if (can_extend) { is_maximal = false; }
    }
    if (!is_maximal) { continue; }
    std::vector<int> clique;
    for (int u = 0; u < n; ++u) {
      if (mask & (uint64_t{1} << u)) { clique.push_back(u); }
    }
    maximal_cliques.push_back(std::move(clique));
  }
  return maximal_cliques;
}

std::vector<std::vector<int>> canonicalize_cliques(std::vector<std::vector<int>> cliques)
{
  for (auto& clique : cliques) {
    std::sort(clique.begin(), clique.end());
  }
  std::sort(cliques.begin(), cliques.end(), [](const auto& a, const auto& b) {
    if (a.size() != b.size()) { return a.size() < b.size(); }
    return a < b;
  });
  cliques.erase(std::unique(cliques.begin(), cliques.end()), cliques.end());
  return cliques;
}

std::vector<std::vector<int>> adjacency_matrix_to_list(const std::vector<std::vector<char>>& adj)
{
  const int n = static_cast<int>(adj.size());
  std::vector<std::vector<int>> adj_list(n);
  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < n; ++j) {
      if (adj[i][j]) { adj_list[i].push_back(j); }
    }
  }
  return adj_list;
}

std::vector<std::vector<int>> maximal_cliques_from_production_algorithm(
  const std::vector<std::vector<char>>& adj)
{
  const auto adj_list = adjacency_matrix_to_list(adj);
  std::vector<double> weights(adj_list.size(), 1.0);
  auto cliques = dual_simplex::find_maximal_cliques_for_test(
    adj_list, weights, 0.0, 100000, std::numeric_limits<double>::infinity());
  return canonicalize_cliques(std::move(cliques));
}

double original_clique_sum(const std::vector<int>& clique_vars,
                           const std::vector<double>& assignment)
{
  double lhs = 0.0;
  for (const auto var : clique_vars) {
    lhs += assignment[var];
  }
  return lhs;
}

std::string format_phase2_panic_dump(const mps_parser::mps_data_model_t<int, double>& problem,
                                     const std::vector<int>& clique_vars,
                                     const std::vector<double>& x_star)
{
  std::ostringstream out;
  const auto& var_lb = problem.get_variable_lower_bounds();
  const auto& var_ub = problem.get_variable_upper_bounds();
  out << "\nClique vars:";
  for (auto v : clique_vars) {
    out << " x" << v << "(value=" << x_star[v] << ", lb=" << var_lb[v] << ", ub=" << var_ub[v]
        << ")";
  }

  std::unordered_set<int> clique_var_set(clique_vars.begin(), clique_vars.end());
  const auto& values = problem.get_constraint_matrix_values();
  const auto& cols   = problem.get_constraint_matrix_indices();
  const auto& rows   = problem.get_constraint_matrix_offsets();
  const auto& clb    = problem.get_constraint_lower_bounds();
  const auto& cub    = problem.get_constraint_upper_bounds();

  out << "\nRelated constraints:";
  for (size_t row = 0; row + 1 < rows.size(); ++row) {
    bool touches_clique = false;
    for (int p = rows[row]; p < rows[row + 1]; ++p) {
      if (clique_var_set.count(cols[p]) > 0) {
        touches_clique = true;
        break;
      }
    }
    if (!touches_clique) { continue; }
    out << "\n  row " << row << ": ";
    for (int p = rows[row]; p < rows[row + 1]; ++p) {
      if (p > rows[row]) { out << " + "; }
      out << values[p] << "*x" << cols[p];
    }
    out << " in [" << clb[row] << ", " << cub[row] << "]";
  }
  return out.str();
}

void disable_non_clique_cuts(mip_solver_settings_t<int, double>& settings)
{
  settings.clique_cuts                = 1;
  settings.max_cut_passes             = 10;
  settings.mixed_integer_gomory_cuts  = 0;
  settings.knapsack_cuts              = 0;
  settings.mir_cuts                   = 0;
  settings.strong_chvatal_gomory_cuts = 0;
}

void disable_all_cuts(mip_solver_settings_t<int, double>& settings)
{
  settings.max_cut_passes             = 0;
  settings.clique_cuts                = 0;
  settings.mixed_integer_gomory_cuts  = 0;
  settings.knapsack_cuts              = 0;
  settings.mir_cuts                   = 0;
  settings.strong_chvatal_gomory_cuts = 0;
}

bool cut_is_invalid_for_incumbent(const std::vector<int>& cut_vars,
                                  const std::vector<double>& incumbent,
                                  double tol)
{
  return original_clique_sum(cut_vars, incumbent) > 1.0 + tol;
}

bool prefix_has_invalid_cut(const std::vector<std::vector<int>>& dumped_cuts,
                            size_t prefix_end_exclusive,
                            const std::vector<double>& incumbent,
                            double tol)
{
  for (size_t i = 0; i < prefix_end_exclusive; ++i) {
    if (cut_is_invalid_for_incumbent(dumped_cuts[i], incumbent, tol)) { return true; }
  }
  return false;
}

std::optional<size_t> isolate_first_invalid_cut_by_bisection(
  const std::vector<std::vector<int>>& dumped_cuts,
  const std::vector<double>& incumbent,
  double tol)
{
  if (!prefix_has_invalid_cut(dumped_cuts, dumped_cuts.size(), incumbent, tol)) {
    return std::nullopt;
  }
  size_t lo = 0;
  size_t hi = dumped_cuts.size() - 1;
  while (lo < hi) {
    const size_t mid = lo + (hi - lo) / 2;
    if (prefix_has_invalid_cut(dumped_cuts, mid + 1, incumbent, tol)) {
      hi = mid;
    } else {
      lo = mid + 1;
    }
  }
  return lo;
}

}  // namespace

// Problem data for the mixed integer linear programming problem
mps_parser::mps_data_model_t<int, double> create_cuts_problem_1()
{
  // Create problem instance
  mps_parser::mps_data_model_t<int, double> problem;

  // Solve the problem
  // minimize -7*x1 -2*x2
  // subject to -1*x1 + 2*x2 <= 4
  //            5*x1 + 1*x2 <= 20
  //            -2*x1 -2*x2 <= -7

  // Set up constraint matrix in CSR format
  std::vector<int> offsets         = {0, 2, 4, 6};
  std::vector<int> indices         = {0, 1, 0, 1, 0, 1};
  std::vector<double> coefficients = {-1.0, 2.0, 5.0, 1.0, -2.0, -2.0};
  problem.set_csr_constraint_matrix(coefficients.data(),
                                    coefficients.size(),
                                    indices.data(),
                                    indices.size(),
                                    offsets.data(),
                                    offsets.size());

  // Set constraint bounds
  std::vector<double> lower_bounds = {-std::numeric_limits<double>::infinity(),
                                      -std::numeric_limits<double>::infinity(),
                                      -std::numeric_limits<double>::infinity()};
  std::vector<double> upper_bounds = {4.0, 20.0, -7.0};
  problem.set_constraint_lower_bounds(lower_bounds.data(), lower_bounds.size());
  problem.set_constraint_upper_bounds(upper_bounds.data(), upper_bounds.size());

  // Set variable bounds
  std::vector<double> var_lower_bounds = {0.0, 0.0};
  std::vector<double> var_upper_bounds = {10.0, 10.0};
  problem.set_variable_lower_bounds(var_lower_bounds.data(), var_lower_bounds.size());
  problem.set_variable_upper_bounds(var_upper_bounds.data(), var_upper_bounds.size());

  // Set objective coefficients (minimize -7*x1 -2*x2)
  std::vector<double> objective_coefficients = {-7.0, -2.0};
  problem.set_objective_coefficients(objective_coefficients.data(), objective_coefficients.size());

  // Set variable types
  std::vector<char> variable_types = {'I', 'I'};
  problem.set_variable_types(variable_types);

  return problem;
}

TEST(cuts, test_cuts_1)
{
  const raft::handle_t handle_{};
  mip_solver_settings_t<int, double> settings;
  constexpr double test_time_limit = 1.;

  // Create the problem
  auto problem = create_cuts_problem_1();

  settings.time_limit                  = test_time_limit;
  settings.max_cut_passes              = 1;
  mip_solution_t<int, double> solution = solve_mip(&handle_, problem, settings);
  EXPECT_EQ(solution.get_termination_status(), mip_termination_status_t::Optimal);

  double obj_val = solution.get_objective_value();
  // Expected objective value from documentation example is approximately -28
  EXPECT_NEAR(-28, obj_val, 1e-3);

  EXPECT_EQ(solution.get_num_nodes(), 0);
}

// Problem data for the mixed integer linear programming problem
mps_parser::mps_data_model_t<int, double> create_cuts_problem_2()
{
  // Create problem instance
  mps_parser::mps_data_model_t<int, double> problem;

  // Solve the problem
  // minimize -86*y1 -4*y2 -40*y3
  // subject to 774*y1 + 76*y2 + 42*y3 <= 875
  //            67*y1 + 27*y2 + 53*y3 <= 875
  //            y1, y2, y3 in {0, 1}

  // Set up constraint matrix in CSR format
  std::vector<int> offsets         = {0, 3, 6};
  std::vector<int> indices         = {0, 1, 2, 0, 1, 2};
  std::vector<double> coefficients = {774.0, 76.0, 42.0, 67.0, 27.0, 53.0};
  problem.set_csr_constraint_matrix(coefficients.data(),
                                    coefficients.size(),
                                    indices.data(),
                                    indices.size(),
                                    offsets.data(),
                                    offsets.size());

  // Set constraint bounds
  std::vector<double> lower_bounds = {-std::numeric_limits<double>::infinity(),
                                      -std::numeric_limits<double>::infinity()};
  std::vector<double> upper_bounds = {875.0, 875.0};
  problem.set_constraint_lower_bounds(lower_bounds.data(), lower_bounds.size());
  problem.set_constraint_upper_bounds(upper_bounds.data(), upper_bounds.size());

  // Set variable bounds
  std::vector<double> var_lower_bounds = {0.0, 0.0, 0.0};
  std::vector<double> var_upper_bounds = {1.0, 1.0, 1.0};
  problem.set_variable_lower_bounds(var_lower_bounds.data(), var_lower_bounds.size());
  problem.set_variable_upper_bounds(var_upper_bounds.data(), var_upper_bounds.size());

  // Set objective coefficients (minimize -86*y1 -4*y2 -40*y3)
  std::vector<double> objective_coefficients = {-86.0, -4.0, -40.0};
  problem.set_objective_coefficients(objective_coefficients.data(), objective_coefficients.size());

  // Set variable types
  std::vector<char> variable_types = {'I', 'I', 'I'};
  problem.set_variable_types(variable_types);

  return problem;
}

TEST(cuts, test_cuts_2)
{
  const raft::handle_t handle_{};
  mip_solver_settings_t<int, double> settings;
  constexpr double test_time_limit = 1.;

  // Create the problem
  auto problem = create_cuts_problem_2();

  settings.time_limit                  = test_time_limit;
  settings.max_cut_passes              = 10;
  settings.presolver                   = presolver_t::None;
  mip_solution_t<int, double> solution = solve_mip(&handle_, problem, settings);
  EXPECT_EQ(solution.get_termination_status(), mip_termination_status_t::Optimal);

  double obj_val = solution.get_objective_value();
  // Expected objective value from documentation example is approximately -126
  EXPECT_NEAR(-126, obj_val, 1e-3);

  EXPECT_EQ(solution.get_num_nodes(), 0);
}

TEST(cuts, clique_phase1_smoke_conflict_graph_edges)
{
  const raft::handle_t handle{};
  auto problem      = create_pairwise_triangle_with_isolated_variable_problem();
  auto clique_table = build_clique_table_for_model(handle, problem);

  // Positive edges from triangle.
  EXPECT_TRUE(clique_table.check_adjacency(0, 1));
  EXPECT_TRUE(clique_table.check_adjacency(1, 0));
  EXPECT_TRUE(clique_table.check_adjacency(1, 2));
  EXPECT_TRUE(clique_table.check_adjacency(2, 1));
  EXPECT_TRUE(clique_table.check_adjacency(0, 2));
  EXPECT_TRUE(clique_table.check_adjacency(2, 0));

  // Negative edges to isolated x3.
  EXPECT_FALSE(clique_table.check_adjacency(0, 3));
  EXPECT_FALSE(clique_table.check_adjacency(3, 0));
  EXPECT_FALSE(clique_table.check_adjacency(1, 3));
  EXPECT_FALSE(clique_table.check_adjacency(3, 1));
  EXPECT_FALSE(clique_table.check_adjacency(2, 3));
  EXPECT_FALSE(clique_table.check_adjacency(3, 2));

  // Self is never an edge.
  EXPECT_FALSE(clique_table.check_adjacency(3, 3));
}

TEST(cuts, clique_phase1_unit_maximal_clique_finder_hardcoded_adj)
{
  // Hardcoded graph:
  // triangle (0,1,2) and an extra edge (2,3)
  std::vector<std::vector<char>> adj = {
    {0, 1, 1, 0},
    {1, 0, 1, 0},
    {1, 1, 0, 1},
    {0, 0, 1, 0},
  };

  auto maximal_bruteforce = canonicalize_cliques(maximal_cliques_bruteforce(adj));
  auto maximal_internal   = maximal_cliques_from_production_algorithm(adj);
  EXPECT_EQ(maximal_internal, maximal_bruteforce);
  bool found_triangle = false;
  for (const auto& clique : maximal_internal) {
    if (clique.size() == 3 && clique[0] == 0 && clique[1] == 1 && clique[2] == 2) {
      found_triangle = true;
      break;
    }
  }
  EXPECT_TRUE(found_triangle);
}

TEST(cuts, clique_phase1_addtl_conflict_symmetry_and_reverse_lookup)
{
  const raft::handle_t handle{};
  auto problem      = create_weighted_addtl_conflict_problem();
  auto clique_table = build_clique_table_for_model_with_min_size(handle, problem, 1);

  ASSERT_FALSE(clique_table.addtl_cliques.empty());

  // Conflict introduced through additional clique path must be symmetric.
  EXPECT_TRUE(clique_table.check_adjacency(1, 3));
  EXPECT_TRUE(clique_table.check_adjacency(3, 1));

  // get_adj_set_of_var() must also include reverse lookup for addtl membership.
  auto adj_of_1 = clique_table.get_adj_set_of_var(1);
  auto adj_of_3 = clique_table.get_adj_set_of_var(3);
  EXPECT_TRUE(adj_of_1.count(3) > 0);
  EXPECT_TRUE(adj_of_3.count(1) > 0);
}

TEST(cuts, clique_phase1_remove_small_cliques_preserves_addtl_conflicts)
{
  const raft::handle_t handle{};
  auto problem = create_weighted_addtl_conflict_problem();
  // Force base clique {x2,x3} to be considered "small" and removed.
  auto clique_table = build_clique_table_for_model_with_min_size(handle, problem, 2);

  EXPECT_TRUE(clique_table.first.empty());
  EXPECT_TRUE(clique_table.addtl_cliques.empty());

  // Conflicts must remain materialized in adj_list_small_cliques after removals.
  EXPECT_TRUE(clique_table.check_adjacency(1, 3));
  EXPECT_TRUE(clique_table.check_adjacency(3, 1));
  EXPECT_TRUE(clique_table.check_adjacency(2, 3));
  EXPECT_TRUE(clique_table.check_adjacency(3, 2));
  EXPECT_FALSE(clique_table.check_adjacency(0, 3));
}

TEST(cuts, clique_phase2_no_cut_off_optimal_solution_validation)
{
  const raft::handle_t handle{};
  auto problem = create_pairwise_triangle_set_packing_problem();

  mip_solver_settings_t<int, double> settings;
  settings.time_limit = 10.0;
  settings.presolver  = presolver_t::None;
  disable_all_cuts(settings);

  auto mip_solution = solve_mip(&handle, problem, settings);
  ASSERT_EQ(mip_solution.get_termination_status(), mip_termination_status_t::Optimal);
  auto x_star = cuopt::host_copy(mip_solution.get_solution(), handle.get_stream());

  auto clique_table = build_clique_table_for_model(handle, problem);
  auto adj          = build_original_adjacency_matrix(clique_table, problem.get_n_variables());
  auto maximal      = maximal_cliques_bruteforce(adj);
  ASSERT_FALSE(maximal.empty());

  for (const auto& clique_vars : maximal) {
    if (clique_vars.size() < 2) { continue; }
    const double lhs = original_clique_sum(clique_vars, x_star);
    ASSERT_LE(lhs, 1.0 + kCliqueTestTol) << format_phase2_panic_dump(problem, clique_vars, x_star);
  }
}

TEST(cuts, clique_phase3_fractional_separation_must_cut_off)
{
  const raft::handle_t handle{};
  auto mip_problem = create_pairwise_triangle_set_packing_problem();

  auto lp_relaxation = mip_problem;
  std::vector<char> all_continuous(lp_relaxation.get_n_variables(), 'C');
  lp_relaxation.set_variable_types(all_continuous);

  pdlp_solver_settings_t<int, double> lp_settings{};
  lp_settings.time_limit = 10.0;
  lp_settings.presolver  = presolver_t::None;
  lp_settings.set_optimality_tolerance(1e-8);

  auto lp_solution = solve_lp(&handle, lp_relaxation, lp_settings);
  ASSERT_EQ(lp_solution.get_termination_status(), pdlp_termination_status_t::Optimal);
  auto x_bar = cuopt::host_copy(lp_solution.get_primal_solution(), handle.get_stream());

  auto clique_table = build_clique_table_for_model(handle, mip_problem);
  auto adj          = build_original_adjacency_matrix(clique_table, mip_problem.get_n_variables());
  auto maximal      = maximal_cliques_from_production_algorithm(adj);

  bool found_separating_clique = false;
  for (const auto& clique_vars : maximal) {
    if (clique_vars.size() < 2) { continue; }
    const double lhs = original_clique_sum(clique_vars, x_bar);
    if (lhs > 1.0 + kCliqueTestTol) {
      found_separating_clique = true;
      break;
    }
  }
  EXPECT_TRUE(found_separating_clique);
}

TEST(cuts, clique_phase4_fault_isolation_binary_search)
{
  // Simulated incumbent x* and dumped cuts.
  // First invalid cut is at index 2: {0,1} gives 2 > 1.
  const std::vector<double> incumbent             = {1.0, 1.0, 0.0, 0.0};
  const std::vector<std::vector<int>> dumped_cuts = {
    {0, 2},  // valid
    {1, 3},  // valid
    {0, 1},  // invalid
    {2, 3},  // valid
  };

  auto first_invalid =
    isolate_first_invalid_cut_by_bisection(dumped_cuts, incumbent, kCliqueTestTol);
  ASSERT_TRUE(first_invalid.has_value());
  EXPECT_EQ(first_invalid.value(), 2);
}

TEST(cuts, clique_phase4_tree_depth_limit_smoke)
{
  const raft::handle_t handle{};
  auto problem = create_pairwise_triangle_set_packing_problem();

  mip_solver_settings_t<int, double> root_only_settings;
  root_only_settings.time_limit = 10.0;
  root_only_settings.presolver  = presolver_t::None;
  root_only_settings.node_limit = 0;
  disable_non_clique_cuts(root_only_settings);

  mip_solver_settings_t<int, double> deeper_settings = root_only_settings;
  deeper_settings.node_limit                         = 100;

  auto root_only_solution = solve_mip(&handle, problem, root_only_settings);
  auto deeper_solution    = solve_mip(&handle, problem, deeper_settings);

  EXPECT_EQ(deeper_solution.get_termination_status(), mip_termination_status_t::Optimal);
  EXPECT_NE(root_only_solution.get_termination_status(), mip_termination_status_t::Infeasible);
  if (root_only_solution.get_termination_status() == mip_termination_status_t::Optimal) {
    EXPECT_NEAR(
      root_only_solution.get_objective_value(), deeper_solution.get_objective_value(), 1e-6);
  }
}

TEST(cuts, clique_phase5_ignores_non_binary_variables)
{
  const raft::handle_t handle{};
  auto problem      = create_binary_continuous_mixed_conflict_problem();
  auto clique_table = build_clique_table_for_model(handle, problem);

  EXPECT_TRUE(clique_table.check_adjacency(0, 2));
  EXPECT_FALSE(clique_table.check_adjacency(0, 1));
  EXPECT_FALSE(clique_table.check_adjacency(1, 2));
}

TEST(cuts, clique_phase5_ignores_fractional_binary_bounds)
{
  const raft::handle_t handle{};
  auto problem      = create_near_binary_bound_conflict_problem();
  auto clique_table = build_clique_table_for_model(handle, problem);

  EXPECT_FALSE(clique_table.check_adjacency(0, 1));
}

}  // namespace cuopt::linear_programming::test
