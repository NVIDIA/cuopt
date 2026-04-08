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
#include <memory>
#include <mutex>
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

mps_parser::mps_data_model_t<int, double>& get_neos8_model_cached()
{
  static std::once_flag init_flag;
  static std::unique_ptr<mps_parser::mps_data_model_t<int, double>> model_ptr;
  std::call_once(init_flag, []() {
    const auto neos8_path = make_path_absolute("mip/neos8.mps");
    auto neos8_model      = cuopt::mps_parser::parse_mps<int, double>(neos8_path, false);
    model_ptr = std::make_unique<mps_parser::mps_data_model_t<int, double>>(std::move(neos8_model));
  });
  cuopt_assert(model_ptr != nullptr, "Failed to initialize cached neos8 model");
  return *model_ptr;
}

detail::clique_table_t<int, double>& get_neos8_clique_table_cached()
{
  static std::once_flag init_flag;
  static std::unique_ptr<detail::clique_table_t<int, double>> clique_table_ptr;
  std::call_once(init_flag, []() {
    const raft::handle_t handle{};
    auto& neos8_model = get_neos8_model_cached();
    auto clique_table = build_clique_table_for_model(handle, neos8_model);
    clique_table_ptr =
      std::make_unique<detail::clique_table_t<int, double>>(std::move(clique_table));
  });
  cuopt_assert(clique_table_ptr != nullptr, "Failed to initialize cached neos8 clique table");
  return *clique_table_ptr;
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

struct neos8_mip_solution_cache_t {
  mip_termination_status_t status;
  std::vector<double> primal;
  double objective;
};

struct neos8_lp_solution_cache_t {
  pdlp_termination_status_t status;
  std::vector<double> primal;
};

neos8_mip_solution_cache_t& get_neos8_optimal_solution_no_cuts_cached()
{
  static std::once_flag init_flag;
  static std::unique_ptr<neos8_mip_solution_cache_t> solution_ptr;
  std::call_once(init_flag, []() {
    const raft::handle_t handle{};
    auto& neos8_model = get_neos8_model_cached();
    mip_solver_settings_t<int, double> settings;
    settings.time_limit = 120.0;
    settings.presolver  = presolver_t::None;
    disable_all_cuts(settings);

    auto mip_solution = solve_mip(&handle, neos8_model, settings);
    auto cache        = std::make_unique<neos8_mip_solution_cache_t>();
    cache->status     = mip_solution.get_termination_status();
    cache->objective  = mip_solution.get_objective_value();
    cache->primal     = cuopt::host_copy(mip_solution.get_solution(), handle.get_stream());
    solution_ptr      = std::move(cache);
  });
  cuopt_assert(solution_ptr != nullptr, "Failed to initialize cached neos8 no-cut MIP solution");
  return *solution_ptr;
}

neos8_lp_solution_cache_t& get_neos8_lp_relaxation_solution_cached()
{
  static std::once_flag init_flag;
  static std::unique_ptr<neos8_lp_solution_cache_t> solution_ptr;
  std::call_once(init_flag, []() {
    const raft::handle_t handle{};
    auto lp_relaxation = get_neos8_model_cached();
    std::vector<char> all_continuous(lp_relaxation.get_n_variables(), 'C');
    lp_relaxation.set_variable_types(all_continuous);

    pdlp_solver_settings_t<int, double> lp_settings{};
    lp_settings.time_limit = 120.0;
    lp_settings.presolver  = presolver_t::None;
    lp_settings.set_optimality_tolerance(1e-8);

    auto lp_solution = solve_lp(&handle, lp_relaxation, lp_settings);
    auto cache       = std::make_unique<neos8_lp_solution_cache_t>();
    cache->status    = lp_solution.get_termination_status();
    cache->primal    = cuopt::host_copy(lp_solution.get_primal_solution(), handle.get_stream());
    solution_ptr     = std::move(cache);
  });
  cuopt_assert(solution_ptr != nullptr, "Failed to initialize cached neos8 LP relaxation solution");
  return *solution_ptr;
}

bool is_binary_var_for_clique_literals(const mps_parser::mps_data_model_t<int, double>& problem,
                                       int var_idx,
                                       double bound_tol)
{
  const auto& var_types = problem.get_variable_types();
  const auto& var_lb    = problem.get_variable_lower_bounds();
  const auto& var_ub    = problem.get_variable_upper_bounds();
  return var_types[var_idx] != 'C' && var_lb[var_idx] >= -bound_tol &&
         var_ub[var_idx] <= 1.0 + bound_tol;
}

std::vector<std::vector<int>> build_fractional_literal_cliques_for_assignment(
  const mps_parser::mps_data_model_t<int, double>& problem,
  detail::clique_table_t<int, double>& clique_table,
  const std::vector<double>& assignment,
  double integer_tol,
  double bound_tol,
  int max_calls)
{
  const int num_vars = problem.get_n_variables();
  cuopt_assert(static_cast<int>(assignment.size()) >= num_vars,
               "Assignment size mismatch in fractional literal clique builder");

  std::vector<int> vertices;
  std::vector<double> weights;
  vertices.reserve(2 * num_vars);
  weights.reserve(2 * num_vars);
  for (int j = 0; j < num_vars; ++j) {
    if (!is_binary_var_for_clique_literals(problem, j, bound_tol)) { continue; }
    const double xj = assignment[j];
    if (std::abs(xj - std::round(xj)) <= integer_tol) { continue; }
    vertices.push_back(j);
    weights.push_back(xj);
    vertices.push_back(j + num_vars);
    weights.push_back(1.0 - xj);
  }
  if (vertices.empty()) { return {}; }

  std::vector<int> vertex_to_local(2 * num_vars, -1);
  std::vector<char> in_subgraph(2 * num_vars, 0);
  for (size_t idx = 0; idx < vertices.size(); ++idx) {
    vertex_to_local[vertices[idx]] = static_cast<int>(idx);
    in_subgraph[vertices[idx]]     = 1;
  }

  std::vector<std::vector<int>> adj_local(vertices.size());
  for (size_t idx = 0; idx < vertices.size(); ++idx) {
    const auto vertex_idx = vertices[idx];
    auto adj_set          = clique_table.get_adj_set_of_var(vertex_idx);
    auto& adj             = adj_local[idx];
    adj.reserve(adj_set.size());
    for (const auto neighbor : adj_set) {
      cuopt_assert(neighbor >= 0 && neighbor < 2 * num_vars,
                   "Neighbor out of range in fractional literal clique builder");
      if (!in_subgraph[neighbor]) { continue; }
      const auto local_neighbor = vertex_to_local[neighbor];
      if (local_neighbor >= 0) { adj.push_back(local_neighbor); }
    }
  }

  auto cliques_local = dual_simplex::find_maximal_cliques_for_test(
    adj_local, weights, 1.0 + kCliqueTestTol, max_calls, std::numeric_limits<double>::infinity());
  std::vector<std::vector<int>> cliques_global;
  cliques_global.reserve(cliques_local.size());
  for (auto& local_clique : cliques_local) {
    std::vector<int> global_clique;
    global_clique.reserve(local_clique.size());
    for (const auto local_idx : local_clique) {
      cuopt_assert(local_idx >= 0 && static_cast<size_t>(local_idx) < vertices.size(),
                   "Local clique index out of range");
      global_clique.push_back(vertices[local_idx]);
    }
    cliques_global.push_back(std::move(global_clique));
  }
  return canonicalize_cliques(std::move(cliques_global));
}

std::vector<std::vector<int>>& get_neos8_fractional_literal_cliques_cached()
{
  static std::once_flag init_flag;
  static std::unique_ptr<std::vector<std::vector<int>>> cliques_ptr;
  std::call_once(init_flag, []() {
    auto& neos8_model   = get_neos8_model_cached();
    auto& clique_table  = get_neos8_clique_table_cached();
    auto& lp_relaxation = get_neos8_lp_relaxation_solution_cached();
    auto cliques        = build_fractional_literal_cliques_for_assignment(
      neos8_model, clique_table, lp_relaxation.primal, kCliqueTestTol, kCliqueTestTol, 100000);
    cliques_ptr = std::make_unique<std::vector<std::vector<int>>>(std::move(cliques));
  });
  cuopt_assert(cliques_ptr != nullptr, "Failed to initialize cached neos8 dumped literal cliques");
  return *cliques_ptr;
}

double literal_clique_cut_violation(const std::vector<int>& literal_clique,
                                    const std::vector<double>& assignment,
                                    int num_vars)
{
  cuopt_assert(static_cast<int>(assignment.size()) >= num_vars,
               "Assignment size mismatch in literal clique violation");
  double dot              = 0.0;
  int num_complement_vars = 0;
  for (const auto literal : literal_clique) {
    cuopt_assert(literal >= 0 && literal < 2 * num_vars, "Literal out of range");
    const int var_idx        = literal % num_vars;
    const bool is_complement = literal >= num_vars;
    if (is_complement) {
      num_complement_vars++;
      dot += assignment[var_idx];
    } else {
      dot -= assignment[var_idx];
    }
  }
  const double rhs = static_cast<double>(num_complement_vars - 1);
  return rhs - dot;
}

std::string format_phase2_literal_panic_dump(const std::vector<int>& literal_clique,
                                             const std::vector<double>& incumbent,
                                             int num_vars)
{
  std::ostringstream out;
  out << "\nLiteral clique:";
  for (const auto literal : literal_clique) {
    const bool is_complement = literal >= num_vars;
    const int var_idx        = literal % num_vars;
    out << " " << (is_complement ? "~x" : "x") << var_idx << "(value=" << incumbent[var_idx] << ")";
  }
  out << "\nViolation: " << literal_clique_cut_violation(literal_clique, incumbent, num_vars);
  return out.str();
}

bool literal_cut_is_invalid_for_incumbent(const std::vector<int>& literal_clique,
                                          const std::vector<double>& incumbent,
                                          int num_vars,
                                          double tol)
{
  return literal_clique_cut_violation(literal_clique, incumbent, num_vars) > tol;
}

bool prefix_has_invalid_literal_cut(const std::vector<std::vector<int>>& dumped_cuts,
                                    size_t prefix_end_exclusive,
                                    const std::vector<double>& incumbent,
                                    int num_vars,
                                    double tol)
{
  for (size_t i = 0; i < prefix_end_exclusive; ++i) {
    if (literal_cut_is_invalid_for_incumbent(dumped_cuts[i], incumbent, num_vars, tol)) {
      return true;
    }
  }
  return false;
}

std::optional<size_t> isolate_first_invalid_literal_cut_by_bisection(
  const std::vector<std::vector<int>>& dumped_cuts,
  const std::vector<double>& incumbent,
  int num_vars,
  double tol)
{
  if (!prefix_has_invalid_literal_cut(dumped_cuts, dumped_cuts.size(), incumbent, num_vars, tol)) {
    return std::nullopt;
  }
  size_t lo = 0;
  size_t hi = dumped_cuts.size() - 1;
  while (lo < hi) {
    const size_t mid = lo + (hi - lo) / 2;
    if (prefix_has_invalid_literal_cut(dumped_cuts, mid + 1, incumbent, num_vars, tol)) {
      hi = mid;
    } else {
      lo = mid + 1;
    }
  }
  return lo;
}

mps_parser::mps_data_model_t<int, double>& get_neos8_lp_relaxation_model_cached()
{
  static std::once_flag init_flag;
  static std::unique_ptr<mps_parser::mps_data_model_t<int, double>> model_ptr;
  std::call_once(init_flag, []() {
    auto lp_relaxation = get_neos8_model_cached();
    std::vector<char> all_continuous(lp_relaxation.get_n_variables(), 'C');
    lp_relaxation.set_variable_types(all_continuous);
    model_ptr =
      std::make_unique<mps_parser::mps_data_model_t<int, double>>(std::move(lp_relaxation));
  });
  cuopt_assert(model_ptr != nullptr, "Failed to initialize cached neos8 LP relaxation model");
  return *model_ptr;
}

mps_parser::mps_data_model_t<int, double> append_literal_cut_prefix_to_lp_model(
  const mps_parser::mps_data_model_t<int, double>& base_lp_model,
  const std::vector<std::vector<int>>& dumped_cuts,
  size_t prefix_end_exclusive,
  int num_vars)
{
  auto model_with_cuts = base_lp_model;
  if (prefix_end_exclusive == 0) { return model_with_cuts; }

  std::vector<double> matrix_values  = base_lp_model.get_constraint_matrix_values();
  std::vector<int> matrix_indices    = base_lp_model.get_constraint_matrix_indices();
  std::vector<int> matrix_offsets    = base_lp_model.get_constraint_matrix_offsets();
  std::vector<double> constraint_lbs = base_lp_model.get_constraint_lower_bounds();
  std::vector<double> constraint_ubs = base_lp_model.get_constraint_upper_bounds();
  std::vector<std::string> row_names = base_lp_model.get_row_names();
  if (matrix_offsets.empty()) { matrix_offsets.push_back(0); }

  const size_t cuts_to_apply = std::min(prefix_end_exclusive, dumped_cuts.size());
  for (size_t cut_idx = 0; cut_idx < cuts_to_apply; ++cut_idx) {
    const auto& literal_cut = dumped_cuts[cut_idx];

    std::vector<int> row_vars;
    std::vector<double> row_coeffs;
    row_vars.reserve(literal_cut.size());
    row_coeffs.reserve(literal_cut.size());

    int num_complements = 0;
    for (const auto literal : literal_cut) {
      cuopt_assert(literal >= 0 && literal < 2 * num_vars,
                   "Literal out of range for LP cut append");
      const int var_idx        = literal % num_vars;
      const bool is_complement = literal >= num_vars;
      if (is_complement) { num_complements++; }
      const double coeff = is_complement ? 1.0 : -1.0;

      bool found = false;
      for (size_t t = 0; t < row_vars.size(); ++t) {
        if (row_vars[t] == var_idx) {
          row_coeffs[t] += coeff;
          found = true;
          break;
        }
      }
      if (!found) {
        row_vars.push_back(var_idx);
        row_coeffs.push_back(coeff);
      }
    }

    std::vector<int> order(row_vars.size());
    std::iota(order.begin(), order.end(), 0);
    std::sort(order.begin(), order.end(), [&](int a, int b) { return row_vars[a] < row_vars[b]; });
    for (const auto pos : order) {
      const double coeff = row_coeffs[pos];
      if (std::abs(coeff) <= 1e-12) { continue; }
      matrix_indices.push_back(row_vars[pos]);
      matrix_values.push_back(coeff);
    }
    matrix_offsets.push_back(static_cast<int>(matrix_indices.size()));
    constraint_lbs.push_back(static_cast<double>(num_complements - 1));
    constraint_ubs.push_back(std::numeric_limits<double>::infinity());
    row_names.push_back("literal_cut_" + std::to_string(cut_idx));
  }

  model_with_cuts.set_csr_constraint_matrix(matrix_values.data(),
                                            matrix_values.size(),
                                            matrix_indices.data(),
                                            matrix_indices.size(),
                                            matrix_offsets.data(),
                                            matrix_offsets.size());
  model_with_cuts.set_constraint_lower_bounds(constraint_lbs.data(), constraint_lbs.size());
  model_with_cuts.set_constraint_upper_bounds(constraint_ubs.data(), constraint_ubs.size());
  model_with_cuts.set_row_names(row_names);
  return model_with_cuts;
}

pdlp_termination_status_t solve_lp_with_literal_cut_prefix(
  const std::vector<std::vector<int>>& dumped_cuts, size_t prefix_end_exclusive, int num_vars)
{
  const raft::handle_t handle{};
  auto& base_lp_model  = get_neos8_lp_relaxation_model_cached();
  auto model_with_cuts = append_literal_cut_prefix_to_lp_model(
    base_lp_model, dumped_cuts, prefix_end_exclusive, num_vars);

  pdlp_solver_settings_t<int, double> lp_settings{};
  lp_settings.time_limit = 120.0;
  lp_settings.presolver  = presolver_t::None;
  lp_settings.set_optimality_tolerance(1e-8);

  auto lp_solution = solve_lp(&handle, model_with_cuts, lp_settings);
  return lp_solution.get_termination_status();
}

bool prefix_makes_lp_relaxation_infeasible(const std::vector<std::vector<int>>& dumped_cuts,
                                           size_t prefix_end_exclusive,
                                           int num_vars)
{
  const auto status = solve_lp_with_literal_cut_prefix(dumped_cuts, prefix_end_exclusive, num_vars);
  return status == pdlp_termination_status_t::PrimalInfeasible;
}

std::optional<size_t> isolate_first_lp_infeasible_literal_cut_by_bisection(
  const std::vector<std::vector<int>>& dumped_cuts, int num_vars)
{
  if (!prefix_makes_lp_relaxation_infeasible(dumped_cuts, dumped_cuts.size(), num_vars)) {
    return std::nullopt;
  }
  size_t lo = 0;
  size_t hi = dumped_cuts.size() - 1;
  while (lo < hi) {
    const size_t mid = lo + (hi - lo) / 2;
    if (prefix_makes_lp_relaxation_infeasible(dumped_cuts, mid + 1, num_vars)) {
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

  EXPECT_LE(solution.get_num_nodes(), 2);
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

TEST(cuts, test_duplicate_cuts_detection)
{
  dual_simplex::simplex_solver_settings_t<int, double> settings;
  dual_simplex::cut_pool_t<int, double> cut_pool(4, settings);
  dual_simplex::inequality_t<int, double> cut1;
  cut1.push_back(0, 1.0);
  cut1.push_back(1, 2.0);
  cut1.rhs = 1.0;
  cut_pool.add_cut(dual_simplex::cut_type_t::MIXED_INTEGER_GOMORY, cut1);
  dual_simplex::inequality_t<int, double> cut2;
  cut2.push_back(0, 2.0);
  cut2.push_back(1, 4.0);
  cut2.rhs = 2.0;
  cut_pool.add_cut(dual_simplex::cut_type_t::MIXED_INTEGER_GOMORY, cut2);
  dual_simplex::inequality_t<int, double> cut3;
  cut3.push_back(0, 0.1);
  cut3.push_back(2, 0.2);
  cut3.rhs = 1.0;
  cut_pool.add_cut(dual_simplex::cut_type_t::MIXED_INTEGER_GOMORY, cut3);
  dual_simplex::inequality_t<int, double> cut4;
  cut4.push_back(0, 0.2);
  cut4.push_back(2, 0.4);
  cut4.rhs = 1.0;
  cut_pool.add_cut(dual_simplex::cut_type_t::MIXED_INTEGER_GOMORY, cut4);
  dual_simplex::inequality_t<int, double> cut5;
  cut5.push_back(1, 10.0);
  cut5.push_back(3, 20.0);
  cut5.rhs = 0.1;
  cut_pool.add_cut(dual_simplex::cut_type_t::MIXED_INTEGER_GOMORY, cut5);
  dual_simplex::inequality_t<int, double> cut6;
  cut6.push_back(1, 20.0);
  cut6.push_back(3, 40.0);
  cut6.rhs = 0.2;
  cut_pool.add_cut(dual_simplex::cut_type_t::MIXED_INTEGER_GOMORY, cut6);
  dual_simplex::inequality_t<int, double> cut7;
  cut7.push_back(0, 1.0);
  cut7.push_back(1, 1.0);
  cut7.push_back(2, 1.0);
  cut7.push_back(3, 1.0);
  cut7.rhs = 1.0;
  cut_pool.add_cut(dual_simplex::cut_type_t::MIXED_INTEGER_GOMORY, cut7);
  dual_simplex::inequality_t<int, double> cut8;
  cut8.push_back(1, 3.0);
  cut8.rhs = 7.0;
  cut_pool.add_cut(dual_simplex::cut_type_t::MIXED_INTEGER_GOMORY, cut8);

  cut_pool.check_for_duplicate_cuts();
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

TEST(cuts, clique_neos8_phase1_addtl_indices_and_nonempty_graph)
{
  auto& clique_table = get_neos8_clique_table_cached();
  EXPECT_TRUE(!clique_table.first.empty() || !clique_table.addtl_cliques.empty());

  const size_t max_addtl_to_check = std::min<size_t>(clique_table.addtl_cliques.size(), 400);
  for (size_t k = 0; k < max_addtl_to_check; ++k) {
    const auto& addtl = clique_table.addtl_cliques[k];
    ASSERT_GE(addtl.clique_idx, 0);
    ASSERT_LT(static_cast<size_t>(addtl.clique_idx), clique_table.first.size());
    const auto& base = clique_table.first[addtl.clique_idx];
    ASSERT_GE(addtl.start_pos_on_clique, 0);
    ASSERT_LE(static_cast<size_t>(addtl.start_pos_on_clique), base.size());
  }
}

TEST(cuts, clique_neos8_phase1_addtl_suffix_conflicts_materialized)
{
  auto& clique_table = get_neos8_clique_table_cached();
  if (clique_table.addtl_cliques.empty()) {
    GTEST_SKIP() << "neos8 produced no additional cliques in this configuration";
  }

  size_t checked_addtl            = 0;
  const size_t max_addtl_to_check = std::min<size_t>(clique_table.addtl_cliques.size(), 200);
  for (size_t k = 0; k < max_addtl_to_check; ++k) {
    const auto& addtl = clique_table.addtl_cliques[k];
    if (addtl.clique_idx < 0 ||
        static_cast<size_t>(addtl.clique_idx) >= clique_table.first.size()) {
      continue;
    }
    const auto& base      = clique_table.first[addtl.clique_idx];
    const size_t start_at = static_cast<size_t>(addtl.start_pos_on_clique);
    if (start_at >= base.size()) { continue; }

    const size_t end_at = std::min(base.size(), start_at + 8);
    for (size_t p = start_at; p < end_at; ++p) {
      EXPECT_TRUE(clique_table.check_adjacency(addtl.vertex_idx, base[p]));
      EXPECT_TRUE(clique_table.check_adjacency(base[p], addtl.vertex_idx));
    }
    checked_addtl++;
  }
  EXPECT_GT(checked_addtl, 0);
}

TEST(cuts, clique_neos8_phase1_symmetry_and_degree_cache_consistency)
{
  auto& clique_table   = get_neos8_clique_table_cached();
  const int n_vertices = static_cast<int>(clique_table.var_clique_map_first.size());
  ASSERT_GT(n_vertices, 0);

  const int sample_size = std::min(n_vertices, 24);
  const int stride      = std::max(1, n_vertices / sample_size);
  std::vector<int> sampled_vertices(sample_size);
  for (int i = 0; i < sample_size; ++i) {
    sampled_vertices[i] = (i * stride) % n_vertices;
  }

  for (const auto v : sampled_vertices) {
    const auto deg_cached = clique_table.get_degree_of_var(v);
    const auto adj_set    = clique_table.get_adj_set_of_var(v);
    EXPECT_EQ(deg_cached, static_cast<int>(adj_set.size()));
    EXPECT_EQ(deg_cached, clique_table.get_degree_of_var(v));
  }

  for (int i = 0; i < sample_size; ++i) {
    for (int j = i + 1; j < sample_size; ++j) {
      const auto v1 = sampled_vertices[i];
      const auto v2 = sampled_vertices[j];
      EXPECT_EQ(clique_table.check_adjacency(v1, v2), clique_table.check_adjacency(v2, v1));
    }
  }
}

// Disabled: hits time limit on ARM (L4) instead of Optimal.
// https://github.com/NVIDIA/cuopt/issues/972
TEST(cuts, DISABLED_clique_neos8_phase2_no_cut_off_optimal_solution_validation)
{
  auto& no_cut_mip = get_neos8_optimal_solution_no_cuts_cached();
  ASSERT_EQ(no_cut_mip.status, mip_termination_status_t::Optimal);

  auto& lp_relaxation = get_neos8_lp_relaxation_solution_cached();
  ASSERT_EQ(lp_relaxation.status, pdlp_termination_status_t::Optimal);

  auto& dumped_literal_cuts = get_neos8_fractional_literal_cliques_cached();
  if (dumped_literal_cuts.empty()) {
    GTEST_SKIP() << "neos8 produced no candidate literal cliques from LP relaxation";
  }

  const int num_vars = get_neos8_model_cached().get_n_variables();
  for (size_t i = 0; i < dumped_literal_cuts.size(); ++i) {
    const double violation =
      literal_clique_cut_violation(dumped_literal_cuts[i], no_cut_mip.primal, num_vars);
    ASSERT_LE(violation, kCliqueTestTol)
      << "Invalid clique cut at index " << i
      << format_phase2_literal_panic_dump(dumped_literal_cuts[i], no_cut_mip.primal, num_vars);
  }
}

TEST(cuts, clique_neos8_phase3_fractional_separation_must_cut_off)
{
  auto& lp_relaxation = get_neos8_lp_relaxation_solution_cached();
  ASSERT_EQ(lp_relaxation.status, pdlp_termination_status_t::Optimal);

  auto& dumped_literal_cuts = get_neos8_fractional_literal_cliques_cached();
  if (dumped_literal_cuts.empty()) {
    GTEST_SKIP() << "neos8 produced no candidate literal cliques from LP relaxation";
  }

  const int num_vars = get_neos8_model_cached().get_n_variables();
  for (size_t i = 0; i < dumped_literal_cuts.size(); ++i) {
    const double violation =
      literal_clique_cut_violation(dumped_literal_cuts[i], lp_relaxation.primal, num_vars);
    ASSERT_GT(violation, kCliqueTestTol)
      << "Non-separating clique cut at index " << i
      << format_phase2_literal_panic_dump(dumped_literal_cuts[i], lp_relaxation.primal, num_vars);
  }
}

// Disabled: depends on phase2 cached result which fails on ARM (L4).
// https://github.com/NVIDIA/cuopt/issues/972
TEST(cuts, DISABLED_clique_neos8_phase4_fault_isolation_binary_search)
{
  auto& no_cut_mip = get_neos8_optimal_solution_no_cuts_cached();
  ASSERT_EQ(no_cut_mip.status, mip_termination_status_t::Optimal);

  auto& dumped_literal_cuts = get_neos8_fractional_literal_cliques_cached();
  if (dumped_literal_cuts.empty()) {
    GTEST_SKIP() << "neos8 produced no candidate literal cliques from LP relaxation";
  }

  const auto& model  = get_neos8_model_cached();
  const int num_vars = model.get_n_variables();

  // Real dumped cuts should not invalidate the no-cut incumbent.
  EXPECT_FALSE(prefix_has_invalid_literal_cut(
    dumped_literal_cuts, dumped_literal_cuts.size(), no_cut_mip.primal, num_vars, kCliqueTestTol));

  // Inject a known-invalid cut and verify bisection isolates it.
  std::vector<int> incumbent_ones;
  incumbent_ones.reserve(2);
  for (int j = 0; j < num_vars && incumbent_ones.size() < 2; ++j) {
    if (!is_binary_var_for_clique_literals(model, j, kCliqueTestTol)) { continue; }
    if (no_cut_mip.primal[j] >= 1.0 - kCliqueTestTol) { incumbent_ones.push_back(j); }
  }
  if (incumbent_ones.size() < 2) {
    GTEST_SKIP() << "Could not find two binary variables fixed to one in neos8 incumbent";
  }

  auto cuts_with_injected_bug = dumped_literal_cuts;
  const size_t injected_index = cuts_with_injected_bug.size();
  cuts_with_injected_bug.push_back({incumbent_ones[0], incumbent_ones[1]});

  auto first_invalid = isolate_first_invalid_literal_cut_by_bisection(
    cuts_with_injected_bug, no_cut_mip.primal, num_vars, kCliqueTestTol);
  ASSERT_TRUE(first_invalid.has_value());
  EXPECT_EQ(first_invalid.value(), injected_index);
}

TEST(cuts, clique_neos8_phase4_lp_infeasibility_binary_search)
{
  auto& dumped_literal_cuts = get_neos8_fractional_literal_cliques_cached();
  if (dumped_literal_cuts.empty()) {
    GTEST_SKIP() << "neos8 produced no candidate literal cliques from LP relaxation";
  }

  const auto& model  = get_neos8_model_cached();
  const int num_vars = model.get_n_variables();

  std::vector<std::vector<int>> cuts_for_lp_search;
  const size_t max_real_cuts = std::min<size_t>(dumped_literal_cuts.size(), 64);
  cuts_for_lp_search.insert(cuts_for_lp_search.end(),
                            dumped_literal_cuts.begin(),
                            dumped_literal_cuts.begin() + max_real_cuts);

  int inject_var = -1;
  for (int j = 0; j < num_vars; ++j) {
    if (is_binary_var_for_clique_literals(model, j, kCliqueTestTol)) {
      inject_var = j;
      break;
    }
  }
  if (inject_var < 0) {
    GTEST_SKIP() << "Could not find a binary variable for LP infeasibility injection";
  }

  const size_t injected_index = cuts_for_lp_search.size();
  cuts_for_lp_search.push_back(
    {inject_var, inject_var, inject_var + num_vars, inject_var + num_vars});

  // Prefix before injected cut should remain LP-feasible.
  const auto status_before_injection =
    solve_lp_with_literal_cut_prefix(cuts_for_lp_search, injected_index, num_vars);
  EXPECT_NE(status_before_injection, pdlp_termination_status_t::PrimalInfeasible);

  // Full prefix should be LP-infeasible due to injected contradictory cut.
  const auto status_with_injection =
    solve_lp_with_literal_cut_prefix(cuts_for_lp_search, cuts_for_lp_search.size(), num_vars);
  EXPECT_EQ(status_with_injection, pdlp_termination_status_t::PrimalInfeasible);

  auto first_infeasible =
    isolate_first_lp_infeasible_literal_cut_by_bisection(cuts_for_lp_search, num_vars);
  ASSERT_TRUE(first_infeasible.has_value());
  EXPECT_EQ(first_infeasible.value(), injected_index);
}

}  // namespace cuopt::linear_programming::test
