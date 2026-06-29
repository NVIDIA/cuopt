/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#include "../linear_programming/utilities/pdlp_test_utilities.cuh"
#include "mip_utils.cuh"

#include <cuopt/mathematical_optimization/io/parser.hpp>
#include <cuopt/mathematical_optimization/pdlp/solver_settings.hpp>
#include <cuopt/mathematical_optimization/pdlp/solver_solution.hpp>
#include <cuopt/mathematical_optimization/solve.hpp>
#include <cuts/cuts.hpp>
#include <mip_heuristics/presolve/conflict_graph/clique_table.cuh>
#include <mip_heuristics/problem/problem.cuh>
#include <utilities/common_utils.hpp>
#include <utilities/copy_helpers.hpp>
#include <utilities/error.hpp>
#include <utilities/inline_lp_test_utils.hpp>
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
#include <span>
#include <sstream>
#include <string>
#include <unordered_set>
#include <vector>

namespace cuopt::mathematical_optimization::test {

namespace {

constexpr double kCliqueTestTol = 1e-6;

// Pairwise binary conflicts forming a triangle.
io::mps_data_model_t<int, double> create_pairwise_triangle_set_packing_problem()
{
  return cuopt::test::parse_inline_lp(R"LP(
Minimize
  obj: -x0 - x1 - x2
Subject To
  c1: x0 + x1 <= 1
  c2: x1 + x2 <= 1
  c3: x0 + x2 <= 1
Binaries
  x0
  x1
  x2
End
)LP");
}

io::mps_data_model_t<int, double> create_pairwise_pentagon_set_packing_problem()
{
  return cuopt::test::parse_inline_lp(R"LP(
Minimize
  obj: -x0 - x1 - x2 - x3 - x4
Subject To
  c1: x0 + x1 <= 1
  c2: x1 + x2 <= 1
  c3: x2 + x3 <= 1
  c4: x3 + x4 <= 1
  c5: x4 + x0 <= 1
Binaries
  x0
  x1
  x2
  x3
  x4
End
)LP");
}

// Same triangle conflicts plus an isolated binary x3 with no conflict rows.
io::mps_data_model_t<int, double> create_pairwise_triangle_with_isolated_variable_problem()
{
  return cuopt::test::parse_inline_lp(R"LP(
Minimize
  obj: -x0 - x1 - x2
Subject To
  c1: x0 + x1 <= 1
  c2: x1 + x2 <= 1
  c3: x0 + x2 <= 1
Binaries
  x0
  x1
  x2
  x3
End
)LP");
}

// x0 + y1 <= 1  (must be ignored for clique graph because y1 is continuous)
// x0 + x2 <= 1  (must generate a conflict edge)
io::mps_data_model_t<int, double> create_binary_continuous_mixed_conflict_problem()
{
  return cuopt::test::parse_inline_lp(R"LP(
Minimize
  obj: 0 x0 + 0 y1 + 0 x2
Subject To
  c1: x0 + y1 <= 1
  c2: x0 + x2 <= 1
Bounds
  0 <= y1 <= 1
Binaries
  x0
  x2
End
)LP");
}

// x0 + x1 <= 1 but x1 has upper bound 0.9999999, so this row should not be
// treated as a binary conflict row.
io::mps_data_model_t<int, double> create_near_binary_bound_conflict_problem()
{
  return cuopt::test::parse_inline_lp(R"LP(
Minimize
  obj: 0 x0 + 0 x1
Subject To
  c1: x0 + x1 <= 1
Bounds
  0 <= x0 <= 1
  0 <= x1 <= 0.9999999
Generals
  x0
  x1
End
)LP");
}

// Creates base clique {x2, x3} and additional clique inducing conflict {x1, x3}.
io::mps_data_model_t<int, double> create_weighted_addtl_conflict_problem()
{
  return cuopt::test::parse_inline_lp(R"LP(
Minimize
  obj: 0 x0 + 0 x1 + 0 x2 + 0 x3
Subject To
  c1: x0 + 2 x1 + 3 x2 + 4 x3 <= 5
Binaries
  x0
  x1
  x2
  x3
End
)LP");
}

mip::clique_table_t<int, double> build_clique_table_for_model_with_min_size(
  const raft::handle_t& handle, const io::mps_data_model_t<int, double>& model, int min_clique_size)
{
  auto op_problem = mps_data_model_to_optimization_problem(&handle, model);
  mip::problem_t<int, double> mip_problem(op_problem);
  simplex::user_problem_t<int, double> host_problem(op_problem.get_handle_ptr());
  mip_problem.get_host_user_problem(host_problem);

  mip::clique_config_t clique_config;
  clique_config.min_clique_size = min_clique_size;
  mip::clique_table_t<int, double> clique_table(2 * host_problem.num_cols,
                                                clique_config.min_clique_size,
                                                clique_config.max_clique_size_for_extension);

  mip_solver_settings_t<int, double> settings;
  cuopt::timer_t timer(std::numeric_limits<double>::infinity());
  mip::build_clique_table(host_problem, clique_table, settings.tolerances, true, true, timer);
  return clique_table;
}

mip::clique_table_t<int, double> build_clique_table_for_model(
  const raft::handle_t& handle, const io::mps_data_model_t<int, double>& model)
{
  return build_clique_table_for_model_with_min_size(handle, model, 1);
}

io::mps_data_model_t<int, double>& get_neos8_model_cached()
{
  static std::once_flag init_flag;
  static std::unique_ptr<io::mps_data_model_t<int, double>> model_ptr;
  std::call_once(init_flag, []() {
    const auto neos8_path = make_path_absolute("mip/neos8.mps");
    auto neos8_model =
      cuopt::mathematical_optimization::io::read_mps<int, double>(neos8_path, false);
    model_ptr = std::make_unique<io::mps_data_model_t<int, double>>(std::move(neos8_model));
  });
  cuopt_assert(model_ptr != nullptr, "Failed to initialize cached neos8 model");
  return *model_ptr;
}

mip::clique_table_t<int, double>& get_neos8_clique_table_cached()
{
  static std::once_flag init_flag;
  static std::unique_ptr<mip::clique_table_t<int, double>> clique_table_ptr;
  std::call_once(init_flag, []() {
    const raft::handle_t handle{};
    auto& neos8_model = get_neos8_model_cached();
    auto clique_table = build_clique_table_for_model(handle, neos8_model);
    clique_table_ptr  = std::make_unique<mip::clique_table_t<int, double>>(std::move(clique_table));
  });
  cuopt_assert(clique_table_ptr != nullptr, "Failed to initialize cached neos8 clique table");
  return *clique_table_ptr;
}

std::vector<std::vector<char>> build_original_adjacency_matrix(
  mip::clique_table_t<int, double>& clique_table, int num_vars)
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
  auto cliques = mip::find_maximal_cliques_for_test(
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

std::string format_phase2_panic_dump(const io::mps_data_model_t<int, double>& problem,
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
  settings.zero_half_cuts             = 0;
  settings.max_cut_passes             = 10;
  settings.mixed_integer_gomory_cuts  = 0;
  settings.knapsack_cuts              = 0;
  settings.mir_cuts                   = 0;
  settings.strong_chvatal_gomory_cuts = 0;
}

void disable_non_zero_half_cuts(mip_solver_settings_t<int, double>& settings)
{
  settings.clique_cuts                = 1;
  settings.zero_half_cuts             = 1;
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
  settings.zero_half_cuts             = 0;
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

bool is_binary_var_for_clique_literals(const io::mps_data_model_t<int, double>& problem,
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
  const io::mps_data_model_t<int, double>& problem,
  mip::clique_table_t<int, double>& clique_table,
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

  auto cliques_local = mip::find_maximal_cliques_for_test(
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

io::mps_data_model_t<int, double>& get_neos8_lp_relaxation_model_cached()
{
  static std::once_flag init_flag;
  static std::unique_ptr<io::mps_data_model_t<int, double>> model_ptr;
  std::call_once(init_flag, []() {
    auto lp_relaxation = get_neos8_model_cached();
    std::vector<char> all_continuous(lp_relaxation.get_n_variables(), 'C');
    lp_relaxation.set_variable_types(all_continuous);
    model_ptr = std::make_unique<io::mps_data_model_t<int, double>>(std::move(lp_relaxation));
  });
  cuopt_assert(model_ptr != nullptr, "Failed to initialize cached neos8 LP relaxation model");
  return *model_ptr;
}

io::mps_data_model_t<int, double> append_literal_cut_prefix_to_lp_model(
  const io::mps_data_model_t<int, double>& base_lp_model,
  const std::vector<std::vector<int>>& dumped_cuts,
  size_t prefix_end_exclusive,
  int num_vars)
{
  auto model_with_cuts = base_lp_model;
  if (prefix_end_exclusive == 0) { return model_with_cuts; }

  std::vector<double> matrix_values  = base_lp_model.get_constraint_matrix_values();
  std::vector<int> matrix_indices    = base_lp_model.get_constraint_matrix_indices();
  std::vector<int> matrix_offsets    = base_lp_model.get_constraint_matrix_offsets();
  std::vector<double> constraint_rhs = base_lp_model.get_constraint_bounds();
  std::vector<double> constraint_lbs = base_lp_model.get_constraint_lower_bounds();
  std::vector<double> constraint_ubs = base_lp_model.get_constraint_upper_bounds();
  std::vector<char> row_types        = base_lp_model.get_row_types();
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
    // Keep RHS / ROWS metadata aligned with appended bounds.
    // Literal cut is lhs >= rhs, so row type is 'G'.
    if (!constraint_rhs.empty()) {
      constraint_rhs.push_back(static_cast<double>(num_complements - 1));
    }
    constraint_lbs.push_back(static_cast<double>(num_complements - 1));
    constraint_ubs.push_back(std::numeric_limits<double>::infinity());
    if (!row_types.empty()) { row_types.push_back('G'); }
    row_names.push_back("literal_cut_" + std::to_string(cut_idx));
  }

  model_with_cuts.set_csr_constraint_matrix(matrix_values, matrix_indices, matrix_offsets);
  if (!constraint_rhs.empty()) { model_with_cuts.set_constraint_bounds(constraint_rhs); }
  model_with_cuts.set_constraint_lower_bounds(constraint_lbs);
  model_with_cuts.set_constraint_upper_bounds(constraint_ubs);
  if (!row_types.empty()) { model_with_cuts.set_row_types(row_types); }
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

io::mps_data_model_t<int, double> create_cuts_problem_1()
{
  return cuopt::test::parse_inline_lp(R"LP(
Minimize
  obj: -7 x1 - 2 x2
Subject To
  c1: -x1 + 2 x2 <= 4
  c2: 5 x1 + x2 <= 20
  c3: -2 x1 - 2 x2 <= -7
Bounds
  0 <= x1 <= 10
  0 <= x2 <= 10
Generals
  x1
  x2
End
)LP");
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

io::mps_data_model_t<int, double> create_cuts_problem_2()
{
  return cuopt::test::parse_inline_lp(R"LP(
Minimize
  obj: -86 y1 - 4 y2 - 40 y3
Subject To
  c1: 774 y1 + 76 y2 + 42 y3 <= 875
  c2: 67 y1 + 27 y2 + 53 y3 <= 875
Binaries
  y1
  y2
  y3
End
)LP");
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
  simplex::simplex_solver_settings_t<int, double> settings;
  mip::cut_pool_t<int, double> cut_pool(4, settings);
  mip::inequality_t<int, double> cut1;
  cut1.push_back(0, 1.0);
  cut1.push_back(1, 2.0);
  cut1.rhs = 1.0;
  cut_pool.add_cut(mip::cut_type_t::MIXED_INTEGER_GOMORY, cut1);
  mip::inequality_t<int, double> cut2;
  cut2.push_back(0, 2.0);
  cut2.push_back(1, 4.0);
  cut2.rhs = 2.0;
  cut_pool.add_cut(mip::cut_type_t::MIXED_INTEGER_GOMORY, cut2);
  mip::inequality_t<int, double> cut3;
  cut3.push_back(0, 0.1);
  cut3.push_back(2, 0.2);
  cut3.rhs = 1.0;
  cut_pool.add_cut(mip::cut_type_t::MIXED_INTEGER_GOMORY, cut3);
  mip::inequality_t<int, double> cut4;
  cut4.push_back(0, 0.2);
  cut4.push_back(2, 0.4);
  cut4.rhs = 1.0;
  cut_pool.add_cut(mip::cut_type_t::MIXED_INTEGER_GOMORY, cut4);
  mip::inequality_t<int, double> cut5;
  cut5.push_back(1, 10.0);
  cut5.push_back(3, 20.0);
  cut5.rhs = 0.1;
  cut_pool.add_cut(mip::cut_type_t::MIXED_INTEGER_GOMORY, cut5);
  mip::inequality_t<int, double> cut6;
  cut6.push_back(1, 20.0);
  cut6.push_back(3, 40.0);
  cut6.rhs = 0.2;
  cut_pool.add_cut(mip::cut_type_t::MIXED_INTEGER_GOMORY, cut6);
  mip::inequality_t<int, double> cut7;
  cut7.push_back(0, 1.0);
  cut7.push_back(1, 1.0);
  cut7.push_back(2, 1.0);
  cut7.push_back(3, 1.0);
  cut7.rhs = 1.0;
  cut_pool.add_cut(mip::cut_type_t::MIXED_INTEGER_GOMORY, cut7);
  mip::inequality_t<int, double> cut8;
  cut8.push_back(1, 3.0);
  cut8.rhs = 7.0;
  cut_pool.add_cut(mip::cut_type_t::MIXED_INTEGER_GOMORY, cut8);

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

  // Conflicts must remain materialized in small_clique_adj after removals.
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
  const int n_vertices = static_cast<int>(clique_table.var_clique_first.n_keys());
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

// ---- Zero-half cut tests --------------------------------------------------

namespace {

std::vector<std::vector<int>> canonicalize_cycles(std::vector<std::vector<int>> cycles)
{
  for (auto& cycle : cycles) {
    if (cycle.empty()) { continue; }
    auto min_it = std::min_element(cycle.begin(), cycle.end());
    std::rotate(cycle.begin(), min_it, cycle.end());
    if (cycle.size() >= 3 && cycle[1] > cycle.back()) {
      std::reverse(cycle.begin() + 1, cycle.end());
    }
  }
  std::sort(cycles.begin(), cycles.end());
  cycles.erase(std::unique(cycles.begin(), cycles.end()), cycles.end());
  return cycles;
}

}  // namespace

TEST(cuts, zero_half_unit_separator_simple_pentagon)
{
  // 5-cycle: 0-1-2-3-4-0. All vertices fractional at 0.5.
  std::vector<std::vector<int>> adj = {
    {1, 4},
    {0, 2},
    {1, 3},
    {2, 4},
    {3, 0},
  };
  std::vector<double> x_values(5, 0.5);
  auto cycles = mip::find_violated_odd_cycles_for_test(
    adj, x_values, 1e-6, std::numeric_limits<double>::infinity());
  ASSERT_FALSE(cycles.empty());
  cycles = canonicalize_cycles(std::move(cycles));
  std::vector<int> expected{0, 1, 2, 3, 4};
  bool found = false;
  for (const auto& cycle : cycles) {
    if (cycle.size() == 5) {
      auto sorted = cycle;
      std::sort(sorted.begin(), sorted.end());
      if (sorted == expected) {
        found = true;
        break;
      }
    }
  }
  EXPECT_TRUE(found);
}

TEST(cuts, zero_half_unit_separator_no_cycle_for_4_cycle)
{
  // Even cycle: 0-1-2-3-0
  std::vector<std::vector<int>> adj = {
    {1, 3},
    {0, 2},
    {1, 3},
    {2, 0},
  };
  std::vector<double> x_values(4, 0.5);
  auto cycles = mip::find_violated_odd_cycles_for_test(
    adj, x_values, 1e-6, std::numeric_limits<double>::infinity());
  EXPECT_TRUE(cycles.empty());
}

TEST(cuts, zero_half_unit_separator_skips_triangle)
{
  // Triangle 0-1-2-0 ; size-3 cycles must be left to the clique separator.
  std::vector<std::vector<int>> adj = {
    {1, 2},
    {0, 2},
    {0, 1},
  };
  std::vector<double> x_values(3, 0.5);
  auto cycles = mip::find_violated_odd_cycles_for_test(
    adj, x_values, 1e-6, std::numeric_limits<double>::infinity());
  for (const auto& cycle : cycles) {
    EXPECT_GE(cycle.size(), 5u);
  }
}

TEST(cuts, zero_half_unit_separator_no_cycle_when_integer_solution)
{
  // 5-cycle but x_values are integer feasible: (1, 0, 1, 0, 0) -- no violation.
  std::vector<std::vector<int>> adj = {
    {1, 4},
    {0, 2},
    {1, 3},
    {2, 4},
    {3, 0},
  };
  std::vector<double> x_values = {1.0, 0.0, 1.0, 0.0, 0.0};
  // x_v interpreted as conflict-graph vertex weight (here just x_j directly).
  auto cycles = mip::find_violated_odd_cycles_for_test(
    adj, x_values, 1e-6, std::numeric_limits<double>::infinity());
  EXPECT_TRUE(cycles.empty());
}

TEST(cuts, zero_half_unit_separator_disjoint_pentagons)
{
  // Two disjoint 5-cycles share no vertices: {0..4} and {5..9}.
  std::vector<std::vector<int>> adj = {
    {1, 4},
    {0, 2},
    {1, 3},
    {2, 4},
    {3, 0},
    {6, 9},
    {5, 7},
    {6, 8},
    {7, 9},
    {8, 5},
  };
  std::vector<double> x_values(10, 0.5);
  auto cycles = mip::find_violated_odd_cycles_for_test(
    adj, x_values, 1e-6, std::numeric_limits<double>::infinity());
  ASSERT_GE(cycles.size(), 2u);
  cycles           = canonicalize_cycles(std::move(cycles));
  bool found_left  = false;
  bool found_right = false;
  for (const auto& cycle : cycles) {
    if (cycle.size() != 5) { continue; }
    auto sorted = cycle;
    std::sort(sorted.begin(), sorted.end());
    if (sorted == std::vector<int>{0, 1, 2, 3, 4}) { found_left = true; }
    if (sorted == std::vector<int>{5, 6, 7, 8, 9}) { found_right = true; }
  }
  EXPECT_TRUE(found_left);
  EXPECT_TRUE(found_right);
}

TEST(cuts, zero_half_unit_separator_overlapping_pentagons)
{
  std::vector<std::vector<int>> adj = {
    {1, 4, 5, 8},
    {0, 2},
    {1, 3},
    {2, 4},
    {3, 0},
    {0, 6},
    {5, 7},
    {6, 8},
    {7, 0},
  };
  std::vector<double> x_values(9, 0.5);
  auto cycles = mip::find_violated_odd_cycles_for_test(
    adj, x_values, 1e-6, std::numeric_limits<double>::infinity());
  cycles = canonicalize_cycles(std::move(cycles));

  EXPECT_NE(std::find(cycles.begin(), cycles.end(), std::vector<int>{0, 1, 2, 3, 4}), cycles.end());
  EXPECT_NE(std::find(cycles.begin(), cycles.end(), std::vector<int>{0, 5, 6, 7, 8}), cycles.end());
}

TEST(cuts, zero_half_end_to_end_pentagon_tightens_lp_relaxation)
{
  const raft::handle_t handle{};
  auto mip_problem = create_pairwise_pentagon_set_packing_problem();

  // First solve the LP relaxation (no cuts) to confirm the baseline value 2.5.
  auto lp_relaxation = mip_problem;
  std::vector<char> all_continuous(lp_relaxation.get_n_variables(), 'C');
  lp_relaxation.set_variable_types(all_continuous);

  pdlp_solver_settings_t<int, double> lp_settings{};
  lp_settings.time_limit = 10.0;
  lp_settings.presolver  = presolver_t::None;
  lp_settings.set_optimality_tolerance(1e-8);
  auto lp_solution = solve_lp(&handle, lp_relaxation, lp_settings);
  ASSERT_EQ(lp_solution.get_termination_status(), pdlp_termination_status_t::Optimal);
  const double lp_obj_no_cuts = lp_solution.get_objective_value();
  EXPECT_NEAR(lp_obj_no_cuts, -2.5, kCliqueTestTol);

  // Optimal IP value is 2 (independent set of size 2), so the LP gap is 0.5.
  mip_solver_settings_t<int, double> settings;
  settings.time_limit = 10.0;
  settings.presolver  = presolver_t::None;
  disable_non_zero_half_cuts(settings);

  auto mip_solution = solve_mip(&handle, mip_problem, settings);
  ASSERT_EQ(mip_solution.get_termination_status(), mip_termination_status_t::Optimal);
  EXPECT_NEAR(mip_solution.get_objective_value(), -2.0, kCliqueTestTol);
}

TEST(cuts, zero_half_unit_separator_seven_cycle_violated_below_half)
{
  // 7-cycle: 0-1-2-3-4-5-6-0, all weights 0.4. Each edge weight = (1-0.4-0.4)/2 = 0.1
  // total path weight from j1 to j2 of length 7 = 0.7 — not below 0.5, so no cut.
  // Make weights slightly higher: 0.45 → edge weight = 0.05, total = 7*0.05 = 0.35 < 0.5.
  std::vector<std::vector<int>> adj = {
    {1, 6},
    {0, 2},
    {1, 3},
    {2, 4},
    {3, 5},
    {4, 6},
    {5, 0},
  };
  std::vector<double> x_values(7, 0.45);
  auto cycles = mip::find_violated_odd_cycles_for_test(
    adj, x_values, 1e-6, std::numeric_limits<double>::infinity());
  ASSERT_FALSE(cycles.empty());
  bool found_seven = false;
  for (const auto& cycle : cycles) {
    if (cycle.size() == 7) {
      found_seven = true;
      break;
    }
  }
  EXPECT_TRUE(found_seven);
}

// Minimal 0-1 single-node-flow relaxation for the flow-cover separator.
//
//   y0 + y1 - y2 <= 4
//   0 <= y0 <= 3*x0, 0 <= y1 <= 6*x1, 0 <= y2 <= 3*x2
//
// The fractional point x* = (1, 2/3, 1), y* = (3, 4, 3) satisfies the relaxation
// but violates the generated c-MIR flow-cover cut. This is a reduced version of a
// standard flow-cover example; the test checks validity instead of exact coefficients
// because the approximate single-node-flow selection may choose a different valid cut.
// Index layout (x0,x1,x2,y0,y1,y2 → 0..5) is load-bearing — downstream test
// helpers index into the primal via point[j] for binaries and point[3+j] for
// flows. Keep the variable order matching that layout.
io::mps_data_model_t<int, double> create_small_single_node_flow_problem()
{
  return cuopt::test::parse_inline_lp(R"LP(
Minimize
  obj: 0 x0 + 0 x1 + 0 x2 + 0 y0 + 0 y1 + 0 y2
Subject To
  c1: y0 + y1 - y2 <= 4
  c2: -3 x0 + y0 <= 0
  c3: -6 x1 + y1 <= 0
  c4: -3 x2 + y2 <= 0
Bounds
  0 <= y0 <= 3
  0 <= y1 <= 6
  0 <= y2 <= 3
Binaries
  x0
  x1
  x2
End
)LP");
}

struct flow_cover_test_problem_t {
  raft::handle_t handle;
  simplex::simplex_solver_settings_t<int, double> settings;
  simplex::lp_problem_t<int, double> lp;
  simplex::csr_matrix_t<int, double> Arow;
  std::vector<int> new_slacks;
  std::vector<simplex::variable_type_t> var_types;

  flow_cover_test_problem_t() : handle(), settings(), lp(&handle, 1, 1, 1), Arow(0, 0, 0) {}
};

flow_cover_test_problem_t build_flow_cover_test_problem(
  const io::mps_data_model_t<int, double>& model)
{
  flow_cover_test_problem_t test_problem;
  auto op_problem = mps_data_model_to_optimization_problem(&test_problem.handle, model);
  mip::problem_t<int, double> mip_problem(op_problem);
  simplex::user_problem_t<int, double> host_problem(op_problem.get_handle_ptr());
  mip_problem.get_host_user_problem(host_problem);

  simplex::dualize_info_t<int, double> dualize_info;
  simplex::convert_user_problem(
    host_problem, test_problem.settings, test_problem.lp, test_problem.new_slacks, dualize_info);
  test_problem.var_types = host_problem.var_types;
  if (test_problem.lp.num_cols > static_cast<int>(test_problem.var_types.size())) {
    test_problem.var_types.resize(test_problem.lp.num_cols, simplex::variable_type_t::CONTINUOUS);
  }
  test_problem.lp.A.to_compressed_row(test_problem.Arow);
  return test_problem;
}

std::vector<double> single_node_flow_fractional_solution(int num_cols)
{
  std::vector<double> xstar(num_cols, 0.0);
  xstar[0] = 1.0;
  xstar[1] = 2.0 / 3.0;
  xstar[2] = 1.0;
  xstar[3] = 3.0;
  xstar[4] = 4.0;
  xstar[5] = 3.0;
  return xstar;
}

bool single_node_flow_y_feasible(const std::vector<double>& y)
{
  const double activity = y[0] + y[1] - y[2];
  return activity <= 4.0 + 1e-8;
}

void expect_single_node_flow_cut_valid_at_point(const mip::inequality_t<int, double>& cut,
                                                const std::vector<double>& point,
                                                const std::string& label)
{
  EXPECT_GE(cut.vector.dot(point), cut.rhs - 1e-7) << label;
}

void expect_single_node_flow_cut_valid_at_extreme_points(const mip::inequality_t<int, double>& cut,
                                                         int num_cols)
{
  const std::vector<double> capacities = {3.0, 6.0, 3.0};
  const std::vector<double> flow_signs = {1.0, 1.0, -1.0};
  int checked_points                   = 0;

  for (int x_mask = 0; x_mask < 8; x_mask++) {
    std::vector<double> y_upper(3, 0.0);
    for (int j = 0; j < 3; j++) {
      if (((x_mask >> j) & 1) != 0) { y_upper[j] = capacities[j]; }
    }

    for (int y_mask = 0; y_mask < 8; y_mask++) {
      std::vector<double> y(3, 0.0);
      for (int j = 0; j < 3; j++) {
        if (((y_mask >> j) & 1) != 0) { y[j] = y_upper[j]; }
      }
      if (!single_node_flow_y_feasible(y)) { continue; }

      std::vector<double> point(num_cols, 0.0);
      for (int j = 0; j < 3; j++) {
        point[j]     = ((x_mask >> j) & 1) != 0 ? 1.0 : 0.0;
        point[3 + j] = y[j];
      }
      expect_single_node_flow_cut_valid_at_point(
        cut,
        point,
        "box vertex x_mask=" + std::to_string(x_mask) + " y_mask=" + std::to_string(y_mask));
      checked_points++;
    }

    for (int free_j = 0; free_j < 3; free_j++) {
      for (int bound_mask = 0; bound_mask < 4; bound_mask++) {
        std::vector<double> y(3, 0.0);
        int bit = 0;
        for (int j = 0; j < 3; j++) {
          if (j == free_j) { continue; }
          if (((bound_mask >> bit) & 1) != 0) { y[j] = y_upper[j]; }
          bit++;
        }

        double fixed_activity = 0.0;
        for (int j = 0; j < 3; j++) {
          if (j != free_j) { fixed_activity += flow_signs[j] * y[j]; }
        }

        const double y_free = (4.0 - fixed_activity) / flow_signs[free_j];
        if (y_free < -1e-8 || y_free > y_upper[free_j] + 1e-8) { continue; }
        y[free_j] = std::max(0.0, std::min(y_upper[free_j], y_free));
        if (!single_node_flow_y_feasible(y)) { continue; }

        std::vector<double> point(num_cols, 0.0);
        for (int j = 0; j < 3; j++) {
          point[j]     = ((x_mask >> j) & 1) != 0 ? 1.0 : 0.0;
          point[3 + j] = y[j];
        }
        expect_single_node_flow_cut_valid_at_point(
          cut,
          point,
          "flow-tight vertex x_mask=" + std::to_string(x_mask) +
            " free_j=" + std::to_string(free_j) + " bound_mask=" + std::to_string(bound_mask));
        checked_points++;
      }
    }
  }

  EXPECT_GT(checked_points, 0);
}

TEST(cuts, flow_cover_generates_valid_single_node_flow_cut)
{
  auto test_problem = build_flow_cover_test_problem(create_small_single_node_flow_problem());
  const std::vector<double> xstar = single_node_flow_fractional_solution(test_problem.lp.num_cols);

  mip::flow_cover_generation_t<int, double> generator(
    test_problem.lp, test_problem.settings, test_problem.Arow, test_problem.new_slacks);
  mip::variable_bounds_t<int, double> variable_bounds(test_problem.lp,
                                                      test_problem.settings,
                                                      test_problem.var_types,
                                                      test_problem.Arow,
                                                      test_problem.new_slacks);
  ASSERT_GT(generator.num_constraints(), 0);

  int generated_cuts = 0;
  for (const auto& flow_cover_row : generator.get_constraints()) {
    mip::inequality_t<int, double> cut(test_problem.lp.num_cols);
    const int status = generator.generate_cut(test_problem.lp,
                                              test_problem.settings,
                                              test_problem.Arow,
                                              variable_bounds,
                                              test_problem.var_types,
                                              xstar,
                                              flow_cover_row,
                                              cut);
    if (status != 0) { continue; }

    EXPECT_LT(cut.vector.dot(xstar), cut.rhs - 1e-6)
      << "row=" << flow_cover_row.row << " reverse=" << flow_cover_row.reverse;
    expect_single_node_flow_cut_valid_at_extreme_points(cut, test_problem.lp.num_cols);
    generated_cuts++;
  }

  EXPECT_GT(generated_cuts, 0);
}

}  // namespace cuopt::mathematical_optimization::test
