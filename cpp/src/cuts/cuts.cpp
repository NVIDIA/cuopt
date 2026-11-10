/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#include <cuts/cuts.hpp>
#include <cuts/cuts_debug.hpp>
#include <cuts/rational.hpp>
#include <dual_simplex/basis_solves.hpp>
#include <math_optimization/tic_toc.hpp>
#include <presolve/conflict_graph/clique_table.cuh>
#include <utilities/logger.hpp>
#include <utilities/macros.cuh>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <iterator>
#include <numeric>
#include <unordered_set>
#include <utility>
#include <vector>

namespace cuopt::mathematical_optimization::mip {

using simplex::basis_update_mpf_t;
using simplex::form_b;
using simplex::lp_problem_t;
using simplex::lp_solution_t;
using simplex::simplex_solver_settings_t;
using simplex::variable_status_t;
using simplex::variable_type_t;

int64_t gcd(const std::vector<int64_t>& integers);

int64_t lcm(const std::vector<int64_t>& integers);

namespace {

// Total probing-edge budget from the byte cap and the remaining work headroom
// (max_work_estimate - work_estimate).
template <typename i_t, typename f_t>
size_t max_probing_edge_budget(size_t num_vertices, f_t work_headroom)
{
  constexpr size_t max_probing_conflict_bytes = size_t{8} << 30;
  const size_t probing_degree_bytes           = num_vertices * sizeof(size_t);
  const size_t probing_edge_bytes =
    sizeof(std::pair<i_t, i_t>) + sizeof(std::pair<f_t, i_t>) + 2 * sizeof(i_t);
  const size_t max_edges_by_memory =
    probing_degree_bytes < max_probing_conflict_bytes
      ? (max_probing_conflict_bytes - probing_degree_bytes) / probing_edge_bytes
      : 0;
  const size_t max_edges_by_work = (size_t)std::max<f_t>(0.0, work_headroom / 4.0);
  return std::min(max_edges_by_memory, max_edges_by_work);
}

}  // namespace

template <typename i_t, typename f_t>
void cut_generation_t<i_t, f_t>::prepare_fractional_sub_conflict_graph(
  const simplex_solver_settings_t<i_t, f_t>& settings,
  const std::vector<f_t>& xstar,
  f_t start_time)
{
  sub_cg_.clear();

  if (settings.clique_cuts == 0 && settings.zero_half_cuts == 0) { return; }
  if (toc(start_time) >= settings.time_limit) { return; }

  // The clique table is produced by a background OpenMP task (spawned in
  // branch_and_bound). Its base cliques are published early, but the extension
  // phase keeps mutating the same table. Now that cut generation needs the
  // table, signal the task to stop extending and block until it finishes, so the
  // table is no longer mutated while we read it below. The wait must happen
  // unconditionally (even when the pointer is already non-null), otherwise we
  // would race the still-running extension.
  if (signal_extend_) {
    signal_extend_->store(true, std::memory_order_release);
#pragma omp taskwait depend(in : *signal_extend_)
  }

  if (clique_table_ == nullptr) { return; }
  const bool has_probing_conflicts =
    !probing_implied_bound_.zero_variables.empty() || !probing_implied_bound_.one_variables.empty();
  if (clique_table_->empty() && !has_probing_conflicts) { return; }

  const i_t num_vars = user_problem_.num_cols;
  cuopt_assert(clique_table_->n_variables == num_vars,
               "prepare_fractional_sub_conflict_graph clique table variable count mismatch");
  cuopt_assert(static_cast<size_t>(num_vars) <= xstar.size(),
               "prepare_fractional_sub_conflict_graph xstar size mismatch");
  cuopt_assert(user_problem_.var_types.size() == static_cast<size_t>(num_vars),
               "prepare_fractional_sub_conflict_graph user problem var_types size mismatch");
  if (has_probing_conflicts) {
    cuopt_assert(probing_implied_bound_.zero_offsets.size() == (size_t)num_vars + 1,
                 "Probing zero-offset column count mismatch");
    cuopt_assert(probing_implied_bound_.one_offsets.size() == (size_t)num_vars + 1,
                 "Probing one-offset column count mismatch");
  }

  const f_t bound_tol         = settings.primal_tol;
  f_t work_estimate           = 0.0;
  const f_t max_work_estimate = 1e7;

  sub_cg_.num_vars = num_vars;
  sub_cg_.vertices.reserve(static_cast<size_t>(num_vars) * 2);
  sub_cg_.weights.reserve(static_cast<size_t>(num_vars) * 2);

  for (i_t j = 0; j < num_vars; ++j) {
    if (user_problem_.var_types[j] == variable_type_t::CONTINUOUS) { continue; }
    const f_t lower_bound = user_problem_.lower[j];
    const f_t upper_bound = user_problem_.upper[j];
    if (lower_bound < -bound_tol || upper_bound > 1 + bound_tol) { continue; }
    const f_t xj = xstar[j];
    if (std::abs(xj - std::round(xj)) <= settings.integer_tol) { continue; }
    sub_cg_.vertices.push_back(j);
    sub_cg_.weights.push_back(xj);
    sub_cg_.vertices.push_back(j + num_vars);
    sub_cg_.weights.push_back(static_cast<f_t>(1.0) - xj);
  }
  work_estimate +=
    4.0 * static_cast<f_t>(num_vars) + 2.0 * static_cast<f_t>(sub_cg_.vertices.size());
  if (work_estimate > max_work_estimate) {
    sub_cg_.clear();
    return;
  }

  if (sub_cg_.vertices.empty()) {
    // No fractional binaries — both separators have nothing to do, but the
    // build itself succeeded. Mark ready so callers don't keep retrying.
    sub_cg_.ready = true;
    return;
  }

  sub_cg_.vertex_to_local.assign(static_cast<size_t>(2 * num_vars), -1);
  sub_cg_.in_subgraph.assign(static_cast<size_t>(2 * num_vars), 0);
  for (size_t idx = 0; idx < sub_cg_.vertices.size(); ++idx) {
    if (toc(start_time) >= settings.time_limit) {
      sub_cg_.clear();
      return;
    }
    const i_t v_idx                = sub_cg_.vertices[idx];
    sub_cg_.vertex_to_local[v_idx] = static_cast<i_t>(idx);
    sub_cg_.in_subgraph[v_idx]     = 1;
  }
  work_estimate += 3.0 * static_cast<f_t>(sub_cg_.vertices.size());
  if (work_estimate > max_work_estimate) {
    sub_cg_.clear();
    return;
  }

  sub_cg_.adj_local.assign(sub_cg_.vertices.size(), {});
  size_t total_adj_entries = 0;
  size_t kept_adj_entries  = 0;
  for (size_t idx = 0; idx < sub_cg_.vertices.size(); ++idx) {
    if (toc(start_time) >= settings.time_limit) {
      sub_cg_.clear();
      return;
    }
    const i_t v_idx                 = sub_cg_.vertices[idx];
    std::unordered_set<i_t> adj_set = clique_table_->get_adj_set_of_var(v_idx);
    total_adj_entries += adj_set.size();
    std::vector<i_t>& adj = sub_cg_.adj_local[idx];
    adj.reserve(adj_set.size());
    for (const i_t neighbor : adj_set) {
      cuopt_assert(neighbor >= 0 && neighbor < 2 * num_vars,
                   "prepare_fractional_sub_conflict_graph neighbor out of range");
      if (!sub_cg_.in_subgraph[neighbor]) { continue; }
      const i_t local_neighbor = sub_cg_.vertex_to_local[neighbor];
      cuopt_assert(local_neighbor >= 0,
                   "prepare_fractional_sub_conflict_graph local_neighbor out of range");
      adj.push_back(local_neighbor);
    }
    kept_adj_entries += adj.size();
  }
  work_estimate += static_cast<f_t>(sub_cg_.vertices.size()) + static_cast<f_t>(total_adj_entries) +
                   2.0 * static_cast<f_t>(kept_adj_entries);
  if (work_estimate > max_work_estimate) {
    sub_cg_.clear();
    return;
  }

  size_t probing_entries_scanned = 0;
  size_t probing_edges_kept      = 0;
  if (has_probing_conflicts) {
    using candidate_t = std::pair<f_t, i_t>;
    // Min-heap on score: front() is the worst kept candidate, so a better one replaces it.
    auto better_candidate = [](const candidate_t& a, const candidate_t& b) {
      return a.first > b.first || (a.first == b.first && a.second < b.second);
    };

    const size_t max_probing_edges =
      max_probing_edge_budget<i_t, f_t>(sub_cg_.vertices.size(), max_work_estimate - work_estimate);
    const size_t max_edges_per_literal =
      max_probing_edges / std::max<size_t>(1, sub_cg_.vertices.size());

    size_t raw_edge_bound = 0;
    for (i_t j = 0; j < num_vars; ++j) {
      if (!sub_cg_.in_subgraph[j]) { continue; }
      raw_edge_bound +=
        probing_implied_bound_.zero_offsets[j + 1] - probing_implied_bound_.zero_offsets[j];
      raw_edge_bound +=
        probing_implied_bound_.one_offsets[j + 1] - probing_implied_bound_.one_offsets[j];
    }

    std::vector<std::pair<i_t, i_t>> probing_edges;
    probing_edges.reserve(std::min(max_probing_edges, raw_edge_bound));

    for (i_t j = 0; j < num_vars; ++j) {
      if (!sub_cg_.in_subgraph[j]) { continue; }
      for (bool one : {false, true}) {
        const auto& offsets =
          one ? probing_implied_bound_.one_offsets : probing_implied_bound_.zero_offsets;
        const auto& variables =
          one ? probing_implied_bound_.one_variables : probing_implied_bound_.zero_variables;
        const auto& lower_bounds =
          one ? probing_implied_bound_.one_lower_bound : probing_implied_bound_.zero_lower_bound;
        const auto& upper_bounds =
          one ? probing_implied_bound_.one_upper_bound : probing_implied_bound_.zero_upper_bound;
        const i_t source       = one ? j : j + num_vars;
        const i_t source_local = sub_cg_.vertex_to_local[source];
        const i_t begin        = offsets[j];
        const i_t end          = offsets[j + 1];
        cuopt_assert(source_local >= 0, "Probing conflict source must be fractional");

        std::vector<candidate_t> candidates;
        candidates.reserve(std::min(max_edges_per_literal, (size_t)(end - begin)));
        for (i_t p = begin; p < end; ++p) {
          probing_entries_scanned++;
          const i_t target_var = variables[p];
          cuopt_assert(target_var >= 0 && target_var < num_vars,
                       "Probing conflict target out of range");
          if (target_var == j) { continue; }
          i_t target = -1;
          if (upper_bounds[p] < 1.0 - settings.integer_tol) {
            target = target_var;
          } else if (lower_bounds[p] > settings.integer_tol) {
            target = target_var + num_vars;
          } else {
            continue;
          }
          if (!sub_cg_.in_subgraph[target]) { continue; }
          const i_t target_local = sub_cg_.vertex_to_local[target];
          candidate_t candidate{sub_cg_.weights[source_local] + sub_cg_.weights[target_local],
                                target_local};
          if (candidates.size() < max_edges_per_literal) {
            candidates.push_back(candidate);
            if (candidates.size() == max_edges_per_literal) {
              std::make_heap(candidates.begin(), candidates.end(), better_candidate);
            }
            continue;
          }
          if (max_edges_per_literal == 0) { continue; }
          if (better_candidate(candidate, candidates.front())) {
            std::pop_heap(candidates.begin(), candidates.end(), better_candidate);
            candidates.back() = candidate;
            std::push_heap(candidates.begin(), candidates.end(), better_candidate);
          }
        }
        for (const auto& candidate : candidates) {
          probing_edges.emplace_back(std::min(source_local, candidate.second),
                                     std::max(source_local, candidate.second));
        }
      }
      if (toc(start_time) >= settings.time_limit) {
        sub_cg_.clear();
        return;
      }
    }
    cuopt_assert(probing_edges.size() <= max_probing_edges,
                 "Selected probing conflicts exceed memory budget");

    std::sort(probing_edges.begin(), probing_edges.end());
    probing_edges.erase(std::unique(probing_edges.begin(), probing_edges.end()),
                        probing_edges.end());
    probing_edges_kept = probing_edges.size();

    std::vector<size_t> probing_degrees(sub_cg_.vertices.size(), 0);
    for (const auto& [a, b] : probing_edges) {
      probing_degrees[a]++;
      probing_degrees[b]++;
    }
    for (size_t i = 0; i < sub_cg_.adj_local.size(); ++i) {
      sub_cg_.adj_local[i].reserve(sub_cg_.adj_local[i].size() + probing_degrees[i]);
    }
    for (const auto& [a, b] : probing_edges) {
      sub_cg_.adj_local[a].push_back(b);
      sub_cg_.adj_local[b].push_back(a);
    }
  }

  kept_adj_entries = 0;
  for (auto& adj : sub_cg_.adj_local) {
    std::sort(adj.begin(), adj.end());
    adj.erase(std::unique(adj.begin(), adj.end()), adj.end());
    kept_adj_entries += adj.size();
  }
  work_estimate = 4.0 * static_cast<f_t>(num_vars) +
                  6.0 * static_cast<f_t>(sub_cg_.vertices.size()) +
                  static_cast<f_t>(total_adj_entries) + 2.0 * static_cast<f_t>(kept_adj_entries);
  if (work_estimate > max_work_estimate) {
    sub_cg_.clear();
    return;
  }

#ifdef ASSERT_MODE
  for (const auto& adj : sub_cg_.adj_local) {
    cuopt_assert(std::adjacent_find(adj.begin(), adj.end()) == adj.end(),
                 "Duplicate neighbor in fractional conflict subgraph");
  }
#endif

  sub_cg_.ready = true;
  CLIQUE_CUTS_DEBUG(
    "prepare_fractional_sub_conflict_graph ready vertices=%lld raw_adj=%lld kept_adj=%lld "
    "probing_scanned=%lld probing_kept=%lld",
    static_cast<long long>(sub_cg_.vertices.size()),
    static_cast<long long>(total_adj_entries),
    static_cast<long long>(kept_adj_entries),
    static_cast<long long>(probing_entries_scanned),
    static_cast<long long>(probing_edges_kept));
  ZERO_HALF_DEBUG(
    "prepare_fractional_sub_conflict_graph ready vertices=%lld raw_adj=%lld kept_adj=%lld "
    "probing_scanned=%lld probing_kept=%lld",
    static_cast<long long>(sub_cg_.vertices.size()),
    static_cast<long long>(total_adj_entries),
    static_cast<long long>(kept_adj_entries),
    static_cast<long long>(probing_entries_scanned),
    static_cast<long long>(probing_edges_kept));
}

template <typename i_t, typename f_t>
bool cut_generation_t<i_t, f_t>::generate_cuts(const lp_problem_t<i_t, f_t>& lp,
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
                                               f_t start_time)
{
  // Generate Gomory and CG Cuts
  if (settings.mixed_integer_gomory_cuts != 0 || settings.strong_chvatal_gomory_cuts != 0) {
    if (toc(start_time) >= settings.time_limit) { return true; }
    f_t cut_start_time = tic();
    generate_gomory_cuts(lp,
                         settings,
                         Arow,
                         new_slacks,
                         var_types,
                         basis_update,
                         xstar,
                         basic_list,
                         nonbasic_list,
                         start_time);
    f_t cut_generation_time = toc(cut_start_time);
    if (cut_generation_time > 1.0) {
      settings.log.debug("Gomory and CG cut generation time %.2f seconds\n", cut_generation_time);
    }
  }

  // Generate Knapsack cuts
  if (settings.knapsack_cuts != 0) {
    if (toc(start_time) >= settings.time_limit) { return true; }
    f_t cut_start_time = tic();
    generate_knapsack_cuts(lp, settings, Arow, new_slacks, var_types, xstar, start_time);
    f_t cut_generation_time = toc(cut_start_time);
    if (cut_generation_time > 1.0) {
      settings.log.debug("Knapsack cut generation time %.2f seconds\n", cut_generation_time);
    }
  }

  // Generate Flow Cover cuts
  if (settings.flow_cover_cuts != 0) {
    if (toc(start_time) >= settings.time_limit) { return true; }
    f_t cut_start_time = tic();
    generate_flow_cover_cuts(lp, settings, Arow, var_types, xstar, variable_bounds, start_time);
    f_t cut_generation_time = toc(cut_start_time);
    if (cut_generation_time > 1.0) {
      settings.log.debug("Flow cover cut generation time %.2f seconds\n", cut_generation_time);
    }
  }

  // Generate MIR and CG cuts
  if (settings.mir_cuts != 0 || settings.strong_chvatal_gomory_cuts != 0) {
    if (toc(start_time) >= settings.time_limit) { return true; }
    f_t cut_start_time = tic();
    generate_mir_cuts(
      lp, settings, Arow, new_slacks, var_types, xstar, ystar, variable_bounds, start_time);
    f_t cut_generation_time = toc(cut_start_time);
    if (cut_generation_time > 1.0) {
      settings.log.debug("MIR and CG cut generation time %.2f seconds\n", cut_generation_time);
    }
  }

  // Generate implied bound cuts
  if (settings.implied_bound_cuts != 0) {
    if (toc(start_time) >= settings.time_limit) { return true; }
    f_t cut_start_time = tic();
    generate_implied_bound_cuts(lp, settings, var_types, xstar, start_time);
    f_t cut_generation_time = toc(cut_start_time);
    if (cut_generation_time > 1.0) {
      settings.log.debug("Implied bounds cut generation time %.2f seconds\n", cut_generation_time);
    }
  }

  // Build the fractional conflict-graph subgraph once (resolving the async
  // clique-table future on the way) so both clique-cut and zero-half cut
  // separators consume the same vertex/weight/adjacency tables instead of
  // each recomputing them. Done here, after the cut routines that don't
  // need the clique table, to give the background clique-table thread as
  // much time as possible to finish before we join it.
  if (toc(start_time) >= settings.time_limit) { return true; }
  prepare_fractional_sub_conflict_graph(settings, xstar, start_time);

  // Generate Clique cuts (last to give background clique table generation maximum time)
  if (settings.clique_cuts != 0) {
    if (toc(start_time) >= settings.time_limit) { return true; }
    f_t cut_start_time = tic();
    bool feasible      = generate_clique_cuts(lp, settings, var_types, xstar, zstar, start_time);
    if (!feasible) {
      settings.log.printf("Clique cuts proved infeasible\n");
      return false;
    }
    f_t cut_generation_time = toc(cut_start_time);
    if (cut_generation_time > 1.0) {
      settings.log.debug("Clique cut generation time %.2f seconds\n", cut_generation_time);
    }
  }

  // Generate Zero-half (odd-cycle / odd-wheel) cuts; reuses the clique table built above
  if (settings.zero_half_cuts != 0) {
    if (toc(start_time) >= settings.time_limit) { return true; }
    ZERO_HALF_DEBUG("generate_cuts: about to call generate_zero_half_cuts");
    f_t cut_start_time = tic();
    bool feasible      = generate_zero_half_cuts(
      lp, settings, Arow, new_slacks, var_types, xstar, zstar, variable_bounds, start_time);
    ZERO_HALF_DEBUG("generate_cuts: returned from generate_zero_half_cuts feasible=%d",
                    static_cast<int>(feasible));
    if (!feasible) {
      settings.log.printf("Zero-half cuts proved infeasible\n");
      return false;
    }
    f_t cut_generation_time = toc(cut_start_time);
    if (cut_generation_time > 1.0) {
      settings.log.debug("Zero-half cut generation time %.2f seconds\n", cut_generation_time);
    }
  } else {
    ZERO_HALF_DEBUG("generate_cuts: zero_half_cuts disabled (setting=%d)",
                    static_cast<int>(settings.zero_half_cuts));
  }
  return true;
}

template <typename i_t, typename f_t>
bool rational_coefficients(const std::vector<variable_type_t>& var_types,
                           const std::vector<f_t>& lower_bounds,
                           const std::vector<f_t>& upper_bounds,
                           inequality_sense_t sense,
                           const inequality_t<i_t, f_t>& input_inequality,
                           inequality_t<i_t, f_t>& rational_inequality)
{
  rational_inequality = input_inequality;

  std::vector<int64_t> numerators;
  std::vector<int64_t> denominators;
  std::vector<i_t> indices;
  for (i_t k = 0; k < input_inequality.size(); k++) {
    const i_t j = rational_inequality.index(k);
    const f_t x = rational_inequality.coeff(k);
    if (var_types[j] == variable_type_t::INTEGER) {
      int64_t numerator, denominator;
      if (!rational_approximation(x, static_cast<int64_t>(1000), numerator, denominator)) {
        return false;
      }
      numerators.push_back(numerator);
      denominators.push_back(denominator);
      indices.push_back(k);
      rational_inequality.vector.x[k] = static_cast<f_t>(numerator) / static_cast<f_t>(denominator);
    }
  }

  int64_t gcd_numerators   = gcd(numerators);
  int64_t lcm_denominators = lcm(denominators);

  f_t scalar = static_cast<f_t>(lcm_denominators) / static_cast<f_t>(gcd_numerators);
  if (!std::isfinite(scalar) || scalar <= 0.0 || scalar > 1000.0) { return false; }

  // The scaled product can land an ulp off the integer it represents.
  rational_inequality.scale(scalar);

  // Suppose sum_j a_j * x_j <= beta, and we compute a scaling s such that
  // A_j = s*a_j + delta_j, with A_j integer. Here delta_j is the error when
  // rationalizing the coefficient. We modify the rhs of the inequality as
  // follows to account for this error. We then have
  // sum_j A_j * x_j <= s*beta
  //                   + sum_{j : delta_j > 0} delta_j * u_j
  //                   + sum_{j : delta_j < 0} delta_j * l_j.
  // For a >= inequality, exchange the upper and lower bounds.
  const i_t sz = indices.size();
  for (i_t k = 0; k < sz; k++) {
    const i_t index = indices[k];
    const i_t j     = input_inequality.index(index);
    const f_t coeff = input_inequality.coeff(index);
    const f_t delta = rational_inequality.coeff(index) - scalar * coeff;
    if (delta == 0.0) { continue; }

    const bool use_upper_bound =
      sense == inequality_sense_t::LESS_EQUAL ? delta > 0.0 : delta < 0.0;
    const f_t bound = use_upper_bound ? upper_bounds[j] : lower_bounds[j];
    if (!std::isfinite(bound)) { return false; }
    rational_inequality.rhs += delta * bound;
  }

  return std::isfinite(rational_inequality.rhs);
}

int64_t gcd(const std::vector<int64_t>& integers)
{
  if (integers.empty()) { return 0; }

  int64_t result = integers[0];
  for (size_t i = 1; i < integers.size(); ++i) {
    result = std::gcd(result, integers[i]);
  }
  return result;
}

int64_t lcm(const std::vector<int64_t>& integers)
{
  if (integers.empty()) { return 0; }
  int64_t result =
    std::reduce(std::next(integers.begin()), integers.end(), integers[0], [](int64_t a, int64_t b) {
      return std::lcm(a, b);
    });
  return result;
}

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
             std::vector<f_t>& edge_norms)

{
  // Given a set of cuts: C*x <= d that are currently violated
  // by the current solution x* (i.e. C*x* > d), this function
  // adds the cuts into the LP and solves again.

#ifdef CHECK_BASIS
  {
    csc_matrix_t<i_t, f_t> Btest(lp.num_rows, lp.num_rows, 1);
    basis_update.multiply_lu(Btest);
    csc_matrix_t<i_t, f_t> B(lp.num_rows, lp.num_rows, 1);
    form_b(lp.A, basic_list, B);
    csc_matrix_t<i_t, f_t> Diff(lp.num_rows, lp.num_rows, 1);
    add(Btest, B, 1.0, -1.0, Diff);
    const f_t err = Diff.norm1();
    settings.log.printf("Before || B - L*U || %e\n", err);
    assert(err <= 1e-6);
  }
#endif

  const i_t p = cuts.m;
  if (cut_rhs.size() != static_cast<size_t>(p)) {
    settings.log.printf("cut_rhs must have the same number of rows as cuts\n");
    assert(cut_rhs.size() == static_cast<size_t>(p));
  }
  settings.log.debug("Number of cuts %d\n", p);
  settings.log.debug("Original lp rows %d\n", lp.num_rows);
  settings.log.debug("Original lp cols %d\n", lp.num_cols);

  csr_matrix_t<i_t, f_t> new_A_row(lp.num_rows, lp.num_cols, 1);
  lp.A.to_compressed_row(new_A_row);

  i_t append_status = new_A_row.append_rows(cuts);
  if (append_status != 0) {
    settings.log.printf("append_rows error: %d\n", append_status);
    assert(append_status == 0);
  }

  csc_matrix_t<i_t, f_t> new_A_col(lp.num_rows + p, lp.num_cols, 1);
  new_A_row.to_compressed_col(new_A_col);

  // Add in slacks variables for the new rows
  lp.lower.resize(lp.num_cols + p);
  lp.upper.resize(lp.num_cols + p);
  lp.objective.resize(lp.num_cols + p);
  edge_norms.resize(lp.num_cols + p);
  i_t nz = new_A_col.col_start[lp.num_cols];
  new_A_col.col_start.resize(lp.num_cols + p + 1);
  new_A_col.i.resize(nz + p);
  new_A_col.x.resize(nz + p);
  i_t k = lp.num_rows;
  for (i_t j = lp.num_cols; j < lp.num_cols + p; j++) {
    new_A_col.col_start[j] = nz;
    new_A_col.i[nz]        = k++;
    new_A_col.x[nz]        = 1.0;
    nz++;
    lp.lower[j]     = 0.0;
    lp.upper[j]     = inf;
    lp.objective[j] = 0.0;
    edge_norms[j]   = 1.0;
    new_slacks.push_back(j);
  }
  settings.log.debug("Done adding slacks\n");
  new_A_col.col_start[lp.num_cols + p] = nz;
  new_A_col.n                          = lp.num_cols + p;

  lp.A = new_A_col;

  // Check that all slack columns have length 1
  for (i_t slack : new_slacks) {
    const i_t col_start = lp.A.col_start[slack];
    const i_t col_end   = lp.A.col_start[slack + 1];
    const i_t col_len   = col_end - col_start;
    if (col_len != 1) {
      settings.log.printf("Add cuts: Slack %d has %d nzs in column\n", slack, col_len);
      assert(col_len == 1);
    }
  }

  i_t old_rows = lp.num_rows;
  lp.num_rows += p;
  i_t old_cols = lp.num_cols;
  lp.num_cols += p;

  lp.rhs.resize(lp.num_rows);
  for (i_t k = old_rows; k < old_rows + p; k++) {
    const i_t h = k - old_rows;
    lp.rhs[k]   = cut_rhs[h];
  }
  settings.log.debug("Done adding rhs\n");

  // Construct C_B = C(:, basic_list)
  std::vector<i_t> C_col_degree(lp.num_cols, 0);
  i_t cuts_nz = cuts.row_start[p];
  for (i_t q = 0; q < cuts_nz; q++) {
    const i_t j = cuts.j[q];
    if (j >= lp.num_cols) {
      settings.log.printf("Cut column index j=%d exceeds num_cols=%d\n", j, lp.num_cols);
      return -1;
    }
    C_col_degree[j]++;
  }
  settings.log.debug("Done computing C_col_degree\n");

  std::vector<i_t> in_basis(old_cols, -1);
  const i_t num_basic = static_cast<i_t>(basic_list.size());
  i_t C_B_nz          = 0;
  for (i_t k = 0; k < num_basic; k++) {
    const i_t j = basic_list[k];
    if (j < 0 || j >= old_cols) {
      settings.log.printf(
        "basic_list[%d] = %d is out of bounds %d old_cols %d\n", k, j, j, old_cols);
      assert(j >= 0 && j < old_cols);
    }
    in_basis[j] = k;
    // The cuts are on the original variables. So it is possible that
    // a slack will be basic and thus not part of the cuts matrix
    if (j < cuts.n) { C_B_nz += C_col_degree[j]; }
  }
  settings.log.debug("Done estimating C_B_nz\n");

  csr_matrix_t<i_t, f_t> C_B(p, num_basic, C_B_nz);
  nz = 0;
  for (i_t i = 0; i < p; i++) {
    C_B.row_start[i]    = nz;
    const i_t row_start = cuts.row_start[i];
    const i_t row_end   = cuts.row_start[i + 1];
    for (i_t q = row_start; q < row_end; q++) {
      const i_t j       = cuts.j[q];
      const i_t j_basis = in_basis[j];
      if (j_basis == -1) { continue; }
      C_B.j[nz] = j_basis;
      C_B.x[nz] = cuts.x[q];
      nz++;
    }
  }
  C_B.row_start[p] = nz;

  if (nz != C_B_nz) {
    settings.log.printf("Add cuts: predicted nz %d actual nz %d\n", C_B_nz, nz);
    assert(nz == C_B_nz);
  }
  settings.log.debug("C_B rows %d cols %d nz %d\n", C_B.m, C_B.n, nz);

  // Adjust the basis update to include the new cuts
  basis_update.append_cuts(C_B);

  basic_list.resize(lp.num_rows, 0);
  i_t h = old_cols;
  for (i_t j = old_rows; j < lp.num_rows; j++) {
    basic_list[j] = h++;
  }

#ifdef CHECK_BASIS
  // Check the basis update
  csc_matrix_t<i_t, f_t> Btest(lp.num_rows, lp.num_rows, 1);
  basis_update.multiply_lu(Btest);

  csc_matrix_t<i_t, f_t> B(lp.num_rows, lp.num_rows, 1);
  form_b(lp.A, basic_list, B);

  csc_matrix_t<i_t, f_t> Diff(lp.num_rows, lp.num_rows, 1);
  add(Btest, B, 1.0, -1.0, Diff);
  const f_t err = Diff.norm1();
  settings.log.printf("After || B - L*U || %e\n", err);
  if (err > 1e-6) {
    settings.log.printf("Diff matrix\n");
    // Diff.print_matrix();
    assert(err <= 1e-6);
  }
#endif
  // Adjust the vstatus
  vstatus.resize(lp.num_cols);
  for (i_t j = old_cols; j < lp.num_cols; j++) {
    vstatus[j] = variable_status_t::BASIC;
  }

  // Adjust the solution
  solution.x.resize(lp.num_cols, 0.0);
  solution.y.resize(lp.num_rows, 0.0);
  solution.z.resize(lp.num_cols, 0.0);

  return 0;
}

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
                basis_update_mpf_t<i_t, f_t>& basis_update)
{
  std::vector<i_t> cuts_to_remove;
  cuts_to_remove.reserve(lp.num_rows - original_rows);
  std::vector<i_t> slacks_to_remove;
  slacks_to_remove.reserve(lp.num_rows - original_rows);
  const f_t dual_tol = 1e-10;

  std::vector<i_t> is_slack(lp.num_cols, 0);
  for (i_t j : new_slacks) {
    is_slack[j] = 1;
#ifdef CHECK_SLACKS
    // Check that slack column length is 1
    const i_t col_start = lp.A.col_start[j];
    const i_t col_end   = lp.A.col_start[j + 1];
    const i_t col_len   = col_end - col_start;
    if (col_len != 1) {
      printf("Remove cuts: Slack %d has %d nzs in column\n", j, col_len);
      assert(col_len == 1);
    }
#endif
  }

  for (i_t k = original_rows; k < lp.num_rows; k++) {
    if (std::abs(y[k]) < dual_tol) {
      const i_t row_start = Arow.row_start[k];
      const i_t row_end   = Arow.row_start[k + 1];
      i_t last_slack      = -1;
      const f_t slack_tol = 1e-3;
      for (i_t p = row_start; p < row_end; p++) {
        const i_t j = Arow.j[p];
        if (is_slack[j]) {
          if (vstatus[j] == variable_status_t::BASIC && x[j] > slack_tol) { last_slack = j; }
        }
      }
      if (last_slack != -1) {
        cuts_to_remove.push_back(k);
        slacks_to_remove.push_back(last_slack);
      }
    }
  }

  if (cuts_to_remove.size() > 0) {
    std::vector<i_t> marked_rows(lp.num_rows, 0);
    for (i_t i : cuts_to_remove) {
      marked_rows[i] = 1;
    }
    std::vector<i_t> marked_cols(lp.num_cols, 0);
    for (i_t j : slacks_to_remove) {
      marked_cols[j] = 1;
    }

    std::vector<f_t> new_rhs(lp.num_rows - cuts_to_remove.size());
    std::vector<f_t> new_solution_y(lp.num_rows - cuts_to_remove.size());
    i_t h = 0;
    for (i_t i = 0; i < lp.num_rows; i++) {
      if (!marked_rows[i]) {
        new_rhs[h]        = lp.rhs[i];
        new_solution_y[h] = y[i];
        h++;
      }
    }
    csr_matrix_t<i_t, f_t> new_Arow(1, 1, 0);
    Arow.remove_rows(marked_rows, new_Arow);
    Arow = new_Arow;
    Arow.to_compressed_col(lp.A);

    std::vector<f_t> new_objective(lp.num_cols - slacks_to_remove.size());
    std::vector<f_t> new_lower(lp.num_cols - slacks_to_remove.size());
    std::vector<f_t> new_upper(lp.num_cols - slacks_to_remove.size());
    std::vector<variable_type_t> new_var_types(lp.num_cols - slacks_to_remove.size());
    std::vector<variable_status_t> new_vstatus(lp.num_cols - slacks_to_remove.size());
    std::vector<f_t> new_edge_norms(lp.num_cols - slacks_to_remove.size());
    std::vector<i_t> new_basic_list;
    new_basic_list.reserve(lp.num_rows - slacks_to_remove.size());
    std::vector<i_t> new_nonbasic_list;
    new_nonbasic_list.reserve(nonbasic_list.size());
    std::vector<f_t> new_solution_x(lp.num_cols - slacks_to_remove.size());
    std::vector<f_t> new_solution_z(lp.num_cols - slacks_to_remove.size());
    std::vector<i_t> new_is_slacks(lp.num_cols - slacks_to_remove.size(), 0);
    h = 0;
    for (i_t k = 0; k < lp.num_cols; k++) {
      if (!marked_cols[k]) {
        new_objective[h]  = lp.objective[k];
        new_lower[h]      = lp.lower[k];
        new_upper[h]      = lp.upper[k];
        new_var_types[h]  = var_types[k];
        new_vstatus[h]    = vstatus[k];
        new_edge_norms[h] = edge_norms[k];
        new_solution_x[h] = x[k];
        new_solution_z[h] = z[k];
        new_is_slacks[h]  = is_slack[k];
        if (new_vstatus[h] != variable_status_t::BASIC) {
          new_nonbasic_list.push_back(h);
        } else {
          new_basic_list.push_back(h);
        }
        h++;
      }
    }
    lp.A.remove_columns(marked_cols);
    lp.A.to_compressed_row(Arow);
    lp.objective = new_objective;
    lp.lower     = new_lower;
    lp.upper     = new_upper;
    lp.rhs       = new_rhs;
    var_types    = new_var_types;
    lp.num_cols  = lp.A.n;
    lp.num_rows  = lp.A.m;

    new_slacks.clear();
    new_slacks.reserve(lp.num_cols);
    for (i_t j = 0; j < lp.num_cols; j++) {
      if (new_is_slacks[j]) { new_slacks.push_back(j); }
    }
    basic_list    = new_basic_list;
    nonbasic_list = new_nonbasic_list;
    vstatus       = new_vstatus;
    edge_norms    = new_edge_norms;
    x             = new_solution_x;
    y             = new_solution_y;
    z             = new_solution_z;

    settings.log.debug("Removed %d cuts. After removal %d rows %d columns %d nonzeros\n",
                       cuts_to_remove.size(),
                       lp.num_rows,
                       lp.num_cols,
                       lp.A.col_start[lp.A.n]);

    basis_update.resize(lp.num_rows);
    i_t deficient_repaired = 0;
    i_t refactor_status    = basis_update.refactor_basis(lp.A,
                                                      settings,
                                                      lp.lower,
                                                      lp.upper,
                                                      start_time,
                                                      basic_list,
                                                      nonbasic_list,
                                                      vstatus,
                                                      deficient_repaired);
    if (refactor_status == CONCURRENT_HALT_RETURN) { return CONCURRENT_HALT_RETURN; }
    if (refactor_status == TIME_LIMIT_RETURN) { return TIME_LIMIT_RETURN; }
    if (refactor_status != 0 || deficient_repaired > 0) { return -1; }
  }

  return 0;
}

template <typename i_t, typename f_t>
void read_saved_solution_for_cut_verification(const lp_problem_t<i_t, f_t>& lp,
                                              const simplex_solver_settings_t<i_t, f_t>& settings,
                                              std::vector<f_t>& saved_solution)
{
  settings.log.printf("Trying to open solution.dat\n");
  FILE* fid = NULL;
  fid       = fopen("solution.dat", "r");
  if (fid != NULL) {
    i_t n_solution_dat;
    i_t count = fscanf(fid, "%d\n", &n_solution_dat);
    settings.log.printf(
      "Solution.dat variables %d =? %d =? count %d\n", n_solution_dat, lp.num_cols, count);
    bool good = true;
    if (count == 1 && n_solution_dat == lp.num_cols) {
      settings.log.printf("Opened solution.dat with %d number of variables\n", n_solution_dat);
      saved_solution.resize(n_solution_dat);
      for (i_t j = 0; j < n_solution_dat; j++) {
        count = fscanf(fid, "%lf", &saved_solution[j]);
        if (count != 1) {
          settings.log.printf("bad read solution.dat: j %d count %d\n", j, count);
          good = false;
          break;
        }
      }
    } else {
      good = false;
    }
    fclose(fid);

    if (!good) {
      saved_solution.resize(0);
      settings.log.printf("Solution.dat is bad\n");
    } else {
      settings.log.printf("Read solution file\n");

      auto hash_combine_f = [](size_t seed, f_t x) {
        seed ^= std::hash<f_t>{}(x) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
        return seed;
      };
      size_t seed = lp.num_cols;
      for (i_t j = 0; j < lp.num_cols; ++j) {
        seed = hash_combine_f(seed, saved_solution[j]);
      }
      settings.log.printf("Saved solution hash: %20zx\n", seed);

      // Compute || A * x - b ||_inf
      std::vector<f_t> residual = lp.rhs;
      matrix_vector_multiply(lp.A, 1.0, saved_solution, -1.0, residual);
      settings.log.printf("Saved solution: || A*x - b ||_inf %e\n",
                          vector_norm_inf<i_t, f_t>(residual));
      f_t infeas = 0;
      for (i_t j = 0; j < lp.num_cols; j++) {
        if (saved_solution[j] < lp.lower[j] - 1e-6) {
          f_t curr_infeas = (lp.lower[j] - saved_solution[j]);
          infeas += curr_infeas;
          settings.log.printf(
            "j: %d saved solution %e lower %e\n", j, saved_solution[j], lp.lower[j]);
        }
        if (saved_solution[j] > lp.upper[j] + 1e-6) {
          f_t curr_infeas = (saved_solution[j] - lp.upper[j]);
          infeas += curr_infeas;
          settings.log.printf(
            "j %d saved solution %e upper %e\n", j, saved_solution[j], lp.upper[j]);
        }
      }
      settings.log.printf("Bound infeasibility %e\n", infeas);
    }
  } else {
    settings.log.printf("Could not open solution.dat\n");
  }
}

template <typename i_t, typename f_t>
void write_solution_for_cut_verification(const lp_problem_t<i_t, f_t>& lp,
                                         const std::vector<f_t>& solution)
{
  FILE* fid = NULL;
  fid       = fopen("solution.dat", "w");
  if (fid != NULL) {
    printf("Writing solution.dat\n");

    std::vector<f_t> residual = lp.rhs;
    matrix_vector_multiply(lp.A, 1.0, solution, -1.0, residual);
    printf("|| A*x - b ||_inf %e\n", vector_norm_inf<i_t, f_t>(residual));
    auto hash_combine_f = [](size_t seed, f_t x) {
      seed ^= std::hash<f_t>{}(x) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
      return seed;
    };
    printf("incumbent size %ld original lp cols %d\n", solution.size(), lp.num_cols);
    i_t n       = lp.num_cols;
    size_t seed = n;
    fprintf(fid, "%d\n", n);
    for (i_t j = 0; j < n; ++j) {
      fprintf(fid, "%.17g\n", solution[j]);
      seed = hash_combine_f(seed, solution[j]);
    }
    printf("Solution hash: %20zx\n", seed);
    fclose(fid);
  }
}

template <typename i_t, typename f_t>
void verify_cuts_against_saved_solution(const csr_matrix_t<i_t, f_t>& cuts,
                                        const std::vector<f_t>& cut_rhs,
                                        const std::vector<f_t>& saved_solution)
{
  if (saved_solution.size() > 0) {
    csc_matrix_t<i_t, f_t> cuts_to_add_col(cuts.m, cuts.n, cuts.row_start[cuts.m]);
    cuts.to_compressed_col(cuts_to_add_col);
    std::vector<f_t> Cx(cuts.m);
    matrix_vector_multiply(cuts_to_add_col, 1.0, saved_solution, 0.0, Cx);
    const i_t num_cuts = cuts.m;
    for (i_t k = 0; k < num_cuts; k++) {
      if (Cx[k] > cut_rhs[k] + 1e-6) {
        printf("Cut %d is violated by saved solution. Cx %e cut_rhs %e Diff: %e\n",
               k,
               Cx[k],
               cut_rhs[k],
               Cx[k] - cut_rhs[k]);
      }
    }
  }
}

#ifdef DUAL_SIMPLEX_INSTANTIATE_DOUBLE
template class cut_generation_t<int, double>;

template bool rational_coefficients<int, double>(const std::vector<variable_type_t>& var_types,
                                                 const std::vector<double>& lower_bounds,
                                                 const std::vector<double>& upper_bounds,
                                                 inequality_sense_t sense,
                                                 const inequality_t<int, double>& inequality,
                                                 inequality_t<int, double>& rational_inequality);

template int add_cuts(const simplex_solver_settings_t<int, double>& settings,
                      const csr_matrix_t<int, double>& cuts,
                      const std::vector<double>& cut_rhs,
                      lp_problem_t<int, double>& lp,
                      std::vector<int>& new_slacks,
                      lp_solution_t<int, double>& solution,
                      basis_update_mpf_t<int, double>& basis_update,
                      std::vector<int>& basic_list,
                      std::vector<int>& nonbasic_list,
                      std::vector<variable_status_t>& vstatus,
                      std::vector<double>& edge_norms);

template int remove_cuts<int, double>(lp_problem_t<int, double>& lp,
                                      const simplex_solver_settings_t<int, double>& settings,
                                      double start_time,
                                      csr_matrix_t<int, double>& Arow,
                                      std::vector<int>& new_slacks,
                                      int original_rows,
                                      std::vector<variable_type_t>& var_types,
                                      std::vector<variable_status_t>& vstatus,
                                      std::vector<double>& edge_norms,
                                      std::vector<double>& x,
                                      std::vector<double>& y,
                                      std::vector<double>& z,
                                      std::vector<int>& basic_list,
                                      std::vector<int>& nonbasic_list,
                                      basis_update_mpf_t<int, double>& basis_update);

template void read_saved_solution_for_cut_verification<int, double>(
  const lp_problem_t<int, double>& lp,
  const simplex_solver_settings_t<int, double>& settings,
  std::vector<double>& saved_solution);

template void write_solution_for_cut_verification<int, double>(const lp_problem_t<int, double>& lp,
                                                               const std::vector<double>& solution);

template void verify_cuts_against_saved_solution<int, double>(
  const csr_matrix_t<int, double>& cuts,
  const std::vector<double>& cut_rhs,
  const std::vector<double>& saved_solution);
#endif

}  // namespace cuopt::mathematical_optimization::mip
