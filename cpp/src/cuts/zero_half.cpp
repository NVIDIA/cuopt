/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#include <cuts/conflict_graph_cut_helpers.hpp>
#include <cuts/cuts.hpp>
#include <cuts/cuts_debug.hpp>
#include <math_optimization/tic_toc.hpp>
#include <presolve/conflict_graph/clique_table.cuh>
#include <utilities/macros.cuh>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <functional>
#include <limits>
#include <queue>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace cuopt::mathematical_optimization::mip {

using simplex::lp_problem_t;
using simplex::simplex_solver_settings_t;
using simplex::variable_type_t;

namespace {

// Build a zero-half (odd-cycle / odd-wheel) cut from a cycle and optional wheel
// centers. cycle_vertices is a simple odd cycle in the conflict graph using the
// 2*num_vars vertex indexing (var j and complement j+num_vars). wheel_centers
// are extra vertices each adjacent to every vertex in cycle_vertices. The
// resulting cut is stored in the form a^T x >= rhs to match cut_pool_t.
template <typename i_t, typename f_t>
clique_cut_build_status_t build_zero_half_cut(const std::vector<i_t>& cycle_vertices,
                                              const std::vector<i_t>& wheel_centers,
                                              i_t num_vars,
                                              const std::vector<variable_type_t>& var_types,
                                              const std::vector<f_t>& lower_bounds,
                                              const std::vector<f_t>& upper_bounds,
                                              const std::vector<f_t>& xstar,
                                              f_t bound_tol,
                                              f_t min_violation,
                                              sparse_vector_t<i_t, f_t>& cut,
                                              f_t& cut_rhs,
                                              f_t* work_estimate,
                                              f_t max_work_estimate)
{
  const size_t cycle_size = cycle_vertices.size();
  if (cycle_size < 5 || (cycle_size % 2) == 0) {
    ZERO_HALF_DEBUG("build_zero_half_cut reject cycle_size=%zu", cycle_size);
    return clique_cut_build_status_t::NO_CUT;
  }
  cuopt_assert(num_vars > 0, "Zero-half cut num_vars must be positive");
  cuopt_assert(static_cast<size_t>(num_vars) <= lower_bounds.size(),
               "Zero-half cut lower bounds size mismatch");
  cuopt_assert(static_cast<size_t>(num_vars) <= xstar.size(), "Zero-half cut xstar size mismatch");

  const i_t m   = static_cast<i_t>((cycle_size - 1) / 2);
  const f_t f_m = static_cast<f_t>(m);
  // The guard above rejects even or <5 cycles, so the cycle decomposes as
  // exactly 2m+1 literals with m >= 2. The whole zero-half lift (rhs = -m,
  // unit cycle coefficients, m-weighted wheel centers) depends on this.
  cuopt_assert(2 * m + 1 == static_cast<i_t>(cycle_size),
               "Zero-half cut: cycle_size must equal 2m+1 (odd cycle)");
  cuopt_assert(m >= 2, "Zero-half cut: odd cycle must have length >= 5 (m >= 2)");
  const f_t total_size     = static_cast<f_t>(cycle_size + wheel_centers.size());
  const f_t estimated_work = 8.0 * total_size + 2.0 * total_size * std::log2(total_size + 1.0);
  if (add_work_estimate(estimated_work, work_estimate, max_work_estimate)) {
    ZERO_HALF_DEBUG("build_zero_half_cut work_limit hit");
    return clique_cut_build_status_t::NO_CUT;
  }

  cut.i.clear();
  cut.x.clear();

  std::unordered_map<i_t, f_t> coeff_by_var;
  std::unordered_set<i_t> seen_original;
  std::unordered_set<i_t> seen_complement;
  coeff_by_var.reserve(cycle_size + wheel_centers.size());
  seen_original.reserve(cycle_size + wheel_centers.size());
  seen_complement.reserve(cycle_size + wheel_centers.size());

  f_t rhs_acc = -f_m;

  auto accumulate =
    [&](const std::vector<i_t>& verts, f_t weight, bool is_cycle) -> clique_cut_build_status_t {
    ZERO_HALF_DEBUG("build_zero_half_cut accumulate verts.size=%zu weight=%g is_cycle=%d",
                    verts.size(),
                    static_cast<double>(weight),
                    static_cast<int>(is_cycle));
    for (const i_t vertex_idx : verts) {
      ZERO_HALF_DEBUG("  acc vertex_idx=%lld (range [0, %lld))",
                      static_cast<long long>(vertex_idx),
                      static_cast<long long>(2 * num_vars));
      cuopt_assert(vertex_idx >= 0 && vertex_idx < 2 * num_vars, "Zero-half vertex out of range");
      const i_t var_idx     = vertex_idx % num_vars;
      const bool complement = vertex_idx >= num_vars;
      const f_t lower_bound = lower_bounds[var_idx];
      const f_t upper_bound = upper_bounds[var_idx];
      cuopt_assert(var_types[var_idx] != variable_type_t::CONTINUOUS,
                   "Zero-half cut contains continuous variable");
      cuopt_assert(lower_bound >= -bound_tol, "Zero-half variable lower bound below zero");
      cuopt_assert(upper_bound <= 1 + bound_tol, "Zero-half variable upper bound above one");

      if (complement) {
        if (seen_original.count(var_idx) > 0) { return clique_cut_build_status_t::NO_CUT; }
        seen_complement.insert(var_idx);
        coeff_by_var[var_idx] += weight;
        rhs_acc += weight;
      } else {
        if (seen_complement.count(var_idx) > 0) { return clique_cut_build_status_t::NO_CUT; }
        seen_original.insert(var_idx);
        coeff_by_var[var_idx] -= weight;
      }
    }
    return clique_cut_build_status_t::CUT_ADDED;
  };

  if (accumulate(cycle_vertices, static_cast<f_t>(1), true) !=
      clique_cut_build_status_t::CUT_ADDED) {
    ZERO_HALF_DEBUG("build_zero_half_cut cycle accumulate failed");
    return clique_cut_build_status_t::NO_CUT;
  }
  if (m > 0 && !wheel_centers.empty()) {
    if (accumulate(wheel_centers, f_m, false) != clique_cut_build_status_t::CUT_ADDED) {
      ZERO_HALF_DEBUG("build_zero_half_cut wheel accumulate failed");
      return clique_cut_build_status_t::NO_CUT;
    }
  }

  const f_t coeff_zero_tol = static_cast<f_t>(1e-12);
  cut.i.reserve(coeff_by_var.size());
  cut.x.reserve(coeff_by_var.size());
  for (const std::pair<const i_t, f_t>& kv : coeff_by_var) {
    if (std::abs(kv.second) <= coeff_zero_tol) { continue; }
    // Each variable appears at most once on the cycle (contributing +/-1) and
    // at most once among the wheel centers (contributing +/-m), so no final
    // coefficient can exceed 1 + m in magnitude. A larger value means a vertex
    // was double-counted in accumulation.
    cuopt_assert(std::abs(kv.second) <= f_m + static_cast<f_t>(1) + bound_tol,
                 "Zero-half coefficient exceeds 1 + m (vertex double-counted?)");
    cut.i.push_back(kv.first);
    cut.x.push_back(kv.second);
  }
  // Support is bounded by the number of distinct accumulated vertices.
  cuopt_assert(cut.i.size() <= cycle_size + wheel_centers.size(),
               "Zero-half cut support exceeds accumulated vertex count");

  if (cut.i.empty()) {
    ZERO_HALF_DEBUG("build_zero_half_cut empty support after accumulation");
    return clique_cut_build_status_t::NO_CUT;
  }

  cut_rhs = rhs_acc;
  cut.sort();

  const f_t dot       = cut.dot(xstar);
  const f_t violation = cut_rhs - dot;
  ZERO_HALF_DEBUG(
    "build_zero_half_cut nz=%lld rhs=%g dot=%g violation=%g threshold=%g cycle=%lld wheel=%lld",
    static_cast<long long>(cut.i.size()),
    static_cast<double>(cut_rhs),
    static_cast<double>(dot),
    static_cast<double>(violation),
    static_cast<double>(min_violation),
    static_cast<long long>(cycle_size),
    static_cast<long long>(wheel_centers.size()));
  cuopt_assert(violation > -bound_tol, "Zero-half cut violation flipped sign unexpectedly");
  if (violation > min_violation) { return clique_cut_build_status_t::CUT_ADDED; }
  return clique_cut_build_status_t::NO_CUT;
}

// Reusable scratch for dijkstra_odd_cycle. The separation loop runs Dijkstra
// once per source vertex; re-allocating and re-initializing dist/prev. Instead
// we allocate the buffers once and reset them in O(1) using a generation stamp:
// dist[v]/prev[v] are considered valid for the current call only when
// stamp[v] == gen.
template <typename i_t, typename f_t>
struct dijkstra_scratch_t {
  std::vector<f_t> dist;
  std::vector<i_t> prev;
  std::vector<std::uint64_t> stamp;  // stamp[v] == gen  <=>  dist[v]/prev[v] valid this call
  std::uint64_t gen{0};

  void ensure_size(std::size_t n)
  {
    if (stamp.size() < n) {
      dist.resize(n);
      prev.resize(n);
      stamp.assign(n, 0);
      gen = 0;
    }
  }
};

template <typename i_t, typename f_t>
bool dijkstra_odd_cycle(i_t source_local,
                        const std::vector<std::vector<i_t>>& local_adj,
                        const std::vector<f_t>& weights,
                        f_t cutoff,
                        std::vector<i_t>& path,
                        f_t& total_weight,
                        f_t* work_estimate,
                        f_t max_work_estimate,
                        dijkstra_scratch_t<i_t, f_t>& scratch)
{
  const i_t num_local = static_cast<i_t>(local_adj.size());
  if (source_local < 0 || source_local >= num_local) { return false; }
  if (weights.size() != static_cast<size_t>(num_local)) { return false; }
  cuopt_assert(source_local >= 0 && source_local < num_local,
               "Zero-half Dijkstra source out of range");
  cuopt_assert(weights.size() == static_cast<size_t>(num_local),
               "Zero-half Dijkstra weights size mismatch");

  const i_t source_idx = source_local;
  const i_t target_idx = source_local + num_local;
  const i_t total_idx  = 2 * num_local;
  const f_t f_inf      = std::numeric_limits<f_t>::infinity();

  scratch.ensure_size(static_cast<std::size_t>(total_idx));
  ++scratch.gen;
  const std::uint64_t gen           = scratch.gen;
  std::vector<f_t>& dist            = scratch.dist;
  std::vector<i_t>& prev            = scratch.prev;
  std::vector<std::uint64_t>& stamp = scratch.stamp;
  // dist[v]/prev[v] are valid only if last written this call (stamp[v] == gen);
  // otherwise the node is unreached, i.e. distance infinity.
  auto cur_dist = [&](i_t v) -> f_t { return stamp[v] == gen ? dist[v] : f_inf; };

  dist[source_idx]  = 0;
  prev[source_idx]  = -1;
  stamp[source_idx] = gen;

  using node_t = std::pair<f_t, i_t>;
  std::priority_queue<node_t, std::vector<node_t>, std::greater<node_t>> pq;
  pq.emplace(static_cast<f_t>(0), source_idx);

  i_t pops = 0;
  while (!pq.empty()) {
    auto [d, u] = pq.top();
    pq.pop();
    ++pops;
    if (d > cur_dist(u)) { continue; }
    if (u == target_idx) { break; }
    if (cutoff > 0 && d >= cutoff) { break; }

    const i_t u_local = u % num_local;
    const i_t u_part  = u / num_local;
    const i_t v_part  = 1 - u_part;
    cuopt_assert(u_part == 0 || u_part == 1, "Bipartite part out of range");

    const std::vector<i_t>& neigh = local_adj[u_local];
    if (add_work_estimate(static_cast<f_t>(neigh.size()) + 4.0, work_estimate, max_work_estimate)) {
      ZERO_HALF_DEBUG("dijkstra_odd_cycle work_limit hit pops=%lld", static_cast<long long>(pops));
      return false;
    }
    for (const i_t v_local : neigh) {
      cuopt_assert(v_local >= 0 && v_local < num_local, "Zero-half Dijkstra neighbor out of range");
      f_t edge_w = (static_cast<f_t>(1) - weights[u_local] - weights[v_local]) / 2;
      if (edge_w < 0) { edge_w = 0; }
      const i_t v  = v_local + v_part * num_local;
      const f_t nd = d + edge_w;
      if (nd < cur_dist(v)) {
        dist[v]  = nd;
        prev[v]  = u;
        stamp[v] = gen;
        pq.emplace(nd, v);
      }
    }
  }

  const f_t target_dist = cur_dist(target_idx);
  if (!std::isfinite(target_dist)) {
    ZERO_HALF_DEBUG("dijkstra_odd_cycle no path pops=%lld", static_cast<long long>(pops));
    return false;
  }
  total_weight = target_dist;
  // All G' edge weights are clamped to >= 0, so the shortest-path distance must
  // be non-negative; a negative total means the clamp/relaxation invariant broke.
  cuopt_assert(total_weight >= -static_cast<f_t>(1e-9),
               "Zero-half Dijkstra shortest-path distance must be non-negative");
  if (cutoff > 0 && total_weight >= cutoff) {
    ZERO_HALF_DEBUG("dijkstra_odd_cycle path too long total=%g cutoff=%g",
                    static_cast<double>(total_weight),
                    static_cast<double>(cutoff));
    return false;
  }

  path.clear();
  for (i_t cur = target_idx; cur != -1; cur = prev[cur]) {
    path.push_back(cur);
    if (cur == source_idx) { break; }
  }
  cuopt_assert(!path.empty(), "Zero-half Dijkstra path empty");
  cuopt_assert(path.back() == source_idx, "Zero-half Dijkstra path missing source");
  std::reverse(path.begin(), path.end());
  cuopt_assert(path.front() == source_idx, "Zero-half Dijkstra path must start at source");
  cuopt_assert(path.back() == target_idx, "Zero-half Dijkstra path must end at target");
  // bipartite path from j1 to j2 must have odd number of edges
  cuopt_assert((path.size() % 2) == 0, "Zero-half bipartite path must have even node count");
#ifdef ASSERT_MODE
  // Every G' edge crosses between the two bipartite copies, so consecutive path
  // nodes must live in opposite parts (part = bipartite_idx / num_local).
  for (size_t k = 0; k + 1 < path.size(); ++k) {
    cuopt_assert((path[k] / num_local) != (path[k + 1] / num_local),
                 "Zero-half Dijkstra path must alternate bipartite parts");
  }
#endif
  ZERO_HALF_DEBUG("dijkstra_odd_cycle done path.size=%zu total_weight=%g pops=%lld",
                  path.size(),
                  static_cast<double>(total_weight),
                  static_cast<long long>(pops));
  return true;
}

template <typename i_t, typename f_t>
bool path_to_odd_cycle(const std::vector<i_t>& bipartite_path,
                       const std::vector<i_t>& vertices,
                       i_t num_local,
                       i_t num_vars,
                       std::vector<i_t>& cycle_vertices,
                       f_t* work_estimate,
                       f_t max_work_estimate)
{
  ZERO_HALF_DEBUG(
    "path_to_odd_cycle enter bipartite_path.size=%zu vertices.size=%zu num_local=%lld "
    "num_vars=%lld",
    bipartite_path.size(),
    vertices.size(),
    static_cast<long long>(num_local),
    static_cast<long long>(num_vars));
  cycle_vertices.clear();
  if (bipartite_path.size() < 4) {
    ZERO_HALF_DEBUG("path_to_odd_cycle reject short path");
    return false;
  }
  if (add_work_estimate(
        static_cast<f_t>(bipartite_path.size()) * 2.0, work_estimate, max_work_estimate)) {
    ZERO_HALF_DEBUG("path_to_odd_cycle work_limit hit");
    return false;
  }

  std::vector<i_t> local_seq;
  local_seq.reserve(bipartite_path.size());
  for (const i_t bv : bipartite_path) {
    local_seq.push_back(bv % num_local);
  }
  cuopt_assert(local_seq.front() == local_seq.back(), "Zero-half cycle path endpoints must match");

  // Drop the duplicate end so we have a sequence covering each cycle vertex once
  local_seq.pop_back();

  std::unordered_set<i_t> seen_local;
  seen_local.reserve(local_seq.size());
  for (const i_t lv : local_seq) {
    if (!seen_local.insert(lv).second) {
      // Same CG vertex appears twice in the path; reject (degenerate cycle)
      ZERO_HALF_DEBUG("path_to_odd_cycle duplicate local vertex lv=%lld",
                      static_cast<long long>(lv));
      return false;
    }
  }

  cycle_vertices.reserve(local_seq.size());
  std::unordered_set<i_t> seen_var;
  seen_var.reserve(local_seq.size());
  for (const i_t lv : local_seq) {
    const i_t global = vertices[lv];
    cuopt_assert(global >= 0 && global < 2 * num_vars, "Zero-half global vertex out of range");
    const i_t var_idx = global % num_vars;
    if (!seen_var.insert(var_idx).second) {
      // Variable appears as both x and ¯x in the cycle; reject (degenerate)
      ZERO_HALF_DEBUG("path_to_odd_cycle duplicate var_idx=%lld", static_cast<long long>(var_idx));
      return false;
    }
    cycle_vertices.push_back(global);
  }
  cuopt_assert(cycle_vertices.size() == local_seq.size(),
               "Zero-half cycle dropped vertices during global mapping");
  cuopt_assert((cycle_vertices.size() % 2) == 1, "Zero-half extracted cycle must have odd length");
  ZERO_HALF_DEBUG("path_to_odd_cycle done cycle_vertices.size=%zu", cycle_vertices.size());
  return cycle_vertices.size() >= 5;
}

// Greedy lifting: extend an odd cycle by attaching a clique of "wheel center"
// vertices that are adjacent (in CG) to every vertex of the cycle.
template <typename i_t, typename f_t>
void extend_to_odd_wheel(const std::vector<i_t>& cycle_vertices,
                         std::vector<i_t>& wheel_centers,
                         mip::clique_table_t<i_t, f_t>& graph,
                         const std::vector<f_t>& reduced_costs,
                         i_t num_vars,
                         f_t start_time,
                         f_t time_limit,
                         f_t* work_estimate,
                         f_t max_work_estimate)
{
  ZERO_HALF_DEBUG(
    "extend_to_odd_wheel enter cycle.size=%zu num_vars=%lld reduced_costs.size=%zu "
    "graph.n_variables=%lld",
    cycle_vertices.size(),
    static_cast<long long>(num_vars),
    reduced_costs.size(),
    static_cast<long long>(graph.n_variables));
  wheel_centers.clear();
  if (cycle_vertices.empty()) { return; }
  if (toc(start_time) >= time_limit) { return; }

  const i_t smallest_degree_var = min_degree_anchor(cycle_vertices, graph, start_time, time_limit);
  ZERO_HALF_DEBUG("extend_to_odd_wheel smallest_degree_var=%lld",
                  static_cast<long long>(smallest_degree_var));
  if (smallest_degree_var < 0) { return; }

  std::unordered_set<i_t> adj_set = graph.get_adj_set_of_var(smallest_degree_var);
  ZERO_HALF_DEBUG("extend_to_odd_wheel adj_set.size=%zu", adj_set.size());
  std::vector<char> cycle_members(2 * num_vars, 0);
  for (const i_t v : cycle_vertices) {
    cuopt_assert(v >= 0 && v < 2 * num_vars, "Zero-half cycle vertex out of range");
    cycle_members[v] = 1;
  }
  std::vector<i_t> candidates;
  candidates.reserve(adj_set.size());
  for (const i_t candidate : adj_set) {
    if (toc(start_time) >= time_limit) { return; }
    if (cycle_members[candidate] != 0) { continue; }
    bool adj_to_all = true;
    for (const i_t v : cycle_vertices) {
      if (candidate == v) {
        adj_to_all = false;
        break;
      }
      if (!graph.check_adjacency(candidate, v)) {
        adj_to_all = false;
        break;
      }
    }
    if (adj_to_all) { candidates.push_back(candidate); }
  }
  ZERO_HALF_DEBUG("extend_to_odd_wheel candidates.size=%zu", candidates.size());
  if (candidates.empty()) { return; }

  const f_t candidate_size = static_cast<f_t>(candidates.size());
  const f_t cycle_size_f   = static_cast<f_t>(cycle_vertices.size());
  const f_t adj_set_cost   = 2.0 * static_cast<f_t>(adj_set.size());
  const f_t sort_cost =
    candidate_size > 0.0 ? 2.0 * candidate_size * std::log2(candidate_size + 1.0) : 0.0;
  if (add_work_estimate(adj_set_cost + cycle_size_f * candidate_size + sort_cost,
                        work_estimate,
                        max_work_estimate)) {
    ZERO_HALF_DEBUG("extend_to_odd_wheel work_limit hit pre-sort");
    return;
  }

  sort_candidates_by_reduced_cost(candidates, reduced_costs, num_vars);

  // Candidates are already adjacent to every cycle vertex (filtered above), so
  // growing a clique among them yields centers adjacent to the whole cycle and
  // to each other.
  const f_t adj_check_cost = 5.0;
  greedy_extend_clique(wheel_centers,
                       candidates,
                       graph,
                       adj_check_cost,
                       start_time,
                       time_limit,
                       work_estimate,
                       max_work_estimate);
#ifdef ASSERT_MODE
  // Post-condition: the selected centers must form a clique that is fully
  // adjacent to the cycle — each center adjacent to every cycle vertex and to
  // every other center. This is exactly what makes the m-weighted wheel lift a
  // valid zero-half inequality.
  for (size_t a = 0; a < wheel_centers.size(); ++a) {
    for (const i_t cv : cycle_vertices) {
      cuopt_assert(graph.check_adjacency(wheel_centers[a], cv),
                   "Zero-half wheel center not adjacent to every cycle vertex");
    }
    for (size_t b = a + 1; b < wheel_centers.size(); ++b) {
      cuopt_assert(graph.check_adjacency(wheel_centers[a], wheel_centers[b]),
                   "Zero-half wheel centers must be mutually adjacent (clique)");
    }
  }
#endif
  ZERO_HALF_DEBUG("extend_to_odd_wheel done wheel_centers.size=%zu", wheel_centers.size());
}

}  // namespace

// This function is only used in tests
std::vector<std::vector<int>> find_violated_odd_cycles_for_test(
  const std::vector<std::vector<int>>& adjacency_list,
  const std::vector<double>& x_values,
  double min_violation,
  double time_limit)
{
  const size_t n_vertices = adjacency_list.size();
  if (n_vertices == 0) { return {}; }
  cuopt_assert(x_values.size() == n_vertices, "x_values size mismatch in odd-cycle test helper");

  const int num_local = static_cast<int>(n_vertices);
  std::vector<std::vector<int>> adj_local(n_vertices);
  for (size_t v = 0; v < n_vertices; ++v) {
    adj_local[v].reserve(adjacency_list[v].size());
    for (const int nbr : adjacency_list[v]) {
      cuopt_assert(nbr >= 0 && static_cast<size_t>(nbr) < n_vertices,
                   "Neighbor index out of range in odd-cycle test helper");
      adj_local[v].push_back(nbr);
    }
  }

  double work_estimate           = 0.0;
  const double max_work_estimate = std::numeric_limits<double>::infinity();
  const double start_time        = tic();
  const double cutoff            = 0.5 - min_violation;

  std::vector<std::vector<int>> result;
  std::vector<int> bipartite_path;
  dijkstra_scratch_t<int, double> dijkstra_scratch;

  for (int s = 0; s < num_local; ++s) {
    if (toc(start_time) >= time_limit) { break; }

    double total_weight = 0;
    if (!dijkstra_odd_cycle<int, double>(s,
                                         adj_local,
                                         x_values,
                                         cutoff,
                                         bipartite_path,
                                         total_weight,
                                         &work_estimate,
                                         max_work_estimate,
                                         dijkstra_scratch)) {
      continue;
    }
    if (bipartite_path.size() < 4) { continue; }
    std::vector<int> seq;
    seq.reserve(bipartite_path.size());
    for (const int bv : bipartite_path) {
      seq.push_back(bv % num_local);
    }
    cuopt_assert(seq.front() == seq.back(), "Odd-cycle test helper path endpoints must match");
    seq.pop_back();
    if ((seq.size() % 2) == 0 || seq.size() < 5) { continue; }
    bool simple = true;
    std::unordered_set<int> seen;
    seen.reserve(seq.size());
    for (const int v : seq) {
      if (!seen.insert(v).second) {
        simple = false;
        break;
      }
    }
    if (!simple) { continue; }
    result.push_back(seq);
  }
  return result;
}

template <typename i_t, typename f_t>
bool cut_generation_t<i_t, f_t>::generate_zero_half_cuts(
  const lp_problem_t<i_t, f_t>& lp,
  const simplex_solver_settings_t<i_t, f_t>& settings,
  csr_matrix_t<i_t, f_t>& Arow,
  const std::vector<i_t>& new_slacks,
  const std::vector<variable_type_t>& var_types,
  const std::vector<f_t>& xstar,
  const std::vector<f_t>& reduced_costs,
  variable_bounds_t<i_t, f_t>& variable_bounds,
  f_t start_time)
{
  if (settings.zero_half_cuts == 0) { return true; }
  if (toc(start_time) >= settings.time_limit) { return true; }

  const i_t num_vars = user_problem_.num_cols;
  ZERO_HALF_DEBUG(
    "generate_zero_half_cuts ENTER num_vars=%lld elapsed=%g time_limit=%g xstar.size=%zu "
    "reduced_costs.size=%zu var_types.size=%zu user_problem_.lower.size=%zu "
    "user_problem_.upper.size=%zu user_problem_.var_types.size=%zu lp.num_cols=%lld "
    "sub_cg_.ready=%d sub_cg_.vertices=%zu",
    static_cast<long long>(num_vars),
    static_cast<double>(toc(start_time)),
    static_cast<double>(settings.time_limit),
    xstar.size(),
    reduced_costs.size(),
    var_types.size(),
    user_problem_.lower.size(),
    user_problem_.upper.size(),
    user_problem_.var_types.size(),
    static_cast<long long>(lp.num_cols),
    static_cast<int>(sub_cg_.ready),
    sub_cg_.vertices.size());

  f_t mod2_work_estimate    = 0.0;
  const bool mod2_completed = generate_mod2_zero_half_cuts(cut_pool_,
                                                           lp,
                                                           settings,
                                                           Arow,
                                                           new_slacks,
                                                           var_types,
                                                           xstar,
                                                           variable_bounds,
                                                           start_time,
                                                           mod2_work_estimate);
  if (!mod2_completed) { return true; }

  // The fractional conflict-graph subgraph is built once per cut pass in
  // prepare_fractional_sub_conflict_graph() and remains a complementary
  // odd-cycle / odd-wheel separator. If no conflict graph is available, the
  // general row-parity cuts above are still retained.
  if (!sub_cg_.ready) {
    ZERO_HALF_DEBUG("sub_cg_ not ready, skipping odd-cycle path");
    return true;
  }
  if (sub_cg_.empty_subgraph()) {
    ZERO_HALF_DEBUG("no fractional binary vertices");
    return true;
  }
  if (clique_table_ == nullptr) {
    ZERO_HALF_DEBUG("no clique table available, skipping");
    return true;
  }
  cuopt_assert(sub_cg_.num_vars == num_vars, "generate_zero_half_cuts sub_cg_ num_vars mismatch");
  cuopt_assert(clique_table_->n_variables == num_vars,
               "Zero-half clique table variable count mismatch");
  cuopt_assert(static_cast<size_t>(num_vars) <= xstar.size(), "Zero-half xstar size mismatch");
  cuopt_assert(user_problem_.var_types.size() == static_cast<size_t>(num_vars),
               "Zero-half user problem var_types size mismatch");

  constexpr f_t min_violation = (f_t)1e-6;
  const f_t bound_tol         = settings.primal_tol;
  // shortest path of length >= 0.5 - min_violation cannot yield a violated cut
  const f_t cutoff            = static_cast<f_t>(0.5) - min_violation;
  f_t work_estimate           = 0.0;
  const f_t max_work_estimate = 1e8;

  const std::vector<i_t>& vertices               = sub_cg_.vertices;
  const std::vector<f_t>& weights                = sub_cg_.weights;
  const std::vector<std::vector<i_t>>& adj_local = sub_cg_.adj_local;
  const std::vector<i_t>& vertex_to_local        = sub_cg_.vertex_to_local;
  const i_t num_local                            = sub_cg_.num_local();
  ZERO_HALF_DEBUG("starting separation loop num_local=%lld", static_cast<long long>(num_local));

  sparse_vector_t<i_t, f_t> cut(lp.num_cols, 0);
  f_t cut_rhs = 0.0;
  std::vector<i_t> bipartite_path;
  std::vector<i_t> cycle_vertices;
  std::vector<i_t> wheel_centers;

  i_t cycles_found  = 0;
  i_t cuts_added    = 0;
  i_t added_per_var = 0;
  std::vector<char> already_used(num_local, 0);
  dijkstra_scratch_t<i_t, f_t> dijkstra_scratch;

  for (i_t s = 0; s < num_local; ++s) {
    if (toc(start_time) >= settings.time_limit) { break; }
    if (work_estimate > max_work_estimate) { break; }
    if (already_used[s]) { continue; }
    ZERO_HALF_DEBUG("separation loop s=%lld / %lld",
                    static_cast<long long>(s),
                    static_cast<long long>(num_local));

    f_t total_weight = 0;
    if (!dijkstra_odd_cycle<i_t, f_t>(s,
                                      adj_local,
                                      weights,
                                      cutoff,
                                      bipartite_path,
                                      total_weight,
                                      &work_estimate,
                                      max_work_estimate,
                                      dijkstra_scratch)) {
      continue;
    }
    if (!path_to_odd_cycle<i_t, f_t>(bipartite_path,
                                     vertices,
                                     num_local,
                                     num_vars,
                                     cycle_vertices,
                                     &work_estimate,
                                     max_work_estimate)) {
      continue;
    }
    cycles_found++;
    cuopt_assert(cycle_vertices.size() >= 5 && (cycle_vertices.size() % 2) == 1,
                 "Zero-half separated cycle must be odd with length >= 5");
    // dijkstra_odd_cycle only returns true when the path stays below the
    // half-integer cutoff, the precondition for the cycle to yield a violation.
    cuopt_assert(cutoff <= static_cast<f_t>(0) || total_weight < cutoff,
                 "Zero-half cycle weight must be below cutoff");
    ZERO_HALF_DEBUG("cycle found s=%lld cycle_vertices.size=%zu",
                    static_cast<long long>(s),
                    cycle_vertices.size());

    extend_to_odd_wheel<i_t, f_t>(cycle_vertices,
                                  wheel_centers,
                                  *clique_table_,
                                  reduced_costs,
                                  num_vars,
                                  start_time,
                                  settings.time_limit,
                                  &work_estimate,
                                  max_work_estimate);

    ZERO_HALF_DEBUG("calling build_zero_half_cut cycle=%zu wheel=%zu",
                    cycle_vertices.size(),
                    wheel_centers.size());
    const clique_cut_build_status_t build_status =
      build_zero_half_cut<i_t, f_t>(cycle_vertices,
                                    wheel_centers,
                                    num_vars,
                                    var_types,
                                    user_problem_.lower,
                                    user_problem_.upper,
                                    xstar,
                                    bound_tol,
                                    min_violation,
                                    cut,
                                    cut_rhs,
                                    &work_estimate,
                                    max_work_estimate);
    ZERO_HALF_DEBUG("build_zero_half_cut returned status=%d", static_cast<int>(build_status));
    if (work_estimate > max_work_estimate) { break; }
    if (build_status == clique_cut_build_status_t::INFEASIBLE) {
      ZERO_HALF_DEBUG("infeasible cycle detected, returning false");
      return false;
    }
    if (build_status == clique_cut_build_status_t::CUT_ADDED) {
      // Only violated cuts are worth pooling; build_zero_half_cut promised a
      // violation > min_violation, so re-check it before we commit.
      cuopt_assert(cut_rhs - cut.dot(xstar) > min_violation - bound_tol,
                   "Zero-half cut added to pool must be violated by xstar");
      inequality_t<i_t, f_t> cut_inequality;
      cut_inequality.vector = cut;
      cut_inequality.rhs    = cut_rhs;
      ZERO_HALF_DEBUG(
        "adding cut to pool nz=%zu rhs=%g", cut.i.size(), static_cast<double>(cut_rhs));
      cut_pool_.add_cut(cut_type_t::ZERO_HALF, cut_inequality);
      ZERO_HALF_DEBUG("cut added to pool");
      cuts_added++;
      added_per_var++;
      // mark all CG vertices that participated so we do not re-derive the same
      // cycle from a different source vertex
      for (const i_t v : cycle_vertices) {
        if (v < 0 || v >= 2 * num_vars) {
          ZERO_HALF_DEBUG("mark already_used: cycle v OUT_OF_RANGE v=%lld",
                          static_cast<long long>(v));
          continue;
        }
        const i_t lv = vertex_to_local[v];
        if (lv >= 0 && lv < num_local) { already_used[lv] = 1; }
      }
    }
  }

  ZERO_HALF_DEBUG("generate_zero_half_cuts EXIT cycles=%lld cuts=%lld work=%g",
                  static_cast<long long>(cycles_found),
                  static_cast<long long>(cuts_added),
                  static_cast<double>(work_estimate));
  return true;
}

#ifdef DUAL_SIMPLEX_INSTANTIATE_DOUBLE
template bool cut_generation_t<int, double>::generate_zero_half_cuts(
  const lp_problem_t<int, double>& lp,
  const simplex_solver_settings_t<int, double>& settings,
  csr_matrix_t<int, double>& Arow,
  const std::vector<int>& new_slacks,
  const std::vector<variable_type_t>& var_types,
  const std::vector<double>& xstar,
  const std::vector<double>& reduced_costs,
  variable_bounds_t<int, double>& variable_bounds,
  double start_time);
#endif

}  // namespace cuopt::mathematical_optimization::mip
