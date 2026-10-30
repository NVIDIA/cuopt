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
#include <limits>
#include <unordered_set>
#include <vector>

namespace cuopt::mathematical_optimization::mip {

using simplex::lp_problem_t;
using simplex::simplex_solver_settings_t;
using simplex::variable_type_t;

namespace {

template <typename i_t, typename f_t>
clique_cut_build_status_t build_clique_cut(const std::vector<i_t>& clique_vertices,
                                           i_t num_vars,
                                           const std::vector<variable_type_t>& var_types,
                                           [[maybe_unused]] const std::vector<f_t>& lower_bounds,
                                           [[maybe_unused]] const std::vector<f_t>& upper_bounds,
                                           const std::vector<f_t>& xstar,
                                           f_t bound_tol,
                                           f_t min_violation,
                                           sparse_vector_t<i_t, f_t>& cut,
                                           f_t& cut_rhs,
                                           f_t* work_estimate,
                                           f_t max_work_estimate)
{
  if (clique_vertices.size() < 2) { return clique_cut_build_status_t::NO_CUT; }
  const f_t clique_size = static_cast<f_t>(clique_vertices.size());
  CLIQUE_CUTS_DEBUG("build_clique_cut start clique_size=%lld",
                    static_cast<long long>(clique_vertices.size()));
  const f_t sort_work = clique_size > 0.0 ? 2.0 * clique_size * std::log2(clique_size + 1.0) : 0.0;
  const f_t dot_work  = 2.0 * clique_size;
  const f_t estimated_work = 9.0 * clique_size + sort_work + dot_work;
  if (add_work_estimate(estimated_work, work_estimate, max_work_estimate)) {
    CLIQUE_CUTS_DEBUG("build_clique_cut skip work_limit clique_size=%lld work=%g limit=%g",
                      static_cast<long long>(clique_vertices.size()),
                      work_estimate == nullptr ? -1.0 : static_cast<double>(*work_estimate),
                      static_cast<double>(max_work_estimate));
    return clique_cut_build_status_t::NO_CUT;
  }

  cuopt_assert(num_vars > 0, "Clique cut num_vars must be positive");
  cuopt_assert(static_cast<size_t>(num_vars) <= lower_bounds.size(),
               "Clique cut lower bounds size mismatch");
  cuopt_assert(static_cast<size_t>(num_vars) <= xstar.size(), "Clique cut xstar size mismatch");

  cut.i.clear();
  cut.x.clear();

  // First pass: collect literal/complement occurrences per variable and the
  // set of variables that appear both as themselves and as their complement
  // in the clique. Bounds / var_type sanity checks are folded in here so we
  // touch each clique vertex once.
  std::unordered_set<i_t> seen_original;
  std::unordered_set<i_t> seen_complement;
  std::unordered_set<i_t> complement_pairs;
  seen_original.reserve(clique_vertices.size());
  seen_complement.reserve(clique_vertices.size());
  for (const i_t vertex_idx : clique_vertices) {
    cuopt_assert(vertex_idx >= 0 && vertex_idx < 2 * num_vars, "Clique vertex out of range");
    const i_t var_idx                      = vertex_idx % num_vars;
    const bool complement                  = vertex_idx >= num_vars;
    [[maybe_unused]] const f_t lower_bound = lower_bounds[var_idx];
    [[maybe_unused]] const f_t upper_bound = upper_bounds[var_idx];
    cuopt_assert(var_types[var_idx] != variable_type_t::CONTINUOUS,
                 "Clique contains continuous variable");
    cuopt_assert(lower_bound >= -bound_tol, "Clique variable lower bound below zero");
    cuopt_assert(upper_bound <= 1 + bound_tol, "Clique variable upper bound above one");

    if (complement) {
      cuopt_assert(seen_complement.count(var_idx) == 0, "Duplicate complement in clique");
      if (seen_original.count(var_idx) > 0) { complement_pairs.insert(var_idx); }
      seen_complement.insert(var_idx);
    } else {
      cuopt_assert(seen_original.count(var_idx) == 0, "Duplicate variable in clique");
      if (seen_complement.count(var_idx) > 0) { complement_pairs.insert(var_idx); }
      seen_original.insert(var_idx);
    }
  }

  // >= 2 complement pairs force two distinct variables each into
  // {0} \cap {1} simultaneously => node is LP-infeasible. The caller is
  // expected to short-circuit the rest of cut generation on INFEASIBLE.
  if (complement_pairs.size() >= 2) {
    CLIQUE_CUTS_DEBUG("build_clique_cut infeasible: %lld complement-pairs",
                      static_cast<long long>(complement_pairs.size()));
    return clique_cut_build_status_t::INFEASIBLE;
  }

  // Exactly one complement pair (x + (1-x) = 1) contributes nothing to the
  // sum but forces every other clique member to 0. We drop the paired
  // variable from the support and bump rhs by 1, producing a fixing cut.
  const bool has_pair = complement_pairs.size() == 1;
  i_t num_complements = 0;
  for (const i_t vertex_idx : clique_vertices) {
    const i_t var_idx     = vertex_idx % num_vars;
    const bool complement = vertex_idx >= num_vars;
    if (has_pair && complement_pairs.count(var_idx) > 0) { continue; }
    cut.i.push_back(var_idx);
    cut.x.push_back(complement ? static_cast<f_t>(1.0) : static_cast<f_t>(-1.0));
    if (complement) { num_complements++; }
  }

  if (cut.i.empty()) {
    CLIQUE_CUTS_DEBUG("build_clique_cut no_cut empty support");
    return clique_cut_build_status_t::NO_CUT;
  }

  cut_rhs = has_pair ? static_cast<f_t>(num_complements) : static_cast<f_t>(num_complements - 1);
  cut.sort();

  const f_t dot       = cut.dot(xstar);
  const f_t violation = cut_rhs - dot;
  if (violation > min_violation) {
    CLIQUE_CUTS_DEBUG(
      "build_clique_cut accepted has_pair=%d nz=%lld rhs=%g dot=%g violation=%g threshold=%g "
      "complements=%lld",
      has_pair ? 1 : 0,
      static_cast<long long>(cut.i.size()),
      static_cast<double>(cut_rhs),
      static_cast<double>(dot),
      static_cast<double>(violation),
      static_cast<double>(min_violation),
      static_cast<long long>(num_complements));
    return clique_cut_build_status_t::CUT_ADDED;
  }
  CLIQUE_CUTS_DEBUG(
    "build_clique_cut rejected has_pair=%d nz=%lld rhs=%g dot=%g violation=%g threshold=%g "
    "complements=%lld",
    has_pair ? 1 : 0,
    static_cast<long long>(cut.i.size()),
    static_cast<double>(cut_rhs),
    static_cast<double>(dot),
    static_cast<double>(violation),
    static_cast<double>(min_violation),
    static_cast<long long>(num_complements));
  return clique_cut_build_status_t::NO_CUT;
}

template <typename i_t, typename f_t>
struct bk_bitset_context_t {
  const std::vector<std::vector<uint64_t>>& adj;
  const std::vector<f_t>& weights;
  f_t min_weight;
  i_t max_calls;
  f_t start_time;
  f_t time_limit;
  size_t words;
  f_t* work_estimate;
  f_t max_work_estimate;
  i_t num_calls{0};
  bool work_limit_reached{false};
  bool call_limit_reached{false};
  std::vector<std::vector<i_t>> cliques;

  bool add_work(f_t accesses)
  {
    return add_work_estimate(accesses, work_estimate, max_work_estimate, &work_limit_reached);
  }

  bool over_work_limit() const
  {
    if (work_limit_reached) { return true; }
    if (work_estimate == nullptr) { return false; }
    return *work_estimate > max_work_estimate;
  }

  bool over_call_limit() const { return call_limit_reached || num_calls >= max_calls; }
};

inline size_t bitset_words(size_t n) { return (n + 63) / 64; }

inline bool bitset_any(const std::vector<uint64_t>& bs)
{
  for (const uint64_t word : bs) {
    if (word != 0) { return true; }
  }
  return false;
}

inline void bitset_set(std::vector<uint64_t>& bs, size_t idx)
{
  bs[idx >> 6] |= (uint64_t(1) << (idx & 63));
}

inline void bitset_clear(std::vector<uint64_t>& bs, size_t idx)
{
  bs[idx >> 6] &= ~(uint64_t(1) << (idx & 63));
}

template <typename i_t, typename f_t>
f_t sum_weights_bitset(const std::vector<uint64_t>& bs, const std::vector<f_t>& weights)
{
  f_t sum = 0.0;
  for (size_t w = 0; w < bs.size(); ++w) {
    uint64_t word = bs[w];
    while (word) {
      const int bit    = __builtin_ctzll(word);
      const size_t idx = w * 64 + static_cast<size_t>(bit);
      sum += weights[idx];
      word &= (word - 1);
    }
  }
  return sum;
}

template <typename i_t, typename f_t>
void bron_kerbosch(bk_bitset_context_t<i_t, f_t>& ctx,
                   std::vector<i_t>& R,       // current clique
                   std::vector<uint64_t>& P,  // potential candidates
                   std::vector<uint64_t>& X,  // already in the clique
                   f_t weight_R)
{
  if (ctx.over_work_limit() || ctx.over_call_limit()) { return; }
  if (toc(ctx.start_time) >= ctx.time_limit) { return; }
  ctx.num_calls++;
  // stop the recursion, for perf reasons
  if (ctx.num_calls > ctx.max_calls) {
    ctx.call_limit_reached = true;
    return;
  }
  if (ctx.add_work(static_cast<f_t>(4 * ctx.words))) { return; }

  // if P and X are empty, we are at maximal clique
  if (!bitset_any(P) && !bitset_any(X)) {
    // if the weight is enough, add and exit
    if (weight_R >= ctx.min_weight) {
      ctx.add_work(static_cast<f_t>(R.size()));
      ctx.cliques.push_back(R);
    }
    return;
  }

  const f_t sumP = sum_weights_bitset<i_t, f_t>(P, ctx.weights);
  // check if all P is added to clique, would we exceed the weight?
  if (weight_R + sumP < ctx.min_weight) { return; }

  i_t pivot                   = -1;
  i_t max_deg                 = -1;
  i_t pivot_vertices_examined = 0;
  // pivoting rule according to the highest degree vertex
  // TODO try other pivoting strategies, we can also implement some online learning like MAB
  for (size_t w = 0; w < ctx.words; ++w) {
    // union of P and X
    uint64_t word = P[w] | X[w];
    while (word) {
      pivot_vertices_examined++;
      // least significant set bit idnex
      const int bit = __builtin_ctzll(word);
      // overall vertex index
      const i_t v = static_cast<i_t>(w * 64 + static_cast<size_t>(bit));
      // clear the least significant set bit (v)
      word &= (word - 1);
      i_t count = 0;
      // count the number of neighbors of v in P
      for (size_t k = 0; k < ctx.words; ++k) {
        count += __builtin_popcountll(P[k] & ctx.adj[v][k]);
      }
      // chose the highest degree v as the pivot
      // we choose the highest degree as the pivot to reduce the recursion size
      // later in this function we recurse on the candidate P / N(v)
      // so it is good to maximize P n N(v)
      if (count > max_deg) {
        max_deg = count;
        pivot   = v;
      }
    }
  }
  ctx.add_work(static_cast<f_t>(2 * ctx.words) +
               static_cast<f_t>(pivot_vertices_examined) * static_cast<f_t>(2 * ctx.words));

  std::vector<i_t> candidates;
  candidates.reserve(ctx.weights.size());
  cuopt_assert(pivot >= 0, "Pivot must be valid when P or X is non-empty");
  for (size_t w = 0; w < ctx.words; ++w) {
    // P / N(pivot)
    uint64_t word = P[w] & ~ctx.adj[pivot][w];
    while (word) {
      const int bit = __builtin_ctzll(word);
      const i_t v   = static_cast<i_t>(w * 64 + static_cast<size_t>(bit));
      word &= (word - 1);
      candidates.push_back(v);
    }
  }
  const i_t num_candidates = static_cast<i_t>(candidates.size());
  ctx.add_work(static_cast<f_t>(2 * ctx.words + num_candidates));
  ctx.add_work(static_cast<f_t>(num_candidates) * static_cast<f_t>(7 * ctx.words + 6));
  // note that candidates will include pivot if it is in P
  for (const i_t v : candidates) {
    if (ctx.over_call_limit()) {
      ctx.call_limit_reached = true;
      return;
    }
    if (toc(ctx.start_time) >= ctx.time_limit) { return; }

    R.push_back(v);
    std::vector<uint64_t> P_next(ctx.words, 0);
    std::vector<uint64_t> X_next(ctx.words, 0);
    for (size_t k = 0; k < ctx.words; ++k) {
      P_next[k] = P[k] & ctx.adj[v][k];
      X_next[k] = X[k] & ctx.adj[v][k];
    }

    bron_kerbosch(ctx, R, P_next, X_next, weight_R + ctx.weights[v]);
    if (ctx.over_work_limit()) { return; }
    if (ctx.over_call_limit()) {
      ctx.call_limit_reached = true;
      return;
    }
    R.pop_back();
    bitset_clear(P, static_cast<size_t>(v));
    bitset_set(X, static_cast<size_t>(v));
  }
}

template <typename i_t, typename f_t>
void extend_clique_vertices(std::vector<i_t>& clique_vertices,
                            mip::clique_table_t<i_t, f_t>& graph,
                            const std::vector<f_t>& xstar,
                            const std::vector<f_t>& reduced_costs,
                            i_t num_vars,
                            f_t integer_tol,
                            f_t start_time,
                            f_t time_limit,
                            f_t* work_estimate,
                            f_t max_work_estimate)
{
  if (toc(start_time) >= time_limit) { return; }
  if (clique_vertices.empty()) { return; }
#if DEBUG_CLIQUE_CUTS
  const size_t initial_clique_vertices = clique_vertices.size();
#endif
  CLIQUE_CUTS_DEBUG("extend_clique_vertices start size=%lld",
                    static_cast<long long>(clique_vertices.size()));
  const f_t initial_clique_size = static_cast<f_t>(clique_vertices.size());

  const i_t smallest_degree_var = min_degree_anchor(clique_vertices, graph, start_time, time_limit);
  if (smallest_degree_var < 0) { return; }

  std::unordered_set<i_t> adj_set = graph.get_adj_set_of_var(smallest_degree_var);
  std::unordered_set<i_t> clique_members(clique_vertices.begin(), clique_vertices.end());
  std::vector<i_t> candidates;
  candidates.reserve(adj_set.size());
  // the candidate list if only the integer valued vertices
  for (const i_t& candidate : adj_set) {
    if (toc(start_time) >= time_limit) { return; }
    if (clique_members.count(candidate) != 0) { continue; }
    i_t var_idx = candidate % num_vars;
    f_t value   = candidate >= num_vars ? (1.0 - xstar[var_idx]) : xstar[var_idx];
    if (std::abs(value - std::round(value)) <= integer_tol) { candidates.push_back(candidate); }
  }
  CLIQUE_CUTS_DEBUG("extend_clique_vertices anchor=%lld adj_size=%lld integer_candidates=%lld",
                    static_cast<long long>(smallest_degree_var),
                    static_cast<long long>(adj_set.size()),
                    static_cast<long long>(candidates.size()));
  const f_t candidate_size = static_cast<f_t>(candidates.size());
  const f_t sort_work =
    candidate_size > 0.0 ? 2.0 * candidate_size * std::log2(candidate_size + 1.0) : 0.0;
  const f_t adj_set_build_cost = 2.0 * static_cast<f_t>(adj_set.size());
  const f_t addtl_cliques_scan_cost =
    1.0 + static_cast<f_t>(graph.var_clique_addtl.avg_slice_size());
  const f_t adj_check_cost = 5.0 + addtl_cliques_scan_cost;
  const f_t estimated_preloop_work =
    2.0 * initial_clique_size + adj_set_build_cost + 3.0 * static_cast<f_t>(adj_set.size()) +
    sort_work + 2.0 * candidate_size + addtl_cliques_scan_cost * initial_clique_size +
    addtl_cliques_scan_cost;
  if (add_work_estimate(estimated_preloop_work, work_estimate, max_work_estimate)) {
    CLIQUE_CUTS_DEBUG("extend_clique_vertices skip work_limit work=%g limit=%g",
                      work_estimate == nullptr ? -1.0 : static_cast<double>(*work_estimate),
                      static_cast<double>(max_work_estimate));
    return;
  }

  // sort the candidates by reduced cost.
  // smaller reduce cost disturbs dual simplex less
  // less refactors and less iterations after resolve.
  // it also increases the cut's effectiveness by keeping xstar not disturbed much
  // if it is disturbed too much, the cut might become non-binding
  sort_candidates_by_reduced_cost(candidates, reduced_costs, num_vars);

  // adj_check_cost folds in addtl_cliques_scan_cost so each check_adjacency
  // charges its own addtl scan cost as the clique grows.
  greedy_extend_clique(clique_vertices,
                       candidates,
                       graph,
                       adj_check_cost,
                       start_time,
                       time_limit,
                       work_estimate,
                       max_work_estimate);
  CLIQUE_CUTS_DEBUG("extend_clique_vertices done start=%lld final=%lld added=%lld",
                    static_cast<long long>(initial_clique_vertices),
                    static_cast<long long>(clique_vertices.size()),
                    static_cast<long long>(clique_vertices.size() - initial_clique_vertices));
}

}  // namespace

// This function is only used in tests
std::vector<std::vector<int>> find_maximal_cliques_for_test(
  const std::vector<std::vector<int>>& adjacency_list,
  const std::vector<double>& weights,
  double min_weight,
  int max_calls,
  double time_limit)
{
  const size_t n_vertices = adjacency_list.size();
  if (n_vertices == 0) { return {}; }
  cuopt_assert(weights.size() == n_vertices, "Weights size mismatch in clique test helper");
  cuopt_assert(max_calls > 0, "max_calls must be positive in clique test helper");

  const size_t words = bitset_words(n_vertices);
  std::vector<std::vector<uint64_t>> adj_bitset(n_vertices, std::vector<uint64_t>(words, 0));
  for (size_t v = 0; v < n_vertices; ++v) {
    for (const int& nbr : adjacency_list[v]) {
      cuopt_assert(nbr >= 0 && static_cast<size_t>(nbr) < n_vertices,
                   "Neighbor index out of range in clique test helper");
      bitset_set(adj_bitset[v], static_cast<size_t>(nbr));
    }
  }

  double work_estimate           = 0.0;
  const double max_work_estimate = std::numeric_limits<double>::infinity();
  const double start_time        = tic();

  bk_bitset_context_t<int, double> ctx{adj_bitset,
                                       weights,
                                       min_weight,
                                       max_calls,
                                       start_time,
                                       time_limit,
                                       words,
                                       &work_estimate,
                                       max_work_estimate};

  std::vector<int> R;
  std::vector<uint64_t> P(words, 0);
  std::vector<uint64_t> X(words, 0);
  for (size_t idx = 0; idx < n_vertices; ++idx) {
    bitset_set(P, idx);
  }
  bron_kerbosch<int, double>(ctx, R, P, X, 0.0);
  return ctx.cliques;
}

template <typename i_t, typename f_t>
bool cut_generation_t<i_t, f_t>::generate_clique_cuts(
  const lp_problem_t<i_t, f_t>& lp,
  const simplex_solver_settings_t<i_t, f_t>& settings,
  const std::vector<variable_type_t>& var_types,
  const std::vector<f_t>& xstar,
  const std::vector<f_t>& reduced_costs,
  f_t start_time)
{
  if (settings.clique_cuts == 0) { return true; }
  if (toc(start_time) >= settings.time_limit) { return true; }

  const i_t num_vars = user_problem_.num_cols;
  CLIQUE_CUTS_DEBUG("generate_clique_cuts start num_vars=%lld time_limit=%g elapsed=%g",
                    static_cast<long long>(num_vars),
                    static_cast<double>(settings.time_limit),
                    static_cast<double>(toc(start_time)));

  // The fractional conflict-graph subgraph is built once per cut pass in
  // prepare_fractional_sub_conflict_graph() (called from generate_cuts) and shared with
  // the zero-half cut separator. Skip if the build was unable to produce a
  // useable sub-CG (clique table missing/empty, work/time budget hit, etc.).
  if (!sub_cg_.ready) {
    CLIQUE_CUTS_DEBUG("generate_clique_cuts sub_cg_ not ready, skipping");
    return true;
  }
  if (sub_cg_.empty_subgraph()) {
    CLIQUE_CUTS_DEBUG("generate_clique_cuts no fractional binary vertices");
    return true;
  }
  cuopt_assert(sub_cg_.num_vars == num_vars, "generate_clique_cuts sub_cg_ num_vars mismatch");
  cuopt_assert(static_cast<size_t>(num_vars) <= xstar.size(), "Clique cut xstar size mismatch");
  cuopt_assert(user_problem_.var_types.size() == static_cast<size_t>(num_vars),
               "User problem var_types size mismatch");

  const f_t min_violation = std::max(settings.primal_tol, static_cast<f_t>(1e-6));
  const f_t bound_tol     = settings.primal_tol;
  const f_t min_weight    = 1.0 + min_violation;
  // TODO this can be problem dependent
  const i_t max_calls         = 100000;
  f_t work_estimate           = 0.0;
  const f_t max_work_estimate = 1e8;

  const std::vector<i_t>& vertices               = sub_cg_.vertices;
  const std::vector<f_t>& weights                = sub_cg_.weights;
  const std::vector<std::vector<i_t>>& adj_local = sub_cg_.adj_local;

  CLIQUE_CUTS_DEBUG("generate_clique_cuts fractional subgraph vertices=%lld (literals=%lld)",
                    static_cast<long long>(vertices.size() / 2),
                    static_cast<long long>(vertices.size()));

  const size_t words = bitset_words(vertices.size());
  std::vector<std::vector<uint64_t>> adj_bitset(vertices.size(), std::vector<uint64_t>(words, 0));
  size_t local_adj_entries = 0;
  for (size_t v = 0; v < adj_local.size(); ++v) {
    local_adj_entries += adj_local[v].size();
    for (const i_t neighbor : adj_local[v]) {
      bitset_set(adj_bitset[v], static_cast<size_t>(neighbor));
    }
  }
  work_estimate += static_cast<f_t>(adj_local.size()) + 3.0 * static_cast<f_t>(local_adj_entries);
  if (work_estimate > max_work_estimate) { return true; }
  CLIQUE_CUTS_DEBUG("generate_clique_cuts bitset graph words=%lld local_entries=%lld",
                    static_cast<long long>(words),
                    static_cast<long long>(local_adj_entries));

  bk_bitset_context_t<i_t, f_t> ctx{adj_bitset,
                                    weights,
                                    min_weight,
                                    max_calls,
                                    start_time,
                                    settings.time_limit,
                                    words,
                                    &work_estimate,
                                    max_work_estimate};
  std::vector<i_t> R;
  std::vector<uint64_t> P(words, 0);
  std::vector<uint64_t> X(words, 0);
  for (size_t idx = 0; idx < vertices.size(); ++idx) {
    bitset_set(P, idx);
  }
  work_estimate += 2.0 * static_cast<f_t>(vertices.size());
  if (work_estimate > max_work_estimate) { return true; }
  bron_kerbosch<i_t, f_t>(ctx, R, P, X, 0.0);
  CLIQUE_CUTS_DEBUG(
    "generate_clique_cuts maximal cliques found=%lld bk_calls=%lld work=%g work_limit=%d "
    "call_limit=%d",
    static_cast<long long>(ctx.cliques.size()),
    static_cast<long long>(ctx.num_calls),
    static_cast<double>(work_estimate),
    ctx.over_work_limit() ? 1 : 0,
    ctx.over_call_limit() ? 1 : 0);
  if (ctx.over_call_limit()) { return true; }
  if (ctx.over_work_limit()) { return true; }
  if (toc(start_time) >= settings.time_limit) { return true; }
  if (work_estimate > max_work_estimate) { return true; }

  sparse_vector_t<i_t, f_t> cut(lp.num_cols, 0);
  f_t cut_rhs = 0.0;
#if DEBUG_CLIQUE_CUTS
  size_t candidate_cliques = 0;
  size_t added_cuts        = 0;
  size_t rejected_cliques  = 0;
  size_t extension_gain    = 0;
#endif
  for (std::vector<i_t>& clique_local : ctx.cliques) {
    if (toc(start_time) >= settings.time_limit) { return true; }
#if DEBUG_CLIQUE_CUTS
    candidate_cliques++;
#endif
    std::vector<i_t> clique_vertices;
    clique_vertices.reserve(clique_local.size());
    for (const i_t local_idx : clique_local) {
      clique_vertices.push_back(vertices[local_idx]);
    }
    work_estimate += 3.0 * static_cast<f_t>(clique_local.size());
    if (work_estimate > max_work_estimate) { return true; }
#if DEBUG_CLIQUE_CUTS
    const size_t size_before_extension = clique_vertices.size();
#endif
    extend_clique_vertices<i_t, f_t>(clique_vertices,
                                     *clique_table_,
                                     xstar,
                                     reduced_costs,
                                     num_vars,
                                     settings.integer_tol,
                                     start_time,
                                     settings.time_limit,
                                     &work_estimate,
                                     max_work_estimate);
#if DEBUG_CLIQUE_CUTS
    extension_gain += clique_vertices.size() - size_before_extension;
#endif
    if (work_estimate > max_work_estimate) { return true; }
    if (toc(start_time) >= settings.time_limit) { return true; }
    const clique_cut_build_status_t build_status = build_clique_cut<i_t, f_t>(clique_vertices,
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
    if (work_estimate > max_work_estimate) { return true; }
    if (build_status == clique_cut_build_status_t::INFEASIBLE) {
      settings.log.debug("Detected contradictory variable/complement clique\n");
      CLIQUE_CUTS_DEBUG(
        "generate_clique_cuts infeasible clique detected after processing=%lld cliques",
        static_cast<long long>(candidate_cliques));
      return false;
    }
    if (build_status == clique_cut_build_status_t::CUT_ADDED) {
      inequality_t<i_t, f_t> cut_inequality;
      cut_inequality.vector = cut;
      cut_inequality.rhs    = cut_rhs;
      cut_pool_.add_cut(cut_type_t::CLIQUE, cut_inequality);
#if DEBUG_CLIQUE_CUTS
      added_cuts++;
      CLIQUE_CUTS_DEBUG("generate_clique_cuts added cut nz=%lld rhs=%g clique_size=%lld",
                        static_cast<long long>(cut.i.size()),
                        static_cast<double>(cut_rhs),
                        static_cast<long long>(clique_vertices.size()));
#endif
    }
#if DEBUG_CLIQUE_CUTS
    else {
      rejected_cliques++;
    }
#endif
  }
#if DEBUG_CLIQUE_CUTS
  CLIQUE_CUTS_DEBUG(
    "generate_clique_cuts done candidate_cliques=%lld added=%lld rejected=%lld extension_gain=%lld "
    "final_work=%g",
    static_cast<long long>(candidate_cliques),
    static_cast<long long>(added_cuts),
    static_cast<long long>(rejected_cliques),
    static_cast<long long>(extension_gain),
    static_cast<double>(work_estimate));
#endif
  return true;
}

#ifdef DUAL_SIMPLEX_INSTANTIATE_DOUBLE
template bool cut_generation_t<int, double>::generate_clique_cuts(
  const lp_problem_t<int, double>& lp,
  const simplex_solver_settings_t<int, double>& settings,
  const std::vector<variable_type_t>& var_types,
  const std::vector<double>& xstar,
  const std::vector<double>& reduced_costs,
  double start_time);
#endif

}  // namespace cuopt::mathematical_optimization::mip
