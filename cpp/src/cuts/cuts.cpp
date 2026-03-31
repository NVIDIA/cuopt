/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#include <cuts/cuts.hpp>

#include <dual_simplex/basis_solves.hpp>
#include <dual_simplex/tic_toc.hpp>
#include <mip_heuristics/presolve/conflict_graph/clique_table.cuh>
#include <utilities/macros.cuh>

#include <cstdint>
#include <cstdio>
#include <limits>
#include <unordered_set>

#include <barrier/dense_matrix.hpp>

#include <numeric>
#include <queue>

namespace cuopt::linear_programming::dual_simplex {

namespace {

#define DEBUG_CLIQUE_CUTS 0
#define CHECK_WORKSPACE   0

enum class clique_cut_build_status_t : int8_t { NO_CUT = 0, CUT_ADDED = 1, INFEASIBLE = 2 };

#if DEBUG_CLIQUE_CUTS
#define CLIQUE_CUTS_DEBUG(...)                    \
  do {                                            \
    std::fprintf(stderr, "[DEBUG_CLIQUE_CUTS] "); \
    std::fprintf(stderr, __VA_ARGS__);            \
    std::fprintf(stderr, "\n");                   \
  } while (0)
#else
#define CLIQUE_CUTS_DEBUG(...) \
  do {                         \
  } while (0)
#endif

template <typename i_t, typename f_t>
clique_cut_build_status_t build_clique_cut(const std::vector<i_t>& clique_vertices,
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
  i_t num_complements = 0;
  std::unordered_set<i_t> seen_original;
  std::unordered_set<i_t> seen_complement;
  seen_original.reserve(clique_vertices.size());
  seen_complement.reserve(clique_vertices.size());
  for (const auto vertex_idx : clique_vertices) {
    cuopt_assert(vertex_idx >= 0 && vertex_idx < 2 * num_vars, "Clique vertex out of range");
    const i_t var_idx     = vertex_idx % num_vars;
    const bool complement = vertex_idx >= num_vars;
    const f_t lower_bound = lower_bounds[var_idx];
    const f_t upper_bound = upper_bounds[var_idx];

    cuopt_assert(var_types[var_idx] != variable_type_t::CONTINUOUS,
                 "Clique contains continuous variable");
    cuopt_assert(lower_bound >= -bound_tol, "Clique variable lower bound below zero");
    cuopt_assert(upper_bound <= 1 + bound_tol, "Clique variable upper bound above one");

    // we store the cut in the form of >= 1, for easy violation check with dot product
    // that's why compelements have 1 as coeff and normal vars have -1
    if (complement) {
      if (seen_original.count(var_idx) > 0) {
        // FIXME: this is temporary, fix all the vars of all other vars in the clique
        return clique_cut_build_status_t::NO_CUT;
        CLIQUE_CUTS_DEBUG("build_clique_cut infeasible var=%lld appears as variable and complement",
                          static_cast<long long>(var_idx));
        return clique_cut_build_status_t::INFEASIBLE;
      }
      cuopt_assert(seen_complement.count(var_idx) == 0, "Duplicate complement in clique");
      seen_complement.insert(var_idx);
      num_complements++;
      cut.i.push_back(var_idx);
      cut.x.push_back(1.0);
    } else {
      if (seen_complement.count(var_idx) > 0) {
        // FIXME: this is temporary, fix all the vars of all other vars in the clique
        return clique_cut_build_status_t::NO_CUT;
        CLIQUE_CUTS_DEBUG("build_clique_cut infeasible var=%lld appears as variable and complement",
                          static_cast<long long>(var_idx));
        return clique_cut_build_status_t::INFEASIBLE;
      }
      cuopt_assert(seen_original.count(var_idx) == 0, "Duplicate variable in clique");
      seen_original.insert(var_idx);
      cut.i.push_back(var_idx);
      cut.x.push_back(-1.0);
    }
  }

  if (cut.i.empty()) {
    CLIQUE_CUTS_DEBUG("build_clique_cut no_cut empty support");
    return clique_cut_build_status_t::NO_CUT;
  }

  cut_rhs = static_cast<f_t>(num_complements - 1);
  cut.sort();

  const f_t dot       = cut.dot(xstar);
  const f_t violation = cut_rhs - dot;
  if (violation > min_violation) {
    CLIQUE_CUTS_DEBUG(
      "build_clique_cut accepted nz=%lld rhs=%g dot=%g violation=%g threshold=%g complements=%lld",
      static_cast<long long>(cut.i.size()),
      static_cast<double>(cut_rhs),
      static_cast<double>(dot),
      static_cast<double>(violation),
      static_cast<double>(min_violation),
      static_cast<long long>(num_complements));
    return clique_cut_build_status_t::CUT_ADDED;
  }
  CLIQUE_CUTS_DEBUG(
    "build_clique_cut rejected nz=%lld rhs=%g dot=%g violation=%g threshold=%g complements=%lld",
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
  for (auto word : bs) {
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
  for (auto v : candidates) {
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
                            detail::clique_table_t<i_t, f_t>& graph,
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

  i_t smallest_degree     = std::numeric_limits<i_t>::max();
  i_t smallest_degree_var = -1;
  for (auto v : clique_vertices) {
    if (toc(start_time) >= time_limit) { return; }
    i_t degree = graph.get_degree_of_var(v);
    if (degree < smallest_degree) {
      smallest_degree     = degree;
      smallest_degree_var = v;
    }
  }

  auto adj_set = graph.get_adj_set_of_var(smallest_degree_var);
  std::unordered_set<i_t> clique_members(clique_vertices.begin(), clique_vertices.end());
  std::vector<i_t> candidates;
  candidates.reserve(adj_set.size());
  // the candidate list if only the integer valued vertices
  for (const auto& candidate : adj_set) {
    if (toc(start_time) >= time_limit) { return; }
    if (clique_members.count(candidate) != 0) { continue; }
    i_t var_idx = candidate % num_vars;
    f_t value   = candidate >= num_vars ? (1.0 - xstar[var_idx]) : xstar[var_idx];
    if (std::abs(value - std::round(value)) <= integer_tol) { candidates.push_back(candidate); }
  }
  CLIQUE_CUTS_DEBUG(
    "extend_clique_vertices anchor=%lld degree=%lld adj_size=%lld integer_candidates=%lld",
    static_cast<long long>(smallest_degree_var),
    static_cast<long long>(smallest_degree),
    static_cast<long long>(adj_set.size()),
    static_cast<long long>(candidates.size()));
  const f_t candidate_size = static_cast<f_t>(candidates.size());
  const f_t sort_work =
    candidate_size > 0.0 ? 2.0 * candidate_size * std::log2(candidate_size + 1.0) : 0.0;
  const f_t adj_set_build_cost     = 2.0 * static_cast<f_t>(adj_set.size());
  const f_t adj_check_cost         = 5.0;
  const f_t estimated_preloop_work = 2.0 * initial_clique_size + adj_set_build_cost +
                                     3.0 * static_cast<f_t>(adj_set.size()) + sort_work +
                                     2.0 * candidate_size;
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
  auto reduced_cost = [&](i_t vertex_idx) -> f_t {
    i_t var_idx = vertex_idx % num_vars;
    cuopt_assert(var_idx >= 0 && var_idx < static_cast<i_t>(reduced_costs.size()),
                 "Variable index out of range");
    f_t rc = reduced_costs[var_idx];
    if (!std::isfinite(rc)) { rc = 0.0; }
    return vertex_idx >= num_vars ? -rc : rc;
  };

  std::sort(candidates.begin(), candidates.end(), [&](i_t a, i_t b) {
    return reduced_cost(a) < reduced_cost(b);
  });

  for (const auto candidate : candidates) {
    bool add   = true;
    i_t checks = 0;
    for (const auto v : clique_vertices) {
      checks++;
      if (!graph.check_adjacency(candidate, v)) {
        add = false;
        break;
      }
    }
    if (add_work_estimate(
          adj_check_cost * static_cast<f_t>(checks), work_estimate, max_work_estimate)) {
      break;
    }
    if (add) {
      clique_vertices.push_back(candidate);
      clique_members.insert(candidate);
    }
  }
  CLIQUE_CUTS_DEBUG("extend_clique_vertices done start=%lld final=%lld added=%lld",
                    static_cast<long long>(initial_clique_vertices),
                    static_cast<long long>(clique_vertices.size()),
                    static_cast<long long>(clique_vertices.size() - initial_clique_vertices));
}

}  // namespace

template <typename i_t, typename f_t>
bool rational_coefficients(const std::vector<variable_type_t>& var_types,
                           const inequality_t<i_t, f_t>& inequality,
                           inequality_t<i_t, f_t>& rational_inequality);

template <typename f_t>
bool rational_approximation(f_t x,
                            int64_t max_denominator,
                            int64_t& numerator,
                            int64_t& denominator);

int64_t gcd(const std::vector<int64_t>& integers);

int64_t lcm(const std::vector<int64_t>& integers);

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
    for (const auto& nbr : adjacency_list[v]) {
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
void cut_pool_t<i_t, f_t>::add_cut(cut_type_t cut_type, const inequality_t<i_t, f_t>& cut)
{
  // TODO: Add fast duplicate check and only add if the cut is not already in the pool

  for (i_t p = 0; p < cut.size(); p++) {
    const i_t j = cut.index(p);
    if (j >= original_vars_) {
      settings_.log.printf(
        "Cut has variable %d that is greater than original_vars_ %d\n", j, original_vars_);
      return;
    }
  }

  inequality_t<i_t, f_t> cut_squeezed;
  cut.squeeze(cut_squeezed);
  if (cut_squeezed.size() == 0) {
    settings_.log.printf("Cut has no coefficients\n");
    return;
  }
  cut_storage_.append_row(cut_squeezed.vector);
  rhs_storage_.push_back(cut_squeezed.rhs);
  cut_type_.push_back(cut_type);
  cut_age_.push_back(0);
}

template <typename i_t, typename f_t>
f_t cut_pool_t<i_t, f_t>::cut_distance(i_t row,
                                       const std::vector<f_t>& x,
                                       f_t& cut_violation,
                                       f_t& cut_norm)
{
  const i_t row_start = cut_storage_.row_start[row];
  const i_t row_end   = cut_storage_.row_start[row + 1];
  f_t cut_x           = 0.0;
  f_t dot             = 0.0;
  for (i_t p = row_start; p < row_end; p++) {
    const i_t j         = cut_storage_.j[p];
    const f_t cut_coeff = cut_storage_.x[p];
    cut_x += cut_coeff * x[j];
    dot += cut_coeff * cut_coeff;
  }
  cut_violation      = rhs_storage_[row] - cut_x;
  cut_norm           = std::sqrt(dot);
  const f_t distance = cut_violation / cut_norm;
  return distance;
}

template <typename i_t, typename f_t>
f_t cut_pool_t<i_t, f_t>::cut_density(i_t row)
{
  const i_t row_start     = cut_storage_.row_start[row];
  const i_t row_end       = cut_storage_.row_start[row + 1];
  const i_t cut_nz        = row_end - row_start;
  const i_t original_vars = original_vars_;
  return static_cast<f_t>(cut_nz) / original_vars;
}

template <typename i_t, typename f_t>
f_t cut_pool_t<i_t, f_t>::cut_orthogonality(i_t i, i_t j)
{
  const i_t i_start = cut_storage_.row_start[i];
  const i_t i_end   = cut_storage_.row_start[i + 1];
  const i_t i_nz    = i_end - i_start;
  const i_t j_start = cut_storage_.row_start[j];
  const i_t j_end   = cut_storage_.row_start[j + 1];
  const i_t j_nz    = j_end - j_start;

  f_t dot = sparse_dot(cut_storage_.j.data() + i_start,
                       cut_storage_.x.data() + i_start,
                       i_nz,
                       cut_storage_.j.data() + j_start,
                       cut_storage_.x.data() + j_start,
                       j_nz);

  f_t norm_i = cut_norms_[i];
  f_t norm_j = cut_norms_[j];
  return 1.0 - std::abs(dot) / (norm_i * norm_j);
}

template <typename i_t, typename f_t>
void cut_pool_t<i_t, f_t>::check_for_duplicate_cuts()
{
  // Algorithm from Finding Duplicate Rows in a Linear Programming Model
  // by J. A. Tomlin and J.S. Welch
  // Operations Research Letters Volume 5, Number 1, June 1986
  std::vector<f_t> divisors(cut_storage_.m, 0.0);
  std::vector<i_t> sets(cut_storage_.m, 0);

  csc_matrix_t<i_t, f_t> cut_storage_csc(0, 0, 1);
  cut_storage_.to_compressed_col(cut_storage_csc);
  i_t n = cut_storage_csc.n;
  i_t m = cut_storage_csc.m;

  const i_t sentinel = std::numeric_limits<i_t>::max();

  i_t new_set                        = 1;
  i_t remaining_potential_duplicates = cut_storage_.m;
  for (i_t j = 0; j < n; j++) {
    i_t r0        = -1;
    i_t new_rows  = 0;
    i_t new_set_0 = new_set;
    new_set++;
    const i_t col_start = cut_storage_csc.col_start[j];
    const i_t col_end   = cut_storage_csc.col_start[j + 1];
    for (i_t p = col_start; p < col_end; p++) {
      const i_t r    = cut_storage_csc.i[p];
      const f_t a_rj = cut_storage_csc.x[p];
      const f_t f_r  = divisors[r];
      if (sets[r] == 0) {
        r0          = r;  // To enable use to find this new set later
        sets[r]     = new_set_0;
        divisors[r] = a_rj;
        new_rows++;
      } else if (sets[r] < new_set_0) {
        // Look over indices a_ij with i > r
        for (i_t q = p + 1; q < col_end; q++) {
          const i_t i    = cut_storage_csc.i[q];
          const f_t a_ij = cut_storage_csc.x[q];
          if (sets[i] == sets[r]) {
            // These two rows are currently in the same set
            // Check to see if the coefficients still match
            const f_t f_i     = divisors[i];
            const f_t val     = (a_rj / f_r) * (f_i / a_ij);
            const f_t epsilon = 1e-10;
            if ((val >= 1.0 - epsilon && val <= 1.0 + epsilon)) {
              sets[r] = new_set;
              sets[i] = new_set;
            }
          }
        }
        if (sets[r] >= new_set_0) {  // This is only true if a match was found inside the above loop
          new_set++;
        } else {
          sets[r] = sentinel;
          remaining_potential_duplicates--;
          if (remaining_potential_duplicates == 0) { break; }
        }
      }
    }
    if (remaining_potential_duplicates == 0) { break; }
    if (new_rows == 1) {
      sets[r0] = sentinel;
      remaining_potential_duplicates--;
      if (remaining_potential_duplicates == 0) { break; }
    }
  }

  // The cuts are stored in the form: sum_j d_ij x_j >= rhs_i
  // We now look for cuts that are duplicates of each other and remove them
  std::vector<i_t> cuts_to_remove(m, 0);
  i_t num_cuts_to_remove = 0;
  for (i_t r = 0; r < m; r++) {
    const i_t set_r = sets[r];
    if (set_r > 0 && set_r < sentinel && cuts_to_remove[r] == 0) {
      // This cut has a duplicate
      for (i_t i = r + 1; i < m; i++) {
        if (sets[i] == set_r) {
          const f_t f_r     = divisors[r];
          const f_t f_i     = divisors[i];
          const f_t theta_r = rhs_storage_[r] / f_r;
          const f_t theta_i = rhs_storage_[i] / f_i;
          if (f_r > 0 && f_i > 0) {
            // We have sum_j d_rj / f_r x_j >= rhs_r / f_r = theta_r
            //    and  sum_j d_ij / f_i x_j >= rhs_i / f_i = theta_i
            if (theta_r <= theta_i) {
              // Cut i is either the same or stronger than cut r
              if (cuts_to_remove[r] == 0) { num_cuts_to_remove++; }
              cuts_to_remove[r] = 1;  // Remove row r
            } else {
              // theta_r > theta_i, so cut r is stricly stronger than cut i
              if (cuts_to_remove[i] == 0) { num_cuts_to_remove++; }
              cuts_to_remove[i] = 1;  // Remove row i
            }
          } else if (f_r < 0 && f_i < 0) {
            // We have sum_j d_rj / f_r x_j <= rhs_r / f_r = theta_r
            //    and  sum_j d_ij / f_i x_j <= rhs_i / f_i = theta_i
            if (theta_r >= theta_i) {
              // Cut i is either the same or stronger than cut r
              if (cuts_to_remove[r] == 0) { num_cuts_to_remove++; }
              cuts_to_remove[r] = 1;  // Remove row r
            } else {
              // theta_r < theta_i, so cut r is strictly stronger than cut i
              if (cuts_to_remove[i] == 0) { num_cuts_to_remove++; }
              cuts_to_remove[i] = 1;  // Remove row i
            }
          }
        }
      }
    }
  }

  if (num_cuts_to_remove > 0) {
    settings_.log.debug("Removing %d duplicate cuts\n", num_cuts_to_remove);
    csr_matrix_t<i_t, f_t> new_cut_storage(0, 0, 0);
    cut_storage_.remove_rows(cuts_to_remove, new_cut_storage);
    cut_storage_ = new_cut_storage;
    i_t write    = 0;
    for (i_t i = 0; i < m; i++) {
      if (cuts_to_remove[i] == 0) {
        rhs_storage_[write] = rhs_storage_[i];
        cut_type_[write]    = cut_type_[i];
        cut_age_[write]     = cut_age_[i];
        write++;
      }
    }
    rhs_storage_.resize(write);
    cut_type_.resize(write);
    cut_age_.resize(write);
  }
}

template <typename i_t, typename f_t>
void cut_pool_t<i_t, f_t>::score_cuts(std::vector<f_t>& x_relax)
{
  check_for_duplicate_cuts();
  cut_distances_.resize(cut_storage_.m, 0.0);
  cut_norms_.resize(cut_storage_.m, 0.0);

  const bool verbose = false;
  for (i_t i = 0; i < cut_storage_.m; i++) {
    f_t violation;
    f_t cut_dist      = cut_distance(i, x_relax, violation, cut_norms_[i]);
    cut_distances_[i] = cut_dist <= min_cut_distance_ ? 0.0 : cut_dist;
    if (verbose) {
      settings_.log.printf("Cut %d type %d distance %+e violation %+e cut_norm %e\n",
                           i,
                           static_cast<int>(cut_type_[i]),
                           cut_distances_[i],
                           violation,
                           cut_norms_[i]);
    }
  }

  std::vector<i_t> sorted_indices;
  best_score_last_permutation(cut_distances_, sorted_indices);

  const i_t max_cuts          = 2000;
  const f_t min_orthogonality = settings_.cut_min_orthogonality;
  best_cuts_.reserve(std::min(max_cuts, cut_storage_.m));
  best_cuts_.clear();
  scored_cuts_ = 0;

  if (!sorted_indices.empty()) {
    const i_t i = sorted_indices.back();
    sorted_indices.pop_back();
    best_cuts_.push_back(i);
    scored_cuts_++;
  }

  while (scored_cuts_ < max_cuts && !sorted_indices.empty()) {
    const i_t i = sorted_indices.back();
    sorted_indices.pop_back();

    if (cut_distances_[i] <= min_cut_distance_) { break; }

    f_t cut_ortho            = 1.0;
    const i_t best_cuts_size = best_cuts_.size();
    for (i_t k = 0; k < best_cuts_size; k++) {
      const i_t j = best_cuts_[k];
      cut_ortho   = std::min(cut_ortho, cut_orthogonality(i, j));
    }
    if (cut_ortho >= min_orthogonality) {
      best_cuts_.push_back(i);
      scored_cuts_++;
    }
  }
}

template <typename i_t, typename f_t>
i_t cut_pool_t<i_t, f_t>::get_best_cuts(csr_matrix_t<i_t, f_t>& best_cuts,
                                        std::vector<f_t>& best_rhs,
                                        std::vector<cut_type_t>& best_cut_types)
{
  best_cuts.m = 0;
  best_cuts.n = original_vars_;
  best_cuts.row_start.clear();
  best_cuts.j.clear();
  best_cuts.x.clear();
  best_cuts.row_start.reserve(scored_cuts_ + 1);
  best_cuts.row_start.push_back(0);
  best_rhs.clear();
  best_rhs.reserve(scored_cuts_);
  best_cut_types.clear();
  best_cut_types.reserve(scored_cuts_);

  for (i_t i : best_cuts_) {
    if (cut_distances_[i] <= min_cut_distance_) { continue; }
    sparse_vector_t<i_t, f_t> cut(cut_storage_, i);
    cut.negate();
    best_cuts.append_row(cut);
    best_rhs.push_back(-rhs_storage_[i]);
    best_cut_types.push_back(cut_type_[i]);
  }

  age_cuts();

  return static_cast<i_t>(best_rhs.size());
}

template <typename i_t, typename f_t>
void cut_pool_t<i_t, f_t>::age_cuts()
{
  for (i_t i = 0; i < cut_age_.size(); i++) {
    cut_age_[i]++;
  }
}

template <typename i_t, typename f_t>
void cut_pool_t<i_t, f_t>::drop_cuts()
{
  // TODO: Implement this
}

template <typename i_t, typename f_t>
knapsack_generation_t<i_t, f_t>::knapsack_generation_t(
  const lp_problem_t<i_t, f_t>& lp,
  const simplex_solver_settings_t<i_t, f_t>& settings,
  csr_matrix_t<i_t, f_t>& Arow,
  const std::vector<i_t>& new_slacks,
  const std::vector<variable_type_t>& var_types)
  : is_slack_(lp.num_cols, 0),
    is_complemented_(lp.num_cols, 0),
    is_marked_(lp.num_cols, 0),
    workspace_(lp.num_cols, 0.0),
    complemented_xstar_(lp.num_cols, 0.0),
    settings_(settings)
{
  const bool verbose = false;
  knapsack_constraints_.reserve(lp.num_rows);

  for (i_t j : new_slacks) {
    is_slack_[j] = 1;
  }

  for (i_t i = 0; i < lp.num_rows; i++) {
    inequality_t<i_t, f_t> inequality(Arow, i, lp.rhs[i]);
    inequality_t<i_t, f_t> rational_inequality = inequality;
    if (!rational_coefficients(var_types, inequality, rational_inequality)) { continue; }
    inequality = rational_inequality;

    const i_t row_len = rational_inequality.size();
    if (row_len < 3) { continue; }
    bool is_knapsack = true;
    f_t sum_pos      = 0.0;
    f_t sum_neg      = 0.0;
    for (i_t p = 0; p < row_len; p++) {
      const i_t j = inequality.index(p);
      if (is_slack_[j]) {
        if (inequality.coeff(p) < 0.0) {
          is_knapsack = false;
          break;
        }
        continue;
      }
      const f_t aj = inequality.coeff(p);
      if (var_types[j] != variable_type_t::INTEGER || lp.lower[j] != 0.0 || lp.upper[j] != 1.0) {
        is_knapsack = false;
        break;
      }
      if (aj < 0.0) {
        sum_pos += -aj;
        sum_neg += -aj;
      } else {
        sum_pos += aj;
      }
    }

    if (is_knapsack) {
      const f_t beta = inequality.rhs + sum_neg;
      if (beta > 0.0 && beta <= sum_pos && std::abs(sum_pos / (row_len - 1) - beta) > 1e-3) {
        if (verbose) {
          settings.log.printf(
            "Knapsack constraint %d row len %d beta %e sum_neg %e sum_pos %e sum_pos / (row_len - "
            "1) %e\n",
            i,
            row_len,
            beta,
            sum_neg,
            sum_pos,
            sum_pos / (row_len - 1));
        }
        knapsack_constraints_.push_back(i);
      }
    }
  }

#ifdef PRINT_KNAPSACK_INFO
  i_t num_knapsack_constraints = knapsack_constraints_.size();
  settings.log.printf("Number of knapsack constraints %d\n", num_knapsack_constraints);
#endif
}

template <typename i_t, typename f_t>
i_t knapsack_generation_t<i_t, f_t>::generate_knapsack_cut(
  const lp_problem_t<i_t, f_t>& lp,
  const simplex_solver_settings_t<i_t, f_t>& settings,
  csr_matrix_t<i_t, f_t>& Arow,
  const std::vector<i_t>& new_slacks,
  const std::vector<variable_type_t>& var_types,
  const std::vector<f_t>& xstar,
  i_t knapsack_row,
  inequality_t<i_t, f_t>& cut)
{
  const bool verbose = false;
  // Get the row associated with the knapsack constraint
  inequality_t<i_t, f_t> knapsack_inequality(Arow, knapsack_row, lp.rhs[knapsack_row]);
  inequality_t<i_t, f_t> rational_knapsack_inequality = knapsack_inequality;
  if (!rational_coefficients(var_types, knapsack_inequality, rational_knapsack_inequality)) {
    return -1;
  }
  knapsack_inequality = rational_knapsack_inequality;

  // Given the following knapsack constraint:
  // sum_j a_j x_j <= beta
  //
  // We solve the following separation problem:
  // minimize   sum_j (1 - xstar_j) z_j
  // subject to sum_j a_j z_j > beta
  //            z_j in {0, 1}
  // When z_j = 1, then j is in the cover.
  // Let phi_star be the optimal objective of this problem.
  // We have a violated cover when phi_star < 1.0
  //
  // We convert this problem into a 0-1 knapsack problem
  // maximize     sum_j (1 - xstar_j) zbar_j
  // subject to   sum_j a_j zbar_j <= sum_j a_j - (beta + 1)
  //            zbar_j in {0, 1}
  // where zbar_j = 1 - z_j
  // This problem is in the form of a 0-1 knapsack problem
  // which we can solve with dynamic programming or generate
  // a heuristic solution with a greedy algorithm.

  // Remove the slacks from the inequality
  f_t seperation_rhs = 0.0;
  if (verbose) { settings.log.printf(" Knapsack : "); }
  std::vector<i_t> complemented_variables;
  complemented_variables.reserve(knapsack_inequality.size());
  for (i_t k = 0; k < knapsack_inequality.size(); k++) {
    const i_t j = knapsack_inequality.index(k);
    if (is_slack_[j]) {
      knapsack_inequality.vector.x[k] = 0.0;
    } else {
      const f_t aj = knapsack_inequality.vector.x[k];
      if (aj < 0.0) {
        knapsack_inequality.rhs -= aj;
        knapsack_inequality.vector.x[k] *= -1.0;
        complemented_variables.push_back(j);
        is_complemented_[j] = 1;
      }
      if (verbose) { settings.log.printf(" %g x%d +", knapsack_inequality.vector.x[k], j); }
      seperation_rhs += knapsack_inequality.vector.x[k];
    }
  }
  if (verbose) { settings.log.printf(" <= %g\n", knapsack_inequality.rhs); }
  seperation_rhs -= (knapsack_inequality.rhs + 1);

  if (verbose) {
    settings.log.printf("\t");
    for (i_t k = 0; k < knapsack_inequality.size(); k++) {
      const i_t j = knapsack_inequality.index(k);
      if (!is_slack_[j]) {
        if (std::abs(xstar[j]) > 1e-3) { settings.log.printf("x_relax[%d]= %g ", j, xstar[j]); }
      }
    }
    settings.log.printf("\n");

    settings.log.printf("seperation_rhs %g\n", seperation_rhs);
  }

  if (seperation_rhs <= 0.0) {
    restore_complemented(complemented_variables);
    return -1;
  }

  std::vector<f_t> values;
  values.reserve(knapsack_inequality.size() - 1);
  std::vector<f_t> weights;
  weights.reserve(knapsack_inequality.size() - 1);
  std::vector<i_t> indices;
  indices.reserve(knapsack_inequality.size() - 1);
  f_t objective_constant = 0.0;
  std::vector<i_t> fixed_variables;
  std::vector<f_t> fixed_values;
  const f_t x_tol = 1e-5;
  for (i_t k = 0; k < knapsack_inequality.size(); k++) {
    const i_t j = knapsack_inequality.index(k);
    if (!is_slack_[j]) {
      const f_t xstar_j      = is_complemented_[j] ? 1.0 - xstar[j] : xstar[j];
      complemented_xstar_[j] = xstar_j;
      const f_t vj           = std::min(1.0, std::max(0.0, 1.0 - xstar_j));
      if (xstar_j < x_tol) {
        // if xstar_j is close to 0, then we can fix z to zero
        fixed_variables.push_back(j);
        fixed_values.push_back(0.0);
        seperation_rhs -= knapsack_inequality.vector.x[k];
        // No need to adjust the objective constant
        continue;
      }
      if (xstar_j > 1.0 - x_tol) {
        // if xstar_j is close to 1, then we can fix z to 1
        fixed_variables.push_back(j);
        fixed_values.push_back(1.0);
        // Note seperation rhs is unchanged
        objective_constant += vj;
        continue;
      }
      objective_constant += vj;
      values.push_back(vj);
      weights.push_back(knapsack_inequality.vector.x[k]);
      indices.push_back(j);
    }
  }
  std::vector<f_t> solution;
  solution.resize(values.size());

  if (seperation_rhs <= 0.0) {
    restore_complemented(complemented_variables);
    return -1;
  }

  f_t objective = 0.0;
  if (!values.empty()) {
    if (verbose) { settings.log.printf("Calling solve_knapsack_problem\n"); }

    objective = solve_knapsack_problem(values, weights, seperation_rhs, solution);
  } else {
    solution.clear();
  }
  if (std::isnan(objective)) {
    restore_complemented(complemented_variables);
    return -1;
  }
  if (verbose) {
    settings.log.printf("objective %e objective_constant %e\n", objective, objective_constant);
  }
  f_t seperation_value = -objective + objective_constant;
  if (verbose) { settings.log.printf("seperation_value %e\n", seperation_value); }
  const f_t tol = 1e-6;
  if (seperation_value >= 1.0 - tol) {
    restore_complemented(complemented_variables);
    return -1;
  }

  i_t cover_size = 0;
  for (i_t k = 0; k < solution.size(); k++) {
    if (solution[k] == 0.0) { cover_size++; }
  }
  for (i_t k = 0; k < fixed_values.size(); k++) {
    if (fixed_values[k] == 1.0) { cover_size++; }
  }

  cut.reserve(cover_size);
  cut.clear();

  for (i_t k = 0; k < solution.size(); k++) {
    const i_t j = indices[k];
    if (solution[k] == 0.0) { cut.push_back(j, -1.0); }
  }
  for (i_t k = 0; k < fixed_variables.size(); k++) {
    const i_t j = fixed_variables[k];
    if (fixed_values[k] == 1.0) { cut.push_back(j, -1.0); }
  }
  cut.rhs = -cover_size + 1;

  // The cut is in the form: - sum_{j in cover} x_j >= -cover_size + 1
  // Which is equivalent to: sum_{j in cover} x_j <= cover_size - 1

  // Compute the minimal cover and partition the variables into C1 and C2
  inequality_t<i_t, f_t> minimal_cover_cut(lp.num_cols);
  std::vector<i_t> c1_partition;
  std::vector<i_t> c2_partition;
  minimal_cover_and_partition(
    knapsack_inequality, cut, complemented_xstar_, minimal_cover_cut, c1_partition, c2_partition);

  // Lift the cut
  inequality_t<i_t, f_t> lifted_cut(lp.num_cols);
  lift_knapsack_cut(knapsack_inequality, minimal_cover_cut, c1_partition, c2_partition, lifted_cut);
  lifted_cut.negate();

  // The cut is now in the form:
  // -\sum_{j in C} x_j - \sum_{j in F} alpha_j x_j >= -cover_size + 1
  for (i_t k = 0; k < lifted_cut.size(); k++) {
    const i_t j = lifted_cut.index(k);
    // \sum_{k!=j} d_k x_k + d_j xbar_j >= gamma
    // xbar_j = 1 - x_j
    // \sum_{k!=j} d_k x_k + d_j (1 - x_j) >= gamma
    // \sum_{k!=j} d_k x_k + d_j - d_j x_j >= gamma
    // \sum_{k!=j} d_k x_k  + d_j x_j >= gamma - d_j
    if (is_complemented_[j]) {
      lifted_cut.rhs -= lifted_cut.vector.x[k];
      lifted_cut.vector.x[k] *= -1.0;
    }
  }
  lifted_cut.sort();

  // Verify the cut is violated
  f_t lifted_dot       = lifted_cut.vector.dot(xstar);
  f_t lifted_violation = lifted_dot - lifted_cut.rhs;
  if (verbose) {
    settings.log.printf(
      "Knapsack cut %d lifted violation %e < 0\n", knapsack_row, lifted_violation);
  }

  if (lifted_violation >= -tol) {
    restore_complemented(complemented_variables);
    return -1;
  }

  cut = lifted_cut;
  restore_complemented(complemented_variables);
  return 0;
}

template <typename i_t, typename f_t>
bool knapsack_generation_t<i_t, f_t>::is_minimal_cover(f_t cover_sum,
                                                       f_t beta,
                                                       const std::vector<f_t>& cover_coefficients)
{
  // Check if the cover is minimial
  // A set C is a cover if
  // sum_{j in C} a_j > beta
  // A set C is a minimal cover if
  // sum_{k in C \ {j}} a_k <= beta for all j in C
  bool minimal = true;

  // cover_sum = sum_{j in C} a_j

  // A cover is minimal if cover_sum - a_j <= beta for all j in C

  for (i_t k = 0; k < cover_coefficients.size(); k++) {
    const f_t a_j = cover_coefficients[k];
    if (a_j == 0.0) { continue; }
    if (cover_sum - a_j > beta) {
      minimal = false;
      break;
    }
  }
  return minimal;
}

template <typename i_t, typename f_t>
void knapsack_generation_t<i_t, f_t>::minimal_cover_and_partition(
  const inequality_t<i_t, f_t>& knapsack_inequality,
  const inequality_t<i_t, f_t>& negated_base_cut,
  const std::vector<f_t>& xstar,
  inequality_t<i_t, f_t>& minimal_cover_cut,
  std::vector<i_t>& c1_partition,
  std::vector<i_t>& c2_partition)
{
  // Compute the minimal cover cut
  inequality_t<i_t, f_t> base_cut = negated_base_cut;
  base_cut.negate();

  std::vector<i_t> cover_indicies;
  cover_indicies.reserve(base_cut.size());

  std::vector<f_t> cover_coefficients;
  cover_coefficients.reserve(base_cut.size());

  std::vector<f_t> score;
  score.reserve(base_cut.size());

  for (i_t k = 0; k < knapsack_inequality.size(); k++) {
    const i_t j   = knapsack_inequality.index(k);
    workspace_[j] = knapsack_inequality.coeff(k);
  }

  for (i_t k = 0; k < base_cut.size(); k++) {
    const i_t j       = base_cut.index(k);
    const f_t xstar_j = xstar[j];
    score.push_back((1.0 - xstar_j) / workspace_[j]);
    cover_indicies.push_back(j);
    cover_coefficients.push_back(workspace_[j]);
  }

  f_t cover_sum = std::accumulate(cover_coefficients.begin(), cover_coefficients.end(), 0.0);

  bool is_minimal = is_minimal_cover(cover_sum, knapsack_inequality.rhs, cover_coefficients);

  if (is_minimal) {
    minimal_cover_cut = base_cut;
    return;
  }

  // We don't have a minimal cover. So sort the score from smallest to largest breaking ties by
  // largest to smallest a_j
  std::vector<i_t> permuation(cover_indicies.size());
  std::iota(permuation.begin(), permuation.end(), 0);
  std::sort(permuation.begin(), permuation.end(), [&](i_t a, i_t b) {
    if (score[a] < score[b]) {
      return true;
    } else if (score[a] == score[b]) {
      return cover_coefficients[a] > cover_coefficients[b];
    } else {
      return false;
    }
  });

  const f_t beta = knapsack_inequality.rhs;
  for (i_t k = 0; k < permuation.size(); k++) {
    const i_t h   = permuation[k];
    const f_t a_j = cover_coefficients[h];
    if (cover_sum - a_j > beta) {
      // sum_{k in C} a_k - a_j > beta
      // so sum_{k in C \ {k}} a_k > beta and C \ {k} remains a cover

      cover_sum -= a_j;
      // Set the coefficient to 0 to remove it from the cover
      cover_coefficients[h] = 0.0;

      is_minimal = is_minimal_cover(cover_sum, beta, cover_coefficients);
      if (is_minimal) { break; }
    } else {
      // C \ {j} is no longer a cover.
      continue;
    }
  }

  // Go through and correct cover_indicies and cover_coefficients
  for (i_t k = 0; k < cover_coefficients.size();) {
    if (cover_coefficients[k] == 0.0) {
      cover_indicies[k] = cover_indicies.back();
      cover_indicies.pop_back();
      cover_coefficients[k] = cover_coefficients.back();
      cover_coefficients.pop_back();
    } else {
      k++;
    }
  }

  // We now have a minimal cover cut
  // sum_{j in C} x_j <= |C| - 1
  minimal_cover_cut.vector.i = cover_indicies;
  minimal_cover_cut.vector.x.resize(cover_indicies.size(), 1.0);
  minimal_cover_cut.rhs = cover_coefficients.size() - 1;

  // Now we need to partition the variables into C1 and C2
  // C2 = {j in C | x_j = 1}
  // C1 = C / C2

  const f_t x_tol = 1e-5;
  for (i_t j : cover_indicies) {
    if (xstar[j] > 1.0 - x_tol) {
      c2_partition.push_back(j);
    } else {
      c1_partition.push_back(j);
    }
  }
}

template <typename i_t, typename f_t>
void knapsack_generation_t<i_t, f_t>::lift_knapsack_cut(
  const inequality_t<i_t, f_t>& knapsack_inequality,
  const inequality_t<i_t, f_t>& base_cut,
  const std::vector<i_t>& c1_partition,
  const std::vector<i_t>& c2_partition,
  inequality_t<i_t, f_t>& lifted_cut)
{
  // The base cut is in the form: sum_{j in cover} x_j <= |cover| - 1

  // We will attempt to lift the cut by adding a new variable x_k with k not in C to the base cut
  // so that the cut becomes
  // sum_{j in cover} x_j + alpha_k * x_k <= |cover| - 1
  //
  // We can do this for multiple variables so that in the end the cut becomes
  //
  // sum_{j in cover} x_j + sum_{k in F} alpha_k * x_k <= |cover| - 1

  // Determine which variables are in the knapsack constraint and not in the cover
  std::vector<i_t> marked_variables;
  marked_variables.reserve(knapsack_inequality.size());
  for (i_t k = 0; k < knapsack_inequality.size(); k++) {
    const i_t j = knapsack_inequality.index(k);
    if (!is_marked_[j]) {
      is_marked_[j] = 1;  // is_marked_[j] = 1 for all j in N
      marked_variables.push_back(j);
    }
  }
  for (i_t k = 0; k < base_cut.size(); k++) {
    const i_t j = base_cut.index(k);
    if (is_marked_[j]) {
      is_marked_[j] = 0;  // is_marked_[j] = 1 for all j in N \ C
      // OK to leave marked_variables unchanged as marked_variables will be a superset of all dirty
      // is_marked
    }
  }
  std::vector<i_t> remaining_variables;
  std::vector<i_t> remaining_indices;
  std::vector<f_t> remaining_coefficients;
  remaining_variables.reserve(knapsack_inequality.size());
  remaining_indices.reserve(knapsack_inequality.size());
  remaining_coefficients.reserve(knapsack_inequality.size());

  for (i_t k = 0; k < knapsack_inequality.size(); k++) {
    const i_t j = knapsack_inequality.index(k);
    if (is_marked_[j]) {
      if (is_slack_[j]) { continue; }
      remaining_variables.push_back(j);
      remaining_indices.push_back(k);
      remaining_coefficients.push_back(knapsack_inequality.coeff(k));
    }
  }

  // We start with F = {} and lift remaining variables one by one
  // For a variable k not in C union F, the inequality
  //
  // alpha_k * x_k + sum_{j in C} x_j <= |C| - 1
  // is trivially satisfied when x_k = 0.
  // If x_k = 1, then the inequality will be valid for all alpha_k
  // such that
  // alpha_k +  maximize sum_{j in C} x_j                          <= |C| - 1
  //            subject to a_k + sum_{j in C} a_j x_j <= beta
  //
  // where here we require a_k + sum_{j in C} a_j x_j <= beta so that the inequality
  // is valid for the set { x_j in {0, 1}^(|C| + 1) | sum_{j in C union k} a_j x_j <= beta}
  //
  // Let phi^star_k denote the optimal objective value of the problem
  //
  // maximize sum_{j in C} x_j
  // subject to a_k + sum_{j in C} a_j x_j <= beta
  //             x_j in {0, 1} for all j in C
  // Then alpha_k <= |C| - 1 - phi^star_k
  // and we can set alpha_k = |C| - 1 - phi^star_k
  //
  // We can continue this process for each variable k not in C union F
  //
  // Assume the valid inequality
  // sum_{j in C} x_j + sum_{j in F} alpha_j * x_j <= |C| - 1
  // has been obtained so far. We now add the variable x_k with k not in C union F to the
  // inequality. So we have alpha_k * x_k + sum_{j in C} x_j + sum_{j in F} alpha_j * x_j <= |C| - 1
  //
  // Again, this is trivially satisfied when x_k = 0. And we can determine the max value of alpha_k
  // by solving the 0-1 knapsack problem:
  //
  // maximize sum_{j in C} x_j + sum_{j in F} alpha_j * x_j
  // subject to sum_{j in C} a_j x_j + sum_{j in F} a_j * x_j <= beta - a_k
  //            x_j in {0, 1} for all j in C union F
  //
  // Let phi^star_k denote the optimal objective value of the knapsack problem.
  // The lifted coefficient alpha_k = |C| - 1 - phi^star_k

  // Construct weight and values for C
  std::vector<i_t> values;
  values.reserve(knapsack_inequality.size());

  std::vector<f_t> weights;
  weights.reserve(knapsack_inequality.size());

  for (i_t k = 0; k < knapsack_inequality.size(); k++) {
    const i_t j = knapsack_inequality.index(k);
    if (!is_marked_[j]) {
      // j is in C
      weights.push_back(knapsack_inequality.coeff(k));
      values.push_back(1);
    }
  }

  std::vector<i_t> F;
  std::vector<f_t> alpha;

  std::vector<f_t> solution;

  f_t cover_size = base_cut.size();

  lifted_cut = base_cut;

  // Sort the coefficients such that the largest coefficients are lifted first
  // We will pop the largest coefficients from the back of the permutation
  std::vector<i_t> permutation;
  best_score_last_permutation(remaining_coefficients, permutation);

  while (permutation.size() > 0) {
    const i_t h   = permutation.back();
    const i_t k   = remaining_variables[h];
    const f_t a_k = remaining_coefficients[h];

    f_t capacity = knapsack_inequality.rhs - a_k;

    f_t objective =
      exact_knapsack_problem_integer_values_fraction_values(values, weights, capacity, solution);
    if (std::isnan(objective)) {
      settings_.log.debug("lifting knapsack problem failed\n");
      break;
    }

    f_t alpha_k = std::max(0.0, cover_size - 1.0 - objective);

    if (alpha_k > 0.0) {
      settings_.log.debug("Lifted variable %d with alpha %g\n", k, alpha_k);
      F.push_back(k);
      alpha.push_back(alpha_k);
      values.push_back(static_cast<i_t>(std::round(alpha_k)));
      weights.push_back(a_k);

      lifted_cut.vector.i.push_back(k);
      lifted_cut.vector.x.push_back(alpha_k);
    }

    // Remove the variable from the permutation
    permutation.pop_back();
  }
  // Restore is_marked_
  for (i_t j : marked_variables) {
    is_marked_[j] = 0;
  }
}

template <typename i_t, typename f_t>
f_t knapsack_generation_t<i_t, f_t>::greedy_knapsack_problem(const std::vector<f_t>& values,
                                                             const std::vector<f_t>& weights,
                                                             f_t rhs,
                                                             std::vector<f_t>& solution)
{
  i_t n = weights.size();
  solution.assign(n, 0.0);

  // Build permutation
  std::vector<i_t> perm(n);
  std::iota(perm.begin(), perm.end(), 0);

  std::vector<f_t> ratios;
  ratios.resize(n);
  for (i_t i = 0; i < n; i++) {
    ratios[i] = values[i] / weights[i];
  }

  // Sort by value / weight ratio
  std::sort(perm.begin(), perm.end(), [&](i_t i, i_t j) { return ratios[i] > ratios[j]; });

  // Greedy select items with the best value / weight ratio until the remaining capacity is
  // exhausted
  f_t remaining   = rhs;
  f_t total_value = 0.0;

  for (i_t j : perm) {
    if (weights[j] <= remaining) {
      solution[j] = 1.0;
      remaining -= weights[j];
      total_value += values[j];
    }
  }

  // Best single-item fallback
  f_t best_single_value = 0.0;
  i_t best_single_idx   = -1;

  for (i_t j = 0; j < n; ++j) {
    if (weights[j] <= rhs && values[j] > best_single_value) {
      best_single_value = values[j];
      best_single_idx   = j;
    }
  }

  if (best_single_value > total_value) {
    solution.assign(n, 0.0);
    solution[best_single_idx] = 1.0;
    return best_single_value;
  }

  return total_value;
}

template <typename i_t, typename f_t>
f_t knapsack_generation_t<i_t, f_t>::solve_knapsack_problem(const std::vector<f_t>& values,
                                                            const std::vector<f_t>& weights,
                                                            f_t rhs,
                                                            std::vector<f_t>& solution)
{
  // Solve the knapsack problem
  // maximize sum_{j=0}^n values[j] * solution[j]
  // subject to sum_{j=0}^n weights[j] * solution[j] <= rhs
  // values: values of the items
  // weights: weights of the items
  // return the value of the solution

  // Using approximate dynamic programming

  i_t n         = weights.size();
  f_t objective = std::numeric_limits<f_t>::quiet_NaN();

  // Compute the maximum value
  f_t vmax = *std::max_element(values.begin(), values.end());

  // Check if all the values are integers
  bool all_integers     = true;
  const f_t integer_tol = 1e-5;
  for (i_t j = 0; j < n; j++) {
    if (std::abs(values[j] - std::round(values[j])) > integer_tol) {
      all_integers = false;
      break;
    }
  }

  const bool verbose = false;

  if (verbose) { settings_.log.printf("all_integers %d\n", all_integers); }

  // Compute the scaling factor and comptue the scaled integer values
  f_t scale = 1.0;
  std::vector<i_t> scaled_values(n);
  if (all_integers) {
    for (i_t j = 0; j < n; j++) {
      scaled_values[j] = static_cast<i_t>(std::floor(values[j]));
    }
  } else {
    const f_t epsilon = 0.1;
    scale             = epsilon * vmax / static_cast<f_t>(n);
    if (scale <= 0.0) { return std::numeric_limits<f_t>::quiet_NaN(); }
    if (verbose) {
      settings_.log.printf("scale %g epsilon %g vmax %g n %d\n", scale, epsilon, vmax, n);
    }
    for (i_t i = 0; i < n; ++i) {
      scaled_values[i] = static_cast<i_t>(std::floor(values[i] / scale));
    }
  }

  i_t sum_value     = std::accumulate(scaled_values.begin(), scaled_values.end(), 0);
  const i_t INT_INF = std::numeric_limits<i_t>::max() / 2;
  if (verbose) { settings_.log.printf("sum value %d\n", sum_value); }
  const i_t max_size = 10000;
  if (sum_value <= 0.0 || sum_value >= max_size) {
    if (verbose) {
      settings_.log.printf("sum value %d is negative or too large using greedy solution\n",
                           sum_value);
    }
    return greedy_knapsack_problem(values, weights, rhs, solution);
  }

  solution.assign(n, 0.0);

  // dp(j, v) = minimum weight using first j items to get value v
  dense_matrix_t<i_t, i_t> dp(n + 1, sum_value + 1, INT_INF);
  dense_matrix_t<i_t, uint8_t> take(n + 1, sum_value + 1, 0);
  dp(0, 0) = 0;

  // 4. Dynamic programming
  for (i_t j = 1; j <= n; ++j) {
    for (i_t v = 0; v <= sum_value; ++v) {
      // Do not take item i-1
      dp(j, v) = dp(j - 1, v);

      // Take item j-1 if possible
      if (v >= scaled_values[j - 1]) {
        i_t candidate =
          dp(j - 1, v - scaled_values[j - 1]) + static_cast<i_t>(std::floor(weights[j - 1]));
        if (candidate < dp(j, v)) {
          dp(j, v)   = candidate;
          take(j, v) = 1;
        }
      }
    }
  }

  // 5. Find best achievable value within capacity
  i_t best_value = 0;
  for (i_t v = 0; v <= sum_value; ++v) {
    if (dp(n, v) <= rhs) { best_value = v; }
  }

  // 6. Backtrack to recover solution
  i_t v = best_value;
  for (i_t j = n; j >= 1; --j) {
    if (take(j, v)) {
      solution[j - 1] = 1.0;
      v -= scaled_values[j - 1];
    } else {
      solution[j - 1] = 0.0;
    }
  }

  objective = best_value * scale;
  return objective;
}

template <typename i_t, typename f_t>
f_t knapsack_generation_t<i_t, f_t>::exact_knapsack_problem_integer_values_fraction_values(
  const std::vector<i_t>& values,
  const std::vector<f_t>& weights,
  f_t rhs,
  std::vector<f_t>& solution)
{
  // Solve the knapsack problem
  // maximize sum_{j=0}^n values[j] * solution[j]
  // subject to sum_{j=0}^n weights[j] * solution[j] <= rhs
  // values: values of the items
  // weights: weights of the items
  // return the value of the solution

  const i_t n = weights.size();

  const bool verbose = false;
  i_t sum_value      = std::accumulate(values.begin(), values.end(), 0);
  if (verbose) { settings_.log.printf("sum value %d\n", sum_value); }
  const i_t max_size = 10000;
  if (sum_value <= 0.0 || sum_value >= max_size) {
    if (verbose) { settings_.log.printf("sum value %d is negative or too large\n", sum_value); }
    return std::numeric_limits<f_t>::quiet_NaN();
  }

  solution.assign(n, 0.0);

  // dp(j, v) = minimum weight using first j items to get value v
  dense_matrix_t<i_t, f_t> dp(n + 1, sum_value + 1, inf);
  dense_matrix_t<i_t, uint8_t> take(n + 1, sum_value + 1, 0);
  dp(0, 0) = 0;

  // 4. Dynamic programming
  for (i_t j = 1; j <= n; ++j) {
    for (i_t v = 0; v <= sum_value; ++v) {
      // Do not take item i-1
      dp(j, v) = dp(j - 1, v);

      // Take item j-1 if possible
      if (v >= values[j - 1]) {
        f_t candidate = dp(j - 1, v - values[j - 1]) + weights[j - 1];
        if (candidate < dp(j, v)) {
          dp(j, v)   = candidate;
          take(j, v) = 1;
        }
      }
    }
  }

  // 5. Find best achievable value within capacity
  i_t best_value = 0;
  for (i_t v = 0; v <= sum_value; ++v) {
    if (dp(n, v) <= rhs) { best_value = v; }
  }

  // 6. Backtrack to recover solution
  i_t v = best_value;
  for (i_t j = n; j >= 1; --j) {
    if (take(j, v)) {
      solution[j - 1] = 1.0;
      v -= values[j - 1];
    } else {
      solution[j - 1] = 0.0;
    }
  }

  return f_t(best_value);
}

template <typename i_t, typename f_t>
void cut_generation_t<i_t, f_t>::generate_implied_bound_cuts(
  const lp_problem_t<i_t, f_t>& lp,
  const simplex_solver_settings_t<i_t, f_t>& settings,
  const std::vector<variable_type_t>& var_types,
  const std::vector<f_t>& xstar,
  f_t start_time)
{
  if (probing_implied_bound_.zero_offsets.empty()) { return; }

  const f_t tol      = 1e-4;
  i_t num_cuts       = 0;
  const i_t pib_cols = static_cast<i_t>(probing_implied_bound_.zero_offsets.size()) - 1;
  const i_t n_cols   = std::min(lp.num_cols, pib_cols);

  for (i_t j = 0; j < n_cols; j++) {
    if (var_types[j] == variable_type_t::CONTINUOUS) { continue; }
    const f_t xstar_j = xstar[j];

    // x_j = 0 implications
    const i_t zero_begin = probing_implied_bound_.zero_offsets[j];
    const i_t zero_end   = probing_implied_bound_.zero_offsets[j + 1];
    for (i_t p = zero_begin; p < zero_end; p++) {
      const i_t i = probing_implied_bound_.zero_variables[p];
      if (i == j) { continue; }
      const f_t l_i = lp.lower[i];
      const f_t u_i = lp.upper[i];

      // Tightened upper bound: x_j = 0 implies y_i <= b, where b < u_i
      // Valid inequality: y_i <= b + (u_i - b)*x_j  or  -y_i + (u_i - b)*x_j >= -b
      const f_t b_ub = probing_implied_bound_.zero_upper_bound[p];
      if (b_ub < u_i - tol) {
        const f_t coeff_j = u_i - b_ub;
        const f_t y_i     = xstar[i];
        const f_t lhs_val = -y_i + coeff_j * xstar_j;
        const f_t rhs_val = -b_ub;
        if (lhs_val < rhs_val - tol) {
          inequality_t<i_t, f_t> cut;
          cut.push_back(i, -1.0);
          cut.push_back(j, coeff_j);
          cut.rhs = -b_ub;
          cut_pool_.add_cut(cut_type_t::IMPLIED_BOUND, cut);
          num_cuts++;
        }
      }

      // Tightened lower bound: x_j = 0 implies y_i >= b, where b > l_i
      // Valid inequality: y_i >= b - (b - l_i)*x_j  or  y_i + (b - l_i)*x_j >= b
      const f_t b_lb = probing_implied_bound_.zero_lower_bound[p];
      if (b_lb > l_i + tol) {
        const f_t coeff_j = b_lb - l_i;
        const f_t y_i     = xstar[i];
        const f_t lhs_val = y_i + coeff_j * xstar_j;
        const f_t rhs_val = b_lb;
        if (lhs_val < rhs_val - tol) {
          inequality_t<i_t, f_t> cut;
          cut.push_back(i, 1.0);
          cut.push_back(j, coeff_j);
          cut.rhs = b_lb;
          cut_pool_.add_cut(cut_type_t::IMPLIED_BOUND, cut);
          num_cuts++;
        }
      }
    }

    // x_j = 1 implications
    const i_t one_begin = probing_implied_bound_.one_offsets[j];
    const i_t one_end   = probing_implied_bound_.one_offsets[j + 1];
    for (i_t p = one_begin; p < one_end; p++) {
      const i_t i = probing_implied_bound_.one_variables[p];
      if (i == j) { continue; }
      const f_t l_i = lp.lower[i];
      const f_t u_i = lp.upper[i];

      // Tightened upper bound: x_j = 1 implies y_i <= b, where b < u_i
      // Valid inequality: y_i <= u_i - (u_i - b)*x_j  or  -y_i - (u_i - b)*x_j >= -u_i
      const f_t b_ub = probing_implied_bound_.one_upper_bound[p];
      if (b_ub < u_i - tol) {
        const f_t coeff_j = -(u_i - b_ub);
        const f_t y_i     = xstar[i];
        const f_t lhs_val = -y_i + coeff_j * xstar_j;
        const f_t rhs_val = -u_i;
        if (lhs_val < rhs_val - tol) {
          inequality_t<i_t, f_t> cut;
          cut.push_back(i, -1.0);
          cut.push_back(j, coeff_j);
          cut.rhs = -u_i;
          cut_pool_.add_cut(cut_type_t::IMPLIED_BOUND, cut);
          num_cuts++;
        }
      }

      // Tightened lower bound: x_j = 1 implies y_i >= b, where b > l_i
      // Valid inequality: y_i >= l_i + (b - l_i)*x_j  or  y_i - (b - l_i)*x_j >= l_i
      const f_t b_lb = probing_implied_bound_.one_lower_bound[p];
      if (b_lb > l_i + tol) {
        const f_t coeff_j = -(b_lb - l_i);
        const f_t lhs_val = xstar[i] + coeff_j * xstar_j;
        const f_t rhs_val = l_i;
        if (lhs_val < rhs_val - tol) {
          inequality_t<i_t, f_t> cut;
          cut.push_back(i, 1.0);
          cut.push_back(j, coeff_j);
          cut.rhs = rhs_val;
          cut_pool_.add_cut(cut_type_t::IMPLIED_BOUND, cut);
          num_cuts++;
        }
      }
    }
  }

  if (num_cuts > 0) {
    settings.log.debug("Generated %d implied bounds cuts from probing\n", num_cuts);
  }
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
    f_t cut_start_time = tic();
    generate_gomory_cuts(
      lp, settings, Arow, new_slacks, var_types, basis_update, xstar, basic_list, nonbasic_list);
    f_t cut_generation_time = toc(cut_start_time);
    if (cut_generation_time > 1.0) {
      settings.log.debug("Gomory and CG cut generation time %.2f seconds\n", cut_generation_time);
    }
  }

  // Generate Knapsack cuts
  if (settings.knapsack_cuts != 0) {
    f_t cut_start_time = tic();
    generate_knapsack_cuts(lp, settings, Arow, new_slacks, var_types, xstar, start_time);
    f_t cut_generation_time = toc(cut_start_time);
    if (cut_generation_time > 1.0) {
      settings.log.debug("Knapsack cut generation time %.2f seconds\n", cut_generation_time);
    }
  }

  // Generate MIR and CG cuts
  if (settings.mir_cuts != 0 || settings.strong_chvatal_gomory_cuts != 0) {
    f_t cut_start_time = tic();
    generate_mir_cuts(lp, settings, Arow, new_slacks, var_types, xstar, ystar, variable_bounds);
    f_t cut_generation_time = toc(cut_start_time);
    if (cut_generation_time > 1.0) {
      settings.log.debug("MIR and CG cut generation time %.2f seconds\n", cut_generation_time);
    }
  }

  // Generate Clique cuts (last to give background clique table generation maximum time)
  if (settings.clique_cuts != 0) {
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

  // Generate implied bound cuts
  if (settings.implied_bound_cuts != 0) {
    f_t cut_start_time = tic();
    generate_implied_bound_cuts(lp, settings, var_types, xstar, start_time);
    f_t cut_generation_time = toc(cut_start_time);
    if (cut_generation_time > 1.0) {
      settings.log.debug("Implied bounds cut generation time %.2f seconds\n", cut_generation_time);
    }
  }
  return true;
}

template <typename i_t, typename f_t>
void cut_generation_t<i_t, f_t>::generate_knapsack_cuts(
  const lp_problem_t<i_t, f_t>& lp,
  const simplex_solver_settings_t<i_t, f_t>& settings,
  csr_matrix_t<i_t, f_t>& Arow,
  const std::vector<i_t>& new_slacks,
  const std::vector<variable_type_t>& var_types,
  const std::vector<f_t>& xstar,
  f_t start_time)
{
  if (knapsack_generation_.num_knapsack_constraints() > 0) {
    for (i_t knapsack_row : knapsack_generation_.get_knapsack_constraints()) {
      if (toc(start_time) >= settings.time_limit) { return; }
      inequality_t<i_t, f_t> cut(lp.num_cols);
      i_t knapsack_status = knapsack_generation_.generate_knapsack_cut(
        lp, settings, Arow, new_slacks, var_types, xstar, knapsack_row, cut);
      if (knapsack_status == 0) { cut_pool_.add_cut(cut_type_t::KNAPSACK, cut); }
    }
  }
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

  if (clique_table_ == nullptr && clique_table_future_ != nullptr &&
      clique_table_future_->valid()) {
    CLIQUE_CUTS_DEBUG("generate_clique_cuts signaling background thread and waiting");
    if (signal_extend_) { signal_extend_->store(true, std::memory_order_release); }
    clique_table_        = clique_table_future_->get();
    clique_table_future_ = nullptr;
    if (clique_table_) {
      CLIQUE_CUTS_DEBUG("generate_clique_cuts received clique table first=%lld addtl=%lld",
                        static_cast<long long>(clique_table_->first.size()),
                        static_cast<long long>(clique_table_->addtl_cliques.size()));
    }
  }

  if (clique_table_ == nullptr) {
    CLIQUE_CUTS_DEBUG("generate_clique_cuts no clique table available, skipping");
    return true;
  }
  CLIQUE_CUTS_DEBUG("generate_clique_cuts using clique table first=%lld addtl=%lld",
                    static_cast<long long>(clique_table_->first.size()),
                    static_cast<long long>(clique_table_->addtl_cliques.size()));

  if (clique_table_->first.empty() && clique_table_->addtl_cliques.empty()) {
    CLIQUE_CUTS_DEBUG("generate_clique_cuts empty clique table, nothing to separate");
    return true;
  }

  cuopt_assert(clique_table_->n_variables == num_vars, "Clique table variable count mismatch");
  cuopt_assert(static_cast<size_t>(num_vars) <= xstar.size(), "Clique cut xstar size mismatch");

  const f_t min_violation = std::max(settings.primal_tol, static_cast<f_t>(1e-6));
  const f_t bound_tol     = settings.primal_tol;
  const f_t min_weight    = 1.0 + min_violation;
  // TODO this can be problem dependent
  const i_t max_calls         = 100000;
  f_t work_estimate           = 0.0;
  const f_t max_work_estimate = 1e8;

  cuopt_assert(user_problem_.var_types.size() == static_cast<size_t>(num_vars),
               "User problem var_types size mismatch");

  std::vector<i_t> vertices;
  std::vector<f_t> weights;
  vertices.reserve(num_vars * 2);
  weights.reserve(num_vars * 2);

  // create the sub graph induced by fractional binary variables
  for (i_t j = 0; j < num_vars; ++j) {
    if (user_problem_.var_types[j] == variable_type_t::CONTINUOUS) { continue; }
    const f_t lower_bound = user_problem_.lower[j];
    const f_t upper_bound = user_problem_.upper[j];
    if (lower_bound < -bound_tol || upper_bound > 1 + bound_tol) { continue; }
    const f_t xj = xstar[j];
    if (std::abs(xj - std::round(xj)) <= settings.integer_tol) { continue; }
    vertices.push_back(j);
    weights.push_back(xj);
    vertices.push_back(j + num_vars);
    weights.push_back(1.0 - xj);
  }
  // Coarse loop estimate: variable scans + selected vertex/weight writes
  work_estimate += 4.0 * static_cast<f_t>(num_vars) + 2.0 * static_cast<f_t>(vertices.size());
  if (work_estimate > max_work_estimate) { return true; }

  if (vertices.empty()) {
    CLIQUE_CUTS_DEBUG("generate_clique_cuts no fractional binary vertices");
    return true;
  }
  CLIQUE_CUTS_DEBUG("generate_clique_cuts fractional subgraph vertices=%lld (literals=%lld)",
                    static_cast<long long>(vertices.size() / 2),
                    static_cast<long long>(vertices.size()));

  std::vector<i_t> vertex_to_local(2 * num_vars, -1);
  std::vector<char> in_subgraph(2 * num_vars, 0);
  for (size_t idx = 0; idx < vertices.size(); ++idx) {
    if (toc(start_time) >= settings.time_limit) { return true; }
    const i_t vertex_idx        = vertices[idx];
    vertex_to_local[vertex_idx] = static_cast<i_t>(idx);
    in_subgraph[vertex_idx]     = 1;
  }
  work_estimate += 3.0 * static_cast<f_t>(vertices.size());
  if (work_estimate > max_work_estimate) { return true; }

  std::vector<std::vector<i_t>> adj_local(vertices.size());
  size_t total_adj_entries = 0;
  size_t kept_adj_entries  = 0;
  for (size_t idx = 0; idx < vertices.size(); ++idx) {
    if (toc(start_time) >= settings.time_limit) { return true; }
    i_t vertex_idx = vertices[idx];
    // returns the complement as well
    auto adj_set = clique_table_->get_adj_set_of_var(vertex_idx);
    total_adj_entries += adj_set.size();
    auto& adj = adj_local[idx];
    adj.reserve(adj_set.size());
    for (const auto neighbor : adj_set) {
      if (toc(start_time) >= settings.time_limit) { return true; }
      cuopt_assert(neighbor >= 0 && neighbor < 2 * num_vars, "Neighbor out of range");
      if (!in_subgraph[neighbor]) { continue; }
      i_t local_neighbor = vertex_to_local[neighbor];
      cuopt_assert(local_neighbor >= 0, "Local neighbor out of range");
      adj.push_back(local_neighbor);
    }
    kept_adj_entries += adj.size();
#ifdef ASSERT_MODE
    {
      std::unordered_set<i_t> adj_global;
      adj_global.reserve(adj.size());
      for (const auto neighbor : adj) {
        i_t v = vertices[neighbor];
        cuopt_assert(adj_global.insert(v).second, "Duplicate neighbor in adjacency list");
        i_t complement = (v >= num_vars) ? (v - num_vars) : (v + num_vars);
        cuopt_assert(adj_global.find(complement) == adj_global.end(),
                     "Adjacency list contains complementing variable");
      }
    }
#endif
  }
  work_estimate += static_cast<f_t>(vertices.size()) + static_cast<f_t>(total_adj_entries) +
                   2.0 * static_cast<f_t>(kept_adj_entries);
  if (work_estimate > max_work_estimate) { return true; }
  CLIQUE_CUTS_DEBUG("generate_clique_cuts adjacency raw_entries=%lld kept_entries=%lld",
                    static_cast<long long>(total_adj_entries),
                    static_cast<long long>(kept_adj_entries));

  const size_t words = bitset_words(vertices.size());
  std::vector<std::vector<uint64_t>> adj_bitset(vertices.size(), std::vector<uint64_t>(words, 0));
  size_t local_adj_entries = 0;
  for (size_t v = 0; v < adj_local.size(); ++v) {
    local_adj_entries += adj_local[v].size();
    for (const auto neighbor : adj_local[v]) {
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
  for (auto& clique_local : ctx.cliques) {
    if (toc(start_time) >= settings.time_limit) { return true; }
#if DEBUG_CLIQUE_CUTS
    candidate_cliques++;
#endif
    std::vector<i_t> clique_vertices;
    clique_vertices.reserve(clique_local.size());
    for (auto local_idx : clique_local) {
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
    const auto build_status = build_clique_cut<i_t, f_t>(clique_vertices,
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

template <typename i_t, typename f_t>
void cut_generation_t<i_t, f_t>::generate_mir_cuts(
  const lp_problem_t<i_t, f_t>& lp,
  const simplex_solver_settings_t<i_t, f_t>& settings,
  csr_matrix_t<i_t, f_t>& Arow,
  const std::vector<i_t>& new_slacks,
  const std::vector<variable_type_t>& var_types,
  const std::vector<f_t>& xstar,
  const std::vector<f_t>& ystar,
  variable_bounds_t<i_t, f_t>& variable_bounds)
{
  f_t mir_start_time     = tic();
  constexpr bool verbose = false;
  complemented_mixed_integer_rounding_cut_t<i_t, f_t> complemented_mir(lp, settings, new_slacks);
  strong_cg_cut_t<i_t, f_t> cg(lp, var_types, xstar);

  std::vector<f_t> scores;
  complemented_mir.compute_initial_scores_for_rows(lp, settings, Arow, xstar, ystar, scores);

  // Push all the scores onto the priority queue
  std::priority_queue<std::pair<f_t, i_t>> score_queue;
  for (i_t i = 0; i < lp.num_rows; i++) {
    score_queue.push(std::make_pair(scores[i], i));
  }

  // These data structures are used to track the rows that have been aggregated
  // The invariant is that aggregated_rows is empty and aggregated_mark is all zeros
  // at the beginning of each iteration of the for loop below
  std::vector<i_t> aggregated_rows;
  std::vector<i_t> aggregated_mark(lp.num_rows, 0);

  // Transform the relaxation solution
  std::vector<f_t> transformed_xstar;
  complemented_mir.bound_substitution(lp, variable_bounds, var_types, xstar, transformed_xstar);

  const i_t max_cuts = std::min(lp.num_rows, 100000);
  f_t work_estimate  = 0.0;
  i_t num_cuts       = 0;
  while (num_cuts < max_cuts && !score_queue.empty()) {
    // Get the row with the highest score from the queue
    auto [max_score, i] = score_queue.top();
    score_queue.pop();
    // skip stale score entries
    if (max_score != scores[i]) { continue; }

    // Add the current row to the aggregated set
    aggregated_mark[i] = 1;
    aggregated_rows.push_back(i);

    const i_t row_nz      = Arow.row_length(i);
    const i_t slack       = complemented_mir.slack_cols(i);
    const f_t slack_value = xstar[slack];

    if (max_score <= 0.0) { break; }
    if (work_estimate > 2e9) { break; }

    inequality_t<i_t, f_t> inequality(Arow, i, lp.rhs[i]);
    work_estimate += inequality.size();

    const bool generate_cg_cut = settings.strong_chvatal_gomory_cuts != 0;
    f_t fractional_part_rhs    = fractional_part(inequality.rhs);
    if (generate_cg_cut && fractional_part_rhs > 1e-6 && fractional_part_rhs < (1 - 1e-6)) {
      // Try to generate a CG cut

      inequality_t<i_t, f_t> cg_inequality = inequality;
      if (fractional_part(inequality.rhs) < 0.5) {
        // Multiply by -1 to force the fractional part to be greater than 0.5
        cg_inequality.negate();
      }
      inequality_t<i_t, f_t> cg_cut;
      i_t cg_status =
        cg.generate_strong_cg_cut(lp, settings, var_types, cg_inequality, xstar, cg_cut);
      if (cg_status == 0) { cut_pool_.add_cut(cut_type_t::CHVATAL_GOMORY, cg_cut); }
    }

    if (settings.mir_cuts == 0) { continue; }

    // Remove the slack from the equality to get an inequality
    work_estimate += inequality.size();
    i_t negate_inequality = 1;
    for (i_t k = 0; k < inequality.size(); k++) {
      const i_t j = inequality.index(k);
      if (j == slack) {
        if (inequality.coeff(k) != 1.0) {
          if (inequality.coeff(k) == -1.0 && lp.lower[j] >= 0.0) {
            negate_inequality = 0;
          } else {
            settings.log.debug("Bad slack %d in inequality: aj %e lo %e up %e\n",
                               j,
                               inequality.coeff(k),
                               lp.lower[j],
                               lp.upper[j]);
            negate_inequality = -1;
            break;
          }
        }
        inequality.vector.x[k] = 0.0;
      }
    }

    if (negate_inequality == -1) { continue; }

    if (negate_inequality) {
      // inequaility'*x <= inequality_rhs
      // But for MIR we need: inequality'*x >= inequality_rhs
      inequality.negate();
      work_estimate += inequality.size();
    }
    // We should now have: inequality'*x >= inequality_rhs

    for (i_t k = 0; k < inequality.size(); k++) {
      const i_t j = inequality.index(k);
      if (var_types[j] == variable_type_t::INTEGER) {
        if (transformed_xstar[j] > complemented_mir.new_upper(j) / 2.0) {
          settings.log.printf("!!!!!! j %d transformed x_j %e new_upper_j/2.0 %e\n",
                              j,
                              transformed_xstar[j],
                              complemented_mir.new_upper(j) / 2.0);
        }
      }
    }

    bool add_cut             = false;
    i_t num_aggregated       = 0;
    const i_t max_aggregated = 6;
    f_t min_abs_multiplier   = 1.0;
    f_t max_abs_multiplier   = 1.0;
    work_estimate += lp.num_cols;

    while (!add_cut && num_aggregated < max_aggregated) {
      inequality_t<i_t, f_t> transformed_inequality;
      inequality.squeeze(transformed_inequality);
      work_estimate += transformed_inequality.size();

      complemented_mir.transform_inequality(variable_bounds, var_types, transformed_inequality);
      work_estimate += transformed_inequality.size();

      inequality_t<i_t, f_t> cut;
      bool cut_found = complemented_mir.cut_generation_heuristic(
        transformed_inequality, var_types, transformed_xstar, cut, work_estimate);
      // Note cut is in the transformed variables

      if (cut_found) {
        // Transform back to the original variables
        complemented_mir.untransform_inequality(variable_bounds, var_types, cut);
        complemented_mir.remove_small_coefficients(lp.lower, lp.upper, cut);
        complemented_mir.substitute_slacks(lp, Arow, cut);
        complemented_mir.remove_small_coefficients(lp.lower, lp.upper, cut);
        f_t viol = complemented_mir.compute_violation(cut, xstar);
        work_estimate += 10 * cut.size();
        if (viol > 1e-6) { add_cut = true; }
      }

      if (add_cut) {
        if (settings.mir_cuts != 0) { cut_pool_.add_cut(cut_type_t::MIXED_INTEGER_ROUNDING, cut); }
        break;
      } else {
        // Perform aggregation to try and find a cut

        // Find all the continuous variables in the inequality
        i_t num_continuous    = 0;
        f_t max_off_bound     = 0.0;
        i_t max_off_bound_var = -1;
        for (i_t p = 0; p < inequality.size(); p++) {
          const i_t j  = inequality.index(p);
          const f_t aj = inequality.coeff(p);
          if (aj == 0.0) { continue; }
          if (var_types[j] == variable_type_t::CONTINUOUS) {
            num_continuous++;

            const f_t lb_star_j = complemented_mir.get_lb_star(j);
            const f_t ub_star_j = complemented_mir.get_ub_star(j);
            const f_t off_lower = lb_star_j > -inf ? xstar[j] - lb_star_j : std::abs(xstar[j]);
            const f_t off_upper = ub_star_j < inf ? ub_star_j - xstar[j] : std::abs(xstar[j]);
            const f_t off_bound = std::min(off_lower, off_upper);
            const i_t col_len   = lp.A.col_length(j);
            if (off_bound > max_off_bound && col_len > 1) {
              max_off_bound     = off_bound;
              max_off_bound_var = j;
            }
          }
        }
        work_estimate += 10 * inequality.size();

        if (num_continuous == 0 || max_off_bound < 1e-6) { break; }

        // The variable that is farthest from its bound is used as a pivot
        if (max_off_bound_var >= 0) {
          const i_t col_start          = lp.A.col_start[max_off_bound_var];
          const i_t col_end            = lp.A.col_start[max_off_bound_var + 1];
          const i_t col_len            = lp.A.col_length(max_off_bound_var);
          const i_t max_potential_rows = col_len;
          if (col_len > 1) {
            std::vector<i_t> potential_rows;
            potential_rows.reserve(col_len);

            const f_t threshold = 1e-4;
            for (i_t q = col_start; q < col_end; q++) {
              const i_t i   = lp.A.i[q];
              const f_t val = lp.A.x[q];
              // Can't use rows that have already been aggregated
              if (std::abs(val) > threshold && !aggregated_mark[i]) { potential_rows.push_back(i); }
              if (potential_rows.size() >= max_potential_rows) { break; }
            }
            work_estimate += 5 * (col_end - col_start);

            bool did_aggregate = false;
            while (!potential_rows.empty()) {
              const i_t pivot_row =
                *std::max_element(potential_rows.begin(), potential_rows.end(), [&](i_t a, i_t b) {
                  return scores[a] < scores[b];
                });
              work_estimate += potential_rows.size();

              inequality_t<i_t, f_t> pivot_row_inequality(Arow, pivot_row, lp.rhs[pivot_row]);
              work_estimate += pivot_row_inequality.size();
              // Save inequality before combine_rows mutates it, so we can restore on rejection
              inequality_t<i_t, f_t> saved_inequality = inequality;
              f_t multiplier                          = complemented_mir.combine_rows(
                lp, Arow, max_off_bound_var, pivot_row_inequality, inequality);
              if (max_abs_multiplier / std::abs(multiplier) > 10000 ||
                  std::abs(multiplier) / min_abs_multiplier > 10000) {
                inequality = saved_inequality;
                // Erase the pivot row from the potential rows
                potential_rows.erase(
                  std::remove(potential_rows.begin(), potential_rows.end(), pivot_row),
                  potential_rows.end());
                continue;
              }
              max_abs_multiplier = std::max(max_abs_multiplier, std::abs(multiplier));
              min_abs_multiplier = std::min(min_abs_multiplier, std::abs(multiplier));
              aggregated_rows.push_back(pivot_row);
              aggregated_mark[pivot_row] = 1;
              work_estimate += inequality.size() + pivot_row_inequality.size();
              did_aggregate = true;
              break;
            }

            if (!did_aggregate) {
              // No potential rows to aggregate
              break;
            }
          }
        }
        num_aggregated++;  // Always increase so the loop terminates
      }
    }

    if (add_cut) {
      // We were successful in generating a cut.

      // Set the score of the aggregated rows to a lower value
      for (i_t row : aggregated_rows) {
        scores[row] = 0.99 * scores[row];
        score_queue.push(std::make_pair(scores[row], row));
      }
      work_estimate += aggregated_rows.size() * std::log2(score_queue.size());
    }

    // Clear the aggregated mark
    work_estimate += 2 * aggregated_rows.size();
    for (i_t row : aggregated_rows) {
      aggregated_mark[row] = 0;
    }
    // Clear the aggregated rows
    aggregated_rows.clear();

    scores[i] = 0.0;
    score_queue.push(std::make_pair(scores[i], i));
    work_estimate += std::log2(std::max(1, static_cast<i_t>(score_queue.size())));
  }
}

template <typename i_t, typename f_t>
void cut_generation_t<i_t, f_t>::generate_gomory_cuts(
  const lp_problem_t<i_t, f_t>& lp,
  const simplex_solver_settings_t<i_t, f_t>& settings,
  csr_matrix_t<i_t, f_t>& Arow,
  const std::vector<i_t>& new_slacks,
  const std::vector<variable_type_t>& var_types,
  basis_update_mpf_t<i_t, f_t>& basis_update,
  const std::vector<f_t>& xstar,
  const std::vector<i_t>& basic_list,
  const std::vector<i_t>& nonbasic_list)
{
  tableau_equality_t<i_t, f_t> tableau(lp, basis_update, nonbasic_list);
  mixed_integer_gomory_cut_t<i_t, f_t> gomory_cut;
  complemented_mixed_integer_rounding_cut_t<i_t, f_t> complemented_mir(lp, settings, new_slacks);
  simplex_solver_settings_t<i_t, f_t> variable_settings = settings;
  variable_settings.sub_mip                             = 1;
  variable_bounds_t<i_t, f_t> variable_bounds(lp, variable_settings, var_types, Arow, new_slacks);
  strong_cg_cut_t<i_t, f_t> cg(lp, var_types, xstar);
  std::vector<f_t> transformed_xstar;
  complemented_mir.bound_substitution(lp, variable_bounds, var_types, xstar, transformed_xstar);

  for (i_t i = 0; i < lp.num_rows; i++) {
    inequality_t<i_t, f_t> inequality(lp.num_cols);
    const i_t j = basic_list[i];
    if (var_types[j] != variable_type_t::INTEGER) { continue; }
    const f_t x_j = xstar[j];
    if (fractional_part(x_j) < 0.05 || fractional_part(x_j) > 0.95) { continue; }

    i_t tableau_status = tableau.generate_base_equality(
      lp, settings, Arow, var_types, basis_update, xstar, basic_list, nonbasic_list, i, inequality);
    if (tableau_status == 0) {
      // Generate a CG cut
      const bool generate_cg_cut = settings.strong_chvatal_gomory_cuts != 0;
      if (generate_cg_cut) {
        // Try to generate a CG cut
        inequality_t<i_t, f_t> cg_inequality = inequality;
        if (fractional_part(inequality.rhs) < 0.5) {
          // Multiply by -1 to force the fractional part to be greater than 0.5
          cg_inequality.negate();
        }
        inequality_t<i_t, f_t> cg_cut(lp.num_cols);
        i_t cg_status =
          cg.generate_strong_cg_cut(lp, settings, var_types, cg_inequality, xstar, cg_cut);
        if (cg_status == 0) { cut_pool_.add_cut(cut_type_t::CHVATAL_GOMORY, cg_cut); }
      }

      if (settings.mixed_integer_gomory_cuts == 0) { continue; }

      // Transform the inequality
      inequality_t<i_t, f_t> transformed_inequality = inequality;
      complemented_mir.transform_inequality(variable_bounds, var_types, transformed_inequality);

      // Generate a MIR cut from the transformed inequality
      inequality_t<i_t, f_t> cut_A_float(lp.num_cols);
      bool cut_ok = complemented_mir.generate_cut_nonnegative_maintain_indicies(
        transformed_inequality, var_types, cut_A_float);

      // Transform the cut back to the original variables
      complemented_mir.untransform_inequality(variable_bounds, var_types, cut_A_float);
      complemented_mir.remove_small_coefficients(lp.lower, lp.upper, cut_A_float);

      inequality_t<i_t, f_t> cut_A(lp.num_cols);
      if (cut_ok) { cut_ok = rational_coefficients(var_types, cut_A_float, cut_A); }

      // See if the inequality is violated by the original relaxation solution
      f_t cut_A_violation = complemented_mir.compute_violation(cut_A, xstar);
      bool A_valid        = false;
      f_t cut_A_distance  = 0.0;
      if (cut_ok && cut_A_violation > 1e-6) {
        if (cut_A.size() == 0) { continue; }
        complemented_mir.substitute_slacks(lp, Arow, cut_A);
        complemented_mir.remove_small_coefficients(lp.lower, lp.upper, cut_A);
        if (cut_A.size() == 0) {
          A_valid = false;
        } else {
          // Check that the cut is violated
          f_t dot      = cut_A.vector.dot(xstar);
          f_t cut_norm = cut_A.vector.norm2_squared();
          if (dot >= cut_A.rhs) { continue; }
          cut_A_distance = (cut_A.rhs - dot) / std::sqrt(cut_norm);
          A_valid        = true;
        }
      }

      // Negate the base inequality
      inequality.negate();

      inequality_t<i_t, f_t> cut_B_float(lp.num_cols);

      transformed_inequality = inequality;
      complemented_mir.transform_inequality(variable_bounds, var_types, transformed_inequality);

      cut_ok = complemented_mir.generate_cut_nonnegative_maintain_indicies(
        transformed_inequality, var_types, cut_B_float);
      // Transform the cut back to the original variables
      complemented_mir.untransform_inequality(variable_bounds, var_types, cut_B_float);
      complemented_mir.remove_small_coefficients(lp.lower, lp.upper, cut_B_float);

      inequality_t<i_t, f_t> cut_B(lp.num_cols);
      if (cut_ok) { cut_ok = rational_coefficients(var_types, cut_B_float, cut_B); }

      bool B_valid        = false;
      f_t cut_B_distance  = 0.0;
      f_t cut_B_violation = complemented_mir.compute_violation(cut_B, xstar);
      if (cut_ok && cut_B_violation > 1e-6) {
        if (cut_B.size() == 0) { continue; }
        complemented_mir.substitute_slacks(lp, Arow, cut_B);
        complemented_mir.remove_small_coefficients(lp.lower, lp.upper, cut_B);
        if (cut_B.size() == 0) {
          B_valid = false;
        } else {
          // Check that the cut is violated
          f_t dot      = cut_B.vector.dot(xstar);
          f_t cut_norm = cut_B.vector.norm2_squared();
          if (dot >= cut_B.rhs) { continue; }
          cut_B_distance = (cut_B.rhs - dot) / std::sqrt(cut_norm);
          B_valid        = true;
        }
      }

      if ((cut_A_distance > cut_B_distance) && A_valid) {
        cut_pool_.add_cut(cut_type_t::MIXED_INTEGER_GOMORY, cut_A);
      } else if (B_valid) {
        cut_pool_.add_cut(cut_type_t::MIXED_INTEGER_GOMORY, cut_B);
      }
    }
  }
}

template <typename i_t, typename f_t>
i_t tableau_equality_t<i_t, f_t>::generate_base_equality(
  const lp_problem_t<i_t, f_t>& lp,
  const simplex_solver_settings_t<i_t, f_t>& settings,
  csr_matrix_t<i_t, f_t>& Arow,
  const std::vector<variable_type_t>& var_types,
  basis_update_mpf_t<i_t, f_t>& basis_update,
  const std::vector<f_t>& xstar,
  const std::vector<i_t>& basic_list,
  const std::vector<i_t>& nonbasic_list,
  i_t i,
  inequality_t<i_t, f_t>& inequality)
{
  // Let's look for Gomory cuts
  const i_t j = basic_list[i];
  if (var_types[j] != variable_type_t::INTEGER) { return -1; }
  const f_t x_j = xstar[j];
  if (std::abs(x_j - std::round(x_j)) < settings.integer_tol) { return -1; }
#ifdef PRINT_CUT_INFO
  settings_.log.printf("Generating cut for variable %d relaxed value %e row %d\n", j, x_j, i);
#endif

  // Solve B^T u_bar = e_i
  sparse_vector_t<i_t, f_t> e_i(lp.num_rows, 1);
  e_i.i[0] = i;
  e_i.x[0] = 1.0;
  sparse_vector_t<i_t, f_t> u_bar(lp.num_rows, 0);
  basis_update.b_transpose_solve(e_i, u_bar);

#ifdef CHECK_B_TRANSPOSE_SOLVE
  std::vector<f_t> u_bar_dense(lp.num_rows);
  u_bar.to_dense(u_bar_dense);

  std::vector<f_t> BTu_bar(lp.num_rows);
  b_transpose_multiply(lp, basic_list, u_bar_dense, BTu_bar);
  for (i_t k = 0; k < lp.num_rows; k++) {
    if (k == i) {
      settings.log.printf("BTu_bar %d error %e\n", k, std::abs(BTu_bar[k] - 1.0));
      if (std::abs(BTu_bar[k] - 1.0) > 1e-10) {
        settings.log.printf("BTu_bar[%d] = %e i %d\n", k, BTu_bar[k], i);
        assert(false);
      }
    } else {
      settings.log.printf("BTu_bar %d error %e\n", k, std::abs(BTu_bar[k]));
      if (std::abs(BTu_bar[k]) > 1e-10) {
        settings.log.printf("BTu_bar[%d] = %e i %d\n", k, BTu_bar[k], i);
        assert(false);
      }
    }
  }
#endif

  // Compute a_bar = N^T u_bar
  // TODO: This is similar to a function in phase2 of dual simplex. See if it can be reused.
  const i_t nz_ubar = u_bar.i.size();
  std::vector<i_t> abar_indices;
  abar_indices.reserve(nz_ubar);
  for (i_t k = 0; k < nz_ubar; k++) {
    const i_t ii        = u_bar.i[k];
    const f_t u_bar_i   = u_bar.x[k];
    const i_t row_start = Arow.row_start[ii];
    const i_t row_end   = Arow.row_start[ii + 1];
    for (i_t p = row_start; p < row_end; p++) {
      const i_t jj = Arow.j[p];
      if (nonbasic_mark_[jj] == 1) {
        const f_t val    = u_bar_i * Arow.x[p];
        const f_t y      = val - c_workspace_[jj];
        const f_t t      = x_workspace_[jj] + y;
        c_workspace_[jj] = (t - x_workspace_[jj]) - y;
        x_workspace_[jj] = t;
        if (!x_mark_[jj]) {
          x_mark_[jj] = 1;
          abar_indices.push_back(jj);
        }
      }
    }
  }
  // TODO: abar has lots of small coefficients. Double check that
  // we do not accidently create a base (in)equality
  // that cuts off an integer solution, when we drop the small coefficients.

  i_t small_coeff              = 0;
  const f_t drop_tol           = 1e-12;
  const bool drop_coefficients = true;
  sparse_vector_t<i_t, f_t> a_bar(lp.num_cols, 0);
  a_bar.i.reserve(abar_indices.size() + 1);
  a_bar.x.reserve(abar_indices.size() + 1);
  for (i_t k = 0; k < abar_indices.size(); k++) {
    const i_t jj = abar_indices[k];
    if (drop_coefficients && std::abs(x_workspace_[jj]) < drop_tol) {
      small_coeff++;
    } else {
      a_bar.i.push_back(jj);
      a_bar.x.push_back(x_workspace_[jj]);
    }
  }
  const bool verbose = false;
  if (verbose && small_coeff > 0) { settings.log.printf("Small coeff dropped %d\n", small_coeff); }

  // Clear the workspace
  for (i_t jj : abar_indices) {
    x_workspace_[jj] = 0.0;
    x_mark_[jj]      = 0;
    c_workspace_[jj] = 0.0;
  }
  abar_indices.clear();

  // We should now have the base inequality
  // x_j + a_bar^T x_N >= b_bar_i
  // We add x_j into a_bar so that everything is in a single sparse_vector_t
  a_bar.i.push_back(j);
  a_bar.x.push_back(1.0);

  // Check that the tableau equality is satisfied
  const f_t tableau_tol = 1e-6;
  f_t a_bar_dot_xstar   = a_bar.dot(xstar);
  if (std::abs(a_bar_dot_xstar - b_bar_[i]) > tableau_tol) {
    settings.log.debug("bad tableau equality. error %e\n", std::abs(a_bar_dot_xstar - b_bar_[i]));
    return -1;
  }

  // We have that x_j + a_bar^T x_N == b_bar_i
  // So x_j + a_bar^T x_N >= b_bar_i
  // And x_j + a_bar^T x_N <= b_bar_i
  // Or -x_j - a_bar^T x_N >= -b_bar_i

  // Skip cuts that are shallow
  const f_t shallow_tol = 1e-2;
  if (std::abs(x_j - std::round(x_j)) < shallow_tol) {
    // Skip cuts where integer variable has small fractional part
    return -1;
  }

  const f_t f_val = b_bar_[i] - std::floor(b_bar_[i]);
  if (f_val < 0.01 || f_val > 0.99) {
    // Skip cuts with rhs has small fractional part
    return -1;
  }

#ifdef PRINT_BASE_INEQUALITY
  // Print out the base inequality
  for (i_t k = 0; k < a_bar.i.size(); k++) {
    const i_t jj = a_bar.i[k];
    const f_t aj = a_bar.x[k];
    settings_.log.printf("a_bar[%d] = %e\n", k, aj);
  }
  settings_.log.printf("b_bar[%d] = %e\n", i, b_bar[i]);
#endif

  inequality.vector = a_bar;
  inequality.rhs    = b_bar_[i];

  return 0;
}

template <typename f_t>
bool rational_approximation(f_t x,
                            int64_t max_denominator,
                            int64_t& numerator,
                            int64_t& denominator)
{
  int64_t a, p0 = 0, q0 = 1, p1 = 1, q1 = 0;
  f_t val       = x;
  bool negative = false;

  if (x < 0) {
    negative = true;
    val      = -val;
  }

  while (1) {
    a = (int64_t)std::floor(val);
    if (a < 0 || a > INT64_MAX) { return false; }  // Protect against overflow
    int64_t p2 = a * p1 + p0;
    int64_t q2 = a * q1 + q0;
    if (q2 > max_denominator) { break; }
    p0 = p1;
    q0 = q1;
    p1 = p2;
    q1 = q2;

    f_t rem = val - a;
    if (rem < 1e-14) { break; }
    val = 1.0 / rem;
  }

  numerator   = negative ? -p1 : p1;
  denominator = q1;

  f_t approx = static_cast<f_t>(numerator) / static_cast<f_t>(denominator);
  f_t err    = std::abs(approx - x);
  return err <= 1e-14;
}

template <typename i_t, typename f_t>
bool rational_coefficients(const std::vector<variable_type_t>& var_types,
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
  if (scalar < 0) { return false; }
  if (std::abs(scalar) > 1000) { return false; }

  rational_inequality.scale(scalar);

  return true;
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
variable_bounds_t<i_t, f_t>::variable_bounds_t(const lp_problem_t<i_t, f_t>& lp,
                                               const simplex_solver_settings_t<i_t, f_t>& settings,
                                               const std::vector<variable_type_t>& var_types,
                                               const csr_matrix_t<i_t, f_t>& Arow,
                                               const std::vector<i_t>& new_slacks)
  : upper_offsets(lp.num_cols + 1, 0),
    lower_offsets(lp.num_cols + 1, 0),
    upper_activities_(lp.num_rows, 0.0),
    lower_activities_(lp.num_rows, 0.0),
    num_pos_inf_(lp.num_rows, 0),
    num_neg_inf_(lp.num_rows, 0)
{
  if (settings.sub_mip) {
    return;  // Don't compute the variable upper/lower bounds inside sub-MIP
  }
  f_t start_time = tic();

  // Construct the slack map
  slack_map_.resize(lp.num_rows, -1);
  std::vector<f_t> slack_coeff(lp.num_rows, 0.0);
  for (i_t j : new_slacks) {
    const i_t col_start = lp.A.col_start[j];
    const i_t col_end   = lp.A.col_start[j + 1];
    const i_t col_len   = col_end - col_start;
    assert(col_len == 1);
    const i_t i    = lp.A.i[col_start];
    slack_map_[i]  = j;
    slack_coeff[i] = lp.A.x[col_start];
  }

  // The constraints are in the form:
  // sum_j a_j x_j + sigma * slack = beta

  std::vector<i_t> num_integer_in_row(lp.num_rows, 0);
  // Compute the upper activities of the constraints
  for (i_t i = 0; i < lp.num_rows; i++) {
    const i_t row_start   = Arow.row_start[i];
    const i_t row_end     = Arow.row_start[i + 1];
    const i_t slack_index = slack_map_[i];
    f_t activity          = 0.0;
    for (i_t p = row_start; p < row_end; p++) {
      const i_t j = Arow.j[p];
      if (j == slack_index) { continue; }
      const f_t aj = Arow.x[p];
      const f_t uj = lp.upper[j];
      const f_t lj = lp.lower[j];

      if (aj > 0.0) {
        if (uj < inf) {
          activity += aj * uj;
        } else {
          num_pos_inf_[i]++;
        }
      } else {  // a_j < 0.0
        if (lj > -inf) {
          activity += aj * lj;
        } else {
          num_pos_inf_[i]++;
        }
      }

      if (var_types[j] == variable_type_t::INTEGER) { num_integer_in_row[i]++; }
    }
    upper_activities_[i] = activity;
  }

  // Compute the lower activities of the constraints
  for (i_t i = 0; i < lp.num_rows; i++) {
    const i_t row_start   = Arow.row_start[i];
    const i_t row_end     = Arow.row_start[i + 1];
    const i_t slack_index = slack_map_[i];
    f_t activity          = 0.0;
    for (i_t p = row_start; p < row_end; p++) {
      const i_t j = Arow.j[p];
      if (j == slack_index) { continue; }
      const f_t aj = Arow.x[p];
      const f_t uj = lp.upper[j];
      const f_t lj = lp.lower[j];
      if (aj > 0.0) {
        if (lj > -inf) {
          activity += aj * lj;
        } else {
          num_neg_inf_[i]++;
        }
      } else {  // a_j < 0.0
        if (uj < inf) {
          activity += aj * uj;
        } else {
          num_neg_inf_[i]++;
        }
      }
    }
    lower_activities_[i] = activity;
  }

  // Now go through all continuous variables and use the activiites to get upper variable bounds
  i_t upper_edges = 0;
  for (i_t j = 0; j < lp.num_cols; j++) {
    upper_offsets[j] = upper_edges;
    if (var_types[j] != variable_type_t::CONTINUOUS) { continue; }
    const i_t col_start = lp.A.col_start[j];
    const i_t col_end   = lp.A.col_start[j + 1];
    for (i_t p = col_start; p < col_end; p++) {
      const i_t i = lp.A.i[p];
      if (num_integer_in_row[i] < 1) { continue; }
      if (num_neg_inf_[i] > 2 && num_pos_inf_[i] > 2) { continue; }
      const i_t row_start = Arow.row_start[i];
      const i_t row_end   = Arow.row_start[i + 1];
      const i_t row_len   = row_end - row_start;
      if (row_len < 2) { continue; }
      const f_t a_ij              = lp.A.x[p];
      const f_t slack_lower       = lp.lower[slack_map_[i]];
      const f_t slack_upper       = lp.upper[slack_map_[i]];
      const f_t slack_coeff_i     = slack_coeff[i];
      const f_t sigma_slack_lower = slack_coeff_i == 1.0 ? slack_lower : -slack_upper;
      const f_t sigma_slack_upper = slack_coeff_i == 1.0 ? slack_upper : -slack_lower;

      if (sigma_slack_lower > -inf) {
        const f_t beta = lp.rhs[i] - sigma_slack_lower;
        // sum_k a_ik x_k <= beta

        // If we have too many variables in the row that would cause the activity to be infinite,
        // we cannot derive an variable bound
        if (a_ij > 0.0 && num_neg_inf_[i] <= 2) {
          const f_t lower_activity_j = lower_activity(lp.lower[j], lp.upper[j], a_ij);

          // This is inefficient if num_neg_inf_[i] > 0
          // If num_neg_inf_[i] == 1 and var_types[s] != INTEGER, we can't derive a bound
          // If num_neg_inf_[i] == 2 and var_types[s ^ j] != INTEGER, we can't derive a bound
          // If num_neg_inf_[i] == 2 and var_types[s ^ j] == INTEGER, and lower_activity_j != -inf,
          // we can't derive a bound
          for (i_t q = row_start; q < row_end; q++) {
            const i_t l = Arow.j[q];
            if (var_types[l] == variable_type_t::CONTINUOUS) { continue; }
            // sum_{k != l, k != j} a_ik x_k + a_ij x_j + a_il x_l <= beta
            // a_ij x_j <= -a_il x_l + beta - sum_{k != l, k != j} a_ik x_k
            const f_t a_il             = Arow.x[q];
            const f_t lower_activity_l = lower_activity(lp.lower[l], lp.upper[l], a_il);
            const f_t sum              = adjusted_lower_activity(
              lower_activities_[i], num_neg_inf_[i], lower_activity_j, lower_activity_l);
            if (sum > -inf) {
              // We have a valid variable upper bound
              // x_j <= -a_il/a_ij * x_l + beta/a_ij - 1/a_ij * sum_{k != l, k != j} a_ik *
              // bound(x_k)
              upper_variables.push_back(l);
              upper_weights.push_back(-a_il / a_ij);
              upper_biases.push_back(beta / a_ij - (1.0 / a_ij) * sum);
              upper_edges++;
            }
          }
        }
      }

      if (sigma_slack_upper < inf) {
        const f_t beta = lp.rhs[i] - sigma_slack_upper;
        // sum_k a_ik x_k >= beta

        // If we have too many variables in the row that would cause the activity to be infinite,
        // we cannot derive an variable bound
        if (a_ij < 0.0 && num_pos_inf_[i] <= 2) {
          const f_t upper_activity_j = upper_activity(lp.lower[j], lp.upper[j], a_ij);

          for (i_t q = row_start; q < row_end; q++) {
            const i_t l = Arow.j[q];
            if (var_types[l] == variable_type_t::CONTINUOUS) { continue; }
            // sum_{k != l, k != j} a_ik x_k + a_ij x_j + a_il x_l >= beta
            // a_ij x_j >= -a_il x_l + beta - sum_{k != l, k != j} a_ik x_k
            const f_t a_il             = Arow.x[q];
            const f_t upper_activity_l = upper_activity(lp.lower[l], lp.upper[l], a_il);
            const f_t sum              = adjusted_upper_activity(
              upper_activities_[i], num_pos_inf_[i], upper_activity_j, upper_activity_l);
            if (sum < inf) {
              // We have a valid variable upper bound
              // x_j <= -a_il/a_ij * x_l + beta/a_ij - 1/a_ij * sum_{k != l, k != j} a_ik *
              // bound(x_k)
              upper_variables.push_back(l);
              upper_weights.push_back(-a_il / a_ij);
              upper_biases.push_back(beta / a_ij - (1.0 / a_ij) * sum);
              upper_edges++;
            }
          }
        }
      }
    }
  }
  upper_offsets[lp.num_cols] = upper_edges;
  settings.log.printf("%d variable upper bounds in %.2f seconds\n", upper_edges, toc(start_time));

  // Now go through all continuous variables and use the activiites to get lower variable bounds
  i_t lower_edges = 0;
  for (i_t j = 0; j < lp.num_cols; j++) {
    lower_offsets[j] = lower_edges;
    if (var_types[j] != variable_type_t::CONTINUOUS) { continue; }
    const i_t col_start = lp.A.col_start[j];
    const i_t col_end   = lp.A.col_start[j + 1];
    for (i_t p = col_start; p < col_end; p++) {
      const i_t i = lp.A.i[p];
      if (num_integer_in_row[i] < 1) { continue; }
      const i_t row_start = Arow.row_start[i];
      const i_t row_end   = Arow.row_start[i + 1];
      const i_t row_len   = row_end - row_start;
      if (row_len < 2) { continue; }
      const f_t a_ij              = lp.A.x[p];
      const f_t slack_lower       = lp.lower[slack_map_[i]];
      const f_t slack_upper       = lp.upper[slack_map_[i]];
      const f_t slack_coeff_i     = slack_coeff[i];
      const f_t sigma_slack_lower = slack_coeff_i == 1.0 ? slack_lower : -slack_upper;
      const f_t sigma_slack_upper = slack_coeff_i == 1.0 ? slack_upper : -slack_lower;

      if (sigma_slack_lower > -inf) {
        const f_t beta = lp.rhs[i] - sigma_slack_lower;
        // sum_k a_ik x_k <= beta

        // If we have too many variables in the row that would cause the activity to be infinite,
        // we cannot derive a variable bound
        if (a_ij < 0.0 && num_neg_inf_[i] <= 2) {
          const f_t lower_activity_j = lower_activity(lp.lower[j], lp.upper[j], a_ij);

          for (i_t q = row_start; q < row_end; q++) {
            const i_t l = Arow.j[q];
            if (var_types[l] == variable_type_t::CONTINUOUS) { continue; }
            // sum_{k != l, k != j} a_ik x_k + a_ij x_j + a_il x_l <= beta
            // a_ij x_j <= -a_il x_l + beta - sum_{k != l, k != j} a_ik x_k
            // x_j >= -a_il/a_ij * x_l + beta/a_ij - 1/a_ij * sum_{k != l, k != j} a_ik * bound(x_k)
            const f_t a_il             = Arow.x[q];
            const f_t lower_activity_l = lower_activity(lp.lower[l], lp.upper[l], a_il);
            const f_t sum              = adjusted_lower_activity(
              lower_activities_[i], num_neg_inf_[i], lower_activity_j, lower_activity_l);
            if (sum > -inf) {
              // We have a valid variable lower bound
              // x_j >= -a_il/a_ij * x_l + beta/a_ij - 1/a_ij * sum_{k != l, k != j} a_ik *
              // bound(x_k)
              lower_variables.push_back(l);
              lower_weights.push_back(-a_il / a_ij);
              lower_biases.push_back(beta / a_ij - (1.0 / a_ij) * sum);
              lower_edges++;
            }
          }
        }
      }

      if (sigma_slack_upper < inf) {
        const f_t beta = lp.rhs[i] - sigma_slack_upper;
        // sum_k a_ik x_k >= beta

        // If we have too many variables in the row that would cause the activity to be infinite,
        // we cannot derive a variable bound
        if (a_ij > 0.0 && num_pos_inf_[i] <= 2) {
          const f_t upper_activity_j = upper_activity(lp.lower[j], lp.upper[j], a_ij);

          for (i_t q = row_start; q < row_end; q++) {
            const i_t l = Arow.j[q];
            if (var_types[l] == variable_type_t::CONTINUOUS) { continue; }
            // sum_{k != l, k != j} a_ik x_k + a_ij x_j + a_il x_l >= beta
            // a_ij x_j >= -a_il x_l + beta - sum_{k != l, k != j} a_ik x_k
            const f_t a_il             = Arow.x[q];
            const f_t upper_activity_l = upper_activity(lp.lower[l], lp.upper[l], a_il);
            const f_t sum              = adjusted_upper_activity(
              upper_activities_[i], num_pos_inf_[i], upper_activity_j, upper_activity_l);
            if (sum < inf) {
              // We have a valid variable lower bound
              // x_j >= -a_il/a_ij * x_l + beta/a_ij - 1/a_ij * sum_{k != l, k != j} a_ik *
              // bound(x_k)
              lower_variables.push_back(l);
              lower_weights.push_back(-a_il / a_ij);
              lower_biases.push_back(beta / a_ij - (1.0 / a_ij) * sum);
              lower_edges++;
            }
          }
        }
      }
    }
  }
  lower_offsets[lp.num_cols] = lower_edges;
  settings.log.printf("%d variable lower bounds in %.2f seconds\n", lower_edges, toc(start_time));
}

template <typename i_t, typename f_t>
complemented_mixed_integer_rounding_cut_t<i_t, f_t>::complemented_mixed_integer_rounding_cut_t(
  const lp_problem_t<i_t, f_t>& lp,
  const simplex_solver_settings_t<i_t, f_t>& settings,
  const std::vector<i_t>& new_slacks)
  : is_slack_(lp.num_cols, 0),
    slack_rows_(lp.num_cols, -1),
    slack_cols_(lp.num_rows, -1),
    lb_variable_(lp.num_cols, -1),
    lb_star_(lp.num_cols, 0.0),
    ub_variable_(lp.num_cols, -1),
    ub_star_(lp.num_cols, 0.0),
    transformed_upper_(lp.num_cols, inf),
    bound_changed_(lp.num_cols, 0),
    scratch_pad_(lp.num_cols)
{
  for (i_t j : new_slacks) {
    is_slack_[j]        = 1;
    const i_t col_start = lp.A.col_start[j];
    const i_t i         = lp.A.i[col_start];
    slack_rows_[j]      = i;
    slack_cols_[i]      = j;
    assert(std::abs(lp.A.x[col_start]) == 1.0);
  }
}

template <typename i_t, typename f_t>
void complemented_mixed_integer_rounding_cut_t<i_t, f_t>::compute_initial_scores_for_rows(
  const lp_problem_t<i_t, f_t>& lp,
  const simplex_solver_settings_t<i_t, f_t>& settings,
  const csr_matrix_t<i_t, f_t>& Arow,
  const std::vector<f_t>& xstar,
  const std::vector<f_t>& ystar,
  std::vector<f_t>& scores)
{
  const bool verbose  = false;
  const i_t n         = lp.num_cols;
  const f_t obj_norm  = vector_norm2<i_t, f_t>(lp.objective);
  const f_t obj_denom = std::max(1.0, obj_norm);

  // Compute initial scores for all rows
  scores.resize(lp.num_rows, 0.0);
  for (i_t i = 0; i < lp.num_rows; i++) {
    const i_t row_start = Arow.row_start[i];
    const i_t row_end   = Arow.row_start[i + 1];

    const i_t row_nz = row_end - row_start;
    f_t row_norm     = 0.0;
    for (i_t p = row_start; p < row_end; p++) {
      const f_t a_j = Arow.x[p];
      row_norm += a_j * a_j;
    }
    row_norm = std::sqrt(row_norm);

    const f_t density = static_cast<f_t>(row_nz) / static_cast<f_t>(n);
    const f_t dual    = std::abs(ystar[i]);

    const i_t slack = slack_cols_[i];
    assert(slack >= 0);
    const f_t slack_value = std::max(xstar[slack], 0.0);
    const f_t slack_denom = std::max(0.1, std::sqrt(row_norm));

    const f_t nz_weight    = 0.0001;
    const f_t dual_weight  = 1.0;
    const f_t slack_weight = 0.001;

    scores[i] = nz_weight * (1.0 - density) + dual_weight * std::max(dual / obj_denom, 0.0001) +
                slack_weight * (1.0 - slack_value / slack_denom);

    if (verbose) {
      settings.log.printf("Scores[%d] = %e density %.2f dual %e slack %e\n",
                          i,
                          scores[i],
                          density,
                          dual,
                          slack_value);
    }
  }
}

template <typename i_t, typename f_t>
bool complemented_mixed_integer_rounding_cut_t<i_t, f_t>::cut_generation_heuristic(
  const inequality_t<i_t, f_t>& transformed_inequality,
  const std::vector<variable_type_t>& var_types,
  const std::vector<f_t>& transformed_xstar,
  inequality_t<i_t, f_t>& transformed_cut,
  f_t& work_estimate)
{
  std::vector<f_t> deltas_to_try;
  deltas_to_try.reserve(transformed_inequality.size());
  deltas_to_try.push_back(1.0);
  work_estimate += transformed_inequality.size();
  i_t num_integers = 0;
  f_t max_coeff    = 0.0;
  for (i_t k = 0; k < transformed_inequality.size(); k++) {
    const i_t j      = transformed_inequality.index(k);
    const f_t abs_aj = std::abs(transformed_inequality.coeff(k));
    if (var_types[j] == variable_type_t::INTEGER) {
      num_integers++;
      max_coeff                 = std::max(max_coeff, abs_aj);
      const f_t x_j             = transformed_xstar[j];
      const f_t new_upper_j     = new_upper(j);
      const f_t dist_upper      = new_upper_j - x_j;
      const f_t dist_lower      = x_j;
      const bool between_bounds = x_j > 1e-6 && (new_upper_j == inf || dist_upper > 0.0);
      if (between_bounds && abs_aj > 1e-6) { deltas_to_try.push_back(abs_aj); }
    }
  }
  if (max_coeff > 1e-6 && max_coeff != 1.0) {
    deltas_to_try.push_back(max_coeff);
    deltas_to_try.push_back(max_coeff + 1.0);
  }

  std::vector<i_t> complemented_indices;
  complemented_indices.reserve(num_integers);
  std::vector<f_t> distance_from_midpoint;
  distance_from_midpoint.reserve(num_integers);
  std::vector<i_t> integer_indices;
  integer_indices.reserve(num_integers);
  for (i_t k = 0; k < transformed_inequality.size(); k++) {
    const i_t j = transformed_inequality.index(k);
    if (var_types[j] == variable_type_t::INTEGER && new_upper(j) < inf) {
      const f_t x_j         = transformed_xstar[j];
      const f_t new_upper_j = new_upper(j);
      if (x_j > 1e-6 && new_upper_j < inf) {
        const f_t midpoint_j = new_upper_j / 2.0;
        distance_from_midpoint.push_back(x_j - midpoint_j);
        integer_indices.push_back(k);
      }
    }
  }

  std::vector<i_t> perm(integer_indices.size());
  best_score_first_permutation(distance_from_midpoint, perm);
  work_estimate +=
    integer_indices.size() > 0 ? integer_indices.size() * std::log2(integer_indices.size()) : 0;

  bool cut_found = false;

  inequality_t<i_t, f_t> complemented_inequality = transformed_inequality;
  work_estimate += 4 * transformed_inequality.size();

  f_t delta          = 0.0;
  f_t best_violation = 0.0;

  // First try without any complementation
  for (const f_t tmp_delta : deltas_to_try) {
    bool cut_ok = scale_uncomplement_and_generate_cut(var_types,
                                                      transformed_xstar,
                                                      complemented_indices,
                                                      complemented_inequality,
                                                      tmp_delta,
                                                      transformed_cut,
                                                      work_estimate);
    if (!cut_ok) { continue; }
    // Check if the cut is violated
    best_violation = compute_violation(transformed_cut, transformed_xstar);
    work_estimate += 4 * transformed_cut.size();
    if (best_violation > 1e-6) {
      cut_found = true;
      delta     = tmp_delta;
      break;
    }
  }

  if (!cut_found) {
    // Complement an integer variable
    for (const i_t idx : perm) {
      const i_t l = integer_indices[idx];
      const i_t j = complemented_inequality.index(l);
      // We have an integer variable x_j <= b_j
      // We create a new variable xbar_j such that
      // x_j + xbar_j = b_j
      // x_j = b_j - xbar_j, xbar_j = b_j - x_j
      //
      // The inequality
      // sum_{k != j} a_k x_k + a_j x_j >= beta
      // becomes
      // sum_{k != j} a_k x_k + a_j (b_j - xbar_j) >= beta
      // sum_{k != j} a_k x_k - a_j xbar_j >= beta - a_j b_j
      const f_t b_j = new_upper(j);
      const f_t a_j = complemented_inequality.coeff(l);

      complemented_inequality.vector.x[l] = -a_j;
      complemented_inequality.rhs -= a_j * b_j;
      complemented_indices.push_back(l);

      for (const f_t tmp_delta : deltas_to_try) {
        bool cut_ok = scale_uncomplement_and_generate_cut(var_types,
                                                          transformed_xstar,
                                                          complemented_indices,
                                                          complemented_inequality,
                                                          tmp_delta,
                                                          transformed_cut,
                                                          work_estimate);
        if (!cut_ok) { continue; }
        // Check if the cut is violated
        best_violation = compute_violation(transformed_cut, transformed_xstar);
        work_estimate += 4 * transformed_cut.size();
        if (best_violation > 1e-6) {
          cut_found = true;
          delta     = tmp_delta;
          break;
        }
      }
      if (cut_found) { break; }
    }
  }

  if (!cut_found) { return false; }

  // We have found a cut. Now try to improve the violation by scaling the cut by 1/2, 1/4, 1/8, etc.
  std::vector<f_t> scaled_deltas_to_try = {delta / 2.0, delta / 4.0, delta / 8.0};
  for (const f_t tmp_delta : scaled_deltas_to_try) {
    inequality_t<i_t, f_t> tmp_cut_delta;
    bool cut_ok = scale_uncomplement_and_generate_cut(var_types,
                                                      transformed_xstar,
                                                      complemented_indices,
                                                      complemented_inequality,
                                                      tmp_delta,
                                                      tmp_cut_delta,
                                                      work_estimate);
    if (!cut_ok) { continue; }

    // Check if the cut is violated
    f_t violation = compute_violation(tmp_cut_delta, transformed_xstar);
    work_estimate += 4 * tmp_cut_delta.size();
    if (violation > best_violation) {
      best_violation  = violation;
      transformed_cut = tmp_cut_delta;
      delta           = tmp_delta;
    }
  }

  std::vector<i_t> best_complemented_indices = complemented_indices;
  work_estimate += 2 * best_complemented_indices.size();

  // Try to improve the violation by complementing integer variables
  complemented_inequality = transformed_inequality;
  work_estimate += 4 * transformed_inequality.size();
  complemented_indices.clear();
  for (const i_t idx : perm) {
    const i_t l = integer_indices[idx];
    const i_t j = complemented_inequality.index(l);
    // We have an integer variable x_j <= b_j
    // We create a new variable xbar_j such that
    // x_j + xbar_j = b_j
    // x_j = b_j - xbar_j, xbar_j = b_j - x_j
    //
    // The inequality
    // sum_{k != j} a_k x_k + a_j x_j >= beta
    // becomes
    // sum_{k != j} a_k x_k + a_j (b_j - xbar_j) >= beta
    // sum_{k != j} a_k x_k - a_j xbar_j >= beta - a_j b_j
    const f_t b_j = new_upper(j);
    const f_t a_j = complemented_inequality.coeff(l);

    complemented_inequality.vector.x[l] = -a_j;
    complemented_inequality.rhs -= a_j * b_j;
    complemented_indices.push_back(l);

    inequality_t<i_t, f_t> tmp_cut_delta;

    bool cut_ok = scale_uncomplement_and_generate_cut(var_types,
                                                      transformed_xstar,
                                                      complemented_indices,
                                                      complemented_inequality,
                                                      delta,
                                                      tmp_cut_delta,
                                                      work_estimate);
    if (!cut_ok) { continue; }
    // Check if the cut is violated
    f_t violation = compute_violation(tmp_cut_delta, transformed_xstar);
    work_estimate += 4 * tmp_cut_delta.size();
    if (violation > best_violation) {
      best_violation            = violation;
      best_complemented_indices = complemented_indices;
      transformed_cut           = tmp_cut_delta;
    }
  }

  return best_violation > 1e-6;
}

template <typename i_t, typename f_t>
bool complemented_mixed_integer_rounding_cut_t<i_t, f_t>::scale_uncomplement_and_generate_cut(
  const std::vector<variable_type_t>& var_types,
  const std::vector<f_t>& transformed_xstar,
  const std::vector<i_t>& complemented_indices,
  const inequality_t<i_t, f_t>& complemented_inequality,
  f_t delta,
  inequality_t<i_t, f_t>& cut_delta,
  f_t& work_estimate)
{
  inequality_t scaled_inequality = complemented_inequality;
  if (delta != 1.0) { scaled_inequality.scale(1.0 / delta); }
  bool cut_ok = generate_cut_nonnegative_maintain_indicies(scaled_inequality, var_types, cut_delta);
  if (!cut_ok) { return false; }
  work_estimate += 4 * scaled_inequality.size();

  // Now we need to transform the complemented variables back
  for (i_t h = 0; h < complemented_indices.size(); h++) {
    const i_t l = complemented_indices[h];
    const i_t j = complemented_inequality.index(l);
    // Our cut is of the form
    // sum_{k != j} d_k x_k  + d_j xbar_j >= alpha
    // we have that xbar_j = b_j - x_j
    // So
    // sum_{k != j} d_k x_k  + d_j (b_j - x_j) >= alpha
    // Or
    // sum_{k != j} d_k x_k  - d_j x_j >= alpha - d_j b_j

    const f_t b_j         = new_upper(j);
    const f_t d_j         = cut_delta.coeff(l);
    cut_delta.vector.x[l] = -d_j;
    cut_delta.rhs -= d_j * b_j;
  }
  work_estimate += 5 * complemented_indices.size();
  return true;
}

template <typename i_t, typename f_t>
void complemented_mixed_integer_rounding_cut_t<i_t, f_t>::remove_small_coefficients(
  const std::vector<f_t>& lower_bounds,
  const std::vector<f_t>& upper_bounds,
  inequality_t<i_t, f_t>& cut)
{
  const i_t nz = cut.size();
  i_t removed  = 0;
  for (i_t k = 0; k < cut.size(); k++) {
    const i_t j = cut.index(k);

    // Check for small coefficients
    const f_t aj = cut.coeff(k);
    if (std::abs(aj) < 1e-6) {
      if (aj >= 0.0 && upper_bounds[j] < inf) {
        // Move this to the right-hand side
        cut.rhs -= aj * upper_bounds[j];
        cut.vector.x[k] = 0.0;
        removed++;
      } else if (aj <= 0.0 && lower_bounds[j] > -inf) {
        cut.rhs -= aj * lower_bounds[j];
        cut.vector.x[k] = 0.0;
        removed++;
        continue;
      } else {
        // We need to keep the coefficient
      }
    }
  }

  if (removed > 0) {
    inequality_t<i_t, f_t> new_cut(cut.vector.n);
    cut.squeeze(new_cut);
    cut = new_cut;
  }
}

template <typename i_t, typename f_t>
void complemented_mixed_integer_rounding_cut_t<i_t, f_t>::bound_substitution(
  const lp_problem_t<i_t, f_t>& lp,
  const variable_bounds_t<i_t, f_t>& variable_bounds,
  const std::vector<variable_type_t>& var_types,
  const std::vector<f_t>& xstar,
  std::vector<f_t>& transformed_xstar)
{
  transformed_xstar.resize(lp.num_cols);
  // Perform bound substitution for continuous variables
  for (i_t j = 0; j < lp.num_cols; j++) {
    if (var_types[j] != variable_type_t::CONTINUOUS) { continue; }
    // Step 1: Decide whether to use variable or simple bounds
    const f_t uj      = lp.upper[j];
    const f_t lj      = lp.lower[j];
    const f_t xstar_j = xstar[j];

    // Set lb and lb_star to the simple lower bound
    lb_variable_[j] = -1;
    lb_star_[j]     = lj;

    // Set ub and ub_star to the simple upper bound
    ub_variable_[j] = -1;
    ub_star_[j]     = uj;

    // Check the variable lower bound and update lb and lb_star
    // if these yield a tighter bound
    const i_t lower_variable_start = variable_bounds.lower_offsets[j];
    const i_t lower_variable_end   = variable_bounds.lower_offsets[j + 1];
    for (i_t p = lower_variable_start; p < lower_variable_end; p++) {
      const i_t i     = variable_bounds.lower_variables[p];
      const f_t gamma = variable_bounds.lower_weights[p];
      const f_t alpha = variable_bounds.lower_biases[p];
      // x_j >= gamma * x_i + alpha

      const f_t xstar_i = xstar[i];
      const f_t val     = gamma * xstar_i + alpha;
      if (val > lb_star_[j]) {
        lb_variable_[j] = p;
        lb_star_[j]     = val;
      }
    }

    // Check the variable upper bound and update ub and ub_star
    // if these yield a tighter bound
    const i_t upper_variable_start = variable_bounds.upper_offsets[j];
    const i_t upper_variable_end   = variable_bounds.upper_offsets[j + 1];
    for (i_t p = upper_variable_start; p < upper_variable_end; p++) {
      const i_t i     = variable_bounds.upper_variables[p];
      const f_t gamma = variable_bounds.upper_weights[p];
      const f_t alpha = variable_bounds.upper_biases[p];
      // x_j <= gamma * x_i + alpha

      const f_t xstar_i = xstar[i];
      const f_t val     = gamma * xstar_i + alpha;
      if (val < ub_star_[j]) {
        ub_variable_[j] = p;
        ub_star_[j]     = val;
      }
    }

    // Step 2: Decide to use the lower or upper bound
    const bool has_finite_lower_bound = lb_star_[j] > -inf;
    const bool has_finite_upper_bound = ub_star_[j] < inf;
    if (!has_finite_lower_bound && !has_finite_upper_bound) {
      transformed_xstar[j]  = xstar_j;
      transformed_upper_[j] = inf;
      bound_changed_[j]     = 0;
      continue;
    }
    if (has_finite_lower_bound &&
        (!has_finite_upper_bound || (xstar_j - lb_star_[j] <= ub_star_[j] - xstar_j))) {
      // Use the lower bound
      // lb_star_j <= x_j <= ub_star_j
      // v_j = x_j - lb_star_j,
      // 0 <= v_j <= ub_star_j - lb_star_j
      transformed_upper_[j] = ub_star_[j] - lb_star_[j];
      transformed_xstar[j]  = xstar_j - lb_star_[j];
      bound_changed_[j]     = (lb_star_[j] == 0.0) ? 0 : -1;
    } else if (has_finite_upper_bound) {
      // Use the upper bound
      // lb_star_j <= x_j <= ub_star_j
      // x_j + w_j = ub_star_j,
      // w_j = ub_star_j - x_j,
      // x_j = ub_star_j - w_j
      // 0 <= w_j <= ub_star_j - lb_star_j
      transformed_upper_[j] = ub_star_[j] - lb_star_[j];
      transformed_xstar[j]  = ub_star_[j] - xstar_j;
      bound_changed_[j]     = 1;
    }
  }

  // Perform bound substitution for the integer variables
  for (i_t j = 0; j < lp.num_cols; j++) {
    if (var_types[j] != variable_type_t::INTEGER) { continue; }
    const f_t uj      = lp.upper[j];
    const f_t lj      = lp.lower[j];
    const f_t xstar_j = xstar[j];

    lb_star_[j]           = lj;
    ub_star_[j]           = uj;
    transformed_xstar[j]  = xstar_j;
    transformed_upper_[j] = uj;
    bound_changed_[j]     = 0;

    if (uj < inf) {
      if (uj - xstar_j <= xstar_j - lj) {
        // Use the upper bound
        // lj <= x_j <= uj
        // x_j + w_j = uj,
        // w_j = uj - x_j,
        // x_j = uj - w_j
        // 0 <= w_j <= uj - lj
        transformed_upper_[j] = uj - lj;
        transformed_xstar[j]  = uj - xstar_j;
        bound_changed_[j]     = 1;
      } else if (lj != 0.0) {
        // Use the lower bound
        // lj <= x_j <= uj
        // v_j = x_j - lj,
        // 0 <= v_j <= uj - lj
        transformed_upper_[j] = uj - lj;
        transformed_xstar[j]  = xstar_j - lj;
        bound_changed_[j]     = -1;
      }
      continue;
    }

    if (lj > -inf && lj != 0.0) {
      // Use the lower bound
      // lj <= x_j <= uj
      // v_j = x_j - lj,
      // 0 <= v_j <= uj - lj
      transformed_upper_[j] = uj - lj;
      transformed_xstar[j]  = xstar_j - lj;
      bound_changed_[j]     = -1;
    }
  }
}

template <typename i_t, typename f_t>
void complemented_mixed_integer_rounding_cut_t<i_t, f_t>::transform_inequality(
  const variable_bounds_t<i_t, f_t>& variable_bounds,
  const std::vector<variable_type_t>& var_type,
  inequality_t<i_t, f_t>& inequality)
{
  const i_t nz = inequality.size();
  for (i_t k = 0; k < nz; k++) {
    const i_t j  = inequality.index(k);
    const f_t aj = inequality.coeff(k);
    if (var_type[j] != variable_type_t::CONTINUOUS) {
      scratch_pad_.add_to_pad(j, aj);
      continue;
    }
    if (bound_changed_[j] == -1) {
      if (lb_variable_[j] == -1) {
        // v_j = x_j - l_j, v_j >= 0
        // x_j = v_j + l_j
        // sum_{k != j} a_k x_k + a_j x_j >= beta
        // sum_{k != j} a_k x_k + a_j (v_j + l_j) >= beta
        // sum_{k != j} a_k x_k + a_j v_j >= beta - a_j l_j
        const f_t lj = lb_star_[j];
        inequality.rhs -= aj * lj;
        scratch_pad_.add_to_pad(j, aj);
      } else {
        // v_j = x_j - lb*_j, v_j >= 0
        // x_j = v_j + lb*_j
        // lb*_j = gamma * x_i + alpha
        // x_j = v_j + gamma * x_i + alpha
        // sum_{k != j} a_k x_k + a_j x_j >= beta
        // sum_{k != j} a_k x_k + a_j (v_j + gamma * x_i + alpha) >= beta
        // sum_{k != j} a_k x_k + a_j v_j + a_j * gamma * x_i >= beta - a_j alpha
        const i_t p     = lb_variable_[j];
        const f_t alpha = variable_bounds.lower_biases[p];
        const f_t gamma = variable_bounds.lower_weights[p];
        const i_t i     = variable_bounds.lower_variables[p];
        inequality.rhs -= aj * alpha;
        scratch_pad_.add_to_pad(j, aj);
        scratch_pad_.add_to_pad(i, aj * gamma);
      }
    } else if (bound_changed_[j] == 1) {
      if (ub_variable_[j] == -1) {
        // w_j = u_j - x_j, w_j >= 0
        // x_j = u_j - w_j
        // sum_{k != j} a_k x_k + a_j x_j >= beta
        // sum_{k != j} a_k x_k + a_j (u_j - w_j) >= beta
        // sum_{k != j} a_k x_k - a_j w_j >= beta - a_j u_j
        const f_t uj = ub_star_[j];
        inequality.rhs -= aj * uj;
        scratch_pad_.add_to_pad(j, -aj);
      } else {
        // w_j = ub*_j - x_j, w_j >= 0
        // x_j = ub*_j - w_j
        // ub*_j = gamma * x_i + alpha
        // x_j = gamma * x_i + alpha - w_j
        // sum_{k != j} a_k x_k + a_j x_j >= beta
        // sum_{k != j} a_k x_k + a_j (ub*_j - w_j) >= beta
        // sum_{k != j} a_k x_k + a_j (gamma * x_i + alpha) - a_j w_j >= beta
        // sum_{k != j} a_k x_k + a_j gamma * x_i - a_j w_j >= beta - a_j alpha
        const i_t p     = ub_variable_[j];
        const f_t alpha = variable_bounds.upper_biases[p];
        const f_t gamma = variable_bounds.upper_weights[p];
        const i_t i     = variable_bounds.upper_variables[p];
        inequality.rhs -= aj * alpha;
        scratch_pad_.add_to_pad(j, -aj);
        scratch_pad_.add_to_pad(i, aj * gamma);
      }
    } else if (bound_changed_[j] == 0) {
      scratch_pad_.add_to_pad(j, aj);
    }
  }
  scratch_pad_.get_pad(inequality.vector.i, inequality.vector.x);
  // At this point we have converted all the continuous variables to be nonnegative
  // Note that since continuous variables had VUB or VLB, they modified
  // the integer variables.

  // We clear the scratch pad. As it is no longer needed.
  scratch_pad_.clear_pad();

  // We now convert all the integer variables to be nonnegative
  const i_t nz_after = inequality.size();
  for (i_t k = 0; k < nz_after; k++) {
    const i_t j = inequality.index(k);
    if (var_type[j] != variable_type_t::INTEGER) { continue; }
    const f_t aj = inequality.coeff(k);
    if (bound_changed_[j] == -1) {
      // v_j = x_j - l_j, v_j >= 0
      // x_j = v_j + l_j
      // sum_{k != j} a_k x_k + a_j x_j >= beta
      // sum_{k != j} a_k x_k + a_j (v_j + l_j) >= beta
      // sum_{k != j} a_k x_k + a_j v_j >= beta - a_j l_j
      const f_t lj = lb_star_[j];
      inequality.rhs -= aj * lj;
    } else if (bound_changed_[j] == 1) {
      // w_j = u_j - x_j, w_j >= 0
      // x_j = u_j - w_j
      // sum_{k != j} a_k x_j + a_j x_j >= beta
      // sum_{k != j} a_k x_j + a_j (u_j - w_j) >= beta
      // sum_{k != j} a_k x_j - a_j w_j >= beta - a_j u_j
      const f_t uj = ub_star_[j];
      inequality.rhs -= aj * uj;
      inequality.vector.x[k] *= -1.0;
    }
  }
}

template <typename i_t, typename f_t>
void complemented_mixed_integer_rounding_cut_t<i_t, f_t>::untransform_inequality(
  const variable_bounds_t<i_t, f_t>& variable_bounds,
  const std::vector<variable_type_t>& var_type,
  inequality_t<i_t, f_t>& inequality)
{
  // First convert all the integers variables back to their original form: l_j <= x_j <= u_j
  const i_t nz = inequality.size();
  for (i_t k = 0; k < nz; k++) {
    const i_t j = inequality.index(k);
    if (var_type[j] != variable_type_t::INTEGER) { continue; }
    const f_t dj = inequality.coeff(k);
    if (bound_changed_[j] == -1) {
      // v_j = x_j - l_j, v_j >= 0
      // sum_{k != j} d_k x_k + d_j v_j >= beta
      // sum_{k != j} d_k x_k + d_j (x_j - l_j) >= beta
      // sum_{k != j} d_k x_k + d_j x_j >= beta + d_j l_j
      const f_t lj = lb_star_[j];
      inequality.rhs += dj * lj;
    } else if (bound_changed_[j] == 1) {
      // w_j = u_j - x_j, w_j >= 0
      // sum_{k != j} d_k x_k + d_j w_j >= beta
      // sum_{k != j} d_k x_k + d_j (u_j - x_j) >= beta
      // sum_{k != j} d_k x_k - d_j x_j  >= beta - d_j u_j
      const f_t uj = ub_star_[j];
      inequality.rhs -= dj * uj;
      inequality.vector.x[k] *= -1.0;
    }
  }
  // Then undo the VUB/VLB substitions and bring continuous variables back to their original form
  for (i_t k = 0; k < nz; k++) {
    const i_t j  = inequality.index(k);
    const f_t dj = inequality.coeff(k);
    if (var_type[j] != variable_type_t::CONTINUOUS) {
      scratch_pad_.add_to_pad(j, dj);
      continue;
    }
    if (bound_changed_[j] == -1) {
      if (lb_variable_[j] == -1) {
        // v_j = x_j - l_j, v_j >= 0
        // sum_{k != j} d_k x_k + d_j v_j >= beta
        // sum_{k != j} d_k x_k + d_j (x_j - l_j) >= beta
        // sum_{k != j} d_k x_k + d_j x_j >= beta + d_j l_j
        const f_t lj = lb_star_[j];
        inequality.rhs += dj * lj;
        scratch_pad_.add_to_pad(j, dj);
      } else {
        // v_j = x_j - lb*_j, v_j >= 0
        // lb*_j = gamma * x_i + alpha
        // v_j = x_j - gamma * x_i -  alpha
        // sum_{k != j} d_k x_k + d_j v_j >= beta
        // sum_{k != j} d_k x_k + d_j (x_j - gamma * x_i - alpha) >= beta
        // sum_{k != j} d_k x_k + d_j x_j - d_j * gamma * x_i >= beta + d_j alpha
        const i_t p     = lb_variable_[j];
        const f_t alpha = variable_bounds.lower_biases[p];
        const f_t gamma = variable_bounds.lower_weights[p];
        const i_t i     = variable_bounds.lower_variables[p];
        inequality.rhs += dj * alpha;
        scratch_pad_.add_to_pad(j, dj);
        scratch_pad_.add_to_pad(i, -dj * gamma);
      }
    } else if (bound_changed_[j] == 1) {
      if (ub_variable_[j] == -1) {
        // w_j = u_j - x_j, w_j >= 0
        // sum_{k != j} d_k x_k + d_j w_j >= beta
        // sum_{k != j} d_k x_k + d_j (u_j - x_j) >= beta
        // sum_{k != j} d_k x_k - d_j x_j  >= beta - d_j u_j
        const f_t uj = ub_star_[j];
        inequality.rhs -= dj * uj;
        scratch_pad_.add_to_pad(j, -dj);
      } else {
        // w_j = ub*_j - x_j, w_j >= 0
        // ub*_j = gamma * x_i + alpha
        // w_j = gamma * x_i + alpha - x_j
        // sum_{k != j} d_k x_k + d_j w_j >= beta
        // sum_{k != j} d_k x_k + d_j (gamma * x_i + alpha - x_j) >= beta
        // sum_{k != j} d_k x_k + d_j gamma * x_i - d_j x_j >= beta - d_j alpha
        const i_t p     = ub_variable_[j];
        const f_t alpha = variable_bounds.upper_biases[p];
        const f_t gamma = variable_bounds.upper_weights[p];
        const i_t i     = variable_bounds.upper_variables[p];
        inequality.rhs -= dj * alpha;
        scratch_pad_.add_to_pad(j, -dj);
        scratch_pad_.add_to_pad(i, dj * gamma);
      }
    } else {
      scratch_pad_.add_to_pad(j, dj);
    }
  }

  scratch_pad_.get_pad(inequality.vector.i, inequality.vector.x);
  scratch_pad_.clear_pad();
}

template <typename i_t, typename f_t>
bool complemented_mixed_integer_rounding_cut_t<i_t, f_t>::
  generate_cut_nonnegative_maintain_indicies(const inequality_t<i_t, f_t>& inequality,
                                             const std::vector<variable_type_t>& var_types,
                                             inequality_t<i_t, f_t>& cut)
{
  auto f = [](f_t q_1, f_t q_2) -> f_t {
    f_t q_1_hat = q_1 - std::floor(q_1);
    f_t q_2_hat = q_2 - std::floor(q_2);
    return std::min(q_1_hat, q_2_hat) + q_2_hat * std::floor(q_1);
  };

  auto h = [](f_t q) -> f_t { return std::max(q, 0.0); };

  cut.vector       = inequality.vector;
  const f_t beta   = inequality.rhs;
  const f_t f_beta = fractional_part(beta);
  cut.rhs          = f_beta * std::ceil(beta);
  if (f_beta < 0.05 || f_beta > 0.95) { return false; }

  for (i_t k = 0; k < inequality.size(); k++) {
    const i_t j = inequality.index(k);
    f_t aj      = inequality.coeff(k);
    if (var_types[j] == variable_type_t::INTEGER) {
      cut.vector.x[k] = f(aj, beta);
    } else {
      cut.vector.x[k] = h(aj);
    }
    if (cut.vector.x[k] != cut.vector.x[k]) {
      printf("cut.x[%d] %e != cut.x[%d] %e. aj %e beta %e var type %d\n",
             k,
             cut.vector.x[k],
             k,
             cut.vector.x[k],
             aj,
             beta,
             static_cast<int>(var_types[j]));
      exit(1);
    }
  }

  return true;
}

template <typename i_t, typename f_t>
f_t complemented_mixed_integer_rounding_cut_t<i_t, f_t>::compute_violation(
  const inequality_t<i_t, f_t>& cut, const std::vector<f_t>& xstar)
{
  f_t dot           = cut.vector.dot(xstar);
  f_t cut_violation = cut.rhs - dot;
  return cut_violation;
}

template <typename i_t, typename f_t>
void complemented_mixed_integer_rounding_cut_t<i_t, f_t>::substitute_slacks(
  const lp_problem_t<i_t, f_t>& lp, csr_matrix_t<i_t, f_t>& Arow, inequality_t<i_t, f_t>& cut)
{
  // Remove slacks from the cut
  // So that the cut is only over the original variables
  bool found_slack = false;
  i_t cut_nz       = 0;
  std::vector<i_t> cut_indices;
  cut_indices.reserve(cut.size());

  for (i_t k = 0; k < cut.size(); k++) {
    const i_t j  = cut.index(k);
    const f_t cj = cut.coeff(k);
    if (is_slack_[j]) {
      found_slack           = true;
      const i_t slack_start = lp.A.col_start[j];
#ifdef CHECK_SLACKS
      const i_t slack_end = lp.A.col_start[j + 1];
      const i_t slack_len = slack_end - slack_start;
      if (slack_len != 1) {
        printf("Slack %d has %d nzs in colum\n", j, slack_len);
        assert(slack_len == 1);
      }
#endif
      const f_t alpha = lp.A.x[slack_start];
#ifdef CHECK_SLACKS
      if (std::abs(alpha) != 1.0) {
        printf("Slack %d has non-unit coefficient %e\n", j, alpha);
        assert(std::abs(alpha) == 1.0);
      }
#endif

      // Do the substitution
      // Slack variable s_j participates in row i of the constraint matrix
      // Row i is of the form:
      // sum_{k != j} A(i, k) * x_k + alpha * s_j = rhs_i
      // where alpha = +1/-1
      /// So we have that
      // s_j = (rhs_i - sum_{k != j} A(i, k) * x_k)/alpha

      // Our cut is of the form:
      // sum_{k != j} C(k) * x_k + C(j) * s_j >= cut_rhs
      // So the cut becomes
      // sum_{k != j} C(k) * x_k + C(j)/alpha * (rhs_i - sum_{h != j} A(i, h) * x_h) >= cut_rhs
      // This is equivalent to:
      // sum_{k != j} C(k) * x_k + sum_{h != j} -C(j)/alpha * A(i, h) * x_h >= cut_rhs - C(j)/alpha
      // * rhs_i
      const i_t i = slack_rows_[j];
      cut.rhs -= cj * lp.rhs[i] / alpha;
      const i_t row_start = Arow.row_start[i];
      const i_t row_end   = Arow.row_start[i + 1];
      for (i_t q = row_start; q < row_end; q++) {
        const i_t h = Arow.j[q];
        if (h != j) {
          const f_t aih = Arow.x[q];
          scratch_pad_.add_to_pad(h, -cj * aih / alpha);
        } else {
          const f_t aij = Arow.x[q];
          if (std::abs(aij) != 1.0) {
            printf("Slack row %d has non-unit coefficient %e for variable %d\n", i, aij, j);
            assert(std::abs(aij) == 1.0);
          }
        }
      }

    } else {
      scratch_pad_.add_to_pad(j, cj);
    }
  }

  if (found_slack) {
    scratch_pad_.get_pad(cut.vector.i, cut.vector.x);
    // Sort the cut
    cut.sort();
  }

  // Clear the workspace
  scratch_pad_.clear_pad();
}

template <typename i_t, typename f_t>
f_t complemented_mixed_integer_rounding_cut_t<i_t, f_t>::combine_rows(
  const lp_problem_t<i_t, f_t>& lp,
  csr_matrix_t<i_t, f_t>& Arow,
  i_t xj,
  const inequality_t<i_t, f_t>& pivot_row,
  inequality_t<i_t, f_t>& inequality)
{
  // Find the coefficient associated with variable xj in the pivot row
  f_t a_l_j = 0.0;
  for (i_t k = 0; k < pivot_row.size(); k++) {
    const i_t j = pivot_row.index(k);
    if (j == xj) {
      a_l_j = pivot_row.coeff(k);
      break;
    }
  }

  if (a_l_j == 0) {
    printf("Pivot row has no coefficient for variable %d\n", xj);
    return 0.0;
  }

  f_t a_i_j = 0.0;

  // Store the inequality in the workspace
  // and save the coefficient associated with variable xj
  for (i_t k = 0; k < inequality.size(); k++) {
    const i_t j = inequality.index(k);
    if (j != xj) {
      scratch_pad_.add_to_pad(j, inequality.coeff(k));
    } else {
      a_i_j = inequality.coeff(k);
    }
  }
  if (a_i_j == 0.0) {
    printf("Inequality has zero coefficient for variable %d\n", xj);
    scratch_pad_.clear_pad();
    return 0.0;
  }

  f_t pivot_value = a_i_j / a_l_j;
  // Adjust the rhs of the inequality
  inequality.rhs -= pivot_value * pivot_row.rhs;

  // Adjust the coefficients of the inequality
  // based on the nonzeros in the pivot row
  for (i_t k = 0; k < pivot_row.size(); k++) {
    const i_t j = pivot_row.index(k);
    if (j != xj) { scratch_pad_.add_to_pad(j, -pivot_value * pivot_row.coeff(k)); }
  }

  // Store the new inequality
  scratch_pad_.get_pad(inequality.vector.i, inequality.vector.x);

  // Clear the workspace
  scratch_pad_.clear_pad();

  return -pivot_value;
}

template <typename i_t, typename f_t>
strong_cg_cut_t<i_t, f_t>::strong_cg_cut_t(const lp_problem_t<i_t, f_t>& lp,
                                           const std::vector<variable_type_t>& var_types,
                                           const std::vector<f_t>& xstar)
  : transformed_variables_(lp.num_cols, 0)
{
  // Determine the substition for the integer variables
  for (i_t j = 0; j < lp.num_cols; j++) {
    if (var_types[j] == variable_type_t::INTEGER) {
      const f_t l_j = lp.lower[j];
      const f_t u_j = lp.upper[j];
      if (l_j != 0.0) {
        // We need to transform the variable
        // Check the distance to each bound
        const f_t dist_to_lower = std::max(0.0, xstar[j] - l_j);
        const f_t dist_to_upper = std::max(0.0, u_j - xstar[j]);
        if (dist_to_upper >= dist_to_lower || u_j >= inf) {
          // We are closer to the lower bound.
          transformed_variables_[j] = -1;
        } else if (u_j < inf) {
          // We are closer to the finite upper bound
          transformed_variables_[j] = 1;
        }
      }
    }
  }
}

template <typename i_t, typename f_t>
i_t strong_cg_cut_t<i_t, f_t>::remove_continuous_variables_integers_nonnegative(
  const lp_problem_t<i_t, f_t>& lp,
  const simplex_solver_settings_t<i_t, f_t>& settings,
  const std::vector<variable_type_t>& var_types,
  inequality_t<i_t, f_t>& inequality)
{
  const bool verbose = false;
  // Count the number of continuous variables in the inequality
  i_t num_continuous = 0;
  const i_t nz       = inequality.size();
  for (i_t k = 0; k < nz; k++) {
    const i_t j = inequality.index(k);
    if (var_types[j] == variable_type_t::CONTINUOUS) { num_continuous++; }
  }

  if (verbose) { settings.log.printf("num_continuous %d\n", num_continuous); }
  // We assume the inequality is of the form sum_j a_j x_j <= rhs

  for (i_t k = 0; k < nz; k++) {
    const i_t j   = inequality.index(k);
    const f_t l_j = lp.lower[j];
    const f_t u_j = lp.upper[j];
    const f_t a_j = inequality.coeff(k);
    if (var_types[j] == variable_type_t::CONTINUOUS) {
      if (a_j == 0.0) { continue; }

      if (a_j > 0.0 && l_j > -inf) {
        // v_j = x_j - l_j >= 0
        // x_j = v_j + l_j
        // sum_{k != j} a_k x_k + a_j x_j <= rhs
        // sum_{k != j} a_k x_k + a_j (v_j + l_j) <= rhs
        // sum_{k != j} a_k x_k + a_j v_j <= rhs - a_j l_j
        inequality.rhs -= a_j * l_j;
        transformed_variables_[j] = -1;

        // We now have a_j * v_j with a_j, v_j >= 0
        // So we have sum_{k != j} a_k x_k <= sum_{k != j} a_k x_k + a_j v_j <= rhs - a_j l_j
        // So we can now drop the continuous variable v_j
        inequality.vector.x[k] = 0.0;

      } else if (a_j < 0.0 && u_j < inf) {
        // w_j = u_j - x_j >= 0
        // x_j = u_j - w_j
        // sum_{k != j} a_k x_k + a_j x_j <= rhs
        // sum_{k != j} a_k x_k + a_j (u_j - w_j) <= rhs
        // sum_{k != j} a_k x_k - a_j w_j <= rhs - a_j u_j
        inequality.rhs -= a_j * u_j;
        transformed_variables_[j] = 1;

        // We now have a_j * w_j with a_j, w_j >= 0
        // So we have sum_{k != j} a_k x_k <= sum_{k != j} a_k x_k + a_j w_j <= rhs - a_j u_j
        // So we can now drop the continuous variable w_j
        inequality.vector.x[k] = 0.0;
      } else {
        // We can't keep the coefficient of the continuous variable positive
        // This means we can't eliminate the continuous variable
        if (verbose) { settings.log.printf("x%d ak: %e lo: %e up: %e\n", j, a_j, l_j, u_j); }
        return -1;
      }
    } else {
      // The variable is integer. We just need to ensure it is nonnegative
      if (transformed_variables_[j] == -1) {
        // We are closer to the lower bound.
        // v_j = x_j - l_j >= 0
        // x_j = v_j + l_j
        // sum_{k != j} a_k x_k + a_j x_j <= rhs
        // sum_{k != j} a_k x_k + a_j (v_j + l_j) <= rhs
        // sum_{k != j} a_k x_k + a_j v_j <= rhs - a_j l_j
        inequality.rhs -= a_j * l_j;
      } else if (transformed_variables_[j] == 1) {
        // We are closer to the finite upper bound
        // w_j = u_j - x_j >= 0
        // x_j = u_j - w_j
        // sum_{k != j} a_k x_k + a_j x_j <= rhs
        // sum_{k != j} a_k x_k + a_j (u_j - w_j) <= rhs
        // sum_{k != j} a_k x_k - a_j w_j <= rhs - a_j u_j
        inequality.rhs -= a_j * u_j;
        inequality.vector.x[k] *= -1.0;
      }
    }
  }

  // Squeeze out the zero coefficents
  sparse_vector_t<i_t, f_t> new_inequality_vector(inequality.vector.n, 0);
  inequality.vector.squeeze(new_inequality_vector);
  inequality.vector = new_inequality_vector;
  return 0;
}

template <typename i_t, typename f_t>
void strong_cg_cut_t<i_t, f_t>::to_original_integer_variables(const lp_problem_t<i_t, f_t>& lp,
                                                              inequality_t<i_t, f_t>& cut)
{
  // We expect a cut of the form sum_j a_j y_j <= rhs
  // where y_j >= 0 is a transformed variable
  // We need to convert it back into a cut on the original variables

  for (i_t k = 0; k < cut.size(); k++) {
    const i_t j   = cut.index(k);
    const f_t a_j = cut.coeff(k);
    if (transformed_variables_[j] == -1) {
      // sum_{k != j} a_k x_k + a_j v_j <= rhs
      // v_j = x_j - l_j >= 0,
      // sum_{k != j} a_k x_k + a_j (x_j - l_j) <= rhs
      // sum_{k != j} a_k x_k + a_j x_j <= rhs + a_j l_j
      cut.rhs += a_j * lp.lower[j];
    } else if (transformed_variables_[j] == 1) {
      // sum_{k != j} a_k x_k + a_j w_j <= rhs
      // w_j = u_j - x_j >= 0
      // sum_{k != j} a_k x_k + a_j (u_j - x_j) <= rhs
      // sum_{k != j} a_k x_k - a_j x_j <= rhs - a_j u_j
      cut.rhs -= a_j * lp.upper[j];
      cut.vector.x[k] *= -1.0;
    }
  }
}

template <typename i_t, typename f_t>
i_t strong_cg_cut_t<i_t, f_t>::generate_strong_cg_cut_integer_only(
  const simplex_solver_settings_t<i_t, f_t>& settings,
  const std::vector<variable_type_t>& var_types,
  const inequality_t<i_t, f_t>& inequality,
  inequality_t<i_t, f_t>& cut)
{
  // We expect an inequality of the form sum_j a_j x_j <= rhs
  // where all the variables x_j are integer and nonnegative

  // We then apply the CG cut:
  // sum_j floor(a_j) x_j <= floor(rhs)
  cut.reserve(inequality.size());
  cut.clear();

  f_t a_0   = inequality.rhs;
  f_t f_a_0 = fractional_part(a_0);

  if (f_a_0 == 0.0) {
    // f(a_0) == 0.0 so we do a weak CG cut
    cut.reserve(inequality.size());
    cut.clear();
    for (i_t k = 0; k < inequality.size(); k++) {
      const i_t j   = inequality.index(k);
      const f_t a_j = inequality.coeff(k);
      if (var_types[j] == variable_type_t::INTEGER) {
        cut.push_back(j, std::floor(a_j));
      } else {
        return -1;
      }
    }
    cut.rhs = std::floor(inequality.rhs);
  } else {
    return generate_strong_cg_cut_helper(
      inequality.vector.i, inequality.vector.x, inequality.rhs, var_types, cut);
  }
  return 0;
}

template <typename i_t, typename f_t>
i_t strong_cg_cut_t<i_t, f_t>::generate_strong_cg_cut_helper(
  const std::vector<i_t>& indicies,
  const std::vector<f_t>& coefficients,
  f_t rhs,
  const std::vector<variable_type_t>& var_types,
  inequality_t<i_t, f_t>& cut)
{
  const bool verbose = false;
  const i_t nz       = indicies.size();
  const f_t f_a_0    = fractional_part(rhs);

  const f_t min_fractional_part = 1e-2;
  if (f_a_0 < min_fractional_part) { return -1; }
  if (f_a_0 > 1 - min_fractional_part) { return -1; }

  // We will try to generat a strong CG cut.
  // Find the unique integer k such that
  // 1/(k+1) <= f(a_0) < 1/k
  const f_t k_upper = 1.0 / f_a_0;
  i_t k             = static_cast<i_t>(std::ceil(k_upper)) - 1;

  const f_t alpha = 1.0 - f_a_0;
  f_t lower       = 1.0 / static_cast<f_t>(k + 1);
  f_t upper       = 1.0 / static_cast<f_t>(k);
  if (verbose) { printf("f_a_0 %e lower %e upper %e alpha %e\n", f_a_0, lower, upper, alpha); }
  if (f_a_0 >= lower && f_a_0 < upper) {
    cut.reserve(nz);
    cut.clear();
    for (i_t q = 0; q < nz; q++) {
      const i_t j   = indicies[q];
      const f_t a_j = coefficients[q];
      if (var_types[j] == variable_type_t::INTEGER) {
        const f_t f_a_j = fractional_part(a_j);
        const f_t tol   = 1e-4;
        if (f_a_j <= f_a_0 + tol) {
          cut.push_back(j, (k + 1.0) * std::floor(a_j));
          if (verbose) { printf("j %d a_j %e f_a_j %e k %d\n", j, a_j, f_a_j, k); }
        } else {
          // Find p such that p <= k * f(a_j) < p + 1
          i_t p = static_cast<i_t>(std::floor(k * f_a_j));
          // If f(a_j) > f(a_0) + p /k (1 - f(a_0)) then we can increase the cofficient by 1
          const f_t rhs_j = f_a_0 + static_cast<f_t>(p) / static_cast<f_t>(k) * alpha;
          const i_t coeff = (k + 1) * static_cast<i_t>(std::floor(a_j)) + p;
          if (f_a_j > rhs_j + tol) {
            cut.push_back(j, static_cast<f_t>(coeff + 1));
          } else {
            cut.push_back(j, static_cast<f_t>(coeff));
          }
        }
      } else {
        return -1;
      }
    }
  } else {
    if (verbose) { printf("Error: k %d lower %e f(a_0) %e upper %e\n", k, lower, f_a_0, upper); }
    return -1;
  }
  cut.rhs = (k + 1.0) * std::floor(rhs);
  if (verbose) {
    printf("Generated strong CG cut: k %d f_a_0 %e cut_rhs %e\n", k, f_a_0, cut.rhs);
    for (i_t q = 0; q < cut.size(); q++) {
      if (cut.vector.x[q] != 0.0) { printf("%.16e x%d ", cut.vector.x[q], cut.vector.i[q]); }
    }
    printf("\n");
    printf("Original inequality rhs %e nz %ld\n", rhs, coefficients.size());
    for (i_t q = 0; q < nz; q++) {
      printf("%e x%d ", coefficients[q], indicies[q]);
    }
    printf("\n");
  }
  return 0;
}

template <typename i_t, typename f_t>
i_t strong_cg_cut_t<i_t, f_t>::generate_strong_cg_cut(
  const lp_problem_t<i_t, f_t>& lp,
  const simplex_solver_settings_t<i_t, f_t>& settings,
  const std::vector<variable_type_t>& var_types,
  const inequality_t<i_t, f_t>& inequality,
  const std::vector<f_t>& xstar,
  inequality_t<i_t, f_t>& cut)
{
#ifdef PRINT_INEQUALITY_INFO
  for (i_t k = 0; k < inequality.i.size(); k++) {
    printf("%e %c%d ",
           inequality.x[k],
           var_types[inequality.i[k]] == variable_type_t::CONTINUOUS ? 'x' : 'y',
           inequality.i[k]);
  }
  printf("CG inequality rhs %e\n", inequality_rhs);
#endif
  // Try to remove continuous variables from the inequality
  // and transform integer variables to be nonnegative

  // Copy the inequality since remove continuous variables will modify it
  inequality_t<i_t, f_t> cg_inequality = inequality;
  i_t status =
    remove_continuous_variables_integers_nonnegative(lp, settings, var_types, cg_inequality);

  if (status != 0) {
    // Try negating the equality and see if that helps
    cg_inequality = inequality;
    cg_inequality.negate();

    status =
      remove_continuous_variables_integers_nonnegative(lp, settings, var_types, cg_inequality);
  }

  if (status == 0) {
    // We have an inequality with no continuous variables

    // Generate a CG cut
    status = generate_strong_cg_cut_integer_only(settings, var_types, cg_inequality, cut);
    if (status != 0) { return -1; }

    // Convert the CG cut back to the original variables
    to_original_integer_variables(lp, cut);

    // Check for violation
    f_t dot = cut.vector.dot(xstar);
    // If the cut is violated we will have: sum_j a_j xstar_j > rhs
    f_t violation                     = dot - cut.rhs;
    const f_t min_violation_threshold = 1e-6;
    if (violation > min_violation_threshold) {
      //  Note that no slacks are currently present. Since slacks are currently treated as
      //  continuous. However, this may change. We may need to substitute out the slacks here

      // The CG cut is in the form: sum_j a_j x_j <= rhs
      // The cut pool wants the cut in the form: sum_j a_j x_j >= rhs
      cut.negate();
      return 0;
    }
  }
  return -1;
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
    i_t refactor_status = basis_update.refactor_basis(
      lp.A, settings, lp.lower, lp.upper, start_time, basic_list, nonbasic_list, vstatus);
    if (refactor_status == CONCURRENT_HALT_RETURN) { return CONCURRENT_HALT_RETURN; }
    if (refactor_status == TIME_LIMIT_RETURN) { return TIME_LIMIT_RETURN; }
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
template class cut_pool_t<int, double>;
template class cut_generation_t<int, double>;
template class knapsack_generation_t<int, double>;
template class tableau_equality_t<int, double>;
template class complemented_mixed_integer_rounding_cut_t<int, double>;
template class variable_bounds_t<int, double>;

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

}  // namespace cuopt::linear_programming::dual_simplex
