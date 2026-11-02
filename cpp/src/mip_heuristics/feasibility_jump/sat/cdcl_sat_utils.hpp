/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "../fj_cpu_binary.cuh"

#include <utilities/macros.cuh>
#include <utilities/splitmix64.hpp>

#include <algorithm>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <map>
#include <new>
#include <span>
#include <utility>
#include <vector>

// Polarity is the variable value satisfying the literal: 1 for x, 0 for !x.
#define LIT(var, polarity) (((var) << 1) | (polarity))
#define LIT_VAR(lit)       ((lit) >> 1)
#define LIT_POLARITY(lit)  ((lit) & 1)
#define LIT_NEG(lit)       ((lit) ^ 1)

namespace cuopt::mathematical_optimization::mip {

namespace {

static constexpr int encoding_stop_poll_period       = 128;
static constexpr size_t direct_encoding_term_limit   = 12;
static constexpr size_t direct_encoding_clause_limit = 32;
static constexpr size_t bdd_state_limit              = 20000;
static constexpr int sat_variable_limit              = 1000000;
static constexpr size_t bdd_term_limit               = 256;
static constexpr size_t sat_clause_limit             = 2000000;
static constexpr int bdd_false_terminal              = -1;
static constexpr int bdd_true_terminal               = -2;

using pseudoboolean_term_t = std::pair<int, int64_t>;

// Both encoders use sum_i a_i z_i <= rhs, where a_i = |c_i| and z_i is x_i for c_i > 0
// and !x_i otherwise.
// For direct encoding, the idea is, for example, given the pseudo-boolean constraint
// 2x_1+3x_2+5x_3 <= 5
// Some violating assignments are:
// (x_1, x_2, x_3) = {(1, 0, 1), (0, 1, 1), (1, 1, 1)}
// Each forbidden assigmnent becomes a clause. For instance:
// x_1 = 1, x_2 = 0, x_3 = 1 is forbidden by NOT(x_1) AND x_2 AND NOT(x_3).
// For the above example, a simplifed equivalent CNF is:
// (NOT(x_1) OR NOT(x_3)) AND (NOT(x_2) OR NOT(x_3))
// Obviously, this becomes inefficient for large numbers of terms.
static bool try_encode_pb_row_direct(const std::vector<pseudoboolean_term_t>& terms,
                                     int64_t rhs,
                                     std::vector<int64_t>& subset_weight,
                                     std::vector<std::vector<int>>& cnf)
{
  if (terms.size() > direct_encoding_term_limit) return false;

  const size_t row_begin = cnf.size();
  const int count        = 1 << terms.size();
  subset_weight.assign(count, 0);
  for (int mask = 1; mask < count; ++mask) {
    const int bit       = std::countr_zero((unsigned)mask);
    subset_weight[mask] = subset_weight[mask & (mask - 1)] + std::abs(terms[bit].second);
    if (subset_weight[mask] <= rhs) continue;
    bool minimal = true;
    for (size_t k = 0; k < terms.size() && minimal; ++k) {
      if ((mask & (1 << k)) && subset_weight[mask] - std::abs(terms[k].second) > rhs)
        minimal = false;
    }
    if (!minimal) continue;
    std::vector<int> clause;
    for (size_t k = 0; k < terms.size(); ++k) {
      if (mask & (1 << k)) clause.push_back(LIT(terms[k].first, terms[k].second < 0));
    }
    cnf.push_back(std::move(clause));
    if (cnf.size() - row_begin > direct_encoding_clause_limit) {
      cnf.resize(row_begin);
      return false;
    }
  }
  return true;
}

// Two-clause monotone ROBDD encoding (Abio et al., JAIR 2012, Sec. 6).
// Q(i,r) := sum_{j>=i} a_j z_j <= r
// Q(i,r) -> Q(i+1,r) && (!z_i || Q(i+1,r-a_i)).
// The idea of BDD encoding is inspired by dynamic programming, by encoding a "decision diagram"
// whose state represents the accumulated weighted sum of the pseudo-boolean constraint.
static int encode_pb_row_bdd_node(const std::vector<pseudoboolean_term_t>& terms,
                                  const std::vector<int64_t>& suffix,
                                  std::map<std::pair<int, int64_t>, int>& memo,
                                  int& variables,
                                  std::vector<std::vector<int>>& cnf,
                                  bool& overflow,
                                  int i,
                                  int64_t remaining)
{
  if (remaining < 0) return bdd_false_terminal;
  if (suffix[i] <= remaining) return bdd_true_terminal;
  const auto key = std::make_pair(i, remaining);
  if (const auto found = memo.find(key); found != memo.end()) return found->second;
  if (memo.size() > bdd_state_limit || variables >= sat_variable_limit ||
      terms.size() > bdd_term_limit) {
    overflow = true;
    return bdd_false_terminal;
  }
  const int low =
    encode_pb_row_bdd_node(terms, suffix, memo, variables, cnf, overflow, i + 1, remaining);
  const int high = encode_pb_row_bdd_node(
    terms, suffix, memo, variables, cnf, overflow, i + 1, remaining - std::abs(terms[i].second));
  if (overflow) return bdd_false_terminal;
  if (low == high) {
    memo.emplace(key, low);
    return low;
  }
  const int q = LIT(variables++, 1);                       // Q(i, remaining)
  const int z = LIT(terms[i].first, terms[i].second > 0);  // z_i
  if (low != bdd_true_terminal) {
    std::vector<int> clause{LIT_NEG(q)};
    if (low >= 0) clause.push_back(low);
    cnf.push_back(std::move(clause));
  }
  if (high != bdd_true_terminal) {
    std::vector<int> clause{LIT_NEG(q), LIT_NEG(z)};
    if (high >= 0) clause.push_back(high);
    cnf.push_back(std::move(clause));
  }
  memo.emplace(key, q);
  return q;
}

static sat_result_t encode_pb_row_bdd(const std::vector<pseudoboolean_term_t>& terms,
                                      int64_t rhs,
                                      int& variables,
                                      std::vector<std::vector<int>>& cnf)
{
  std::vector<int64_t> suffix(terms.size() + 1, 0);
  for (int k = (int)terms.size() - 1; k >= 0; --k)
    suffix[k] = suffix[k + 1] + std::abs(terms[k].second);
  std::map<std::pair<int, int64_t>, int> memo;
  bool overflow  = false;
  const int root = encode_pb_row_bdd_node(terms, suffix, memo, variables, cnf, overflow, 0, rhs);
  if (overflow) return sat_result_t::declined;
  if (root == bdd_false_terminal) return sat_result_t::infeasible;
  // The root unit clause asserts Q(0,rhs), the normalized PB row.
  if (root >= 0) cnf.push_back({root});
  return sat_result_t::successful;
}

template <typename coef_t, typename Stop>
static sat_result_t encode_pb_model(const fj_bin_problem_t<coef_t>& pb,
                                    int& variables,
                                    std::vector<std::vector<int>>& cnf,
                                    const Stop& stop)
{
  variables = pb.n_variables;
  cnf.clear();
  std::vector<pseudoboolean_term_t> terms;
  std::vector<int64_t> subset_weight;
  for (int r = 0; r < pb.n_constraints; ++r) {
    if (r % encoding_stop_poll_period == 0 && stop()) return sat_result_t::stopped;
    terms.clear();
    for (int p = pb.offsets[r]; p < pb.offsets[r + 1]; ++p)
      terms.emplace_back(pb.variables[p], pb.coefficients[p]);

    for (size_t k = 1; k < terms.size(); ++k)
      cuopt_assert(terms[k - 1].first < terms[k].first,
                   "CSR column indices must be strictly increasing");

    int64_t rhs          = pb.bound[r];
    int64_t max_activity = 0;
    // normalize coefficients to be signed
    for (auto [v, coefficient] : terms) {
      rhs -= std::min<int64_t>(0, coefficient);
      max_activity += std::abs(coefficient);
    }
    if (rhs < 0) return sat_result_t::infeasible;
    if (max_activity <= rhs) continue;  // constraint is always satisfied

    if (!try_encode_pb_row_direct(terms, rhs, subset_weight, cnf)) {
      const auto encoding = encode_pb_row_bdd(terms, rhs, variables, cnf);
      if (encoding != sat_result_t::successful) return encoding;
    }
    if (cnf.size() > sat_clause_limit) return sat_result_t::declined;
  }
  return sat_result_t::successful;
}

// MiniSat-style EVSIDS [Een-Sorensson 2003, Sec. 4.6; Biere-Froehlich 2015].
// This is basically the equivalent of branching decisions/pseudocosts for SAT / the scores in FJ.
// Pick the next variables based on their "conflict relevance". The more a variable causes conflict,
// the higher its score. For this, a heap is maintained
struct sat_variable_order_t {
  static constexpr int not_in_heap                   = -1;
  static constexpr double initial_activity_increment = 1.0;
  static constexpr double activity_rescale_threshold = 1e100;
  static constexpr double activity_rescale_factor    = 1e-100;
  static constexpr double initial_clause_activity    = 0.001;

  sat_variable_order_t(int n, uint64_t rng_seed) : position(n, not_in_heap), activity(n), tie(n)
  {
    cuopt::splitmix64_t tie_rng(rng_seed);
    for (int v = 0; v < n; ++v)
      tie[v] = tie_rng.next_u64();
  }

  bool empty() const { return heap.empty(); }
  void add_initial_activity(int v) { activity[v] += initial_clause_activity; }
  void clear_activity(int v) { activity[v] = 0.0; }

  void build_heap()
  {
    for (size_t v = 0; v < activity.size(); ++v)
      insert(v);
  }

  void insert(int v)
  {
    if (position[v] != not_in_heap) return;
    position[v] = (int)heap.size();
    heap.push_back(v);
    heap_up(position[v]);
  }

  int pop_best()
  {
    const int result = heap.front();
    const int last   = heap.back();
    heap.pop_back();
    position[result] = not_in_heap;
    if (!heap.empty()) {
      int p = 0;
      while (2 * p + 1 < (int)heap.size()) {
        int child = 2 * p + 1;
        if (child + 1 < (int)heap.size() && higher(heap[child + 1], heap[child])) ++child;
        if (!higher(heap[child], last)) break;
        heap[p]           = heap[child];
        position[heap[p]] = p;
        p                 = child;
      }
      heap[p]        = last;
      position[last] = p;
    }
    return result;
  }

  void bump(int v)
  {
    activity[v] += increment;
    if (activity[v] > activity_rescale_threshold) {
      for (double& score : activity)
        score *= activity_rescale_factor;
      increment *= activity_rescale_factor;
    }
    if (position[v] != not_in_heap) heap_up(position[v]);
  }

  void decay_activity(double factor) { increment /= factor; }

 private:
  bool higher(int a, int b) const
  {
    return activity[a] != activity[b] ? activity[a] > activity[b] : tie[a] > tie[b];
  }

  void heap_up(int p)
  {
    const int v = heap[p];
    while (p > 0) {
      const int parent = (p - 1) / 2;
      if (!higher(v, heap[parent])) break;
      heap[p]           = heap[parent];
      position[heap[p]] = p;
      p                 = parent;
    }
    heap[p]     = v;
    position[v] = p;
  }

  std::vector<int> heap, position;
  std::vector<double> activity;
  std::vector<uint64_t> tie;
  double increment{initial_activity_increment};
};

struct trail_t {
  // This is the variable assignment stack + the propagation queue. propagation_head indexes the
  // next assignment whose consequences must be scanned. depth_starts[d] is the first assignment at
  // decision depth d + 1, so truncating there backtracks to depth d. Inspired by MiniSat.
  // EVSIDS ordering
  std::vector<int> literals;
  std::vector<size_t> depth_starts;
  size_t propagation_head{0};

  size_t size() const { return literals.size(); }
  int operator[](size_t index) const { return literals[index]; }
  auto begin() const { return literals.begin(); }
  auto end() const { return literals.end(); }

  int decision_depth() const { return depth_starts.size(); }
  bool at_root() const { return depth_starts.empty(); }

  void enqueue(int lit) { literals.push_back(lit); }

  bool has_pending_propagation() const { return propagation_head < literals.size(); }

  int next_to_propagate()
  {
    cuopt_assert(has_pending_propagation(), "");
    return literals[propagation_head++];
  }

  size_t propagated_count() const { return propagation_head; }

  void start_decision_depth() { depth_starts.push_back(literals.size()); }

  bool above_depth(int target_depth) const { return decision_depth() > target_depth; }

  size_t backtrack_offset(int target_depth) const
  {
    cuopt_assert(target_depth >= 0 && target_depth < decision_depth(), "");
    return depth_starts[target_depth];
  }

  void truncate_to_depth(int target_depth)
  {
    const size_t retained_size = backtrack_offset(target_depth);
    literals.resize(retained_size);
    propagation_head = std::min(propagation_head, retained_size);
    depth_starts.resize(target_depth);
  }
};

using clause_ref_t = int;

// a clause is a series of OR operations over one or more literals
// example: (A ∨ ¬B ∨ C)
struct clause_t {
 public:
  static constexpr size_t header_words = 4;

  static size_t storage_words(size_t literal_count) { return header_words + literal_count; }

  clause_t(std::span<const int> clause_literals, int clause_lbd, int conflict, bool is_learned)
    : size(clause_literals.size()),
      lbd(clause_lbd),
      last_conflict(conflict),
      flags(is_learned ? learned_flag : 0)
  {
    cuopt_assert(size >= 0, "");
    std::copy(clause_literals.begin(), clause_literals.end(), literals);
  }

  bool learned() const { return has_flag(learned_flag); }
  bool deleted() const { return has_flag(deleted_flag); }
  bool locked() const { return has_flag(locked_flag); }

  void touch(int conflict) { last_conflict = conflict; }
  void mark_deleted() { set_flag(deleted_flag); }

  void set_locked(bool locked)
  {
    if (locked)
      set_flag(locked_flag);
    else
      clear_flag(locked_flag);
  }

 private:
  enum flag_t : int {
    learned_flag = 1 << 0,
    deleted_flag = 1 << 1,
    locked_flag  = 1 << 2,
  };

  bool has_flag(flag_t flag) const { return flags & flag; }
  void set_flag(flag_t flag) { flags |= flag; }
  void clear_flag(flag_t flag) { flags &= ~flag; }

 public:
  int size;
  int lbd;
  int last_conflict;
  int flags;
  int literals[];
};

static_assert(sizeof(clause_t) == clause_t::header_words * sizeof(int));
static_assert(alignof(clause_t) == alignof(int));

// clauses are stored in a contiguous arena, header + each clause.
class clause_arena_t {
 public:
  static size_t storage_words(size_t clause_count, size_t literal_count)
  {
    return clause_count * clause_t::storage_words(0) + literal_count;
  }

  void reserve(size_t words) { words_.reserve(words); }

  clause_ref_t add(std::span<const int> literals, bool learned, int lbd, int conflict)
  {
    const clause_ref_t ref = words_.size();
    words_.resize(words_.size() + clause_t::storage_words(literals.size()));

    // placement new inside the arena
    new (words_.data() + ref) clause_t(literals, lbd, conflict, learned);
    return ref;
  }

  clause_t& operator[](clause_ref_t ref)
  {
    cuopt_assert(ref >= 0 && (size_t)ref + clause_t::header_words <= words_.size(), "");
    return *(clause_t*)(words_.data() + ref);
  }

 private:
  std::vector<int> words_;
};

}  // namespace

}  // namespace cuopt::mathematical_optimization::mip
