/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#pragma once

#include <mip_heuristics/structural/early_structural.cuh>

#include <dual_simplex/user_problem.hpp>
#include <utilities/omp_helpers.hpp>
#include <utilities/timer.hpp>

#include <atomic>
#include <cstdint>
#include <limits>
#include <memory>
#include <vector>

namespace cuopt::mathematical_optimization::mip {

/**
 * @brief Solver for markshare / market split models.
 *
 * Recognizes `min w * sum_k s_k  s.t.  sum_j a_kj x_j + s_k = b_k, x binary, s_k >= 0` with small
 * integer data, and solves it by dynamic programming.
 *
 * `solve()` returns true when it find and prove an optimal solution.
 */
template <typename i_t, typename f_t>
class markshare_t : public structural_heuristic_t<i_t, f_t> {
 public:
  // Normalized coefficients, right hand sides and residuals. All search arithmetic is exact
  // integer arithmetic in this type.
  using coefficient_type = int32_t;

  markshare_t() = default;

  const char* name() const override { return "MarkshareDP"; }

  // The problem_t overload stays the base's `false`: this heuristic reads the raw model only.
  using structural_heuristic_t<i_t, f_t>::recognize;

  bool recognize(const optimization_problem_t<i_t, f_t>& op_problem,
                 const typename mip_solver_settings_t<i_t, f_t>::tolerances_t& tolerances) override;

  bool solve(const typename mip_solver_settings_t<i_t, f_t>::tolerances_t& tolerances,
             f_t time_limit,
             std::atomic<bool>& preemption,
             std::vector<f_t>& assignment) override;

  bool exclusive() const override { return true; }

 private:
  struct settings_t {
    // Structural caps. These are the cheap rejects, so keep them tight.
    i_t max_rows{8};
    i_t max_core_cols{64};
    coefficient_type max_normalized_rhs{1 << 20};
    size_t max_table_bytes{size_t{64} << 20};

    i_t node_report_interval{4096};  // node granularity of the budget and progress checks
    double report_interval{5.0};     // seconds between progress rows

    i_t max_level{64};

    // Meet-in-the-middle terminal. Instead of descending the last `hash_depth` levels, the search
    // asks one exact m-dimensional question about the columns below it.
    i_t hash_min_cols{40};  // enable the terminal only when the core column count exceeds this
    // Building the table costs about 2^depth and the search above it about 2^(n - depth), so the
    // optimum sits near n/2.
    i_t hash_depth_offset{1};
    i_t hash_max_depth{30};
    // Pinned budget for the terminal table.
    size_t hash_bytes{size_t{8} << 30};
    // Above this the search cannot finish even with the terminal. markshare2 has 60 core columns.
    i_t max_search_cols{62};

    f_t integrality_tolerance{1e-6};
    // Round trip check on the row normalization. Deliberately far tighter than the integrality
    // tolerance: exact row scaling passes it with room, an inexact model does not.
    double normalization_tolerance{1e-9};
  };

  // The normalized integer model the search runs on. Built by recognize(), read only afterwards.
  struct normalized_model_t {
    i_t m{0};  // rows
    i_t n{0};  // core binaries kept in the search

    std::vector<i_t> core_col;   // search index -> problem column
    std::vector<i_t> slack_col;  // row -> the continuous slack column of that row
    std::vector<i_t> fixed_col;  // columns pinned at a single value
    std::vector<f_t> fixed_val;  // the value each pinned column takes

    std::vector<coefficient_type> Arow;        // m x n, for the table builders
    std::vector<coefficient_type> Acol;        // n x m, the DFS hot layout
    std::vector<coefficient_type> b;           // row -> normalized rhs
    std::vector<coefficient_type> prefix_max;  // m x (n + 1), running column sums
    std::vector<coefficient_type> row_gcd;     // row -> gcd of its coefficients
    f_t weight{0};                             // the common w in c(s_k) = w
    i_t n_variables{0};
  };

  enum class dfs_result_t { FOUND, EXHAUSTED, BUDGET };

  /**
   * @brief Fingerprints of partial sum vectors.
   *
   */
  struct fingerprint_set_t {
    std::vector<uint64_t> slot;  // zero marks an empty slot
    size_t mask{0};

    void init(size_t capacity);
    void insert(uint64_t fingerprint);
    bool contains(uint64_t fingerprint) const;
    size_t bytes() const { return slot.size() * sizeof(uint64_t); }
  };

  // Per-search scratch. Held separately from the model so that several enumerations can run
  // concurrently over the same (read only) model and tables.
  struct dfs_context_t {
    std::vector<coefficient_type> residual;  // (n + 1) * m
    std::vector<uint8_t> branch;             // n + 1
    std::vector<uint8_t> value;              // n
    int64_t nodes{0};
    // How much of `nodes` has been published to live_nodes_. Lives on the context rather than in
    // run_dfs_from so the recursive terminal call shares it and cannot double count.
    int64_t accounted{0};

    void resize(i_t n, i_t m)
    {
      residual.assign(size_t(n + 1) * m, 0);
      branch.assign(n + 1, 0);
      value.assign(n, 0);
      nodes     = 0;
      accounted = 0;
    }
  };

  // One partially fixed subtree: the values of the trailing columns plus the residual they leave.
  struct subtree_t {
    std::vector<uint8_t> value;
    std::vector<coefficient_type> residual;
  };

  void build_tables();
  i_t choose_hash_depth() const;
  void build_hash();
  uint64_t residual_fingerprint(const coefficient_type* residual) const;
  dfs_result_t run_dfs_from(dfs_context_t& ctx,
                            i_t start_depth,
                            const coefficient_type* start_residual,
                            const std::atomic<bool>* stop,
                            i_t terminal_depth);
  // Enumerates the trailing `depth` columns, keeping the subtrees that survive pruning.
  void collect_subtrees(const std::vector<coefficient_type>& target,
                        i_t depth,
                        std::vector<subtree_t>& seeds);
  dfs_result_t run_dfs(const std::vector<coefficient_type>& target);
  bool enumerate_level(i_t level, std::vector<coefficient_type>& slack, i_t index, bool& found);
  // Rebuilds the assignment and verifies it against the untouched host copy of the problem.
  bool reconstruct(std::vector<f_t>& assignment) const;

  uint8_t joint_at(coefficient_type u0, coefficient_type u1) const
  {
    const size_t index = size_t(u0) * joint_stride_ + size_t(u1);
    return joint_[index];
  }

  f_t user_objective(f_t solver_obj) const { return obj_scale_ * (solver_obj + obj_offset_); }

  void print_table_header() const;
  void report(char symbol, bool have_incumbent);
  void maybe_report(double now);

  // The host model recognize() read, kept for the independent verification in reconstruct().
  std::unique_ptr<simplex::user_problem_t<i_t, f_t>> problem_;
  settings_t settings_;
  normalized_model_t model_;
  bool detected_{false};

  f_t obj_scale_{1};
  f_t obj_offset_{0};
  // Objective contribution of the columns pinned at a single value; the search only accounts for
  // the slacks, so this is what turns a level into a solver-space objective.
  f_t obj_offset_fixed_{0};

  std::vector<std::vector<uint8_t>> row_tables_;
  std::vector<uint8_t> joint_;
  std::vector<i_t> extra_rows_;  // rows outside the joint pair
  i_t joint_row0_{0};
  i_t joint_row1_{1};
  size_t joint_stride_{0};

  // Serial DFS state, allocated once and reused across every target vector.
  dfs_context_t context_;
  std::vector<uint8_t> value_;                 // the winning assignment
  std::vector<coefficient_type> found_slack_;  // the slack vector that produced it
  std::vector<coefficient_type> target_;
  i_t num_threads_{1};

  fingerprint_set_t hash_;
  i_t hash_depth_{0};  // zero means the terminal is disabled

  const std::atomic<bool>* preemption_{nullptr};
  timer_t timer_{std::numeric_limits<double>::infinity()};
  bool budget_exhausted_{false};

  // Progress state, written from the seed tasks as well as the driver.
  omp_atomic_t<int64_t> live_nodes_{0};
  omp_atomic_t<double> next_report_{0.0};
  omp_atomic_t<i_t> levels_exhausted_{0};
  f_t incumbent_{0};
};

}  // namespace cuopt::mathematical_optimization::mip
