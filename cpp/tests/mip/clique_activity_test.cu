/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#include "../linear_programming/utilities/pdlp_test_utilities.cuh"
#include "mip_utils.cuh"

#include <cuopt/linear_programming/solve.hpp>
#include <mip_heuristics/mip_constants.hpp>
#include <mip_heuristics/presolve/bounds_presolve.cuh>
#include <mip_heuristics/presolve/clique_activity_corrections.cuh>
#include <mip_heuristics/presolve/conflict_graph/clique_table.cuh>
#include <mip_heuristics/problem/problem.cuh>
#include <mps_parser/parser.hpp>
#include <utilities/common_utils.hpp>
#include <utilities/copy_helpers.hpp>
#include <utilities/error.hpp>

#include <raft/core/handle.hpp>
#include <raft/util/cudart_utils.hpp>
#include <rmm/device_scalar.hpp>

#include <gtest/gtest.h>

#include <algorithm>
#include <cstdint>
#include <limits>
#include <memory>
#include <unordered_set>
#include <vector>

namespace cuopt::linear_programming::test {

namespace {

// -----------------------------------------------------------------------------
// Helpers
// -----------------------------------------------------------------------------

// Build a minimal problem_t with the given CSR, binary-flag and variable types.
// After this call reverse CSR and is_binary_variable are populated, which is
// what clique_group_table_t::build_from_host reads.
std::shared_ptr<detail::problem_t<int, double>> make_problem(const raft::handle_t& handle,
                                                             const std::vector<int>& offsets,
                                                             const std::vector<int>& indices,
                                                             const std::vector<double>& values,
                                                             const std::vector<double>& cnst_lb,
                                                             const std::vector<double>& cnst_ub,
                                                             const std::vector<double>& var_lb,
                                                             const std::vector<double>& var_ub,
                                                             const std::vector<char>& var_types)
{
  mps_parser::mps_data_model_t<int, double> model;
  model.set_csr_constraint_matrix(values.data(),
                                  static_cast<int>(values.size()),
                                  indices.data(),
                                  static_cast<int>(indices.size()),
                                  offsets.data(),
                                  static_cast<int>(offsets.size()));
  model.set_constraint_lower_bounds(cnst_lb.data(), cnst_lb.size());
  model.set_constraint_upper_bounds(cnst_ub.data(), cnst_ub.size());
  model.set_variable_lower_bounds(var_lb.data(), var_lb.size());
  model.set_variable_upper_bounds(var_ub.data(), var_ub.size());
  std::vector<double> obj(var_lb.size(), 0.0);
  model.set_objective_coefficients(obj.data(), obj.size());
  model.set_variable_types(var_types);
  model.set_maximize(false);

  auto op = mps_data_model_to_optimization_problem(&handle, model);
  auto pb = std::make_shared<detail::problem_t<int, double>>(op);
  pb->preprocess_problem();
  return pb;
}

// Attach a manually-built clique_table with the given "large" clique list and
// the given adjacency-list small cliques. `n_vars` is the number of original
// problem variables. Using min_clique_size=1 disables any internal filtering
// (the clique_table_t members we care about are plain vectors/maps).
std::shared_ptr<detail::clique_table_t<int, double>> make_clique_table(
  int n_vars,
  const std::vector<std::vector<int>>& first,
  const std::vector<detail::addtl_clique_t<int, double>>& addtl     = {},
  const std::unordered_map<int, std::unordered_set<int>>& small_adj = {})
{
  auto ct = std::make_shared<detail::clique_table_t<int, double>>(
    /*n_vertices=*/2 * n_vars,
    /*min_clique_size=*/1,
    /*max_clique_size_for_extension=*/1);
  ct->first         = first;
  ct->addtl_cliques = addtl;
  for (auto const& kv : small_adj) {
    ct->adj_list_small_cliques[kv.first] = kv.second;
  }
  // var_clique_map_first is not read by build_from_host any more, but keep it
  // populated so it stays consistent with `first` for any other consumer.
  for (int ci = 0; ci < static_cast<int>(first.size()); ++ci) {
    for (int v : first[ci]) {
      if (v >= 0 && v < n_vars) { ct->var_clique_map_first[v].insert(ci); }
    }
  }
  return ct;
}

// -----------------------------------------------------------------------------
// Tests for clique_group_table_t::build_from_host
// -----------------------------------------------------------------------------

TEST(clique_activity, build_from_host_no_cliques)
{
  // One constraint: x0 + x1 + x2 + y >= 2, all binary, no clique table → no
  // groups should be emitted.
  const raft::handle_t handle{};
  auto pb = make_problem(handle,
                         {0, 4},
                         {0, 1, 2, 3},
                         {1.0, 1.0, 1.0, 1.0},
                         {2.0},
                         {std::numeric_limits<double>::infinity()},
                         {0.0, 0.0, 0.0, 0.0},
                         {1.0, 1.0, 1.0, 1.0},
                         {'I', 'I', 'I', 'I'});
  auto ct = make_clique_table(/*n_vars=*/4, /*first=*/{});

  detail::clique_group_table_t<int, double> data(handle.get_stream());
  data.build_from_host(*pb, *ct);
  EXPECT_TRUE(data.empty());
  EXPECT_EQ(data.n_groups, 0);
}

TEST(clique_activity, build_from_host_basic_large_clique)
{
  // One constraint: x0 + x1 + x2 + y >= 2, all binary. Clique {0,1,2} — we
  // expect exactly one group on constraint 0 with those three members.
  const raft::handle_t handle{};
  auto pb = make_problem(handle,
                         {0, 4},
                         {0, 1, 2, 3},
                         {1.0, 1.0, 1.0, 1.0},
                         {2.0},
                         {std::numeric_limits<double>::infinity()},
                         {0.0, 0.0, 0.0, 0.0},
                         {1.0, 1.0, 1.0, 1.0},
                         {'I', 'I', 'I', 'I'});
  auto ct = make_clique_table(/*n_vars=*/4, /*first=*/{{0, 1, 2}});

  detail::clique_group_table_t<int, double> data(handle.get_stream());
  data.build_from_host(*pb, *ct);
  ASSERT_FALSE(data.empty());
  ASSERT_EQ(data.n_groups, 1);

  auto stream       = handle.get_stream();
  auto h_cnst_ids   = host_copy(data.group_constraint_ids, stream);
  auto h_mem_off    = host_copy(data.group_member_offsets, stream);
  auto h_mem_vars   = host_copy(data.group_member_vars, stream);
  auto h_mem_coeffs = host_copy(data.group_member_coeffs, stream);
  auto h_cg_off     = host_copy(data.constraint_group_offsets, stream);
  auto h_rev_gid    = host_copy(data.reverse_group_id, stream);

  ASSERT_EQ(h_cnst_ids.size(), 1u);
  EXPECT_EQ(h_cnst_ids[0], 0);
  ASSERT_EQ(h_mem_off.size(), 2u);
  EXPECT_EQ(h_mem_off[0], 0);
  EXPECT_EQ(h_mem_off[1], 3);
  ASSERT_EQ(h_mem_vars.size(), 3u);
  // member set must be {0, 1, 2}
  std::vector<int> sorted_vars = h_mem_vars;
  std::sort(sorted_vars.begin(), sorted_vars.end());
  EXPECT_EQ(sorted_vars, (std::vector<int>{0, 1, 2}));
  for (double c : h_mem_coeffs)
    EXPECT_DOUBLE_EQ(c, 1.0);

  // constraint_group_offsets: [0, 1] (1 constraint, 1 group)
  ASSERT_EQ(h_cg_off.size(), 2u);
  EXPECT_EQ(h_cg_off[0], 0);
  EXPECT_EQ(h_cg_off[1], 1);

  // reverse_group_id: length = nnz = 4. For vars 0,1,2 (each in one
  // constraint), reverse_group_id entry should be 0. For y (var 3), it
  // should be -1.
  ASSERT_EQ(h_rev_gid.size(), 4u);
  int count_in_group = 0;
  int count_out      = 0;
  for (int g : h_rev_gid) {
    if (g == 0)
      count_in_group++;
    else if (g == -1)
      count_out++;
  }
  EXPECT_EQ(count_in_group, 3);
  EXPECT_EQ(count_out, 1);
}

TEST(clique_activity, build_from_host_skips_complement_literal_clique)
{
  // Clique contains a complement literal (v >= n_vars). The first-implementation
  // code drops cliques that mix complements, so no group should be emitted.
  const raft::handle_t handle{};
  const int n_vars = 4;
  auto pb          = make_problem(handle,
                                  {0, 4},
                                  {0, 1, 2, 3},
                                  {1.0, 1.0, 1.0, 1.0},
                                  {2.0},
                                  {std::numeric_limits<double>::infinity()},
                                  {0.0, 0.0, 0.0, 0.0},
                                  {1.0, 1.0, 1.0, 1.0},
                                  {'I', 'I', 'I', 'I'});
  auto ct =
    make_clique_table(n_vars, /*first=*/{{0, 1, n_vars + 2}});  // last entry is complement of x2

  detail::clique_group_table_t<int, double> data(handle.get_stream());
  data.build_from_host(*pb, *ct);
  EXPECT_TRUE(data.empty());
}

TEST(clique_activity, build_from_host_small_adj_list_fallback)
{
  // No large cliques, but pairwise edges {(0,1), (1,2), (0,2)} on an adj list.
  // Since all three vars appear in the same constraint, build_from_host should
  // extract the triangle as a single group via its greedy maximal-clique path.
  const raft::handle_t handle{};
  auto pb = make_problem(handle,
                         {0, 3},
                         {0, 1, 2},
                         {1.0, 1.0, 1.0},
                         {-std::numeric_limits<double>::infinity()},
                         {1.0},
                         {0.0, 0.0, 0.0},
                         {1.0, 1.0, 1.0},
                         {'I', 'I', 'I'});

  std::unordered_map<int, std::unordered_set<int>> adj;
  adj[0]  = {1, 2};
  adj[1]  = {0, 2};
  adj[2]  = {0, 1};
  auto ct = make_clique_table(/*n_vars=*/3, /*first=*/{}, /*addtl=*/{}, adj);

  detail::clique_group_table_t<int, double> data(handle.get_stream());
  data.build_from_host(*pb, *ct);
  ASSERT_EQ(data.n_groups, 1);
  auto stream = handle.get_stream();
  auto h_mem  = host_copy(data.group_member_vars, stream);
  std::sort(h_mem.begin(), h_mem.end());
  EXPECT_EQ(h_mem, (std::vector<int>{0, 1, 2}));
}

// -----------------------------------------------------------------------------
// Tests for compute_clique_corrections_kernel
// -----------------------------------------------------------------------------

TEST(clique_activity, compute_corrections_kernel_values)
{
  // One group, three members with coeffs {3, 5, 2}. lb=0, ub=1 for all.
  // Expected:
  //   sum_pos = 10, max1 = 5, max2 = 3 → max_correction = 10 - 5 = 5
  //   sum_neg = 0,  min1 = 0, min2 = 0 → min_correction = 0
  const raft::handle_t handle{};
  auto stream = handle.get_stream();

  std::vector<int> h_offsets   = {0, 3};
  std::vector<int> h_vars      = {0, 1, 2};
  std::vector<double> h_coeffs = {3.0, 5.0, 2.0};
  std::vector<double> h_lb     = {0.0, 0.0, 0.0};
  std::vector<double> h_ub     = {1.0, 1.0, 1.0};

  auto d_offsets = device_copy(h_offsets, stream);
  auto d_vars    = device_copy(h_vars, stream);
  auto d_coeffs  = device_copy(h_coeffs, stream);
  auto d_lb      = device_copy(h_lb, stream);
  auto d_ub      = device_copy(h_ub, stream);

  rmm::device_uvector<double> d_max_corr(1, stream);
  rmm::device_uvector<double> d_min_corr(1, stream);
  rmm::device_uvector<double> d_max1(1, stream);
  rmm::device_uvector<double> d_max2(1, stream);
  rmm::device_uvector<double> d_min1(1, stream);
  rmm::device_uvector<double> d_min2(1, stream);

  constexpr int warp = raft::WarpSize;
  detail::compute_clique_corrections_kernel<int, double, warp>
    <<<1, warp, 0, stream>>>(make_span(d_offsets),
                             make_span(d_vars),
                             make_span(d_coeffs),
                             make_span(d_lb),
                             make_span(d_ub),
                             make_span(d_max_corr),
                             make_span(d_min_corr),
                             make_span(d_max1),
                             make_span(d_max2),
                             make_span(d_min1),
                             make_span(d_min2),
                             /*int_tol=*/1e-6);
  RAFT_CHECK_CUDA(stream);

  auto h_max_corr = host_copy(d_max_corr, stream);
  auto h_min_corr = host_copy(d_min_corr, stream);
  auto h_max1_h   = host_copy(d_max1, stream);
  auto h_max2_h   = host_copy(d_max2, stream);
  auto h_min1_h   = host_copy(d_min1, stream);
  auto h_min2_h   = host_copy(d_min2, stream);

  EXPECT_DOUBLE_EQ(h_max_corr[0], 5.0);
  EXPECT_DOUBLE_EQ(h_min_corr[0], 0.0);
  EXPECT_DOUBLE_EQ(h_max1_h[0], 5.0);
  EXPECT_DOUBLE_EQ(h_max2_h[0], 3.0);
  EXPECT_DOUBLE_EQ(h_min1_h[0], 0.0);
  EXPECT_DOUBLE_EQ(h_min2_h[0], 0.0);
}

TEST(clique_activity, compute_corrections_kernel_skips_fixed_members)
{
  // Same coeffs as above, but x1 (coeff=5) is fixed (lb==ub). With only x0 and
  // x2 unfixed (coeffs 3 and 2), we expect:
  //   sum_pos = 5, max1 = 3, max2 = 2 → max_correction = 5 - 3 = 2
  const raft::handle_t handle{};
  auto stream = handle.get_stream();

  std::vector<int> h_offsets   = {0, 3};
  std::vector<int> h_vars      = {0, 1, 2};
  std::vector<double> h_coeffs = {3.0, 5.0, 2.0};
  std::vector<double> h_lb     = {0.0, 1.0, 0.0};
  std::vector<double> h_ub     = {1.0, 1.0, 1.0};

  auto d_offsets = device_copy(h_offsets, stream);
  auto d_vars    = device_copy(h_vars, stream);
  auto d_coeffs  = device_copy(h_coeffs, stream);
  auto d_lb      = device_copy(h_lb, stream);
  auto d_ub      = device_copy(h_ub, stream);

  rmm::device_uvector<double> d_max_corr(1, stream);
  rmm::device_uvector<double> d_min_corr(1, stream);
  rmm::device_uvector<double> d_max1(1, stream);
  rmm::device_uvector<double> d_max2(1, stream);
  rmm::device_uvector<double> d_min1(1, stream);
  rmm::device_uvector<double> d_min2(1, stream);

  constexpr int warp = raft::WarpSize;
  detail::compute_clique_corrections_kernel<int, double, warp>
    <<<1, warp, 0, stream>>>(make_span(d_offsets),
                             make_span(d_vars),
                             make_span(d_coeffs),
                             make_span(d_lb),
                             make_span(d_ub),
                             make_span(d_max_corr),
                             make_span(d_min_corr),
                             make_span(d_max1),
                             make_span(d_max2),
                             make_span(d_min1),
                             make_span(d_min2),
                             1e-6);
  RAFT_CHECK_CUDA(stream);

  auto h_max_corr = host_copy(d_max_corr, stream);
  auto h_max1_h   = host_copy(d_max1, stream);
  auto h_max2_h   = host_copy(d_max2, stream);

  EXPECT_DOUBLE_EQ(h_max_corr[0], 2.0);
  EXPECT_DOUBLE_EQ(h_max1_h[0], 3.0);
  EXPECT_DOUBLE_EQ(h_max2_h[0], 2.0);
}

// -----------------------------------------------------------------------------
// End-to-end bound propagation: clique should tighten bounds that the stock
// algorithm can't reach.
// -----------------------------------------------------------------------------

TEST(clique_activity, bound_propagation_tightens_with_clique)
{
  // One constraint: x0 + x1 + x2 + y >= 2, all binary [0, 1].
  //
  // Stock activity tightening (no clique):
  //   max_a = 4. For y: max_a_without_y = 3. new_lb_y = ceil((2 - 3)/1) = -1
  //   → no tightening (y stays at [0, 1]).
  //
  // Clique-aware (clique {0, 1, 2}):
  //   sum_pos = 3, max1 = 1 → max_correction = 2. Corrected max_a = 4 - 2 = 2.
  //   For y (not in clique): max_a_without_y = 2 - 1 = 1. new_lb_y =
  //   ceil((2 - 1)/1) = 1 → y's lb tightens to 1.
  const raft::handle_t handle{};
  auto pb = make_problem(handle,
                         {0, 4},
                         {0, 1, 2, 3},
                         {1.0, 1.0, 1.0, 1.0},
                         {2.0},
                         {std::numeric_limits<double>::infinity()},
                         {0.0, 0.0, 0.0, 0.0},
                         {1.0, 1.0, 1.0, 1.0},
                         {'I', 'I', 'I', 'I'});

  mip_solver_settings_t<int, double> settings{};
  detail::mip_solver_context_t<int, double> context(pb->handle_ptr, pb.get(), settings);

  // --- Baseline: no clique table ------------------------------------------
  detail::bound_presolve_t<int, double> bp_no_clique(context);
  bp_no_clique.solve(*pb);
  auto stream    = handle.get_stream();
  auto h_lb_base = host_copy(bp_no_clique.upd.lb, stream);
  auto h_ub_base = host_copy(bp_no_clique.upd.ub, stream);

  // --- With clique: attach {x0, x1, x2} and re-run ------------------------
  pb->clique_table = make_clique_table(/*n_vars=*/4, /*first=*/{{0, 1, 2}});

  detail::bound_presolve_t<int, double> bp_with_clique(context);
  bp_with_clique.solve(*pb);
  auto h_lb_cliq = host_copy(bp_with_clique.upd.lb, stream);
  auto h_ub_cliq = host_copy(bp_with_clique.upd.ub, stream);

  // Baseline should leave y's lb at 0; clique-aware should lift it to 1.
  EXPECT_DOUBLE_EQ(h_lb_base[3], 0.0);
  EXPECT_DOUBLE_EQ(h_lb_cliq[3], 1.0);

  // The three clique members stay in [0, 1] (no direct tightening — the
  // constraint lb of 2 is still achievable by picking any two of them to 1,
  // but that remains feasible per-variable).
  for (int i = 0; i < 3; ++i) {
    EXPECT_DOUBLE_EQ(h_lb_cliq[i], 0.0);
    EXPECT_DOUBLE_EQ(h_ub_cliq[i], 1.0);
  }
}

TEST(clique_activity, bound_propagation_matches_when_clique_is_noop)
{
  // A problem where the clique has no extra tightening power: constraint
  // x0 + x1 <= 1 is already a set-packing constraint and its max_activity is
  // 2, clique-corrected max_activity is 1. But there are no other variables
  // to propagate onto, and x0, x1 already have ub = 1, so bounds can't
  // tighten further. Both runs should produce identical bounds.
  const raft::handle_t handle{};
  auto pb = make_problem(handle,
                         {0, 2},
                         {0, 1},
                         {1.0, 1.0},
                         {-std::numeric_limits<double>::infinity()},
                         {1.0},
                         {0.0, 0.0},
                         {1.0, 1.0},
                         {'I', 'I'});

  mip_solver_settings_t<int, double> settings{};
  detail::mip_solver_context_t<int, double> context(pb->handle_ptr, pb.get(), settings);

  detail::bound_presolve_t<int, double> bp_no(context);
  bp_no.solve(*pb);
  auto stream  = handle.get_stream();
  auto h_lb_no = host_copy(bp_no.upd.lb, stream);
  auto h_ub_no = host_copy(bp_no.upd.ub, stream);

  pb->clique_table = make_clique_table(/*n_vars=*/2, /*first=*/{{0, 1}});
  detail::bound_presolve_t<int, double> bp_yes(context);
  bp_yes.solve(*pb);
  auto h_lb_yes = host_copy(bp_yes.upd.lb, stream);
  auto h_ub_yes = host_copy(bp_yes.upd.ub, stream);

  ASSERT_EQ(h_lb_no.size(), h_lb_yes.size());
  for (size_t i = 0; i < h_lb_no.size(); ++i) {
    EXPECT_DOUBLE_EQ(h_lb_no[i], h_lb_yes[i]) << "lb mismatch at var " << i;
    EXPECT_DOUBLE_EQ(h_ub_no[i], h_ub_yes[i]) << "ub mismatch at var " << i;
  }
}

}  // namespace

}  // namespace cuopt::linear_programming::test
