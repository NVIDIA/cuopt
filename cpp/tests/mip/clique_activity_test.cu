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
#include <utilities/timer.hpp>

#include <raft/core/handle.hpp>
#include <raft/util/cudart_utils.hpp>
#include <rmm/device_scalar.hpp>

#include <gtest/gtest.h>

#include <algorithm>
#include <cstdint>
#include <limits>
#include <memory>
#include <string>
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
  // Convert legacy hash-of-set small adjacency into the CSR. Production code
  // never goes through this path — remove_small_cliques populates the CSR
  // directly. The helper is sufficient here because tests only need a stable
  // read-only view of edges.
  ct->set_small_clique_adj_for_test(small_adj);
  // Materialize var_clique_first / var_clique_addtl / first_var_positions
  // from `first` and `addtl_cliques`. This mirrors what
  // build_clique_table()/find_initial_cliques() do at the end of build.
  detail::fill_var_clique_maps(*ct);
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
  data.build_from_host(*pb, /*primary_reverse_original_ids=*/{}, *ct);
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
  data.build_from_host(*pb, /*primary_reverse_original_ids=*/{}, *ct);
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

TEST(clique_activity, build_from_host_accepts_complement_literal_clique)
{
  // Clique {x0, x1, ~x2} on constraint x0 + x1 + x2 + y >= 2. The third
  // member is a complement literal, so its effective literal coefficient is
  // -1 * 1 = -1 while the first two stay at +1. The group should be emitted
  // with those three underlying vars and coeffs {+1, +1, -1}.
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
  data.build_from_host(*pb, /*primary_reverse_original_ids=*/{}, *ct);
  ASSERT_EQ(data.n_groups, 1);

  auto stream       = handle.get_stream();
  auto h_mem_vars   = host_copy(data.group_member_vars, stream);
  auto h_mem_coeffs = host_copy(data.group_member_coeffs, stream);
  ASSERT_EQ(h_mem_vars.size(), 3u);
  // Build (var → effective coeff) map so we can check by identity.
  std::unordered_map<int, double> got;
  for (std::size_t i = 0; i < h_mem_vars.size(); ++i) {
    got[h_mem_vars[i]] = h_mem_coeffs[i];
  }
  ASSERT_EQ(got.size(), 3u);
  EXPECT_DOUBLE_EQ(got[0], 1.0);
  EXPECT_DOUBLE_EQ(got[1], 1.0);
  EXPECT_DOUBLE_EQ(got[2], -1.0);
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
  data.build_from_host(*pb, /*primary_reverse_original_ids=*/{}, *ct);
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
  std::vector<int> h_gcnst     = {0};
  std::vector<int> h_chgcnst   = {1};
  std::vector<double> h_lb     = {0.0, 0.0, 0.0};
  std::vector<double> h_ub     = {1.0, 1.0, 1.0};

  auto d_offsets = device_copy(h_offsets, stream);
  auto d_vars    = device_copy(h_vars, stream);
  auto d_coeffs  = device_copy(h_coeffs, stream);
  auto d_gcnst   = device_copy(h_gcnst, stream);
  auto d_chgcnst = device_copy(h_chgcnst, stream);
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
                             make_span(d_gcnst),
                             make_span(d_chgcnst),
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
  std::vector<int> h_gcnst     = {0};
  std::vector<int> h_chgcnst   = {1};
  std::vector<double> h_lb     = {0.0, 1.0, 0.0};
  std::vector<double> h_ub     = {1.0, 1.0, 1.0};

  auto d_offsets = device_copy(h_offsets, stream);
  auto d_vars    = device_copy(h_vars, stream);
  auto d_coeffs  = device_copy(h_coeffs, stream);
  auto d_gcnst   = device_copy(h_gcnst, stream);
  auto d_chgcnst = device_copy(h_chgcnst, stream);
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
                             make_span(d_gcnst),
                             make_span(d_chgcnst),
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

TEST(clique_activity, bound_propagation_tightens_with_complement_literal_clique)
{
  // Constraint: x0 + x1 - x2 + y >= 1, all binary.
  //   Stock max_a = 1 + 1 + 0 + 1 = 3.
  // Clique: { x0, x1, ~x2 } — at most one of {x0=1, x1=1, x2=0} is true.
  // Effective literal coeffs b = {+1, +1, -1*(-1)} = {+1, +1, +1}.
  //   sum_pos = 3, max1 = 1 → max_correction = 2. Corrected max_a = 1.
  //   For y (not in the clique): max_a_without_y = 1 - 1 = 0.
  //   new_lb_y = ceil((1 - 0)/1) = 1.
  const raft::handle_t handle{};
  const int n_vars = 4;
  auto pb          = make_problem(handle,
                                  {0, 4},
                                  {0, 1, 2, 3},
                                  {1.0, 1.0, -1.0, 1.0},
                                  {1.0},
                                  {std::numeric_limits<double>::infinity()},
                                  {0.0, 0.0, 0.0, 0.0},
                                  {1.0, 1.0, 1.0, 1.0},
                                  {'I', 'I', 'I', 'I'});

  mip_solver_settings_t<int, double> settings{};
  detail::mip_solver_context_t<int, double> context(pb->handle_ptr, pb.get(), settings);

  detail::bound_presolve_t<int, double> bp_no_clique(context);
  bp_no_clique.solve(*pb);
  auto stream    = handle.get_stream();
  auto h_lb_base = host_copy(bp_no_clique.upd.lb, stream);
  EXPECT_DOUBLE_EQ(h_lb_base[3], 0.0);

  pb->clique_table = make_clique_table(n_vars, /*first=*/{{0, 1, n_vars + 2}});
  detail::bound_presolve_t<int, double> bp_with_clique(context);
  bp_with_clique.solve(*pb);
  auto h_lb_cliq = host_copy(bp_with_clique.upd.lb, stream);
  EXPECT_DOUBLE_EQ(h_lb_cliq[3], 1.0);
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

// -----------------------------------------------------------------------------
// Clique-aware vs stock bound propagation: monotonicity invariant.
//
// Theoretical claim (proved in CLIQUE_PIPELINE_AUDIT.md / kernel comment):
// for any feasible problem and any valid clique table on it, the clique-aware
// bound presolve must produce a feasible region that is a (possibly improper)
// subset of the stock presolve's region. Concretely, on every variable:
//     lb_stock  <=  lb_clique
//     ub_stock  >=  ub_clique
// and on every constraint, the converged stored activity (which in the
// clique case is already clique-corrected by apply_clique_corrections):
//     min_activity_stock  <=  min_activity_clique
//     max_activity_stock  >=  max_activity_clique
// Tighter bounds → tighter raw activity (smaller ub_v shrinks max_a; larger
// lb_v lifts min_a), and the per-iter clique correction can only push
// further in the same direction (max_correction >= 0 subtracted from max_a;
// min_correction <= 0 subtracted from min_a → adds a non-negative amount).
// -----------------------------------------------------------------------------

namespace {

struct presolve_result_t {
  std::vector<double> lb;
  std::vector<double> ub;
  std::vector<double> min_a;
  std::vector<double> max_a;
};

// Run bound_presolve_t::solve once and snapshot the converged buffers.
// `ct` may be null to exercise the stock (no-clique) path; otherwise it is
// attached to `pb` before solve. This helper is intentionally side-effecting
// on `pb->clique_table` so callers can pair it across two runs by simply
// passing nullptr the first time and a real table the second.
presolve_result_t run_presolve(detail::problem_t<int, double>& pb,
                               std::shared_ptr<detail::clique_table_t<int, double>> ct)
{
  pb.clique_table = std::move(ct);
  mip_solver_settings_t<int, double> settings{};
  detail::mip_solver_context_t<int, double> context(pb.handle_ptr, &pb, settings);
  detail::bound_presolve_t<int, double> bp(context);
  bp.solve(pb);
  auto stream = pb.handle_ptr->get_stream();
  return {host_copy(bp.upd.lb, stream),
          host_copy(bp.upd.ub, stream),
          host_copy(bp.upd.min_activity, stream),
          host_copy(bp.upd.max_activity, stream)};
}

// Assert: clique tightens or matches on every var and every constraint.
// `tol` accounts for the FP roundoff that piggy-backs on the apply / undo of
// the per-group correction along long iteration chains; on these tiny test
// problems it should always be safe at 1e-9.
void expect_clique_tighter_or_equal(const presolve_result_t& stock,
                                    const presolve_result_t& clique,
                                    double tol = 1e-9)
{
  ASSERT_EQ(stock.lb.size(), clique.lb.size());
  ASSERT_EQ(stock.ub.size(), clique.ub.size());
  ASSERT_EQ(stock.min_a.size(), clique.min_a.size());
  ASSERT_EQ(stock.max_a.size(), clique.max_a.size());

  for (std::size_t v = 0; v < stock.lb.size(); ++v) {
    EXPECT_LE(stock.lb[v], clique.lb[v] + tol)
      << "lb LOOSER under clique at var " << v << " (stock=" << stock.lb[v]
      << ", clique=" << clique.lb[v] << ")";
    EXPECT_GE(stock.ub[v], clique.ub[v] - tol)
      << "ub LOOSER under clique at var " << v << " (stock=" << stock.ub[v]
      << ", clique=" << clique.ub[v] << ")";
  }
  for (std::size_t c = 0; c < stock.min_a.size(); ++c) {
    // Skip constraints whose stock activity is +/-inf (never happens in the
    // test problems below, but the guard keeps the helper robust if a future
    // test adds a free constraint).
    const double inf = std::numeric_limits<double>::infinity();
    if (std::abs(stock.min_a[c]) == inf || std::abs(stock.max_a[c]) == inf) continue;
    EXPECT_LE(stock.min_a[c], clique.min_a[c] + tol)
      << "min_activity LOOSER under clique at cnst " << c << " (stock=" << stock.min_a[c]
      << ", clique=" << clique.min_a[c] << ")";
    EXPECT_GE(stock.max_a[c], clique.max_a[c] - tol)
      << "max_activity LOOSER under clique at cnst " << c << " (stock=" << stock.max_a[c]
      << ", clique=" << clique.max_a[c] << ")";
  }
}

// Ground-truth check: at least ONE var or constraint must tighten strictly,
// otherwise the test problem is uninformative (a clique that does not move
// any number is a degenerate test for our invariant). Used in tests where we
// intentionally crafted the problem so the clique path is strictly stronger.
void expect_some_strict_tightening(const presolve_result_t& stock,
                                   const presolve_result_t& clique,
                                   double tol = 1e-9)
{
  bool found = false;
  for (std::size_t v = 0; v < stock.lb.size(); ++v) {
    if (clique.lb[v] > stock.lb[v] + tol) {
      found = true;
      break;
    }
    if (clique.ub[v] < stock.ub[v] - tol) {
      found = true;
      break;
    }
  }
  if (!found) {
    for (std::size_t c = 0; c < stock.min_a.size(); ++c) {
      if (clique.min_a[c] > stock.min_a[c] + tol) {
        found = true;
        break;
      }
      if (clique.max_a[c] < stock.max_a[c] - tol) {
        found = true;
        break;
      }
    }
  }
  EXPECT_TRUE(found) << "Test problem was supposed to demonstrate clique-aware tightening but "
                        "every var/constraint matched stock exactly. Either the problem is "
                        "degenerate or the clique is not being consumed.";
}

}  // namespace

TEST(clique_activity, monotonicity_cascading_tightening)
{
  // Two-stage cascade:
  //   c0:  x0 + x1 + x2 + y0 >= 2          (clique {x0,x1,x2})
  //   c1:  y0 + z <= 5                     (no clique, z continuous in [0,10])
  //
  // Stock:
  //   c0: max_a = 4. max_a_without_y0 = 3. new_lb_y0 = ceil((2-3)/1) = -1
  //       → y0 stays at [0, 1].
  //   c1: min_a = 0. min_a_without_z = 0. new_ub_z = (5-0)/1 = 5.
  //       → z tightens to [0, 5] (just clamps the trivial ub).
  //   At fixed point: y0 ∈ [0, 1], z ∈ [0, 5].
  //
  // Clique:
  //   c0 corrected max_a = 4 - (sum_pos - max1) = 4 - (3 - 1) = 2.
  //       max_a_without_y0 = 1. new_lb_y0 = ceil((2-1)/1) = 1 → y0 ∈ [1, 1].
  //   c1: y0 now has lb=1. raw min_a = 1 + 0 = 1. min_a_without_z = 1.
  //       new_ub_z = (5 - 1) / 1 = 4 → z ∈ [0, 4].
  //   Fixed point: y0 = 1, z ∈ [0, 4]. Strictly tighter on y0.lb and z.ub.
  //
  // Invariant must hold on every var and every constraint.
  const raft::handle_t handle{};
  const int n_vars = 5;  // x0, x1, x2, y0, z
  const double inf = std::numeric_limits<double>::infinity();
  auto pb          = make_problem(handle,
                                  {0, 4, 6},
                                  {0, 1, 2, 3, /*c1*/ 3, 4},
                                  {1.0, 1.0, 1.0, 1.0, 1.0, 1.0},
                                  {2.0, -inf},
                                  {inf, 5.0},
                                  {0.0, 0.0, 0.0, 0.0, 0.0},
                                  {1.0, 1.0, 1.0, 1.0, 10.0},
                                  {'I', 'I', 'I', 'I', 'C'});

  auto stock_pb  = run_presolve(*pb, /*ct=*/nullptr);
  auto clique_pb = run_presolve(*pb, make_clique_table(n_vars, /*first=*/{{0, 1, 2}}));

  expect_clique_tighter_or_equal(stock_pb, clique_pb);

  // Sanity-pin the cascade outcomes so a regression in either propagation
  // direction is caught locally, not just via the generic invariant.
  EXPECT_DOUBLE_EQ(stock_pb.lb[3], 0.0);   // y0 unchanged in stock
  EXPECT_DOUBLE_EQ(stock_pb.ub[4], 5.0);   // z capped at constraint ub in stock
  EXPECT_DOUBLE_EQ(clique_pb.lb[3], 1.0);  // y0 lifted by clique correction
  EXPECT_DOUBLE_EQ(clique_pb.ub[4], 4.0);  // z further tightened via raised y0

  expect_some_strict_tightening(stock_pb, clique_pb);
}

TEST(clique_activity, monotonicity_independent_cliques_multi_constraint)
{
  // Two independent cliques on two parallel constraints, plus a third
  // coupling constraint that aggregates the "extras":
  //   c0:  x0 + x1 + x2 + y0 >= 2          (clique {x0,x1,x2})
  //   c1:  x3 + x4 + x5 + y1 >= 2          (clique {x3,x4,x5})
  //   c2:  y0 + y1 + z <= 3                (no clique, z ∈ [0, 10] continuous)
  //
  // Stock:  y0 and y1 stay at [0,1]; min_a(c2) = 0 → z ub stays at 3 (from
  //         the cnst ub minus min_a_without_z).
  // Clique: y0 ≥ 1 and y1 ≥ 1 from c0/c1 cliques; then min_a(c2) = 2 →
  //         new_ub_z = 3 - 2 = 1.
  const raft::handle_t handle{};
  const int n_vars = 9;  // x0..x5, y0, y1, z
  const double inf = std::numeric_limits<double>::infinity();
  auto pb          = make_problem(handle,
                         /*offsets=*/{0, 4, 8, 11},
                         /*indices=*/{0, 1, 2, 6, 3, 4, 5, 7, 6, 7, 8},
                         /*values=*/{1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1},
                         /*cnst_lb=*/{2.0, 2.0, -inf},
                         /*cnst_ub=*/{inf, inf, 3.0},
                         /*var_lb=*/{0, 0, 0, 0, 0, 0, 0, 0, 0},
                         /*var_ub=*/{1, 1, 1, 1, 1, 1, 1, 1, 10},
                         /*var_types=*/{'I', 'I', 'I', 'I', 'I', 'I', 'I', 'I', 'C'});

  auto stock_pb  = run_presolve(*pb, /*ct=*/nullptr);
  auto clique_pb = run_presolve(*pb, make_clique_table(n_vars, /*first=*/{{0, 1, 2}, {3, 4, 5}}));

  expect_clique_tighter_or_equal(stock_pb, clique_pb);

  // Spot-check the cascade: both y's lifted, z reduced by 2.
  EXPECT_DOUBLE_EQ(stock_pb.lb[6], 0.0);
  EXPECT_DOUBLE_EQ(stock_pb.lb[7], 0.0);
  EXPECT_DOUBLE_EQ(clique_pb.lb[6], 1.0);
  EXPECT_DOUBLE_EQ(clique_pb.lb[7], 1.0);
  EXPECT_GE(stock_pb.ub[8] + 1e-9, clique_pb.ub[8]);
  EXPECT_LE(clique_pb.ub[8], 1.0 + 1e-9);  // z ub at most 1 in clique

  expect_some_strict_tightening(stock_pb, clique_pb);
}

TEST(clique_activity, monotonicity_complement_literal_clique)
{
  // Same shape as the "tightens with complement literal clique" test, but we
  // now also pin the activity-side invariant explicitly, since complement
  // members feed the symmetric (min-side) correction path through the
  // signed b_v = sign_v * a_v rewrite. Constraint:
  //   c0:  x0 + x1 - x2 + y >= 1           (clique {x0, x1, ~x2})
  // Stock: y stays at [0,1]; clique lifts y to 1.
  const raft::handle_t handle{};
  const int n_vars = 4;
  const double inf = std::numeric_limits<double>::infinity();
  auto pb          = make_problem(handle,
                                  {0, 4},
                                  {0, 1, 2, 3},
                                  {1.0, 1.0, -1.0, 1.0},
                                  {1.0},
                                  {inf},
                                  {0.0, 0.0, 0.0, 0.0},
                                  {1.0, 1.0, 1.0, 1.0},
                                  {'I', 'I', 'I', 'I'});

  auto stock_pb  = run_presolve(*pb, /*ct=*/nullptr);
  auto clique_pb = run_presolve(*pb, make_clique_table(n_vars, /*first=*/{{0, 1, n_vars + 2}}));

  expect_clique_tighter_or_equal(stock_pb, clique_pb);
  EXPECT_DOUBLE_EQ(stock_pb.lb[3], 0.0);
  EXPECT_DOUBLE_EQ(clique_pb.lb[3], 1.0);
  expect_some_strict_tightening(stock_pb, clique_pb);
}

TEST(clique_activity, monotonicity_holds_when_clique_is_noop)
{
  // Reuse the "no-op clique" shape: the clique structure exists but the
  // problem cannot tighten further on any variable. The invariant must
  // still hold trivially (everything matches), and no LOOSENING must occur
  // on any var/cnst as a side effect of the corrections being applied and
  // then undone by the per-cnst stock-minus-true peel-off in
  // update_bounds_per_cnst_cliq.
  const raft::handle_t handle{};
  const double inf = std::numeric_limits<double>::infinity();
  auto pb          = make_problem(
    handle, {0, 2}, {0, 1}, {1.0, 1.0}, {-inf}, {1.0}, {0.0, 0.0}, {1.0, 1.0}, {'I', 'I'});

  auto stock_pb  = run_presolve(*pb, /*ct=*/nullptr);
  auto clique_pb = run_presolve(*pb, make_clique_table(/*n_vars=*/2, /*first=*/{{0, 1}}));

  expect_clique_tighter_or_equal(stock_pb, clique_pb);

  // Strict equality on the no-op problem: clique correction must not
  // actively MOVE any number, only "fail to" tighten further. This is the
  // double-correction guard of apply_clique_corrections_to_activity_kernel
  // in action — any drift here would mean the kernel is over-applying.
  ASSERT_EQ(stock_pb.lb.size(), clique_pb.lb.size());
  for (std::size_t i = 0; i < stock_pb.lb.size(); ++i) {
    EXPECT_DOUBLE_EQ(stock_pb.lb[i], clique_pb.lb[i]) << "lb drift at var " << i;
    EXPECT_DOUBLE_EQ(stock_pb.ub[i], clique_pb.ub[i]) << "ub drift at var " << i;
  }
}

}  // namespace

// -----------------------------------------------------------------------------
// Clique-aware vs stock bound propagation: real MIPLIB-style instances.
//
// The synthetic monotonicity tests above pin specific propagation chains by
// hand. To guard against regressions that would only fire on production-shape
// problems (dense conflict graphs, many small cliques, mixed sign coeffs,
// non-trivial preprocess output), we additionally run the same monotonicity
// invariant on a handful of small but binary-heavy instances that ship in
// `datasets/mip/` (i.e., are checked in to git so CI will always have them).
//
// Selection criteria for the instances:
//   - git-tracked under `datasets/mip/` (no MIPLIB download required);
//   - mostly or entirely binary (so the conflict-graph build has material
//     to work with — `clique_table_t::extend` and the small-clique CSR are
//     only populated for binary literals);
//   - small enough that the two back-to-back bound presolves stay within a
//     normal unit-test budget on a single GPU.
//
// Each instance test:
//   1) Parses the MPS, builds `problem_t`, runs `preprocess_problem()`.
//   2) Builds a real `clique_table_t` via the production
//      `detail::build_clique_table` entry point, with the same
//      remove_small_cliques/extend toggles the solver uses.
//   3) Runs `bound_presolve_t::solve` once with `pb.clique_table = nullptr`
//      (stock) and once with the real clique table (clique-aware), reusing
//      the same `problem_t` so the two runs see identical preprocessed
//      input.
//   4) Asserts the monotonicity invariant on every var and every constraint.
//      We do NOT require strict tightening on real instances — for some
//      problems the clique structure is redundant with the LP tightening
//      already done by stock prop, and matching is a perfectly valid
//      outcome. We only assert "tighter or equal".
//
// Tolerance: 1e-6. Real instances cycle through many bound updates, and the
// per-iteration apply / undo of clique corrections accumulates O(n_iter * eps)
// FP roundoff that is invisible on the synthetic problems but can cross 1e-9.
// The looser tolerance is still tight enough to catch any actual loosening.
// -----------------------------------------------------------------------------

namespace {

struct mps_problem_with_cliques_t {
  std::shared_ptr<detail::problem_t<int, double>> pb;
  std::shared_ptr<detail::clique_table_t<int, double>> ct;
};

// Parse a checked-in MPS file under datasets/mip/, build the preprocessed
// `problem_t`, and produce a real `clique_table_t` for it via the same code
// path the production solver uses (find_initial_cliques + remove_small +
// extend). Both outputs are returned so the caller can pair them across two
// `bound_presolve_t::solve` invocations.
mps_problem_with_cliques_t load_mps_with_cliques(const raft::handle_t& handle,
                                                 const std::string& rel_mps_path)
{
  const auto abs_path = make_path_absolute(rel_mps_path);
  auto model          = cuopt::mps_parser::parse_mps<int, double>(abs_path, false);
  handle.sync_stream();

  auto op = mps_data_model_to_optimization_problem(&handle, model);
  auto pb = std::make_shared<detail::problem_t<int, double>>(op);
  pb->preprocess_problem();

  // build_clique_table is the production entry point: it lives over a
  // dual_simplex::user_problem_t (host CSR + host bounds), not problem_t.
  // Pull a host_user_problem snapshot off the preprocessed problem so the
  // clique table sees exactly what bound_presolve_t will see.
  dual_simplex::user_problem_t<int, double> host_problem(pb->handle_ptr);
  pb->get_host_user_problem(host_problem);

  // Force min_clique_size = 1 so even the small test instances yield groups
  // through both the "first" path and the small-clique CSR. The production
  // default (512) would short-circuit every clique on these inputs and leave
  // the test exercising no clique logic at all.
  detail::clique_config_t clique_config{};
  clique_config.min_clique_size = 1;
  auto ct                       = std::make_shared<detail::clique_table_t<int, double>>(
    /*n_vertices=*/2 * host_problem.num_cols,
    clique_config.min_clique_size,
    clique_config.max_clique_size_for_extension);

  mip_solver_settings_t<int, double> settings{};
  cuopt::timer_t timer(std::numeric_limits<double>::infinity());
  detail::build_clique_table(host_problem,
                             *ct,
                             settings.tolerances,
                             /*remove_small_cliques=*/true,
                             /*extend=*/true,
                             timer);
  return {std::move(pb), std::move(ct)};
}

// Run stock then clique-aware bound presolve on the given MPS instance and
// assert the monotonicity invariant. Real-instance tolerance is 1e-6 — see
// header comment.
void run_real_mps_monotonicity(const std::string& rel_mps_path)
{
  const raft::handle_t handle{};
  auto bundle = load_mps_with_cliques(handle, rel_mps_path);

  auto stock_pb  = run_presolve(*bundle.pb, /*ct=*/nullptr);
  auto clique_pb = run_presolve(*bundle.pb, bundle.ct);

  expect_clique_tighter_or_equal(stock_pb, clique_pb, /*tol=*/1e-6);
}

}  // namespace

TEST(clique_activity, monotonicity_real_dominating_set)
{
  // Tiny pure-binary set-covering style instance (10 cnsts, 9 binary vars).
  // Useful as a smoke test that the entire MPS → preprocess → clique build →
  // double presolve pipeline runs cleanly under the monotonicity contract.
  run_real_mps_monotonicity("mip/dominating_set.mps");
}

TEST(clique_activity, monotonicity_real_sudoku)
{
  // 729 binary vars, 354 constraints. Sudoku encodings are clique-rich by
  // construction (each row / column / box / cell yields a set-packing
  // constraint), so this exercises the dense-clique path through both
  // remove_small_cliques and the small-clique CSR.
  run_real_mps_monotonicity("mip/sudoku.mps");
}

TEST(clique_activity, monotonicity_real_cod105_max)
{
  // 1024 binary vars, ~1025 constraints. MIPLIB-style coding-theory
  // instance with many short pairwise conflicts. Larger of the four
  // instances; still well within unit-test runtime.
  run_real_mps_monotonicity("mip/cod105_max.mps");
}

TEST(clique_activity, monotonicity_real_50v_10_free_bound)
{
  // 234 constraints × 2014 vars; mostly binary via `MARKER INTORG` plus
  // UP=1 bounds, with a single non-binary continuous variable. Mixes
  // binary cliques with continuous columns, so the clique-aware path has
  // to leave the continuous column's contribution to activity alone while
  // applying corrections to the binary block.
  run_real_mps_monotonicity("mip/50v-10-free-bound.mps");
}

}  // namespace cuopt::linear_programming::test
