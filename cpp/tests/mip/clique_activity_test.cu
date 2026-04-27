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

// Helpers

// Build a minimal problem_t. preprocess_problem() populates reverse CSR and
// is_binary_variable, which clique_group_table_t::build_from_host reads.
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

// Build a clique_table with the given large cliques and small adjacency.
// min_clique_size=1 disables internal filtering for tests.
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
  // Test-only helper: production uses remove_small_cliques to populate the CSR.
  ct->set_small_clique_adj_for_test(small_adj);
  // Materialize var_clique_first / var_clique_addtl / first_var_positions
  // from `first` and `addtl_cliques`. This mirrors what
  // build_clique_table()/find_initial_cliques() do at the end of build.
  detail::fill_var_clique_maps(*ct);
  return ct;
}

// Tests for clique_group_table_t::build_from_host

TEST(clique_activity, build_from_host_no_cliques)
{
  // x0 + x1 + x2 + y >= 2, no clique table → no groups.
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
  // Clique {0,1,2} on x0 + x1 + x2 + y >= 2 → one group on constraint 0.
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

  // reverse_group_id: vars 0,1,2 in group 0; y (var 3) is -1.
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
  // Clique {x0, x1, ~x2}: effective coeffs {+1, +1, -1}.
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
  // No large cliques; triangle of small edges should be extracted as one group.
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

// Tests for compute_clique_corrections_kernel

TEST(clique_activity, compute_corrections_kernel_values)
{
  // Coeffs {3,5,2}, lb=0, ub=1 → max_correction = 10-5 = 5, min_correction = 0.
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
  // x1 (coeff=5) fixed; only x0, x2 active → max_correction = 5-3 = 2.
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

// End-to-end bound propagation: clique tightens what stock can't.

TEST(clique_activity, bound_propagation_tightens_with_clique)
{
  // x0+x1+x2+y >= 2 with clique {x0,x1,x2}:
  // stock leaves y in [0,1]; clique-aware corrects max_a to 2 → y.lb = 1.
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

  detail::bound_presolve_t<int, double> bp_no_clique(context);
  bp_no_clique.solve(*pb);
  auto stream    = handle.get_stream();
  auto h_lb_base = host_copy(bp_no_clique.upd.lb, stream);
  auto h_ub_base = host_copy(bp_no_clique.upd.ub, stream);

  pb->clique_table = make_clique_table(/*n_vars=*/4, /*first=*/{{0, 1, 2}});

  detail::bound_presolve_t<int, double> bp_with_clique(context);
  bp_with_clique.solve(*pb);
  auto h_lb_cliq = host_copy(bp_with_clique.upd.lb, stream);
  auto h_ub_cliq = host_copy(bp_with_clique.upd.ub, stream);

  EXPECT_DOUBLE_EQ(h_lb_base[3], 0.0);
  EXPECT_DOUBLE_EQ(h_lb_cliq[3], 1.0);

  // Clique members stay in [0, 1]: any two=1 remains feasible per-variable.
  for (int i = 0; i < 3; ++i) {
    EXPECT_DOUBLE_EQ(h_lb_cliq[i], 0.0);
    EXPECT_DOUBLE_EQ(h_ub_cliq[i], 1.0);
  }
}

TEST(clique_activity, bound_propagation_tightens_with_complement_literal_clique)
{
  // x0+x1-x2+y >= 1 with clique {x0,x1,~x2}: clique-aware lifts y.lb 0→1.
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
  // x0+x1 <= 1: clique correction has no var to propagate onto → identical bounds.
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

// Clique-aware vs stock monotonicity invariant
// (see CLIQUE_PIPELINE_AUDIT.md for the proof):
//   lb_stock <= lb_clique;  ub_stock >= ub_clique
//   min_activity_stock <= min_activity_clique
//   max_activity_stock >= max_activity_clique

namespace {

struct presolve_result_t {
  std::vector<double> lb;
  std::vector<double> ub;
  std::vector<double> min_a;
  std::vector<double> max_a;
};

// Run bound_presolve_t::solve and snapshot converged buffers. `ct` may be null
// for the stock path; otherwise it's attached to pb before solving.
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

// Assert: clique tightens or matches on every var and constraint.
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
    // Skip free constraints (inf stored activity).
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

// Assert: at least one var or constraint tightens strictly. Guards against
// degenerate tests where the clique fails to consume.
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
  EXPECT_TRUE(found) << "Expected at least one strict tightening; clique not being consumed.";
}

}  // namespace

TEST(clique_activity, monotonicity_cascading_tightening)
{
  // c0: x0+x1+x2+y0 >= 2 (clique {x0,x1,x2})
  // c1: y0+z <= 5
  // Clique lifts y0.lb 0→1, which cascades to tighten z.ub 5→4.
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

  EXPECT_DOUBLE_EQ(stock_pb.lb[3], 0.0);
  EXPECT_DOUBLE_EQ(stock_pb.ub[4], 5.0);
  EXPECT_DOUBLE_EQ(clique_pb.lb[3], 1.0);
  EXPECT_DOUBLE_EQ(clique_pb.ub[4], 4.0);

  expect_some_strict_tightening(stock_pb, clique_pb);
}

TEST(clique_activity, monotonicity_independent_cliques_multi_constraint)
{
  // Two independent cliques + a coupling constraint.
  // Clique lifts y0,y1 to 1; min_a(c2) becomes 2 → z.ub 3→1.
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

  EXPECT_DOUBLE_EQ(stock_pb.lb[6], 0.0);
  EXPECT_DOUBLE_EQ(stock_pb.lb[7], 0.0);
  EXPECT_DOUBLE_EQ(clique_pb.lb[6], 1.0);
  EXPECT_DOUBLE_EQ(clique_pb.lb[7], 1.0);
  EXPECT_GE(stock_pb.ub[8] + 1e-9, clique_pb.ub[8]);
  EXPECT_LE(clique_pb.ub[8], 1.0 + 1e-9);

  expect_some_strict_tightening(stock_pb, clique_pb);
}

TEST(clique_activity, monotonicity_complement_literal_clique)
{
  // c0: x0+x1-x2+y >= 1 with clique {x0,x1,~x2}: clique lifts y to 1.
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

  // No-op problem: drift would mean the kernel is over-applying corrections.
  ASSERT_EQ(stock_pb.lb.size(), clique_pb.lb.size());
  for (std::size_t i = 0; i < stock_pb.lb.size(); ++i) {
    EXPECT_DOUBLE_EQ(stock_pb.lb[i], clique_pb.lb[i]) << "lb drift at var " << i;
    EXPECT_DOUBLE_EQ(stock_pb.ub[i], clique_pb.ub[i]) << "ub drift at var " << i;
  }
}

}  // namespace

// Real-instance monotonicity tests on small binary-heavy MPS files in
// datasets/mip/. Tolerance 1e-6 absorbs FP drift over many iterations.

namespace {

struct mps_problem_with_cliques_t {
  std::shared_ptr<detail::problem_t<int, double>> pb;
  std::shared_ptr<detail::clique_table_t<int, double>> ct;
};

// Parse MPS, build preprocessed problem, build production clique table.
mps_problem_with_cliques_t load_mps_with_cliques(const raft::handle_t& handle,
                                                 const std::string& rel_mps_path)
{
  const auto abs_path = make_path_absolute(rel_mps_path);
  auto model          = cuopt::mps_parser::parse_mps<int, double>(abs_path, false);
  handle.sync_stream();

  auto op = mps_data_model_to_optimization_problem(&handle, model);
  auto pb = std::make_shared<detail::problem_t<int, double>>(op);
  pb->preprocess_problem();

  // build_clique_table operates on dual_simplex::user_problem_t.
  dual_simplex::user_problem_t<int, double> host_problem(pb->handle_ptr);
  pb->get_host_user_problem(host_problem);

  // min_clique_size=1 (vs. production default 512) so small test instances
  // actually emit cliques.
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

// Run stock then clique-aware presolve and assert monotonicity (tol=1e-6).
void run_real_mps_monotonicity(const std::string& rel_mps_path)
{
  const raft::handle_t handle{};
  auto bundle = load_mps_with_cliques(handle, rel_mps_path);

  auto stock_pb  = run_presolve(*bundle.pb, /*ct=*/nullptr);
  auto clique_pb = run_presolve(*bundle.pb, bundle.ct);

  expect_clique_tighter_or_equal(stock_pb, clique_pb, /*tol=*/1e-6);
}

}  // namespace

// Tiny pure-binary set-covering smoke test.
TEST(clique_activity, monotonicity_real_dominating_set)
{
  run_real_mps_monotonicity("mip/dominating_set.mps");
}

// Sudoku: clique-rich set-packing structure exercises dense-clique path.
TEST(clique_activity, monotonicity_real_sudoku) { run_real_mps_monotonicity("mip/sudoku.mps"); }

// Coding-theory instance with many short pairwise conflicts.
TEST(clique_activity, monotonicity_real_cod105_max)
{
  run_real_mps_monotonicity("mip/cod105_max.mps");
}

// Mixed binary cliques + a continuous column.
TEST(clique_activity, monotonicity_real_50v_10_free_bound)
{
  run_real_mps_monotonicity("mip/50v-10-free-bound.mps");
}

}  // namespace cuopt::linear_programming::test
