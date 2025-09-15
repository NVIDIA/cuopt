/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved. SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "../linear_programming/utilities/pdlp_test_utilities.cuh"
#include "mip_utils.cuh"

#include <raft/sparse/detail/cusparse_wrappers.h>
#include <thrust/sort.h>
#include <linear_programming/initial_scaling_strategy/initial_scaling.cuh>
#include <linear_programming/utilities/problem_checking.cuh>
#include <mip/presolve/bounds_presolve.cuh>
#include <mip/presolve/lb_multi_probe.cuh>
#include <mip/presolve/lb_problem.cuh>
#include <mip/presolve/load_balanced_bounds_presolve.cuh>
#include <mip/presolve/multi_probe.cuh>
#include <mip/presolve/trivial_presolve.cuh>
#include <mip/problem/load_balanced_problem.cuh>
#include <mps_parser/parser.hpp>
#include <raft/core/handle.hpp>
#include <raft/util/cudart_utils.hpp>
#include <utilities/common_utils.hpp>
#include <utilities/error.hpp>
#include <utilities/timer.hpp>

#include <rmm/mr/device/cuda_async_memory_resource.hpp>

#include <gtest/gtest.h>

#include <cstdint>
#include <sstream>
#include <string>
#include <vector>

namespace cuopt::linear_programming::test {

template <typename i_t, typename f_t>
std::vector<i_t> rand_id(const raft::handle_t* handle_ptr,
                         rmm::device_uvector<i_t>& reorg_ids,
                         const std::vector<i_t>& bin_offsets,
                         int64_t seed)
{
  std::mt19937 rng(seed);
  std::vector<i_t> r_id;

  for (size_t i = 0; i < bin_offsets.size() - 1; ++i) {
    if (bin_offsets[i] != bin_offsets[i + 1]) {
      std::uniform_int_distribution<i_t> dist(bin_offsets[i], bin_offsets[i + 1] - 1);
      auto id = reorg_ids.element(dist(rng), handle_ptr->get_stream());
      r_id.push_back(id);
    }
  }
  return r_id;
}

template <typename i_t, typename f_t>
std::vector<i_t> get_rand_items(detail::problem_t<i_t, f_t>& lb_problem,
                                bool disp_cnst,
                                int64_t seed)
{
  if (disp_cnst) {
    return rand_id(lb_problem.cnst_csr.reorg_ids, lb_problem.cnst_csr.bin_offsets, seed);
  } else {
    return rand_id(lb_problem.vars_csr.reorg_ids, lb_problem.vars_csr.bin_offsets, seed);
  }
}

inline auto make_async() { return std::make_shared<rmm::mr::cuda_async_memory_resource>(); }

void init_handler(const raft::handle_t* handle_ptr)
{
  // Init cuBlas / cuSparse context here to avoid having it during solving time
  RAFT_CUBLAS_TRY(raft::linalg::detail::cublassetpointermode(
    handle_ptr->get_cublas_handle(), CUBLAS_POINTER_MODE_DEVICE, handle_ptr->get_stream()));
  RAFT_CUSPARSE_TRY(raft::sparse::detail::cusparsesetpointermode(
    handle_ptr->get_cusparse_handle(), CUSPARSE_POINTER_MODE_DEVICE, handle_ptr->get_stream()));
}

std::tuple<std::vector<int>, std::vector<double>, std::vector<double>> select_k_random(
  detail::problem_t<int, double>& problem, int sample_size)
{
  // auto seed = std::random_device{}();
  unsigned long seed = 358004014ul;
  std::cout << "Tested with seed " << seed << "\n";
  problem.compute_n_integer_vars();
  auto v_bnds     = host_copy(problem.variable_bounds);
  auto int_var_id = host_copy(problem.integer_indices);
  int_var_id.erase(
    std::remove_if(int_var_id.begin(),
                   int_var_id.end(),
                   [v_bnds](auto id) {
                     auto v_bnd = v_bnds[id];
                     return !(std::isfinite(get_lower(v_bnd)) && std::isfinite(get_upper(v_bnd)));
                   }),
    int_var_id.end());
  sample_size = std::min(sample_size, static_cast<int>(int_var_id.size()));
  std::vector<int> random_int_vars;
  std::mt19937 m{seed};
  std::sample(
    int_var_id.begin(), int_var_id.end(), std::back_inserter(random_int_vars), sample_size, m);
  std::vector<double> probe_0(sample_size);
  std::vector<double> probe_1(sample_size);
  for (int i = 0; i < static_cast<int>(random_int_vars.size()); ++i) {
    if (i % 2) {
      probe_0[i] = get_lower(v_bnds[random_int_vars[i]]);
      probe_1[i] = get_upper(v_bnds[random_int_vars[i]]);
    } else {
      probe_1[i] = get_lower(v_bnds[random_int_vars[i]]);
      probe_0[i] = get_upper(v_bnds[random_int_vars[i]]);
    }
  }
  return std::make_tuple(std::move(random_int_vars), std::move(probe_0), std::move(probe_1));
}

std::pair<std::vector<thrust::pair<int, double>>, std::vector<thrust::pair<int, double>>>
convert_probe_tuple(std::tuple<std::vector<int>, std::vector<double>, std::vector<double>>& probe)
{
  std::vector<thrust::pair<int, double>> probe_first;
  std::vector<thrust::pair<int, double>> probe_second;
  for (size_t i = 0; i < std::get<0>(probe).size(); ++i) {
    probe_first.emplace_back(thrust::make_pair(std::get<0>(probe)[i], std::get<1>(probe)[i]));
    probe_second.emplace_back(thrust::make_pair(std::get<0>(probe)[i], std::get<2>(probe)[i]));
  }
  return std::make_pair(std::move(probe_first), std::move(probe_second));
}

std::tuple<std::vector<double>, std::vector<double>, std::vector<double>, std::vector<double>>
bounds_probe_results(detail::bound_presolve_t<int, double>& bnd_prb_0,
                     detail::bound_presolve_t<int, double>& bnd_prb_1,
                     detail::problem_t<int, double>& problem,
                     const std::pair<std::vector<thrust::pair<int, double>>,
                                     std::vector<thrust::pair<int, double>>>& probe)
{
  auto& probe_first  = std::get<0>(probe);
  auto& probe_second = std::get<1>(probe);
  rmm::device_uvector<double> b_lb_0(problem.n_variables, problem.handle_ptr->get_stream());
  rmm::device_uvector<double> b_ub_0(problem.n_variables, problem.handle_ptr->get_stream());
  rmm::device_uvector<double> b_lb_1(problem.n_variables, problem.handle_ptr->get_stream());
  rmm::device_uvector<double> b_ub_1(problem.n_variables, problem.handle_ptr->get_stream());
  bnd_prb_0.solve(problem, probe_first);
  bnd_prb_0.set_updated_bounds(problem.handle_ptr, make_span(b_lb_0), make_span(b_ub_0));
  bnd_prb_1.solve(problem, probe_second);
  bnd_prb_1.set_updated_bounds(problem.handle_ptr, make_span(b_lb_1), make_span(b_ub_1));

  auto h_lb_0 = host_copy(b_lb_0);
  auto h_ub_0 = host_copy(b_ub_0);
  auto h_lb_1 = host_copy(b_lb_1);
  auto h_ub_1 = host_copy(b_ub_1);
  return std::make_tuple(
    std::move(h_lb_0), std::move(h_ub_0), std::move(h_lb_1), std::move(h_ub_1));
}

std::tuple<std::vector<double>, std::vector<double>, std::vector<double>, std::vector<double>>
multi_probe_results(
  detail::multi_probe_t<int, double>& prb,
  detail::problem_t<int, double>& problem,
  const std::tuple<std::vector<int>, std::vector<double>, std::vector<double>>& probe_tuple)
{
  {
    nvtxRangePush("solve");
    prb.solve(problem, probe_tuple);
    nvtxRangePop();
  }
  rmm::device_uvector<double> b_lb_0(problem.n_variables, problem.handle_ptr->get_stream());
  rmm::device_uvector<double> b_ub_0(problem.n_variables, problem.handle_ptr->get_stream());
  rmm::device_uvector<double> b_lb_1(problem.n_variables, problem.handle_ptr->get_stream());
  rmm::device_uvector<double> b_ub_1(problem.n_variables, problem.handle_ptr->get_stream());
  prb.set_updated_bounds(problem.handle_ptr, make_span(b_lb_0), make_span(b_ub_0), 0);
  prb.set_updated_bounds(problem.handle_ptr, make_span(b_lb_1), make_span(b_ub_1), 1);

  auto h_lb_0 = host_copy(b_lb_0);
  auto h_ub_0 = host_copy(b_ub_0);
  auto h_lb_1 = host_copy(b_lb_1);
  auto h_ub_1 = host_copy(b_ub_1);
  return std::make_tuple(
    std::move(h_lb_0), std::move(h_ub_0), std::move(h_lb_1), std::move(h_ub_1));
}

std::tuple<std::vector<double2>, std::vector<double2>> multi_probe_results(
  detail::lb_multi_probe_t<int, double>& prb,
  detail::problem_t<int, double>& problem,
  const std::tuple<std::vector<int>, std::vector<double>, std::vector<double>>& probe_tuple)
{
  {
    nvtxRangePush("lb_solve");
    prb.solve(problem, probe_tuple);
    nvtxRangePop();
  }
  rmm::device_uvector<double2> m_bnd_0(problem.n_variables, problem.handle_ptr->get_stream());
  rmm::device_uvector<double2> m_bnd_1(problem.n_variables, problem.handle_ptr->get_stream());
  prb.set_updated_bounds(problem.handle_ptr, make_span(m_bnd_0), 0);
  prb.set_updated_bounds(problem.handle_ptr, make_span(m_bnd_1), 1);

  auto h_bnd_0 = host_copy(m_bnd_0);
  auto h_bnd_1 = host_copy(m_bnd_1);
  return std::make_tuple(std::move(h_bnd_0), std::move(h_bnd_1));
}

bool test_bounds(detail::problem_t<int, double>& problem,
                 detail::lb_bounds_update_data_t<int, double>& lb_upd,
                 detail::bounds_update_data_t<int, double>& upd)
{
  auto stream    = problem.handle_ptr->get_stream();
  auto chg       = host_copy(upd.changed_variables, stream);
  auto lb_chg    = host_copy(lb_upd.changed_variables, stream);
  auto bnd_lb    = host_copy(upd.lb, stream);
  auto bnd_ub    = host_copy(upd.ub, stream);
  auto mpb_v_bnd = host_copy(lb_upd.vars_bnd, stream);

  auto orig_bnds = host_copy(problem.variable_bounds, stream);

  auto off = host_copy(problem.reverse_offsets, stream);

  int lb_chg_count = 0, chg_count = 0;
  bool passed = true;
  double tol  = 1e-8;
  for (int i = 0; i < problem.n_variables; ++i) {
    // if (lb_chg[i]) {
    lb_chg_count += lb_chg[i];
    chg_count += chg[i];
    if (chg[i] != lb_chg[i]) {
      std::cout << "variable changed mismatch " << i << "\n";
      passed = false;
      continue;
    }
    if (true) {
      auto v_lb = bnd_lb[i];
      auto v_ub = bnd_ub[i];

      auto mpb_lb = get_lower(mpb_v_bnd[i]);
      auto mpb_ub = get_upper(mpb_v_bnd[i]);

      bool lb_mismatch =
        ((abs(mpb_lb - v_lb) / abs(v_lb) > tol) || (std::isinf(v_lb) ^ std::isinf(mpb_lb)));
      bool ub_mismatch =
        ((abs(mpb_ub - v_ub) / abs(v_ub) > tol) || (std::isinf(v_ub) ^ std::isinf(mpb_ub)));

      if (lb_mismatch) {
        auto deg = off[i + 1] - off[i];
        std::cout << "lb mismatch " << i << " " << v_lb << " " << mpb_lb
                  << "\tdiff = " << abs(v_lb - mpb_lb) << " " << get_lower(orig_bnds[i]) << " "
                  << deg << "\n";
        passed = false;
      }
      if (ub_mismatch) {
        auto deg = off[i + 1] - off[i];
        std::cout << "ub mismatch " << i << " " << v_ub << " " << mpb_ub
                  << "\tdiff = " << abs(v_ub - mpb_ub) << " " << get_upper(orig_bnds[i]) << " "
                  << deg << "\n";
        passed = false;
      }
      if (passed) { std::cout << "var " << i << " matched\n"; }
    }
  }
  return passed;
}

bool test_activity(detail::problem_t<int, double>& problem,
                   detail::lb_bounds_update_data_t<int, double>& lb_upd,
                   detail::bounds_update_data_t<int, double>& upd,
                   bool ignore_changed_constraints = false)
{
  problem.handle_ptr->sync_stream();
  auto stream   = problem.handle_ptr->get_stream();
  auto chg      = host_copy(upd.changed_constraints, stream);
  auto nxchg    = host_copy(upd.next_changed_constraints, stream);
  auto lb_nxchg = host_copy(lb_upd.next_changed_constraints, stream);
  auto min_act  = host_copy(upd.min_activity, stream);
  auto max_act  = host_copy(upd.max_activity, stream);
  auto c_lb     = host_copy(problem.constraint_lower_bounds, stream);
  auto c_ub     = host_copy(problem.constraint_upper_bounds, stream);
  auto off      = host_copy(problem.offsets, stream);

  auto c_sl        = host_copy(lb_upd.cnst_slack, stream);
  bool passed      = true;
  double tol       = 1e-8;
  int count_chg    = 0;
  int count_lb_chg = 0;
  for (int i = 0; i < problem.n_constraints; ++i) {
    auto deg = off[i + 1] - off[i];
    if ((!ignore_changed_constraints) && (nxchg[i] != lb_nxchg[i])) {
      std::cout << "constraint changed mismatch " << i << " bnd " << nxchg[i] << " lb "
                << lb_nxchg[i] << " " << deg << "\n";
      passed = false;
      count_lb_chg += lb_nxchg[i];
      count_chg += nxchg[i];
      continue;
    }
    if (chg[i]) {
      auto bnd_min_slack = min_act[i];
      auto bnd_max_slack = max_act[i];
      auto lb_min_slack  = c_ub[i] - get_lower(c_sl[i]);
      auto lb_max_slack  = c_lb[i] - get_upper(c_sl[i]);
      if ((abs(lb_min_slack - bnd_min_slack) / abs(bnd_min_slack) > tol) ||
          (std::isinf(lb_min_slack) ^ std::isinf(bnd_min_slack))) {
        std::cout << "min mismatch " << i << " " << bnd_min_slack << " " << lb_min_slack
                  << "\tdiff = " << abs(bnd_min_slack - lb_min_slack) << " " << deg << " " << chg[i]
                  << "\n";
        passed = false;
      }
      if ((abs(lb_max_slack - bnd_max_slack) / abs(bnd_max_slack) > tol) ||
          (std::isinf(lb_max_slack) ^ std::isinf(bnd_max_slack))) {
        std::cout << "max mismatch " << i << " " << bnd_max_slack << " " << lb_max_slack
                  << "\tdiff = " << abs(bnd_max_slack - lb_max_slack) << " " << deg << " " << chg[i]
                  << "\n";
        passed = false;
      }
    }
  }
  std::cout << "count_lb_chg " << count_lb_chg << " count_chg " << count_chg << "\n";
  return passed;
}

void randomize_changed_constraints(rmm::device_uvector<int>& changed_constraints,
                                   int64_t seed,
                                   const raft::handle_t* handle_ptr)
{
  std::cout << "Tested with seed " << seed << "\n";
  std::mt19937 rng(seed);
  std::vector<int> h_cc;
  h_cc.reserve(changed_constraints.size());
  std::uniform_int_distribution<int> dist(0, 1);
  for (size_t i = 0; i < changed_constraints.size(); ++i) {
    h_cc.push_back(dist(rng));
  }
  device_copy(changed_constraints, h_cc, handle_ptr->get_stream());
}

void flip_changed_constraints(rmm::device_uvector<int>& changed_constraints,
                              const raft::handle_t* handle_ptr)
{
  std::vector<int> h_cc(changed_constraints.size(), 1);
  h_cc[0] = 0;
  device_copy(changed_constraints, h_cc, handle_ptr->get_stream());
}

void test_lb_multi_probe(std::string path)
{
  auto memory_resource = make_async();
  rmm::mr::set_current_device_resource(memory_resource.get());
  const raft::handle_t handle_{};
  cuopt::mps_parser::mps_data_model_t<int, double> mps_problem =
    cuopt::mps_parser::parse_mps<int, double>(path, false);
  handle_.sync_stream();
  auto op_problem = mps_data_model_to_optimization_problem(&handle_, mps_problem);
  problem_checking_t<int, double>::check_problem_representation(op_problem);
  detail::problem_t<int, double> problem(op_problem);

  problem.preprocess_problem();
  detail::trivial_presolve(problem);

  mip_solver_settings_t<int, double> default_settings{};
  detail::pdhg_solver_t<int, double> pdhg_solver(problem.handle_ptr, problem);
  detail::pdlp_initial_scaling_strategy_t<int, double> scaling(&handle_,
                                                               problem,
                                                               10,
                                                               1.0,
                                                               pdhg_solver,
                                                               problem.reverse_coefficients,
                                                               problem.reverse_offsets,
                                                               problem.reverse_constraints,
                                                               true);
  detail::mip_solver_t<int, double> solver(problem, default_settings, scaling, cuopt::timer_t(0));
  detail::multi_probe_t<int, double> multi_probe_prs(solver.context);

  detail::lb_problem_t<int, double>& lb_problem = problem.get_load_balanced_problem();
  detail::lb_multi_probe_t<int, double> lb_multi_probe_prs(solver.context);

  auto probe_tuple       = select_k_random(problem, 100);
  auto bounds_probe_vals = convert_probe_tuple(probe_tuple);

  auto [bnd_lb_0, bnd_ub_0, bnd_lb_1, bnd_ub_1] =
    multi_probe_results(multi_probe_prs, problem, probe_tuple);
  auto [m_bnd_0, m_bnd_1] = multi_probe_results(lb_multi_probe_prs, problem, probe_tuple);

  handle_.sync_stream();
  test_activity(problem, lb_multi_probe_prs.upd_0, multi_probe_prs.upd_0);
  test_activity(problem, lb_multi_probe_prs.upd_1, multi_probe_prs.upd_1);
  test_bounds(problem, lb_multi_probe_prs.upd_0, multi_probe_prs.upd_0);
  test_bounds(problem, lb_multi_probe_prs.upd_1, multi_probe_prs.upd_1);
}

TEST(presolve, multi_probe)
{
  std::vector<std::string> test_instances = {
    "mip/50v-10-free-bound.mps", "mip/neos5-free-bound.mps", "mip/neos5.mps"};
  for (const auto& test_instance : test_instances) {
    std::cout << "Running: " << test_instance << std::endl;
    auto path = make_path_absolute(test_instance);
    test_lb_multi_probe(path);
  }
}

}  // namespace cuopt::linear_programming::test
