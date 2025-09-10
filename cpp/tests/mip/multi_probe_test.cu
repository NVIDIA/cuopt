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
#include <linear_programming/initial_scaling_strategy/initial_scaling.cuh>
#include <linear_programming/utilities/problem_checking.cuh>
#include <mip/presolve/bounds_presolve.cuh>
#include <mip/presolve/multi_probe.cuh>
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
  auto seed = std::random_device{}();
  std::cerr << "Tested with seed " << seed << "\n";
  problem.compute_n_integer_vars();
  auto [v_lb, v_ub] = extract_host_bounds<double>(problem.variable_bounds, problem.handle_ptr);
  auto int_var_id   = host_copy(problem.integer_indices);
  int_var_id.erase(
    std::remove_if(int_var_id.begin(),
                   int_var_id.end(),
                   [v_lb_sp = v_lb, v_ub_sp = v_ub](auto id) {
                     return !(std::isfinite(v_lb_sp[id]) && std::isfinite(v_ub_sp[id]));
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
      probe_0[i] = v_lb[random_int_vars[i]];
      probe_1[i] = v_ub[random_int_vars[i]];
    } else {
      probe_1[i] = v_lb[random_int_vars[i]];
      probe_0[i] = v_ub[random_int_vars[i]];
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
  prb.solve(problem, probe_tuple);
  rmm::device_uvector<double> m_lb_0(problem.n_variables, problem.handle_ptr->get_stream());
  rmm::device_uvector<double> m_ub_0(problem.n_variables, problem.handle_ptr->get_stream());
  rmm::device_uvector<double> m_lb_1(problem.n_variables, problem.handle_ptr->get_stream());
  rmm::device_uvector<double> m_ub_1(problem.n_variables, problem.handle_ptr->get_stream());
  prb.set_updated_bounds(problem.handle_ptr, make_span(m_lb_0), make_span(m_ub_0), 0);
  prb.set_updated_bounds(problem.handle_ptr, make_span(m_lb_1), make_span(m_ub_1), 1);

  auto h_lb_0 = host_copy(m_lb_0);
  auto h_ub_0 = host_copy(m_ub_0);
  auto h_lb_1 = host_copy(m_lb_1);
  auto h_ub_1 = host_copy(m_ub_1);
  return std::make_tuple(
    std::move(h_lb_0), std::move(h_ub_0), std::move(h_lb_1), std::move(h_ub_1));
}

void test_multi_probe(std::string path)
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
  mip_solver_settings_t<int, double> default_settings{};
  detail::pdlp_initial_scaling_strategy_t<int, double> scaling(&handle_,
                                                               problem,
                                                               10,
                                                               1.0,
                                                               problem.reverse_coefficients,
                                                               problem.reverse_offsets,
                                                               problem.reverse_constraints,
                                                               nullptr,
                                                               true);
  detail::mip_solver_t<int, double> solver(problem, default_settings, scaling, cuopt::timer_t(0));
  detail::bound_presolve_t<int, double> bnd_prb_0(solver.context);
  detail::bound_presolve_t<int, double> bnd_prb_1(solver.context);
  detail::multi_probe_t<int, double> multi_probe_prs(solver.context);

  auto probe_tuple       = select_k_random(problem, 100);
  auto bounds_probe_vals = convert_probe_tuple(probe_tuple);

  auto [bnd_lb_0, bnd_ub_0, bnd_lb_1, bnd_ub_1] =
    bounds_probe_results(bnd_prb_0, bnd_prb_1, problem, bounds_probe_vals);
  auto [m_lb_0, m_ub_0, m_lb_1, m_ub_1] =
    multi_probe_results(multi_probe_prs, problem, probe_tuple);

  auto bnd_min_act_0 = host_copy(bnd_prb_0.upd.min_activity);
  auto bnd_max_act_0 = host_copy(bnd_prb_0.upd.max_activity);
  auto bnd_min_act_1 = host_copy(bnd_prb_1.upd.min_activity);
  auto bnd_max_act_1 = host_copy(bnd_prb_1.upd.max_activity);

  auto mlp_min_act_0 = host_copy(multi_probe_prs.upd_0.min_activity);
  auto mlp_max_act_0 = host_copy(multi_probe_prs.upd_0.max_activity);
  auto mlp_min_act_1 = host_copy(multi_probe_prs.upd_1.min_activity);
  auto mlp_max_act_1 = host_copy(multi_probe_prs.upd_1.max_activity);

  for (int i = 0; i < (int)bnd_min_act_0.size(); ++i) {
    EXPECT_DOUBLE_EQ(bnd_min_act_0[i], mlp_min_act_0[i]);
    EXPECT_DOUBLE_EQ(bnd_max_act_0[i], mlp_max_act_0[i]);
    EXPECT_DOUBLE_EQ(bnd_min_act_1[i], mlp_min_act_1[i]);
    EXPECT_DOUBLE_EQ(bnd_max_act_1[i], mlp_max_act_1[i]);
  }

  for (int i = 0; i < (int)bnd_lb_0.size(); ++i) {
    EXPECT_DOUBLE_EQ(bnd_lb_0[i], m_lb_0[i]);
    EXPECT_DOUBLE_EQ(bnd_ub_0[i], m_ub_0[i]);
    EXPECT_DOUBLE_EQ(bnd_lb_1[i], m_lb_1[i]);
    EXPECT_DOUBLE_EQ(bnd_ub_1[i], m_ub_1[i]);
  }
}

TEST(presolve, multi_probe)
{
  std::vector<std::string> test_instances = {
    "mip/50v-10-free-bound.mps", "mip/neos5-free-bound.mps", "mip/neos5.mps"};
  for (const auto& test_instance : test_instances) {
    std::cout << "Running: " << test_instance << std::endl;
    auto path = make_path_absolute(test_instance);
    test_multi_probe(path);
  }
}

}  // namespace cuopt::linear_programming::test
