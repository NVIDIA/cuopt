/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#include "../linear_programming/utilities/pdlp_test_utilities.cuh"
#include "determinism_utils.cuh"
#include "mip_utils.cuh"

#include <raft/sparse/detail/cusparse_wrappers.h>
#include <cuopt/linear_programming/pdlp/pdlp_hyper_params.cuh>
#include <cuopt/linear_programming/solve.hpp>
#include <mip_heuristics/presolve/bounds_presolve.cuh>
#include <mip_heuristics/presolve/multi_probe.cuh>
#include <mip_heuristics/presolve/third_party_presolve.hpp>
#include <mip_heuristics/presolve/trivial_presolve.cuh>
#include <mip_heuristics/problem/problem.cuh>
#include <mip_heuristics/utils.cuh>
#include <mps_parser/mps_data_model.hpp>
#include <mps_parser/parser.hpp>
#include <pdlp/initial_scaling_strategy/initial_scaling.cuh>
#include <pdlp/utilities/problem_checking.cuh>
#include <pdlp/utils.cuh>
#include <utilities/common_utils.hpp>
#include <utilities/copy_helpers.hpp>
#include <utilities/error.hpp>

#include <raft/core/handle.hpp>
#include <raft/util/cudart_utils.hpp>

#include <gtest/gtest.h>

#include <cstdint>
#include <sstream>
#include <string>
#include <vector>

namespace cuopt::linear_programming::test {

void init_handler(const raft::handle_t* handle_ptr)
{
  // Init cuBlas / cuSparse context here to avoid having it during solving time
  RAFT_CUBLAS_TRY(raft::linalg::detail::cublassetpointermode(
    handle_ptr->get_cublas_handle(), CUBLAS_POINTER_MODE_DEVICE, handle_ptr->get_stream()));
  RAFT_CUSPARSE_TRY(raft::sparse::detail::cusparsesetpointermode(
    handle_ptr->get_cusparse_handle(), CUSPARSE_POINTER_MODE_DEVICE, handle_ptr->get_stream()));
}

std::tuple<std::vector<int>, std::vector<double>, std::vector<double>> select_k_random(
  detail::problem_t<int, double>& problem,
  int sample_size,
  unsigned long seed = std::random_device{}())
{
  std::cerr << "Tested with seed " << seed << "\n";
  problem.compute_n_integer_vars();
  auto [v_lb, v_ub] = extract_host_bounds<double>(problem.variable_bounds, problem.handle_ptr);
  auto int_var_id   = host_copy(problem.integer_indices, problem.handle_ptr->get_stream());
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

uint32_t test_probing_cache_determinism(std::string path,
                                        unsigned long seed = std::random_device{}())
{
  const raft::handle_t handle_{};
  cuopt::mps_parser::mps_data_model_t<int, double> mps_problem =
    cuopt::mps_parser::parse_mps<int, double>(path, false);
  handle_.sync_stream();
  auto op_problem = mps_data_model_to_optimization_problem(&handle_, mps_problem);
  problem_checking_t<int, double>::check_problem_representation(op_problem);
  detail::problem_t<int, double> problem(op_problem);
  mip_solver_settings_t<int, double> default_settings{};
  default_settings.mip_scaling = false;  // we're not checking scaling determinism here
  auto timer = cuopt::termination_checker_t(0.0, cuopt::termination_checker_t::root_tag_t{});
  detail::mip_solver_t<int, double> solver(problem, default_settings, timer);
  detail::bound_presolve_t<int, double> bnd_prb(solver.context);

  work_limit_context_t work_limit_context("ProbingCache");
  // rely on the iteration limit
  termination_checker_t probing_timer(
    work_limit_context, std::numeric_limits<double>::max(), timer);
  compute_probing_cache(bnd_prb, problem, probing_timer);
  std::vector<std::pair<int, std::array<detail::cache_entry_t<int, double>, 2>>> cached_values(
    bnd_prb.probing_cache.probing_cache.begin(), bnd_prb.probing_cache.probing_cache.end());
  std::sort(cached_values.begin(), cached_values.end(), [](const auto& a, const auto& b) {
    return a.first < b.first;
  });

  std::vector<int> probed_indices;
  std::vector<double> intervals;
  std::vector<int> interval_types;

  std::vector<int> var_to_cached_bound_keys;
  std::vector<double> var_to_cached_bound_lb;
  std::vector<double> var_to_cached_bound_ub;
  for (const auto& a : cached_values) {
    probed_indices.push_back(a.first);
    intervals.push_back(a.second[0].val_interval.val);
    intervals.push_back(a.second[1].val_interval.val);
    interval_types.push_back(a.second[0].val_interval.interval_type);
    interval_types.push_back(a.second[1].val_interval.interval_type);

    auto sorted_map = std::map<int, detail::cached_bound_t<double>>(
      a.second[0].var_to_cached_bound_map.begin(), a.second[0].var_to_cached_bound_map.end());
    for (const auto& [var_id, cached_bound] : sorted_map) {
      var_to_cached_bound_keys.push_back(var_id);
      var_to_cached_bound_lb.push_back(cached_bound.lb);
      var_to_cached_bound_ub.push_back(cached_bound.ub);
    }
  }

  std::vector<uint32_t> hashes;
  hashes.push_back(detail::compute_hash(probed_indices));
  hashes.push_back(detail::compute_hash(intervals));
  hashes.push_back(detail::compute_hash(interval_types));
  hashes.push_back(detail::compute_hash(var_to_cached_bound_keys));
  hashes.push_back(detail::compute_hash(var_to_cached_bound_lb));
  hashes.push_back(detail::compute_hash(var_to_cached_bound_ub));

  // return a composite hash of all the hashes to check for determinism
  return detail::compute_hash(hashes);
}

uint32_t test_scaling_determinism(std::string path, unsigned long seed = std::random_device{}())
{
  const raft::handle_t handle_{};
  cuopt::mps_parser::mps_data_model_t<int, double> mps_problem =
    cuopt::mps_parser::parse_mps<int, double>(path, false);
  handle_.sync_stream();
  auto op_problem = mps_data_model_to_optimization_problem(&handle_, mps_problem);
  problem_checking_t<int, double>::check_problem_representation(op_problem);
  detail::problem_t<int, double> problem(op_problem);

  pdlp_hyper_params::pdlp_hyper_params_t hyper_params{};
  hyper_params.update_primal_weight_on_initial_solution = false;
  hyper_params.update_step_size_on_initial_solution     = true;
  // problem contains unpreprocessed data
  detail::problem_t<int, double> scaled_problem(problem);

  detail::pdlp_initial_scaling_strategy_t<int, double> scaling(
    scaled_problem.handle_ptr,
    scaled_problem,
    hyper_params.default_l_inf_ruiz_iterations,
    (double)hyper_params.default_alpha_pock_chambolle_rescaling,
    scaled_problem.reverse_coefficients,
    scaled_problem.reverse_offsets,
    scaled_problem.reverse_constraints,
    nullptr,
    hyper_params,
    true);

  scaling.scale_problem();

  // generate a random initial solution in order to ensure scaling of solution vectors is
  // deterministic as well as the initial step size
  std::vector<double> initial_solution(scaled_problem.n_variables);
  std::mt19937 m{seed};
  std::generate(initial_solution.begin(), initial_solution.end(), [&m]() { return m(); });
  auto d_initial_solution = device_copy(initial_solution, handle_.get_stream());
  scaling.scale_primal(d_initial_solution);

  scaled_problem.preprocess_problem();

  detail::trivial_presolve(scaled_problem);

  std::vector<uint32_t> hashes;
  hashes.push_back(detail::compute_hash(d_initial_solution, handle_.get_stream()));
  hashes.push_back(scaled_problem.get_fingerprint());
  return detail::compute_hash(hashes);
}

TEST(problem, find_implied_integers)
{
  const raft::handle_t handle_{};

  auto path           = make_path_absolute("mip/fiball.mps");
  auto mps_data_model = cuopt::mps_parser::parse_mps<int, double>(path, false);
  auto op_problem     = mps_data_model_to_optimization_problem(&handle_, mps_data_model);
  auto presolver      = std::make_unique<detail::third_party_presolve_t<int, double>>();
  auto result         = presolver->apply(op_problem,
                                 cuopt::linear_programming::problem_category_t::MIP,
                                 cuopt::linear_programming::presolver_t::Papilo,
                                 false,
                                 1e-6,
                                 1e-12,
                                 20,
                                 1);
  ASSERT_NE(result.status, detail::third_party_presolve_status_t::INFEASIBLE);
  ASSERT_NE(result.status, detail::third_party_presolve_status_t::UNBNDORINFEAS);

  auto problem = detail::problem_t<int, double>(result.reduced_problem);
  problem.set_implied_integers(result.implied_integer_indices);
  ASSERT_TRUE(result.implied_integer_indices.size() > 0);
  auto var_types = host_copy(problem.variable_types, handle_.get_stream());
  // Find the index of the one continuous variable
  auto it = std::find_if(var_types.begin(), var_types.end(), [](var_t var_type) {
    return var_type == var_t::CONTINUOUS;
  });
  ASSERT_NE(it, var_types.end());
  ASSERT_EQ(problem.presolve_data.var_flags.size(), var_types.size());
  // Ensure it is an implied integer
  EXPECT_EQ(problem.presolve_data.var_flags.element(it - var_types.begin(), handle_.get_stream()),
            ((int)detail::problem_t<int, double>::var_flags_t::VAR_IMPLIED_INTEGER));
}

TEST(presolve, probing_cache_deterministic)
{
  spin_stream_raii_t spin_stream_1;

  std::vector<std::string> test_instances = {"mip/50v-10-free-bound.mps",
                                             "mip/neos5-free-bound.mps",
                                             "mip/neos5.mps",
                                             "mip/50v-10.mps",
                                             "mip/gen-ip054.mps",
                                             "mip/rmatr200-p5.mps"};
  for (const auto& test_instance : test_instances) {
    std::cout << "Running: " << test_instance << std::endl;
    unsigned long seed = std::random_device{}();
    std::cerr << "Tested with seed " << seed << "\n";
    auto path          = make_path_absolute(test_instance);
    uint32_t gold_hash = 0;
    for (int i = 0; i < 10; ++i) {
      auto hash = test_probing_cache_determinism(path, seed);
      if (i == 0) {
        gold_hash = hash;
        std::cout << "Gold hash: " << gold_hash << std::endl;
      } else {
        EXPECT_EQ(hash, gold_hash);
      }
    }
  }
}

TEST(presolve, mip_scaling_deterministic)
{
  spin_stream_raii_t spin_stream_1;
  spin_stream_raii_t spin_stream_2;

  std::vector<std::string> test_instances = {"mip/sct2.mps",
                                             "mip/thor50dday.mps",
                                             "mip/uccase9.mps",
                                             "mip/neos5-free-bound.mps",
                                             "mip/neos5.mps",
                                             "mip/50v-10.mps",
                                             "mip/gen-ip054.mps",
                                             "mip/rmatr200-p5.mps"};
  for (const auto& test_instance : test_instances) {
    std::cout << "Running: " << test_instance << std::endl;
    unsigned long seed = std::random_device{}();
    std::cerr << "Tested with seed " << seed << "\n";
    auto path          = make_path_absolute(test_instance);
    uint32_t gold_hash = 0;
    for (int i = 0; i < 10; ++i) {
      auto hash = test_scaling_determinism(path, seed);
      if (i == 0) {
        gold_hash = hash;
        std::cout << "Gold hash: " << gold_hash << std::endl;
      } else {
        EXPECT_EQ(hash, gold_hash);
      }
    }
  }
}

}  // namespace cuopt::linear_programming::test
