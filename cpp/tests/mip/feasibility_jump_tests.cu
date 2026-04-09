/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#include "../linear_programming/utilities/pdlp_test_utilities.cuh"
#include "mip_utils.cuh"

#include <cuopt/error.hpp>
#include <cuopt/linear_programming/solve.hpp>
#include <cuopt/linear_programming/utilities/internals.hpp>
#include <mip_heuristics/feasibility_jump/feasibility_jump.cuh>
#include <mip_heuristics/mip_scaling_strategy.cuh>
#include <mip_heuristics/solution/solution.cuh>
#include <mip_heuristics/solver_context.cuh>
#include <mps_parser/parser.hpp>
#include <pdlp/utilities/problem_checking.cuh>
#include <utilities/common_utils.hpp>
#include <utilities/seed_generator.cuh>

#include <raft/sparse/detail/cusparse_wrappers.h>
#include <raft/core/handle.hpp>
#include <raft/util/cudart_utils.hpp>

#include <gtest/gtest.h>

#include <thrust/count.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/sequence.h>

#include <cstdint>
#include <limits>
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

// Helper function to setup MIP solver and run FJ with given settings and initial solution
static fj_state_t run_fj_instance(std::string test_instance,
                                  const detail::fj_settings_t& fj_settings,
                                  fj_tweaks_t tweaks                   = {},
                                  std::vector<double> initial_solution = {},
                                  int determinism_mode                 = CUOPT_MODE_DETERMINISTIC)
{
  const raft::handle_t handle_{};
  std::cout << "Running: " << test_instance << std::endl;

  auto path = cuopt::test::get_rapids_dataset_root_dir() + ("/mip/" + test_instance);

  if (std::getenv("CUOPT_INSTANCE")) {
    path = make_path_absolute(std::getenv("CUOPT_INSTANCE"));
    std::cout << "Using instance from CUOPT_INSTANCE: " << path << std::endl;
  }

  cuopt::mps_parser::mps_data_model_t<int, double> mps_problem =
    cuopt::mps_parser::parse_mps<int, double>(path, false);
  handle_.sync_stream();
  auto op_problem = mps_data_model_to_optimization_problem(&handle_, mps_problem);
  problem_checking_t<int, double>::check_problem_representation(op_problem);

  init_handler(op_problem.get_handle_ptr());
  // run the problem constructor of MIP, so that we do bounds standardization
  detail::problem_t<int, double> problem(op_problem);
  problem.preprocess_problem();

  return run_fj(problem, fj_settings, tweaks, initial_solution, determinism_mode);
}

// FJ had a bug causing objective/violation values to explode in magnitude in certain scenarios.
// Ensure this is fixed on instances that historically triggered it.
static bool run_fj_check_no_obj_runoff(std::string test_instance)
{
  detail::fj_settings_t fj_settings;
  fj_settings.time_limit             = 30.;
  fj_settings.mode                   = detail::fj_mode_t::EXIT_NON_IMPROVING;
  fj_settings.n_of_minimums_for_exit = 5000;
  fj_settings.update_weights         = true;
  fj_settings.feasibility_run        = false;
  fj_settings.iteration_limit        = 20000;

  auto state = run_fj_instance(test_instance, fj_settings);

  // ensure that the objective and the violation in the FJ state are not too large (<1e60)
  EXPECT_LE(state.incumbent_violation, 1e60) << "FJ violation too large";
  EXPECT_LE(state.incumbent_objective, 1e60) << "FJ objective too large";

  return true;
}

static bool run_fj_check_objective(std::string test_instance, int iter_limit, double obj_target)
{
  // tolerance
  obj_target += 1e-4;

  detail::fj_settings_t fj_settings;
  fj_settings.time_limit             = 30.;
  fj_settings.mode                   = detail::fj_mode_t::EXIT_NON_IMPROVING;
  fj_settings.n_of_minimums_for_exit = 5000;
  fj_settings.update_weights         = true;
  fj_settings.feasibility_run        = obj_target == +std::numeric_limits<double>::infinity();
  fj_settings.iteration_limit        = iter_limit;

  auto state =
    run_fj_instance(test_instance, fj_settings, fj_tweaks_t{}, {}, CUOPT_MODE_DETERMINISTIC);
  auto& solution = state.solution;

  CUOPT_LOG_DEBUG("%s: Solution generated with FJ: is_feasible %d, objective %g (raw %g)",
                  test_instance.c_str(),
                  solution.get_feasible(),
                  solution.get_user_objective(),
                  solution.get_objective());

  EXPECT_TRUE(solution.get_feasible()) << test_instance << " is unexpectedly infeasible";
  EXPECT_LE(solution.get_user_objective(), obj_target)
    << test_instance << " objective " << solution.get_user_objective() << " exceeds target "
    << obj_target;

  return !solution.get_feasible() ? false : solution.get_user_objective() <= obj_target;
}

static bool run_fj_check_feasible(std::string test_instance)
{
  detail::fj_settings_t fj_settings;
  fj_settings.time_limit             = 30.;
  fj_settings.mode                   = detail::fj_mode_t::EXIT_NON_IMPROVING;
  fj_settings.n_of_minimums_for_exit = 5000;
  fj_settings.update_weights         = true;
  fj_settings.feasibility_run        = false;
  fj_settings.iteration_limit        = 25000;

  auto state     = run_fj_instance(test_instance, fj_settings);
  auto& solution = state.solution;

  bool previous_feasible = solution.get_feasible();
  double previous_obj    = solution.get_user_objective();

  EXPECT_TRUE(previous_feasible) << "Initial solution is unexpectedly infeasible";

  // again but with very large obj weight to force FJ into the infeasible region
  fj_tweaks_t tweaks;
  tweaks.objective_weight = 1e6;
  auto new_state     = run_fj_instance(test_instance, fj_settings, tweaks, state.solution_vector);
  auto& new_solution = new_state.solution;

  CUOPT_LOG_DEBUG("%s: Solution generated with FJ: is_feasible %d, objective %g (raw %g)",
                  test_instance.c_str(),
                  new_solution.get_feasible(),
                  new_solution.get_user_objective(),
                  new_solution.get_objective());

  // TODO: check neither worsens nor defeasibizes the solution
  EXPECT_TRUE(new_solution.get_feasible() == previous_feasible) << "FJ feasibility lost";
  EXPECT_LE(new_solution.get_user_objective(), previous_obj) << "FJ objective worsened";

  return true;
}

static bool run_fj_check_determinism(std::string test_instance, int iter_limit)
{
  detail::fj_settings_t fj_settings;
  fj_settings.time_limit             = std::numeric_limits<double>::max();
  fj_settings.mode                   = detail::fj_mode_t::EXIT_NON_IMPROVING;
  fj_settings.n_of_minimums_for_exit = 5000 * 1000;
  // fj_settings.work_limit             = 0.5;  // run for 0.5wu (~0.5s)
  fj_settings.update_weights      = true;
  fj_settings.feasibility_run     = false;
  fj_settings.iteration_limit     = iter_limit;
  fj_settings.load_balancing_mode = detail::fj_load_balancing_mode_t::ALWAYS_ON;
  fj_settings.seed                = cuopt::seed_generator::get_seed();

  auto state     = run_fj_instance(test_instance, fj_settings);
  auto& solution = state.solution;

  printf("%s[seed=%x]: Solution generated with FJ: is_feasible %d, objective %g (raw %g)",
         test_instance.c_str(),
         fj_settings.seed,
         solution.get_feasible(),
         solution.get_user_objective(),
         solution.get_objective());

  static std::unordered_map<std::string, double> first_val_map;
  if (first_val_map.count(test_instance) == 0) {
    first_val_map[test_instance] = solution.get_user_objective();
  }
  EXPECT_NEAR(solution.get_user_objective(), first_val_map[test_instance], 1.0)
    << test_instance << " determinism objective mismatch";

  return true;
}

TEST(mip_solve, feasibility_jump_determinism)
{
  cuopt::init_logger_t log("", true);

  int seed =
    std::getenv("CUOPT_SEED") ? std::stoi(std::getenv("CUOPT_SEED")) : std::random_device{}();

  for (const auto& [instance, iter_limit] : {std::make_pair("thor50dday.mps", 1000),
                                             std::make_pair("gen-ip054.mps", 1000),
                                             std::make_pair("50v-10.mps", 1000),
                                             std::make_pair("seymour1.mps", 1000),
                                             std::make_pair("rmatr200-p5.mps", 1000),
                                             std::make_pair("tr12-30.mps", 1000),
                                             std::make_pair("sct2.mps", 1000)}) {
    for (int i = 0; i < 10; i++) {
      cuopt::seed_generator::set_seed(seed);
      run_fj_check_determinism(instance, iter_limit);
    }
  }
}

}  // namespace cuopt::linear_programming::test
