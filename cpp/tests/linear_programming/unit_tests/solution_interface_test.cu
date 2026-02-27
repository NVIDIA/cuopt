/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

/**
 * @file solution_interface_test.cu
 * @brief Unit tests for solution/problem interface conversions and polymorphism.
 *
 * Tests use small hand-constructed problems and solutions so they run fast,
 * don't require the solver, and assert on exact known values.
 * The one exception is mps_data_model_to_optimization_problem which tests
 * the MPS-parser-to-problem pipeline and legitimately needs a real file.
 */

#include <cuopt/linear_programming/cpu_optimization_problem_solution.hpp>
#include <cuopt/linear_programming/cpu_pdlp_warm_start_data.hpp>
#include <cuopt/linear_programming/gpu_optimization_problem_solution.hpp>
#include <cuopt/linear_programming/optimization_problem_interface.hpp>
#include <cuopt/linear_programming/optimization_problem_utils.hpp>
#include <cuopt/linear_programming/solve.hpp>
#include <mps_parser/parser.hpp>
#include <utilities/common_utils.hpp>
#include <utilities/copy_helpers.hpp>

#include <gtest/gtest.h>

#include <numeric>
#include <stdexcept>

namespace cuopt::linear_programming {

// =============================================================================
// Helpers: build tiny problems and solutions with known data
// =============================================================================

// A trivial 3-variable, 2-constraint LP:
//   min  1*x0 + 2*x1 + 3*x2
//   s.t. 4*x0 + 5*x1           <= 10   (CSR row 0)
//                  6*x1 + 7*x2 <= 20   (CSR row 1)
//   0 <= x0 <= 100,  0 <= x1 <= 200,  0 <= x2 <= 300
//
// CSR (row-major):  values = {4,5,6,7}  col_ind = {0,1,1,2}  offsets = {0,2,4}

static constexpr int kNVars = 3;
static constexpr int kNCons = 2;
static constexpr int kNnz   = 4;

static const double kObj[]    = {1.0, 2.0, 3.0};
static const double kVarLb[]  = {0.0, 0.0, 0.0};
static const double kVarUb[]  = {100.0, 200.0, 300.0};
static const double kRhs[]    = {10.0, 20.0};
static const double kCsrVal[] = {4.0, 5.0, 6.0, 7.0};
static const int kCsrInd[]    = {0, 1, 1, 2};
static const int kCsrOff[]    = {0, 2, 4};

// Populate a problem interface with the tiny LP above
template <typename ProblemT>
void populate_tiny_problem(ProblemT* problem)
{
  problem->set_objective_coefficients(kObj, kNVars);
  problem->set_variable_lower_bounds(kVarLb, kNVars);
  problem->set_variable_upper_bounds(kVarUb, kNVars);
  problem->set_constraint_bounds(kRhs, kNCons);
  problem->set_csr_constraint_matrix(kCsrVal, kNnz, kCsrInd, kNnz, kCsrOff, kNCons + 1);
}

// Build a cpu_lp_solution_t with known values
static std::unique_ptr<cpu_lp_solution_t<int, double>> make_cpu_lp_solution(bool with_warmstart)
{
  std::vector<double> primal = {1.0, 2.0, 3.0};
  std::vector<double> dual   = {0.5, 0.6};
  std::vector<double> rc     = {0.1, 0.2, 0.3};

  if (!with_warmstart) {
    return std::make_unique<cpu_lp_solution_t<int, double>>(std::move(primal),
                                                            std::move(dual),
                                                            std::move(rc),
                                                            pdlp_termination_status_t::Optimal,
                                                            /*primal_obj=*/-42.0,
                                                            /*dual_obj=*/-42.5,
                                                            /*solve_time=*/1.23,
                                                            /*l2_primal_residual=*/1e-8,
                                                            /*l2_dual_residual=*/2e-8,
                                                            /*gap=*/0.5,
                                                            /*num_iterations=*/100,
                                                            /*solved_by_pdlp=*/true);
  }

  cpu_pdlp_warm_start_data_t<int, double> ws;
  ws.current_primal_solution_                  = std::vector<double>(kNVars, 0.1);
  ws.current_dual_solution_                    = std::vector<double>(kNCons, 0.2);
  ws.initial_primal_average_                   = std::vector<double>(kNVars, 0.3);
  ws.initial_dual_average_                     = std::vector<double>(kNCons, 0.4);
  ws.current_ATY_                              = std::vector<double>(kNVars, 0.5);
  ws.sum_primal_solutions_                     = std::vector<double>(kNVars, 0.6);
  ws.sum_dual_solutions_                       = std::vector<double>(kNCons, 0.7);
  ws.last_restart_duality_gap_primal_solution_ = std::vector<double>(kNVars, 0.8);
  ws.last_restart_duality_gap_dual_solution_   = std::vector<double>(kNCons, 0.9);
  ws.initial_primal_weight_                    = 1.0;
  ws.initial_step_size_                        = 0.01;
  ws.total_pdlp_iterations_                    = 100;
  ws.total_pdhg_iterations_                    = 200;
  ws.last_candidate_kkt_score_                 = 1e-4;
  ws.last_restart_kkt_score_                   = 1e-5;
  ws.sum_solution_weight_                      = 50.0;
  ws.iterations_since_last_restart_            = 10;

  return std::make_unique<cpu_lp_solution_t<int, double>>(std::move(primal),
                                                          std::move(dual),
                                                          std::move(rc),
                                                          pdlp_termination_status_t::IterationLimit,
                                                          /*primal_obj=*/-42.0,
                                                          /*dual_obj=*/-42.5,
                                                          /*solve_time=*/1.23,
                                                          /*l2_primal_residual=*/1e-8,
                                                          /*l2_dual_residual=*/2e-8,
                                                          /*gap=*/0.5,
                                                          /*num_iterations=*/100,
                                                          /*solved_by_pdlp=*/true,
                                                          std::move(ws));
}

// Build a cpu_mip_solution_t with known values
static std::unique_ptr<cpu_mip_solution_t<int, double>> make_cpu_mip_solution()
{
  std::vector<double> sol = {1.0, 0.0, 1.0};
  return std::make_unique<cpu_mip_solution_t<int, double>>(std::move(sol),
                                                           mip_termination_status_t::Optimal,
                                                           /*objective=*/-99.0,
                                                           /*mip_gap=*/0.0,
                                                           /*solution_bound=*/-99.0,
                                                           /*total_solve_time=*/2.34,
                                                           /*presolve_time=*/0.1,
                                                           /*max_constraint_violation=*/0.0,
                                                           /*max_int_violation=*/0.0,
                                                           /*max_variable_bound_violation=*/0.0,
                                                           /*num_nodes=*/42,
                                                           /*num_simplex_iterations=*/500);
}

// Build a gpu_lp_solution_t with known device data (no solver needed)
static gpu_lp_solution_t<int, double> make_gpu_lp_solution()
{
  auto stream = rmm::cuda_stream_per_thread;

  rmm::device_uvector<double> primal(kNVars, stream);
  rmm::device_uvector<double> dual(kNCons, stream);
  rmm::device_uvector<double> rc(kNVars, stream);

  std::vector<double> h_primal = {1.0, 2.0, 3.0};
  std::vector<double> h_dual   = {0.5, 0.6};
  std::vector<double> h_rc     = {0.1, 0.2, 0.3};
  raft::copy(primal.data(), h_primal.data(), kNVars, stream);
  raft::copy(dual.data(), h_dual.data(), kNCons, stream);
  raft::copy(rc.data(), h_rc.data(), kNVars, stream);

  using info_t = optimization_problem_solution_t<int, double>::additional_termination_information_t;
  std::vector<info_t> term_stats(1);
  term_stats[0].primal_objective      = -42.0;
  term_stats[0].dual_objective        = -42.5;
  term_stats[0].solve_time            = 1.23;
  term_stats[0].l2_primal_residual    = 1e-8;
  term_stats[0].l2_dual_residual      = 2e-8;
  term_stats[0].gap                   = 0.5;
  term_stats[0].number_of_steps_taken = 100;
  term_stats[0].solved_by_pdlp        = true;

  std::vector<pdlp_termination_status_t> term_status = {pdlp_termination_status_t::Optimal};

  optimization_problem_solution_t<int, double> sol(
    primal, dual, rc, "obj", {}, {}, std::move(term_stats), std::move(term_status));

  return gpu_lp_solution_t<int, double>(std::move(sol));
}

// Build a gpu_mip_solution_t with known device data (no solver needed)
static gpu_mip_solution_t<int, double> make_gpu_mip_solution()
{
  auto stream = rmm::cuda_stream_per_thread;

  rmm::device_uvector<double> sol(kNVars, stream);
  std::vector<double> h_sol = {1.0, 0.0, 1.0};
  raft::copy(sol.data(), h_sol.data(), kNVars, stream);

  solver_stats_t<int, double> stats;
  stats.total_solve_time       = 2.34;
  stats.presolve_time          = 0.1;
  stats.num_nodes              = 42;
  stats.num_simplex_iterations = 500;
  stats.set_solution_bound(-99.0);

  mip_solution_t<int, double> mip_sol(std::move(sol),
                                      {},
                                      /*objective=*/-99.0,
                                      /*mip_gap=*/0.0,
                                      mip_termination_status_t::Optimal,
                                      /*max_constraint_violation=*/0.0,
                                      /*max_int_violation=*/0.0,
                                      /*max_variable_bound_violation=*/0.0,
                                      stats);

  return gpu_mip_solution_t<int, double>(std::move(mip_sol));
}

// =============================================================================
// Test fixture (only mps_data_model test needs files)
// =============================================================================

class SolutionInterfaceTest : public ::testing::Test {
 protected:
  void SetUp() override
  {
    const std::string& dir = cuopt::test::get_rapids_dataset_root_dir();
    lp_file_               = dir + "/linear_programming/afiro_original.mps";
  }
  std::string lp_file_;
};

// =============================================================================
// Polymorphism & method-dispatch tests
// =============================================================================

TEST_F(SolutionInterfaceTest, lp_solution_throws_on_mip_methods)
{
  auto sol                                  = make_gpu_lp_solution();
  lp_solution_interface_t<int, double>* ptr = &sol;

  EXPECT_THROW(ptr->get_mip_gap(), std::logic_error);
  EXPECT_THROW(ptr->get_solution_bound(), std::logic_error);
}

TEST_F(SolutionInterfaceTest, mip_solution_throws_on_lp_methods)
{
  auto sol                                   = make_gpu_mip_solution();
  mip_solution_interface_t<int, double>* ptr = &sol;

  EXPECT_THROW(ptr->get_dual_solution(), std::logic_error);
  EXPECT_THROW(ptr->get_dual_objective_value(), std::logic_error);
  EXPECT_THROW(ptr->get_reduced_costs(), std::logic_error);
}

TEST_F(SolutionInterfaceTest, lp_solution_polymorphic_methods)
{
  auto sol                                                     = make_gpu_lp_solution();
  optimization_problem_solution_interface_t<int, double>* base = &sol;

  EXPECT_FALSE(base->is_mip());
  EXPECT_NO_THROW(base->get_error_status());
  EXPECT_NEAR(base->get_solve_time(), 1.23, 1e-6);

  auto host_sol = base->get_solution_host();
  ASSERT_EQ(host_sol.size(), static_cast<size_t>(kNVars));
  EXPECT_NEAR(host_sol[0], 1.0, 1e-9);
  EXPECT_NEAR(host_sol[1], 2.0, 1e-9);
  EXPECT_NEAR(host_sol[2], 3.0, 1e-9);

  EXPECT_NEAR(base->get_objective_value(), -42.0, 1e-9);

  auto dual = base->get_dual_solution();
  ASSERT_EQ(dual.size(), static_cast<size_t>(kNCons));
  EXPECT_NEAR(dual[0], 0.5, 1e-9);
  EXPECT_NEAR(dual[1], 0.6, 1e-9);
}

TEST_F(SolutionInterfaceTest, mip_solution_polymorphic_methods)
{
  auto sol                                                     = make_gpu_mip_solution();
  optimization_problem_solution_interface_t<int, double>* base = &sol;

  EXPECT_TRUE(base->is_mip());
  EXPECT_NEAR(base->get_objective_value(), -99.0, 1e-9);
  EXPECT_NEAR(base->get_mip_gap(), 0.0, 1e-9);
  EXPECT_NEAR(base->get_solution_bound(), -99.0, 1e-9);

  auto host_sol = base->get_solution_host();
  ASSERT_EQ(host_sol.size(), static_cast<size_t>(kNVars));
  EXPECT_NEAR(host_sol[0], 1.0, 1e-9);
  EXPECT_NEAR(host_sol[2], 1.0, 1e-9);
}

TEST_F(SolutionInterfaceTest, termination_status_int_values)
{
  auto sol                                                     = make_gpu_lp_solution();
  optimization_problem_solution_interface_t<int, double>* base = &sol;

  int status = base->get_termination_status_int();
  EXPECT_EQ(status, CUOPT_TERIMINATION_STATUS_OPTIMAL);
}

// =============================================================================
// Problem conversion tests (hand-constructed tiny LP)
// =============================================================================

TEST_F(SolutionInterfaceTest, gpu_problem_to_optimization_problem)
{
  raft::handle_t handle;
  auto problem = std::make_unique<gpu_optimization_problem_t<int, double>>(&handle);
  populate_tiny_problem(problem.get());

  EXPECT_EQ(problem->get_n_variables(), kNVars);
  EXPECT_EQ(problem->get_n_constraints(), kNCons);

  auto concrete = problem->to_optimization_problem();
  EXPECT_EQ(concrete.get_n_variables(), kNVars);
  EXPECT_EQ(concrete.get_n_constraints(), kNCons);

  auto obj = cuopt::host_copy(concrete.get_objective_coefficients(), handle.get_stream());
  ASSERT_EQ(static_cast<int>(obj.size()), kNVars);
  for (int i = 0; i < kNVars; ++i) {
    EXPECT_NEAR(obj[i], kObj[i], 1e-9);
  }

  auto lb = cuopt::host_copy(concrete.get_variable_lower_bounds(), handle.get_stream());
  auto ub = cuopt::host_copy(concrete.get_variable_upper_bounds(), handle.get_stream());
  ASSERT_EQ(static_cast<int>(lb.size()), kNVars);
  ASSERT_EQ(static_cast<int>(ub.size()), kNVars);
  for (int i = 0; i < kNVars; ++i) {
    EXPECT_NEAR(lb[i], kVarLb[i], 1e-9);
    EXPECT_NEAR(ub[i], kVarUb[i], 1e-9);
  }

  auto vals = cuopt::host_copy(concrete.get_constraint_matrix_values(), handle.get_stream());
  ASSERT_EQ(static_cast<int>(vals.size()), kNnz);
  for (int i = 0; i < kNnz; ++i) {
    EXPECT_NEAR(vals[i], kCsrVal[i], 1e-9);
  }
}

TEST_F(SolutionInterfaceTest, cpu_problem_to_optimization_problem)
{
  raft::handle_t handle;
  auto problem = std::make_unique<cpu_optimization_problem_t<int, double>>(&handle);
  populate_tiny_problem(problem.get());

  EXPECT_EQ(problem->get_n_variables(), kNVars);
  EXPECT_EQ(problem->get_n_constraints(), kNCons);

  auto concrete = problem->to_optimization_problem();
  EXPECT_EQ(concrete.get_n_variables(), kNVars);
  EXPECT_EQ(concrete.get_n_constraints(), kNCons);

  auto obj = cuopt::host_copy(concrete.get_objective_coefficients(), handle.get_stream());
  ASSERT_EQ(static_cast<int>(obj.size()), kNVars);
  for (int i = 0; i < kNVars; ++i) {
    EXPECT_NEAR(obj[i], kObj[i], 1e-9);
  }

  auto lb = cuopt::host_copy(concrete.get_variable_lower_bounds(), handle.get_stream());
  auto ub = cuopt::host_copy(concrete.get_variable_upper_bounds(), handle.get_stream());
  for (int i = 0; i < kNVars; ++i) {
    EXPECT_NEAR(lb[i], kVarLb[i], 1e-9);
    EXPECT_NEAR(ub[i], kVarUb[i], 1e-9);
  }

  auto vals = cuopt::host_copy(concrete.get_constraint_matrix_values(), handle.get_stream());
  ASSERT_EQ(static_cast<int>(vals.size()), kNnz);
  for (int i = 0; i < kNnz; ++i) {
    EXPECT_NEAR(vals[i], kCsrVal[i], 1e-9);
  }
}

// This test legitimately uses the MPS parser since it tests that pipeline
TEST_F(SolutionInterfaceTest, mps_data_model_to_optimization_problem)
{
  auto mps_data = cuopt::mps_parser::parse_mps<int, double>(lp_file_);
  raft::handle_t handle;

  auto problem = mps_data_model_to_optimization_problem(&handle, mps_data);

  EXPECT_EQ(problem.get_n_variables(), mps_data.get_n_variables());
  EXPECT_EQ(problem.get_n_constraints(), mps_data.get_n_constraints());
  EXPECT_EQ(problem.get_nnz(), mps_data.get_nnz());

  auto csr_values  = cuopt::host_copy(problem.get_constraint_matrix_values(), handle.get_stream());
  auto csr_indices = cuopt::host_copy(problem.get_constraint_matrix_indices(), handle.get_stream());
  auto csr_offsets = cuopt::host_copy(problem.get_constraint_matrix_offsets(), handle.get_stream());
  EXPECT_EQ(static_cast<int>(csr_values.size()), mps_data.get_nnz());
  EXPECT_EQ(static_cast<int>(csr_indices.size()), mps_data.get_nnz());
  EXPECT_EQ(static_cast<int>(csr_offsets.size()), mps_data.get_n_constraints() + 1);

  auto obj_host = cuopt::host_copy(problem.get_objective_coefficients(), handle.get_stream());
  ASSERT_EQ(static_cast<int>(obj_host.size()), mps_data.get_n_variables());
  auto mps_obj = mps_data.get_objective_coefficients();
  for (size_t i = 0; i < obj_host.size(); ++i) {
    EXPECT_NEAR(obj_host[i], mps_obj[i], 1e-9) << "Mismatch at objective coeff " << i;
  }

  for (size_t i = 1; i < csr_offsets.size(); ++i) {
    EXPECT_GE(csr_offsets[i], csr_offsets[i - 1]) << "Non-monotonic CSR offset at " << i;
  }
  for (size_t i = 0; i < csr_indices.size(); ++i) {
    EXPECT_GE(csr_indices[i], 0) << "Negative column index at " << i;
    EXPECT_LT(csr_indices[i], mps_data.get_n_variables()) << "Out-of-range column index at " << i;
  }
}

// =============================================================================
// Solution conversion tests (hand-constructed, known values)
// =============================================================================

TEST_F(SolutionInterfaceTest, cpu_lp_solution_to_gpu)
{
  auto cpu_sol = make_cpu_lp_solution(/*with_warmstart=*/false);

  auto orig_primal = cpu_sol->get_primal_solution_host();
  ASSERT_EQ(orig_primal.size(), static_cast<size_t>(kNVars));

  auto gpu_sol = cpu_sol->to_gpu_solution(rmm::cuda_stream_per_thread);

  EXPECT_NEAR(gpu_sol.get_objective_value(0), -42.0, 1e-9);
  EXPECT_EQ(static_cast<int>(gpu_sol.get_termination_status()),
            static_cast<int>(pdlp_termination_status_t::Optimal));

  auto gpu_primal = cuopt::host_copy(gpu_sol.get_primal_solution(), rmm::cuda_stream_per_thread);
  ASSERT_EQ(gpu_primal.size(), orig_primal.size());
  for (size_t i = 0; i < gpu_primal.size(); ++i) {
    EXPECT_NEAR(gpu_primal[i], orig_primal[i], 1e-9);
  }

  // Verify all termination fields survive the round-trip
  auto info = gpu_sol.get_additional_termination_information(0);
  EXPECT_NEAR(info.l2_primal_residual, 1e-8, 1e-12);
  EXPECT_NEAR(info.l2_dual_residual, 2e-8, 1e-12);
  EXPECT_NEAR(info.gap, 0.5, 1e-9);
  EXPECT_EQ(info.number_of_steps_taken, 100);
  EXPECT_TRUE(info.solved_by_pdlp);
}

TEST_F(SolutionInterfaceTest, cpu_mip_solution_to_gpu)
{
  auto cpu_sol = make_cpu_mip_solution();

  auto gpu_sol = cpu_sol->to_gpu_solution(rmm::cuda_stream_per_thread);

  EXPECT_NEAR(gpu_sol.get_objective_value(), -99.0, 1e-9);
  EXPECT_EQ(static_cast<int>(gpu_sol.get_termination_status()),
            static_cast<int>(mip_termination_status_t::Optimal));

  auto host_sol = cuopt::host_copy(gpu_sol.get_solution(), rmm::cuda_stream_per_thread);
  ASSERT_EQ(host_sol.size(), static_cast<size_t>(kNVars));
  EXPECT_NEAR(host_sol[0], 1.0, 1e-9);
  EXPECT_NEAR(host_sol[1], 0.0, 1e-9);
  EXPECT_NEAR(host_sol[2], 1.0, 1e-9);
}

TEST_F(SolutionInterfaceTest, gpu_lp_solution_to_python_ret)
{
  auto sol        = make_gpu_lp_solution();
  auto python_ret = sol.to_python_lp_ret();

  EXPECT_TRUE(python_ret.is_gpu());
  EXPECT_NEAR(python_ret.primal_objective_, -42.0, 1e-9);
}

TEST_F(SolutionInterfaceTest, cpu_lp_solution_to_python_ret)
{
  auto cpu_sol    = make_cpu_lp_solution(/*with_warmstart=*/false);
  auto python_ret = cpu_sol->to_python_lp_ret();

  EXPECT_FALSE(python_ret.is_gpu());
  EXPECT_NEAR(python_ret.primal_objective_, -42.0, 1e-9);
}

TEST_F(SolutionInterfaceTest, gpu_mip_solution_to_python_ret)
{
  auto sol        = make_gpu_mip_solution();
  auto python_ret = sol.to_python_mip_ret();

  EXPECT_TRUE(python_ret.is_gpu());
  EXPECT_NEAR(python_ret.objective_, -99.0, 1e-9);
}

TEST_F(SolutionInterfaceTest, cpu_mip_solution_to_python_ret)
{
  auto cpu_sol    = make_cpu_mip_solution();
  auto python_ret = cpu_sol->to_python_mip_ret();

  EXPECT_FALSE(python_ret.is_gpu());
  EXPECT_NEAR(python_ret.objective_, -99.0, 1e-9);
}

// =============================================================================
// Warmstart conversion tests (hand-constructed)
// =============================================================================

TEST_F(SolutionInterfaceTest, cpu_warmstart_to_gpu)
{
  auto cpu_sol = make_cpu_lp_solution(/*with_warmstart=*/true);

  ASSERT_TRUE(cpu_sol->has_warm_start_data());

  auto orig_primal = cpu_sol->get_current_primal_solution_host();
  auto orig_dual   = cpu_sol->get_current_dual_solution_host();
  ASSERT_EQ(orig_primal.size(), static_cast<size_t>(kNVars));
  ASSERT_EQ(orig_dual.size(), static_cast<size_t>(kNCons));

  auto gpu_sol = cpu_sol->to_gpu_solution(rmm::cuda_stream_per_thread);

  auto& ws_data  = gpu_sol.get_pdlp_warm_start_data();
  auto ws_primal = cuopt::host_copy(ws_data.current_primal_solution_, rmm::cuda_stream_per_thread);
  auto ws_dual   = cuopt::host_copy(ws_data.current_dual_solution_, rmm::cuda_stream_per_thread);

  ASSERT_EQ(ws_primal.size(), orig_primal.size());
  ASSERT_EQ(ws_dual.size(), orig_dual.size());

  for (size_t i = 0; i < ws_primal.size(); ++i) {
    EXPECT_NEAR(ws_primal[i], orig_primal[i], 1e-9);
  }
  for (size_t i = 0; i < ws_dual.size(); ++i) {
    EXPECT_NEAR(ws_dual[i], orig_dual[i], 1e-9);
  }
}

// =============================================================================
// Problem interface copy_to_host tests (hand-constructed)
// =============================================================================

TEST_F(SolutionInterfaceTest, gpu_problem_copy_to_host_methods)
{
  raft::handle_t handle;
  auto problem = std::make_unique<gpu_optimization_problem_t<int, double>>(&handle);
  populate_tiny_problem(problem.get());

  std::vector<double> obj(kNVars);
  problem->copy_objective_coefficients_to_host(obj.data(), kNVars);
  for (int i = 0; i < kNVars; ++i) {
    EXPECT_NEAR(obj[i], kObj[i], 1e-9);
  }

  std::vector<double> lb(kNVars), ub(kNVars);
  problem->copy_variable_lower_bounds_to_host(lb.data(), kNVars);
  problem->copy_variable_upper_bounds_to_host(ub.data(), kNVars);
  for (int i = 0; i < kNVars; ++i) {
    EXPECT_NEAR(lb[i], kVarLb[i], 1e-9);
    EXPECT_NEAR(ub[i], kVarUb[i], 1e-9);
  }

  std::vector<double> rhs(kNCons);
  problem->copy_constraint_bounds_to_host(rhs.data(), kNCons);
  for (int i = 0; i < kNCons; ++i) {
    EXPECT_NEAR(rhs[i], kRhs[i], 1e-9);
  }

  std::vector<double> vals(kNnz);
  std::vector<int> inds(kNnz);
  std::vector<int> offs(kNCons + 1);
  problem->copy_constraint_matrix_to_host(
    vals.data(), inds.data(), offs.data(), kNnz, kNnz, kNCons + 1);
  for (int i = 0; i < kNnz; ++i) {
    EXPECT_NEAR(vals[i], kCsrVal[i], 1e-9);
    EXPECT_EQ(inds[i], kCsrInd[i]);
  }
  for (int i = 0; i <= kNCons; ++i) {
    EXPECT_EQ(offs[i], kCsrOff[i]);
  }
}

TEST_F(SolutionInterfaceTest, cpu_problem_copy_to_host_methods)
{
  auto problem = std::make_unique<cpu_optimization_problem_t<int, double>>();
  populate_tiny_problem(problem.get());

  std::vector<double> obj(kNVars);
  problem->copy_objective_coefficients_to_host(obj.data(), kNVars);
  for (int i = 0; i < kNVars; ++i) {
    EXPECT_NEAR(obj[i], kObj[i], 1e-9);
  }

  std::vector<double> lb(kNVars), ub(kNVars);
  problem->copy_variable_lower_bounds_to_host(lb.data(), kNVars);
  problem->copy_variable_upper_bounds_to_host(ub.data(), kNVars);
  for (int i = 0; i < kNVars; ++i) {
    EXPECT_NEAR(lb[i], kVarLb[i], 1e-9);
    EXPECT_NEAR(ub[i], kVarUb[i], 1e-9);
  }

  std::vector<double> rhs(kNCons);
  problem->copy_constraint_bounds_to_host(rhs.data(), kNCons);
  for (int i = 0; i < kNCons; ++i) {
    EXPECT_NEAR(rhs[i], kRhs[i], 1e-9);
  }

  std::vector<double> vals(kNnz);
  std::vector<int> inds(kNnz);
  std::vector<int> offs(kNCons + 1);
  problem->copy_constraint_matrix_to_host(
    vals.data(), inds.data(), offs.data(), kNnz, kNnz, kNCons + 1);
  for (int i = 0; i < kNnz; ++i) {
    EXPECT_NEAR(vals[i], kCsrVal[i], 1e-9);
    EXPECT_EQ(inds[i], kCsrInd[i]);
  }
  for (int i = 0; i <= kNCons; ++i) {
    EXPECT_EQ(offs[i], kCsrOff[i]);
  }
}

}  // namespace cuopt::linear_programming
