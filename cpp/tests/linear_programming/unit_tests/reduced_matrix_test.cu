/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#include <cuopt/mathematical_optimization/io/mps_data_model.hpp>
#include <cuopt/mathematical_optimization/solve.hpp>
#include <pdlp/reduced_matrix.cuh>
#include <utilities/copy_helpers.hpp>
#include <utilities/logger.hpp>

#include <raft/core/device_setter.hpp>
#include <raft/core/handle.hpp>

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

namespace cuopt::mathematical_optimization::test {
namespace {

using pdlp::reduced_matrix_mode_t;

void append_reduction_log(const char* message, void* data)
{
  auto& log = *static_cast<std::string*>(data);
  log += message;
  log += '\n';
}

class reduced_matrix_log_capture_t {
 public:
  const std::string& text() const { return log_; }
  void clear() { log_.clear(); }

 private:
  cuopt::init_logger_t logger_{"", false};
  std::string log_;
  cuopt::scoped_log_callback_t callback_{append_reduction_log, &log_};
};

io::mps_data_model_t<int, double> make_reduction_test_model()
{
  io::mps_data_model_t<int, double> model;
  model.set_maximize(false);
  const std::vector<double> values{1, 2, 3, 4};
  const std::vector<int> columns{0, 1, 1, 2}, offsets{0, 2, 4};
  const std::vector<double> objective{1, 2, 3}, lower{0, 0, 0}, upper{10, 10, 10};
  const std::vector<double> row_bounds{2, 3};
  model.set_csr_constraint_matrix(values, columns, offsets);
  model.set_objective_coefficients(objective);
  model.set_variable_lower_bounds(lower);
  model.set_variable_upper_bounds(upper);
  model.set_constraint_lower_bounds(row_bounds);
  model.set_constraint_upper_bounds(row_bounds);
  return model;
}

bool reduction_enabled(const pdlp::pdlp_hyper_params_t& params,
                       int64_t nnz,
                       int concurrent_nnz_cutoff)
{
  return pdlp::reduced_matrix_enabled(
    params, nnz, concurrent_nnz_cutoff, false, false, false, false, false);
}

class ReducedMatrixPolicy : public testing::Test {
 protected:
  reduced_matrix_log_capture_t log_;
};

TEST_F(ReducedMatrixPolicy, DefaultUsesConfiguredConcurrentNnzCutoffInclusively)
{
  const pdlp_solver_settings_t<int, double> settings;
  EXPECT_EQ(settings.hyper_params.reduced_matrix, reduced_matrix_mode_t::DEFAULT);
  EXPECT_EQ(settings.concurrent_nnz_cutoff, 50'000'000);
  for (int cutoff : {settings.concurrent_nnz_cutoff, 7}) {
    SCOPED_TRACE(cutoff);
    log_.clear();
    EXPECT_FALSE(reduction_enabled(settings.hyper_params, cutoff - 1, cutoff));
    EXPECT_NE(log_.text().find("Column reduction disabled (DEFAULT)"), std::string::npos);
    EXPECT_EQ(log_.text().find("Column reduction enabled ("), std::string::npos);
    for (int64_t nnz : {int64_t{cutoff}, int64_t{cutoff} + 1}) {
      SCOPED_TRACE(nnz);
      log_.clear();
      EXPECT_TRUE(reduction_enabled(settings.hyper_params, nnz, cutoff));
      EXPECT_NE(log_.text().find("Column reduction enabled (DEFAULT)"), std::string::npos);
      EXPECT_NE(log_.text().find(std::to_string(nnz)), std::string::npos);
    }
  }
}

TEST_F(ReducedMatrixPolicy, DefaultRespectsZeroAndDisabledCutoff)
{
  const pdlp::pdlp_hyper_params_t params;
  for (int64_t nnz : {0LL, 1LL, 50'000'000LL, 100'000'001LL}) {
    SCOPED_TRACE(nnz);
    log_.clear();
    EXPECT_FALSE(reduction_enabled(params, nnz, -1));
    EXPECT_NE(log_.text().find("Column reduction disabled (DEFAULT)"), std::string::npos);
    EXPECT_EQ(log_.text().find("Column reduction enabled ("), std::string::npos);
    log_.clear();
    EXPECT_TRUE(reduction_enabled(params, nnz, 0));
    EXPECT_NE(log_.text().find("Column reduction enabled (DEFAULT)"), std::string::npos);
  }
}

TEST_F(ReducedMatrixPolicy, OffIsSilentRegardlessOfSizeCutoffAndDistributedPath)
{
  pdlp::pdlp_hyper_params_t params;
  params.reduced_matrix = reduced_matrix_mode_t::OFF;
  for (int cutoff : {-1, 0, 7, 50'000'000}) {
    SCOPED_TRACE(cutoff);
    for (int64_t nnz : {0LL, 100'000'001LL}) {
      SCOPED_TRACE(nnz);
      EXPECT_FALSE(reduction_enabled(params, nnz, cutoff));
      EXPECT_FALSE(
        pdlp::reduced_matrix_enabled(params, nnz, cutoff, false, false, false, false, true));
    }
  }
  EXPECT_TRUE(log_.text().empty());
}

TEST_F(ReducedMatrixPolicy, ExplicitColumnReductionBypassesSizeAndDisabledCutoff)
{
  pdlp::pdlp_hyper_params_t params;
  params.reduced_matrix = reduced_matrix_mode_t::COLUMN_REDUCTION;
  for (int cutoff : {-1, 0, 7, 50'000'000}) {
    SCOPED_TRACE(cutoff);
    for (int64_t nnz : {0LL, 1LL, 50'000'000LL, 100'000'001LL}) {
      SCOPED_TRACE(nnz);
      EXPECT_TRUE(reduction_enabled(params, nnz, cutoff));
    }
  }
  EXPECT_NE(log_.text().find("Column reduction enabled (COLUMN_REDUCTION)"), std::string::npos);
}

TEST_F(ReducedMatrixPolicy, UnsupportedPathsDisableBothAutomaticAndExplicitReduction)
{
  const std::array<const char*, 9> reasons{"legacy stream batching is not supported",
                                           "multiple PDLP climbers are not supported",
                                           "mixed-precision SpMV is not supported",
                                           "quadratic objectives are not supported",
                                           "multi-GPU PDLP is not supported",
                                           "reflected primal-dual updates are required",
                                           "adaptive step sizes are not supported",
                                           "artificial restarts in the main loop are not supported",
                                           "reflection coefficient must be 1"};
  for (auto mode : {reduced_matrix_mode_t::DEFAULT, reduced_matrix_mode_t::COLUMN_REDUCTION}) {
    SCOPED_TRACE(static_cast<int>(mode));
    pdlp::pdlp_hyper_params_t params;
    params.reduced_matrix = mode;
    for (int unsupported = 0; unsupported < 9; ++unsupported) {
      SCOPED_TRACE(reasons[unsupported]);
      auto unsupported_params = params;
      bool legacy_batch = false, batch = false, mixed_precision = false;
      bool quadratic = false, distributed = false;
      switch (unsupported) {
        case 0: legacy_batch = true; break;
        case 1: batch = true; break;
        case 2: mixed_precision = true; break;
        case 3: quadratic = true; break;
        case 4: distributed = true; break;
        case 5: unsupported_params.use_reflected_primal_dual = false; break;
        case 6: unsupported_params.use_adaptive_step_size_strategy = true; break;
        case 7: unsupported_params.artificial_restart_in_main_loop = true; break;
        case 8: unsupported_params.reflection_coefficient = 0.5; break;
      }
      for (int cutoff : {-1, 0, 50'000'000}) {
        SCOPED_TRACE(cutoff);
        log_.clear();
        EXPECT_FALSE(pdlp::reduced_matrix_enabled(unsupported_params,
                                                  50'000'000,
                                                  cutoff,
                                                  legacy_batch,
                                                  batch,
                                                  mixed_precision,
                                                  quadratic,
                                                  distributed));
        EXPECT_NE(log_.text().find("Column reduction disabled ("), std::string::npos);
        EXPECT_NE(log_.text().find(reasons[unsupported]), std::string::npos);
        EXPECT_EQ(log_.text().find("Column reduction enabled ("), std::string::npos);
      }
    }
  }
}

TEST_F(ReducedMatrixPolicy, DistributedPathTakesPriorityOverPlaceholderSizeAndCutoff)
{
  for (auto mode : {reduced_matrix_mode_t::DEFAULT, reduced_matrix_mode_t::COLUMN_REDUCTION}) {
    pdlp::pdlp_hyper_params_t params;
    params.reduced_matrix = mode;
    for (int cutoff : {-1, 0, 50'000'000}) {
      SCOPED_TRACE(cutoff);
      for (int64_t nnz : {0LL, 50'000'000LL}) {
        SCOPED_TRACE(nnz);
        log_.clear();
        EXPECT_FALSE(
          pdlp::reduced_matrix_enabled(params, nnz, cutoff, false, false, false, false, true));
        EXPECT_NE(log_.text().find("multi-GPU"), std::string::npos);
      }
    }
  }
}

TEST(ReducedMatrixSettings, SolveUsesConfiguredConcurrentNnzCutoff)
{
  raft::handle_t handle;
  auto model = make_reduction_test_model();
  ASSERT_EQ(model.get_nnz(), 4);
  const std::array<std::pair<int, bool>, 5> cases{
    {{3, true}, {4, true}, {5, false}, {-1, false}, {0, true}}};
  for (const auto& [cutoff, expected_enabled] : cases) {
    SCOPED_TRACE(cutoff);
    pdlp_solver_settings_t<int, double> settings;
    settings.method                = method_t::PDLP;
    settings.presolver             = presolver_t::None;
    settings.crossover             = false;
    settings.log_to_console        = false;
    settings.iteration_limit       = 2000;
    settings.time_limit            = 30;
    settings.concurrent_nnz_cutoff = cutoff;
    settings.set_optimality_tolerance(1e-6);
    ASSERT_EQ(settings.hyper_params.reduced_matrix, reduced_matrix_mode_t::DEFAULT);
    reduced_matrix_log_capture_t log;
    auto result = solve_lp(&handle, model, settings);
    EXPECT_EQ(result.get_termination_status(), pdlp_termination_status_t::Optimal);
    EXPECT_NEAR(result.get_objective_value(), 2.0, 1e-5);
    EXPECT_EQ(log.text().find("Column reduction enabled (DEFAULT)") != std::string::npos,
              expected_enabled);
    EXPECT_EQ(log.text().find("Column reduction disabled (DEFAULT)") != std::string::npos,
              !expected_enabled);
    handle.sync_stream();
  }
}

TEST(ReducedMatrixWorkspace, DisabledOperatorDoesNotAllocateProblemSizedBuffers)
{
  raft::handle_t handle;
  auto model = make_reduction_test_model();
  auto op    = mps_data_model_to_optimization_problem<int, double>(&handle, model);
  mip::problem_t<int, double> problem(op);
  pdlp::reduced_matrix_t<int, double> reduced(&handle, problem, false);
  EXPECT_EQ(reduced.full_workspace_size_bytes(), 0);
  EXPECT_FALSE(reduced.active());
  EXPECT_FALSE(reduced.refresh_pending());
  rmm::device_uvector<double> empty(0, handle.get_stream());
  EXPECT_FALSE(reduced.update_mode(0.0, false, empty, empty, empty));
  EXPECT_FALSE(reduced.request_refresh(true, empty, empty));
  EXPECT_FALSE(reduced.finish_refresh(empty, empty, empty));
  EXPECT_EQ(reduced.full_workspace_size_bytes(), 0);
  handle.sync_stream();
}

TEST(ReducedMatrixDistributed, ExplicitReductionIsDisabledWithOneVisibleGpu)
{
  if (raft::device_setter::get_device_count() != 1) {
    GTEST_SKIP() << "This regression exercises multi-GPU PDLP with exactly one visible GPU";
  }
  raft::handle_t handle;
  auto model = make_reduction_test_model();
  pdlp_solver_settings_t<int, double> settings;
  settings.method                      = method_t::PDLP;
  settings.num_gpus                    = -1;
  settings.presolver                   = presolver_t::None;
  settings.crossover                   = false;
  settings.log_to_console              = false;
  settings.iteration_limit             = 2000;
  settings.time_limit                  = 30;
  settings.hyper_params.reduced_matrix = reduced_matrix_mode_t::COLUMN_REDUCTION;
  settings.set_optimality_tolerance(1e-6);
  reduced_matrix_log_capture_t log;
  auto result = solve_lp(&handle, model, settings);
  EXPECT_EQ(result.get_termination_status(), pdlp_termination_status_t::Optimal);
  EXPECT_NEAR(result.get_objective_value(), 2.0, 1e-5);
  EXPECT_NE(log.text().find("Solving with multi-GPU PDLP on 1 GPUs"), std::string::npos);
  EXPECT_NE(log.text().find("multi-GPU"), std::string::npos);
  const auto disabled = log.text().find("Column reduction disabled (COLUMN_REDUCTION)");
  ASSERT_NE(disabled, std::string::npos);
  EXPECT_EQ(log.text().find("Column reduction disabled (", disabled + 1), std::string::npos);
  EXPECT_EQ(log.text().find("Column reduction enabled ("), std::string::npos);
  EXPECT_EQ(log.text().find("Reduced matrix activated"), std::string::npos);
  handle.sync_stream();
}

}  // namespace

class ReducedMatrix : public testing::TestWithParam<bool> {};

TEST_P(ReducedMatrix, CompactPrimalRefreshReleaseAndRestart)
{
  raft::handle_t handle;
  auto stream = handle.get_stream();
  optimization_problem_t<int, double> op(&handle);
  std::vector<double> values{1, -2, 3, -4, 5, -6, 7, -8, -2, 1, -4, 3, -6, 5, -8, 7};
  std::vector<int> columns{0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3, 4, 5, 6, 7};
  std::vector<int> offsets{0, 8, 16};
  std::vector<double> lower{-2, -3, -4, -5, -6, -7, -8, -9};
  std::vector<double> upper{3, 4, 5, 6, 7, 8, 9, 10};
  std::vector<double> objective{1000, -1000, 1000, -1000, 1000, -1000, 1000, .3};
  std::vector<double> row_lower{-100, -100}, row_upper{100, 100};
  op.set_csr_constraint_matrix(
    values.data(), values.size(), columns.data(), columns.size(), offsets.data(), offsets.size());
  op.set_objective_coefficients(objective.data(), objective.size());
  op.set_variable_lower_bounds(lower.data(), lower.size());
  op.set_variable_upper_bounds(upper.data(), upper.size());
  op.set_constraint_lower_bounds(row_lower.data(), row_lower.size());
  op.set_constraint_upper_bounds(row_upper.data(), row_upper.size());
  mip::problem_t<int, double> problem(op);
  problem.compute_transpose_of_problem();
  mip::problem_t<int, double> shared_structure(problem, false);
  RAFT_CUSPARSE_TRY(
    cusparseSetPointerMode(handle.get_cusparse_handle(), CUSPARSE_POINTER_MODE_DEVICE));

  std::vector<double> projected(8), current(8), initial(8);
  for (int j = 0; j < 8; ++j) {
    projected[j] = j == 7 ? 0.5 : (j % 2 == 0 ? lower[j] : upper[j]);
    current[j]   = projected[j] + 0.07 * (j + 1);
    initial[j]   = projected[j] - 0.11 * (j + 1);
  }
  auto d_projected = device_copy(projected, stream);
  auto d_current   = device_copy(current, stream);
  auto d_initial   = device_copy(initial, stream);
  std::vector<double> dual{-.2, .3};
  auto d_dual                 = device_copy(dual, stream);
  auto d_step                 = device_copy(std::vector<double>{.05}, stream);
  const double initial_weight = 0.5;
  rmm::device_scalar<double> d_weight(initial_weight, stream);
  rmm::device_uvector<double> activity(2, stream);
  pdlp::reduced_matrix_t<int, double> reduced(&handle, problem, true);
  if (GetParam()) {
    reduced.redirect_csr_structure(shared_structure);
    for (auto* structure : {&problem.offsets,
                            &problem.variables,
                            &problem.reverse_offsets,
                            &problem.reverse_constraints}) {
      structure->resize(0, stream);
      structure->shrink_to_fit(stream);
    }
    // Poison the shared problem's values: only its CSR structure may be used.
    RAFT_CUDA_TRY(cudaMemsetAsync(shared_structure.coefficients.data(),
                                  0,
                                  sizeof(double) * shared_structure.coefficients.size(),
                                  stream.get()));
    RAFT_CUDA_TRY(cudaMemsetAsync(shared_structure.reverse_coefficients.data(),
                                  0,
                                  sizeof(double) * shared_structure.reverse_coefficients.size(),
                                  stream.get()));
  }
  reduced_matrix_log_capture_t log;
  ASSERT_TRUE(reduced.update_mode(1e-4, false, d_projected, d_current, d_initial, 42));
  ASSERT_EQ(reduced.free_count(), 1);
  EXPECT_NE(log.text().find("Reduced matrix activated"), std::string::npos);
  EXPECT_NE(log.text().find("columns=1/8"), std::string::npos);
  EXPECT_NE(log.text().find("nnz=2/16"), std::string::npos);
  EXPECT_NE(log.text().find("iteration=42"), std::string::npos);

  int iteration = 0;
  for (int epoch = 0; epoch < 3; ++epoch) {
    // Odd/even segment lengths exercise both signs of the fixed-state coefficient.
    const int steps = epoch == 0 ? 1 : (epoch == 1 ? 2 : 197);
    for (int step = 0; step < steps; ++step, ++iteration) {
      const double weight = double(iteration + 1) / (iteration + 2);
      d_weight.set_value_async(weight, stream);
      reduced.compute_At_y(d_dual);
      reduced.update_primal(d_step, d_weight);
      reduced.compute_A_x(activity, d_weight);
      std::vector<double> expected_activity(2, 0.0);
      for (int j = 0; j < 8; ++j) {
        const double aty = values[j] * dual[0] + values[8 + j] * dual[1];
        projected[j]     = std::clamp(current[j] - .05 * (objective[j] - aty), lower[j], upper[j]);
        const double reflected = 2 * projected[j] - current[j];
        current[j]             = weight * reflected + (1 - weight) * initial[j];
        for (int row = 0; row < 2; ++row) {
          expected_activity[row] += values[8 * row + j] * reflected;
        }
      }
      const auto actual_activity = host_copy(activity, stream);
      for (int row = 0; row < 2; ++row) {
        EXPECT_NEAR(actual_activity[row], expected_activity[row], 1e-10);
      }
    }
    ASSERT_TRUE(reduced.request_refresh(true, d_current, d_initial));
    const auto actual_current = host_copy(d_current, stream);
    for (int j = 0; j < 8; ++j) {
      EXPECT_NEAR(actual_current[j], current[j], 1e-11);
    }

    // Supply the result of a full refresh. An interior projection releases column 0;
    // releasing all columns later must fall back to the full operator.
    if (epoch == 0) { projected[0] = current[0] = 0.0; }
    if (epoch == 2) { std::fill(projected.begin(), projected.end(), 0.0); }
    raft::update_device(d_projected.data(), projected.data(), projected.size(), stream);
    raft::update_device(d_current.data(), current.data(), current.size(), stream);
    ASSERT_TRUE(reduced.finish_refresh(d_projected, d_current, d_initial));
    if (epoch == 0) { EXPECT_EQ(reduced.free_count(), 2); }
    if (epoch == 1) {
      // Restart overwrites both full current and anchor vectors, then re-gathers compact state.
      current = initial = projected;
      raft::update_device(d_current.data(), current.data(), current.size(), stream);
      raft::update_device(d_initial.data(), initial.data(), initial.size(), stream);
      reduced.update_mode(1e-4, true, d_projected, d_current, d_initial);
      iteration = 0;
    }
  }
  EXPECT_FALSE(reduced.active());
}

INSTANTIATE_TEST_SUITE_P(SharedStructure, ReducedMatrix, testing::Bool());

}  // namespace cuopt::mathematical_optimization::test
