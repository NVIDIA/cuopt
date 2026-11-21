/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#include <cuopt/mathematical_optimization/io/mps_data_model.hpp>
#include <cuopt/mathematical_optimization/solve.hpp>

#include <pdlp/batched_halpern.cuh>
#include <pdlp/reflected_projection_direct.cuh>
#include <utilities/manual_cuda_graph.cuh>

#include <cub/device/device_transform.cuh>
#include <raft/core/handle.hpp>
#include <rmm/device_uvector.hpp>

#include <gtest/gtest.h>

#include <bit>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <optional>
#include <string>
#include <tuple>
#include <vector>

namespace cuopt::mathematical_optimization::pdlp::test {
namespace {

template <typename t>
void upload(rmm::device_uvector<t>& target, const std::vector<t>& values, cuda::stream_ref stream)
{
  RAFT_CUDA_TRY(cudaMemcpyAsync(
    target.data(), values.data(), values.size() * sizeof(t), cudaMemcpyHostToDevice, stream.get()));
}

template <typename t>
std::vector<t> download(const rmm::device_uvector<t>& source, cuda::stream_ref stream)
{
  std::vector<t> result(source.size());
  RAFT_CUDA_TRY(cudaMemcpyAsync(
    result.data(), source.data(), source.size() * sizeof(t), cudaMemcpyDeviceToHost, stream.get()));
  stream.sync();
  return result;
}

void expect_same_bits(const std::vector<double>& expected, const std::vector<double>& actual)
{
  ASSERT_EQ(expected.size(), actual.size());
  for (size_t i = 0; i < expected.size(); ++i) {
    EXPECT_EQ(std::bit_cast<uint64_t>(expected[i]), std::bit_cast<uint64_t>(actual[i])) << i;
  }
}

struct scalar_state {
  explicit scalar_state(int batch_size)
    : stream{handle.get_stream()},
      batch_size{batch_size},
      age_reference{1, stream},
      age_batched{1, stream},
      weights_reference{static_cast<size_t>(batch_size + 2), stream},
      weights_batched{static_cast<size_t>(batch_size + 2), stream},
      final_weight{1, stream}
  {
  }

  void reset(int age)
  {
    const std::vector<int> ages{age};
    const std::vector<double> guards(batch_size + 2, -123.25);
    const std::vector<double> final_guard{-456.5};
    upload(age_reference, ages, stream);
    upload(age_batched, ages, stream);
    upload(weights_reference, guards, stream);
    upload(weights_batched, guards, stream);
    upload(final_weight, final_guard, stream);
    stream.sync();
  }

  void reference_step(int step)
  {
    RAFT_CUDA_TRY(cub::DeviceTransform::Transform(age_reference.data(),
                                                  weights_reference.data() + step,
                                                  1,
                                                  halpern_weight_from_iteration<int, double>{},
                                                  stream.get()));
    RAFT_CUDA_TRY(cub::DeviceTransform::Transform(
      age_reference.data(),
      age_reference.data(),
      1,
      [] __device__(int k) { return k + int(1); },
      stream.get()));
  }

  void reference()
  {
    for (int step = 0; step < batch_size; ++step) {
      reference_step(step);
    }
  }

  void prepare()
  {
    detail::prepare_batched_halpern_weights(
      age_batched.data(), weights_batched.data(), batch_size, stream);
  }

  void finish()
  {
    detail::finish_batched_halpern(
      age_batched.data(), final_weight.data(), weights_batched.data(), batch_size, stream);
  }

  void check(int expected_age)
  {
    const auto expected = download(weights_reference, stream);
    const auto observed = download(weights_batched, stream);
    expect_same_bits(expected, observed);
    EXPECT_EQ(download(age_reference, stream).at(0), expected_age);
    EXPECT_EQ(download(age_batched, stream).at(0), expected_age);
    EXPECT_EQ(std::bit_cast<uint64_t>(download(final_weight, stream).at(0)),
              std::bit_cast<uint64_t>(expected.at(batch_size - 1)));
  }

  raft::handle_t handle;
  cuda::stream_ref stream;
  int batch_size;
  rmm::device_uvector<int> age_reference, age_batched;
  rmm::device_uvector<double> weights_reference, weights_batched, final_weight;
};

TEST(BatchedHalpern, WeightsMatchSequentialDeviceArithmetic)
{
  for (int batch : {1, 4, 8, 16, 24}) {
    scalar_state state(batch);
    for (int age : {0, 1, 51, 199, 99999, std::numeric_limits<int>::max() - batch - 2}) {
      state.reset(age);
      state.reference();
      state.prepare();
      EXPECT_EQ(download(state.age_batched, state.stream).at(0), age);
      EXPECT_EQ(download(state.final_weight, state.stream).at(0), -456.5);
      state.finish();
      state.check(age + batch);
    }
  }
}

TEST(BatchedHalpern, CapturedReplayReadsLiveRestartAge)
{
  scalar_state state(24);
  state.reset(0);
  state.reference();
  state.prepare();
  state.finish();
  state.check(24);  // Warm the exact operations before capture.
  state.reset(0);
  manual_cuda_graph_t graph;
  const auto body = [&]() {
    state.reference();
    state.prepare();
    state.finish();
  };
  for (int age : {24, 48, 72}) {
    graph.run(state.stream, body);
    state.check(age);
    EXPECT_TRUE(graph.is_initialized());
  }
  for (int restart_age : {0, 199, 100000}) {
    state.reset(restart_age);
    graph.run(state.stream, body);
    state.check(restart_age + 24);
  }
}

TEST(BatchedHalpern, CaptureRecoveryAdvancesAgeOnce)
{
  scalar_state state(24);
  state.reset(0);
  state.reference();
  state.prepare();
  state.finish();
  state.check(24);
  state.reset(0);
  manual_cuda_graph_t graph;
  int calls       = 0;
  const auto body = [&]() {
    ++calls;  // Test-only injection bookkeeping, not production solver state.
    state.prepare();
    if (calls == 1) {
      // Synchronizing the captured stream deliberately invalidates this capture.
      RAFT_CUDA_TRY(cudaStreamSynchronize(state.stream.get()));
    }
    state.reference();
    state.finish();
  };
  graph.run(state.stream, body);
  EXPECT_EQ(calls, 2);
  EXPECT_FALSE(graph.is_initialized());
  state.check(24);
  graph.run(state.stream, body);
  EXPECT_TRUE(graph.is_initialized());
  state.check(48);
}

template <template <typename> class projection_t, typename bound_t>
void compare_projection_sequence(const std::vector<bound_t>& bounds)
{
  scalar_state state(24);
  state.reset(0);
  const size_t size = bounds.size();
  std::vector<double> current(size), initial(size), input3(size), input4(size);
  for (size_t i = 0; i < size; ++i) {
    current[i] = 0.31 * (static_cast<int>(i % 17) - 8);
    initial[i] = -0.2 * current[i];
    input3[i]  = 0.1 * (static_cast<int>(i % 7) - 3);
    input4[i]  = 0.01 * (static_cast<int>(i % 11) - 5);
  }
  rmm::device_uvector<double> reference(size, state.stream), candidate(size, state.stream);
  rmm::device_uvector<double> ref_reflected(size, state.stream), new_reflected(size, state.stream);
  rmm::device_uvector<double> d_initial(size, state.stream), d_input3(size, state.stream),
    d_input4(size, state.stream), step_size(1, state.stream);
  rmm::device_uvector<bound_t> d_bounds(size, state.stream);
  const std::vector<double> step{0.125};
  upload(reference, current, state.stream);
  upload(candidate, current, state.stream);
  upload(d_initial, initial, state.stream);
  upload(d_input3, input3, state.stream);
  upload(d_input4, input4, state.stream);
  upload(d_bounds, bounds, state.stream);
  upload(step_size, step, state.stream);
  state.stream.sync();
  state.prepare();
  for (int iteration = 0; iteration < 24; ++iteration) {
    state.reference_step(iteration);
    EXPECT_TRUE(detail::try_direct_reflected_projection(
      true,
      reference.data(),
      d_initial.data(),
      d_input3.data(),
      d_input4.data(),
      d_bounds.data(),
      ref_reflected.data(),
      size,
      projection_t<double>(step_size.data(), state.weights_reference.data() + iteration, 1.5),
      state.stream));
    EXPECT_TRUE(detail::try_direct_reflected_projection(
      true,
      candidate.data(),
      d_initial.data(),
      d_input3.data(),
      d_input4.data(),
      d_bounds.data(),
      new_reflected.data(),
      size,
      projection_t<double>(step_size.data(), state.weights_batched.data() + iteration, 1.5),
      state.stream));
  }
  state.finish();
  state.check(24);
  expect_same_bits(download(reference, state.stream), download(candidate, state.stream));
  expect_same_bits(download(ref_reflected, state.stream), download(new_reflected, state.stream));
}

TEST(BatchedHalpern, DirectProjectionSequencePreservesBits)
{
  std::vector<double2> primal_bounds(257, double2{-1.0, 1.0});
  std::vector<double> dual_upper_bounds(257, 2.0);
  compare_projection_sequence<primal_reflected_projection>(primal_bounds);
  compare_projection_sequence<dual_reflected_projection>(dual_upper_bounds);
}

class OrdinaryGraphBatching : public testing::Test {
 protected:
  void SetUp() override
  {
    const char* value = std::getenv("CUOPT_EXPERIMENTAL_GRAPH_BATCH_SIZE");
    if (value != nullptr) { previous_override_ = value; }
    ASSERT_EQ(unsetenv("CUOPT_EXPERIMENTAL_GRAPH_BATCH_SIZE"), 0);
  }

  void TearDown() override
  {
    if (previous_override_.has_value()) {
      EXPECT_EQ(setenv("CUOPT_EXPERIMENTAL_GRAPH_BATCH_SIZE", previous_override_->c_str(), 1), 0);
    } else {
      EXPECT_EQ(unsetenv("CUOPT_EXPERIMENTAL_GRAPH_BATCH_SIZE"), 0);
    }
  }

  static io::mps_data_model_t<int, double> make_model()
  {
    // Nearly parallel equality constraints keep this solve active beyond the initial
    // frequent-checkpoint phase, so the default path must actually replay graph batches.
    io::mps_data_model_t<int, double> model;
    model.set_maximize(false);
    model.set_csr_constraint_matrix(std::vector<double>{1, 1, 1, 1.0001},
                                    std::vector<int>{0, 1, 0, 1},
                                    std::vector<int>{0, 2, 4});
    model.set_objective_coefficients(std::vector<double>{1, 2.0001});
    model.set_variable_lower_bounds(std::vector<double>{0, 0});
    model.set_variable_upper_bounds(std::vector<double>{10, 10});
    model.set_constraint_lower_bounds(std::vector<double>{5, 5.0002});
    model.set_constraint_upper_bounds(std::vector<double>{5, 5.0002});
    return model;
  }

  static pdlp_solver_settings_t<int, double> make_settings()
  {
    pdlp_solver_settings_t<int, double> settings;
    settings.method          = method_t::PDLP;
    settings.presolver       = presolver_t::None;
    settings.log_to_console  = false;
    settings.iteration_limit = 1400;
    settings.time_limit      = 30;
    settings.set_optimality_tolerance(1e-14);
    return settings;
  }

 private:
  std::optional<std::string> previous_override_;
};

TEST_F(OrdinaryGraphBatching, DefaultExecutionMatchesSingleIterationGraphs)
{
  raft::handle_t handle;
  auto model    = make_model();
  auto settings = make_settings();
  ASSERT_EQ(setenv("CUOPT_EXPERIMENTAL_GRAPH_BATCH_SIZE", "1", 1), 0);
  auto sequential = solve_lp(&handle, model, settings);
  ASSERT_EQ(unsetenv("CUOPT_EXPERIMENTAL_GRAPH_BATCH_SIZE"), 0);

  auto batched = solve_lp(&handle, model, settings);
  ASSERT_EQ(sequential.get_termination_status(), pdlp_termination_status_t::IterationLimit);
  EXPECT_EQ(batched.get_termination_status(), sequential.get_termination_status());
  EXPECT_EQ(batched.get_additional_termination_information().number_of_steps_taken,
            sequential.get_additional_termination_information().number_of_steps_taken);
  expect_same_bits(download(sequential.get_primal_solution(), handle.get_stream()),
                   download(batched.get_primal_solution(), handle.get_stream()));
  expect_same_bits(download(sequential.get_dual_solution(), handle.get_stream()),
                   download(batched.get_dual_solution(), handle.get_stream()));
}

TEST_F(OrdinaryGraphBatching, AutomaticColumnReductionPreservesSolutionAndIterationLimit)
{
  raft::handle_t handle;
  auto model                   = make_model();
  auto settings                = make_settings();
  settings.large_nnz_threshold = 0;

  ASSERT_EQ(setenv("CUOPT_EXPERIMENTAL_GRAPH_BATCH_SIZE", "1", 1), 0);
  auto sequential = solve_lp(&handle, model, settings);
  ASSERT_EQ(unsetenv("CUOPT_EXPERIMENTAL_GRAPH_BATCH_SIZE"), 0);
  auto automatic = solve_lp(&handle, model, settings);
  ASSERT_EQ(sequential.get_termination_status(), pdlp_termination_status_t::IterationLimit);
  EXPECT_EQ(automatic.get_termination_status(), sequential.get_termination_status());
  EXPECT_EQ(automatic.get_additional_termination_information().number_of_steps_taken,
            sequential.get_additional_termination_information().number_of_steps_taken);
  expect_same_bits(download(sequential.get_primal_solution(), handle.get_stream()),
                   download(automatic.get_primal_solution(), handle.get_stream()));
  expect_same_bits(download(sequential.get_dual_solution(), handle.get_stream()),
                   download(automatic.get_dual_solution(), handle.get_stream()));
}

class OrdinaryGraphBatchingCheckpoints
  : public OrdinaryGraphBatching,
    public testing::WithParamInterface<std::tuple<int, int, bool>> {};

TEST_P(OrdinaryGraphBatchingCheckpoints, PreservesCheckpointAndIterationLimitResults)
{
  raft::handle_t handle;
  auto model                                  = make_model();
  auto settings                               = make_settings();
  settings.iteration_limit                    = std::get<0>(GetParam());
  settings.hyper_params.major_iteration       = std::get<1>(GetParam());
  settings.hyper_params.use_conditional_major = std::get<2>(GetParam());

  // Preserve the test's checkpoint schedule instead of replacing it with a solver-mode preset.
  ASSERT_EQ(setenv("CUOPT_EXPERIMENTAL_GRAPH_BATCH_SIZE", "1", 1), 0);
  auto sequential = solve_lp(&handle, model, settings, true, false);
  ASSERT_EQ(unsetenv("CUOPT_EXPERIMENTAL_GRAPH_BATCH_SIZE"), 0);
  auto batched = solve_lp(&handle, model, settings, true, false);

  ASSERT_EQ(sequential.get_termination_status(), pdlp_termination_status_t::IterationLimit);
  EXPECT_EQ(batched.get_termination_status(), sequential.get_termination_status());
  EXPECT_EQ(batched.get_additional_termination_information().number_of_steps_taken,
            sequential.get_additional_termination_information().number_of_steps_taken);
  expect_same_bits(download(sequential.get_primal_solution(), handle.get_stream()),
                   download(batched.get_primal_solution(), handle.get_stream()));
  expect_same_bits(download(sequential.get_dual_solution(), handle.get_stream()),
                   download(batched.get_dual_solution(), handle.get_stream()));
}

INSTANTIATE_TEST_SUITE_P(CheckpointBoundaries,
                         OrdinaryGraphBatchingCheckpoints,
                         testing::Values(std::tuple{1407, 200, true},
                                         std::tuple{1407, 211, true},
                                         std::tuple{1407, 211, false},
                                         std::tuple{10007, 200, true}));

}  // namespace
}  // namespace cuopt::mathematical_optimization::pdlp::test
