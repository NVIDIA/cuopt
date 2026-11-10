/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#include <pdlp/reflected_projection_direct.cuh>

#include <thrust/iterator/zip_iterator.h>
#include <cub/device/device_transform.cuh>
#include <raft/core/handle.hpp>
#include <rmm/device_uvector.hpp>

#include <gtest/gtest.h>

#include <array>
#include <bit>
#include <cstdint>
#include <limits>
#include <vector>

namespace cuopt::mathematical_optimization::pdlp::test {
namespace {

template <typename t>
void upload(rmm::device_uvector<t>& target, const std::vector<t>& source, cuda::stream_ref stream)
{
  RAFT_CUDA_TRY(cudaMemcpyAsync(
    target.data(), source.data(), source.size() * sizeof(t), cudaMemcpyHostToDevice, stream.get()));
}

void expect_same_bits(const rmm::device_uvector<double>& reference,
                      const rmm::device_uvector<double>& actual,
                      cuda::stream_ref stream)
{
  std::vector<double> expected(reference.size()), observed(actual.size());
  RAFT_CUDA_TRY(cudaMemcpyAsync(expected.data(),
                                reference.data(),
                                reference.size() * sizeof(double),
                                cudaMemcpyDeviceToHost,
                                stream.get()));
  RAFT_CUDA_TRY(cudaMemcpyAsync(observed.data(),
                                actual.data(),
                                actual.size() * sizeof(double),
                                cudaMemcpyDeviceToHost,
                                stream.get()));
  stream.sync();
  ASSERT_EQ(expected.size(), observed.size());
  for (size_t i = 0; i < expected.size(); ++i) {
    EXPECT_EQ(std::bit_cast<uint64_t>(expected[i]), std::bit_cast<uint64_t>(observed[i])) << i;
  }
}

template <template <typename> class projection_t, typename last_input_t>
void compare_with_cub(size_t count,
                      const std::vector<last_input_t>& last_input,
                      double reflection,
                      bool capture_graph)
{
  raft::handle_t handle;
  cuda::stream_ref stream{handle.get_stream()};
  const size_t storage = count + 2;  // Guard the tail, including the empty-input case.
  std::vector<double> current(storage), initial(storage), input3(storage), input4(storage);
  for (size_t i = 0; i < storage; ++i) {
    const int centered = static_cast<int>(i % 17) - 8;
    current[i]         = i % 11 == 0 ? -0.0 : 0.37 * centered;
    initial[i]         = -0.19 * centered;
    input3[i]          = 0.23 * (static_cast<int>(i % 7) - 3);
    input4[i]          = 0.31 * (static_cast<int>(i % 13) - 6);
  }
  rmm::device_uvector<double> reference(storage, stream), actual(storage, stream);
  rmm::device_uvector<double> reflected_reference(storage, stream),
    reflected_actual(storage, stream);
  rmm::device_uvector<double> d_initial(storage, stream), d_input3(storage, stream),
    d_input4(storage, stream), parameters(2, stream);
  rmm::device_uvector<last_input_t> d_last(storage, stream);
  upload(reference, current, stream);
  upload(actual, current, stream);
  upload(reflected_reference, initial, stream);
  upload(reflected_actual, initial, stream);
  upload(d_initial, initial, stream);
  upload(d_input3, input3, stream);
  upload(d_input4, input4, stream);
  upload(d_last, last_input, stream);
  const std::vector<double> initial_parameters{0.125, 0.5};
  upload(parameters, initial_parameters, stream);
  stream.sync();
  auto projection = projection_t<double>(parameters.data(), parameters.data() + 1, reflection);

  auto launch_both = [&]() {
    if (count != 0) {
      RAFT_CUDA_TRY(cub::DeviceTransform::Transform(
        cuda::std::make_tuple(
          reference.data(), d_initial.data(), d_input3.data(), d_input4.data(), d_last.data()),
        thrust::make_zip_iterator(reflected_reference.data(), reference.data()),
        count,
        projection,
        stream.get()));
    }
    EXPECT_TRUE(detail::try_direct_reflected_projection(true,
                                                        actual.data(),
                                                        d_initial.data(),
                                                        d_input3.data(),
                                                        d_input4.data(),
                                                        d_last.data(),
                                                        reflected_actual.data(),
                                                        count,
                                                        projection,
                                                        stream));
  };
  launch_both();
  expect_same_bits(reference, actual, stream);
  expect_same_bits(reflected_reference, reflected_actual, stream);

  if (capture_graph) {
    cudaGraph_t graph{};
    cudaGraphExec_t executable{};
    RAFT_CUDA_TRY(cudaStreamBeginCapture(stream.get(), cudaStreamCaptureModeThreadLocal));
    for (int step = 0; step < 24; ++step) {
      launch_both();
    }
    RAFT_CUDA_TRY(cudaStreamEndCapture(stream.get(), &graph));
    RAFT_CUDA_TRY(cudaGraphInstantiateWithFlags(&executable, graph, 0));
    for (const auto& values :
         std::array<std::array<double, 2>, 3>{{{1.3, 0.0}, {0.7, 0.999999}, {0.125, 0.5}}}) {
      RAFT_CUDA_TRY(cudaMemcpyAsync(parameters.data(),
                                    values.data(),
                                    2 * sizeof(double),
                                    cudaMemcpyHostToDevice,
                                    stream.get()));
      RAFT_CUDA_TRY(cudaGraphLaunch(executable, stream.get()));
      expect_same_bits(reference, actual, stream);
      expect_same_bits(reflected_reference, reflected_actual, stream);
    }
    RAFT_CUDA_TRY(cudaGraphExecDestroy(executable));
    RAFT_CUDA_TRY(cudaGraphDestroy(graph));
  }
}

std::vector<double2> primal_bounds(size_t count)
{
  const double infinity = std::numeric_limits<double>::infinity();
  const std::array<double2, 5> cases{
    {{-infinity, infinity}, {0.0, infinity}, {-infinity, 0.0}, {-0.5, 0.75}, {0.25, 0.25}}};
  std::vector<double2> bounds(count + 2);
  for (size_t i = 0; i < bounds.size(); ++i) {
    bounds[i] = cases[i % cases.size()];
  }
  return bounds;
}

TEST(DirectReflectedProjection, PrimalMatchesCubBitsAndPreservesTail)
{
  for (size_t count : {0, 1, 255, 256, 257, 1025}) {
    for (double reflection : {0.0, 1.0, 1.5}) {
      SCOPED_TRACE(count);
      SCOPED_TRACE(reflection);
      compare_with_cub<primal_reflected_projection>(count, primal_bounds(count), reflection, false);
    }
  }
}

TEST(DirectReflectedProjection, DualMatchesCubBitsAndPreservesTail)
{
  for (size_t count : {0, 1, 255, 256, 257, 1025}) {
    std::vector<double> upper(count + 2);
    for (size_t i = 0; i < upper.size(); ++i) {
      upper[i] = i % 3 == 0 ? std::numeric_limits<double>::infinity() : 3.0;
    }
    for (double reflection : {0.0, 1.0, 1.5}) {
      SCOPED_TRACE(count);
      SCOPED_TRACE(reflection);
      compare_with_cub<dual_reflected_projection>(count, upper, reflection, false);
    }
  }
}

TEST(DirectReflectedProjection, CapturedBatchReadsLiveScalarParameters)
{
  constexpr size_t count = 257;
  compare_with_cub<primal_reflected_projection>(count, primal_bounds(count), 1.0, true);
  compare_with_cub<dual_reflected_projection>(
    count, std::vector<double>(count + 2, 3.0), 1.0, true);
}

TEST(DirectReflectedProjection, UnsupportedTypeAndDisabledDispatchDoNotLaunch)
{
  raft::handle_t handle;
  cuda::stream_ref stream{handle.get_stream()};
  // Deliberately null pointers with a nonzero count make accidental dispatch observable.
  EXPECT_FALSE(detail::try_direct_reflected_projection(
    true,
    static_cast<float*>(nullptr),
    static_cast<const float*>(nullptr),
    static_cast<const float*>(nullptr),
    static_cast<const float*>(nullptr),
    static_cast<const float2*>(nullptr),
    static_cast<float*>(nullptr),
    257,
    primal_reflected_projection<float>(nullptr, nullptr, 1.0f),
    stream));
  EXPECT_FALSE(detail::try_direct_reflected_projection(
    false,
    static_cast<double*>(nullptr),
    static_cast<const double*>(nullptr),
    static_cast<const double*>(nullptr),
    static_cast<const double*>(nullptr),
    static_cast<const double2*>(nullptr),
    static_cast<double*>(nullptr),
    257,
    primal_reflected_projection<double>(nullptr, nullptr, 1.0),
    stream));
  stream.sync();
}

}  // namespace
}  // namespace cuopt::mathematical_optimization::pdlp::test
