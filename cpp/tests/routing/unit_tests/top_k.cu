/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#include <routing/local_search/cycle_finder/cycle_graph.hpp>
#include <routing/routing_test.cuh>
#include <routing/util_kernels/top_k.cuh>
#include <routing/utilities/test_utilities.hpp>

#include <cub/block/block_load.cuh>
#include <cub/block/block_radix_sort.cuh>
#include <cub/block/block_shuffle.cuh>
#include <cub/block/block_store.cuh>

#include <thrust/sort.h>
#include <cub/cub.cuh>

#include <algorithm>
#include <iostream>
#include <random>
#include <vector>

namespace cuopt {
namespace routing {
namespace test {

struct time_it {
  double* elapsed_ms{nullptr};
  std::chrono::time_point<std::chrono::steady_clock> start;

  time_it(double* elapsed_ms) : elapsed_ms(elapsed_ms), start(std::chrono::steady_clock::now()) {}

  ~time_it()
  {
    if (elapsed_ms != nullptr) {
      *elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                      std::chrono::steady_clock::now() - start)
                      .count() /
                    1000.f;
    }
  }
};

auto constexpr write_diagonal = true;

template <int TPB, typename i_t>
__global__ void top_k_indices(i_t width,
                              raft::device_span<const double> row_costs,
                              raft::device_span<double> out_costs,
                              raft::device_span<i_t> out_indices)
{
  using ::cuopt::routing::detail::max_graph_nodes_per_row;
  const i_t row_id = blockIdx.x;

  i_t n_items_in_row = cuopt::routing::detail::
    top_k_indices_per_row<i_t, double, max_graph_nodes_per_row, TPB, write_diagonal>(
      row_id,
      row_costs.subspan(row_id * width, width),
      out_costs.subspan(row_id * max_graph_nodes_per_row, max_graph_nodes_per_row),
      out_indices.subspan(row_id * max_graph_nodes_per_row, max_graph_nodes_per_row));
}

template <typename i_t, typename f_t>
class top_cand_test_t : public routing_test_t<i_t, f_t>, public ::testing::TestWithParam<int> {
  int width;
  std::vector<double> cost;

  std::mt19937 rng;
  std::uniform_real_distribution<double> large_random;

  static constexpr int max_graph_nodes_per_row = ::cuopt::routing::detail::max_graph_nodes_per_row;

 public:
  top_cand_test_t()
    : width(GetParam()),
      cost(width * width),
      large_random(2 * width, std::numeric_limits<double>::max())
  {
  }

  void SetUp() override
  {
    // TODO : seed
    rng.seed(0);
    large_random.reset();
  }

  void test_top_k()
  {
    auto h_input_cost = create_data();

    rmm::device_uvector<double> d_input_cost(width * width, this->stream_view_);
    rmm::device_uvector<double> d_output_cost(width * max_graph_nodes_per_row, this->stream_view_);
    rmm::device_uvector<i_t> d_out_index(width * max_graph_nodes_per_row, this->stream_view_);

    raft::copy(d_input_cost.data(), h_input_cost.data(), h_input_cost.size(), this->stream_view_);

    this->stream_view_.synchronize();
    call_top_k(d_input_cost, d_output_cost, d_out_index);

    verify_top_k(h_input_cost, d_output_cost, d_out_index);
  }

  const std::vector<double>& create_data()
  {
    std::generate(cost.begin(), cost.end(), [&]() { return large_random(rng); });

    // corner cases
    top_k_on_right(0);
    top_k_on_left(1);
    top_k_split(2);
    // top_k requires diagonal elements to be double::max()
    for (int i = 0; i < width; ++i) {
      cost[i * width + i] = std::numeric_limits<double>::max();
    }

    return cost;
  }

  void top_k_split(int row)
  {
    assert(row < width);
    auto row_begin = cost.begin() + row * width;

    if (max_graph_nodes_per_row < width) {
      std::generate(row_begin + (max_graph_nodes_per_row / 2),
                    row_begin + width - (max_graph_nodes_per_row / 2),
                    [&]() { return large_random(rng); });
    }
  }

  void top_k_on_left(int row)
  {
    assert(row < width);

    auto row_begin = cost.begin() + row * width;

    std::generate(row_begin, row_begin + width, [&]() { return large_random(rng); });
    std::iota(row_begin, row_begin + std::min(max_graph_nodes_per_row, width), 0);
    std::shuffle(row_begin, row_begin + std::min(max_graph_nodes_per_row, width), rng);
  }

  void top_k_on_right(int row)
  {
    assert(row < width);

    auto row_end = cost.begin() + (row + 1) * width;

    std::generate(row_end - width, row_end, [&]() { return large_random(rng); });
    std::iota(row_end - std::min(max_graph_nodes_per_row, width), row_end, 0);
    std::shuffle(row_end - std::min(max_graph_nodes_per_row, width), row_end, rng);
  }

  void call_top_k(rmm::device_uvector<double>& input_cost,
                  rmm::device_uvector<double>& output_cost,
                  rmm::device_uvector<i_t>& out_index)
  {
    constexpr int TPB = 128;
    top_k_indices<TPB, i_t><<<width, TPB, 0, this->stream_view_>>>(width,
                                                                   cuopt::make_span(input_cost),
                                                                   cuopt::make_span(output_cost),
                                                                   cuopt::make_span(out_index));
    this->stream_view_.synchronize();
    RAFT_CUDA_TRY(cudaGetLastError());
  }

  void verify_top_k(const std::vector<double>& h_input_cost,
                    rmm::device_uvector<double>& d_output_cost,
                    rmm::device_uvector<i_t>& d_out_index)
  {
    this->stream_view_.synchronize();
    std::vector<double> h_output_cost(d_output_cost.size());
    raft::copy(
      h_output_cost.data(), d_output_cost.data(), d_output_cost.size(), this->stream_view_);
    this->stream_view_.synchronize();

    std::vector<i_t> h_sorted_index(d_out_index.size());
    raft::copy(h_sorted_index.data(), d_out_index.data(), d_out_index.size(), this->stream_view_);
    this->stream_view_.synchronize();
    std::vector<double> sorted_data(width);
    for (int i = 0; i < width; ++i) {
      // copy row i
      auto cost_row = h_input_cost.begin() + width * i;
      std::copy(cost_row, cost_row + width, sorted_data.begin());

      std::sort(sorted_data.begin(), sorted_data.end());

      auto output_width = std::min(width, max_graph_nodes_per_row);
      // check if costs are sorted appropriately
      for (int j = 0; j < output_width; ++j) {
        ASSERT_FLOAT_EQ(sorted_data[j], h_output_cost[j + max_graph_nodes_per_row * i]);
      }

      // check if elements pointed by h_sorted_index are the smallest
      //(within max_graph_nodes_per_row of sorted data)
      for (int j = 0; j < output_width; ++j) {
        EXPECT_EQ(cost_row[h_sorted_index.at(j + max_graph_nodes_per_row * i)], sorted_data[j]);
      }
    }
  }

  double bench_cub(int iter, const std::vector<double>& input_cost)
  {
    rmm::device_uvector<double> d_input_cost(input_cost.size(), this->stream_view_);
    rmm::device_uvector<double> d_output_cost(input_cost.size(), this->stream_view_);

    rmm::device_uvector<i_t> segment_marker(width + 1, this->stream_view_);
    rmm::device_uvector<i_t> in_index(input_cost.size(), this->stream_view_);
    rmm::device_uvector<i_t> out_sorted_index(input_cost.size(), this->stream_view_);

    thrust::tabulate(rmm::exec_policy(this->stream_view_),
                     segment_marker.begin(),
                     segment_marker.end(),
                     [width = width] __device__(auto i) { return i * width; });

    thrust::sequence(rmm::exec_policy(this->stream_view_), in_index.begin(), in_index.end());

    d_output_cost.resize(input_cost.size(), this->stream_view_);
    raft::copy(d_input_cost.data(), input_cost.data(), input_cost.size(), this->stream_view_);

    auto num_segments = segment_marker.size() - 1;
    auto num_items    = input_cost.size();  // width*width
    size_t tmp_storage_bytes;
    cub::DeviceSegmentedSort::SortPairs(static_cast<void*>(nullptr),
                                        tmp_storage_bytes,
                                        d_input_cost.data(),
                                        d_output_cost.data(),
                                        in_index.data(),
                                        out_sorted_index.data(),
                                        num_items,
                                        num_segments,
                                        segment_marker.data(),
                                        segment_marker.data() + 1,
                                        this->stream_view_);
    rmm::device_uvector<std::byte> d_cub_storage_bytes(0, this->stream_view_);
    d_cub_storage_bytes.resize(tmp_storage_bytes, this->stream_view_);
    double elapsed_ms;
    this->stream_view_.synchronize();
    {
      time_it t(&elapsed_ms);
      for (int i = 0; i < iter; ++i) {
        cub::DeviceSegmentedSort::SortPairs(d_cub_storage_bytes.data(),
                                            tmp_storage_bytes,
                                            d_input_cost.data(),
                                            d_output_cost.data(),
                                            in_index.data(),
                                            out_sorted_index.data(),
                                            num_items,
                                            num_segments,
                                            segment_marker.data(),
                                            segment_marker.data() + 1,
                                            this->stream_view_);
      }
      this->stream_view_.synchronize();
    }
    return elapsed_ms;
  }

  double bench_top_k(int iter, const std::vector<double>& input_cost)
  {
    rmm::device_uvector<double> d_input_cost(width * width, this->stream_view_);
    rmm::device_uvector<double> d_output_cost(width * max_graph_nodes_per_row, this->stream_view_);
    rmm::device_uvector<i_t> d_out_index(width * max_graph_nodes_per_row, this->stream_view_);

    raft::copy(d_input_cost.data(), input_cost.data(), input_cost.size(), this->stream_view_);

    double elapsed_ms;
    this->stream_view_.synchronize();
    {
      time_it t(&elapsed_ms);
      for (int i = 0; i < iter; ++i) {
        call_top_k(d_input_cost, d_output_cost, d_out_index);
      }
      this->stream_view_.synchronize();
    }
    return elapsed_ms;
  }

  void compare_bench()
  {
    auto h_input_cost = create_data();

    constexpr int iter = 30;

    auto top_k_time    = bench_top_k(iter, h_input_cost) / iter;
    auto cub_sort_time = bench_cub(iter, h_input_cost) / iter;
    std::cerr << "time taken for width " << width << "\ttop_k " << top_k_time << " ms\tcub_sort "
              << cub_sort_time << " ms\n";
  }
};

using top_cand_test = top_cand_test_t<int, double>;

// TEST_P(top_cand_test, bench) { compare_bench(); }

TEST_P(top_cand_test, test_top_k) { test_top_k(); }

INSTANTIATE_TEST_SUITE_P(top_k,
                         top_cand_test,
                         ::testing::Values(15000,
                                           10000,
                                           8000,
                                           6854,
                                           5000,
                                           4000,
                                           3500,
                                           3000,
                                           2500,
                                           2200,
                                           2000,
                                           1750,
                                           1500,
                                           1350,
                                           1200,
                                           1024,
                                           512,
                                           256));

}  // namespace test
}  // namespace routing
}  // namespace cuopt

CUOPT_TEST_PROGRAM_MAIN()
