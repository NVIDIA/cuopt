/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
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

#pragma once

#include <algorithm>
#include <climits>
#include <vector>

#include <dual_simplex/mip_node.hpp>
#include <utilities/pcg.hpp>

namespace cuopt::linear_programming::dual_simplex {

template <typename i_t, typename f_t>
struct diving_root_t {
  mip_node_t<i_t, f_t> node;
  std::vector<f_t> lower;
  std::vector<f_t> upper;

  diving_root_t(mip_node_t<i_t, f_t>&& node, std::vector<f_t>&& lower, std::vector<f_t>&& upper)
    : node(std::move(node)), lower(std::move(lower)), upper(std::move(upper))
  {
  }

  friend bool operator>(const diving_root_t<i_t, f_t>& a, const diving_root_t<i_t, f_t>& b)
  {
    return a.node.objective_estimate > b.node.objective_estimate;
  }
};

// A min-heap for storing the starting nodes for the dives.
// This has a maximum size of INT16_MAX, such that the container
// will discard the least promising node if the queue is full.
template <typename i_t, typename f_t>
class diving_queue_t {
 private:
  std::vector<diving_root_t<i_t, f_t>> buffer;
  static constexpr i_t max_size_ = INT16_MAX;
  PCG rng;
  const double epsilon = 0.1;  // Probability to grab a random node

 public:
  diving_queue_t() { buffer.reserve(max_size_); }

  void set_rng_seed(uint64_t seed) { rng.set_seed(PCG::default_seed + seed); }

  void push(diving_root_t<i_t, f_t>&& node)
  {
    buffer.push_back(std::move(node));
    std::push_heap(buffer.begin(), buffer.end(), std::greater<>());
    if (buffer.size() > max_size() - 1) { buffer.pop_back(); }
  }

  void emplace(mip_node_t<i_t, f_t>&& node, std::vector<f_t>&& lower, std::vector<f_t>&& upper)
  {
    buffer.emplace_back(std::move(node), std::move(lower), std::move(upper));
    std::push_heap(buffer.begin(), buffer.end(), std::greater<>());
    if (buffer.size() > max_size() - 1) { buffer.pop_back(); }
  }

  diving_root_t<i_t, f_t> pop()
  {
    if (rng.next<f_t>() <= epsilon) {
      i_t idx = rng.uniform<i_t>(0, buffer.size());
      std::swap(buffer[idx], buffer.back());
      diving_root_t<i_t, f_t> node = std::move(buffer.back());
      buffer.pop_back();
      std::make_heap(buffer.begin(), buffer.end(), std::greater<>());
      return node;

    } else {
      std::pop_heap(buffer.begin(), buffer.end(), std::greater<>());
      diving_root_t<i_t, f_t> node = std::move(buffer.back());
      buffer.pop_back();
      return node;
    }
  }

  i_t size() const { return buffer.size(); }
  constexpr i_t max_size() const { return max_size_; }
  const diving_root_t<i_t, f_t>& top() const { return buffer.front(); }
  void clear() { buffer.clear(); }
};

}  // namespace cuopt::linear_programming::dual_simplex
