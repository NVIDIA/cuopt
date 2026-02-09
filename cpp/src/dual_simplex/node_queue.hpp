/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved. SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <algorithm>
#include <memory>
#include <mutex>
#include <optional>
#include <utility>
#include <vector>

#include <dual_simplex/mip_node.hpp>

namespace cuopt::linear_programming::dual_simplex {

// This is a generic heap implementation based
// on the STL functions. The main benefit here is
// that we access the underlying container.
template <typename T, typename Comp>
class heap_t {
 public:
  heap_t()          = default;
  virtual ~heap_t() = default;

  void push(const T& node)
  {
    buffer.push_back(node);
    std::push_heap(buffer.begin(), buffer.end(), comp);
  }

  void push(T&& node)
  {
    buffer.push_back(std::move(node));
    std::push_heap(buffer.begin(), buffer.end(), comp);
  }

  template <typename... Args>
  void emplace(Args&&... args)
  {
    buffer.emplace_back(std::forward<Args>(args)...);
    std::push_heap(buffer.begin(), buffer.end(), comp);
  }

  std::optional<T> pop()
  {
    if (buffer.empty()) return std::nullopt;

    std::pop_heap(buffer.begin(), buffer.end(), comp);
    T node = std::move(buffer.back());
    buffer.pop_back();
    return node;
  }

  size_t size() const { return buffer.size(); }
  T& top() { return buffer.front(); }
  void clear() { buffer.clear(); }
  bool empty() const { return buffer.empty(); }

 private:
  std::vector<T> buffer;
  Comp comp;
};

// A queue storing the nodes waiting to be explored/dived from.
template <typename i_t, typename f_t>
class node_queue_t {
 private:
  struct heap_entry_t {
    mip_node_t<i_t, f_t>* node = nullptr;
    f_t lower_bound            = -inf;
    f_t score                  = inf;

    heap_entry_t(mip_node_t<i_t, f_t>* new_node)
      : node(new_node), lower_bound(new_node->lower_bound), score(new_node->objective_estimate)
    {
    }
  };

  // Comparision function for ordering the nodes based on their lower bound with
  // lowest one being explored first.
  struct lower_bound_comp {
    bool operator()(const std::shared_ptr<heap_entry_t>& a, const std::shared_ptr<heap_entry_t>& b)
    {
      // `a` will be placed after `b`
      return a->lower_bound > b->lower_bound;
    }
  };

  // Comparision function for ordering the nodes based on some score (currently the pseudocost
  // estimate) with the lowest being explored first.
  struct score_comp {
    bool operator()(const std::shared_ptr<heap_entry_t>& a, const std::shared_ptr<heap_entry_t>& b)
    {
      // `a` will be placed after `b`
      return a->score > b->score;
    }
  };

  heap_t<std::shared_ptr<heap_entry_t>, lower_bound_comp> best_first_heap;
  heap_t<std::shared_ptr<heap_entry_t>, score_comp> diving_heap;
  omp_mutex_t mutex;

 public:
  void push(mip_node_t<i_t, f_t>* new_node)
  {
    std::lock_guard<omp_mutex_t> lock(mutex);
    auto entry = std::make_shared<heap_entry_t>(new_node);
    best_first_heap.push(entry);
    diving_heap.push(entry);
  }

  std::optional<mip_node_t<i_t, f_t>*> pop_best_first()
  {
    std::lock_guard<omp_mutex_t> lock(mutex);
    auto entry = best_first_heap.pop();

    if (entry.has_value()) { return std::exchange(entry.value()->node, nullptr); }

    return std::nullopt;
  }

  std::optional<mip_node_t<i_t, f_t>*> pop_diving()
  {
    std::lock_guard<omp_mutex_t> lock(mutex);

    while (!diving_heap.empty()) {
      auto entry = diving_heap.pop();

      if (entry.has_value()) {
        if (auto node_ptr = entry.value()->node; node_ptr != nullptr) { return node_ptr; }
      }
    }

    return std::nullopt;
  }

  i_t diving_queue_size()
  {
    std::lock_guard<omp_mutex_t> lock(mutex);
    return diving_heap.size();
  }

  i_t best_first_queue_size()
  {
    std::lock_guard<omp_mutex_t> lock(mutex);
    return best_first_heap.size();
  }

  f_t get_lower_bound()
  {
    std::lock_guard<omp_mutex_t> lock(mutex);
    return best_first_heap.empty() ? inf : best_first_heap.top()->lower_bound;
  }

  mip_node_t<i_t, f_t>* bfs_top()
  {
    std::lock_guard<omp_mutex_t> lock(mutex);
    return best_first_heap.empty() ? nullptr : best_first_heap.top()->node;
  }
};

}  // namespace cuopt::linear_programming::dual_simplex
