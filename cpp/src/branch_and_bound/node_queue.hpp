/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved. SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <branch_and_bound/mip_node.hpp>

#include <algorithm>
#include <memory>
#include <mutex>
#include <utility>
#include <vector>

namespace cuopt::mathematical_optimization::mip {

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
    ++num_entries_;
  }

  void push(T&& node)
  {
    buffer.push_back(std::move(node));
    std::push_heap(buffer.begin(), buffer.end(), comp);
    ++num_entries_;
  }

  template <typename... Args>
  void emplace(Args&&... args)
  {
    buffer.emplace_back(std::forward<Args>(args)...);
    std::push_heap(buffer.begin(), buffer.end(), comp);
    ++num_entries_;
    assert(num_entries_.load() == buffer.size());
  }

  T pop()
  {
    --num_entries_;
    std::pop_heap(buffer.begin(), buffer.end(), comp);
    T node = std::move(buffer.back());
    buffer.pop_back();
    assert(num_entries_.load() == buffer.size());
    return node;
  }

  size_t size() const { return num_entries_; }
  T& top() { return buffer.front(); }

  void clear()
  {
    buffer.clear();
    num_entries_ = 0;
  }

  bool empty() const { return num_entries_ == 0; }

  // Read-only access to underlying buffer for iteration without modification
  const std::vector<T>& data() const { return buffer; }

 private:
  std::vector<T> buffer;
  omp_atomic_t<size_t> num_entries_{0};
  Comp comp;
};

// A queue storing the nodes waiting to be explored.
template <typename i_t, typename f_t>
class node_queue_t {
 public:
  void push_atomic(mip_node_t<i_t, f_t>* new_node)
  {
    std::lock_guard lock(mutex_);
    push_lockfree(new_node);
  }

  void push_lockfree(mip_node_t<i_t, f_t>* new_node)
  {
    assert(new_node != nullptr);
    auto entry = std::make_shared<heap_entry_t>(new_node);
    best_first_heap_.push(entry);
    if (new_node->can_dive) diving_heap_.push(entry);
    lower_bound_ = best_first_heap_.top()->lower_bound;
  }

  mip_node_t<i_t, f_t>* pop()
  {
    std::lock_guard lock(mutex_);
    if (best_first_heap_.empty()) { return nullptr; }
    auto entry                 = best_first_heap_.pop();
    lower_bound_               = best_first_heap_.empty() ? std::numeric_limits<f_t>::infinity()
                                                          : best_first_heap_.top()->lower_bound;
    mip_node_t<i_t, f_t>* node = std::exchange(entry->node, nullptr);
    assert(node != nullptr);
    return node;
  }

  bool diving_init(const simplex::lp_problem_t<i_t, f_t>& lp,
                   mip_node_t<i_t, f_t>& start_node,
                   std::vector<f_t>& start_lower,
                   std::vector<f_t>& start_upper,
                   std::vector<bool>& bounds_changed)
  {
    std::lock_guard lock(mutex_);

    auto node = pop_diving();
    if (!node) return false;

    start_node  = node->detach_copy();
    start_lower = lp.lower;
    start_upper = lp.upper;
    std::fill(bounds_changed.begin(), bounds_changed.end(), false);
    node->get_variable_bounds(start_lower, start_upper, bounds_changed);
    return true;
  }

  bool steal_from(node_queue_t& victim, i_t nodes_to_steal)
  {
    assert(this != &victim);

    bool steal = false;
    std::scoped_lock lock(mutex_, victim.mutex_);

    for (i_t k = 0; k < nodes_to_steal; ++k) {
      if (victim.best_first_heap_.size() < nodes_to_steal) break;
      auto entry = victim.best_first_heap_.pop();

      // Invalidate the node for diving
      mip_node_t<i_t, f_t>* node = std::exchange(entry->node, nullptr);
      push_lockfree(node);
      victim.lower_bound_ = victim.best_first_heap_.empty()
                              ? std::numeric_limits<f_t>::infinity()
                              : victim.best_first_heap_.top()->lower_bound;
      steal               = true;
    }
    return steal;
  }

  i_t diving_queue_size() { return diving_heap_.size(); }
  i_t best_first_queue_size() { return best_first_heap_.size(); }

  f_t get_lower_bound()
  {
    return best_first_heap_.empty() ? std::numeric_limits<f_t>::infinity() : lower_bound_.load();
  }

 private:
  struct heap_entry_t {
    mip_node_t<i_t, f_t>* node = nullptr;
    f_t lower_bound            = -std::numeric_limits<f_t>::infinity();
    f_t score                  = std::numeric_limits<f_t>::infinity();

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

  mip_node_t<i_t, f_t>* pop_diving()
  {
    while (!diving_heap_.empty()) {
      auto entry = diving_heap_.pop();
      if (entry->node != nullptr) {
        entry->node->can_dive = false;
        return entry->node;
      }
    }

    return nullptr;
  }

  heap_t<std::shared_ptr<heap_entry_t>, lower_bound_comp> best_first_heap_;
  heap_t<std::shared_ptr<heap_entry_t>, score_comp> diving_heap_;
  omp_mutex_t mutex_;

  omp_atomic_t<f_t> lower_bound_{std::numeric_limits<f_t>::infinity()};
};

}  // namespace cuopt::mathematical_optimization::mip
