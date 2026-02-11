/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#pragma once

#include <dual_simplex/bb_event.hpp>
#include <dual_simplex/branch_and_bound_worker.hpp>
#include <dual_simplex/diving_heuristics.hpp>
#include <dual_simplex/node_queue.hpp>
#include <utilities/work_limit_context.hpp>

#include <optional>

#include <cmath>
#include <deque>
#include <limits>
#include <memory>
#include <queue>
#include <vector>

namespace cuopt::linear_programming::dual_simplex {

template <typename i_t, typename f_t>
struct backlog_node_compare_t {
  bool operator()(const mip_node_t<i_t, f_t>* a, const mip_node_t<i_t, f_t>* b) const
  {
    if (a->lower_bound != b->lower_bound) { return a->lower_bound > b->lower_bound; }
    if (a->origin_worker_id != b->origin_worker_id) {
      return a->origin_worker_id > b->origin_worker_id;
    }
    return a->creation_seq > b->creation_seq;
  }
};

template <typename i_t, typename f_t>
struct queued_integer_solution_t {
  f_t objective;
  std::vector<f_t> solution;
  i_t depth{0};
  int worker_id{-1};
  int sequence_id{0};
  double work_timestamp{0.0};

  bool operator<(const queued_integer_solution_t& other) const
  {
    if (objective != other.objective) return objective < other.objective;
    if (worker_id != other.worker_id) return worker_id < other.worker_id;
    return sequence_id < other.sequence_id;
  }
};

template <typename i_t, typename f_t>
struct deterministic_snapshot_t {
  f_t upper_bound;
  pseudo_cost_snapshot_t<i_t, f_t> pc_snapshot;
  std::vector<f_t> incumbent;
  i_t total_lp_iters;
};

template <typename i_t, typename f_t, typename Derived>
class deterministic_worker_base_t : public branch_and_bound_worker_t<i_t, f_t> {
  using base_t = branch_and_bound_worker_t<i_t, f_t>;

 public:
  double clock{0.0};
  work_limit_context_t work_context;

  pseudo_cost_snapshot_t<i_t, f_t> pc_snapshot;
  f_t local_upper_bound{std::numeric_limits<f_t>::infinity()};

  // Diving-specific snapshots (ignored by BFS workers)
  std::vector<f_t> incumbent_snapshot;
  i_t total_lp_iters_snapshot{0};

  std::vector<queued_integer_solution_t<i_t, f_t>> integer_solutions;
  int next_solution_seq{0};

  i_t total_nodes_processed{0};
  i_t total_integer_solutions{0};
  double total_runtime{0.0};
  double total_nowork_time{0.0};

  deterministic_worker_base_t(int id,
                              const lp_problem_t<i_t, f_t>& original_lp,
                              const csr_matrix_t<i_t, f_t>& Arow,
                              const std::vector<variable_type_t>& var_types,
                              const simplex_solver_settings_t<i_t, f_t>& settings,
                              const std::string& context_name)
    : base_t(id, original_lp, Arow, var_types, settings), work_context(context_name)
  {
    work_context.deterministic = true;
  }

  void set_snapshots(const deterministic_snapshot_t<i_t, f_t>& snap)
  {
    local_upper_bound       = snap.upper_bound;
    pc_snapshot             = snap.pc_snapshot;
    incumbent_snapshot      = snap.incumbent;
    total_lp_iters_snapshot = snap.total_lp_iters;
  }

  bool has_work() const { return static_cast<const Derived*>(this)->has_work_impl(); }
};

template <typename i_t, typename f_t>
class deterministic_bfs_worker_t
  : public deterministic_worker_base_t<i_t, f_t, deterministic_bfs_worker_t<i_t, f_t>> {
  using base_t = deterministic_worker_base_t<i_t, f_t, deterministic_bfs_worker_t<i_t, f_t>>;

 public:
  // Node management
  std::deque<mip_node_t<i_t, f_t>*> plunge_stack;
  heap_t<mip_node_t<i_t, f_t>*, backlog_node_compare_t<i_t, f_t>> backlog;
  mip_node_t<i_t, f_t>* current_node{nullptr};
  mip_node_t<i_t, f_t>* last_solved_node{nullptr};

  // Event logging for deterministic replay
  bb_event_batch_t<i_t, f_t> events;
  int event_sequence{0};
  int32_t next_creation_seq{0};

  // BFS-specific state
  f_t local_lower_bound_ceiling{std::numeric_limits<f_t>::infinity()};
  bool recompute_bounds_and_basis{true};
  i_t nodes_processed_this_horizon{0};

  // BFS statistics
  i_t total_nodes_pruned{0};
  i_t total_nodes_branched{0};
  i_t total_nodes_infeasible{0};
  i_t total_nodes_assigned{0};

  explicit deterministic_bfs_worker_t(int id,
                                      const lp_problem_t<i_t, f_t>& original_lp,
                                      const csr_matrix_t<i_t, f_t>& Arow,
                                      const std::vector<variable_type_t>& var_types,
                                      const simplex_solver_settings_t<i_t, f_t>& settings)
    : base_t(id, original_lp, Arow, var_types, settings, "BB_Worker_" + std::to_string(id))
  {
  }

  bool has_work_impl() const
  {
    return current_node != nullptr || !plunge_stack.empty() || !backlog.empty();
  }

  void enqueue_node(mip_node_t<i_t, f_t>* node)
  {
    plunge_stack.push_front(node);
    ++total_nodes_assigned;
  }

  mip_node_t<i_t, f_t>* enqueue_children_for_plunge(mip_node_t<i_t, f_t>* down_child,
                                                    mip_node_t<i_t, f_t>* up_child,
                                                    rounding_direction_t preferred_direction)
  {
    if (!plunge_stack.empty()) {
      backlog.push(plunge_stack.back());
      plunge_stack.pop_back();
    }

    down_child->origin_worker_id = this->worker_id;
    down_child->creation_seq     = next_creation_seq++;
    up_child->origin_worker_id   = this->worker_id;
    up_child->creation_seq       = next_creation_seq++;

    mip_node_t<i_t, f_t>* first_child;
    if (preferred_direction == rounding_direction_t::UP) {
      plunge_stack.push_front(down_child);
      plunge_stack.push_front(up_child);
      first_child = up_child;
    } else {
      plunge_stack.push_front(up_child);
      plunge_stack.push_front(down_child);
      first_child = down_child;
    }
    return first_child;
  }

  mip_node_t<i_t, f_t>* dequeue_node()
  {
    if (current_node != nullptr) {
      mip_node_t<i_t, f_t>* node = current_node;
      current_node               = nullptr;
      return node;
    }
    if (!plunge_stack.empty()) {
      mip_node_t<i_t, f_t>* node = plunge_stack.front();
      plunge_stack.pop_front();
      return node;
    }
    auto node_opt = backlog.pop();
    return node_opt.has_value() ? node_opt.value() : nullptr;
  }

  size_t queue_size() const
  {
    return plunge_stack.size() + backlog.size() + (current_node != nullptr ? 1 : 0);
  }

  void record_event(bb_event_t<i_t, f_t> event)
  {
    event.event_sequence = event_sequence++;
    events.add(std::move(event));
  }

  void record_branched(
    mip_node_t<i_t, f_t>* node, i_t down_child_id, i_t up_child_id, i_t branch_var, f_t branch_val)
  {
    record_event(bb_event_t<i_t, f_t>::make_branched(this->clock,
                                                     this->worker_id,
                                                     node->creation_seq,
                                                     down_child_id,
                                                     up_child_id,
                                                     node->lower_bound,
                                                     branch_var,
                                                     branch_val));
    ++nodes_processed_this_horizon;
    ++this->total_nodes_processed;
    ++total_nodes_branched;
  }

  void record_integer_solution(mip_node_t<i_t, f_t>* node, f_t objective)
  {
    record_event(bb_event_t<i_t, f_t>::make_integer_solution(
      this->clock, this->worker_id, node->creation_seq, objective));
    ++nodes_processed_this_horizon;
    ++this->total_nodes_processed;
    ++this->total_integer_solutions;
  }

  void record_fathomed(mip_node_t<i_t, f_t>* node, f_t lower_bound)
  {
    record_event(bb_event_t<i_t, f_t>::make_fathomed(
      this->clock, this->worker_id, node->creation_seq, lower_bound));
    ++nodes_processed_this_horizon;
    ++this->total_nodes_processed;
    ++total_nodes_pruned;
  }

  void record_infeasible(mip_node_t<i_t, f_t>* node)
  {
    record_event(
      bb_event_t<i_t, f_t>::make_infeasible(this->clock, this->worker_id, node->creation_seq));
    ++nodes_processed_this_horizon;
    ++this->total_nodes_processed;
    ++total_nodes_infeasible;
  }

  void record_numerical(mip_node_t<i_t, f_t>* node)
  {
    record_event(
      bb_event_t<i_t, f_t>::make_numerical(this->clock, this->worker_id, node->creation_seq));
    ++nodes_processed_this_horizon;
    ++this->total_nodes_processed;
  }
};

template <typename i_t, typename f_t>
struct dive_queue_entry_t {
  mip_node_t<i_t, f_t> node;
  std::vector<f_t> resolved_lower;
  std::vector<f_t> resolved_upper;
};

template <typename i_t, typename f_t>
class deterministic_diving_worker_t
  : public deterministic_worker_base_t<i_t, f_t, deterministic_diving_worker_t<i_t, f_t>> {
  using base_t = deterministic_worker_base_t<i_t, f_t, deterministic_diving_worker_t<i_t, f_t>>;

 public:
  search_strategy_t diving_type{search_strategy_t::PSEUDOCOST_DIVING};

  // Diving-specific node management
  std::deque<dive_queue_entry_t<i_t, f_t>> dive_queue;
  std::vector<f_t> dive_lower;
  std::vector<f_t> dive_upper;

  // Root LP relaxation solution (constant, set once at construction)
  const std::vector<f_t>* root_solution{nullptr};

  // Diving state
  bool recompute_bounds_and_basis{true};

  // Diving statistics
  i_t total_nodes_explored{0};
  i_t total_dives{0};
  i_t lp_iters_this_dive{0};

  explicit deterministic_diving_worker_t(int id,
                                         search_strategy_t type,
                                         const lp_problem_t<i_t, f_t>& original_lp,
                                         const csr_matrix_t<i_t, f_t>& Arow,
                                         const std::vector<variable_type_t>& var_types,
                                         const simplex_solver_settings_t<i_t, f_t>& settings,
                                         const std::vector<f_t>* root_sol)
    : base_t(id, original_lp, Arow, var_types, settings, "Diving_Worker_" + std::to_string(id)),
      diving_type(type),
      root_solution(root_sol)
  {
    dive_lower = original_lp.lower;
    dive_upper = original_lp.upper;
  }

  deterministic_diving_worker_t(const deterministic_diving_worker_t&)            = delete;
  deterministic_diving_worker_t& operator=(const deterministic_diving_worker_t&) = delete;
  deterministic_diving_worker_t(deterministic_diving_worker_t&&)                 = default;
  deterministic_diving_worker_t& operator=(deterministic_diving_worker_t&&)      = default;

  bool has_work_impl() const { return !dive_queue.empty(); }

  void enqueue_dive_node(mip_node_t<i_t, f_t>* node, const lp_problem_t<i_t, f_t>& original_lp)
  {
    dive_queue_entry_t<i_t, f_t> entry;
    entry.resolved_lower = original_lp.lower;
    entry.resolved_upper = original_lp.upper;
    std::vector<bool> bounds_changed(original_lp.num_cols, false);
    node->get_variable_bounds(entry.resolved_lower, entry.resolved_upper, bounds_changed);
    entry.node = node->detach_copy();
    dive_queue.push_back(std::move(entry));
  }

  std::optional<dive_queue_entry_t<i_t, f_t>> dequeue_dive_node()
  {
    if (dive_queue.empty()) return std::nullopt;
    auto entry = std::move(dive_queue.front());
    dive_queue.pop_front();
    ++total_dives;
    return entry;
  }

  size_t dive_queue_size() const { return dive_queue.size(); }
  size_t queue_size() const { return dive_queue_size(); }  // Unified interface for pool

  void queue_integer_solution(f_t objective, const std::vector<f_t>& solution, i_t depth)
  {
    this->integer_solutions.push_back(
      {objective, solution, depth, this->worker_id, this->next_solution_seq++});
    ++this->total_integer_solutions;
  }

  branch_variable_t<i_t> variable_selection_from_snapshot(const std::vector<i_t>& fractional,
                                                          const std::vector<f_t>& solution) const
  {
    assert(root_solution != nullptr);
    return this->pc_snapshot.pseudocost_diving(fractional, solution, *root_solution);
  }

  branch_variable_t<i_t> guided_variable_selection(const std::vector<i_t>& fractional,
                                                   const std::vector<f_t>& solution) const
  {
    if (this->incumbent_snapshot.empty()) {
      return variable_selection_from_snapshot(fractional, solution);
    }
    return this->pc_snapshot.guided_diving(fractional, solution, this->incumbent_snapshot);
  }
};

template <typename i_t, typename f_t, typename WorkerT, typename Derived>
class deterministic_worker_pool_base_t {
 protected:
  std::vector<WorkerT> workers_;

 public:
  WorkerT& operator[](int worker_id) { return workers_[worker_id]; }
  const WorkerT& operator[](int worker_id) const { return workers_[worker_id]; }
  int size() const { return static_cast<int>(workers_.size()); }

  bool any_has_work() const
  {
    for (const auto& worker : workers_) {
      if (worker.has_work()) return true;
    }
    return false;
  }

  size_t total_queue_size() const
  {
    size_t total = 0;
    for (const auto& worker : workers_) {
      total += worker.queue_size();
    }
    return total;
  }

  bb_event_batch_t<i_t, f_t> collect_and_sort_events()
  {
    bb_event_batch_t<i_t, f_t> all_events;
    for (auto& worker : workers_) {
      static_cast<Derived*>(this)->collect_worker_events(worker, all_events);
    }
    all_events.sort_for_replay();
    return all_events;
  }

  auto begin() { return workers_.begin(); }
  auto end() { return workers_.end(); }
  auto begin() const { return workers_.begin(); }
  auto end() const { return workers_.end(); }
};

template <typename i_t, typename f_t>
class deterministic_bfs_worker_pool_t
  : public deterministic_worker_pool_base_t<i_t,
                                            f_t,
                                            deterministic_bfs_worker_t<i_t, f_t>,
                                            deterministic_bfs_worker_pool_t<i_t, f_t>> {
  using base_t = deterministic_worker_pool_base_t<i_t,
                                                  f_t,
                                                  deterministic_bfs_worker_t<i_t, f_t>,
                                                  deterministic_bfs_worker_pool_t<i_t, f_t>>;

 public:
  deterministic_bfs_worker_pool_t(int num_workers,
                                  const lp_problem_t<i_t, f_t>& original_lp,
                                  const csr_matrix_t<i_t, f_t>& Arow,
                                  const std::vector<variable_type_t>& var_types,
                                  const simplex_solver_settings_t<i_t, f_t>& settings)
  {
    this->workers_.reserve(num_workers);
    for (int i = 0; i < num_workers; ++i) {
      this->workers_.emplace_back(i, original_lp, Arow, var_types, settings);
    }
  }

  void collect_worker_events(deterministic_bfs_worker_t<i_t, f_t>& worker,
                             bb_event_batch_t<i_t, f_t>& all_events)
  {
    for (auto& event : worker.events.events) {
      all_events.add(std::move(event));
    }
    worker.events.clear();
  }
};

template <typename i_t, typename f_t>
class deterministic_diving_worker_pool_t
  : public deterministic_worker_pool_base_t<i_t,
                                            f_t,
                                            deterministic_diving_worker_t<i_t, f_t>,
                                            deterministic_diving_worker_pool_t<i_t, f_t>> {
  using base_t = deterministic_worker_pool_base_t<i_t,
                                                  f_t,
                                                  deterministic_diving_worker_t<i_t, f_t>,
                                                  deterministic_diving_worker_pool_t<i_t, f_t>>;

 public:
  deterministic_diving_worker_pool_t(int num_workers,
                                     const std::vector<search_strategy_t>& diving_types,
                                     const lp_problem_t<i_t, f_t>& original_lp,
                                     const csr_matrix_t<i_t, f_t>& Arow,
                                     const std::vector<variable_type_t>& var_types,
                                     const simplex_solver_settings_t<i_t, f_t>& settings,
                                     const std::vector<f_t>* root_solution)
  {
    this->workers_.reserve(num_workers);
    for (int i = 0; i < num_workers; ++i) {
      search_strategy_t type = diving_types[i % diving_types.size()];
      this->workers_.emplace_back(i, type, original_lp, Arow, var_types, settings, root_solution);
    }
  }

  void collect_worker_events(deterministic_diving_worker_t<i_t, f_t>&, bb_event_batch_t<i_t, f_t>&)
  {
  }
};

}  // namespace cuopt::linear_programming::dual_simplex
