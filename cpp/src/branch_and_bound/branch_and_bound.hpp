/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#pragma once

#include <branch_and_bound/bb_event.hpp>
#include <branch_and_bound/deterministic_workers.hpp>
#include <branch_and_bound/mip_node.hpp>
#include <branch_and_bound/node_queue.hpp>
#include <branch_and_bound/pseudo_costs.hpp>
#include <branch_and_bound/worker.hpp>
#include <branch_and_bound/worker_pool.hpp>

#include <cuts/cuts.hpp>

#include <dual_simplex/initial_basis.hpp>
#include <dual_simplex/phase2.hpp>
#include <dual_simplex/simplex_solver_settings.hpp>
#include <dual_simplex/solution.hpp>
#include <dual_simplex/solve.hpp>

#include <math_optimization/types.hpp>

#include <utilities/macros.cuh>
#include <utilities/omp_helpers.hpp>
#include <utilities/producer_sync.hpp>
#include <utilities/work_limit_context.hpp>
#include <utilities/work_unit_scheduler.hpp>

#include <cuopt/mathematical_optimization/pdlp/solver_settings.hpp>

#include <mip_heuristics/presolve/third_party_presolve.hpp>

#include <omp.h>

#include <atomic>
#include <functional>
#include <future>
#include <memory>
#include <vector>

namespace cuopt::mathematical_optimization::mip {

enum class mip_status_t {
  OPTIMAL         = 0,  // The optimal integer solution was found
  UNBOUNDED       = 1,  // The problem is unbounded
  INFEASIBLE      = 2,  // The problem is infeasible
  TIME_LIMIT      = 3,  // The solver reached a time limit
  NODE_LIMIT      = 4,  // The maximum number of nodes was reached
  ITERATION_LIMIT = 5,  // The maximum number of simplex iterations was reached
  NUMERICAL       = 6,  // The solver encountered a numerical error
  UNSET           = 7,  // The status is not set
  WORK_LIMIT      = 8,  // The solver reached a deterministic work limit
  SUBMIP_HALT     = 9   // Halt the solver
};

inline std::string mip_status_to_string(mip_status_t status)
{
  switch (status) {
    case mip_status_t::OPTIMAL: return "OPTIMAL";
    case mip_status_t::UNBOUNDED: return "UNBOUNDED";
    case mip_status_t::INFEASIBLE: return "INFEASIBLE";
    case mip_status_t::TIME_LIMIT: return "TIME_LIMIT";
    case mip_status_t::NODE_LIMIT: return "NODE_LIMIT";
    case mip_status_t::ITERATION_LIMIT: return "ITERATION_LIMIT";
    case mip_status_t::NUMERICAL: return "NUMERICAL";
    case mip_status_t::UNSET: return "UNSET";
    case mip_status_t::WORK_LIMIT: return "WORK_LIMIT";
    case mip_status_t::SUBMIP_HALT: return "SUBMIP_HALT";
  }
  return "UNKNOWN";
}

template <typename i_t, typename f_t>
struct clique_table_t;

template <typename i_t, typename f_t>
struct mip_symmetry_t;

template <typename i_t, typename f_t>
struct nondeterministic_policy_t;
template <typename i_t, typename f_t, typename WorkerT>
struct deterministic_policy_base_t;
template <typename i_t, typename f_t>
struct deterministic_bfs_policy_t;
template <typename i_t, typename f_t>
struct deterministic_diving_policy_t;

template <typename i_t, typename f_t>
class branch_and_bound_t {
 public:
  branch_and_bound_t(const simplex::user_problem_t<i_t, f_t>& user_problem,
                     const simplex::simplex_solver_settings_t<i_t, f_t>& solver_settings,
                     f_t start_time,
                     const probing_implied_bound_t<i_t, f_t>& probing_implied_bound,
                     std::shared_ptr<mip::clique_table_t<i_t, f_t>> clique_table = nullptr,
                     mip_symmetry_t<i_t, f_t>* symmetry                          = nullptr);

  // Set an initial guess based on the user_problem. This should be called before solve.
  void set_initial_guess(const std::vector<f_t>& user_guess) { guess_ = user_guess; }

  void set_submip_halt_callback(std::function<bool(f_t, f_t)> callback)
  {
    submip_halt_callback_ = std::move(callback);
  }

  // Set the root solution found by PDLP
  void set_root_relaxation_solution(const std::vector<f_t>& primal,
                                    const std::vector<f_t>& dual,
                                    const std::vector<f_t>& reduced_costs,
                                    f_t objective,
                                    f_t user_objective,
                                    i_t iterations,
                                    method_t method)
  {
    if (!is_root_solution_set) {
      root_crossover_soln_.x              = primal;
      root_crossover_soln_.y              = dual;
      root_crossover_soln_.z              = reduced_costs;
      root_objective_                     = objective;
      root_crossover_soln_.objective      = objective;
      root_crossover_soln_.user_objective = user_objective;
      root_crossover_soln_.iterations     = iterations;
      root_relax_solved_by                = method;
      root_crossover_solution_set_.store(true, std::memory_order_release);
    }
  }

  // Set a solution based on the user problem during the course of the solve
  bool set_solution_from_heuristics(const std::vector<f_t>& solution, heuristics_origin_t origin);

  // Apply a solution found by a CPU FJ worker.
  void set_solution_from_cpu_fj(f_t obj, const std::vector<f_t>& assignment, double work_units);

  // This queues the solution to be processed at the correct work unit timestamp
  void queue_external_solution_deterministic(const std::vector<f_t>& solution, double work_unit_ts);

  void set_user_bound_callback(std::function<void(f_t)> callback)
  {
    user_bound_callback_ = std::move(callback);
  }

  void set_concurrent_lp_root_solve(bool enable) { enable_concurrent_lp_root_solve_ = enable; }

  // Seed the global upper bound from an external source (e.g., early FJ during presolve).
  // `bound` must be in B&B's internal objective space.
  void set_initial_upper_bound(f_t bound);

  void set_initial_pseudocost(const pseudo_costs_t<i_t, f_t>& parent_pc,
                              const std::vector<i_t>& reduced_to_original);

  f_t get_upper_bound() const { return upper_bound_.load(); }
  bool has_solver_space_incumbent() const { return incumbent_.has_incumbent; }

  // Repair a low-quality solution from the heuristics.
  bool repair_solution(const std::vector<f_t>& leaf_edge_norms,
                       const std::vector<f_t>& potential_solution,
                       f_t& repaired_obj,
                       std::vector<f_t>& repaired_solution);

  f_t get_lower_bound();
  bool enable_concurrent_lp_root_solve() const { return enable_concurrent_lp_root_solve_; }
  std::atomic<int>* get_root_concurrent_halt() { return &root_concurrent_halt_; }
  void set_root_concurrent_halt(int value) { root_concurrent_halt_ = value; }
  simplex::lp_status_t solve_root_relaxation(
    simplex::simplex_solver_settings_t<i_t, f_t> const& lp_settings,
    simplex::lp_solution_t<i_t, f_t>& root_relax_soln,
    std::vector<simplex::variable_status_t>& root_vstatus,
    simplex::basis_update_mpf_t<i_t, f_t>& basis_update,
    std::vector<i_t>& basic_list,
    std::vector<i_t>& nonbasic_list,
    std::vector<f_t>& edge_norms);

  i_t find_reduced_cost_fixings(f_t upper_bound,
                                std::vector<f_t>& lower_bounds,
                                std::vector<f_t>& upper_bounds);

  // The main entry routine. Returns the solver status and populates solution with the incumbent.
  mip_status_t solve(simplex::mip_solution_t<i_t, f_t>& solution);

  work_limit_context_t& get_work_unit_context() { return work_unit_context_; }

  // Get producer sync for external heuristics (e.g., CPUFJ) to register
  producer_sync_t& get_producer_sync() { return producer_sync_; }

 private:
  const simplex::user_problem_t<i_t, f_t>& original_problem_;
  const simplex::simplex_solver_settings_t<i_t, f_t> settings_;
  const probing_implied_bound_t<i_t, f_t>& probing_implied_bound_;
  std::shared_ptr<mip::clique_table_t<i_t, f_t>> clique_table_;
  omp_atomic_t<bool> signal_extend_cliques_{false};
  mip_symmetry_t<i_t, f_t>* symmetry_;

  work_limit_context_t work_unit_context_{"B&B"};

  // Initial guess.
  std::vector<f_t> guess_;

  // LP relaxation
  csr_matrix_t<i_t, f_t> Arow_;
  simplex::lp_problem_t<i_t, f_t> original_lp_;
  std::vector<i_t> new_slacks_;
  std::vector<simplex::variable_type_t> var_types_;

  // Variable locks (see definition 3.3 from T. Achterberg, “Constraint Integer Programming,”
  // PhD, Technischen Universität Berlin, Berlin, 2007. doi: 10.14279/depositonce-1634).
  // Here we assume that the constraints are in the form `Ax = b, l <= x <= u`.
  std::vector<i_t> var_up_locks_;
  std::vector<i_t> var_down_locks_;

  // Mutex for the original LP
  // The heuristics threads look at the original LP. But the main thread modifies the
  // size of the original LP by adding slacks for cuts. Heuristic threads should lock
  // this mutex when accessing the original LP. The main thread should lock this mutex
  // when modifying the original LP.
  omp_mutex_t mutex_original_lp_;

  // Mutex for upper bound
  omp_mutex_t mutex_upper_;

  // Global upper bound in B&B's internal objective space.
  // A finite value implies an incumbent exists somewhere (solver-space in incumbent_, or
  // original-space in the mip_solver_context_t), but does NOT imply incumbent_.has_incumbent.
  omp_atomic_t<f_t> upper_bound_;

  // Callback for halting the solver. This passes the current upper and lower bound of the solver
  // in user space. The main use of this callback is to stop the sub-MIP solve when
  // the status of the main solve has changed (optimal, time/node/work limit, etc.) or
  // the sub-MIP become suboptimal (lower bound is greater than the current incumbent)
  std::function<bool(f_t, f_t)> submip_halt_callback_;

  // Solver-space incumbent tracked directly by B&B.
  simplex::mip_solution_t<i_t, f_t> incumbent_;

  // Structure with the general info of the solver.
  branch_and_bound_stats_t<i_t, f_t> exploration_stats_;

  // Mutex for repair
  omp_mutex_t mutex_repair_;
  std::vector<std::vector<f_t>> repair_queue_;

  // Variables for the root node in the search tree.
  std::vector<simplex::variable_status_t> root_vstatus_;
  std::vector<simplex::variable_status_t> crossover_vstatus_;
  f_t root_objective_;
  simplex::lp_solution_t<i_t, f_t> root_relax_soln_;
  simplex::lp_solution_t<i_t, f_t> root_crossover_soln_;
  method_t root_relax_solved_by{Unset};
  std::vector<f_t> edge_norms_;
  std::atomic<bool> root_crossover_solution_set_{false};
  omp_atomic_t<f_t> root_lp_current_lower_bound_;
  omp_atomic_t<bool> solving_root_relaxation_{false};
  bool enable_concurrent_lp_root_solve_{false};
  std::atomic<int> root_concurrent_halt_{0};
  std::atomic<int> node_concurrent_halt_{0};
  bool is_root_solution_set{false};
  bool has_initial_pseudocost_{false};

  // Pseudocosts
  pseudo_costs_t<i_t, f_t> pc_;

  // Search tree
  search_tree_t<i_t, f_t> search_tree_;

  // Worker pool dedicated to the best-first search
  bfs_worker_pool_t<i_t, f_t> bfs_worker_pool_;

  // Worker pool dedicated to diving
  diving_worker_pool_t<i_t, f_t> diving_worker_pool_;

  // Worker pool dedicated to recursive RINS
  diving_worker_pool_t<i_t, f_t> rins_worker_pool_;
  submip_stats_t rins_stats_;

  // Global status of the solver.
  omp_atomic_t<mip_status_t> solver_status_;
  omp_atomic_t<bool> is_running_{false};

  // Minimum number of node in the queue. When the queue size is less than
  // this variable, the nodes are added directly to the queue instead of
  // the local stack. This also determines the end of the ramp-up phase.
  i_t min_node_queue_size_;

  // In case, a best-first thread encounters a numerical issue when solving a node,
  // its blocks the progression of the lower bound as it cannot explore the
  // corresponding subtree.
  omp_atomic_t<f_t> lower_bound_numerical_;
  std::function<void(f_t)> user_bound_callback_;

  void print_table_header();
  void report_heuristic(f_t obj, heuristics_origin_t origin);
  void report(char symbol,
              f_t obj,
              f_t lower_bound,
              i_t node_depth,
              i_t node_int_infeas,
              double work_time = -1);

  enum class cut_pass_action_t { CONTINUE, BREAK, RETURN };
  struct cut_pass_result_t {
    cut_pass_action_t action{cut_pass_action_t::CONTINUE};
    mip_status_t status{mip_status_t::UNSET};
  };

  cut_pass_result_t do_cut_pass(i_t cut_pass,
                                simplex::mip_solution_t<i_t, f_t>& solution,
                                i_t& num_fractional,
                                std::vector<i_t>& fractional,
                                cut_generation_t<i_t, f_t>& cut_generation,
                                simplex::basis_update_mpf_t<i_t, f_t>& basis_update,
                                std::vector<i_t>& basic_list,
                                std::vector<i_t>& nonbasic_list,
                                variable_bounds_t<i_t, f_t>& variable_bounds,
                                cut_pool_t<i_t, f_t>& cut_pool,
                                cut_info_t<i_t, f_t>& cut_info,
                                simplex::simplex_solver_settings_t<i_t, f_t>& lp_settings,
                                i_t original_rows,
                                f_t& last_upper_bound,
                                f_t& last_objective,
                                f_t root_relax_objective,
                                i_t& cut_pool_size,
                                const std::vector<f_t>& saved_solution);

  // Set the solution when found at the root node
  void set_solution_at_root(simplex::mip_solution_t<i_t, f_t>& solution,
                            const cut_info_t<i_t, f_t>& cut_info);
  void update_user_bound(f_t lower_bound);

  // Set the final solution.
  void set_final_solution(simplex::mip_solution_t<i_t, f_t>& solution, f_t lower_bound);

  // Update the incumbent solution with the new feasible solution
  // found during branch and bound.
  void add_feasible_solution(f_t leaf_objective,
                             const std::vector<f_t>& leaf_solution,
                             i_t leaf_depth,
                             search_strategy_t thread_type);

  // Repairs low-quality solutions from the heuristics, if it is applicable.
  void repair_heuristic_solutions();

  // Launch a new diving worker from a given best-first worker.
  bool launch_diving_worker(bfs_worker_t<i_t, f_t>* bfs_worker);

  void snap_to_lattice(mip_node_t<i_t, f_t>* node_ptr, f_t leaf_obj);

  // Launch a new best-first worker from a given bfs worker.
  void launch_bfs_worker(bfs_worker_t<i_t, f_t>* worker);

  // Perform best-first search with a given bfs worker.
  void best_first_search_with(bfs_worker_t<i_t, f_t>* worker);

  // We use best-first to pick the `start_node` and then perform a depth-first search
  // from this node (i.e., a plunge). It can only backtrack to a sibling node.
  // Unexplored nodes in the subtree are inserted back into the global heap.
  void plunge_with(bfs_worker_t<i_t, f_t>* worker, mip_node_t<i_t, f_t>* start_node);

  // A worker attempts to steal nodes from another worker
  void work_stealing(bfs_worker_t<i_t, f_t>* worker);

  // Perform a deep dive in the subtree determined by the `start_node` in order
  // to find integer feasible solutions.
  void dive_with(diving_worker_t<i_t, f_t>* worker, i_t backtrack_limit);

  // Launch a new RINS worker
  bool launch_rins_worker(const std::vector<f_t>& sol);
  void set_solution_from_submip(const std::vector<f_t>& solution,
                                const third_party_presolve_t<i_t, f_t>& presolver,
                                f_t fixrate,
                                f_t obj);

  // Solve the RINS sub-MIP
  void solve_submip(diving_worker_t<i_t, f_t>* worker,
                    const std::vector<f_t>& current_incumbent,
                    i_t num_var_fixed,
                    i_t num_integers,
                    i_t submip_level,
                    std::string_view log_prefix);

  // Creates and solves the RINS sub-MIP
  void rins(diving_worker_t<i_t, f_t>* rins_worker, const std::vector<f_t>& node_solution);

  // Get the simplex settings for solving the LP of a single node
  simplex::simplex_solver_settings_t<i_t, f_t> get_node_lp_settings();

  // Solve the LP relaxation of a leaf node
  simplex::dual_status_t solve_node_lp(mip_node_t<i_t, f_t>* node_ptr,
                                       branch_and_bound_worker_t<i_t, f_t>* worker,
                                       branch_and_bound_stats_t<i_t, f_t>& stats,
                                       simplex::logger_t& log,
                                       i_t iter_limit = std::numeric_limits<i_t>::max());

  // Apply symmetry-based bound reductions (orbital fixing and, when
  // settings_.symmetry == 2, lexical reduction) to the current node.
  // Tightens worker->leaf_problem bounds and updates stats. Returns false
  // if lexical reduction proves the node infeasible.
  bool apply_symmetry_reductions(mip_node_t<i_t, f_t>* node_ptr,
                                 branch_and_bound_worker_t<i_t, f_t>* worker,
                                 branch_and_bound_stats_t<i_t, f_t>& stats);

  // Selects the variable to branch on.
  branch_variable_t<i_t> variable_selection(mip_node_t<i_t, f_t>* node_ptr,
                                            const std::vector<i_t>& fractional,
                                            branch_and_bound_worker_t<i_t, f_t>* worker);

  // Policy-based tree update shared between opportunistic and deterministic codepaths.
  template <typename WorkerT, typename Policy>
  std::pair<node_status_t, branch_direction_t> update_tree_impl(
    mip_node_t<i_t, f_t>* node_ptr,
    search_tree_t<i_t, f_t>& search_tree,
    WorkerT* worker,
    simplex::dual_status_t lp_status,
    Policy& policy);

  // Opportunistic tree update wrapper.
  std::pair<node_status_t, branch_direction_t> update_tree(
    mip_node_t<i_t, f_t>* node_ptr,
    search_tree_t<i_t, f_t>& search_tree,
    branch_and_bound_worker_t<i_t, f_t>* worker,
    simplex::dual_status_t lp_status,
    simplex::logger_t& log);

  // ============================================================================
  // Deterministic BSP (Bulk Synchronous Parallel) methods for deterministic parallel B&B
  // ============================================================================

  // Main deterministic coordinator loop
  void run_deterministic_coordinator(const csr_matrix_t<i_t, f_t>& Arow);

  // Gather all events generated, sort by WU timestamp, apply
  void deterministic_sort_replay_events(const bb_event_batch_t<i_t, f_t>& events);

  // Prune nodes held by workers based on new incumbent
  void deterministic_prune_worker_nodes_vs_incumbent();

  // Balance worker loads - redistribute nodes only if significant imbalance detected
  void deterministic_balance_worker_loads();

  node_status_t solve_node_deterministic(deterministic_bfs_worker_t<i_t, f_t>& worker,
                                         mip_node_t<i_t, f_t>* node_ptr,
                                         search_tree_t<i_t, f_t>& search_tree);

  f_t deterministic_compute_lower_bound();

  void run_deterministic_bfs_loop(deterministic_bfs_worker_t<i_t, f_t>& worker,
                                  search_tree_t<i_t, f_t>& search_tree);

  // Executed when all workers reach barrier
  // Handles termination logic serially in deterministic mode
  void deterministic_sync_callback();

  void run_deterministic_diving_loop(deterministic_diving_worker_t<i_t, f_t>& worker);

  void deterministic_dive(deterministic_diving_worker_t<i_t, f_t>& worker,
                          dive_queue_entry_t<i_t, f_t> entry);

  // Populate diving heap from BFS worker backlogs at sync
  void deterministic_populate_diving_heap();

  // Assign starting nodes to diving workers from diving heap
  void deterministic_assign_diving_nodes();

  // Collect and merge diving solutions at sync
  void deterministic_collect_diving_solutions_and_update_pseudocosts();

  template <typename PoolT, typename WorkerTypeGetter>
  void deterministic_process_worker_solutions(PoolT& pool, WorkerTypeGetter get_worker_type);

  template <typename PoolT>
  void deterministic_merge_pseudo_cost_updates(PoolT& pool);

  template <typename PoolT>
  void deterministic_broadcast_snapshots(PoolT& pool, const std::vector<f_t>& incumbent_snapshot);

  friend struct nondeterministic_policy_t<i_t, f_t>;
  friend struct deterministic_bfs_policy_t<i_t, f_t>;
  friend struct deterministic_diving_policy_t<i_t, f_t>;

 private:
  // unique_ptr as we only want to initialize these if we're in the deterministic codepath
  std::unique_ptr<deterministic_bfs_worker_pool_t<i_t, f_t>> deterministic_workers_;
  std::unique_ptr<cuopt::work_unit_scheduler_t> deterministic_scheduler_;
  mip_status_t deterministic_global_termination_status_{mip_status_t::UNSET};
  double deterministic_horizon_step_{5.0};     // Work unit step per horizon (tunable)
  double deterministic_current_horizon_{0.0};  // Current horizon target
  bool deterministic_mode_enabled_{false};
  int deterministic_horizon_number_{0};  // Current horizon number (for debugging)

  // Producer synchronization for external heuristics (CPUFJ)
  // B&B waits for registered producers at each horizon sync
  producer_sync_t producer_sync_;

  // Producer wait time statistics
  double total_producer_wait_time_{0.0};
  double max_producer_wait_time_{0.0};
  i_t producer_wait_count_{0};

  // Determinism heuristic solution queue - solutions received from GPU heuristics
  // Stored with work unit timestamp for deterministic ordering
  omp_mutex_t mutex_heuristic_queue_;
  std::vector<queued_integer_solution_t<i_t, f_t>> heuristic_solution_queue_;

  // ============================================================================
  // Determinism Diving state
  // ============================================================================

  // Diving worker pool
  // unique_ptr as we only want to initialize these if we're in the deterministic codepath
  std::unique_ptr<deterministic_diving_worker_pool_t<i_t, f_t>> deterministic_diving_workers_;

  // Diving heap - nodes available for diving, sorted by objective estimate
  struct diving_entry_t {
    mip_node_t<i_t, f_t>* node;
    f_t score;
  };
  struct diving_score_comp {
    bool operator()(const diving_entry_t& a, const diving_entry_t& b) const
    {
      if (a.score != b.score) return a.score > b.score;  // Min-heap by score
      if (a.node->origin_worker_id != b.node->origin_worker_id) {
        return a.node->origin_worker_id > b.node->origin_worker_id;
      }
      return a.node->creation_seq > b.node->creation_seq;
    }
  };
  heap_t<diving_entry_t, diving_score_comp> diving_heap_;
};

}  // namespace cuopt::mathematical_optimization::mip
