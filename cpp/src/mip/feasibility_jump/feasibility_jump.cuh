/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved. SPDX-License-Identifier: Apache-2.0
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

#include "utils.cuh"

#include <cuopt/linear_programming/mip/solver_settings.hpp>
#include <mip/diversity/weights.cuh>
#include <mip/logger.cuh>
#include <mip/problem/problem.cuh>
#include <mip/solution/solution.cuh>
#include <mip/solver.cuh>
#include <mip/utils.cuh>

#include <utilities/event_handler.cuh>

#define FJ_DEBUG_LOAD_BALANCING 0
#define FJ_SINGLE_STEP          0

namespace cuopt::linear_programming::detail {

static constexpr int TPB_resetmoves  = raft::WarpSize * 4;
static constexpr int TPB_heavyvars   = raft::WarpSize * 16;
static constexpr int TPB_heavycstrs  = raft::WarpSize * 4;
static constexpr int TPB_localmin    = raft::WarpSize * 4;
static constexpr int TPB_setval      = raft::WarpSize * 16;
static constexpr int TPB_liftmoves   = raft::WarpSize * 4;
static constexpr int TPB_loadbalance = raft::WarpSize * 4;

struct fj_hyper_parameters_t {
  // The number of moves to evaluate, if there are many positive-score
  // variables available.
  int max_sampled_moves = raft::WarpSize * 16;
  // The probability of choosing a random positive-score variable.
  double random_var_probability = 0.04;
  // The probability of choosing a variable using a random constraint's
  // non-zero coefficient after updating weights.
  double random_cstr_probability = 0.16;
  // The period in iterations of each global move value update
  // (all variables being updated vs. considering only the selected one)
  int global_move_update_period      = 10;
  int heavy_move_update_period       = 50;
  int sync_period                    = 200;
  int lhs_refresh_period             = 1000;
  int allow_infeasibility_iterations = 200;
  // The value added to the objective weight everytime a new best solution is
  // found in order to move towards better solutions
  double objective_weight_increment       = 0.01;
  int load_balancing_variable_threshold   = 300;
  int load_balancing_constraint_threshold = 5000;
  int load_balancing_variable_split_size  = 50;

  double breakthrough_move_epsilon    = 1e-4;
  int tabu_tenure_min                 = 3;
  int tabu_tenure_max                 = 13;
  double excess_improvement_weight    = (1.0 / 2.0);
  double weight_smoothing_probability = 0.0003;

  double fractional_score_multiplier = 100;
  double rounding_second_stage_split = 0.1;

  double small_move_tabu_threshold = 1e-6;
  int small_move_tabu_tenure       = 4;

  // load-balancing related settings
  int old_codepath_total_var_to_relvar_ratio_threshold = 200;
  int load_balancing_codepath_min_varcount             = 3200;
};

enum fj_move_type_t {
  FJ_MOVE_BEGIN = 0,
  FJ_MOVE_LIFT  = FJ_MOVE_BEGIN,
  FJ_MOVE_BREAKTHROUGH,
  FJ_MOVE_SIZE,
};

enum class fj_mode_t {
  FIRST_FEASIBLE,     // iterate until a feasible solution is found, then return
  GREEDY_DESCENT,     // single descent until no improving jumps can be made
  TREE,               // tree mode
  ROUNDING,           // FJ as rounding procedure for fractionals
  EXIT_NON_IMPROVING  // iterate until we are don't improve the best
};

enum class fj_load_balancing_mode_t { ALWAYS_ON, AUTO, ALWAYS_OFF };

enum class fj_candidate_selection_t { WEIGHTED_SCORE, FEASIBLE_FIRST };

struct fj_settings_t {
  int seed{0};
  fj_mode_t mode{fj_mode_t::FIRST_FEASIBLE};
  fj_candidate_selection_t candidate_selection{fj_candidate_selection_t::WEIGHTED_SCORE};
  double time_limit{60.0};
  int iteration_limit{std::numeric_limits<int>::max()};
  fj_hyper_parameters_t parameters{};
  int n_of_minimums_for_exit  = 7000;
  double infeasibility_weight = 1.0;
  bool update_weights         = true;
  bool feasibility_run        = true;
  fj_load_balancing_mode_t load_balancing_mode{fj_load_balancing_mode_t::AUTO};
  double baseline_objective_for_longer_run{std::numeric_limits<double>::lowest()};
};

struct fj_move_t {
  int var_idx;
  double value;

  bool operator<(const fj_move_t& rhs) const { return value < rhs.value; }
};

// TODO: use 32bit integers instead,
// as we dont need them to be floating point per the FJ2 scoring scheme
// sizeof(fj_staged_score_t) <= 8 is needed to allow for atomic loads
struct fj_staged_score_t {
  float base{-std::numeric_limits<float>::infinity()};
  float bonus{-std::numeric_limits<float>::infinity()};

  HDI bool operator<(fj_staged_score_t other) const noexcept
  {
    return base == other.base ? bonus < other.bonus : base < other.base;
  }
  HDI bool operator>(fj_staged_score_t other) const noexcept
  {
    return base == other.base ? bonus > other.bonus : base > other.base;
  }
  HDI bool operator==(fj_staged_score_t other) const noexcept
  {
    return base == other.base && bonus == other.bonus;
  }
  HDI bool operator!=(fj_staged_score_t other) const noexcept { return !(*this == other); }

  HDI static fj_staged_score_t invalid()
  {
    return {-std::numeric_limits<float>::infinity(), -std::numeric_limits<float>::infinity()};
  }
  HDI static fj_staged_score_t zero() { return {0, 0}; }

  HDI bool valid() const { return *this != invalid(); }
};

template <typename f_t>
struct fj_move_score_info_base_t {
  fj_staged_score_t score;
  f_t infeasibility;

  DI void invalidate()
  {
    score         = fj_staged_score_t::invalid();
    infeasibility = -std::numeric_limits<double>::infinity();
  }

  static DI fj_move_score_info_base_t<f_t> invalid()
  {
    fj_move_score_info_base_t<f_t> invalid_move;
    invalid_move.invalidate();
    return invalid_move;
  }
};

struct fj_load_balancing_workid_mapping_t {
  uint32_t var_idx;
  uint32_t subworkid;  // index among all workids assigned to this specific var_idx
  // bundle some variable-related data to avoid dependend loads
  uint32_t offset_begin;
  uint32_t offset_end;
};

template <typename f_t>
struct fj_move_candidate_t {
  fj_staged_score_t score;
  f_t delta;
};

template <typename i_t, typename f_t>
class fj_t {
 public:
  using move_score_t      = fj_staged_score_t;
  using move_score_info_t = fj_move_score_info_base_t<f_t>;
  using move_candidate_t  = fj_move_candidate_t<f_t>;

  fj_t(mip_solver_context_t<i_t, f_t>& context, fj_settings_t settings = fj_settings_t{});
  ~fj_t();
  void reset_cuda_graph();
  i_t solve(solution_t<i_t, f_t>& solution);
  i_t alloc_max_climbers(i_t desired_climbers);
  void resize_vectors(const raft::handle_t* handle_ptr);
  void device_init(const rmm::cuda_stream_view& stream);
  void climber_init(i_t climber_idx);
  void climber_init(i_t climber_idx, const rmm::cuda_stream_view& stream);
  void set_fj_settings(fj_settings_t settings_);
  void reset_weights(const rmm::cuda_stream_view& stream, f_t weight = 10.);
  void randomize_weights(const raft::handle_t* handle_ptr);
  void copy_weights(const weight_t<i_t, f_t>& weights, const raft::handle_t* handle_ptr);
  i_t host_loop(solution_t<i_t, f_t>& solution, i_t climber_idx = 0);
  void run_step_device(i_t climber_idx = 0, bool use_graph = true);
  void run_step_device(const rmm::cuda_stream_view& stream,
                       i_t climber_idx = 0,
                       bool use_graph  = true);
  void refresh_lhs_and_violation(const rmm::cuda_stream_view& stream, i_t climber_idx = 0);
  // load balancing
  void load_balancing_score_update(const rmm::cuda_stream_view& stream, i_t climber_idx = 0);
  // executed after a roudning FJ run if any fractionals remain to eliminate them
  void round_remaining_fractionals(solution_t<i_t, f_t>& solution, i_t climber_idx = 0);

 public:
  mip_solver_context_t<i_t, f_t>& context;
  problem_t<i_t, f_t>* pb_ptr;
  raft::handle_t* handle_ptr;

  // cstr weights is the sum of both -1 (as they all start with 1) but to save from the perf we
  // always keep all cstr_weight is accessed multiple times in the code, so single access it better
  rmm::device_uvector<f_t> cstr_weights;
  rmm::device_uvector<f_t> cstr_right_weights;
  rmm::device_uvector<f_t> cstr_left_weights;
  rmm::device_scalar<f_t> max_cstr_weight;
  f_t weight_update_increment;
  rmm::device_scalar<f_t> objective_weight;
  f_t stop_threshold = 0.;
  rmm::device_uvector<i_t> objective_vars;

  // array to directly map a CSR entry index to the corresponding constraint bound
  // to save on an indirect likely-uncoalesced load
  rmm::device_uvector<f_t> constraint_lower_bounds_csr;
  rmm::device_uvector<f_t> constraint_upper_bounds_csr;
  rmm::device_uvector<f_t> cstr_coeff_reciprocal;

  // load balancing structures
  rmm::device_uvector<i_t> row_size_bin_prefix_sum;
  rmm::device_uvector<i_t> row_size_nonbin_prefix_sum;
  rmm::device_uvector<fj_load_balancing_workid_mapping_t> work_id_to_bin_var_idx;
  rmm::device_uvector<fj_load_balancing_workid_mapping_t> work_id_to_nonbin_var_idx;
  rmm::device_uvector<i_t> work_ids_for_related_vars;

  cudaGraphExec_t graph_instance;
  bool graph_created = false;

  // kernel launch dimensions, computed once inside the constructor
  std::pair<dim3, dim3> setval_launch_dims;
  std::pair<dim3, dim3> resetmoves_launch_dims;
  std::pair<dim3, dim3> resetmoves_bin_launch_dims;
  std::pair<dim3, dim3> update_weights_launch_dims;
  std::pair<dim3, dim3> heavy_vars_launch_dims;
  std::pair<dim3, dim3> heavy_cstrs_launch_dims;
  std::pair<dim3, dim3> lift_move_launch_dims;
  std::pair<dim3, dim3> load_balancing_workid_map_launch_dims;
  std::pair<dim3, dim3> load_balancing_binary_launch_dims;
  std::pair<dim3, dim3> load_balancing_mtm_compute_candidates_launch_dims;
  std::pair<dim3, dim3> load_balancing_mtm_compute_scores_launch_dims;
  std::pair<dim3, dim3> load_balancing_prepare_launch_dims;

  i_t load_balancing_variable_count;
  i_t load_balancing_constraint_count;

  // data that is specific to each individual climber running on its own seed
  // boilerplate for now, will become useful when support for parallel descents is added
  struct climber_data_t {
    fj_t& fj;

    rmm::cuda_stream stream;
    rmm::cuda_stream load_balancing_bin_stream;
    rmm::cuda_stream load_balancing_nonbin_stream;
    event_handler_t load_balancing_start_event{cudaEventDisableTiming};
    event_handler_t load_balancing_bin_finished_event{cudaEventDisableTiming};
    event_handler_t load_balancing_nonbin_finished_event{cudaEventDisableTiming};

    rmm::device_scalar<i_t> selected_var;
    rmm::device_scalar<f_t> violation_score;
    rmm::device_scalar<f_t> weighted_violation_score;
    rmm::device_scalar<i_t> constraints_changed_count;
    rmm::device_scalar<i_t> local_minimums_reached;
    rmm::device_scalar<i_t> iterations;
    rmm::device_scalar<f_t> best_excess;
    rmm::device_scalar<f_t> best_objective;
    rmm::device_scalar<f_t> saved_solution_objective;
    rmm::device_scalar<f_t> incumbent_quality;
    rmm::device_scalar<f_t> incumbent_objective;
    rmm::device_scalar<i_t> last_minimum_iteration;
    rmm::device_scalar<i_t> last_improving_minimum;
    rmm::device_scalar<i_t> last_iter_candidates;
    rmm::device_scalar<i_t> break_condition;
    // this is needed to prevent race condition in update_weights_kernel
    rmm::device_scalar<i_t> temp_break_condition;
    rmm::device_scalar<cub::KeyValuePair<i_t, f_t>> best_jump_idx;
    // used with FEASIBLE_FIRST in order to allow infeasible moves for a certain number of
    // iterations
    rmm::device_scalar<i_t> iterations_until_feasible_counter;
    rmm::device_scalar<i_t> full_refresh_iteration;
    rmm::device_scalar<i_t> relvar_count_last_update;
    rmm::device_scalar<i_t> load_balancing_skip;

    contiguous_set_t<i_t, f_t> violated_constraints;
    contiguous_set_t<i_t, f_t> candidate_variables;
    bitmap_t<uint32_t> iteration_related_variables;
    rmm::device_uvector<i_t> constraints_changed;
    rmm::device_uvector<f_t> best_assignment;
    rmm::device_uvector<f_t> incumbent_assignment;
    rmm::device_uvector<f_t> incumbent_lhs;
    // compensation term of the Kahan summation algorithm
    rmm::device_uvector<f_t> incumbent_lhs_sumcomp;
    rmm::device_uvector<i_t> move_last_update;
    rmm::device_uvector<f_t> move_delta;
    rmm::device_uvector<move_score_t> move_score;
    rmm::device_uvector<move_score_t> jump_move_scores;
    rmm::device_uvector<f_t> jump_move_delta;
    rmm::device_uvector<f_t> jump_move_delta_check;
    rmm::device_uvector<move_score_t> jump_move_score_check;
    rmm::device_uvector<f_t> jump_move_infeasibility;
    rmm::device_uvector<i_t> tabu_nodec_until;
    rmm::device_uvector<i_t> tabu_noinc_until;
    rmm::device_uvector<i_t> tabu_lastdec;
    rmm::device_uvector<i_t> tabu_lastinc;
    rmm::device_uvector<move_candidate_t> jump_candidates;
    rmm::device_uvector<i_t> jump_candidate_count;
    rmm::device_uvector<i_t> jump_locks;
    rmm::device_scalar<i_t> small_move_tabu;

    // ROUNDING mode related members
    contiguous_set_t<i_t, f_t> fractional_variables;
    rmm::device_scalar<i_t> saved_best_fractional_count;
    rmm::device_scalar<bool> handle_fractionals_only;

    rmm::device_uvector<move_score_t> grid_score_buf;
    rmm::device_uvector<i_t> grid_var_buf;
    rmm::device_uvector<f_t> grid_delta_buf;

    rmm::device_uvector<i_t> candidate_arrived_workids;

    rmm::device_uvector<std::byte> cub_storage_bytes;
    rmm::device_uvector<f_t> dot_product_buffer;

    climber_data_t(fj_t& in_fj)
      : fj(in_fj),
        selected_var(std::numeric_limits<i_t>::max(), fj.handle_ptr->get_stream()),
        violation_score(0, fj.handle_ptr->get_stream()),
        weighted_violation_score(0, fj.handle_ptr->get_stream()),
        constraints_changed_count(0, fj.handle_ptr->get_stream()),
        local_minimums_reached(0, fj.handle_ptr->get_stream()),
        iterations(0, fj.handle_ptr->get_stream()),
        best_excess(-std::numeric_limits<f_t>::infinity(), fj.handle_ptr->get_stream()),
        best_objective(+std::numeric_limits<f_t>::infinity(), fj.handle_ptr->get_stream()),
        saved_solution_objective(+std::numeric_limits<f_t>::infinity(),
                                 fj.handle_ptr->get_stream()),
        incumbent_quality(+std::numeric_limits<f_t>::infinity(), fj.handle_ptr->get_stream()),
        incumbent_objective(0.0, fj.handle_ptr->get_stream()),
        iterations_until_feasible_counter(0, fj.handle_ptr->get_stream()),
        full_refresh_iteration(0, fj.handle_ptr->get_stream()),
        best_jump_idx(cub::KeyValuePair<i_t, f_t>{}, fj.handle_ptr->get_stream()),
        violated_constraints(fj.pb_ptr->n_constraints, fj.handle_ptr->get_stream()),
        candidate_variables(fj.pb_ptr->n_variables, fj.handle_ptr->get_stream()),
        iteration_related_variables(fj.pb_ptr->n_variables, fj.handle_ptr->get_stream()),
        constraints_changed(fj.pb_ptr->n_constraints, fj.handle_ptr->get_stream()),
        best_assignment(fj.pb_ptr->n_variables, fj.handle_ptr->get_stream()),
        tabu_nodec_until(fj.pb_ptr->n_variables, fj.handle_ptr->get_stream()),
        tabu_noinc_until(fj.pb_ptr->n_variables, fj.handle_ptr->get_stream()),
        tabu_lastdec(fj.pb_ptr->n_variables, fj.handle_ptr->get_stream()),
        tabu_lastinc(fj.pb_ptr->n_variables, fj.handle_ptr->get_stream()),
        incumbent_assignment(fj.pb_ptr->n_variables, fj.handle_ptr->get_stream()),
        incumbent_lhs(fj.pb_ptr->n_constraints, fj.handle_ptr->get_stream()),
        incumbent_lhs_sumcomp(fj.pb_ptr->n_constraints, fj.handle_ptr->get_stream()),
        jump_move_scores(fj.pb_ptr->n_variables, fj.handle_ptr->get_stream()),
        jump_move_infeasibility(fj.pb_ptr->n_variables, fj.handle_ptr->get_stream()),
        jump_move_delta(fj.pb_ptr->n_variables, fj.handle_ptr->get_stream()),
        jump_move_delta_check(fj.pb_ptr->n_variables, fj.handle_ptr->get_stream()),
        jump_move_score_check(fj.pb_ptr->n_variables, fj.handle_ptr->get_stream()),
        move_last_update(fj.pb_ptr->n_variables * FJ_MOVE_SIZE, fj.handle_ptr->get_stream()),
        move_delta(fj.pb_ptr->n_variables * FJ_MOVE_SIZE, fj.handle_ptr->get_stream()),
        move_score(fj.pb_ptr->n_variables * FJ_MOVE_SIZE, fj.handle_ptr->get_stream()),
        jump_candidates(fj.pb_ptr->coefficients.size(), fj.handle_ptr->get_stream()),
        jump_candidate_count(fj.pb_ptr->n_variables, fj.handle_ptr->get_stream()),
        jump_locks(fj.pb_ptr->n_variables, fj.handle_ptr->get_stream()),
        fractional_variables(fj.pb_ptr->n_variables, fj.handle_ptr->get_stream()),
        small_move_tabu(0, fj.handle_ptr->get_stream()),
        handle_fractionals_only(false, fj.handle_ptr->get_stream()),
        saved_best_fractional_count(0, fj.handle_ptr->get_stream()),
        candidate_arrived_workids(fj.pb_ptr->coefficients.size(), fj.handle_ptr->get_stream()),
        grid_score_buf(0, fj.handle_ptr->get_stream()),
        grid_var_buf(0, fj.handle_ptr->get_stream()),
        grid_delta_buf(0, fj.handle_ptr->get_stream()),
        last_minimum_iteration(0, fj.handle_ptr->get_stream()),
        last_improving_minimum(0, fj.handle_ptr->get_stream()),
        last_iter_candidates(0, fj.handle_ptr->get_stream()),
        relvar_count_last_update(0, fj.handle_ptr->get_stream()),
        load_balancing_skip(0, fj.handle_ptr->get_stream()),
        break_condition(0, fj.handle_ptr->get_stream()),
        temp_break_condition(0, fj.handle_ptr->get_stream()),
        cub_storage_bytes(0, fj.handle_ptr->get_stream()),
        dot_product_buffer(fj.pb_ptr->n_variables, fj.handle_ptr->get_stream())
    {
      // Allocate space for the objective dot product reduction
      size_t temp_storage_bytes = 0;
      cub::DeviceReduce::Sum(nullptr,
                             temp_storage_bytes,
                             dot_product_buffer.data(),
                             incumbent_objective.data(),
                             fj.pb_ptr->n_variables,
                             fj.handle_ptr->get_stream());

      // Allocate temporary storage
      cub_storage_bytes.resize(temp_storage_bytes, fj.handle_ptr->get_stream());
    }

    struct view_t {
      typename problem_t<i_t, f_t>::view_t pb;

      raft::device_span<i_t> constraints_changed;
      raft::device_span<f_t> cstr_weights;
      raft::device_span<f_t> cstr_right_weights;
      raft::device_span<f_t> cstr_left_weights;
      raft::device_span<f_t> incumbent_assignment;
      raft::device_span<f_t> best_assignment;
      raft::device_span<f_t> incumbent_lhs;
      raft::device_span<f_t> incumbent_lhs_sumcomp;
      raft::device_span<move_score_t> jump_move_scores;
      raft::device_span<f_t> jump_move_infeasibility;
      raft::device_span<f_t> jump_move_delta_check;
      raft::device_span<move_score_t> jump_move_score_check;
      raft::device_span<f_t> jump_move_delta;
      raft::device_mdspan<i_t, raft::extents<i_t, FJ_MOVE_SIZE, raft::dynamic_extent>>
        move_last_update;
      raft::device_mdspan<f_t, raft::extents<i_t, FJ_MOVE_SIZE, raft::dynamic_extent>> move_delta;
      raft::device_mdspan<move_score_t, raft::extents<i_t, FJ_MOVE_SIZE, raft::dynamic_extent>>
        move_score;
      raft::device_span<i_t> tabu_nodec_until;
      raft::device_span<i_t> tabu_noinc_until;
      raft::device_span<i_t> tabu_lastinc;
      raft::device_span<i_t> tabu_lastdec;
      raft::device_span<move_score_t> grid_score_buf;
      raft::device_span<i_t> grid_var_buf;
      raft::device_span<f_t> grid_delta_buf;
      raft::device_span<i_t> objective_vars;
      raft::device_span<move_candidate_t> jump_candidates;
      raft::device_span<i_t> jump_candidate_count;
      raft::device_span<i_t> jump_locks;
      raft::device_span<i_t> work_ids_for_related_vars;
      raft::device_span<i_t> candidate_arrived_workids;

      raft::device_span<f_t> cstr_coeff_reciprocal;
      raft::device_span<f_t> constraint_lower_bounds_csr;
      raft::device_span<f_t> constraint_upper_bounds_csr;

      typename contiguous_set_t<i_t, f_t>::view_t violated_constraints;
      typename contiguous_set_t<i_t, f_t>::view_t candidate_variables;
      typename bitmap_t<uint32_t>::view_t iteration_related_variables;

      // ROUNDING mode related members
      typename contiguous_set_t<i_t, f_t>::view_t fractional_variables;
      i_t* saved_best_fractional_count;
      bool* handle_fractionals_only;
      // load balancing structures
      raft::device_span<i_t> row_size_bin_prefix_sum;
      raft::device_span<i_t> row_size_nonbin_prefix_sum;
      raft::device_span<fj_load_balancing_workid_mapping_t> work_id_to_bin_var_idx;
      raft::device_span<fj_load_balancing_workid_mapping_t> work_id_to_nonbin_var_idx;

      i_t* selected_var;
      i_t* constraints_changed_count;
      f_t* violation_score;
      f_t* weighted_violation_score;
      i_t* local_minimums_reached;
      i_t* iterations;
      i_t* last_iter_candidates;
      f_t* best_excess;
      f_t* best_objective;
      f_t* saved_solution_objective;
      f_t* incumbent_quality;
      f_t* incumbent_objective;
      f_t weight_update_increment;
      f_t* objective_weight;
      i_t* iterations_until_feasible_counter;
      i_t* full_refresh_iteration;
      cub::KeyValuePair<i_t, f_t>* best_jump_idx;
      f_t stop_threshold;
      i_t* small_move_tabu;

      i_t* last_minimum_iteration;
      i_t* last_improving_minimum;
      i_t* break_condition;
      i_t* temp_break_condition;
      i_t* relvar_count_last_update;
      i_t* load_balancing_skip;
      f_t* max_cstr_weight;

      fj_settings_t* settings;

      DI f_t lower_excess_score(i_t cstr, f_t lhs) const
      {
        return raft::min(lhs - pb.constraint_lower_bounds[cstr], (f_t)0);
      }

      DI f_t upper_excess_score(i_t cstr, f_t lhs) const
      {
        return raft::min(pb.constraint_upper_bounds[cstr] - lhs, (f_t)0);
      }

      // Computes the constraint's contribution to the feasibility score:
      // If the constraint is satisfied by the given LHS value, returns 0.
      // If the constraint is violated by the given LHS value, returns -|lhs-rhs|.
      // caution: is inverted compared to solution_t's excess convention
      DI f_t excess_score(i_t cstr, f_t lhs) const
      {
        f_t right_score = upper_excess_score(cstr, lhs);
        if (right_score < 0.) { return right_score; }
        return lower_excess_score(cstr, lhs);
      }

      // FJ relies on maintaining a running LHS value for each constraint
      // which may suffer from numerical errors and lead to very slight (~machine epsilon)
      // violations of the actual bounds.
      // Use a slightly tightened tolerance in FJ to account for this.
      DI f_t get_corrected_tolerance(i_t cstr) const
      {
        f_t cstr_tolerance = get_cstr_tolerance<i_t, f_t>(pb.constraint_lower_bounds[cstr],
                                                          pb.constraint_upper_bounds[cstr],
                                                          pb.tolerances.absolute_tolerance,
                                                          pb.tolerances.relative_tolerance);
        return max((f_t)0, cstr_tolerance - MACHINE_EPSILON);
      }

      DI bool cstr_satisfied(i_t cstr, f_t lhs) const
      {
        f_t cstr_tolerance = get_cstr_tolerance<i_t, f_t>(pb.constraint_lower_bounds[cstr],
                                                          pb.constraint_upper_bounds[cstr],
                                                          pb.tolerances.absolute_tolerance,
                                                          pb.tolerances.relative_tolerance);
        return excess_score(cstr, lhs) >= -cstr_tolerance;
      }

      DI bool move_numerically_stable(f_t old_val, f_t new_val, f_t infeasibility) const
      {
        return fabs(new_val - old_val) < 1e6 && fabs(new_val) < 1e20 &&
               fabs(*violation_score - infeasibility) < 1e20;
      }

      DI bool admits_move(i_t var_idx) const
      {
        f_t delta = jump_move_delta[var_idx];
        f_t iter  = *iterations;

        // affected by tabu
        if ((delta < 0 && iter < tabu_nodec_until[var_idx]) ||
            (delta > 0 && iter < tabu_noinc_until[var_idx]))
          return false;

        if (*handle_fractionals_only && !fractional_variables.contains(var_idx)) return false;

        // give priority to MTM moves.
        if (jump_move_scores[var_idx].base > 0) return true;

#pragma unroll
        for (i_t move_type = FJ_MOVE_BEGIN; move_type < FJ_MOVE_SIZE; ++move_type) {
          if (move_type == FJ_MOVE_LIFT && violated_constraints.size() > 0) continue;
          if (move_type == FJ_MOVE_BREAKTHROUGH &&
              (*best_objective == std::numeric_limits<f_t>::infinity() ||
               *incumbent_objective <
                 *best_objective + 1e-6 - settings->parameters.breakthrough_move_epsilon))
            continue;

          // only select moves that were updated during this iteration
          if (move_last_update(move_type, var_idx) == iter &&
              move_score(move_type, var_idx).base > 0) {
            if (move_type == FJ_MOVE_LIFT) {
              f_t obj_delta = pb.objective_coefficients[var_idx] * move_delta(move_type, var_idx);
              cuopt_assert(obj_delta < 0, "lift move should not worsen objective");
            }

            jump_move_scores[var_idx] = move_score(move_type, var_idx);
            jump_move_delta[var_idx]  = move_delta(move_type, var_idx);
            return true;
          }
        }

        return false;
      }
    };

    view_t view();
    void clear_sets(const rmm::cuda_stream_view& stream);
  };
  void populate_climber_views();

  std::vector<std::unique_ptr<climber_data_t>> climbers;
  rmm::device_uvector<typename climber_data_t::view_t> climber_views;
  fj_settings_t settings;
};

}  // namespace cuopt::linear_programming::detail
