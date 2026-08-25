/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

// Device-facing members of solver_settings_t, split out of solver_settings.cu.
//
// Everything else in that class is host-only parameter handling, so the remainder now
// builds as solver_settings.cpp into the CUDA-free cuopt_client library. Only these
// members take an rmm::cuda_stream_view or hand back a device_uvector, so they are the
// only ones that must stay in a CUDA TU inside libcuopt.
//
// solver_settings.cpp deliberately has no `template class` at all -- that would instantiate
// the constructor, which needs CUDA (see the note there) -- so it cannot emit these members
// either. The `template class` below covers the class as a whole for libcuopt; members
// defined in the CUDA-free TU are instantiated individually there.

#include <cuopt/mathematical_optimization/solver_settings.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_uvector.hpp>

#include <mip_heuristics/mip_constants.hpp>

namespace cuopt {
namespace CUOPT_EXPORT mathematical_optimization {

template <typename i_t, typename f_t>
void solver_settings_t<i_t, f_t>::set_initial_pdlp_primal_solution(const f_t* solution,
                                                                   i_t size,
                                                                   rmm::cuda_stream_view stream)
{
  pdlp_settings.set_initial_primal_solution(solution, size, stream);
}

template <typename i_t, typename f_t>
void solver_settings_t<i_t, f_t>::set_initial_pdlp_dual_solution(const f_t* solution,
                                                                 i_t size,
                                                                 rmm::cuda_stream_view stream)
{
  pdlp_settings.set_initial_dual_solution(solution, size, stream);
}

template <typename i_t, typename f_t>
void solver_settings_t<i_t, f_t>::set_pdlp_warm_start_data(
  const f_t* current_primal_solution,
  const f_t* current_dual_solution,
  const f_t* initial_primal_average,
  const f_t* initial_dual_average,
  const f_t* current_ATY,
  const f_t* sum_primal_solutions,
  const f_t* sum_dual_solutions,
  const f_t* last_restart_duality_gap_primal_solution,
  const f_t* last_restart_duality_gap_dual_solution,
  i_t primal_size,
  i_t dual_size,
  f_t initial_primal_weight,
  f_t initial_step_size,
  i_t total_pdlp_iterations,
  i_t total_pdhg_iterations,
  f_t last_candidate_kkt_score,
  f_t last_restart_kkt_score,
  f_t sum_solution_weight,
  i_t iterations_since_last_restart)
{
  pdlp_settings.set_pdlp_warm_start_data(current_primal_solution,
                                         current_dual_solution,
                                         initial_primal_average,
                                         initial_dual_average,
                                         current_ATY,
                                         sum_primal_solutions,
                                         sum_dual_solutions,
                                         last_restart_duality_gap_primal_solution,
                                         last_restart_duality_gap_dual_solution,
                                         primal_size,
                                         dual_size,
                                         initial_primal_weight,
                                         initial_step_size,
                                         total_pdlp_iterations,
                                         total_pdhg_iterations,
                                         last_candidate_kkt_score,
                                         last_restart_kkt_score,
                                         sum_solution_weight,
                                         iterations_since_last_restart);
}

template <typename i_t, typename f_t>
const rmm::device_uvector<f_t>& solver_settings_t<i_t, f_t>::get_initial_pdlp_primal_solution()
  const
{
  return pdlp_settings.get_initial_primal_solution();
}

template <typename i_t, typename f_t>
const rmm::device_uvector<f_t>& solver_settings_t<i_t, f_t>::get_initial_pdlp_dual_solution() const
{
  return pdlp_settings.get_initial_dual_solution();
}

template <typename i_t, typename f_t>
void solver_settings_t<i_t, f_t>::add_initial_mip_solution(const f_t* solution,
                                                           i_t size,
                                                           rmm::cuda_stream_view stream)
{
  mip_settings.add_initial_solution(solution, size, stream);
}

// The constructor lives here, not in solver_settings.cpp, even though its body is pure
// host code building the parameter tables. It default-constructs pdlp_settings, which holds
// a pdlp_warm_start_data_t by value whose ctor is CUDA-side -- so defining it in the
// CUDA-free library would leave libcuopt_client.so with an undefined symbol that only
// surfaces at call time. Moving it here is what lets that library resolve standalone.
template <typename i_t, typename f_t>
solver_settings_t<i_t, f_t>::solver_settings_t() : pdlp_settings(), mip_settings()
{
  // clang-format off
  // Float parameters
  float_parameters = {
    {CUOPT_TIME_LIMIT, &mip_settings.time_limit, f_t(0.0), std::numeric_limits<f_t>::infinity(), std::numeric_limits<f_t>::infinity()},
    {CUOPT_TIME_LIMIT, &pdlp_settings.time_limit, f_t(0.0), std::numeric_limits<f_t>::infinity(), std::numeric_limits<f_t>::infinity()},
    {CUOPT_WORK_LIMIT, &mip_settings.work_limit, f_t(0.0), std::numeric_limits<f_t>::infinity(), std::numeric_limits<f_t>::infinity()},
    {CUOPT_ABSOLUTE_DUAL_TOLERANCE, &pdlp_settings.tolerances.absolute_dual_tolerance, f_t(0.0), f_t(1e-1), f_t(1e-4)},
    {CUOPT_RELATIVE_DUAL_TOLERANCE, &pdlp_settings.tolerances.relative_dual_tolerance, f_t(0.0), f_t(1e-1), f_t(1e-4)},
    {CUOPT_ABSOLUTE_PRIMAL_TOLERANCE, &pdlp_settings.tolerances.absolute_primal_tolerance, f_t(0.0), f_t(1e-1), f_t(1e-4)},
    {CUOPT_RELATIVE_PRIMAL_TOLERANCE, &pdlp_settings.tolerances.relative_primal_tolerance, f_t(0.0), f_t(1e-1), f_t(1e-4)},
    {CUOPT_ABSOLUTE_GAP_TOLERANCE, &pdlp_settings.tolerances.absolute_gap_tolerance, f_t(0.0), f_t(1e-1), f_t(1e-4)},
    {CUOPT_RELATIVE_GAP_TOLERANCE, &pdlp_settings.tolerances.relative_gap_tolerance, f_t(0.0), f_t(1e-1), f_t(1e-4)},
    {CUOPT_MIP_ABSOLUTE_TOLERANCE, &mip_settings.tolerances.absolute_tolerance, f_t(0.0), f_t(1e-1), f_t(1e-6)},
    {CUOPT_MIP_RELATIVE_TOLERANCE, &mip_settings.tolerances.relative_tolerance, f_t(0.0), f_t(1e-1), f_t(1e-12)},
    {CUOPT_MIP_INTEGRALITY_TOLERANCE, &mip_settings.tolerances.integrality_tolerance, f_t(0.0), f_t(1e-1), f_t(1e-5)},
    {CUOPT_MIP_ABSOLUTE_GAP, &mip_settings.tolerances.absolute_mip_gap, f_t(0.0), std::numeric_limits<f_t>::infinity(), std::max(f_t(1e-10), std::numeric_limits<f_t>::epsilon())},
    {CUOPT_MIP_RELATIVE_GAP, &mip_settings.tolerances.relative_mip_gap, f_t(0.0), f_t(1e-1), f_t(1e-4)},
    {CUOPT_PRIMAL_INFEASIBLE_TOLERANCE, &pdlp_settings.tolerances.primal_infeasible_tolerance, f_t(0.0), f_t(1e-1), std::max(f_t(1e-10), std::numeric_limits<f_t>::epsilon())},
    {CUOPT_DUAL_INFEASIBLE_TOLERANCE, &pdlp_settings.tolerances.dual_infeasible_tolerance, f_t(0.0), f_t(1e-1), std::max(f_t(1e-10), std::numeric_limits<f_t>::epsilon())},
    {CUOPT_MIP_CUT_CHANGE_THRESHOLD, &mip_settings.cut_change_threshold, f_t(-1.0), std::numeric_limits<f_t>::infinity(), f_t(-1.0)},
    {CUOPT_MIP_CUT_MIN_ORTHOGONALITY, &mip_settings.cut_min_orthogonality, f_t(0.0), f_t(1.0), f_t(0.5)},
    {CUOPT_BARRIER_STEP_SCALE, &pdlp_settings.barrier_step_scale, f_t(0.5), f_t(0.9999), f_t(0.9)},
    // MIP heuristic hyper-parameters (hidden from default --help: name contains "hyper_")
    {CUOPT_MIP_HYPER_HEURISTIC_ROOT_LP_TIME_RATIO, &mip_settings.heuristic_params.root_lp_time_ratio, f_t(0.0), f_t(1.0), f_t(0.1), "fraction of total time for root LP"},
    {CUOPT_MIP_HYPER_HEURISTIC_ROOT_LP_MAX_TIME, &mip_settings.heuristic_params.root_lp_max_time, f_t(0.0), std::numeric_limits<f_t>::infinity(), f_t(15.0), "hard cap on root LP seconds"},
    {CUOPT_MIP_HYPER_HEURISTIC_RINS_TIME_LIMIT, &mip_settings.heuristic_params.rins_time_limit, f_t(0.0), std::numeric_limits<f_t>::infinity(), f_t(3.0), "per-call RINS sub-MIP time"},
    {CUOPT_MIP_HYPER_HEURISTIC_RINS_MAX_TIME_LIMIT, &mip_settings.heuristic_params.rins_max_time_limit, f_t(0.0), std::numeric_limits<f_t>::infinity(), f_t(20.0), "ceiling for RINS adaptive time budget"},
    {CUOPT_MIP_HYPER_HEURISTIC_RINS_FIX_RATE, &mip_settings.heuristic_params.rins_fix_rate, f_t(0.0), f_t(1.0), f_t(0.5), "RINS variable fix rate"},
    {CUOPT_MIP_HYPER_HEURISTIC_INITIAL_INFEASIBILITY_WEIGHT, &mip_settings.heuristic_params.initial_infeasibility_weight, f_t(1e-9), std::numeric_limits<f_t>::infinity(), f_t(1000.0), "constraint violation penalty seed"},
    {CUOPT_MIP_HYPER_HEURISTIC_RELAXED_LP_TIME_LIMIT, &mip_settings.heuristic_params.relaxed_lp_time_limit, f_t(1e-9), std::numeric_limits<f_t>::infinity(), f_t(1.0), "base relaxed LP time cap in heuristics"},
    {CUOPT_MIP_HYPER_HEURISTIC_RELATED_VARS_TIME_LIMIT, &mip_settings.heuristic_params.related_vars_time_limit, f_t(1e-9), std::numeric_limits<f_t>::infinity(), f_t(30.0), "time for related-variable structure build"},
    {CUOPT_MIP_SEMICONTINUOUS_BIG_M, &mip_settings.semi_continuous_big_m, f_t(1.0), std::numeric_limits<f_t>::infinity(), f_t(1e10), "big-M value for semi-continuous variables with no finite upper bound"},
    // Diving heuristic hyper-parameters (hidden from default --help: name contains "hyper_")
    {CUOPT_MIP_HYPER_DIVING_ITERATION_LIMIT_FACTOR, &mip_settings.diving_params.iteration_limit_factor, f_t(0.0), f_t(1.0), f_t(0.05), "fraction of best-first iterations allowed per dive"},
    // Recursive sub-MIP (RINS) hyper-parameters (hidden from default --help: name contains "hyper_")
    {CUOPT_MIP_HYPER_SUBMIP_BASE_TARGET_FIXRATE, &mip_settings.submip_params.base_target_fixrate, f_t(0.0), f_t(1.0), f_t(0.6), "base target fix rate for the RINS neighbourhood"},
    {CUOPT_MIP_HYPER_SUBMIP_MIN_FIXRATE, &mip_settings.submip_params.min_fixrate, f_t(0.0), f_t(1.0), f_t(0.25), "minimum fix rate for accepting the RINS neighbourhood"},
    {CUOPT_MIP_HYPER_SUBMIP_MIN_FIXRATE_CAP, &mip_settings.submip_params.min_fixrate_cap, f_t(0.0), f_t(1.0), f_t(0.1), "hard cap on the minimum fix rate for solving a sub-MIP"},
    {CUOPT_MIP_HYPER_SUBMIP_TARGET_MIP_GAP, &mip_settings.submip_params.target_mip_gap, f_t(0.0), f_t(1.0), f_t(0.01), "MIP gap target for the sub-MIP"},
    {CUOPT_MIP_HYPER_SUBMIP_ITERATION_LIMIT_RATIO, &mip_settings.submip_params.iteration_limit_ratio, f_t(0.0), f_t(1.0), f_t(0.8), "sub-MIP simplex-iteration limit as a factor of parent B&B iterations"},
    {CUOPT_MIP_HYPER_SUBMIP_ROUND_CLOSE_RATIO, &mip_settings.submip_params.round_close_ratio, f_t(0.0), f_t(1.0), f_t(0.8), "share of the still-unfixed integers left for later neighbourhood rounds (0 reaches the target fix rate in a single round)"},
   };

  // Int parameters
  // TODO should we have Stable2 and Methodolical1 here?
  int_parameters = {
    {CUOPT_ITERATION_LIMIT, &pdlp_settings.iteration_limit, 0, std::numeric_limits<i_t>::max(), std::numeric_limits<i_t>::max()},
    {CUOPT_NODE_LIMIT, &mip_settings.node_limit, 0, std::numeric_limits<i_t>::max(), std::numeric_limits<i_t>::max()},
    {CUOPT_PDLP_SOLVER_MODE, reinterpret_cast<int*>(&pdlp_settings.pdlp_solver_mode), CUOPT_PDLP_SOLVER_MODE_STABLE1, CUOPT_PDLP_SOLVER_MODE_STABLE3, CUOPT_PDLP_SOLVER_MODE_STABLE3},
    {CUOPT_METHOD, reinterpret_cast<int*>(&pdlp_settings.method), CUOPT_METHOD_CONCURRENT, CUOPT_METHOD_BARRIER, CUOPT_METHOD_CONCURRENT},
    {CUOPT_NUM_CPU_THREADS, &mip_settings.num_cpu_threads, -1, std::numeric_limits<i_t>::max(), -1},
    {CUOPT_AUGMENTED, &pdlp_settings.augmented, -1, 1, -1},
    {CUOPT_FOLDING, &pdlp_settings.folding, -1, 1, -1},
    {CUOPT_DUALIZE, &pdlp_settings.dualize, -1, 1, -1},
    {CUOPT_ORDERING, &pdlp_settings.ordering, -1, 1, -1},
    {CUOPT_BARRIER_DUAL_INITIAL_POINT, &pdlp_settings.barrier_dual_initial_point, -1, 1, -1},
    {CUOPT_POSTSOLVE_INFO, &pdlp_settings.postsolve_info, -1, 1, -1},
    {CUOPT_MIP_CUT_PASSES, &mip_settings.max_cut_passes, -1, std::numeric_limits<i_t>::max(), 10},
    {CUOPT_MIP_MIXED_INTEGER_ROUNDING_CUTS, &mip_settings.mir_cuts, -1, 1, -1},
    {CUOPT_MIP_MIXED_INTEGER_GOMORY_CUTS, &mip_settings.mixed_integer_gomory_cuts, -1, 1, -1},
    {CUOPT_MIP_KNAPSACK_CUTS, &mip_settings.knapsack_cuts, -1, 1, -1},
    {CUOPT_MIP_FLOW_COVER_CUTS, &mip_settings.flow_cover_cuts, -1, 1, -1},
    {CUOPT_MIP_CLIQUE_CUTS, &mip_settings.clique_cuts, -1, 1, -1},
    {CUOPT_MIP_ZERO_HALF_CUTS, &mip_settings.zero_half_cuts, -1, 1, -1},
    {CUOPT_MIP_IMPLIED_BOUND_CUTS, &mip_settings.implied_bound_cuts, -1, 1, -1},
    {CUOPT_MIP_STRONG_CHVATAL_GOMORY_CUTS, &mip_settings.strong_chvatal_gomory_cuts, -1, 1, -1},
    {CUOPT_MIP_REDUCED_COST_STRENGTHENING, &mip_settings.reduced_cost_strengthening, -1, std::numeric_limits<i_t>::max(), -1},
    {CUOPT_MIP_RINS, &mip_settings.submip_params.rins, -1, 1, -1},
    {CUOPT_MIP_RENS, &mip_settings.submip_params.rens, -1, 1, -1},
    {CUOPT_MIP_OBJECTIVE_STEP, &mip_settings.objective_step, 0, 1, 1},
    {CUOPT_NUM_GPUS, &pdlp_settings.num_gpus, -1, 72, 1},
    {CUOPT_NUM_GPUS, &mip_settings.num_gpus, -1, 72, 1},
    {CUOPT_MIP_BATCH_PDLP_STRONG_BRANCHING, &mip_settings.mip_batch_pdlp_strong_branching, 0, 2, 0},
    {CUOPT_MIP_BATCH_PDLP_RELIABILITY_BRANCHING, &mip_settings.mip_batch_pdlp_reliability_branching, 0, 2, 0},
    {CUOPT_MIP_STRONG_BRANCHING_SIMPLEX_ITERATION_LIMIT, &mip_settings.strong_branching_simplex_iteration_limit, -1,std::numeric_limits<i_t>::max(), -1},
    {CUOPT_PRESOLVE, reinterpret_cast<int*>(&pdlp_settings.presolver), CUOPT_PRESOLVE_DEFAULT, CUOPT_PRESOLVE_PSLP, CUOPT_PRESOLVE_DEFAULT},
    {CUOPT_PRESOLVE, reinterpret_cast<int*>(&mip_settings.presolver), CUOPT_PRESOLVE_DEFAULT, CUOPT_PRESOLVE_PSLP, CUOPT_PRESOLVE_DEFAULT},
    {CUOPT_DISTRIBUTED_PDLP_PARTITIONER, reinterpret_cast<int*>(&pdlp_settings.distributed_pdlp_partitioner), CUOPT_DISTRIBUTED_PDLP_PARTITIONER_AUTO, CUOPT_DISTRIBUTED_PDLP_PARTITIONER_ROUND_ROBIN, CUOPT_DISTRIBUTED_PDLP_PARTITIONER_AUTO},
    {CUOPT_MIP_DETERMINISM_MODE, &mip_settings.determinism_mode, CUOPT_MODE_OPPORTUNISTIC, CUOPT_MODE_DETERMINISTIC, CUOPT_MODE_OPPORTUNISTIC},
    {CUOPT_RANDOM_SEED, &mip_settings.seed, -1, std::numeric_limits<i_t>::max(), -1},
    {CUOPT_MIP_RELIABILITY_BRANCHING, &mip_settings.reliability_branching, -1, std::numeric_limits<i_t>::max(), -1},
    {CUOPT_PDLP_PRECISION, reinterpret_cast<int*>(&pdlp_settings.pdlp_precision), CUOPT_PDLP_DEFAULT_PRECISION, CUOPT_PDLP_MIXED_PRECISION, CUOPT_PDLP_DEFAULT_PRECISION},
    {CUOPT_MIP_SYMMETRY, &mip_settings.symmetry, -1, 2, -1},
    {CUOPT_MIP_SCALING, &mip_settings.mip_scaling, CUOPT_MIP_SCALING_OFF, CUOPT_MIP_SCALING_NO_OBJECTIVE, CUOPT_MIP_SCALING_NO_OBJECTIVE},
    // MIP heuristic hyper-parameters (hidden from default --help: name contains "hyper_")
    {CUOPT_MIP_HYPER_HEURISTIC_POPULATION_SIZE, &mip_settings.heuristic_params.population_size, 1, std::numeric_limits<i_t>::max(), 32, "max solutions in pool"},
    {CUOPT_MIP_HYPER_HEURISTIC_NUM_CPUFJ_THREADS, &mip_settings.heuristic_params.num_cpufj_threads, 0, std::numeric_limits<i_t>::max(), 8, "parallel CPU FJ climbers"},
    {CUOPT_MIP_HYPER_HEURISTIC_PRESOLVE_MAX_ROUNDS, &mip_settings.heuristic_params.presolve_max_rounds, -1, std::numeric_limits<i_t>::max(), -1, "Papilo presolve rounds cap (<0 derives it from the problem, 0 keeps Papilo default)"},
    {CUOPT_MIP_HYPER_HEURISTIC_PAPILO_PROBING_MAX_BADGESIZE, &mip_settings.heuristic_params.papilo_probing_max_badgesize, -1, std::numeric_limits<i_t>::max(), -1, "ceiling on Papilo probing.minbadgesize (<0 derives it from the problem, 0 leaves it uncapped)"},
    {CUOPT_MIP_HYPER_HEURISTIC_STAGNATION_TRIGGER, &mip_settings.heuristic_params.stagnation_trigger, 1, std::numeric_limits<i_t>::max(), 3, "FP loops w/o improvement before recombination"},
    {CUOPT_MIP_HYPER_HEURISTIC_MAX_ITERS_WITHOUT_IMPROVEMENT, &mip_settings.heuristic_params.max_iterations_without_improvement, 1, std::numeric_limits<i_t>::max(), 8, "diversity step depth after stagnation"},
    {CUOPT_MIP_HYPER_HEURISTIC_N_OF_MINIMUMS_FOR_EXIT, &mip_settings.heuristic_params.n_of_minimums_for_exit, 1, std::numeric_limits<i_t>::max(), 7000, "FJ baseline local-minima exit threshold"},
    {CUOPT_MIP_HYPER_HEURISTIC_ENABLED_RECOMBINERS, &mip_settings.heuristic_params.enabled_recombiners, 0, 15, 15, "bitmask: 1=BP 2=FP 4=LS 8=SubMIP"},
    {CUOPT_MIP_HYPER_HEURISTIC_CYCLE_DETECTION_LENGTH, &mip_settings.heuristic_params.cycle_detection_length, 1, std::numeric_limits<i_t>::max(), 30, "FP assignment cycle ring buffer length"},
    // Diving heuristic hyper-parameters (hidden from default --help: name contains "hyper_")
    {CUOPT_MIP_HYPER_DIVING_LINE_SEARCH, &mip_settings.diving_params.line_search_diving, -1, 1, -1, "line-search diving toggle: -1 automatic, 0 disabled, 1 enabled"},
    {CUOPT_MIP_HYPER_DIVING_PSEUDOCOST, &mip_settings.diving_params.pseudocost_diving, -1, 1, -1, "pseudocost diving toggle: -1 automatic, 0 disabled, 1 enabled"},
    {CUOPT_MIP_HYPER_DIVING_GUIDED, &mip_settings.diving_params.guided_diving, -1, 1, -1, "guided diving toggle: -1 automatic, 0 disabled, 1 enabled"},
    {CUOPT_MIP_HYPER_DIVING_COEFFICIENT, &mip_settings.diving_params.coefficient_diving, -1, 1, -1, "coefficient diving toggle: -1 automatic, 0 disabled, 1 enabled"},
    {CUOPT_MIP_HYPER_DIVING_FARKAS, &mip_settings.diving_params.farkas_diving, -1, 1, -1, "Farkas diving toggle: -1 automatic, 0 disabled, 1 enabled"},
    {CUOPT_MIP_HYPER_DIVING_VECTOR_LENGTH, &mip_settings.diving_params.vector_length_diving, -1, 1, -1, "vector-length diving toggle: -1 automatic, 0 disabled, 1 enabled"},
    {CUOPT_MIP_HYPER_DIVING_MIN_NODE_DEPTH, &mip_settings.diving_params.min_node_depth, 0, std::numeric_limits<i_t>::max(), 10, "minimum depth at which to start diving"},
    {CUOPT_MIP_HYPER_DIVING_NODE_LIMIT, &mip_settings.diving_params.node_limit, 0, std::numeric_limits<i_t>::max(), 500, "maximum nodes explored per dive"},
    {CUOPT_MIP_HYPER_DIVING_BACKTRACK_LIMIT, &mip_settings.diving_params.backtrack_limit, 0, std::numeric_limits<int16_t>::max(), 5, "maximum backtracking allowed per dive"},
    // Recursive sub-MIP (RINS) hyper-parameters (hidden from default --help: name contains "hyper_")
    {CUOPT_MIP_HYPER_SUBMIP_NODE_LIMIT_OFFSET, &mip_settings.submip_params.node_limit_offset, 0, std::numeric_limits<i_t>::max(), 200, "base node limit for the sub-MIP"},
    {CUOPT_MIP_HYPER_SUBMIP_ITERATION_LIMIT_OFFSET, &mip_settings.submip_params.iteration_limit_offset, 0, std::numeric_limits<i_t>::max(), 10000, "base sub-MIP simplex-iteration limit for root heuristics"},
    {CUOPT_MIP_HYPER_SUBMIP_MAX_LEVEL, &mip_settings.submip_params.max_level, 0, std::numeric_limits<i_t>::max(), 10, "maximum sub-MIP recursion level"},
    {CUOPT_BARRIER_PRESOLVE_BOUND_FREE_VARIABLES, &pdlp_settings.barrier_presolve_bound_free_variables, -1, 1, -1, "Bound free variables during barrier presolve: -1 automatic (default behavior), 0 disabled, 1 enabled"},
    {CUOPT_BARRIER_ADAPTIVE_REGULARIZATION, &pdlp_settings.barrier_adaptive_regularization, -1, 1, -1, "Adaptive regularization for barrier method: -1 automatic (default behavior), 0 disabled, 1 enabled"},
    // QCQP (barrier) scaling hyper-parameter
    {CUOPT_QCQP_HYPER_RUIZ_EQUILIBRATION, &pdlp_settings.qcqp_ruiz_equilibration, -1, 1, -1, "Ruiz equilibration for QCQP barrier scaling: -1 automatic (row/column imbalance heuristic), 0 disabled, 1 enabled"},
  };

    // Bool parameters
  bool_parameters = {
    {CUOPT_INFEASIBILITY_DETECTION, &pdlp_settings.detect_infeasibility, false},
    {CUOPT_STRICT_INFEASIBILITY, &pdlp_settings.strict_infeasibility, false},
    {CUOPT_PER_CONSTRAINT_RESIDUAL, &pdlp_settings.per_constraint_residual, false},
    {CUOPT_SAVE_BEST_PRIMAL_SO_FAR, &pdlp_settings.save_best_primal_so_far, false},
    {CUOPT_FIRST_PRIMAL_FEASIBLE, &pdlp_settings.first_primal_feasible, false},
    {CUOPT_MIP_HEURISTICS_ONLY, &mip_settings.heuristics_only, false},
    {CUOPT_LOG_TO_CONSOLE, &pdlp_settings.log_to_console, true},
    {CUOPT_LOG_TO_CONSOLE, &mip_settings.log_to_console, true},
    {CUOPT_CROSSOVER, &pdlp_settings.crossover, false},
    {CUOPT_ELIMINATE_DENSE_COLUMNS, &pdlp_settings.eliminate_dense_columns, true},
    {CUOPT_CUDSS_DETERMINISTIC, &pdlp_settings.cudss_deterministic, false},
    {CUOPT_DUAL_POSTSOLVE, &pdlp_settings.dual_postsolve, true},
    {CUOPT_BARRIER_ITERATIVE_REFINEMENT, &pdlp_settings.barrier_iterative_refinement, true},
    {CUOPT_MIP_PROBING, &mip_settings.probing, true},
    {CUOPT_USE_DISTRIBUTED_PDLP, &pdlp_settings.use_distributed_pdlp, false},
    // Diving heuristic hyper-parameters (hidden from default --help: name contains "hyper_")
    {CUOPT_MIP_HYPER_DIVING_SHOW_TYPE, &mip_settings.diving_params.show_type, false, "log diving heuristic type when it finds a new incumbent"},
    // Recursive sub-MIP (RINS) hyper-parameters (hidden from default --help: name contains "hyper_")
    {CUOPT_MIP_HYPER_SUBMIP_ENABLE_CPUFJ, &mip_settings.submip_params.enable_cpufj, true, "run CPU FJ over the sub-MIP"},
    {CUOPT_MIP_HYPER_BLOCK_BVE, &mip_settings.block_bve, true, "eliminate blocks of binaries in cuOpt's MIP presolve (needs " CUOPT_MIP_PROBING ")"},
  };
  // String parameters
  string_parameters = {
    {CUOPT_LOG_FILE,  &mip_settings.log_file, ""},
    {CUOPT_LOG_FILE,  &pdlp_settings.log_file, ""},
    {CUOPT_SOLUTION_FILE,  &mip_settings.sol_file, ""},
    {CUOPT_SOLUTION_FILE,  &pdlp_settings.sol_file, ""},
    {CUOPT_USER_PROBLEM_FILE, &mip_settings.user_problem_file, ""},
    {CUOPT_USER_PROBLEM_FILE, &pdlp_settings.user_problem_file, ""},
    {CUOPT_PRESOLVE_FILE, &mip_settings.presolve_file, ""},
    {CUOPT_PRESOLVE_FILE, &pdlp_settings.presolve_file, ""},
  };
  // clang-format on
}

#if MIP_INSTANTIATE_FLOAT
// Emits the ctor/dtor/copy for the whole class; solver_settings.cpp deliberately does not,
// because those need CUDA (see the note there).
template class CUOPT_EXPORT solver_settings_t<int, float>;
#endif

#if MIP_INSTANTIATE_DOUBLE
// Emits the ctor/dtor/copy for the whole class; solver_settings.cpp deliberately does not,
// because those need CUDA (see the note there).
template class CUOPT_EXPORT solver_settings_t<int, double>;
#endif

}  // namespace CUOPT_EXPORT mathematical_optimization
}  // namespace cuopt
