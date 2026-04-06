/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#include <cuopt/error.hpp>
#include <cuopt/linear_programming/solver_settings.hpp>
#include <mip_heuristics/mip_constants.hpp>
#include <utilities/logger.hpp>

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <sstream>

namespace cuopt::linear_programming {

namespace {

bool string_to_int(const std::string& value, int& result)
{
  try {
    size_t pos = 0;
    result     = std::stoi(value, &pos);
    return pos == value.size();
  } catch (const std::exception&) {
    return false;
  }
}

template <typename f_t>
bool string_to_float(const std::string& value, f_t& result)
{
  try {
    size_t pos = 0;
    if constexpr (std::is_same_v<f_t, float>) { result = std::stof(value, &pos); }
    if constexpr (std::is_same_v<f_t, double>) { result = std::stod(value, &pos); }
    if (std::isnan(result)) { return false; }
    return pos == value.size();
  } catch (const std::exception&) {
    return false;
  }
}

std::string quote_if_needed(const std::string& s)
{
  bool needs_quoting = s.empty() || s.find(' ') != std::string::npos ||
                       s.find('"') != std::string::npos || s.find('\t') != std::string::npos;
  if (!needs_quoting) return s;
  std::string out = "\"";
  for (char c : s) {
    if (c == '"')
      out += "\\\"";
    else
      out += c;
  }
  out += '"';
  return out;
}

bool string_to_bool(const std::string& value, bool& result)
{
  if (value == "true" || value == "True" || value == "TRUE" || value == "1" || value == "t" ||
      value == "T") {
    result = true;
    return true;
  } else if (value == "false" || value == "False" || value == "FALSE" || value == "0" ||
             value == "f" || value == "F") {
    result = false;
    return true;
  } else {
    return false;
  }
}

}  // namespace

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
    {CUOPT_MIP_ABSOLUTE_TOLERANCE, &mip_settings.tolerances.absolute_tolerance, f_t(0.0), f_t(1e-1), f_t(1e-4)},
    {CUOPT_MIP_RELATIVE_TOLERANCE, &mip_settings.tolerances.relative_tolerance, f_t(0.0), f_t(1e-1), f_t(1e-4)},
    {CUOPT_MIP_INTEGRALITY_TOLERANCE, &mip_settings.tolerances.integrality_tolerance, f_t(0.0), f_t(1e-1), f_t(1e-5)},
    {CUOPT_MIP_ABSOLUTE_GAP, &mip_settings.tolerances.absolute_mip_gap, f_t(0.0), std::numeric_limits<f_t>::infinity(), std::max(f_t(1e-10), std::numeric_limits<f_t>::epsilon())},
    {CUOPT_MIP_RELATIVE_GAP, &mip_settings.tolerances.relative_mip_gap, f_t(0.0), f_t(1e-1), f_t(1e-4)},
    {CUOPT_PRIMAL_INFEASIBLE_TOLERANCE, &pdlp_settings.tolerances.primal_infeasible_tolerance, f_t(0.0), f_t(1e-1), std::max(f_t(1e-10), std::numeric_limits<f_t>::epsilon())},
    {CUOPT_DUAL_INFEASIBLE_TOLERANCE, &pdlp_settings.tolerances.dual_infeasible_tolerance, f_t(0.0), f_t(1e-1), std::max(f_t(1e-10), std::numeric_limits<f_t>::epsilon())},
    {CUOPT_MIP_CUT_CHANGE_THRESHOLD, &mip_settings.cut_change_threshold, f_t(-1.0), std::numeric_limits<f_t>::infinity(), f_t(-1.0)},
    {CUOPT_MIP_CUT_MIN_ORTHOGONALITY, &mip_settings.cut_min_orthogonality, f_t(0.0), f_t(1.0), f_t(0.5)},
    // MIP heuristic hyper-parameters (hidden from default --help: name contains "hyper_")
    {CUOPT_MIP_HYPER_HEURISTIC_PRESOLVE_TIME_RATIO, &mip_settings.heuristic_params.presolve_time_ratio, f_t(0.0), f_t(1.0), f_t(0.1), "fraction of total time for presolve"},
    {CUOPT_MIP_HYPER_HEURISTIC_PRESOLVE_MAX_TIME, &mip_settings.heuristic_params.presolve_max_time, f_t(0.0), std::numeric_limits<f_t>::infinity(), f_t(60.0), "hard cap on presolve seconds"},
    {CUOPT_MIP_HYPER_HEURISTIC_ROOT_LP_TIME_RATIO, &mip_settings.heuristic_params.root_lp_time_ratio, f_t(0.0), f_t(1.0), f_t(0.1), "fraction of total time for root LP"},
    {CUOPT_MIP_HYPER_HEURISTIC_ROOT_LP_MAX_TIME, &mip_settings.heuristic_params.root_lp_max_time, f_t(0.0), std::numeric_limits<f_t>::infinity(), f_t(15.0), "hard cap on root LP seconds"},
    {CUOPT_MIP_HYPER_HEURISTIC_RINS_TIME_LIMIT, &mip_settings.heuristic_params.rins_time_limit, f_t(0.0), std::numeric_limits<f_t>::infinity(), f_t(3.0), "per-call RINS sub-MIP time"},
    {CUOPT_MIP_HYPER_HEURISTIC_RINS_MAX_TIME_LIMIT, &mip_settings.heuristic_params.rins_max_time_limit, f_t(0.0), std::numeric_limits<f_t>::infinity(), f_t(20.0), "ceiling for RINS adaptive time budget"},
    {CUOPT_MIP_HYPER_HEURISTIC_RINS_FIX_RATE, &mip_settings.heuristic_params.rins_fix_rate, f_t(0.0), f_t(1.0), f_t(0.5), "RINS variable fix rate"},
    {CUOPT_MIP_HYPER_HEURISTIC_INITIAL_INFEASIBILITY_WEIGHT, &mip_settings.heuristic_params.initial_infeasibility_weight, f_t(1e-9), std::numeric_limits<f_t>::infinity(), f_t(1000.0), "constraint violation penalty seed"},
    {CUOPT_MIP_HYPER_HEURISTIC_RELAXED_LP_TIME_LIMIT, &mip_settings.heuristic_params.relaxed_lp_time_limit, f_t(1e-9), std::numeric_limits<f_t>::infinity(), f_t(1.0), "base relaxed LP time cap in heuristics"},
    {CUOPT_MIP_HYPER_HEURISTIC_RELATED_VARS_TIME_LIMIT, &mip_settings.heuristic_params.related_vars_time_limit, f_t(1e-9), std::numeric_limits<f_t>::infinity(), f_t(30.0), "time for related-variable structure build"},
   };

  // Int parameters
  // TODO should we have Stable2 and Methodolical1 here?
  int_parameters = {
    {CUOPT_ITERATION_LIMIT, &pdlp_settings.iteration_limit, 0, std::numeric_limits<i_t>::max(), std::numeric_limits<i_t>::max()},
    {CUOPT_PDLP_SOLVER_MODE, reinterpret_cast<int*>(&pdlp_settings.pdlp_solver_mode), CUOPT_PDLP_SOLVER_MODE_STABLE1, CUOPT_PDLP_SOLVER_MODE_STABLE3, CUOPT_PDLP_SOLVER_MODE_STABLE3},
    {CUOPT_METHOD, reinterpret_cast<int*>(&pdlp_settings.method), CUOPT_METHOD_CONCURRENT, CUOPT_METHOD_BARRIER, CUOPT_METHOD_CONCURRENT},
    {CUOPT_NUM_CPU_THREADS, &mip_settings.num_cpu_threads, -1, std::numeric_limits<i_t>::max(), -1},
    {CUOPT_AUGMENTED, &pdlp_settings.augmented, -1, 1, -1},
    {CUOPT_FOLDING, &pdlp_settings.folding, -1, 1, -1},
    {CUOPT_DUALIZE, &pdlp_settings.dualize, -1, 1, -1},
    {CUOPT_ORDERING, &pdlp_settings.ordering, -1, 1, -1},
    {CUOPT_BARRIER_DUAL_INITIAL_POINT, &pdlp_settings.barrier_dual_initial_point, -1, 1, -1},
    {CUOPT_MIP_CUT_PASSES, &mip_settings.max_cut_passes, -1, std::numeric_limits<i_t>::max(), 10},
    {CUOPT_MIP_MIXED_INTEGER_ROUNDING_CUTS, &mip_settings.mir_cuts, -1, 1, -1},
    {CUOPT_MIP_MIXED_INTEGER_GOMORY_CUTS, &mip_settings.mixed_integer_gomory_cuts, -1, 1, -1},
    {CUOPT_MIP_KNAPSACK_CUTS, &mip_settings.knapsack_cuts, -1, 1, -1},
    {CUOPT_MIP_CLIQUE_CUTS, &mip_settings.clique_cuts, -1, 1, -1},
    {CUOPT_MIP_IMPLIED_BOUND_CUTS, &mip_settings.implied_bound_cuts, -1, 1, -1},
    {CUOPT_MIP_STRONG_CHVATAL_GOMORY_CUTS, &mip_settings.strong_chvatal_gomory_cuts, -1, 1, -1},
    {CUOPT_MIP_REDUCED_COST_STRENGTHENING, &mip_settings.reduced_cost_strengthening, -1, std::numeric_limits<i_t>::max(), -1},
    {CUOPT_NUM_GPUS, &pdlp_settings.num_gpus, 1, 2, 1},
    {CUOPT_NUM_GPUS, &mip_settings.num_gpus, 1, 2, 1},
    {CUOPT_MIP_BATCH_PDLP_STRONG_BRANCHING, &mip_settings.mip_batch_pdlp_strong_branching, 0, 2, 0},
    {CUOPT_MIP_BATCH_PDLP_RELIABILITY_BRANCHING, &mip_settings.mip_batch_pdlp_reliability_branching, 0, 2, 0},
    {CUOPT_MIP_STRONG_BRANCHING_SIMPLEX_ITERATION_LIMIT, &mip_settings.strong_branching_simplex_iteration_limit, -1,std::numeric_limits<i_t>::max(), -1},
    {CUOPT_PRESOLVE, reinterpret_cast<int*>(&pdlp_settings.presolver), CUOPT_PRESOLVE_DEFAULT, CUOPT_PRESOLVE_PSLP, CUOPT_PRESOLVE_DEFAULT},
    {CUOPT_PRESOLVE, reinterpret_cast<int*>(&mip_settings.presolver), CUOPT_PRESOLVE_DEFAULT, CUOPT_PRESOLVE_PSLP, CUOPT_PRESOLVE_DEFAULT},
    {CUOPT_MIP_DETERMINISM_MODE, &mip_settings.determinism_mode, CUOPT_MODE_OPPORTUNISTIC, CUOPT_MODE_DETERMINISTIC, CUOPT_MODE_OPPORTUNISTIC},
    {CUOPT_RANDOM_SEED, &mip_settings.seed, -1, std::numeric_limits<i_t>::max(), -1},
    {CUOPT_MIP_RELIABILITY_BRANCHING, &mip_settings.reliability_branching, -1, std::numeric_limits<i_t>::max(), -1},
    {CUOPT_PDLP_PRECISION, reinterpret_cast<int*>(&pdlp_settings.pdlp_precision), CUOPT_PDLP_DEFAULT_PRECISION, CUOPT_PDLP_MIXED_PRECISION, CUOPT_PDLP_DEFAULT_PRECISION},
    {CUOPT_MIP_SCALING, &mip_settings.mip_scaling, CUOPT_MIP_SCALING_OFF, CUOPT_MIP_SCALING_NO_OBJECTIVE, CUOPT_MIP_SCALING_ON},
    // MIP heuristic hyper-parameters (hidden from default --help: name contains "hyper_")
    {CUOPT_MIP_HYPER_HEURISTIC_POPULATION_SIZE, &mip_settings.heuristic_params.population_size, 1, std::numeric_limits<i_t>::max(), 32, "max solutions in pool"},
    {CUOPT_MIP_HYPER_HEURISTIC_NUM_CPUFJ_THREADS, &mip_settings.heuristic_params.num_cpufj_threads, 0, std::numeric_limits<i_t>::max(), 8, "parallel CPU FJ climbers"},
    {CUOPT_MIP_HYPER_HEURISTIC_STAGNATION_TRIGGER, &mip_settings.heuristic_params.stagnation_trigger, 1, std::numeric_limits<i_t>::max(), 3, "FP loops w/o improvement before recombination"},
    {CUOPT_MIP_HYPER_HEURISTIC_MAX_ITERS_WITHOUT_IMPROVEMENT, &mip_settings.heuristic_params.max_iterations_without_improvement, 1, std::numeric_limits<i_t>::max(), 8, "diversity step depth after stagnation"},
    {CUOPT_MIP_HYPER_HEURISTIC_N_OF_MINIMUMS_FOR_EXIT, &mip_settings.heuristic_params.n_of_minimums_for_exit, 1, std::numeric_limits<i_t>::max(), 7000, "FJ baseline local-minima exit threshold"},
    {CUOPT_MIP_HYPER_HEURISTIC_ENABLED_RECOMBINERS, &mip_settings.heuristic_params.enabled_recombiners, 0, 15, 15, "bitmask: 1=BP 2=FP 4=LS 8=SubMIP"},
    {CUOPT_MIP_HYPER_HEURISTIC_CYCLE_DETECTION_LENGTH, &mip_settings.heuristic_params.cycle_detection_length, 1, std::numeric_limits<i_t>::max(), 30, "FP assignment cycle ring buffer length"},
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
    {CUOPT_PRESOLVE_FILE, &pdlp_settings.presolve_file, ""}
  };
  // clang-format on
}

template <typename i_t, typename f_t>
void solver_settings_t<i_t, f_t>::set_parameter_from_string(const std::string& name,
                                                            const std::string& value)
{
  bool found  = false;
  bool output = false;
  for (auto& param : int_parameters) {
    if (param.param_name == name) {
      i_t value_int;
      if (string_to_int(value, value_int)) {
        if (value_int < param.min_value || value_int > param.max_value) {
          throw std::invalid_argument("Parameter " + name + " value " + value + " out of range");
        }
        *param.value_ptr = value_int;
        found            = true;
        if (!output) {
          CUOPT_LOG_INFO("Setting parameter %s to %d", name.c_str(), value_int);
          output = true;
        }
      } else {
        throw std::invalid_argument("Parameter " + name + " value " + value + " is not an integer");
      }
    }
  }
  for (auto& param : float_parameters) {
    if (param.param_name == name) {
      f_t value_float;
      if (string_to_float<f_t>(value, value_float)) {
        if (value_float < param.min_value || value_float > param.max_value) {
          throw std::invalid_argument("Parameter " + name + " value " + value + " out of range");
        }
        *param.value_ptr = value_float;
        found            = true;
        if (!output) {
          CUOPT_LOG_INFO("Setting parameter %s to %e", name.c_str(), value_float);
          output = true;
        }
      } else {
        throw std::invalid_argument("Parameter " + name + " value " + value + " is not a float");
      }
    }
  }
  for (auto& param : bool_parameters) {
    if (param.param_name == name) {
      bool value_bool;
      if (string_to_bool(value, value_bool)) {
        *param.value_ptr = value_bool;
        found            = true;
        if (!output) {
          CUOPT_LOG_INFO("Setting parameter %s to %s", name.c_str(), value_bool ? "true" : "false");
          output = true;
        }
      } else {
        throw std::invalid_argument("Parameter " + name + " value " + value +
                                    " must be true or false");
      }
    }
  }

  for (auto& param : string_parameters) {
    if (param.param_name == name) {
      *param.value_ptr = value;
      if (!output) {
        CUOPT_LOG_INFO("Setting parameter %s to %s", name.c_str(), value.c_str());
        output = true;
      }
      found = true;
    }
  }
  if (!found) { throw std::invalid_argument("Parameter " + name + " not found"); }
}

template <typename i_t, typename f_t>
template <typename T>
void solver_settings_t<i_t, f_t>::set_parameter(const std::string& name, T value)
{
  bool found  = false;
  bool output = false;
  if constexpr (std::is_same_v<T, i_t>) {
    for (auto& param : int_parameters) {
      if (param.param_name == name) {
        if (value < param.min_value || value > param.max_value) {
          throw std::invalid_argument("Parameter " + name + " out of range");
        }
        *param.value_ptr = value;
        if (!output) {
          CUOPT_LOG_INFO("Setting parameter %s to %d", name.c_str(), value);
          output = true;
        }
        found = true;
      }
    }
  }
  if constexpr (std::is_same_v<T, f_t>) {
    for (auto& param : float_parameters) {
      if (param.param_name == name) {
        if (value < param.min_value || value > param.max_value) {
          throw std::invalid_argument("Parameter " + name + " out of range");
        }
        *param.value_ptr = value;
        if (!output) {
          CUOPT_LOG_INFO("Setting parameter %s to %e", name.c_str(), value);
          output = true;
        }
        found = true;
      }
    }
  }
  if constexpr (std::is_same_v<T, bool>) {
    for (auto& param : bool_parameters) {
      if (param.param_name == name) {
        *param.value_ptr = value;
        if (!output) {
          CUOPT_LOG_INFO("Setting parameter %s to %s", name.c_str(), value ? "true" : "false");
          output = true;
        }
        found = true;
      }
    }
  }
  if constexpr (std::is_same_v<T, std::string>) {
    for (auto& param : string_parameters) {
      if (param.param_name == name) {
        *param.value_ptr = value;
        if (!output) {
          CUOPT_LOG_INFO("Setting parameter %s to %s", name.c_str(), value.c_str());
          output = true;
        }
        found = true;
      }
    }
  }
  if (!found) { throw std::invalid_argument("Parameter " + name + " not found"); }
}

template <typename i_t, typename f_t>
template <typename T>
T solver_settings_t<i_t, f_t>::get_parameter(const std::string& name) const
{
  if constexpr (std::is_same_v<T, i_t>) {
    for (auto& param : int_parameters) {
      if (param.param_name == name) { return *param.value_ptr; }
    }
  }
  if constexpr (std::is_same_v<T, f_t>) {
    for (auto& param : float_parameters) {
      if (param.param_name == name) { return *param.value_ptr; }
    }
  }
  if constexpr (std::is_same_v<T, bool>) {
    for (auto& param : bool_parameters) {
      if (param.param_name == name) { return *param.value_ptr; }
    }
  }
  if constexpr (std::is_same_v<T, std::string>) {
    for (auto& param : string_parameters) {
      if (param.param_name == name) { return *param.value_ptr; }
    }
  }
  throw std::invalid_argument("Parameter " + name + " not found");
}

template <typename i_t, typename f_t>
std::string solver_settings_t<i_t, f_t>::get_parameter_as_string(const std::string& name) const
{
  for (auto& param : int_parameters) {
    if (param.param_name == name) { return std::to_string(*param.value_ptr); }
  }
  for (auto& param : float_parameters) {
    if (param.param_name == name) { return std::to_string(*param.value_ptr); }
  }
  for (auto& param : bool_parameters) {
    if (param.param_name == name) { return *param.value_ptr ? "true" : "false"; }
  }
  for (auto& param : string_parameters) {
    if (param.param_name == name) { return *param.value_ptr; }
  }
  throw std::invalid_argument("Parameter " + name + " not found");
}

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

template <typename i_t, typename f_t>
void solver_settings_t<i_t, f_t>::set_mip_callback(internals::base_solution_callback_t* callback,
                                                   void* user_data)
{
  mip_settings.set_mip_callback(callback, user_data);
}

template <typename i_t, typename f_t>
const std::vector<internals::base_solution_callback_t*>
solver_settings_t<i_t, f_t>::get_mip_callbacks() const
{
  return mip_settings.get_mip_callbacks();
}

template <typename i_t, typename f_t>
pdlp_solver_settings_t<i_t, f_t>& solver_settings_t<i_t, f_t>::get_pdlp_settings()
{
  return pdlp_settings;
}

template <typename i_t, typename f_t>
mip_solver_settings_t<i_t, f_t>& solver_settings_t<i_t, f_t>::get_mip_settings()
{
  return mip_settings;
}

template <typename i_t, typename f_t>
const pdlp_warm_start_data_view_t<i_t, f_t>&
solver_settings_t<i_t, f_t>::get_pdlp_warm_start_data_view() const noexcept
{
  return pdlp_settings.get_pdlp_warm_start_data_view();
}

template <typename i_t, typename f_t>
const std::vector<parameter_info_t<f_t>>& solver_settings_t<i_t, f_t>::get_float_parameters() const
{
  return float_parameters;
}

template <typename i_t, typename f_t>
const std::vector<parameter_info_t<i_t>>& solver_settings_t<i_t, f_t>::get_int_parameters() const
{
  return int_parameters;
}

template <typename i_t, typename f_t>
const std::vector<parameter_info_t<bool>>& solver_settings_t<i_t, f_t>::get_bool_parameters() const
{
  return bool_parameters;
}

template <typename i_t, typename f_t>
const std::vector<parameter_info_t<std::string>>&
solver_settings_t<i_t, f_t>::get_string_parameters() const
{
  return string_parameters;
}

template <typename i_t, typename f_t>
const std::vector<std::string> solver_settings_t<i_t, f_t>::get_parameter_names() const
{
  std::vector<std::string> parameter_names;
  for (auto& param : int_parameters) {
    parameter_names.push_back(param.param_name);
  }
  for (auto& param : float_parameters) {
    parameter_names.push_back(param.param_name);
  }
  for (auto& param : bool_parameters) {
    parameter_names.push_back(param.param_name);
  }
  for (auto& param : string_parameters) {
    parameter_names.push_back(param.param_name);
  }
  return parameter_names;
}

template <typename i_t, typename f_t>
void solver_settings_t<i_t, f_t>::load_parameters_from_file(const std::string& path)
{
  cuopt_expects(!std::filesystem::is_directory(path) && std::filesystem::exists(path),
                error_type_t::ValidationError,
                "Parameter config: not a valid file: %s",
                path.c_str());
  std::ifstream file(path);
  cuopt_expects(file.is_open(),
                error_type_t::ValidationError,
                "Parameter config: cannot open: %s",
                path.c_str());
  std::string line;
  while (std::getline(file, line)) {
    auto first_non_ws = std::find_if_not(line.begin(), line.end(), ::isspace);
    if (first_non_ws == line.end() || *first_non_ws == '#') continue;
    line.erase(line.begin(), first_non_ws);

    std::istringstream iss(line);
    std::string key;
    cuopt_expects(iss >> key >> std::ws && iss.get() == '=',
                  error_type_t::ValidationError,
                  "Parameter config: bad line: %s",
                  line.c_str());
    iss >> std::ws;
    cuopt_expects(!iss.eof(),
                  error_type_t::ValidationError,
                  "Parameter config: missing value: %s",
                  line.c_str());
    std::string val;
    if (iss.peek() == '"') {
      iss.get();
      val.clear();
      char ch;
      bool closed = false;
      while (iss.get(ch)) {
        if (ch == '\\' && iss.peek() == '"') {
          iss.get(ch);
          val += '"';
        } else if (ch == '"') {
          closed = true;
          break;
        } else {
          val += ch;
        }
      }
      cuopt_expects(closed,
                    error_type_t::ValidationError,
                    "Parameter config: unterminated quote: %s",
                    line.c_str());
    } else {
      iss >> val;
    }
    std::string trailing;
    cuopt_expects(!bool(iss >> trailing),
                  error_type_t::ValidationError,
                  "Parameter config: trailing junk: %s",
                  line.c_str());
    try {
      set_parameter_from_string(key, val);
    } catch (const std::invalid_argument& e) {
      cuopt_expects(false, error_type_t::ValidationError, "Parameter config: %s", e.what());
    }
  }
  CUOPT_LOG_INFO("Parameters loaded from: %s", path.c_str());
}

template <typename i_t, typename f_t>
bool solver_settings_t<i_t, f_t>::dump_parameters_to_file(const std::string& path,
                                                          bool hyperparameters_only) const
{
  std::ofstream file(path);
  if (!file.is_open()) {
    CUOPT_LOG_ERROR("Cannot open file for writing: %s", path.c_str());
    return false;
  }
  file << "# cuOpt parameter configuration (auto-generated)\n";
  file << "# Uncomment and change the values you wish to override.\n\n";
  for (const auto& p : int_parameters) {
    if (hyperparameters_only && p.param_name.find("hyper_") == std::string::npos) continue;
    if (p.description && p.description[0] != '\0')
      file << "# " << p.description << " (int, range: [" << p.min_value << ", " << p.max_value
           << "])\n";
    file << "# " << p.param_name << " = " << *p.value_ptr << "\n\n";
  }
  for (const auto& p : float_parameters) {
    if (hyperparameters_only && p.param_name.find("hyper_") == std::string::npos) continue;
    if (p.description && p.description[0] != '\0')
      file << "# " << p.description << " (double, range: [" << p.min_value << ", " << p.max_value
           << "])\n";
    file << "# " << p.param_name << " = " << *p.value_ptr << "\n\n";
  }
  for (const auto& p : bool_parameters) {
    if (hyperparameters_only && p.param_name.find("hyper_") == std::string::npos) continue;
    if (p.description && p.description[0] != '\0') file << "# " << p.description << " (bool)\n";
    file << "# " << p.param_name << " = " << (*p.value_ptr ? "true" : "false") << "\n\n";
  }
  for (const auto& p : string_parameters) {
    if (hyperparameters_only && p.param_name.find("hyper_") == std::string::npos) continue;
    if (p.description && p.description[0] != '\0') file << "# " << p.description << " (string)\n";
    file << "# " << p.param_name << " = " << quote_if_needed(*p.value_ptr) << "\n\n";
  }
  return true;
}

#if MIP_INSTANTIATE_FLOAT
template class solver_settings_t<int, float>;
template void solver_settings_t<int, float>::set_parameter(const std::string& name, int value);
template void solver_settings_t<int, float>::set_parameter(const std::string& name, float value);
template void solver_settings_t<int, float>::set_parameter(const std::string& name, bool value);
template int solver_settings_t<int, float>::get_parameter(const std::string& name) const;
template float solver_settings_t<int, float>::get_parameter(const std::string& name) const;
template bool solver_settings_t<int, float>::get_parameter(const std::string& name) const;
template std::string solver_settings_t<int, float>::get_parameter(const std::string& name) const;
#endif

#if MIP_INSTANTIATE_DOUBLE
template class solver_settings_t<int, double>;
template void solver_settings_t<int, double>::set_parameter(const std::string& name, int value);
template void solver_settings_t<int, double>::set_parameter(const std::string& name, double value);
template void solver_settings_t<int, double>::set_parameter(const std::string& name, bool value);
template int solver_settings_t<int, double>::get_parameter(const std::string& name) const;
template double solver_settings_t<int, double>::get_parameter(const std::string& name) const;
template bool solver_settings_t<int, double>::get_parameter(const std::string& name) const;
template std::string solver_settings_t<int, double>::get_parameter(const std::string& name) const;
#endif

}  // namespace cuopt::linear_programming
