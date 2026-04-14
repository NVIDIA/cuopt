/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved. SPDX-License-Identifier: Apache-2.0
 */

#include "grpc_solution_mapper.hpp"

#include <cuopt/linear_programming/constants.h>
#include <cuopt_remote.pb.h>
#include <cuopt_remote_service.pb.h>
#include <cuopt/linear_programming/cpu_optimization_problem_solution.hpp>
#include <cuopt/linear_programming/mip/solver_solution.hpp>
#include <cuopt/linear_programming/pdlp/solver_solution.hpp>

#include <cstring>
#include <map>

namespace cuopt::linear_programming {

// Convert cuOpt termination status to protobuf enum
cuopt::remote::PDLPTerminationStatus to_proto_pdlp_status(pdlp_termination_status_t status)
{
  switch (status) {
    case pdlp_termination_status_t::NoTermination: return cuopt::remote::PDLP_NO_TERMINATION;
    case pdlp_termination_status_t::NumericalError: return cuopt::remote::PDLP_NUMERICAL_ERROR;
    case pdlp_termination_status_t::Optimal: return cuopt::remote::PDLP_OPTIMAL;
    case pdlp_termination_status_t::PrimalInfeasible: return cuopt::remote::PDLP_PRIMAL_INFEASIBLE;
    case pdlp_termination_status_t::DualInfeasible: return cuopt::remote::PDLP_DUAL_INFEASIBLE;
    case pdlp_termination_status_t::IterationLimit: return cuopt::remote::PDLP_ITERATION_LIMIT;
    case pdlp_termination_status_t::TimeLimit: return cuopt::remote::PDLP_TIME_LIMIT;
    case pdlp_termination_status_t::ConcurrentLimit: return cuopt::remote::PDLP_CONCURRENT_LIMIT;
    case pdlp_termination_status_t::PrimalFeasible: return cuopt::remote::PDLP_PRIMAL_FEASIBLE;
    default: return cuopt::remote::PDLP_NO_TERMINATION;
  }
}

// Convert protobuf enum to cuOpt termination status
pdlp_termination_status_t from_proto_pdlp_status(cuopt::remote::PDLPTerminationStatus status)
{
  switch (status) {
    case cuopt::remote::PDLP_NO_TERMINATION: return pdlp_termination_status_t::NoTermination;
    case cuopt::remote::PDLP_NUMERICAL_ERROR: return pdlp_termination_status_t::NumericalError;
    case cuopt::remote::PDLP_OPTIMAL: return pdlp_termination_status_t::Optimal;
    case cuopt::remote::PDLP_PRIMAL_INFEASIBLE: return pdlp_termination_status_t::PrimalInfeasible;
    case cuopt::remote::PDLP_DUAL_INFEASIBLE: return pdlp_termination_status_t::DualInfeasible;
    case cuopt::remote::PDLP_ITERATION_LIMIT: return pdlp_termination_status_t::IterationLimit;
    case cuopt::remote::PDLP_TIME_LIMIT: return pdlp_termination_status_t::TimeLimit;
    case cuopt::remote::PDLP_CONCURRENT_LIMIT: return pdlp_termination_status_t::ConcurrentLimit;
    case cuopt::remote::PDLP_PRIMAL_FEASIBLE: return pdlp_termination_status_t::PrimalFeasible;
    default: return pdlp_termination_status_t::NoTermination;
  }
}

// Convert MIP termination status
cuopt::remote::MIPTerminationStatus to_proto_mip_status(mip_termination_status_t status)
{
  switch (status) {
    case mip_termination_status_t::NoTermination: return cuopt::remote::MIP_NO_TERMINATION;
    case mip_termination_status_t::Optimal: return cuopt::remote::MIP_OPTIMAL;
    case mip_termination_status_t::FeasibleFound: return cuopt::remote::MIP_FEASIBLE_FOUND;
    case mip_termination_status_t::Infeasible: return cuopt::remote::MIP_INFEASIBLE;
    case mip_termination_status_t::Unbounded: return cuopt::remote::MIP_UNBOUNDED;
    case mip_termination_status_t::TimeLimit: return cuopt::remote::MIP_TIME_LIMIT;
    case mip_termination_status_t::WorkLimit: return cuopt::remote::MIP_WORK_LIMIT;
    default: return cuopt::remote::MIP_NO_TERMINATION;
  }
}

mip_termination_status_t from_proto_mip_status(cuopt::remote::MIPTerminationStatus status)
{
  switch (status) {
    case cuopt::remote::MIP_NO_TERMINATION: return mip_termination_status_t::NoTermination;
    case cuopt::remote::MIP_OPTIMAL: return mip_termination_status_t::Optimal;
    case cuopt::remote::MIP_FEASIBLE_FOUND: return mip_termination_status_t::FeasibleFound;
    case cuopt::remote::MIP_INFEASIBLE: return mip_termination_status_t::Infeasible;
    case cuopt::remote::MIP_UNBOUNDED: return mip_termination_status_t::Unbounded;
    case cuopt::remote::MIP_TIME_LIMIT: return mip_termination_status_t::TimeLimit;
    case cuopt::remote::MIP_WORK_LIMIT: return mip_termination_status_t::WorkLimit;
    default: return mip_termination_status_t::NoTermination;
  }
}

template <typename i_t, typename f_t>
void map_lp_solution_to_proto(const cpu_lp_solution_t<i_t, f_t>& solution,
                              cuopt::remote::LPSolution* pb_solution)
{
  pb_solution->set_lp_termination_status(to_proto_pdlp_status(solution.get_termination_status()));
  pb_solution->set_error_message(solution.get_error_status().what());

  // Solution vectors - CPU solution already has data in host memory
  const auto& primal       = solution.get_primal_solution_host();
  const auto& dual         = solution.get_dual_solution_host();
  const auto& reduced_cost = solution.get_reduced_cost_host();

  for (const auto& v : primal) {
    pb_solution->add_primal_solution(static_cast<double>(v));
  }
  for (const auto& v : dual) {
    pb_solution->add_dual_solution(static_cast<double>(v));
  }
  for (const auto& v : reduced_cost) {
    pb_solution->add_reduced_cost(static_cast<double>(v));
  }

  // Statistics
  pb_solution->set_l2_primal_residual(solution.get_l2_primal_residual());
  pb_solution->set_l2_dual_residual(solution.get_l2_dual_residual());
  pb_solution->set_primal_objective(solution.get_objective_value());
  pb_solution->set_dual_objective(solution.get_dual_objective_value());
  pb_solution->set_gap(solution.get_gap());
  pb_solution->set_nb_iterations(solution.get_num_iterations());
  pb_solution->set_solve_time(solution.get_solve_time());
  pb_solution->set_solved_by(static_cast<int32_t>(solution.solved_by()));

  if (solution.has_warm_start_data()) {
    auto* pb_ws    = pb_solution->mutable_warm_start_data();
    const auto& ws = solution.get_cpu_pdlp_warm_start_data();

    for (const auto& v : ws.current_primal_solution_)
      pb_ws->add_current_primal_solution(static_cast<double>(v));
    for (const auto& v : ws.current_dual_solution_)
      pb_ws->add_current_dual_solution(static_cast<double>(v));
    for (const auto& v : ws.initial_primal_average_)
      pb_ws->add_initial_primal_average(static_cast<double>(v));
    for (const auto& v : ws.initial_dual_average_)
      pb_ws->add_initial_dual_average(static_cast<double>(v));
    for (const auto& v : ws.current_ATY_)
      pb_ws->add_current_aty(static_cast<double>(v));
    for (const auto& v : ws.sum_primal_solutions_)
      pb_ws->add_sum_primal_solutions(static_cast<double>(v));
    for (const auto& v : ws.sum_dual_solutions_)
      pb_ws->add_sum_dual_solutions(static_cast<double>(v));
    for (const auto& v : ws.last_restart_duality_gap_primal_solution_)
      pb_ws->add_last_restart_duality_gap_primal_solution(static_cast<double>(v));
    for (const auto& v : ws.last_restart_duality_gap_dual_solution_)
      pb_ws->add_last_restart_duality_gap_dual_solution(static_cast<double>(v));

    pb_ws->set_initial_primal_weight(static_cast<double>(ws.initial_primal_weight_));
    pb_ws->set_initial_step_size(static_cast<double>(ws.initial_step_size_));
    pb_ws->set_total_pdlp_iterations(static_cast<int32_t>(ws.total_pdlp_iterations_));
    pb_ws->set_total_pdhg_iterations(static_cast<int32_t>(ws.total_pdhg_iterations_));
    pb_ws->set_last_candidate_kkt_score(static_cast<double>(ws.last_candidate_kkt_score_));
    pb_ws->set_last_restart_kkt_score(static_cast<double>(ws.last_restart_kkt_score_));
    pb_ws->set_sum_solution_weight(static_cast<double>(ws.sum_solution_weight_));
    pb_ws->set_iterations_since_last_restart(
      static_cast<int32_t>(ws.iterations_since_last_restart_));
  }
}

template <typename i_t, typename f_t>
cpu_lp_solution_t<i_t, f_t> map_proto_to_lp_solution(const cuopt::remote::LPSolution& pb_solution)
{
  // Convert solution vectors
  std::vector<f_t> primal(pb_solution.primal_solution().begin(),
                          pb_solution.primal_solution().end());
  std::vector<f_t> dual(pb_solution.dual_solution().begin(), pb_solution.dual_solution().end());
  std::vector<f_t> reduced_cost(pb_solution.reduced_cost().begin(),
                                pb_solution.reduced_cost().end());

  auto status    = from_proto_pdlp_status(pb_solution.lp_termination_status());
  auto obj       = static_cast<f_t>(pb_solution.primal_objective());
  auto dual_obj  = static_cast<f_t>(pb_solution.dual_objective());
  auto solve_t   = pb_solution.solve_time();
  auto l2_pr     = static_cast<f_t>(pb_solution.l2_primal_residual());
  auto l2_dr     = static_cast<f_t>(pb_solution.l2_dual_residual());
  auto g         = static_cast<f_t>(pb_solution.gap());
  auto iters     = static_cast<i_t>(pb_solution.nb_iterations());
  auto solved_by = static_cast<method_t>(pb_solution.solved_by());

  if (pb_solution.has_warm_start_data()) {
    const auto& pb_ws = pb_solution.warm_start_data();
    cpu_pdlp_warm_start_data_t<i_t, f_t> ws;

    ws.current_primal_solution_.assign(pb_ws.current_primal_solution().begin(),
                                       pb_ws.current_primal_solution().end());
    ws.current_dual_solution_.assign(pb_ws.current_dual_solution().begin(),
                                     pb_ws.current_dual_solution().end());
    ws.initial_primal_average_.assign(pb_ws.initial_primal_average().begin(),
                                      pb_ws.initial_primal_average().end());
    ws.initial_dual_average_.assign(pb_ws.initial_dual_average().begin(),
                                    pb_ws.initial_dual_average().end());
    ws.current_ATY_.assign(pb_ws.current_aty().begin(), pb_ws.current_aty().end());
    ws.sum_primal_solutions_.assign(pb_ws.sum_primal_solutions().begin(),
                                    pb_ws.sum_primal_solutions().end());
    ws.sum_dual_solutions_.assign(pb_ws.sum_dual_solutions().begin(),
                                  pb_ws.sum_dual_solutions().end());
    ws.last_restart_duality_gap_primal_solution_.assign(
      pb_ws.last_restart_duality_gap_primal_solution().begin(),
      pb_ws.last_restart_duality_gap_primal_solution().end());
    ws.last_restart_duality_gap_dual_solution_.assign(
      pb_ws.last_restart_duality_gap_dual_solution().begin(),
      pb_ws.last_restart_duality_gap_dual_solution().end());

    ws.initial_primal_weight_         = static_cast<f_t>(pb_ws.initial_primal_weight());
    ws.initial_step_size_             = static_cast<f_t>(pb_ws.initial_step_size());
    ws.total_pdlp_iterations_         = static_cast<i_t>(pb_ws.total_pdlp_iterations());
    ws.total_pdhg_iterations_         = static_cast<i_t>(pb_ws.total_pdhg_iterations());
    ws.last_candidate_kkt_score_      = static_cast<f_t>(pb_ws.last_candidate_kkt_score());
    ws.last_restart_kkt_score_        = static_cast<f_t>(pb_ws.last_restart_kkt_score());
    ws.sum_solution_weight_           = static_cast<f_t>(pb_ws.sum_solution_weight());
    ws.iterations_since_last_restart_ = static_cast<i_t>(pb_ws.iterations_since_last_restart());

    return cpu_lp_solution_t<i_t, f_t>(std::move(primal),
                                       std::move(dual),
                                       std::move(reduced_cost),
                                       status,
                                       obj,
                                       dual_obj,
                                       solve_t,
                                       l2_pr,
                                       l2_dr,
                                       g,
                                       iters,
                                       solved_by,
                                       std::move(ws));
  }

  return cpu_lp_solution_t<i_t, f_t>(std::move(primal),
                                     std::move(dual),
                                     std::move(reduced_cost),
                                     status,
                                     obj,
                                     dual_obj,
                                     solve_t,
                                     l2_pr,
                                     l2_dr,
                                     g,
                                     iters,
                                     solved_by);
}

template <typename i_t, typename f_t>
void map_mip_solution_to_proto(const cpu_mip_solution_t<i_t, f_t>& solution,
                               cuopt::remote::MIPSolution* pb_solution)
{
  pb_solution->set_mip_termination_status(to_proto_mip_status(solution.get_termination_status()));
  pb_solution->set_mip_error_message(solution.get_error_status().what());

  // Solution vector - CPU solution already has data in host memory
  const auto& sol_vec = solution.get_solution_host();
  for (const auto& v : sol_vec) {
    pb_solution->add_mip_solution(static_cast<double>(v));
  }

  // Solution statistics
  pb_solution->set_mip_objective(solution.get_objective_value());
  pb_solution->set_mip_gap(solution.get_mip_gap());
  pb_solution->set_solution_bound(solution.get_solution_bound());
  pb_solution->set_total_solve_time(solution.get_solve_time());
  pb_solution->set_presolve_time(solution.get_presolve_time());
  pb_solution->set_max_constraint_violation(solution.get_max_constraint_violation());
  pb_solution->set_max_int_violation(solution.get_max_int_violation());
  pb_solution->set_max_variable_bound_violation(solution.get_max_variable_bound_violation());
  pb_solution->set_nodes(solution.get_num_nodes());
  pb_solution->set_simplex_iterations(solution.get_num_simplex_iterations());
}

template <typename i_t, typename f_t>
cpu_mip_solution_t<i_t, f_t> map_proto_to_mip_solution(
  const cuopt::remote::MIPSolution& pb_solution)
{
  // Convert solution vector
  std::vector<f_t> solution_vec(pb_solution.mip_solution().begin(),
                                pb_solution.mip_solution().end());

  // Create CPU MIP solution with data
  return cpu_mip_solution_t<i_t, f_t>(std::move(solution_vec),
                                      from_proto_mip_status(pb_solution.mip_termination_status()),
                                      static_cast<f_t>(pb_solution.mip_objective()),
                                      static_cast<f_t>(pb_solution.mip_gap()),
                                      static_cast<f_t>(pb_solution.solution_bound()),
                                      pb_solution.total_solve_time(),
                                      pb_solution.presolve_time(),
                                      static_cast<f_t>(pb_solution.max_constraint_violation()),
                                      static_cast<f_t>(pb_solution.max_int_violation()),
                                      static_cast<f_t>(pb_solution.max_variable_bound_violation()),
                                      static_cast<i_t>(pb_solution.nodes()),
                                      static_cast<i_t>(pb_solution.simplex_iterations()));
}

// ============================================================================
// Size estimation
// ============================================================================

template <typename i_t, typename f_t>
size_t estimate_lp_solution_proto_size(const cpu_lp_solution_t<i_t, f_t>& solution)
{
  size_t est = 0;
  est += static_cast<size_t>(solution.get_primal_solution_size()) * sizeof(double);
  est += static_cast<size_t>(solution.get_dual_solution_size()) * sizeof(double);
  est += static_cast<size_t>(solution.get_reduced_cost_size()) * sizeof(double);
  if (solution.has_warm_start_data()) {
    const auto& ws = solution.get_cpu_pdlp_warm_start_data();
    est += ws.current_primal_solution_.size() * sizeof(double);
    est += ws.current_dual_solution_.size() * sizeof(double);
    est += ws.initial_primal_average_.size() * sizeof(double);
    est += ws.initial_dual_average_.size() * sizeof(double);
    est += ws.current_ATY_.size() * sizeof(double);
    est += ws.sum_primal_solutions_.size() * sizeof(double);
    est += ws.sum_dual_solutions_.size() * sizeof(double);
    est += ws.last_restart_duality_gap_primal_solution_.size() * sizeof(double);
    est += ws.last_restart_duality_gap_dual_solution_.size() * sizeof(double);
  }
  est += 512;  // scalars + tags overhead
  return est;
}

template <typename i_t, typename f_t>
size_t estimate_mip_solution_proto_size(const cpu_mip_solution_t<i_t, f_t>& solution)
{
  size_t est = 0;
  est += static_cast<size_t>(solution.get_solution_size()) * sizeof(double);
  est += 256;  // scalars + tags overhead
  return est;
}

// ============================================================================
// Chunked result header population
// ============================================================================

namespace {
void add_result_array_descriptor(cuopt::remote::ChunkedResultHeader* header,
                                 cuopt::remote::ResultFieldId fid,
                                 int64_t count,
                                 int64_t elem_size)
{
  if (count <= 0) return;
  auto* desc = header->add_arrays();
  desc->set_field_id(fid);
  desc->set_total_elements(count);
  desc->set_element_size_bytes(elem_size);
}

template <typename f_t>
std::vector<uint8_t> doubles_to_bytes(const std::vector<f_t>& vec)
{
  std::vector<double> tmp(vec.begin(), vec.end());
  std::vector<uint8_t> bytes(tmp.size() * sizeof(double));
  std::memcpy(bytes.data(), tmp.data(), bytes.size());
  return bytes;
}
}  // namespace

template <typename i_t, typename f_t>
void populate_chunked_result_header_lp(const cpu_lp_solution_t<i_t, f_t>& solution,
                                       cuopt::remote::ChunkedResultHeader* header)
{
  header->set_problem_category(cuopt::remote::LP);
  header->set_lp_termination_status(to_proto_pdlp_status(solution.get_termination_status()));
  header->set_error_message(solution.get_error_status().what());
  header->set_l2_primal_residual(solution.get_l2_primal_residual());
  header->set_l2_dual_residual(solution.get_l2_dual_residual());
  header->set_primal_objective(solution.get_objective_value());
  header->set_dual_objective(solution.get_dual_objective_value());
  header->set_gap(solution.get_gap());
  header->set_nb_iterations(solution.get_num_iterations());
  header->set_solve_time(solution.get_solve_time());
  header->set_solved_by(static_cast<int32_t>(solution.solved_by()));

  const auto& primal       = solution.get_primal_solution_host();
  const auto& dual         = solution.get_dual_solution_host();
  const auto& reduced_cost = solution.get_reduced_cost_host();

  add_result_array_descriptor(
    header, cuopt::remote::RESULT_PRIMAL_SOLUTION, primal.size(), sizeof(double));
  add_result_array_descriptor(
    header, cuopt::remote::RESULT_DUAL_SOLUTION, dual.size(), sizeof(double));
  add_result_array_descriptor(
    header, cuopt::remote::RESULT_REDUCED_COST, reduced_cost.size(), sizeof(double));

  if (solution.has_warm_start_data()) {
    const auto& ws = solution.get_cpu_pdlp_warm_start_data();
    header->set_ws_initial_primal_weight(static_cast<double>(ws.initial_primal_weight_));
    header->set_ws_initial_step_size(static_cast<double>(ws.initial_step_size_));
    header->set_ws_total_pdlp_iterations(static_cast<int32_t>(ws.total_pdlp_iterations_));
    header->set_ws_total_pdhg_iterations(static_cast<int32_t>(ws.total_pdhg_iterations_));
    header->set_ws_last_candidate_kkt_score(static_cast<double>(ws.last_candidate_kkt_score_));
    header->set_ws_last_restart_kkt_score(static_cast<double>(ws.last_restart_kkt_score_));
    header->set_ws_sum_solution_weight(static_cast<double>(ws.sum_solution_weight_));
    header->set_ws_iterations_since_last_restart(
      static_cast<int32_t>(ws.iterations_since_last_restart_));

    add_result_array_descriptor(header,
                                cuopt::remote::RESULT_WS_CURRENT_PRIMAL,
                                ws.current_primal_solution_.size(),
                                sizeof(double));
    add_result_array_descriptor(header,
                                cuopt::remote::RESULT_WS_CURRENT_DUAL,
                                ws.current_dual_solution_.size(),
                                sizeof(double));
    add_result_array_descriptor(header,
                                cuopt::remote::RESULT_WS_INITIAL_PRIMAL_AVG,
                                ws.initial_primal_average_.size(),
                                sizeof(double));
    add_result_array_descriptor(header,
                                cuopt::remote::RESULT_WS_INITIAL_DUAL_AVG,
                                ws.initial_dual_average_.size(),
                                sizeof(double));
    add_result_array_descriptor(
      header, cuopt::remote::RESULT_WS_CURRENT_ATY, ws.current_ATY_.size(), sizeof(double));
    add_result_array_descriptor(
      header, cuopt::remote::RESULT_WS_SUM_PRIMAL, ws.sum_primal_solutions_.size(), sizeof(double));
    add_result_array_descriptor(
      header, cuopt::remote::RESULT_WS_SUM_DUAL, ws.sum_dual_solutions_.size(), sizeof(double));
    add_result_array_descriptor(header,
                                cuopt::remote::RESULT_WS_LAST_RESTART_GAP_PRIMAL,
                                ws.last_restart_duality_gap_primal_solution_.size(),
                                sizeof(double));
    add_result_array_descriptor(header,
                                cuopt::remote::RESULT_WS_LAST_RESTART_GAP_DUAL,
                                ws.last_restart_duality_gap_dual_solution_.size(),
                                sizeof(double));
  }
}

template <typename i_t, typename f_t>
void populate_chunked_result_header_mip(const cpu_mip_solution_t<i_t, f_t>& solution,
                                        cuopt::remote::ChunkedResultHeader* header)
{
  header->set_problem_category(cuopt::remote::MIP);
  header->set_mip_termination_status(to_proto_mip_status(solution.get_termination_status()));
  header->set_mip_error_message(solution.get_error_status().what());
  header->set_mip_objective(solution.get_objective_value());
  header->set_mip_gap(solution.get_mip_gap());
  header->set_solution_bound(solution.get_solution_bound());
  header->set_total_solve_time(solution.get_solve_time());
  header->set_presolve_time(solution.get_presolve_time());
  header->set_max_constraint_violation(solution.get_max_constraint_violation());
  header->set_max_int_violation(solution.get_max_int_violation());
  header->set_max_variable_bound_violation(solution.get_max_variable_bound_violation());
  header->set_nodes(solution.get_num_nodes());
  header->set_simplex_iterations(solution.get_num_simplex_iterations());

  add_result_array_descriptor(header,
                              cuopt::remote::RESULT_MIP_SOLUTION,
                              solution.get_solution_host().size(),
                              sizeof(double));
}

// ============================================================================
// Collect solution arrays as raw bytes
// ============================================================================

template <typename i_t, typename f_t>
std::map<int32_t, std::vector<uint8_t>> collect_lp_solution_arrays(
  const cpu_lp_solution_t<i_t, f_t>& solution)
{
  std::map<int32_t, std::vector<uint8_t>> arrays;

  const auto& primal       = solution.get_primal_solution_host();
  const auto& dual         = solution.get_dual_solution_host();
  const auto& reduced_cost = solution.get_reduced_cost_host();

  if (!primal.empty()) { arrays[cuopt::remote::RESULT_PRIMAL_SOLUTION] = doubles_to_bytes(primal); }
  if (!dual.empty()) { arrays[cuopt::remote::RESULT_DUAL_SOLUTION] = doubles_to_bytes(dual); }
  if (!reduced_cost.empty()) {
    arrays[cuopt::remote::RESULT_REDUCED_COST] = doubles_to_bytes(reduced_cost);
  }

  if (solution.has_warm_start_data()) {
    const auto& ws = solution.get_cpu_pdlp_warm_start_data();
    if (!ws.current_primal_solution_.empty()) {
      arrays[cuopt::remote::RESULT_WS_CURRENT_PRIMAL] =
        doubles_to_bytes(ws.current_primal_solution_);
    }
    if (!ws.current_dual_solution_.empty()) {
      arrays[cuopt::remote::RESULT_WS_CURRENT_DUAL] = doubles_to_bytes(ws.current_dual_solution_);
    }
    if (!ws.initial_primal_average_.empty()) {
      arrays[cuopt::remote::RESULT_WS_INITIAL_PRIMAL_AVG] =
        doubles_to_bytes(ws.initial_primal_average_);
    }
    if (!ws.initial_dual_average_.empty()) {
      arrays[cuopt::remote::RESULT_WS_INITIAL_DUAL_AVG] =
        doubles_to_bytes(ws.initial_dual_average_);
    }
    if (!ws.current_ATY_.empty()) {
      arrays[cuopt::remote::RESULT_WS_CURRENT_ATY] = doubles_to_bytes(ws.current_ATY_);
    }
    if (!ws.sum_primal_solutions_.empty()) {
      arrays[cuopt::remote::RESULT_WS_SUM_PRIMAL] = doubles_to_bytes(ws.sum_primal_solutions_);
    }
    if (!ws.sum_dual_solutions_.empty()) {
      arrays[cuopt::remote::RESULT_WS_SUM_DUAL] = doubles_to_bytes(ws.sum_dual_solutions_);
    }
    if (!ws.last_restart_duality_gap_primal_solution_.empty()) {
      arrays[cuopt::remote::RESULT_WS_LAST_RESTART_GAP_PRIMAL] =
        doubles_to_bytes(ws.last_restart_duality_gap_primal_solution_);
    }
    if (!ws.last_restart_duality_gap_dual_solution_.empty()) {
      arrays[cuopt::remote::RESULT_WS_LAST_RESTART_GAP_DUAL] =
        doubles_to_bytes(ws.last_restart_duality_gap_dual_solution_);
    }
  }

  return arrays;
}

template <typename i_t, typename f_t>
std::map<int32_t, std::vector<uint8_t>> collect_mip_solution_arrays(
  const cpu_mip_solution_t<i_t, f_t>& solution)
{
  std::map<int32_t, std::vector<uint8_t>> arrays;
  const auto& sol_vec = solution.get_solution_host();
  if (!sol_vec.empty()) { arrays[cuopt::remote::RESULT_MIP_SOLUTION] = doubles_to_bytes(sol_vec); }
  return arrays;
}

// ============================================================================
// Chunked result -> solution (client-side)
// ============================================================================

namespace {

template <typename T>
std::vector<T> bytes_to_typed(const std::map<int32_t, std::vector<uint8_t>>& arrays,
                              int32_t field_id)
{
  auto it = arrays.find(field_id);
  if (it == arrays.end() || it->second.empty()) return {};

  const auto& raw = it->second;
  if constexpr (std::is_same_v<T, float>) {
    if (raw.size() % sizeof(double) != 0) return {};
    size_t n = raw.size() / sizeof(double);
    std::vector<double> tmp(n);
    std::memcpy(tmp.data(), raw.data(), n * sizeof(double));
    return std::vector<T>(tmp.begin(), tmp.end());
  } else if constexpr (std::is_same_v<T, double>) {
    if (raw.size() % sizeof(double) != 0) return {};
    size_t n = raw.size() / sizeof(double);
    std::vector<double> v(n);
    std::memcpy(v.data(), raw.data(), n * sizeof(double));
    return v;
  } else {
    if (raw.size() % sizeof(T) != 0) return {};
    size_t n = raw.size() / sizeof(T);
    std::vector<T> v(n);
    std::memcpy(v.data(), raw.data(), n * sizeof(T));
    return v;
  }
}

}  // namespace

template <typename i_t, typename f_t>
cpu_lp_solution_t<i_t, f_t> chunked_result_to_lp_solution(
  const cuopt::remote::ChunkedResultHeader& h,
  const std::map<int32_t, std::vector<uint8_t>>& arrays)
{
  auto primal       = bytes_to_typed<f_t>(arrays, cuopt::remote::RESULT_PRIMAL_SOLUTION);
  auto dual         = bytes_to_typed<f_t>(arrays, cuopt::remote::RESULT_DUAL_SOLUTION);
  auto reduced_cost = bytes_to_typed<f_t>(arrays, cuopt::remote::RESULT_REDUCED_COST);

  auto status    = from_proto_pdlp_status(h.lp_termination_status());
  auto obj       = static_cast<f_t>(h.primal_objective());
  auto dual_obj  = static_cast<f_t>(h.dual_objective());
  auto solve_t   = h.solve_time();
  auto l2_pr     = static_cast<f_t>(h.l2_primal_residual());
  auto l2_dr     = static_cast<f_t>(h.l2_dual_residual());
  auto g         = static_cast<f_t>(h.gap());
  auto iters     = static_cast<i_t>(h.nb_iterations());
  auto solved_by = static_cast<method_t>(h.solved_by());

  auto ws_primal = bytes_to_typed<f_t>(arrays, cuopt::remote::RESULT_WS_CURRENT_PRIMAL);
  if (!ws_primal.empty()) {
    cpu_pdlp_warm_start_data_t<i_t, f_t> ws;
    ws.current_primal_solution_ = std::move(ws_primal);
    ws.current_dual_solution_ = bytes_to_typed<f_t>(arrays, cuopt::remote::RESULT_WS_CURRENT_DUAL);
    ws.initial_primal_average_ =
      bytes_to_typed<f_t>(arrays, cuopt::remote::RESULT_WS_INITIAL_PRIMAL_AVG);
    ws.initial_dual_average_ =
      bytes_to_typed<f_t>(arrays, cuopt::remote::RESULT_WS_INITIAL_DUAL_AVG);
    ws.current_ATY_          = bytes_to_typed<f_t>(arrays, cuopt::remote::RESULT_WS_CURRENT_ATY);
    ws.sum_primal_solutions_ = bytes_to_typed<f_t>(arrays, cuopt::remote::RESULT_WS_SUM_PRIMAL);
    ws.sum_dual_solutions_   = bytes_to_typed<f_t>(arrays, cuopt::remote::RESULT_WS_SUM_DUAL);
    ws.last_restart_duality_gap_primal_solution_ =
      bytes_to_typed<f_t>(arrays, cuopt::remote::RESULT_WS_LAST_RESTART_GAP_PRIMAL);
    ws.last_restart_duality_gap_dual_solution_ =
      bytes_to_typed<f_t>(arrays, cuopt::remote::RESULT_WS_LAST_RESTART_GAP_DUAL);

    ws.initial_primal_weight_         = static_cast<f_t>(h.ws_initial_primal_weight());
    ws.initial_step_size_             = static_cast<f_t>(h.ws_initial_step_size());
    ws.total_pdlp_iterations_         = static_cast<i_t>(h.ws_total_pdlp_iterations());
    ws.total_pdhg_iterations_         = static_cast<i_t>(h.ws_total_pdhg_iterations());
    ws.last_candidate_kkt_score_      = static_cast<f_t>(h.ws_last_candidate_kkt_score());
    ws.last_restart_kkt_score_        = static_cast<f_t>(h.ws_last_restart_kkt_score());
    ws.sum_solution_weight_           = static_cast<f_t>(h.ws_sum_solution_weight());
    ws.iterations_since_last_restart_ = static_cast<i_t>(h.ws_iterations_since_last_restart());

    return cpu_lp_solution_t<i_t, f_t>(std::move(primal),
                                       std::move(dual),
                                       std::move(reduced_cost),
                                       status,
                                       obj,
                                       dual_obj,
                                       solve_t,
                                       l2_pr,
                                       l2_dr,
                                       g,
                                       iters,
                                       solved_by,
                                       std::move(ws));
  }

  return cpu_lp_solution_t<i_t, f_t>(std::move(primal),
                                     std::move(dual),
                                     std::move(reduced_cost),
                                     status,
                                     obj,
                                     dual_obj,
                                     solve_t,
                                     l2_pr,
                                     l2_dr,
                                     g,
                                     iters,
                                     solved_by);
}

template <typename i_t, typename f_t>
cpu_mip_solution_t<i_t, f_t> chunked_result_to_mip_solution(
  const cuopt::remote::ChunkedResultHeader& h,
  const std::map<int32_t, std::vector<uint8_t>>& arrays)
{
  auto sol_vec = bytes_to_typed<f_t>(arrays, cuopt::remote::RESULT_MIP_SOLUTION);

  return cpu_mip_solution_t<i_t, f_t>(std::move(sol_vec),
                                      from_proto_mip_status(h.mip_termination_status()),
                                      static_cast<f_t>(h.mip_objective()),
                                      static_cast<f_t>(h.mip_gap()),
                                      static_cast<f_t>(h.solution_bound()),
                                      h.total_solve_time(),
                                      h.presolve_time(),
                                      static_cast<f_t>(h.max_constraint_violation()),
                                      static_cast<f_t>(h.max_int_violation()),
                                      static_cast<f_t>(h.max_variable_bound_violation()),
                                      static_cast<i_t>(h.nodes()),
                                      static_cast<i_t>(h.simplex_iterations()));
}

// ============================================================================
// Build full protobuf from stored header + arrays (server-side GetResult RPC)
// ============================================================================

template <typename i_t, typename f_t>
void build_lp_solution_proto(const cuopt::remote::ChunkedResultHeader& header,
                             const std::map<int32_t, std::vector<uint8_t>>& arrays,
                             cuopt::remote::LPSolution* proto)
{
  auto cpu_sol = chunked_result_to_lp_solution<i_t, f_t>(header, arrays);
  map_lp_solution_to_proto(cpu_sol, proto);
}

template <typename i_t, typename f_t>
void build_mip_solution_proto(const cuopt::remote::ChunkedResultHeader& header,
                              const std::map<int32_t, std::vector<uint8_t>>& arrays,
                              cuopt::remote::MIPSolution* proto)
{
  auto cpu_sol = chunked_result_to_mip_solution<i_t, f_t>(header, arrays);
  map_mip_solution_to_proto(cpu_sol, proto);
}

// Explicit template instantiations
#if CUOPT_INSTANTIATE_FLOAT
template void map_lp_solution_to_proto(const cpu_lp_solution_t<int32_t, float>& solution,
                                       cuopt::remote::LPSolution* pb_solution);
template cpu_lp_solution_t<int32_t, float> map_proto_to_lp_solution(
  const cuopt::remote::LPSolution& pb_solution);
template void map_mip_solution_to_proto(const cpu_mip_solution_t<int32_t, float>& solution,
                                        cuopt::remote::MIPSolution* pb_solution);
template cpu_mip_solution_t<int32_t, float> map_proto_to_mip_solution(
  const cuopt::remote::MIPSolution& pb_solution);
template size_t estimate_lp_solution_proto_size(const cpu_lp_solution_t<int32_t, float>& solution);
template size_t estimate_mip_solution_proto_size(
  const cpu_mip_solution_t<int32_t, float>& solution);
template void populate_chunked_result_header_lp(const cpu_lp_solution_t<int32_t, float>& solution,
                                                cuopt::remote::ChunkedResultHeader* header);
template void populate_chunked_result_header_mip(const cpu_mip_solution_t<int32_t, float>& solution,
                                                 cuopt::remote::ChunkedResultHeader* header);
template std::map<int32_t, std::vector<uint8_t>> collect_lp_solution_arrays(
  const cpu_lp_solution_t<int32_t, float>& solution);
template std::map<int32_t, std::vector<uint8_t>> collect_mip_solution_arrays(
  const cpu_mip_solution_t<int32_t, float>& solution);
template cpu_lp_solution_t<int32_t, float> chunked_result_to_lp_solution(
  const cuopt::remote::ChunkedResultHeader& header,
  const std::map<int32_t, std::vector<uint8_t>>& arrays);
template cpu_mip_solution_t<int32_t, float> chunked_result_to_mip_solution(
  const cuopt::remote::ChunkedResultHeader& header,
  const std::map<int32_t, std::vector<uint8_t>>& arrays);
template void build_lp_solution_proto<int32_t, float>(
  const cuopt::remote::ChunkedResultHeader& header,
  const std::map<int32_t, std::vector<uint8_t>>& arrays,
  cuopt::remote::LPSolution* proto);
template void build_mip_solution_proto<int32_t, float>(
  const cuopt::remote::ChunkedResultHeader& header,
  const std::map<int32_t, std::vector<uint8_t>>& arrays,
  cuopt::remote::MIPSolution* proto);
#endif

#if CUOPT_INSTANTIATE_DOUBLE
template void map_lp_solution_to_proto(const cpu_lp_solution_t<int32_t, double>& solution,
                                       cuopt::remote::LPSolution* pb_solution);
template cpu_lp_solution_t<int32_t, double> map_proto_to_lp_solution(
  const cuopt::remote::LPSolution& pb_solution);
template void map_mip_solution_to_proto(const cpu_mip_solution_t<int32_t, double>& solution,
                                        cuopt::remote::MIPSolution* pb_solution);
template cpu_mip_solution_t<int32_t, double> map_proto_to_mip_solution(
  const cuopt::remote::MIPSolution& pb_solution);
template size_t estimate_lp_solution_proto_size(const cpu_lp_solution_t<int32_t, double>& solution);
template size_t estimate_mip_solution_proto_size(
  const cpu_mip_solution_t<int32_t, double>& solution);
template void populate_chunked_result_header_lp(const cpu_lp_solution_t<int32_t, double>& solution,
                                                cuopt::remote::ChunkedResultHeader* header);
template void populate_chunked_result_header_mip(
  const cpu_mip_solution_t<int32_t, double>& solution, cuopt::remote::ChunkedResultHeader* header);
template std::map<int32_t, std::vector<uint8_t>> collect_lp_solution_arrays(
  const cpu_lp_solution_t<int32_t, double>& solution);
template std::map<int32_t, std::vector<uint8_t>> collect_mip_solution_arrays(
  const cpu_mip_solution_t<int32_t, double>& solution);
template cpu_lp_solution_t<int32_t, double> chunked_result_to_lp_solution(
  const cuopt::remote::ChunkedResultHeader& header,
  const std::map<int32_t, std::vector<uint8_t>>& arrays);
template cpu_mip_solution_t<int32_t, double> chunked_result_to_mip_solution(
  const cuopt::remote::ChunkedResultHeader& header,
  const std::map<int32_t, std::vector<uint8_t>>& arrays);
template void build_lp_solution_proto<int32_t, double>(
  const cuopt::remote::ChunkedResultHeader& header,
  const std::map<int32_t, std::vector<uint8_t>>& arrays,
  cuopt::remote::LPSolution* proto);
template void build_mip_solution_proto<int32_t, double>(
  const cuopt::remote::ChunkedResultHeader& header,
  const std::map<int32_t, std::vector<uint8_t>>& arrays,
  cuopt::remote::MIPSolution* proto);
#endif

}  // namespace cuopt::linear_programming
