/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved. SPDX-License-Identifier: Apache-2.0
 */

#include "grpc_settings_mapper.hpp"

#include <cuopt/linear_programming/constants.h>
#include <cuopt_remote.pb.h>
#include <cuopt/linear_programming/mip/solver_settings.hpp>
#include <cuopt/linear_programming/pdlp/solver_settings.hpp>
#include <cuopt/linear_programming/solver_settings.hpp>

#include <limits>
#include <stdexcept>
#include <string>

namespace cuopt::linear_programming {

namespace {

// Convert cuOpt pdlp_solver_mode_t to protobuf enum
cuopt::remote::PDLPSolverMode to_proto_pdlp_mode(pdlp_solver_mode_t mode)
{
  switch (mode) {
    case pdlp_solver_mode_t::Stable1: return cuopt::remote::Stable1;
    case pdlp_solver_mode_t::Stable2: return cuopt::remote::Stable2;
    case pdlp_solver_mode_t::Methodical1: return cuopt::remote::Methodical1;
    case pdlp_solver_mode_t::Fast1: return cuopt::remote::Fast1;
    case pdlp_solver_mode_t::Stable3: return cuopt::remote::Stable3;
  }
  throw std::invalid_argument("Unknown pdlp_solver_mode_t: " +
                              std::to_string(static_cast<int>(mode)));
}

// Convert protobuf enum to cuOpt pdlp_solver_mode_t
pdlp_solver_mode_t from_proto_pdlp_mode(cuopt::remote::PDLPSolverMode mode)
{
  switch (mode) {
    case cuopt::remote::Stable1: return pdlp_solver_mode_t::Stable1;
    case cuopt::remote::Stable2: return pdlp_solver_mode_t::Stable2;
    case cuopt::remote::Methodical1: return pdlp_solver_mode_t::Methodical1;
    case cuopt::remote::Fast1: return pdlp_solver_mode_t::Fast1;
    case cuopt::remote::Stable3: return pdlp_solver_mode_t::Stable3;
  }
  throw std::invalid_argument("Unknown PDLPSolverMode: " + std::to_string(static_cast<int>(mode)));
}

// Convert cuOpt method_t to protobuf enum
cuopt::remote::LPMethod to_proto_method(method_t method)
{
  switch (method) {
    case method_t::Concurrent: return cuopt::remote::Concurrent;
    case method_t::PDLP: return cuopt::remote::PDLP;
    case method_t::DualSimplex: return cuopt::remote::DualSimplex;
    case method_t::Barrier: return cuopt::remote::Barrier;
  }
  throw std::invalid_argument("Unknown method_t: " + std::to_string(static_cast<int>(method)));
}

// Convert protobuf enum to cuOpt method_t
method_t from_proto_method(cuopt::remote::LPMethod method)
{
  switch (method) {
    case cuopt::remote::Concurrent: return method_t::Concurrent;
    case cuopt::remote::PDLP: return method_t::PDLP;
    case cuopt::remote::DualSimplex: return method_t::DualSimplex;
    case cuopt::remote::Barrier: return method_t::Barrier;
  }
  throw std::invalid_argument("Unknown LPMethod: " + std::to_string(static_cast<int>(method)));
}

}  // anonymous namespace

template <typename i_t, typename f_t>
void map_pdlp_settings_to_proto(const pdlp_solver_settings_t<i_t, f_t>& settings,
                                cuopt::remote::PDLPSolverSettings* pb_settings)
{
  // Termination tolerances (all names match cuOpt API)
  pb_settings->set_absolute_gap_tolerance(settings.tolerances.absolute_gap_tolerance);
  pb_settings->set_relative_gap_tolerance(settings.tolerances.relative_gap_tolerance);
  pb_settings->set_primal_infeasible_tolerance(settings.tolerances.primal_infeasible_tolerance);
  pb_settings->set_dual_infeasible_tolerance(settings.tolerances.dual_infeasible_tolerance);
  pb_settings->set_absolute_dual_tolerance(settings.tolerances.absolute_dual_tolerance);
  pb_settings->set_relative_dual_tolerance(settings.tolerances.relative_dual_tolerance);
  pb_settings->set_absolute_primal_tolerance(settings.tolerances.absolute_primal_tolerance);
  pb_settings->set_relative_primal_tolerance(settings.tolerances.relative_primal_tolerance);

  // Limits
  pb_settings->set_time_limit(settings.time_limit);
  // Avoid emitting a huge number when the iteration limit is the library default.
  // Use -1 sentinel for "unset/use server defaults".
  if (settings.iteration_limit == std::numeric_limits<i_t>::max()) {
    pb_settings->set_iteration_limit(-1);
  } else {
    pb_settings->set_iteration_limit(static_cast<int64_t>(settings.iteration_limit));
  }

  // Solver configuration
  pb_settings->set_log_to_console(settings.log_to_console);
  pb_settings->set_detect_infeasibility(settings.detect_infeasibility);
  pb_settings->set_strict_infeasibility(settings.strict_infeasibility);
  pb_settings->set_pdlp_solver_mode(to_proto_pdlp_mode(settings.pdlp_solver_mode));
  pb_settings->set_method(to_proto_method(settings.method));
  pb_settings->set_presolver(static_cast<int32_t>(settings.presolver));
  pb_settings->set_dual_postsolve(settings.dual_postsolve);
  pb_settings->set_crossover(settings.crossover);
  pb_settings->set_num_gpus(settings.num_gpus);

  pb_settings->set_per_constraint_residual(settings.per_constraint_residual);
  pb_settings->set_cudss_deterministic(settings.cudss_deterministic);
  pb_settings->set_folding(settings.folding);
  pb_settings->set_augmented(settings.augmented);
  pb_settings->set_dualize(settings.dualize);
  pb_settings->set_ordering(settings.ordering);
  pb_settings->set_barrier_dual_initial_point(settings.barrier_dual_initial_point);
  pb_settings->set_eliminate_dense_columns(settings.eliminate_dense_columns);
  pb_settings->set_pdlp_precision(static_cast<int32_t>(settings.pdlp_precision));
  pb_settings->set_save_best_primal_so_far(settings.save_best_primal_so_far);
  pb_settings->set_first_primal_feasible(settings.first_primal_feasible);
}

template <typename i_t, typename f_t>
void map_proto_to_pdlp_settings(const cuopt::remote::PDLPSolverSettings& pb_settings,
                                pdlp_solver_settings_t<i_t, f_t>& settings)
{
  // Termination tolerances (all names match cuOpt API)
  settings.tolerances.absolute_gap_tolerance      = pb_settings.absolute_gap_tolerance();
  settings.tolerances.relative_gap_tolerance      = pb_settings.relative_gap_tolerance();
  settings.tolerances.primal_infeasible_tolerance = pb_settings.primal_infeasible_tolerance();
  settings.tolerances.dual_infeasible_tolerance   = pb_settings.dual_infeasible_tolerance();
  settings.tolerances.absolute_dual_tolerance     = pb_settings.absolute_dual_tolerance();
  settings.tolerances.relative_dual_tolerance     = pb_settings.relative_dual_tolerance();
  settings.tolerances.absolute_primal_tolerance   = pb_settings.absolute_primal_tolerance();
  settings.tolerances.relative_primal_tolerance   = pb_settings.relative_primal_tolerance();

  // Limits
  settings.time_limit = pb_settings.time_limit();
  // proto3 defaults numeric fields to 0; treat negative iteration_limit as "unset"
  // so the server keeps the library default (typically max()).
  if (pb_settings.iteration_limit() >= 0) {
    const auto limit         = pb_settings.iteration_limit();
    settings.iteration_limit = (limit > static_cast<int64_t>(std::numeric_limits<i_t>::max()))
                                 ? std::numeric_limits<i_t>::max()
                                 : static_cast<i_t>(limit);
  }

  // Solver configuration
  settings.log_to_console       = pb_settings.log_to_console();
  settings.detect_infeasibility = pb_settings.detect_infeasibility();
  settings.strict_infeasibility = pb_settings.strict_infeasibility();
  settings.pdlp_solver_mode     = from_proto_pdlp_mode(pb_settings.pdlp_solver_mode());
  settings.method               = from_proto_method(pb_settings.method());
  {
    auto pv            = pb_settings.presolver();
    settings.presolver = (pv >= CUOPT_PRESOLVE_DEFAULT && pv <= CUOPT_PRESOLVE_PSLP)
                           ? static_cast<presolver_t>(pv)
                           : presolver_t::Default;
  }
  settings.dual_postsolve = pb_settings.dual_postsolve();
  settings.crossover      = pb_settings.crossover();
  settings.num_gpus       = pb_settings.num_gpus();

  settings.per_constraint_residual    = pb_settings.per_constraint_residual();
  settings.cudss_deterministic        = pb_settings.cudss_deterministic();
  settings.folding                    = pb_settings.folding();
  settings.augmented                  = pb_settings.augmented();
  settings.dualize                    = pb_settings.dualize();
  settings.ordering                   = pb_settings.ordering();
  settings.barrier_dual_initial_point = pb_settings.barrier_dual_initial_point();
  settings.eliminate_dense_columns    = pb_settings.eliminate_dense_columns();
  {
    auto pv = pb_settings.pdlp_precision();
    settings.pdlp_precision =
      (pv >= CUOPT_PDLP_DEFAULT_PRECISION && pv <= CUOPT_PDLP_MIXED_PRECISION)
        ? static_cast<pdlp_precision_t>(pv)
        : pdlp_precision_t::DefaultPrecision;
  }
  settings.save_best_primal_so_far = pb_settings.save_best_primal_so_far();
  settings.first_primal_feasible   = pb_settings.first_primal_feasible();
}

template <typename i_t, typename f_t>
void map_mip_settings_to_proto(const mip_solver_settings_t<i_t, f_t>& settings,
                               cuopt::remote::MIPSolverSettings* pb_settings)
{
  // Limits
  pb_settings->set_time_limit(settings.time_limit);

  // Tolerances (all names match cuOpt API)
  pb_settings->set_relative_mip_gap(settings.tolerances.relative_mip_gap);
  pb_settings->set_absolute_mip_gap(settings.tolerances.absolute_mip_gap);
  pb_settings->set_integrality_tolerance(settings.tolerances.integrality_tolerance);
  pb_settings->set_absolute_tolerance(settings.tolerances.absolute_tolerance);
  pb_settings->set_relative_tolerance(settings.tolerances.relative_tolerance);
  pb_settings->set_presolve_absolute_tolerance(settings.tolerances.presolve_absolute_tolerance);

  // Solver configuration
  pb_settings->set_log_to_console(settings.log_to_console);
  pb_settings->set_heuristics_only(settings.heuristics_only);
  pb_settings->set_num_cpu_threads(settings.num_cpu_threads);
  pb_settings->set_num_gpus(settings.num_gpus);
  pb_settings->set_presolver(static_cast<int32_t>(settings.presolver));
  pb_settings->set_mip_scaling(settings.mip_scaling);
}

template <typename i_t, typename f_t>
void map_proto_to_mip_settings(const cuopt::remote::MIPSolverSettings& pb_settings,
                               mip_solver_settings_t<i_t, f_t>& settings)
{
  // Limits
  settings.time_limit = pb_settings.time_limit();

  // Tolerances (all names match cuOpt API)
  settings.tolerances.relative_mip_gap            = pb_settings.relative_mip_gap();
  settings.tolerances.absolute_mip_gap            = pb_settings.absolute_mip_gap();
  settings.tolerances.integrality_tolerance       = pb_settings.integrality_tolerance();
  settings.tolerances.absolute_tolerance          = pb_settings.absolute_tolerance();
  settings.tolerances.relative_tolerance          = pb_settings.relative_tolerance();
  settings.tolerances.presolve_absolute_tolerance = pb_settings.presolve_absolute_tolerance();

  // Solver configuration
  settings.log_to_console  = pb_settings.log_to_console();
  settings.heuristics_only = pb_settings.heuristics_only();
  settings.num_cpu_threads = pb_settings.num_cpu_threads();
  settings.num_gpus        = pb_settings.num_gpus();
  {
    auto pv            = pb_settings.presolver();
    settings.presolver = (pv >= CUOPT_PRESOLVE_DEFAULT && pv <= CUOPT_PRESOLVE_PSLP)
                           ? static_cast<presolver_t>(pv)
                           : presolver_t::Default;
  }
  settings.mip_scaling = pb_settings.mip_scaling();
}

// Explicit template instantiations
#if CUOPT_INSTANTIATE_FLOAT
template void map_pdlp_settings_to_proto(const pdlp_solver_settings_t<int32_t, float>& settings,
                                         cuopt::remote::PDLPSolverSettings* pb_settings);
template void map_proto_to_pdlp_settings(const cuopt::remote::PDLPSolverSettings& pb_settings,
                                         pdlp_solver_settings_t<int32_t, float>& settings);
template void map_mip_settings_to_proto(const mip_solver_settings_t<int32_t, float>& settings,
                                        cuopt::remote::MIPSolverSettings* pb_settings);
template void map_proto_to_mip_settings(const cuopt::remote::MIPSolverSettings& pb_settings,
                                        mip_solver_settings_t<int32_t, float>& settings);
#endif

#if CUOPT_INSTANTIATE_DOUBLE
template void map_pdlp_settings_to_proto(const pdlp_solver_settings_t<int32_t, double>& settings,
                                         cuopt::remote::PDLPSolverSettings* pb_settings);
template void map_proto_to_pdlp_settings(const cuopt::remote::PDLPSolverSettings& pb_settings,
                                         pdlp_solver_settings_t<int32_t, double>& settings);
template void map_mip_settings_to_proto(const mip_solver_settings_t<int32_t, double>& settings,
                                        cuopt::remote::MIPSolverSettings* pb_settings);
template void map_proto_to_mip_settings(const cuopt::remote::MIPSolverSettings& pb_settings,
                                        mip_solver_settings_t<int32_t, double>& settings);
#endif

}  // namespace cuopt::linear_programming
