/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */
// Original-MPS LP relaxation -> native cuOpt presolve/solve/postsolve.
// The driver never calls or links the standalone PSLP API.
#include <cuopt/mathematical_optimization/cuopt_c.h>
#include <chrono>
#include <cstdlib>
#include <type_traits>
#include "compressible_rmm.hpp"
#include "original_checks.hpp"

using Clock = std::chrono::steady_clock;
static double seconds(Clock::time_point a, Clock::time_point b)
{
  return std::chrono::duration<double>(b - a).count();
}
static void check(cuopt_int_t code, const char* call)
{
  if (code != 0) throw std::runtime_error(std::string(call) + " returned " + std::to_string(code));
}
#define API(call) check((call), #call)

struct Resources {
  cuOptOptimizationProblem original = nullptr, lp = nullptr;
  cuOptSolverSettings settings = nullptr;
  cuOptSolution solution       = nullptr;
  ~Resources()
  {
    if (solution) cuOptDestroySolution(&solution);
    if (settings) cuOptDestroySolverSettings(&settings);
    if (lp) cuOptDestroyProblem(&lp);
    if (original) cuOptDestroyProblem(&original);
  }
};

static int integer_argument(const std::string& text)
{
  size_t used     = 0;
  const int value = std::stoi(text, &used);
  if (used != text.size()) throw std::runtime_error("Invalid integer argument: " + text);
  return value;
}
static double real_argument(const std::string& text)
{
  size_t used        = 0;
  const double value = std::stod(text, &used);
  if (used != text.size() || !std::isfinite(value))
    throw std::runtime_error("Invalid real argument: " + text);
  return value;
}
static void set_integer(cuOptSolverSettings settings, const char* name, int expected)
{
  API(cuOptSetIntegerParameter(settings, name, expected));
  cuopt_int_t returned = -1234567;
  API(cuOptGetIntegerParameter(settings, name, &returned));
  if (returned != expected)
    throw std::runtime_error(std::string("Parameter readback mismatch: ") + name);
}
static void set_real(cuOptSolverSettings settings, const char* name, double expected)
{
  API(cuOptSetFloatParameter(settings, name, expected));
  cuopt_float_t returned = std::numeric_limits<double>::quiet_NaN();
  API(cuOptGetFloatParameter(settings, name, &returned));
  if (returned != expected)
    throw std::runtime_error(std::string("Parameter readback mismatch: ") + name);
}

int main(int argc, char** argv)
{
  static_assert(std::is_same<cuopt_float_t, double>::value, "Expected FP64 cuOpt API");
  static_assert(std::is_same<cuopt_int_t, int>::value, "Expected int32 cuOpt API");
  const auto program_start = Clock::now();
  std::string file;
  int method = 0, threads = 8;
  double time_limit = 300, relative_tolerance = 1e-6, absolute_tolerance = 1e-4;
  try {
    for (int i = 1; i < argc; ++i) {
      const std::string arg = argv[i];
      auto value            = [&]() -> std::string {
        if (++i == argc) throw std::runtime_error("Missing value for " + arg);
        return argv[i];
      };
      if (arg == "--method")
        method = integer_argument(value());
      else if (arg == "--threads")
        threads = integer_argument(value());
      else if (arg == "--time-limit")
        time_limit = real_argument(value());
      else if (arg == "--relative-tolerance")
        relative_tolerance = real_argument(value());
      else if (arg == "--absolute-tolerance")
        absolute_tolerance = real_argument(value());
      else if (arg == "--help") {
        std::cout << "Usage: native_driver MODEL.mps[.gz] [--method 0|1] [--threads 8] "
                     "[--time-limit 300] [--relative-tolerance 1e-6] [--absolute-tolerance 1e-4]\n"
                     "C API methods: 0=concurrent/default, 1=PDLP. Native presolve enabled.\n";
        return 0;
      } else if (arg.empty() || arg.front() == '-' || !file.empty())
        throw std::runtime_error("Unknown or extra argument: " + arg);
      else
        file = arg;
    }
    if (file.empty() || (method != 0 && method != 1) || threads < 1 || time_limit <= 0 ||
        relative_tolerance < 0 || relative_tolerance > 0.1 || absolute_tolerance < 0 ||
        absolute_tolerance > 0.1)
      throw std::runtime_error("Invalid arguments; use --help");
    for (const char* name : {"LPFEAS_DUMP_DIR",
                             "LPFEAS_PARAM_FILE",
                             "LPFEAS_IMPORT_MODEL_DIR",
                             "LPFEAS_IMPORT_SOLUTION_DIR"}) {
      if (const char* requested = std::getenv(name))
        if (*requested)
          throw std::runtime_error(std::string("Unsupported diagnostic environment: ") + name);
    }

    // All cuOpt objects are destroyed before the allocator resource is restored.
    lpfeas_compression::allocator_scope_t allocator;
    Model a;
    Resources r;
    if (cuOptGetIntSize() != sizeof(int) || cuOptGetFloatSize() != sizeof(double))
      throw std::runtime_error("cuOpt runtime numeric types do not match driver");
    const auto read_start = Clock::now();
    API(cuOptReadProblem(file.c_str(), &r.original));
    const auto read_end = Clock::now();
    API(cuOptGetNumConstraints(r.original, &a.m));
    API(cuOptGetNumVariables(r.original, &a.n));
    API(cuOptGetNumNonZeros(r.original, &a.nnz));
    API(cuOptGetObjectiveSense(r.original, &a.sense));
    API(cuOptGetObjectiveOffset(r.original, &a.offset));
    if (a.m < 0 || a.n < 0 || a.nnz < 0 || (a.sense != 1 && a.sense != -1))
      throw std::runtime_error("Invalid original problem dimensions or sense");
    a.ap.resize(size_t(a.m) + 1);
    a.ai.resize(std::max(1, a.nnz));
    a.ax.resize(std::max(1, a.nnz));
    a.c.resize(std::max(1, a.n));
    a.lb.resize(std::max(1, a.n));
    a.ub.resize(std::max(1, a.n));
    a.types.resize(a.n);
    a.lhs.resize(std::max(1, a.m));
    a.rhs.resize(std::max(1, a.m));
    API(cuOptGetConstraintMatrix(r.original, a.ap.data(), a.ai.data(), a.ax.data()));
    if (a.n) {
      API(cuOptGetObjectiveCoefficients(r.original, a.c.data()));
      API(cuOptGetVariableLowerBounds(r.original, a.lb.data()));
      API(cuOptGetVariableUpperBounds(r.original, a.ub.data()));
      API(cuOptGetVariableTypes(r.original, a.types.data()));
    }
    if (a.m) {
      API(cuOptGetConstraintLowerBounds(r.original, a.lhs.data()));
      API(cuOptGetConstraintUpperBounds(r.original, a.rhs.data()));
    }
    const auto input_nnz = a.nnz;
    a.canonicalize();
    // cuOptCreateRangedProblem rejects null array pointers even for zero lengths.
    if (a.ai.empty()) a.ai.resize(1);
    if (a.ax.empty()) a.ax.resize(1);
    cuOptDestroyProblem(&r.original);
    const auto canonical_end = Clock::now();

    // NULL variable_types creates an LP with all variables continuous. Passing
    // the parsed original problem directly would dispatch integer-marked MPS to MIP.
    API(cuOptCreateRangedProblem(a.m,
                                 a.n,
                                 a.sense,
                                 a.offset,
                                 a.c.data(),
                                 a.ap.data(),
                                 a.ai.data(),
                                 a.ax.data(),
                                 a.lhs.data(),
                                 a.rhs.data(),
                                 a.lb.data(),
                                 a.ub.data(),
                                 nullptr,
                                 &r.lp));
    cuopt_int_t is_mip = 1;
    API(cuOptIsMIP(r.lp, &is_mip));
    if (is_mip) throw std::runtime_error("Recreated problem is not the continuous LP relaxation");
    API(cuOptCreateSolverSettings(&r.settings));
    set_integer(r.settings, CUOPT_PRESOLVE, CUOPT_PRESOLVE_PSLP);
    set_integer(r.settings, CUOPT_METHOD, method);
    set_integer(r.settings, CUOPT_NUM_GPUS, 1);
    set_integer(r.settings, CUOPT_NUM_CPU_THREADS, threads);
    set_integer(r.settings, CUOPT_FOLDING, -1);
    set_integer(r.settings, CUOPT_CROSSOVER, 0);
    set_integer(r.settings, CUOPT_LOG_TO_CONSOLE, 1);
    for (auto name : {CUOPT_RELATIVE_PRIMAL_TOLERANCE,
                      CUOPT_RELATIVE_DUAL_TOLERANCE,
                      CUOPT_RELATIVE_GAP_TOLERANCE})
      set_real(r.settings, name, relative_tolerance);
    for (auto name : {CUOPT_ABSOLUTE_PRIMAL_TOLERANCE,
                      CUOPT_ABSOLUTE_DUAL_TOLERANCE,
                      CUOPT_ABSOLUTE_GAP_TOLERANCE})
      set_real(r.settings, name, absolute_tolerance);
    // Native cuOpt owns the complete presolve/solve budget. Driver model read,
    // reconstruction and validation are measured separately, never subtracted.
    set_real(r.settings, CUOPT_TIME_LIMIT, time_limit);
    const auto solve_start = Clock::now();
    API(cuOptSolve(r.lp, r.settings, &r.solution));
    const auto solve_end = Clock::now();
    allocator.report("after_solve");
    cuopt_int_t error = 0, status = 0;
    API(cuOptGetErrorStatus(r.solution, &error));
    if (error) {
      char message[4096] = {};
      API(cuOptGetErrorString(r.solution, message, sizeof(message)));
      throw std::runtime_error("cuOpt solution error " + std::to_string(error) + ": " + message);
    }
    API(cuOptGetTerminationStatus(r.solution, &status));
    double reported_time = 0;
    API(cuOptGetSolveTime(r.solution, &reported_time));

    Json j;
    j.str("record_type", "cuopt_builtin_presolve_result");
    j.num("schema_version", 1);
    j.str("file", file);
    j.str("driver_variant", "original_mps_native_presolve");
    j.str("presolve_owner", "cuopt");
    j.flag("external_pslp_invoked", false);
    j.flag("original_coordinates_preserved", true);
    j.flag("lp_relaxation_verified", !is_mip);
    j.str("objective_handling",
          "Original objective sense and offset passed to native cuOpt; signed-minimum conversion "
          "is diagnostic-only");
    j.str("native_timer_scope",
          "cuOptGetSolveTime: native single-GPU shared timer includes presolve and solver setup; "
          "generally excludes outer native postsolve");
    j.str("api_timer_scope",
          "cuOptSolve wall time, including native presolve and postsolve; excludes read, "
          "recreation, fetch, validation, allocator initialization and teardown");
    j.str("budget_scope",
          "CUOPT_TIME_LIMIT passed unchanged to native cuOpt; API/process wall time is not a "
          "timeout classifier");
    j.str("allocator_timing_scope",
          "Initialization and teardown are separate ALLOCATOR_JSON records; monitor process wall "
          "includes both");
    j.num("method", method);
    j.num("solver_num_gpus", 1);
    j.num("solver_cpu_threads", threads);
    j.flag("solver_presolve", true);
    j.num("solver_presolve_parameter", CUOPT_PRESOLVE_PSLP);
    j.flag("solver_crossover", false);
    j.num("solver_barrier_folding", -1);
    j.flag("settings_readback_verified", true);
    j.num("solver_time_limit", time_limit);
    j.num("solver_relative_tolerances", relative_tolerance);
    j.num("relative_primal_tolerance", relative_tolerance);
    j.num("relative_dual_tolerance", relative_tolerance);
    j.num("relative_gap_tolerance", relative_tolerance);
    j.num("solver_absolute_tolerances", absolute_tolerance);
    j.num("comparison_diagnostic_relative_tolerance", 1e-6);
    j.num("comparison_diagnostic_absolute_tolerance", 1e-4);
    j.num("comparison_strict_diagnostic_tolerance", 1e-6);
    j.flag("diagnostic_dump_enabled", false);
    j.flag("diagnostic_external_solution_import", false);
    j.num("diagnostic_parameter_override_count", 0);
    j.num("original_rows", a.m);
    j.num("original_cols", a.n);
    j.num("input_nnz", input_nnz);
    j.num("original_nnz", a.nnz);
    j.num("original_objective_sense", a.sense);
    j.num("original_objective_offset", a.offset);
    j.num("integer_markers_relaxed", std::count_if(a.types.begin(), a.types.end(), [](char type) {
            return type != CUOPT_CONTINUOUS;
          }));
    j.num("read_wall_seconds", seconds(read_start, read_end));
    j.num("extract_canonicalize_wall_seconds", seconds(read_end, canonical_end));
    j.num("solver_setup_wall_seconds", seconds(canonical_end, solve_start));
    j.num("solve_api_wall_seconds", seconds(solve_start, solve_end));
    j.num("cuopt_reported_seconds", reported_time);
    j.num("cuopt_termination_status", status);
    j.flag("cuopt_termination_status_available", true);
    j.flag("cuopt_solve_invoked", true);

    const bool can_fetch_point = status != CUOPT_TERMINATION_STATUS_NO_TERMINATION &&
                                 status != CUOPT_TERMINATION_STATUS_INFEASIBLE &&
                                 status != CUOPT_TERMINATION_STATUS_UNBOUNDED &&
                                 status != CUOPT_TERMINATION_STATUS_UNBOUNDED_OR_INFEASIBLE &&
                                 status != CUOPT_TERMINATION_STATUS_NUMERICAL_ERROR;
    if (!can_fetch_point) {
      j.str("outcome", "solver_terminal_without_validated_point");
      j.flag("validation_available", false);
      j.flag("validation_finite", false);
      j.num("total_wall_seconds", seconds(program_start, Clock::now()));
      j.emit();
      return 0;
    }
    const auto fetch_start    = Clock::now();
    double reported_objective = 0, reported_dual_objective = 0;
    API(cuOptGetObjectiveValue(r.solution, &reported_objective));
    API(cuOptGetDualObjectiveValue(r.solution, &reported_dual_objective));
    std::vector<double> x(std::max(1, a.n), 0), y(std::max(1, a.m), 0), z(std::max(1, a.n), 0);
    if (a.n) {
      API(cuOptGetPrimalSolution(r.solution, x.data()));
      API(cuOptGetReducedCosts(r.solution, z.data()));
    }
    if (a.m) API(cuOptGetDualSolution(r.solution, y.data()));
    const auto fetch_end = Clock::now();
    j.num("solution_fetch_wall_seconds", seconds(fetch_start, fetch_end));
    j.num("cuopt_reported_objective", reported_objective);
    j.num("cuopt_reported_dual_objective", reported_dual_objective);
    j.num("cuopt_reported_signed_min_objective", a.sense * reported_objective);
    j.num("cuopt_reported_signed_min_dual_objective", a.sense * reported_dual_objective);
    j.str("outcome",
          status == CUOPT_TERMINATION_STATUS_OPTIMAL ? "native_optimal_original_solution"
                                                     : "native_nonoptimal_original_point");
    j.flag("validation_available", true);
    j.flag("benchmark_acceptance_reproduced", false);
    const auto validation_start = Clock::now();
    signed_minimum_diagnostics(a, y, z);
    const Solution original_solution{x.data(), y.data(), z.data(), size_t(a.n), size_t(a.m)};
    validate(a, original_solution, j, absolute_tolerance, relative_tolerance);
    j.num("validation_wall_seconds", seconds(validation_start, Clock::now()));
    j.num("total_wall_seconds", seconds(program_start, Clock::now()));
    j.emit();
    return 0;
  } catch (const std::exception& error) {
    Json j;
    j.str("record_type", "cuopt_builtin_presolve_error");
    j.num("schema_version", 1);
    j.str("file", file);
    j.str("error", error.what());
    j.num("total_wall_seconds", seconds(program_start, Clock::now()));
    j.emit();
    return 1;
  }
}
