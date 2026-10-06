/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "grpc_settings_mapper.hpp"

#include <cuopt/export.hpp>

#include <cuopt/mathematical_optimization/constants.h>
#include <cuopt_remote.pb.h>
#include <cuopt/mathematical_optimization/mip/solver_settings.hpp>
#include <cuopt/mathematical_optimization/pdlp/solver_settings.hpp>
#include <cuopt/mathematical_optimization/solver_settings.hpp>

#include <cmath>
#include <cstddef>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace cuopt::mathematical_optimization {

namespace {
#include "generated_enum_converters_settings.inc"

template <typename f_t>
std::string format_parameter_float(f_t value)
{
  if (std::isnan(value)) { return "nan"; }
  if (std::isinf(value)) { return std::signbit(value) ? "-inf" : "inf"; }
  std::ostringstream os;
  os.precision(std::numeric_limits<f_t>::max_digits10);
  os << value;
  return os.str();
}

template <typename f_t>
void copy_repeated(const google::protobuf::RepeatedField<double>& in, std::vector<f_t>& out)
{
  out.assign(in.begin(), in.end());
}

template <typename i_t, typename f_t>
void write_settings_warm_start(const pdlp_solver_settings_t<i_t, f_t>& settings,
                               cuopt::remote::PDLPSolverSettings* pb_settings)
{
  const auto& ws = settings.get_cpu_pdlp_warm_start_data();
  if (!ws.is_populated()) { return; }
  auto* pb_ws = pb_settings->mutable_warm_start_data();
  for (const auto& v : ws.current_primal_solution_) {
    pb_ws->add_current_primal_solution(static_cast<double>(v));
  }
  for (const auto& v : ws.current_dual_solution_) {
    pb_ws->add_current_dual_solution(static_cast<double>(v));
  }
  for (const auto& v : ws.initial_primal_average_) {
    pb_ws->add_initial_primal_average(static_cast<double>(v));
  }
  for (const auto& v : ws.initial_dual_average_) {
    pb_ws->add_initial_dual_average(static_cast<double>(v));
  }
  for (const auto& v : ws.current_ATY_) {
    pb_ws->add_current_aty(static_cast<double>(v));
  }
  for (const auto& v : ws.sum_primal_solutions_) {
    pb_ws->add_sum_primal_solutions(static_cast<double>(v));
  }
  for (const auto& v : ws.sum_dual_solutions_) {
    pb_ws->add_sum_dual_solutions(static_cast<double>(v));
  }
  for (const auto& v : ws.last_restart_duality_gap_primal_solution_) {
    pb_ws->add_last_restart_duality_gap_primal_solution(static_cast<double>(v));
  }
  for (const auto& v : ws.last_restart_duality_gap_dual_solution_) {
    pb_ws->add_last_restart_duality_gap_dual_solution(static_cast<double>(v));
  }
  pb_ws->set_initial_primal_weight(static_cast<double>(ws.initial_primal_weight_));
  pb_ws->set_initial_step_size(static_cast<double>(ws.initial_step_size_));
  pb_ws->set_total_pdlp_iterations(static_cast<int32_t>(ws.total_pdlp_iterations_));
  pb_ws->set_total_pdhg_iterations(static_cast<int32_t>(ws.total_pdhg_iterations_));
  pb_ws->set_last_candidate_kkt_score(static_cast<double>(ws.last_candidate_kkt_score_));
  pb_ws->set_last_restart_kkt_score(static_cast<double>(ws.last_restart_kkt_score_));
  pb_ws->set_sum_solution_weight(static_cast<double>(ws.sum_solution_weight_));
  pb_ws->set_iterations_since_last_restart(static_cast<int32_t>(ws.iterations_since_last_restart_));
}

// Packed repeated doubles are 8 bytes each, plus a tag and a length varint.
// The scalar fields and the embedding tag are covered by the fixed slack.
template <typename f_t>
size_t warm_start_vector_bytes(const std::vector<f_t>& values)
{
  constexpr size_t kFieldOverhead = 8;
  return values.size() * sizeof(double) + (values.empty() ? 0 : kFieldOverhead);
}

template <typename f_t>
void require_finite_warm_start(f_t value, const char* name)
{
  if (!std::isfinite(value)) {
    throw std::invalid_argument(std::string("PDLP warm start ") + name + " is not finite");
  }
}

template <typename f_t>
void require_finite_warm_start(const std::vector<f_t>& values, const char* name)
{
  for (const auto& value : values) {
    require_finite_warm_start(value, name);
  }
}

// -1 is the unset sentinel on cpu_pdlp_warm_start_data_t. Any smaller value is
// not a count the solver can resume from.
template <typename i_t>
void require_warm_start_iteration(i_t value, const char* name)
{
  if (value < static_cast<i_t>(-1)) {
    throw std::invalid_argument(std::string("PDLP warm start ") + name +
                                " must be -1 or non-negative");
  }
}

template <typename f_t, typename i_t>
void require_warm_start_length(const std::vector<f_t>& values,
                               i_t expected,
                               const char* name,
                               const char* what)
{
  if (values.size() != static_cast<size_t>(expected)) {
    throw std::invalid_argument(std::string("PDLP warm start ") + name + " has " +
                                std::to_string(values.size()) + " values; expected " +
                                std::to_string(static_cast<long long>(expected)) + " " + what);
  }
}

template <typename i_t, typename f_t>
void validate_settings_warm_start(const cpu_pdlp_warm_start_data_t<i_t, f_t>& ws,
                                  i_t n_variables,
                                  i_t n_constraints)
{
  require_finite_warm_start(ws.current_primal_solution_, "current_primal_solution");
  require_finite_warm_start(ws.current_dual_solution_, "current_dual_solution");
  require_finite_warm_start(ws.initial_primal_average_, "initial_primal_average");
  require_finite_warm_start(ws.initial_dual_average_, "initial_dual_average");
  require_finite_warm_start(ws.current_ATY_, "current_ATY");
  require_finite_warm_start(ws.sum_primal_solutions_, "sum_primal_solutions");
  require_finite_warm_start(ws.sum_dual_solutions_, "sum_dual_solutions");
  require_finite_warm_start(ws.last_restart_duality_gap_primal_solution_,
                            "last_restart_duality_gap_primal_solution");
  require_finite_warm_start(ws.last_restart_duality_gap_dual_solution_,
                            "last_restart_duality_gap_dual_solution");
  require_finite_warm_start(ws.initial_primal_weight_, "initial_primal_weight");
  require_finite_warm_start(ws.initial_step_size_, "initial_step_size");
  require_finite_warm_start(ws.last_candidate_kkt_score_, "last_candidate_kkt_score");
  require_finite_warm_start(ws.last_restart_kkt_score_, "last_restart_kkt_score");
  require_finite_warm_start(ws.sum_solution_weight_, "sum_solution_weight");
  require_warm_start_iteration(ws.total_pdlp_iterations_, "total_pdlp_iterations");
  require_warm_start_iteration(ws.total_pdhg_iterations_, "total_pdhg_iterations");
  require_warm_start_iteration(ws.iterations_since_last_restart_, "iterations_since_last_restart");

  // A negative count means the caller has no problem yet (mapper round-trip).
  // The worker always passes the reconstructed dimensions.
  if (n_variables < 0 || n_constraints < 0) { return; }
  require_warm_start_length(
    ws.current_primal_solution_, n_variables, "current_primal_solution", "variables");
  require_warm_start_length(
    ws.initial_primal_average_, n_variables, "initial_primal_average", "variables");
  require_warm_start_length(ws.current_ATY_, n_variables, "current_ATY", "variables");
  require_warm_start_length(
    ws.sum_primal_solutions_, n_variables, "sum_primal_solutions", "variables");
  require_warm_start_length(ws.last_restart_duality_gap_primal_solution_,
                            n_variables,
                            "last_restart_duality_gap_primal_solution",
                            "variables");
  require_warm_start_length(
    ws.current_dual_solution_, n_constraints, "current_dual_solution", "constraints");
  require_warm_start_length(
    ws.initial_dual_average_, n_constraints, "initial_dual_average", "constraints");
  require_warm_start_length(
    ws.sum_dual_solutions_, n_constraints, "sum_dual_solutions", "constraints");
  require_warm_start_length(ws.last_restart_duality_gap_dual_solution_,
                            n_constraints,
                            "last_restart_duality_gap_dual_solution",
                            "constraints");
}

template <typename i_t, typename f_t>
void read_settings_warm_start(const cuopt::remote::PDLPSolverSettings& pb_settings,
                              pdlp_solver_settings_t<i_t, f_t>& settings,
                              i_t n_variables,
                              i_t n_constraints)
{
  if (!pb_settings.has_warm_start_data()) { return; }
  const auto& pb_ws = pb_settings.warm_start_data();
  cpu_pdlp_warm_start_data_t<i_t, f_t> decoded;
  copy_repeated(pb_ws.current_primal_solution(), decoded.current_primal_solution_);
  copy_repeated(pb_ws.current_dual_solution(), decoded.current_dual_solution_);
  copy_repeated(pb_ws.initial_primal_average(), decoded.initial_primal_average_);
  copy_repeated(pb_ws.initial_dual_average(), decoded.initial_dual_average_);
  copy_repeated(pb_ws.current_aty(), decoded.current_ATY_);
  copy_repeated(pb_ws.sum_primal_solutions(), decoded.sum_primal_solutions_);
  copy_repeated(pb_ws.sum_dual_solutions(), decoded.sum_dual_solutions_);
  copy_repeated(pb_ws.last_restart_duality_gap_primal_solution(),
                decoded.last_restart_duality_gap_primal_solution_);
  copy_repeated(pb_ws.last_restart_duality_gap_dual_solution(),
                decoded.last_restart_duality_gap_dual_solution_);
  decoded.initial_primal_weight_         = static_cast<f_t>(pb_ws.initial_primal_weight());
  decoded.initial_step_size_             = static_cast<f_t>(pb_ws.initial_step_size());
  decoded.total_pdlp_iterations_         = static_cast<i_t>(pb_ws.total_pdlp_iterations());
  decoded.total_pdhg_iterations_         = static_cast<i_t>(pb_ws.total_pdhg_iterations());
  decoded.last_candidate_kkt_score_      = static_cast<f_t>(pb_ws.last_candidate_kkt_score());
  decoded.last_restart_kkt_score_        = static_cast<f_t>(pb_ws.last_restart_kkt_score());
  decoded.sum_solution_weight_           = static_cast<f_t>(pb_ws.sum_solution_weight());
  decoded.iterations_since_last_restart_ = static_cast<i_t>(pb_ws.iterations_since_last_restart());
  validate_settings_warm_start(decoded, n_variables, n_constraints);
  settings.get_cpu_pdlp_warm_start_data() = std::move(decoded);
}

}  // namespace

template <typename i_t, typename f_t>
void map_pdlp_settings_to_proto(const pdlp_solver_settings_t<i_t, f_t>& settings,
                                cuopt::remote::PDLPSolverSettings* pb_settings)
{
#include "generated_pdlp_settings_to_proto.inc"
  write_settings_warm_start(settings, pb_settings);
}

template <typename i_t, typename f_t>
size_t estimate_pdlp_warm_start_proto_size(const pdlp_solver_settings_t<i_t, f_t>& settings)
{
  const auto& ws = settings.get_cpu_pdlp_warm_start_data();
  if (!ws.is_populated()) { return 0; }
  constexpr size_t kScalarAndEmbedSlack = 256;
  return warm_start_vector_bytes(ws.current_primal_solution_) +
         warm_start_vector_bytes(ws.current_dual_solution_) +
         warm_start_vector_bytes(ws.initial_primal_average_) +
         warm_start_vector_bytes(ws.initial_dual_average_) +
         warm_start_vector_bytes(ws.current_ATY_) +
         warm_start_vector_bytes(ws.sum_primal_solutions_) +
         warm_start_vector_bytes(ws.sum_dual_solutions_) +
         warm_start_vector_bytes(ws.last_restart_duality_gap_primal_solution_) +
         warm_start_vector_bytes(ws.last_restart_duality_gap_dual_solution_) + kScalarAndEmbedSlack;
}

template <typename i_t, typename f_t>
void map_proto_to_pdlp_settings(const cuopt::remote::PDLPSolverSettings& pb_settings,
                                pdlp_solver_settings_t<i_t, f_t>& settings,
                                i_t n_variables,
                                i_t n_constraints)
{
#include "generated_proto_to_pdlp_settings.inc"

  // Post-decode input sanitization: the generated code does raw static_cast
  // on int32 -> enum, which is UB for values outside the enum range. Clamp
  // out-of-range values from buggy/untrusted encoders to safe defaults, and
  // guard the int64 -> i_t conversion of iteration_limit against overflow.
  {
    auto pv = pb_settings.presolver();
    if (pv < CUOPT_PRESOLVE_DEFAULT || pv > CUOPT_PRESOLVE_PSLP) {
      settings.presolver = presolver_t::Default;
    }
  }
  {
    auto pv = pb_settings.pdlp_precision();
    if (pv < CUOPT_PDLP_DEFAULT_PRECISION || pv > CUOPT_PDLP_MIXED_PRECISION) {
      settings.pdlp_precision = pdlp_precision_t::DefaultPrecision;
    }
  }
  if (pb_settings.iteration_limit() > static_cast<int64_t>(std::numeric_limits<i_t>::max())) {
    settings.iteration_limit = std::numeric_limits<i_t>::max();
  }
  read_settings_warm_start(pb_settings, settings, n_variables, n_constraints);
}

template <typename i_t, typename f_t>
void map_mip_settings_to_proto(const mip_solver_settings_t<i_t, f_t>& settings,
                               cuopt::remote::MIPSolverSettings* pb_settings)
{
#include "generated_mip_settings_to_proto.inc"
}

template <typename i_t, typename f_t>
void map_proto_to_mip_settings(const cuopt::remote::MIPSolverSettings& pb_settings,
                               mip_solver_settings_t<i_t, f_t>& settings)
{
#include "generated_proto_to_mip_settings.inc"

  // Post-decode input sanitization: clamp out-of-range enum / mode values
  // from buggy/untrusted encoders to safe defaults.
  {
    auto pv = pb_settings.presolver();
    if (pv < CUOPT_PRESOLVE_DEFAULT || pv > CUOPT_PRESOLVE_PSLP) {
      settings.presolver = presolver_t::Default;
    }
  }
  {
    auto sv = pb_settings.mip_scaling();
    if (sv < CUOPT_MIP_SCALING_OFF || sv > CUOPT_MIP_SCALING_NO_OBJECTIVE) {
      settings.mip_scaling = CUOPT_MIP_SCALING_ON;
    }
  }
  {
    // symmetry: valid range matches the local-solve binding in
    // solver_settings.cu ({CUOPT_MIP_SYMMETRY, ..., -1, 2, -1}).
    auto sv = pb_settings.symmetry();
    if (sv < -1 || sv > 2) { settings.symmetry = -1; }
  }
}

namespace {

// A protobuf map keeps one value per key. A repeated name must already hold
// the same value on every registration. set_parameter() writes all of them.
template <typename T, typename Format>
void append_parameter_list(const std::vector<parameter_info_t<T>>& parameters,
                           google::protobuf::Map<std::string, std::string>* out,
                           Format format)
{
  std::unordered_map<std::string, T> seen;
  for (const auto& p : parameters) {
    const T value             = *p.value_ptr;
    const auto [it, inserted] = seen.emplace(p.param_name, value);
    if (!inserted) {
      if (it->second != value) {
        throw std::invalid_argument("Parameter " + p.param_name +
                                    " differs between LP and MIP settings");
      }
      continue;
    }
    (*out)[p.param_name] = format(value);
  }
}

}  // namespace

template <typename i_t, typename f_t>
void append_solver_parameters(const solver_settings_t<i_t, f_t>& settings,
                              google::protobuf::Map<std::string, std::string>* out)
{
  append_parameter_list(
    settings.get_float_parameters(), out, [](f_t value) { return format_parameter_float(value); });
  append_parameter_list(
    settings.get_int_parameters(), out, [](i_t value) { return std::to_string(value); });
  append_parameter_list(
    settings.get_bool_parameters(), out, [](bool value) { return value ? "true" : "false"; });
  append_parameter_list(
    settings.get_string_parameters(), out, [](const std::string& value) { return value; });
}

template <typename i_t, typename f_t>
void apply_parameter_overrides(solver_settings_t<i_t, f_t>& settings,
                               const google::protobuf::Map<std::string, std::string>& parameters)
{
  // A protobuf map has one entry per name. Each name is registered at least
  // once, so a complete map is never larger than these four lists. Twice that
  // length is spare room. The lists are the source of the count, so adding a
  // parameter raises the limit with no separate constant to update.
  constexpr std::size_t kParameterMapHeadroom = 2;
  const std::size_t registered =
    settings.get_float_parameters().size() + settings.get_int_parameters().size() +
    settings.get_bool_parameters().size() + settings.get_string_parameters().size();
  if (static_cast<std::size_t>(parameters.size()) > registered * kParameterMapHeadroom) {
    throw std::invalid_argument("Too many solver parameters");
  }

  // After the deprecated typed fields have been copied onto `settings`.
  // set_parameter_from_string is the same path the CLI and C API use, so a
  // key here wins over those fields and a parameter with no typed field is
  // still applied.
  for (const auto& entry : parameters) {
    settings.set_parameter_from_string(entry.first, entry.second);
  }
}

// Explicit template instantiations
#if CUOPT_INSTANTIATE_FLOAT
template CUOPT_EXPORT void map_pdlp_settings_to_proto(
  const pdlp_solver_settings_t<int32_t, float>& settings,
  cuopt::remote::PDLPSolverSettings* pb_settings);
template CUOPT_EXPORT void map_proto_to_pdlp_settings(
  const cuopt::remote::PDLPSolverSettings& pb_settings,
  pdlp_solver_settings_t<int32_t, float>& settings,
  int32_t n_variables,
  int32_t n_constraints);
template CUOPT_EXPORT size_t
estimate_pdlp_warm_start_proto_size(const pdlp_solver_settings_t<int32_t, float>& settings);
template CUOPT_EXPORT void map_mip_settings_to_proto(
  const mip_solver_settings_t<int32_t, float>& settings,
  cuopt::remote::MIPSolverSettings* pb_settings);
template CUOPT_EXPORT void map_proto_to_mip_settings(
  const cuopt::remote::MIPSolverSettings& pb_settings,
  mip_solver_settings_t<int32_t, float>& settings);
template CUOPT_EXPORT void apply_parameter_overrides(
  solver_settings_t<int32_t, float>& settings,
  const google::protobuf::Map<std::string, std::string>& parameters);
template CUOPT_EXPORT void append_solver_parameters(
  const solver_settings_t<int32_t, float>& settings,
  google::protobuf::Map<std::string, std::string>* out);
#endif

#if CUOPT_INSTANTIATE_DOUBLE
template CUOPT_EXPORT void map_pdlp_settings_to_proto(
  const pdlp_solver_settings_t<int32_t, double>& settings,
  cuopt::remote::PDLPSolverSettings* pb_settings);
template CUOPT_EXPORT void map_proto_to_pdlp_settings(
  const cuopt::remote::PDLPSolverSettings& pb_settings,
  pdlp_solver_settings_t<int32_t, double>& settings,
  int32_t n_variables,
  int32_t n_constraints);
template CUOPT_EXPORT size_t
estimate_pdlp_warm_start_proto_size(const pdlp_solver_settings_t<int32_t, double>& settings);
template CUOPT_EXPORT void map_mip_settings_to_proto(
  const mip_solver_settings_t<int32_t, double>& settings,
  cuopt::remote::MIPSolverSettings* pb_settings);
template CUOPT_EXPORT void map_proto_to_mip_settings(
  const cuopt::remote::MIPSolverSettings& pb_settings,
  mip_solver_settings_t<int32_t, double>& settings);
template CUOPT_EXPORT void apply_parameter_overrides(
  solver_settings_t<int32_t, double>& settings,
  const google::protobuf::Map<std::string, std::string>& parameters);
template CUOPT_EXPORT void append_solver_parameters(
  const solver_settings_t<int32_t, double>& settings,
  google::protobuf::Map<std::string, std::string>* out);
#endif

}  // namespace cuopt::mathematical_optimization
