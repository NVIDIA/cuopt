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

template <typename i_t, typename f_t>
void read_settings_warm_start(const cuopt::remote::PDLPSolverSettings& pb_settings,
                              pdlp_solver_settings_t<i_t, f_t>& settings)
{
  if (!pb_settings.has_warm_start_data()) { return; }
  const auto& pb_ws = pb_settings.warm_start_data();
  auto& ws          = settings.get_cpu_pdlp_warm_start_data();
  copy_repeated(pb_ws.current_primal_solution(), ws.current_primal_solution_);
  copy_repeated(pb_ws.current_dual_solution(), ws.current_dual_solution_);
  copy_repeated(pb_ws.initial_primal_average(), ws.initial_primal_average_);
  copy_repeated(pb_ws.initial_dual_average(), ws.initial_dual_average_);
  copy_repeated(pb_ws.current_aty(), ws.current_ATY_);
  copy_repeated(pb_ws.sum_primal_solutions(), ws.sum_primal_solutions_);
  copy_repeated(pb_ws.sum_dual_solutions(), ws.sum_dual_solutions_);
  copy_repeated(pb_ws.last_restart_duality_gap_primal_solution(),
                ws.last_restart_duality_gap_primal_solution_);
  copy_repeated(pb_ws.last_restart_duality_gap_dual_solution(),
                ws.last_restart_duality_gap_dual_solution_);
  ws.initial_primal_weight_         = static_cast<f_t>(pb_ws.initial_primal_weight());
  ws.initial_step_size_             = static_cast<f_t>(pb_ws.initial_step_size());
  ws.total_pdlp_iterations_         = static_cast<i_t>(pb_ws.total_pdlp_iterations());
  ws.total_pdhg_iterations_         = static_cast<i_t>(pb_ws.total_pdhg_iterations());
  ws.last_candidate_kkt_score_      = static_cast<f_t>(pb_ws.last_candidate_kkt_score());
  ws.last_restart_kkt_score_        = static_cast<f_t>(pb_ws.last_restart_kkt_score());
  ws.sum_solution_weight_           = static_cast<f_t>(pb_ws.sum_solution_weight());
  ws.iterations_since_last_restart_ = static_cast<i_t>(pb_ws.iterations_since_last_restart());
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
                                pdlp_solver_settings_t<i_t, f_t>& settings)
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
  read_settings_warm_start(pb_settings, settings);
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

template <typename i_t, typename f_t>
void append_solver_parameters(const solver_settings_t<i_t, f_t>& settings,
                              google::protobuf::Map<std::string, std::string>* out)
{
  // A protobuf map keeps one value per key. Walking in registration order
  // means a shared name keeps the later registration. set_parameter() writes
  // every registration of a name, so those values already agree.
  for (const auto& p : settings.get_float_parameters()) {
    (*out)[p.param_name] = format_parameter_float(*p.value_ptr);
  }
  for (const auto& p : settings.get_int_parameters()) {
    (*out)[p.param_name] = std::to_string(*p.value_ptr);
  }
  for (const auto& p : settings.get_bool_parameters()) {
    (*out)[p.param_name] = *p.value_ptr ? "true" : "false";
  }
  for (const auto& p : settings.get_string_parameters()) {
    (*out)[p.param_name] = *p.value_ptr;
  }
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
  pdlp_solver_settings_t<int32_t, float>& settings);
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
  pdlp_solver_settings_t<int32_t, double>& settings);
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
