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

#include <limits>
#include <stdexcept>
#include <string>

namespace cuopt::mathematical_optimization {

namespace {
#include "generated_enum_converters_settings.inc"
}  // namespace

template <typename i_t, typename f_t>
void map_pdlp_settings_to_proto(const pdlp_solver_settings_t<i_t, f_t>& settings,
                                cuopt::remote::PDLPSolverSettings* pb_settings)
{
#include "generated_pdlp_settings_to_proto.inc"
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
template CUOPT_EXPORT void map_mip_settings_to_proto(
  const mip_solver_settings_t<int32_t, float>& settings,
  cuopt::remote::MIPSolverSettings* pb_settings);
template CUOPT_EXPORT void map_proto_to_mip_settings(
  const cuopt::remote::MIPSolverSettings& pb_settings,
  mip_solver_settings_t<int32_t, float>& settings);
template CUOPT_EXPORT void apply_parameter_overrides(
  solver_settings_t<int32_t, float>& settings,
  const google::protobuf::Map<std::string, std::string>& parameters);
#endif

#if CUOPT_INSTANTIATE_DOUBLE
template CUOPT_EXPORT void map_pdlp_settings_to_proto(
  const pdlp_solver_settings_t<int32_t, double>& settings,
  cuopt::remote::PDLPSolverSettings* pb_settings);
template CUOPT_EXPORT void map_proto_to_pdlp_settings(
  const cuopt::remote::PDLPSolverSettings& pb_settings,
  pdlp_solver_settings_t<int32_t, double>& settings);
template CUOPT_EXPORT void map_mip_settings_to_proto(
  const mip_solver_settings_t<int32_t, double>& settings,
  cuopt::remote::MIPSolverSettings* pb_settings);
template CUOPT_EXPORT void map_proto_to_mip_settings(
  const cuopt::remote::MIPSolverSettings& pb_settings,
  mip_solver_settings_t<int32_t, double>& settings);
template CUOPT_EXPORT void apply_parameter_overrides(
  solver_settings_t<int32_t, double>& settings,
  const google::protobuf::Map<std::string, std::string>& parameters);
#endif

}  // namespace cuopt::mathematical_optimization
