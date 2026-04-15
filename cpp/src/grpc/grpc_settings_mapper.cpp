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
