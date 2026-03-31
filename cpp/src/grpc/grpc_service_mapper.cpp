/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved. SPDX-License-Identifier: Apache-2.0
 */

#include "grpc_service_mapper.hpp"

#include <cuopt/linear_programming/constants.h>
#include <cuopt_remote.pb.h>
#include <cuopt_remote_service.pb.h>
#include <cuopt/linear_programming/cpu_optimization_problem.hpp>
#include <cuopt/linear_programming/optimization_problem_interface.hpp>
#include "grpc_problem_mapper.hpp"
#include "grpc_settings_mapper.hpp"

namespace cuopt::linear_programming {

template <typename i_t, typename f_t>
cuopt::remote::SubmitJobRequest build_lp_submit_request(
  const cpu_optimization_problem_t<i_t, f_t>& cpu_problem,
  const pdlp_solver_settings_t<i_t, f_t>& settings)
{
  cuopt::remote::SubmitJobRequest submit_request;

  // Get the lp_request from the oneof
  auto* lp_request = submit_request.mutable_lp_request();

  // Set header
  auto* header = lp_request->mutable_header();
  header->set_version(1);
  header->set_problem_category(cuopt::remote::LP);

  // Map problem data to protobuf
  map_problem_to_proto(cpu_problem, lp_request->mutable_problem());

  // Map settings to protobuf
  map_pdlp_settings_to_proto(settings, lp_request->mutable_settings());

  return submit_request;
}

template <typename i_t, typename f_t>
cuopt::remote::SubmitJobRequest build_mip_submit_request(
  const cpu_optimization_problem_t<i_t, f_t>& cpu_problem,
  const mip_solver_settings_t<i_t, f_t>& settings,
  bool enable_incumbents)
{
  cuopt::remote::SubmitJobRequest submit_request;

  // Get the mip_request from the oneof
  auto* mip_request = submit_request.mutable_mip_request();

  // Set header
  auto* header = mip_request->mutable_header();
  header->set_version(1);
  header->set_problem_category(cuopt::remote::MIP);

  // Map problem data to protobuf
  map_problem_to_proto(cpu_problem, mip_request->mutable_problem());

  // Map settings to protobuf
  map_mip_settings_to_proto(settings, mip_request->mutable_settings());

  // Set enable_incumbents flag
  mip_request->set_enable_incumbents(enable_incumbents);

  return submit_request;
}

// Explicit template instantiations
#if CUOPT_INSTANTIATE_FLOAT
template cuopt::remote::SubmitJobRequest build_lp_submit_request(
  const cpu_optimization_problem_t<int32_t, float>& cpu_problem,
  const pdlp_solver_settings_t<int32_t, float>& settings);
template cuopt::remote::SubmitJobRequest build_mip_submit_request(
  const cpu_optimization_problem_t<int32_t, float>& cpu_problem,
  const mip_solver_settings_t<int32_t, float>& settings,
  bool enable_incumbents);
#endif

#if CUOPT_INSTANTIATE_DOUBLE
template cuopt::remote::SubmitJobRequest build_lp_submit_request(
  const cpu_optimization_problem_t<int32_t, double>& cpu_problem,
  const pdlp_solver_settings_t<int32_t, double>& settings);
template cuopt::remote::SubmitJobRequest build_mip_submit_request(
  const cpu_optimization_problem_t<int32_t, double>& cpu_problem,
  const mip_solver_settings_t<int32_t, double>& settings,
  bool enable_incumbents);
#endif

}  // namespace cuopt::linear_programming
