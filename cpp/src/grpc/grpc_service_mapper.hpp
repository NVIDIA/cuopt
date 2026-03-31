/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved. SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cuopt_remote.pb.h>
#include <cuopt_remote_service.pb.h>

#include <string>

namespace cuopt::linear_programming {

// Forward declarations
template <typename i_t, typename f_t>
class cpu_optimization_problem_t;

template <typename i_t, typename f_t>
struct pdlp_solver_settings_t;

template <typename i_t, typename f_t>
struct mip_solver_settings_t;

/**
 * @brief Build a gRPC SubmitJobRequest for an LP problem.
 *
 * Creates a SubmitJobRequest containing the LP problem and settings using
 * the problem and settings mappers. Serialization is handled by the protobuf library.
 */
template <typename i_t, typename f_t>
cuopt::remote::SubmitJobRequest build_lp_submit_request(
  const cpu_optimization_problem_t<i_t, f_t>& cpu_problem,
  const pdlp_solver_settings_t<i_t, f_t>& settings);

/**
 * @brief Build a gRPC SubmitJobRequest for a MIP problem.
 *
 * Creates a SubmitJobRequest containing the MIP problem and settings using
 * the problem and settings mappers. Serialization is handled by the protobuf library.
 */
template <typename i_t, typename f_t>
cuopt::remote::SubmitJobRequest build_mip_submit_request(
  const cpu_optimization_problem_t<i_t, f_t>& cpu_problem,
  const mip_solver_settings_t<i_t, f_t>& settings,
  bool enable_incumbents = false);

/**
 * @brief Build a gRPC StatusRequest.
 *
 * Simple helper to create a status check request.
 *
 * @param job_id The job ID to check status for
 * @return StatusRequest protobuf message
 */
inline cuopt::remote::StatusRequest build_status_request(const std::string& job_id)
{
  cuopt::remote::StatusRequest request;
  request.set_job_id(job_id);
  return request;
}

/**
 * @brief Build a gRPC GetResultRequest.
 *
 * Simple helper to create a result retrieval request.
 *
 * @param job_id The job ID to get results for
 * @return GetResultRequest protobuf message
 */
inline cuopt::remote::GetResultRequest build_get_result_request(const std::string& job_id)
{
  cuopt::remote::GetResultRequest request;
  request.set_job_id(job_id);
  return request;
}

/**
 * @brief Build a gRPC CancelRequest.
 *
 * Simple helper to create a job cancellation request.
 *
 * @param job_id The job ID to cancel
 * @return CancelRequest protobuf message
 */
inline cuopt::remote::CancelRequest build_cancel_request(const std::string& job_id)
{
  cuopt::remote::CancelRequest request;
  request.set_job_id(job_id);
  return request;
}

/**
 * @brief Build a gRPC DeleteRequest.
 *
 * Simple helper to create a result deletion request.
 *
 * @param job_id The job ID whose result should be deleted
 * @return DeleteRequest protobuf message
 */
inline cuopt::remote::DeleteRequest build_delete_request(const std::string& job_id)
{
  cuopt::remote::DeleteRequest request;
  request.set_job_id(job_id);
  return request;
}

}  // namespace cuopt::linear_programming
