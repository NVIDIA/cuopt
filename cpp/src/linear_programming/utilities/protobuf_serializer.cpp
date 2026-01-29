/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved. SPDX-License-Identifier: Apache-2.0
 */

#include <cuopt/linear_programming/constants.h>
#include <cuopt_remote.pb.h>
#include <cuopt/linear_programming/utilities/remote_serialization.hpp>

#include <utilities/logger.hpp>

// Required for cudaMemcpy when compiled as C++ (not CUDA)
#include <cuda_runtime_api.h>

#include <dlfcn.h>
#include <cmath>
#include <limits>

namespace cuopt::linear_programming {

namespace {

// Convert cuOpt termination status to protobuf enum
cuopt::remote::PDLPTerminationStatus to_proto_status(pdlp_termination_status_t status)
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
pdlp_termination_status_t from_proto_status(cuopt::remote::PDLPTerminationStatus status)
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
    default: return mip_termination_status_t::NoTermination;
  }
}

}  // namespace

/**
 * @brief Default Protocol Buffers serializer implementation.
 */
template <typename i_t, typename f_t>
class protobuf_serializer_t : public remote_serializer_t<i_t, f_t> {
 public:
  using job_status_t = typename remote_serializer_t<i_t, f_t>::job_status_t;

  protobuf_serializer_t()           = default;
  ~protobuf_serializer_t() override = default;

  //============================================================================
  // Problem Serialization
  //============================================================================

  std::vector<uint8_t> serialize_lp_request(
    const mps_parser::data_model_view_t<i_t, f_t>& view,
    const pdlp_solver_settings_t<i_t, f_t>& settings) override
  {
    cuopt::remote::SolveLPRequest request;

    // Set header
    auto* header = request.mutable_header();
    header->set_version(protocol_version());
    header->set_problem_type(cuopt::remote::LP);
    header->set_index_type(sizeof(i_t) == 4 ? cuopt::remote::INT32 : cuopt::remote::INT64);
    header->set_float_type(sizeof(f_t) == 4 ? cuopt::remote::FLOAT32 : cuopt::remote::DOUBLE);

    // Serialize problem data
    serialize_problem_to_proto(view, request.mutable_problem());

    // Serialize settings
    serialize_lp_settings_to_proto(settings, request.mutable_settings());

    // Serialize to bytes
    std::vector<uint8_t> result(request.ByteSizeLong());
    request.SerializeToArray(result.data(), result.size());
    return result;
  }

  std::vector<uint8_t> serialize_mip_request(
    const mps_parser::data_model_view_t<i_t, f_t>& view,
    const mip_solver_settings_t<i_t, f_t>& settings) override
  {
    cuopt::remote::SolveMIPRequest request;

    // Set header
    auto* header = request.mutable_header();
    header->set_version(protocol_version());
    header->set_problem_type(cuopt::remote::MIP);
    header->set_index_type(sizeof(i_t) == 4 ? cuopt::remote::INT32 : cuopt::remote::INT64);
    header->set_float_type(sizeof(f_t) == 4 ? cuopt::remote::FLOAT32 : cuopt::remote::DOUBLE);

    // Serialize problem data
    serialize_problem_to_proto(view, request.mutable_problem());

    // Serialize all MIP settings (names match cuOpt API)
    auto* pb_settings = request.mutable_settings();
    pb_settings->set_time_limit(settings.time_limit);
    pb_settings->set_relative_mip_gap(settings.tolerances.relative_mip_gap);
    pb_settings->set_absolute_mip_gap(settings.tolerances.absolute_mip_gap);
    pb_settings->set_integrality_tolerance(settings.tolerances.integrality_tolerance);
    pb_settings->set_absolute_tolerance(settings.tolerances.absolute_tolerance);
    pb_settings->set_relative_tolerance(settings.tolerances.relative_tolerance);
    pb_settings->set_presolve_absolute_tolerance(settings.tolerances.presolve_absolute_tolerance);
    pb_settings->set_log_to_console(settings.log_to_console);
    pb_settings->set_heuristics_only(settings.heuristics_only);
    pb_settings->set_num_cpu_threads(settings.num_cpu_threads);
    pb_settings->set_num_gpus(settings.num_gpus);
    pb_settings->set_presolve(settings.presolve);
    pb_settings->set_mip_scaling(settings.mip_scaling);

    request.set_enable_incumbents(!settings.get_mip_callbacks().empty());

    // Serialize to bytes
    std::vector<uint8_t> result(request.ByteSizeLong());
    request.SerializeToArray(result.data(), result.size());
    return result;
  }

  //============================================================================
  // Solution Deserialization
  //============================================================================

  optimization_problem_solution_t<i_t, f_t> deserialize_lp_solution(
    const std::vector<uint8_t>& data) override
  {
    cuopt::remote::LPSolution pb_solution;
    if (!pb_solution.ParseFromArray(data.data(), data.size())) {
      return optimization_problem_solution_t<i_t, f_t>(
        cuopt::logic_error("Failed to parse LP solution", cuopt::error_type_t::RuntimeError));
    }

    return proto_to_lp_solution(pb_solution);
  }

  mip_solution_t<i_t, f_t> deserialize_mip_solution(const std::vector<uint8_t>& data) override
  {
    cuopt::remote::MIPSolution pb_solution;
    if (!pb_solution.ParseFromArray(data.data(), data.size())) {
      return mip_solution_t<i_t, f_t>(
        cuopt::logic_error("Failed to parse MIP solution", cuopt::error_type_t::RuntimeError));
    }

    return proto_to_mip_solution(pb_solution);
  }

  //============================================================================
  // Server-side Operations
  //============================================================================

  bool is_mip_request(const std::vector<uint8_t>& data) override
  {
    // Try to parse as async request first
    cuopt::remote::AsyncRequest async_request;
    if (async_request.ParseFromArray(data.data(), data.size())) {
      if (async_request.has_mip_request()) { return true; }
      if (async_request.has_lp_request()) { return false; }
    }

    // Try to parse as direct request and check the header's problem_type
    // MIP request - check if header indicates MIP
    cuopt::remote::SolveMIPRequest mip_request;
    if (mip_request.ParseFromArray(data.data(), data.size()) && mip_request.has_header()) {
      if (mip_request.header().problem_type() == cuopt::remote::MIP) { return true; }
    }

    // LP request - check if header indicates LP
    cuopt::remote::SolveLPRequest lp_request;
    if (lp_request.ParseFromArray(data.data(), data.size()) && lp_request.has_header()) {
      if (lp_request.header().problem_type() == cuopt::remote::LP) { return false; }
    }

    return false;  // Default to LP if can't determine
  }

  bool deserialize_lp_request(const std::vector<uint8_t>& data,
                              cuopt::mps_parser::mps_data_model_t<i_t, f_t>& mps_data,
                              pdlp_solver_settings_t<i_t, f_t>& settings) override
  {
    // Try async request first
    cuopt::remote::AsyncRequest async_request;
    if (async_request.ParseFromArray(data.data(), data.size()) && async_request.has_lp_request()) {
      const auto& lp_request = async_request.lp_request();
      proto_to_mps_data(lp_request.problem(), mps_data);
      proto_to_lp_settings(lp_request.settings(), settings);
      return true;
    }

    // Try direct LP request
    cuopt::remote::SolveLPRequest request;
    if (!request.ParseFromArray(data.data(), data.size())) {
      CUOPT_LOG_ERROR("[protobuf_serializer] Failed to parse LP request");
      return false;
    }

    proto_to_mps_data(request.problem(), mps_data);
    proto_to_lp_settings(request.settings(), settings);
    return true;
  }

  bool deserialize_mip_request(const std::vector<uint8_t>& data,
                               cuopt::mps_parser::mps_data_model_t<i_t, f_t>& mps_data,
                               mip_solver_settings_t<i_t, f_t>& settings) override
  {
    // Try async request first
    cuopt::remote::AsyncRequest async_request;
    if (async_request.ParseFromArray(data.data(), data.size()) && async_request.has_mip_request()) {
      const auto& mip_request = async_request.mip_request();
      proto_to_mps_data(mip_request.problem(), mps_data);
      proto_to_mip_settings(mip_request.settings(), settings);
      return true;
    }

    // Try direct MIP request
    cuopt::remote::SolveMIPRequest request;
    if (!request.ParseFromArray(data.data(), data.size())) {
      CUOPT_LOG_ERROR("[protobuf_serializer] Failed to parse MIP request");
      return false;
    }

    proto_to_mps_data(request.problem(), mps_data);
    proto_to_mip_settings(request.settings(), settings);
    return true;
  }

  std::vector<uint8_t> serialize_lp_solution(
    const optimization_problem_solution_t<i_t, f_t>& solution) override
  {
    cuopt::remote::LPSolution pb_solution;
    lp_solution_to_proto(solution, &pb_solution);

    std::vector<uint8_t> result(pb_solution.ByteSizeLong());
    pb_solution.SerializeToArray(result.data(), result.size());
    return result;
  }

  std::vector<uint8_t> serialize_mip_solution(const mip_solution_t<i_t, f_t>& solution) override
  {
    cuopt::remote::MIPSolution pb_solution;
    mip_solution_to_proto(solution, &pb_solution);

    std::vector<uint8_t> result(pb_solution.ByteSizeLong());
    pb_solution.SerializeToArray(result.data(), result.size());
    return result;
  }

  //============================================================================
  // Async Operations
  //============================================================================

  std::vector<uint8_t> serialize_async_lp_request(
    const mps_parser::data_model_view_t<i_t, f_t>& view,
    const pdlp_solver_settings_t<i_t, f_t>& settings,
    bool blocking) override
  {
    cuopt::remote::AsyncRequest request;
    request.set_request_type(cuopt::remote::SUBMIT_JOB);
    request.set_blocking(blocking);

    auto* lp_request = request.mutable_lp_request();

    // Set header
    auto* header = lp_request->mutable_header();
    header->set_version(protocol_version());
    header->set_problem_type(cuopt::remote::LP);
    header->set_index_type(sizeof(i_t) == 4 ? cuopt::remote::INT32 : cuopt::remote::INT64);
    header->set_float_type(sizeof(f_t) == 4 ? cuopt::remote::FLOAT32 : cuopt::remote::DOUBLE);

    serialize_problem_to_proto(view, lp_request->mutable_problem());
    serialize_lp_settings_to_proto(settings, lp_request->mutable_settings());

    std::vector<uint8_t> result(request.ByteSizeLong());
    request.SerializeToArray(result.data(), result.size());
    return result;
  }

  std::vector<uint8_t> serialize_async_mip_request(
    const mps_parser::data_model_view_t<i_t, f_t>& view,
    const mip_solver_settings_t<i_t, f_t>& settings,
    bool blocking) override
  {
    cuopt::remote::AsyncRequest request;
    request.set_request_type(cuopt::remote::SUBMIT_JOB);
    request.set_blocking(blocking);

    auto* mip_request = request.mutable_mip_request();

    // Set header
    auto* header = mip_request->mutable_header();
    header->set_version(protocol_version());
    header->set_problem_type(cuopt::remote::MIP);
    header->set_index_type(sizeof(i_t) == 4 ? cuopt::remote::INT32 : cuopt::remote::INT64);
    header->set_float_type(sizeof(f_t) == 4 ? cuopt::remote::FLOAT32 : cuopt::remote::DOUBLE);

    serialize_problem_to_proto(view, mip_request->mutable_problem());

    // Serialize all MIP settings (names match cuOpt API)
    auto* pb_settings = mip_request->mutable_settings();
    pb_settings->set_time_limit(settings.time_limit);
    pb_settings->set_relative_mip_gap(settings.tolerances.relative_mip_gap);
    pb_settings->set_absolute_mip_gap(settings.tolerances.absolute_mip_gap);
    pb_settings->set_integrality_tolerance(settings.tolerances.integrality_tolerance);
    pb_settings->set_absolute_tolerance(settings.tolerances.absolute_tolerance);
    pb_settings->set_relative_tolerance(settings.tolerances.relative_tolerance);
    pb_settings->set_presolve_absolute_tolerance(settings.tolerances.presolve_absolute_tolerance);
    pb_settings->set_log_to_console(settings.log_to_console);
    pb_settings->set_heuristics_only(settings.heuristics_only);
    pb_settings->set_num_cpu_threads(settings.num_cpu_threads);
    pb_settings->set_num_gpus(settings.num_gpus);
    pb_settings->set_presolve(settings.presolve);
    pb_settings->set_mip_scaling(settings.mip_scaling);

    std::vector<uint8_t> result(request.ByteSizeLong());
    request.SerializeToArray(result.data(), result.size());
    return result;
  }

  std::vector<uint8_t> serialize_status_request(const std::string& job_id) override
  {
    cuopt::remote::AsyncRequest request;
    request.set_request_type(cuopt::remote::CHECK_STATUS);
    request.set_job_id(job_id);

    std::vector<uint8_t> result(request.ByteSizeLong());
    request.SerializeToArray(result.data(), result.size());
    return result;
  }

  std::vector<uint8_t> serialize_get_result_request(const std::string& job_id) override
  {
    cuopt::remote::AsyncRequest request;
    request.set_request_type(cuopt::remote::GET_RESULT);
    request.set_job_id(job_id);

    std::vector<uint8_t> result(request.ByteSizeLong());
    request.SerializeToArray(result.data(), result.size());
    return result;
  }

  std::vector<uint8_t> serialize_delete_request(const std::string& job_id) override
  {
    cuopt::remote::AsyncRequest request;
    request.set_request_type(cuopt::remote::DELETE_RESULT);
    request.set_job_id(job_id);

    std::vector<uint8_t> result(request.ByteSizeLong());
    request.SerializeToArray(result.data(), result.size());
    return result;
  }

  std::vector<uint8_t> serialize_get_logs_request(const std::string& job_id,
                                                  int64_t frombyte) override
  {
    cuopt::remote::AsyncRequest request;
    request.set_request_type(cuopt::remote::GET_LOGS);
    request.set_job_id(job_id);
    request.set_frombyte(frombyte);

    std::vector<uint8_t> result(request.ByteSizeLong());
    request.SerializeToArray(result.data(), result.size());
    return result;
  }

  std::vector<uint8_t> serialize_cancel_request(const std::string& job_id) override
  {
    cuopt::remote::AsyncRequest request;
    request.set_request_type(cuopt::remote::CANCEL_JOB);
    request.set_job_id(job_id);

    std::vector<uint8_t> result(request.ByteSizeLong());
    request.SerializeToArray(result.data(), result.size());
    return result;
  }

  bool deserialize_submit_response(const std::vector<uint8_t>& data,
                                   std::string& job_id,
                                   std::string& error_message) override
  {
    cuopt::remote::AsyncResponse response;
    if (!response.ParseFromArray(data.data(), data.size())) {
      error_message = "Failed to parse submit response";
      return false;
    }

    if (!response.has_submit_response()) {
      error_message = "Response is not a submit response";
      return false;
    }

    const auto& submit = response.submit_response();
    if (submit.status() != cuopt::remote::SUCCESS) {
      error_message = submit.message();
      return false;
    }

    job_id = submit.job_id();
    return true;
  }

  job_status_t deserialize_status_response(const std::vector<uint8_t>& data) override
  {
    cuopt::remote::AsyncResponse response;
    if (!response.ParseFromArray(data.data(), data.size()) || !response.has_status_response()) {
      return job_status_t::NOT_FOUND;
    }

    switch (response.status_response().job_status()) {
      case cuopt::remote::QUEUED: return job_status_t::QUEUED;
      case cuopt::remote::PROCESSING: return job_status_t::PROCESSING;
      case cuopt::remote::COMPLETED: return job_status_t::COMPLETED;
      case cuopt::remote::FAILED: return job_status_t::FAILED;
      case cuopt::remote::CANCELLED: return job_status_t::CANCELLED;
      case cuopt::remote::NOT_FOUND:
      default: return job_status_t::NOT_FOUND;
    }
  }

  optimization_problem_solution_t<i_t, f_t> deserialize_lp_result_response(
    const std::vector<uint8_t>& data) override
  {
    cuopt::remote::AsyncResponse response;
    if (!response.ParseFromArray(data.data(), data.size())) {
      return optimization_problem_solution_t<i_t, f_t>(
        cuopt::logic_error("Failed to parse result response", cuopt::error_type_t::RuntimeError));
    }

    if (!response.has_result_response()) {
      return optimization_problem_solution_t<i_t, f_t>(
        cuopt::logic_error("Response is not a result response", cuopt::error_type_t::RuntimeError));
    }

    const auto& result = response.result_response();
    if (result.status() != cuopt::remote::SUCCESS) {
      return optimization_problem_solution_t<i_t, f_t>(
        cuopt::logic_error(result.error_message(), cuopt::error_type_t::RuntimeError));
    }

    if (!result.has_lp_solution()) {
      return optimization_problem_solution_t<i_t, f_t>(cuopt::logic_error(
        "Response does not contain LP solution", cuopt::error_type_t::RuntimeError));
    }

    return proto_to_lp_solution(result.lp_solution());
  }

  mip_solution_t<i_t, f_t> deserialize_mip_result_response(
    const std::vector<uint8_t>& data) override
  {
    cuopt::remote::AsyncResponse response;
    if (!response.ParseFromArray(data.data(), data.size())) {
      return mip_solution_t<i_t, f_t>(
        cuopt::logic_error("Failed to parse result response", cuopt::error_type_t::RuntimeError));
    }

    if (!response.has_result_response()) {
      return mip_solution_t<i_t, f_t>(
        cuopt::logic_error("Response is not a result response", cuopt::error_type_t::RuntimeError));
    }

    const auto& result = response.result_response();
    if (result.status() != cuopt::remote::SUCCESS) {
      return mip_solution_t<i_t, f_t>(
        cuopt::logic_error(result.error_message(), cuopt::error_type_t::RuntimeError));
    }

    if (!result.has_mip_solution()) {
      return mip_solution_t<i_t, f_t>(cuopt::logic_error("Response does not contain MIP solution",
                                                         cuopt::error_type_t::RuntimeError));
    }

    return proto_to_mip_solution(result.mip_solution());
  }

  typename remote_serializer_t<i_t, f_t>::logs_result_t deserialize_logs_response(
    const std::vector<uint8_t>& data) override
  {
    typename remote_serializer_t<i_t, f_t>::logs_result_t result;
    result.nbytes     = 0;
    result.job_exists = false;

    cuopt::remote::AsyncResponse response;
    if (!response.ParseFromArray(data.data(), data.size()) || !response.has_logs_response()) {
      return result;
    }

    const auto& logs  = response.logs_response();
    result.job_exists = logs.job_exists();
    result.nbytes     = logs.nbytes();

    result.log_lines.reserve(logs.log_lines_size());
    for (int i = 0; i < logs.log_lines_size(); ++i) {
      result.log_lines.push_back(logs.log_lines(i));
    }

    return result;
  }

  typename remote_serializer_t<i_t, f_t>::cancel_result_t deserialize_cancel_response(
    const std::vector<uint8_t>& data) override
  {
    typename remote_serializer_t<i_t, f_t>::cancel_result_t result;
    result.success    = false;
    result.message    = "Failed to parse response";
    result.job_status = job_status_t::NOT_FOUND;

    cuopt::remote::AsyncResponse response;
    if (!response.ParseFromArray(data.data(), data.size()) || !response.has_cancel_response()) {
      return result;
    }

    const auto& cancel = response.cancel_response();
    result.success     = (cancel.status() == cuopt::remote::SUCCESS);
    result.message     = cancel.message();

    switch (cancel.job_status()) {
      case cuopt::remote::QUEUED: result.job_status = job_status_t::QUEUED; break;
      case cuopt::remote::PROCESSING: result.job_status = job_status_t::PROCESSING; break;
      case cuopt::remote::COMPLETED: result.job_status = job_status_t::COMPLETED; break;
      case cuopt::remote::FAILED: result.job_status = job_status_t::FAILED; break;
      case cuopt::remote::CANCELLED: result.job_status = job_status_t::CANCELLED; break;
      case cuopt::remote::NOT_FOUND:
      default: result.job_status = job_status_t::NOT_FOUND; break;
    }

    return result;
  }

  //============================================================================
  // Server-side Async Request Handling
  //============================================================================

  bool is_async_request(const std::vector<uint8_t>& data) override
  {
    // An AsyncRequest is characterized by having the request_type field set
    // and containing either lp_request or mip_request.
    // We can detect it by checking if it parses as AsyncRequest AND has a job_data set.
    cuopt::remote::AsyncRequest request;
    if (!request.ParseFromArray(data.data(), data.size())) { return false; }

    // AsyncRequest must have either lp_request or mip_request set
    // (the job_data oneof). If neither is set, it's not an async request
    // or it's a status/result/delete request that has job_id instead.
    bool has_job_data = request.has_lp_request() || request.has_mip_request();
    bool has_job_id   = !request.job_id().empty();

    // It's an async request if it has job_data OR job_id (for non-submit requests)
    return has_job_data || has_job_id;
  }

  int get_async_request_type(const std::vector<uint8_t>& data) override
  {
    cuopt::remote::AsyncRequest request;
    if (!request.ParseFromArray(data.data(), data.size())) { return -1; }

    switch (request.request_type()) {
      case cuopt::remote::SUBMIT_JOB: return 0;
      case cuopt::remote::CHECK_STATUS: return 1;
      case cuopt::remote::GET_RESULT: return 2;
      case cuopt::remote::DELETE_RESULT: return 3;
      case cuopt::remote::GET_LOGS: return 4;
      case cuopt::remote::CANCEL_JOB: return 5;
      case cuopt::remote::WAIT_FOR_RESULT: return 6;
      default: return -1;
    }
  }

  bool is_blocking_request(const std::vector<uint8_t>& data) override
  {
    cuopt::remote::AsyncRequest request;
    if (!request.ParseFromArray(data.data(), data.size())) { return false; }
    return request.blocking();
  }

  std::vector<uint8_t> extract_problem_data(const std::vector<uint8_t>& data) override
  {
    cuopt::remote::AsyncRequest request;
    if (!request.ParseFromArray(data.data(), data.size())) { return {}; }

    std::string serialized;
    if (request.has_lp_request()) {
      serialized = request.lp_request().SerializeAsString();
    } else if (request.has_mip_request()) {
      serialized = request.mip_request().SerializeAsString();
    } else {
      return {};
    }

    return std::vector<uint8_t>(serialized.begin(), serialized.end());
  }

  std::string get_job_id(const std::vector<uint8_t>& data) override
  {
    cuopt::remote::AsyncRequest request;
    if (!request.ParseFromArray(data.data(), data.size())) { return ""; }
    return request.job_id();
  }

  int64_t get_frombyte(const std::vector<uint8_t>& data) override
  {
    cuopt::remote::AsyncRequest request;
    if (!request.ParseFromArray(data.data(), data.size())) { return 0; }
    return request.frombyte();
  }

  std::vector<uint8_t> serialize_submit_response(bool success, const std::string& result) override
  {
    cuopt::remote::AsyncResponse response;
    response.set_request_type(cuopt::remote::SUBMIT_JOB);

    auto* submit = response.mutable_submit_response();
    if (success) {
      submit->set_status(cuopt::remote::SUCCESS);
      submit->set_job_id(result);
      submit->set_message("Job submitted successfully");
    } else {
      submit->set_status(cuopt::remote::ERROR_INTERNAL);
      submit->set_message(result);
    }

    std::vector<uint8_t> bytes(response.ByteSizeLong());
    response.SerializeToArray(bytes.data(), bytes.size());
    return bytes;
  }

  std::vector<uint8_t> serialize_status_response(int status_code,
                                                 const std::string& message) override
  {
    cuopt::remote::AsyncResponse response;
    response.set_request_type(cuopt::remote::CHECK_STATUS);

    auto* status = response.mutable_status_response();

    switch (status_code) {
      case 0: status->set_job_status(cuopt::remote::QUEUED); break;
      case 1: status->set_job_status(cuopt::remote::PROCESSING); break;
      case 2: status->set_job_status(cuopt::remote::COMPLETED); break;
      case 3: status->set_job_status(cuopt::remote::FAILED); break;
      case 4: status->set_job_status(cuopt::remote::NOT_FOUND); break;
      case 5: status->set_job_status(cuopt::remote::CANCELLED); break;
      default: status->set_job_status(cuopt::remote::NOT_FOUND); break;
    }
    status->set_message(message);

    std::vector<uint8_t> bytes(response.ByteSizeLong());
    response.SerializeToArray(bytes.data(), bytes.size());
    return bytes;
  }

  std::vector<uint8_t> serialize_result_response(bool success,
                                                 const std::vector<uint8_t>& result_data,
                                                 const std::string& error_message,
                                                 bool is_mip = false) override
  {
    cuopt::remote::AsyncResponse response;
    response.set_request_type(cuopt::remote::GET_RESULT);

    auto* result = response.mutable_result_response();

    if (success) {
      result->set_status(cuopt::remote::SUCCESS);
      // Parse and embed the solution based on problem type
      if (is_mip) {
        cuopt::remote::MIPSolution mip_sol;
        if (mip_sol.ParseFromArray(result_data.data(), result_data.size())) {
          result->mutable_mip_solution()->CopyFrom(mip_sol);
        }
      } else {
        cuopt::remote::LPSolution lp_sol;
        if (lp_sol.ParseFromArray(result_data.data(), result_data.size())) {
          result->mutable_lp_solution()->CopyFrom(lp_sol);
        }
      }
    } else {
      result->set_status(cuopt::remote::ERROR_INTERNAL);
      result->set_error_message(error_message);
    }

    std::vector<uint8_t> bytes(response.ByteSizeLong());
    response.SerializeToArray(bytes.data(), bytes.size());
    return bytes;
  }

  std::vector<uint8_t> serialize_delete_response(bool success) override
  {
    cuopt::remote::AsyncResponse response;
    response.set_request_type(cuopt::remote::DELETE_RESULT);

    auto* del = response.mutable_delete_response();
    del->set_status(success ? cuopt::remote::SUCCESS : cuopt::remote::ERROR_NOT_FOUND);
    del->set_message(success ? "Job deleted" : "Job not found");

    std::vector<uint8_t> bytes(response.ByteSizeLong());
    response.SerializeToArray(bytes.data(), bytes.size());
    return bytes;
  }

  std::vector<uint8_t> serialize_logs_response(const std::string& job_id,
                                               const std::vector<std::string>& log_lines,
                                               int64_t nbytes,
                                               bool job_exists) override
  {
    cuopt::remote::AsyncResponse response;
    response.set_request_type(cuopt::remote::GET_LOGS);

    auto* logs = response.mutable_logs_response();
    logs->set_status(job_exists ? cuopt::remote::SUCCESS : cuopt::remote::ERROR_NOT_FOUND);
    logs->set_job_id(job_id);
    logs->set_nbytes(nbytes);
    logs->set_job_exists(job_exists);

    for (const auto& line : log_lines) {
      logs->add_log_lines(line);
    }

    std::vector<uint8_t> bytes(response.ByteSizeLong());
    response.SerializeToArray(bytes.data(), bytes.size());
    return bytes;
  }

  std::vector<uint8_t> serialize_cancel_response(bool success,
                                                 const std::string& message,
                                                 int status_code) override
  {
    cuopt::remote::AsyncResponse response;
    response.set_request_type(cuopt::remote::CANCEL_JOB);

    auto* cancel = response.mutable_cancel_response();
    cancel->set_status(success ? cuopt::remote::SUCCESS : cuopt::remote::ERROR_INTERNAL);
    cancel->set_message(message);

    switch (status_code) {
      case 0: cancel->set_job_status(cuopt::remote::QUEUED); break;
      case 1: cancel->set_job_status(cuopt::remote::PROCESSING); break;
      case 2: cancel->set_job_status(cuopt::remote::COMPLETED); break;
      case 3: cancel->set_job_status(cuopt::remote::FAILED); break;
      case 4: cancel->set_job_status(cuopt::remote::NOT_FOUND); break;
      case 5: cancel->set_job_status(cuopt::remote::CANCELLED); break;
      default: cancel->set_job_status(cuopt::remote::NOT_FOUND); break;
    }

    std::vector<uint8_t> bytes(response.ByteSizeLong());
    response.SerializeToArray(bytes.data(), bytes.size());
    return bytes;
  }

  //============================================================================
  // Metadata
  //============================================================================

  std::string format_name() const override { return "protobuf"; }

  uint32_t protocol_version() const override { return 1; }

 private:
  //============================================================================
  // Helper Methods - Problem Serialization
  //============================================================================

  void serialize_problem_to_proto(const mps_parser::data_model_view_t<i_t, f_t>& view,
                                  cuopt::remote::OptimizationProblem* pb_problem)
  {
    // Note: view must point to CPU memory for serialization
    // The solve functions ensure this by copying GPU data to CPU if needed

    pb_problem->set_problem_name(view.get_problem_name());
    pb_problem->set_objective_name(view.get_objective_name());
    pb_problem->set_maximize(view.get_sense());  // get_sense() returns true for maximize
    pb_problem->set_objective_scaling_factor(view.get_objective_scaling_factor());
    pb_problem->set_objective_offset(view.get_objective_offset());

    // Get spans for constraint matrix (CSR format)
    auto values_span  = view.get_constraint_matrix_values();
    auto indices_span = view.get_constraint_matrix_indices();
    auto offsets_span = view.get_constraint_matrix_offsets();

    auto nnz    = static_cast<i_t>(values_span.size());
    auto n_rows = static_cast<i_t>(offsets_span.size()) - 1;

    const f_t* values_ptr  = values_span.data();
    const i_t* indices_ptr = indices_span.data();
    const i_t* offsets_ptr = offsets_span.data();

    // Constraint matrix A in CSR format (field names match data_model_view_t)
    for (i_t i = 0; i < nnz; ++i) {
      pb_problem->add_a(static_cast<double>(values_ptr[i]));
    }
    for (i_t i = 0; i < nnz; ++i) {
      pb_problem->add_a_indices(static_cast<int32_t>(indices_ptr[i]));
    }
    for (i_t i = 0; i <= n_rows; ++i) {
      pb_problem->add_a_offsets(static_cast<int32_t>(offsets_ptr[i]));
    }

    // Objective coefficients c (field name matches data_model_view_t: c_)
    auto obj_span      = view.get_objective_coefficients();
    auto n_cols        = static_cast<i_t>(obj_span.size());
    const f_t* obj_ptr = obj_span.data();
    for (i_t i = 0; i < n_cols; ++i) {
      pb_problem->add_c(static_cast<double>(obj_ptr[i]));
    }

    // Variable bounds
    auto lb_span      = view.get_variable_lower_bounds();
    auto ub_span      = view.get_variable_upper_bounds();
    const f_t* lb_ptr = lb_span.data();
    const f_t* ub_ptr = ub_span.data();
    for (i_t i = 0; i < n_cols; ++i) {
      pb_problem->add_variable_lower_bounds(static_cast<double>(lb_ptr[i]));
      pb_problem->add_variable_upper_bounds(static_cast<double>(ub_ptr[i]));
    }

    // Constraint bounds - need to handle both formats:
    // 1. Direct lower/upper bounds (set_constraint_lower/upper_bounds)
    // 2. RHS + row_types format (set_constraint_bounds + set_row_types)
    auto c_lb_span = view.get_constraint_lower_bounds();
    auto c_ub_span = view.get_constraint_upper_bounds();

    if (c_lb_span.size() == static_cast<size_t>(n_rows) &&
        c_ub_span.size() == static_cast<size_t>(n_rows)) {
      // Direct format - use as-is
      const f_t* c_lb_ptr = c_lb_span.data();
      const f_t* c_ub_ptr = c_ub_span.data();
      for (i_t i = 0; i < n_rows; ++i) {
        pb_problem->add_constraint_lower_bounds(static_cast<double>(c_lb_ptr[i]));
        pb_problem->add_constraint_upper_bounds(static_cast<double>(c_ub_ptr[i]));
      }
    } else {
      // RHS + row_types format - compute lower/upper bounds
      auto b_span         = view.get_constraint_bounds();
      auto row_types_span = view.get_row_types();
      const f_t* b_ptr    = b_span.data();
      const char* rt_ptr  = row_types_span.data();

      constexpr f_t inf = std::numeric_limits<f_t>::infinity();

      for (i_t i = 0; i < n_rows; ++i) {
        f_t lb, ub;
        char row_type = (rt_ptr && row_types_span.size() > 0) ? rt_ptr[i] : 'E';
        f_t rhs       = (b_ptr && b_span.size() > 0) ? b_ptr[i] : 0;

        switch (row_type) {
          case 'E':  // Equality: lb = ub = rhs
            lb = rhs;
            ub = rhs;
            break;
          case 'L':  // Less-than-or-equal: -inf <= Ax <= rhs
            lb = -inf;
            ub = rhs;
            break;
          case 'G':  // Greater-than-or-equal: rhs <= Ax <= inf
            lb = rhs;
            ub = inf;
            break;
          case 'N':  // Non-constraining (free)
            lb = -inf;
            ub = inf;
            break;
          default:  // Default to equality
            lb = rhs;
            ub = rhs;
            break;
        }
        pb_problem->add_constraint_lower_bounds(static_cast<double>(lb));
        pb_problem->add_constraint_upper_bounds(static_cast<double>(ub));
      }
    }

    // Variable names (if available)
    const auto& var_names = view.get_variable_names();
    for (const auto& name : var_names) {
      pb_problem->add_variable_names(name);
    }

    // Row names (if available)
    const auto& row_names = view.get_row_names();
    for (const auto& name : row_names) {
      pb_problem->add_row_names(name);
    }

    // Variable types (for MIP problems) - stored as bytes to match data_model_view_t
    auto var_types_span = view.get_variable_types();
    if (var_types_span.size() > 0) {
      pb_problem->set_variable_types(std::string(var_types_span.data(), var_types_span.size()));
    }

    // Row types - store directly as bytes
    auto row_types_span = view.get_row_types();
    if (row_types_span.size() > 0) {
      pb_problem->set_row_types(std::string(row_types_span.data(), row_types_span.size()));
    }

    // Constraint bounds b (RHS) - store directly if available
    auto b_span = view.get_constraint_bounds();
    if (b_span.size() > 0) {
      const f_t* b_ptr = b_span.data();
      for (size_t i = 0; i < b_span.size(); ++i) {
        pb_problem->add_b(static_cast<double>(b_ptr[i]));
      }
    }

    // Initial solutions (if available)
    auto init_primal_span = view.get_initial_primal_solution();
    if (init_primal_span.size() > 0) {
      const f_t* init_primal_ptr = init_primal_span.data();
      for (size_t i = 0; i < init_primal_span.size(); ++i) {
        pb_problem->add_initial_primal_solution(static_cast<double>(init_primal_ptr[i]));
      }
    }

    auto init_dual_span = view.get_initial_dual_solution();
    if (init_dual_span.size() > 0) {
      const f_t* init_dual_ptr = init_dual_span.data();
      for (size_t i = 0; i < init_dual_span.size(); ++i) {
        pb_problem->add_initial_dual_solution(static_cast<double>(init_dual_ptr[i]));
      }
    }

    // Quadratic objective matrix Q (for QPS problems)
    if (view.has_quadratic_objective()) {
      auto q_values_span  = view.get_quadratic_objective_values();
      auto q_indices_span = view.get_quadratic_objective_indices();
      auto q_offsets_span = view.get_quadratic_objective_offsets();

      const f_t* q_values_ptr  = q_values_span.data();
      const i_t* q_indices_ptr = q_indices_span.data();
      const i_t* q_offsets_ptr = q_offsets_span.data();

      for (size_t i = 0; i < q_values_span.size(); ++i) {
        pb_problem->add_q_values(static_cast<double>(q_values_ptr[i]));
      }
      for (size_t i = 0; i < q_indices_span.size(); ++i) {
        pb_problem->add_q_indices(static_cast<int32_t>(q_indices_ptr[i]));
      }
      for (size_t i = 0; i < q_offsets_span.size(); ++i) {
        pb_problem->add_q_offsets(static_cast<int32_t>(q_offsets_ptr[i]));
      }
    }
  }

  // Convert cuOpt pdlp_solver_mode_t to protobuf enum
  cuopt::remote::PDLPSolverMode to_proto_pdlp_mode(pdlp_solver_mode_t mode)
  {
    switch (mode) {
      case pdlp_solver_mode_t::Stable1: return cuopt::remote::Stable1;
      case pdlp_solver_mode_t::Stable2: return cuopt::remote::Stable2;
      case pdlp_solver_mode_t::Methodical1: return cuopt::remote::Methodical1;
      case pdlp_solver_mode_t::Fast1: return cuopt::remote::Fast1;
      case pdlp_solver_mode_t::Stable3: return cuopt::remote::Stable3;
      default: return cuopt::remote::Stable3;
    }
  }

  // Convert cuOpt method_t to protobuf enum
  cuopt::remote::LPMethod to_proto_method(method_t method)
  {
    switch (method) {
      case method_t::Concurrent: return cuopt::remote::Concurrent;
      case method_t::PDLP: return cuopt::remote::PDLP;
      case method_t::DualSimplex: return cuopt::remote::DualSimplex;
      case method_t::Barrier: return cuopt::remote::Barrier;
      default: return cuopt::remote::Concurrent;
    }
  }

  void serialize_lp_settings_to_proto(const pdlp_solver_settings_t<i_t, f_t>& settings,
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
    pb_settings->set_presolve(settings.presolve);
    pb_settings->set_dual_postsolve(settings.dual_postsolve);
    pb_settings->set_crossover(settings.crossover);
    pb_settings->set_num_gpus(settings.num_gpus);

    // Advanced options
    pb_settings->set_per_constraint_residual(settings.per_constraint_residual);
    pb_settings->set_cudss_deterministic(settings.cudss_deterministic);
    pb_settings->set_folding(settings.folding);
    pb_settings->set_augmented(settings.augmented);
    pb_settings->set_dualize(settings.dualize);
    pb_settings->set_ordering(settings.ordering);
    pb_settings->set_barrier_dual_initial_point(settings.barrier_dual_initial_point);
    pb_settings->set_eliminate_dense_columns(settings.eliminate_dense_columns);
    pb_settings->set_save_best_primal_so_far(settings.save_best_primal_so_far);
    pb_settings->set_first_primal_feasible(settings.first_primal_feasible);
  }

  //============================================================================
  // Helper Methods - Problem Deserialization
  //============================================================================

  void proto_to_mps_data(const cuopt::remote::OptimizationProblem& pb_problem,
                         cuopt::mps_parser::mps_data_model_t<i_t, f_t>& mps_data)
  {
    mps_data.set_problem_name(pb_problem.problem_name());
    mps_data.set_objective_name(pb_problem.objective_name());
    mps_data.set_maximize(pb_problem.maximize());
    mps_data.set_objective_scaling_factor(pb_problem.objective_scaling_factor());
    mps_data.set_objective_offset(pb_problem.objective_offset());

    // Constraint matrix A in CSR format (field names match data_model_view_t)
    std::vector<f_t> values(pb_problem.a().begin(), pb_problem.a().end());
    std::vector<i_t> indices(pb_problem.a_indices().begin(), pb_problem.a_indices().end());
    std::vector<i_t> offsets(pb_problem.a_offsets().begin(), pb_problem.a_offsets().end());

    mps_data.set_csr_constraint_matrix(values.data(),
                                       static_cast<i_t>(values.size()),
                                       indices.data(),
                                       static_cast<i_t>(indices.size()),
                                       offsets.data(),
                                       static_cast<i_t>(offsets.size()));

    // Objective coefficients c
    std::vector<f_t> obj(pb_problem.c().begin(), pb_problem.c().end());
    mps_data.set_objective_coefficients(obj.data(), static_cast<i_t>(obj.size()));

    // Variable bounds
    std::vector<f_t> var_lb(pb_problem.variable_lower_bounds().begin(),
                            pb_problem.variable_lower_bounds().end());
    std::vector<f_t> var_ub(pb_problem.variable_upper_bounds().begin(),
                            pb_problem.variable_upper_bounds().end());
    mps_data.set_variable_lower_bounds(var_lb.data(), static_cast<i_t>(var_lb.size()));
    mps_data.set_variable_upper_bounds(var_ub.data(), static_cast<i_t>(var_ub.size()));

    // Constraint bounds (prefer lower/upper bounds if available)
    if (pb_problem.constraint_lower_bounds_size() > 0) {
      std::vector<f_t> con_lb(pb_problem.constraint_lower_bounds().begin(),
                              pb_problem.constraint_lower_bounds().end());
      std::vector<f_t> con_ub(pb_problem.constraint_upper_bounds().begin(),
                              pb_problem.constraint_upper_bounds().end());
      mps_data.set_constraint_lower_bounds(con_lb.data(), static_cast<i_t>(con_lb.size()));
      mps_data.set_constraint_upper_bounds(con_ub.data(), static_cast<i_t>(con_ub.size()));
    } else if (pb_problem.b_size() > 0) {
      // Use b (RHS) + row_types format
      std::vector<f_t> b(pb_problem.b().begin(), pb_problem.b().end());
      mps_data.set_constraint_bounds(b.data(), static_cast<i_t>(b.size()));

      if (!pb_problem.row_types().empty()) {
        const std::string& row_types = pb_problem.row_types();
        mps_data.set_row_types(row_types.data(), static_cast<i_t>(row_types.size()));
      }
    }

    // Variable names
    if (pb_problem.variable_names_size() > 0) {
      std::vector<std::string> var_names(pb_problem.variable_names().begin(),
                                         pb_problem.variable_names().end());
      mps_data.set_variable_names(var_names);
    }

    // Row names
    if (pb_problem.row_names_size() > 0) {
      std::vector<std::string> row_names(pb_problem.row_names().begin(),
                                         pb_problem.row_names().end());
      mps_data.set_row_names(row_names);
    }

    // Variable types (stored as bytes, matching data_model_view_t)
    if (!pb_problem.variable_types().empty()) {
      const std::string& var_types_str = pb_problem.variable_types();
      std::vector<char> var_types(var_types_str.begin(), var_types_str.end());
      mps_data.set_variable_types(var_types);
    }

    // Initial solutions (if provided)
    if (pb_problem.initial_primal_solution_size() > 0) {
      std::vector<f_t> init_primal(pb_problem.initial_primal_solution().begin(),
                                   pb_problem.initial_primal_solution().end());
      mps_data.set_initial_primal_solution(init_primal.data(),
                                           static_cast<i_t>(init_primal.size()));
    }

    if (pb_problem.initial_dual_solution_size() > 0) {
      std::vector<f_t> init_dual(pb_problem.initial_dual_solution().begin(),
                                 pb_problem.initial_dual_solution().end());
      mps_data.set_initial_dual_solution(init_dual.data(), static_cast<i_t>(init_dual.size()));
    }

    // Quadratic objective matrix Q (for QPS problems)
    if (pb_problem.q_values_size() > 0) {
      std::vector<f_t> q_values(pb_problem.q_values().begin(), pb_problem.q_values().end());
      std::vector<i_t> q_indices(pb_problem.q_indices().begin(), pb_problem.q_indices().end());
      std::vector<i_t> q_offsets(pb_problem.q_offsets().begin(), pb_problem.q_offsets().end());

      mps_data.set_quadratic_objective_matrix(q_values.data(),
                                              static_cast<i_t>(q_values.size()),
                                              q_indices.data(),
                                              static_cast<i_t>(q_indices.size()),
                                              q_offsets.data(),
                                              static_cast<i_t>(q_offsets.size()));
    }
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
      default: return pdlp_solver_mode_t::Stable3;
    }
  }

  // Convert protobuf enum to cuOpt method_t
  method_t from_proto_method(cuopt::remote::LPMethod method)
  {
    switch (method) {
      case cuopt::remote::Concurrent: return method_t::Concurrent;
      case cuopt::remote::PDLP: return method_t::PDLP;
      case cuopt::remote::DualSimplex: return method_t::DualSimplex;
      case cuopt::remote::Barrier: return method_t::Barrier;
      default: return method_t::Concurrent;
    }
  }

  void proto_to_lp_settings(const cuopt::remote::PDLPSolverSettings& pb_settings,
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
      settings.iteration_limit = static_cast<i_t>(pb_settings.iteration_limit());
    }

    // Solver configuration
    settings.log_to_console       = pb_settings.log_to_console();
    settings.detect_infeasibility = pb_settings.detect_infeasibility();
    settings.strict_infeasibility = pb_settings.strict_infeasibility();
    settings.pdlp_solver_mode     = from_proto_pdlp_mode(pb_settings.pdlp_solver_mode());
    settings.method               = from_proto_method(pb_settings.method());
    settings.presolve             = pb_settings.presolve();
    settings.dual_postsolve       = pb_settings.dual_postsolve();
    settings.crossover            = pb_settings.crossover();
    settings.num_gpus             = pb_settings.num_gpus();

    // Advanced options
    settings.per_constraint_residual    = pb_settings.per_constraint_residual();
    settings.cudss_deterministic        = pb_settings.cudss_deterministic();
    settings.folding                    = pb_settings.folding();
    settings.augmented                  = pb_settings.augmented();
    settings.dualize                    = pb_settings.dualize();
    settings.ordering                   = pb_settings.ordering();
    settings.barrier_dual_initial_point = pb_settings.barrier_dual_initial_point();
    settings.eliminate_dense_columns    = pb_settings.eliminate_dense_columns();
    settings.save_best_primal_so_far    = pb_settings.save_best_primal_so_far();
    settings.first_primal_feasible      = pb_settings.first_primal_feasible();
  }

  void proto_to_mip_settings(const cuopt::remote::MIPSolverSettings& pb_settings,
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
    settings.presolve        = pb_settings.presolve();
    settings.mip_scaling     = pb_settings.mip_scaling();
  }

  //============================================================================
  // Helper Methods - Solution Conversion
  //============================================================================

  optimization_problem_solution_t<i_t, f_t> proto_to_lp_solution(
    const cuopt::remote::LPSolution& pb_solution)
  {
    // Create CPU-based solution
    std::vector<f_t> primal(pb_solution.primal_solution().begin(),
                            pb_solution.primal_solution().end());
    std::vector<f_t> dual(pb_solution.dual_solution().begin(), pb_solution.dual_solution().end());
    std::vector<f_t> reduced_cost(pb_solution.reduced_cost().begin(),
                                  pb_solution.reduced_cost().end());

    optimization_problem_solution_t<i_t, f_t> solution(
      from_proto_status(pb_solution.termination_status()));

    // Set solution data
    solution.set_primal_solution_host(std::move(primal));
    solution.set_dual_solution_host(std::move(dual));
    solution.set_reduced_cost_host(std::move(reduced_cost));

    // Set statistics
    solution.set_l2_primal_residual(pb_solution.l2_primal_residual());
    solution.set_l2_dual_residual(pb_solution.l2_dual_residual());
    solution.set_primal_objective(pb_solution.primal_objective());
    solution.set_dual_objective(pb_solution.dual_objective());
    solution.set_gap(pb_solution.gap());
    solution.set_nb_iterations(pb_solution.nb_iterations());
    solution.set_solve_time(pb_solution.solve_time());
    solution.set_solved_by_pdlp(pb_solution.solved_by_pdlp());

    return solution;
  }

  void lp_solution_to_proto(const optimization_problem_solution_t<i_t, f_t>& solution,
                            cuopt::remote::LPSolution* pb_solution)
  {
    pb_solution->set_termination_status(to_proto_status(solution.get_termination_status()));
    pb_solution->set_error_message(solution.get_error_string());

    // Solution vectors - handle both device and host memory
    if (solution.is_device_memory()) {
      // Copy from device to host
      const auto& d_primal = solution.get_primal_solution();
      const auto& d_dual   = solution.get_dual_solution();
      // Note: reduced_cost getter is non-const, so we need to work around this

      // Copy primal solution from device
      if (d_primal.size() > 0) {
        std::vector<f_t> h_primal(d_primal.size());
        cudaMemcpy(
          h_primal.data(), d_primal.data(), d_primal.size() * sizeof(f_t), cudaMemcpyDeviceToHost);
        for (const auto& v : h_primal) {
          pb_solution->add_primal_solution(static_cast<double>(v));
        }
      }

      // Copy dual solution from device
      if (d_dual.size() > 0) {
        std::vector<f_t> h_dual(d_dual.size());
        cudaMemcpy(
          h_dual.data(), d_dual.data(), d_dual.size() * sizeof(f_t), cudaMemcpyDeviceToHost);
        for (const auto& v : h_dual) {
          pb_solution->add_dual_solution(static_cast<double>(v));
        }
      }

      // For reduced cost, we can access via const cast since we're just reading
      auto& nc_solution    = const_cast<optimization_problem_solution_t<i_t, f_t>&>(solution);
      auto& d_reduced_cost = nc_solution.get_reduced_cost();
      if (d_reduced_cost.size() > 0) {
        std::vector<f_t> h_reduced_cost(d_reduced_cost.size());
        cudaMemcpy(h_reduced_cost.data(),
                   d_reduced_cost.data(),
                   d_reduced_cost.size() * sizeof(f_t),
                   cudaMemcpyDeviceToHost);
        for (const auto& v : h_reduced_cost) {
          pb_solution->add_reduced_cost(static_cast<double>(v));
        }
      }
    } else {
      // Data is already on host
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
    }

    // Statistics
    pb_solution->set_l2_primal_residual(solution.get_l2_primal_residual());
    pb_solution->set_l2_dual_residual(solution.get_l2_dual_residual());
    pb_solution->set_primal_objective(solution.get_primal_objective());
    pb_solution->set_dual_objective(solution.get_dual_objective());
    pb_solution->set_gap(solution.get_gap());
    pb_solution->set_nb_iterations(solution.get_nb_iterations());
    pb_solution->set_solve_time(solution.get_solve_time());
    pb_solution->set_solved_by_pdlp(solution.get_solved_by_pdlp());
  }

  mip_solution_t<i_t, f_t> proto_to_mip_solution(const cuopt::remote::MIPSolution& pb_solution)
  {
    std::vector<f_t> solution_vec(pb_solution.solution().begin(), pb_solution.solution().end());

    // Create stats from protobuf data
    solver_stats_t<i_t, f_t> stats;
    stats.total_solve_time       = pb_solution.total_solve_time();
    stats.presolve_time          = pb_solution.presolve_time();
    stats.solution_bound         = pb_solution.solution_bound();
    stats.num_nodes              = pb_solution.nodes();
    stats.num_simplex_iterations = pb_solution.simplex_iterations();

    mip_solution_t<i_t, f_t> solution(from_proto_mip_status(pb_solution.termination_status()),
                                      stats);

    solution.set_solution_host(std::move(solution_vec));
    solution.set_objective(pb_solution.objective());
    solution.set_mip_gap(pb_solution.mip_gap());
    solution.set_max_constraint_violation(pb_solution.max_constraint_violation());
    solution.set_max_int_violation(pb_solution.max_int_violation());
    solution.set_max_variable_bound_violation(pb_solution.max_variable_bound_violation());

    return solution;
  }

  void mip_solution_to_proto(const mip_solution_t<i_t, f_t>& solution,
                             cuopt::remote::MIPSolution* pb_solution)
  {
    pb_solution->set_termination_status(to_proto_mip_status(solution.get_termination_status()));
    pb_solution->set_error_message(solution.get_error_string());

    // Handle both device and host memory
    if (solution.is_device_memory()) {
      const auto& d_sol = solution.get_solution();
      if (d_sol.size() > 0) {
        std::vector<f_t> h_sol(d_sol.size());
        cudaMemcpy(h_sol.data(), d_sol.data(), d_sol.size() * sizeof(f_t), cudaMemcpyDeviceToHost);
        for (const auto& v : h_sol) {
          pb_solution->add_solution(static_cast<double>(v));
        }
      }
    } else {
      const auto& sol_vec = solution.get_solution_host();
      for (const auto& v : sol_vec) {
        pb_solution->add_solution(static_cast<double>(v));
      }
    }

    pb_solution->set_objective(solution.get_objective_value());
    pb_solution->set_mip_gap(solution.get_mip_gap());
    pb_solution->set_solution_bound(solution.get_solution_bound());
    pb_solution->set_total_solve_time(solution.get_total_solve_time());
    pb_solution->set_presolve_time(solution.get_presolve_time());
    pb_solution->set_max_constraint_violation(solution.get_max_constraint_violation());
    pb_solution->set_max_int_violation(solution.get_max_int_violation());
    pb_solution->set_max_variable_bound_violation(solution.get_max_variable_bound_violation());
    pb_solution->set_nodes(solution.get_num_nodes());
    pb_solution->set_simplex_iterations(solution.get_num_simplex_iterations());
  }
};

//============================================================================
// Template Instantiations
// Note: Only int32_t and double types are instantiated to avoid adding
// int64_t instantiations throughout the codebase
//============================================================================

#if CUOPT_INSTANTIATE_FLOAT
template class protobuf_serializer_t<int32_t, float>;
#endif

#if CUOPT_INSTANTIATE_DOUBLE
template class protobuf_serializer_t<int32_t, double>;
#endif

//============================================================================
// Factory Functions
//============================================================================

template <typename i_t, typename f_t>
std::shared_ptr<remote_serializer_t<i_t, f_t>> get_default_serializer()
{
  static auto instance = std::make_shared<protobuf_serializer_t<i_t, f_t>>();
  return instance;
}

// Explicit instantiations for factory functions
#if CUOPT_INSTANTIATE_FLOAT
template std::shared_ptr<remote_serializer_t<int32_t, float>> get_default_serializer();
#endif

#if CUOPT_INSTANTIATE_DOUBLE
template std::shared_ptr<remote_serializer_t<int32_t, double>> get_default_serializer();
#endif

// Custom serializer loader (lazy-initialized)
template <typename i_t, typename f_t>
std::shared_ptr<remote_serializer_t<i_t, f_t>> get_serializer()
{
  static std::shared_ptr<remote_serializer_t<i_t, f_t>> instance;
  static std::once_flag init_flag;

  std::call_once(init_flag, []() {
    const char* custom_lib = std::getenv("CUOPT_SERIALIZER_LIB");

    if (custom_lib && custom_lib[0] != '\0') {
      // Try to load custom serializer
      CUOPT_LOG_INFO(std::string("[remote_solve] Loading custom serializer from: ") + custom_lib);

      // Open the shared library
      void* handle = dlopen(custom_lib, RTLD_NOW | RTLD_LOCAL);
      if (!handle) {
        CUOPT_LOG_ERROR(std::string("[remote_solve] Failed to load serializer library: ") +
                        dlerror());
        instance = get_default_serializer<i_t, f_t>();
        return;
      }

      // Look for the factory function
      // The function name includes template types for proper linking
      std::string factory_name = "create_cuopt_serializer";
      if constexpr (std::is_same_v<i_t, int32_t> && std::is_same_v<f_t, double>) {
        factory_name = "create_cuopt_serializer_i32_f64";
      } else if constexpr (std::is_same_v<i_t, int32_t> && std::is_same_v<f_t, float>) {
        factory_name = "create_cuopt_serializer_i32_f32";
      } else if constexpr (std::is_same_v<i_t, int64_t> && std::is_same_v<f_t, double>) {
        factory_name = "create_cuopt_serializer_i64_f64";
      } else if constexpr (std::is_same_v<i_t, int64_t> && std::is_same_v<f_t, float>) {
        factory_name = "create_cuopt_serializer_i64_f32";
      }

      using factory_fn_t = std::unique_ptr<remote_serializer_t<i_t, f_t>> (*)();
      auto factory       = reinterpret_cast<factory_fn_t>(dlsym(handle, factory_name.c_str()));

      if (!factory) {
        CUOPT_LOG_ERROR(std::string("[remote_solve] Factory function '") + factory_name +
                        "' not found: " + dlerror());
        dlclose(handle);
        instance = get_default_serializer<i_t, f_t>();
        return;
      }

      auto custom_serializer = factory();
      if (custom_serializer) {
        CUOPT_LOG_INFO(std::string("[remote_solve] Using custom serializer: ") +
                       custom_serializer->format_name());
        instance = std::move(custom_serializer);
      } else {
        CUOPT_LOG_ERROR("[remote_solve] Factory returned null, using default");
        dlclose(handle);
        instance = get_default_serializer<i_t, f_t>();
      }
      // Note: We intentionally don't dlclose(handle) here to keep the library loaded
    } else {
      instance = get_default_serializer<i_t, f_t>();
    }
  });

  return instance;
}

// Explicit instantiations
#if CUOPT_INSTANTIATE_FLOAT
template std::shared_ptr<remote_serializer_t<int32_t, float>> get_serializer();
#endif

#if CUOPT_INSTANTIATE_DOUBLE
template std::shared_ptr<remote_serializer_t<int32_t, double>> get_serializer();
#endif

}  // namespace cuopt::linear_programming
