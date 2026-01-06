/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved. SPDX-License-Identifier: Apache-2.0
 */

#include <cuopt/linear_programming/constants.h>
#include <cuopt_remote.pb.h>
#include <cuopt/linear_programming/utilities/remote_serialization.hpp>

#include <utilities/logger.hpp>

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

    // Serialize settings
    auto* pb_settings = request.mutable_settings();
    pb_settings->set_time_limit(settings.time_limit);
    pb_settings->set_mip_gap(settings.tolerances.relative_mip_gap);
    // Note: verbosity not directly available in mip_solver_settings_t
    pb_settings->set_verbosity(0);

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
      return async_request.has_mip_request();
    }

    // Try LP request
    cuopt::remote::SolveLPRequest lp_request;
    if (lp_request.ParseFromArray(data.data(), data.size())) { return false; }

    // Try MIP request
    cuopt::remote::SolveMIPRequest mip_request;
    if (mip_request.ParseFromArray(data.data(), data.size())) { return true; }

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

    auto* pb_settings = mip_request->mutable_settings();
    pb_settings->set_time_limit(settings.time_limit);
    pb_settings->set_mip_gap(settings.tolerances.relative_mip_gap);
    pb_settings->set_verbosity(0);

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

    for (i_t i = 0; i < nnz; ++i) {
      pb_problem->add_constraint_matrix_values(static_cast<double>(values_ptr[i]));
    }
    for (i_t i = 0; i < nnz; ++i) {
      pb_problem->add_constraint_matrix_indices(static_cast<int32_t>(indices_ptr[i]));
    }
    for (i_t i = 0; i <= n_rows; ++i) {
      pb_problem->add_constraint_matrix_offsets(static_cast<int32_t>(offsets_ptr[i]));
    }

    // Objective coefficients
    auto obj_span      = view.get_objective_coefficients();
    auto n_cols        = static_cast<i_t>(obj_span.size());
    const f_t* obj_ptr = obj_span.data();
    for (i_t i = 0; i < n_cols; ++i) {
      pb_problem->add_objective_coefficients(static_cast<double>(obj_ptr[i]));
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

    // Constraint bounds
    auto c_lb_span      = view.get_constraint_lower_bounds();
    auto c_ub_span      = view.get_constraint_upper_bounds();
    const f_t* c_lb_ptr = c_lb_span.data();
    const f_t* c_ub_ptr = c_ub_span.data();
    for (i_t i = 0; i < n_rows; ++i) {
      pb_problem->add_constraint_lower_bounds(static_cast<double>(c_lb_ptr[i]));
      pb_problem->add_constraint_upper_bounds(static_cast<double>(c_ub_ptr[i]));
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
  }

  void serialize_lp_settings_to_proto(const pdlp_solver_settings_t<i_t, f_t>& settings,
                                      cuopt::remote::PDLPSolverSettings* pb_settings)
  {
    // Map from cuOpt tolerances to protobuf settings
    pb_settings->set_eps_optimal_absolute(settings.tolerances.absolute_gap_tolerance);
    pb_settings->set_eps_optimal_relative(settings.tolerances.relative_gap_tolerance);
    pb_settings->set_eps_primal_infeasible(settings.tolerances.primal_infeasible_tolerance);
    pb_settings->set_eps_dual_infeasible(settings.tolerances.dual_infeasible_tolerance);

    // Handle infinity time_limit: use -1 to signal "no limit" in the protobuf
    // This avoids undefined behavior when casting infinity to int32_t
    int32_t time_limit_sec = -1;  // -1 means no limit
    if (std::isfinite(settings.time_limit) && settings.time_limit > 0) {
      time_limit_sec = static_cast<int32_t>(
        std::min(settings.time_limit, static_cast<double>(std::numeric_limits<int32_t>::max())));
    }
    pb_settings->set_time_sec_limit(time_limit_sec);

    // Handle max iteration limit similarly
    int32_t iter_limit = -1;  // -1 means no limit
    if (settings.iteration_limit > 0 &&
        settings.iteration_limit < std::numeric_limits<i_t>::max()) {
      iter_limit =
        static_cast<int32_t>(std::min(static_cast<int64_t>(settings.iteration_limit),
                                      static_cast<int64_t>(std::numeric_limits<int32_t>::max())));
    }
    pb_settings->set_iteration_limit(iter_limit);

    // initial_primal_weight and initial_step_size are not directly accessible, use defaults
    pb_settings->set_initial_primal_weight(1.0);
    pb_settings->set_initial_step_size(1.0);
    pb_settings->set_verbosity(settings.log_to_console ? 1 : 0);
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

    // Copy constraint matrix - mps_data_model_t copies the data
    std::vector<f_t> values(pb_problem.constraint_matrix_values().begin(),
                            pb_problem.constraint_matrix_values().end());
    std::vector<i_t> indices(pb_problem.constraint_matrix_indices().begin(),
                             pb_problem.constraint_matrix_indices().end());
    std::vector<i_t> offsets(pb_problem.constraint_matrix_offsets().begin(),
                             pb_problem.constraint_matrix_offsets().end());

    mps_data.set_csr_constraint_matrix(values.data(),
                                       static_cast<i_t>(values.size()),
                                       indices.data(),
                                       static_cast<i_t>(indices.size()),
                                       offsets.data(),
                                       static_cast<i_t>(offsets.size()));

    // Objective coefficients
    std::vector<f_t> obj(pb_problem.objective_coefficients().begin(),
                         pb_problem.objective_coefficients().end());
    mps_data.set_objective_coefficients(obj.data(), static_cast<i_t>(obj.size()));

    // Variable bounds
    std::vector<f_t> var_lb(pb_problem.variable_lower_bounds().begin(),
                            pb_problem.variable_lower_bounds().end());
    std::vector<f_t> var_ub(pb_problem.variable_upper_bounds().begin(),
                            pb_problem.variable_upper_bounds().end());
    mps_data.set_variable_lower_bounds(var_lb.data(), static_cast<i_t>(var_lb.size()));
    mps_data.set_variable_upper_bounds(var_ub.data(), static_cast<i_t>(var_ub.size()));

    // Constraint bounds
    std::vector<f_t> con_lb(pb_problem.constraint_lower_bounds().begin(),
                            pb_problem.constraint_lower_bounds().end());
    std::vector<f_t> con_ub(pb_problem.constraint_upper_bounds().begin(),
                            pb_problem.constraint_upper_bounds().end());
    mps_data.set_constraint_lower_bounds(con_lb.data(), static_cast<i_t>(con_lb.size()));
    mps_data.set_constraint_upper_bounds(con_ub.data(), static_cast<i_t>(con_ub.size()));

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
  }

  void proto_to_lp_settings(const cuopt::remote::PDLPSolverSettings& pb_settings,
                            pdlp_solver_settings_t<i_t, f_t>& settings)
  {
    // Map from protobuf settings to cuOpt tolerances
    settings.tolerances.absolute_gap_tolerance      = pb_settings.eps_optimal_absolute();
    settings.tolerances.relative_gap_tolerance      = pb_settings.eps_optimal_relative();
    settings.tolerances.primal_infeasible_tolerance = pb_settings.eps_primal_infeasible();
    settings.tolerances.dual_infeasible_tolerance   = pb_settings.eps_dual_infeasible();

    // Handle time limit: -1 or 0 means no limit (infinity)
    int32_t time_limit_sec = pb_settings.time_sec_limit();
    if (time_limit_sec <= 0) {
      settings.time_limit = std::numeric_limits<double>::infinity();
    } else {
      settings.time_limit = static_cast<double>(time_limit_sec);
    }

    // Handle iteration limit: -1 or 0 means no limit
    int32_t iter_limit = pb_settings.iteration_limit();
    if (iter_limit <= 0) {
      settings.iteration_limit = std::numeric_limits<i_t>::max();
    } else {
      settings.iteration_limit = static_cast<i_t>(iter_limit);
    }

    settings.log_to_console = pb_settings.verbosity() > 0;
  }

  void proto_to_mip_settings(const cuopt::remote::MIPSolverSettings& pb_settings,
                             mip_solver_settings_t<i_t, f_t>& settings)
  {
    settings.time_limit                  = pb_settings.time_limit();
    settings.tolerances.relative_mip_gap = pb_settings.mip_gap();
    // Note: verbosity not directly supported in mip_solver_settings_t
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
      CUOPT_LOG_INFO("[remote_solve] Loading custom serializer from: {}", custom_lib);

      // Dynamic loading would go here
      // For now, fall back to default
      CUOPT_LOG_WARN("[remote_solve] Custom serializer loading not yet implemented, using default");
      instance = get_default_serializer<i_t, f_t>();
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
