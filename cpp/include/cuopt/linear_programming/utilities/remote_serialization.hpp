/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved. SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cuopt/linear_programming/mip/solver_settings.hpp>
#include <cuopt/linear_programming/mip/solver_solution.hpp>
#include <cuopt/linear_programming/pdlp/solver_settings.hpp>
#include <cuopt/linear_programming/pdlp/solver_solution.hpp>
#include <mps_parser/data_model_view.hpp>
#include <mps_parser/mps_data_model.hpp>

#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace cuopt::linear_programming {

/**
 * @brief Abstract interface for serializing/deserializing cuOpt problems and solutions.
 *
 * This interface allows users to provide custom serialization implementations
 * for different wire formats (protobuf, JSON, msgpack, custom binary, etc.).
 *
 * The default implementation uses Protocol Buffers and is built into libcuopt.
 * Users can provide their own implementation by:
 * 1. Implementing this interface
 * 2. Compiling to a shared library
 * 3. Setting CUOPT_SERIALIZER_LIB environment variable to the library path
 *
 * @tparam i_t Index type (int32_t or int64_t)
 * @tparam f_t Float type (float or double)
 */
template <typename i_t, typename f_t>
class remote_serializer_t {
 public:
  virtual ~remote_serializer_t() = default;

  //============================================================================
  // Problem Serialization
  //============================================================================

  /**
   * @brief Serialize an LP problem with settings to a byte buffer.
   *
   * @param view The problem data view (can point to CPU or GPU memory)
   * @param settings Solver settings
   * @return Serialized byte buffer ready for network transmission
   */
  virtual std::vector<uint8_t> serialize_lp_request(
    const mps_parser::data_model_view_t<i_t, f_t>& view,
    const pdlp_solver_settings_t<i_t, f_t>& settings) = 0;

  /**
   * @brief Serialize a MIP problem with settings to a byte buffer.
   *
   * @param view The problem data view (can point to CPU or GPU memory)
   * @param settings Solver settings
   * @return Serialized byte buffer ready for network transmission
   */
  virtual std::vector<uint8_t> serialize_mip_request(
    const mps_parser::data_model_view_t<i_t, f_t>& view,
    const mip_solver_settings_t<i_t, f_t>& settings) = 0;

  //============================================================================
  // Solution Deserialization
  //============================================================================

  /**
   * @brief Deserialize an LP solution from a byte buffer.
   *
   * @param data The serialized solution bytes received from the server
   * @return The deserialized LP solution object
   */
  virtual optimization_problem_solution_t<i_t, f_t> deserialize_lp_solution(
    const std::vector<uint8_t>& data) = 0;

  /**
   * @brief Deserialize a MIP solution from a byte buffer.
   *
   * @param data The serialized solution bytes received from the server
   * @return The deserialized MIP solution object
   */
  virtual mip_solution_t<i_t, f_t> deserialize_mip_solution(const std::vector<uint8_t>& data) = 0;

  //============================================================================
  // Server-side: Request Deserialization & Response Serialization
  //============================================================================

  /**
   * @brief Check if serialized data is an LP or MIP request.
   *
   * @param data The serialized request bytes
   * @return true if MIP request, false if LP request
   */
  virtual bool is_mip_request(const std::vector<uint8_t>& data) = 0;

  /**
   * @brief Deserialize an LP request (problem + settings) from bytes.
   *
   * This is used by the server to receive problems from clients.
   *
   * @param data The serialized request bytes
   * @param[out] view_data CPU storage that will be populated with problem data
   * @param[out] settings Settings will be populated here
   * @return true on success, false on parse error
   */
  virtual bool deserialize_lp_request(const std::vector<uint8_t>& data,
                                      mps_parser::mps_data_model_t<i_t, f_t>& view_data,
                                      pdlp_solver_settings_t<i_t, f_t>& settings) = 0;

  /**
   * @brief Deserialize a MIP request (problem + settings) from bytes.
   *
   * @param data The serialized request bytes
   * @param[out] view_data CPU storage that will be populated with problem data
   * @param[out] settings Settings will be populated here
   * @return true on success, false on parse error
   */
  virtual bool deserialize_mip_request(const std::vector<uint8_t>& data,
                                       mps_parser::mps_data_model_t<i_t, f_t>& view_data,
                                       mip_solver_settings_t<i_t, f_t>& settings) = 0;

  /**
   * @brief Serialize an LP solution to bytes for sending to client.
   *
   * @param solution The LP solution to serialize
   * @return Serialized byte buffer
   */
  virtual std::vector<uint8_t> serialize_lp_solution(
    const optimization_problem_solution_t<i_t, f_t>& solution) = 0;

  /**
   * @brief Serialize a MIP solution to bytes for sending to client.
   *
   * @param solution The MIP solution to serialize
   * @return Serialized byte buffer
   */
  virtual std::vector<uint8_t> serialize_mip_solution(const mip_solution_t<i_t, f_t>& solution) = 0;

  //============================================================================
  // Async Operations
  //============================================================================

  /**
   * @brief Serialize a job submission request with async options.
   *
   * @param view Problem data
   * @param settings LP solver settings
   * @param blocking If true, server should wait and return result synchronously
   * @return Serialized async request bytes
   */
  virtual std::vector<uint8_t> serialize_async_lp_request(
    const mps_parser::data_model_view_t<i_t, f_t>& view,
    const pdlp_solver_settings_t<i_t, f_t>& settings,
    bool blocking) = 0;

  /**
   * @brief Serialize a job submission request with async options.
   *
   * @param view Problem data
   * @param settings MIP solver settings
   * @param blocking If true, server should wait and return result synchronously
   * @return Serialized async request bytes
   */
  virtual std::vector<uint8_t> serialize_async_mip_request(
    const mps_parser::data_model_view_t<i_t, f_t>& view,
    const mip_solver_settings_t<i_t, f_t>& settings,
    bool blocking) = 0;

  /**
   * @brief Serialize a status check request.
   *
   * @param job_id The job ID to check
   * @return Serialized request bytes
   */
  virtual std::vector<uint8_t> serialize_status_request(const std::string& job_id) = 0;

  /**
   * @brief Serialize a get result request.
   *
   * @param job_id The job ID to get results for
   * @return Serialized request bytes
   */
  virtual std::vector<uint8_t> serialize_get_result_request(const std::string& job_id) = 0;

  /**
   * @brief Serialize a delete request.
   *
   * @param job_id The job ID to delete
   * @return Serialized request bytes
   */
  virtual std::vector<uint8_t> serialize_delete_request(const std::string& job_id) = 0;

  /**
   * @brief Serialize a get logs request.
   *
   * @param job_id The job ID to get logs for
   * @param frombyte Byte offset to start reading from (0 = beginning)
   * @return Serialized request bytes
   */
  virtual std::vector<uint8_t> serialize_get_logs_request(const std::string& job_id,
                                                          int64_t frombyte = 0) = 0;

  /**
   * @brief Serialize a cancel job request.
   *
   * @param job_id The job ID to cancel
   * @return Serialized request bytes
   */
  virtual std::vector<uint8_t> serialize_cancel_request(const std::string& job_id) = 0;

  /**
   * @brief Job status enumeration.
   */
  enum class job_status_t { QUEUED, PROCESSING, COMPLETED, FAILED, NOT_FOUND, CANCELLED };

  /**
   * @brief Structure to hold log retrieval results.
   */
  struct logs_result_t {
    std::vector<std::string> log_lines;  ///< Log lines read from file
    int64_t nbytes;                      ///< Ending byte position (use as frombyte next time)
    bool job_exists;                     ///< False if job_id not found
  };

  /**
   * @brief Structure to hold cancel response results.
   */
  struct cancel_result_t {
    bool success;             ///< True if cancel request was processed
    std::string message;      ///< Success/error message
    job_status_t job_status;  ///< Status of job after cancel attempt
  };

  /**
   * @brief Deserialize job submission response.
   *
   * @param data Response bytes
   * @param[out] job_id Job ID assigned by server (on success)
   * @param[out] error_message Error message (on failure)
   * @return true if submission succeeded
   */
  virtual bool deserialize_submit_response(const std::vector<uint8_t>& data,
                                           std::string& job_id,
                                           std::string& error_message) = 0;

  /**
   * @brief Deserialize status check response.
   *
   * @param data Response bytes
   * @return Job status
   */
  virtual job_status_t deserialize_status_response(const std::vector<uint8_t>& data) = 0;

  /**
   * @brief Deserialize result response as LP solution.
   *
   * @param data Response bytes
   * @return LP solution, or error solution if failed
   */
  virtual optimization_problem_solution_t<i_t, f_t> deserialize_lp_result_response(
    const std::vector<uint8_t>& data) = 0;

  /**
   * @brief Deserialize result response as MIP solution.
   *
   * @param data Response bytes
   * @return MIP solution, or error solution if failed
   */
  virtual mip_solution_t<i_t, f_t> deserialize_mip_result_response(
    const std::vector<uint8_t>& data) = 0;

  /**
   * @brief Deserialize logs response.
   *
   * @param data Response bytes
   * @return Logs result structure
   */
  virtual logs_result_t deserialize_logs_response(const std::vector<uint8_t>& data) = 0;

  /**
   * @brief Deserialize cancel response.
   *
   * @param data Response bytes
   * @return Cancel result structure
   */
  virtual cancel_result_t deserialize_cancel_response(const std::vector<uint8_t>& data) = 0;

  //============================================================================
  // Server-side Async Request Handling
  //============================================================================

  /**
   * @brief Check if serialized data is an async protocol request.
   *
   * Async requests contain RequestType field (SUBMIT_JOB, CHECK_STATUS, etc.)
   *
   * @param data The serialized request bytes
   * @return true if this is an async protocol request
   */
  virtual bool is_async_request(const std::vector<uint8_t>& data) = 0;

  /**
   * @brief Get the async request type.
   *
   * @param data The serialized request bytes
   * @return Request type: 0=SUBMIT_JOB, 1=CHECK_STATUS, 2=GET_RESULT, 3=DELETE_RESULT,
   *         4=GET_LOGS, 5=CANCEL_JOB, 6=WAIT_FOR_RESULT
   */
  virtual int get_async_request_type(const std::vector<uint8_t>& data) = 0;

  /**
   * @brief Check if async request has blocking flag set.
   *
   * @param data The serialized request bytes
   * @return true if blocking mode is requested
   */
  virtual bool is_blocking_request(const std::vector<uint8_t>& data) = 0;

  /**
   * @brief Extract problem data from an async SUBMIT_JOB request.
   *
   * @param data The serialized async request bytes
   * @return The extracted problem data (LP or MIP request)
   */
  virtual std::vector<uint8_t> extract_problem_data(const std::vector<uint8_t>& data) = 0;

  /**
   * @brief Get job_id from async request (for CHECK_STATUS, GET_RESULT, DELETE_RESULT, GET_LOGS).
   *
   * @param data The serialized request bytes
   * @return The job ID string
   */
  virtual std::string get_job_id(const std::vector<uint8_t>& data) = 0;

  /**
   * @brief Get frombyte from GET_LOGS request.
   *
   * @param data The serialized request bytes
   * @return The byte offset to start reading from
   */
  virtual int64_t get_frombyte(const std::vector<uint8_t>& data) = 0;

  /**
   * @brief Serialize a job submission response.
   *
   * @param success Whether submission succeeded
   * @param result On success: job_id, on failure: error message
   * @return Serialized response bytes
   */
  virtual std::vector<uint8_t> serialize_submit_response(bool success,
                                                         const std::string& result) = 0;

  /**
   * @brief Serialize a status check response.
   *
   * @param status_code Job status: 0=QUEUED, 1=PROCESSING, 2=COMPLETED, 3=FAILED, 4=NOT_FOUND
   * @param message Status message
   * @return Serialized response bytes
   */
  virtual std::vector<uint8_t> serialize_status_response(int status_code,
                                                         const std::string& message) = 0;

  /**
   * @brief Serialize a get result response.
   *
   * @param success Whether result retrieval succeeded
   * @param result_data The solution data (if success)
   * @param error_message Error message (if failure)
   * @param is_mip Whether this is a MIP solution (vs LP)
   * @return Serialized response bytes
   */
  virtual std::vector<uint8_t> serialize_result_response(bool success,
                                                         const std::vector<uint8_t>& result_data,
                                                         const std::string& error_message,
                                                         bool is_mip = false) = 0;

  /**
   * @brief Serialize a delete response.
   *
   * @param success Whether deletion succeeded
   * @return Serialized response bytes
   */
  virtual std::vector<uint8_t> serialize_delete_response(bool success) = 0;

  /**
   * @brief Serialize a logs response.
   *
   * @param job_id The job ID
   * @param log_lines Log lines read from file
   * @param nbytes Ending byte position in log file
   * @param job_exists False if job_id not found
   * @return Serialized response bytes
   */
  virtual std::vector<uint8_t> serialize_logs_response(const std::string& job_id,
                                                       const std::vector<std::string>& log_lines,
                                                       int64_t nbytes,
                                                       bool job_exists) = 0;

  /**
   * @brief Serialize a cancel response.
   *
   * @param success Whether cancel was successful
   * @param message Success/error message
   * @param status_code Job status after cancel: 0=QUEUED, 1=PROCESSING, 2=COMPLETED, 3=FAILED,
   * 4=NOT_FOUND, 5=CANCELLED
   * @return Serialized response bytes
   */
  virtual std::vector<uint8_t> serialize_cancel_response(bool success,
                                                         const std::string& message,
                                                         int status_code) = 0;

  //============================================================================
  // Metadata
  //============================================================================

  /**
   * @brief Get the serialization format name (for logging/debugging).
   *
   * @return Format name string (e.g., "protobuf", "json", "msgpack")
   */
  virtual std::string format_name() const = 0;

  /**
   * @brief Get version of the serialization protocol.
   *
   * @return Protocol version number
   */
  virtual uint32_t protocol_version() const = 0;
};

/**
 * @brief Factory function type for creating serializer instances.
 *
 * Custom serializer libraries must export a function with this signature
 * named "create_cuopt_serializer".
 */
template <typename i_t, typename f_t>
using serializer_factory_t = std::unique_ptr<remote_serializer_t<i_t, f_t>> (*)();

/**
 * @brief Get the default (protobuf) serializer instance.
 *
 * @return Shared pointer to the default serializer
 */
template <typename i_t, typename f_t>
std::shared_ptr<remote_serializer_t<i_t, f_t>> get_default_serializer();

/**
 * @brief Get the currently configured serializer.
 *
 * Returns the custom serializer if CUOPT_SERIALIZER_LIB is set,
 * otherwise returns the default protobuf serializer.
 *
 * @return Shared pointer to the serializer
 */
template <typename i_t, typename f_t>
std::shared_ptr<remote_serializer_t<i_t, f_t>> get_serializer();

}  // namespace cuopt::linear_programming
