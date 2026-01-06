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
   * @brief Job status enumeration.
   */
  enum class job_status_t { QUEUED, PROCESSING, COMPLETED, FAILED, NOT_FOUND };

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
