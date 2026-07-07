/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved. SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cuopt/mathematical_optimization/utilities/cython_solve.hpp>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace cuopt::mathematical_optimization {
template <typename i_t, typename f_t>
class solver_settings_t;
namespace io {
template <typename i_t, typename f_t>
class data_model_view_t;
}  // namespace io
}  // namespace cuopt::mathematical_optimization

namespace cuopt::cython {

/** Mirrors cuopt::mathematical_optimization::job_status_t for the Python bindings. */
enum class grpc_job_status_t : int {
  QUEUED     = 0,
  PROCESSING = 1,
  COMPLETED  = 2,
  FAILED     = 3,
  CANCELLED  = 4,
  NOT_FOUND  = 5,
};

struct grpc_submit_result_t {
  bool success = false;
  std::string error_message;
  std::string job_id;
  bool is_mip = false;
};

struct grpc_status_result_t {
  bool success = false;
  std::string error_message;
  grpc_job_status_t status = grpc_job_status_t::NOT_FOUND;
  std::string message;
  int64_t result_size_bytes = 0;
};

struct grpc_result_outcome_t {
  bool not_ready = false;
  bool success   = false;
  std::string error_message;
  std::unique_ptr<solver_ret_t> solution;
};

struct grpc_logs_result_t {
  bool success = false;
  std::string error_message;
  std::vector<std::string> lines;
};

struct grpc_incumbent_entry_t {
  int64_t index    = 0;
  double objective = 0.0;
  std::vector<double> assignment;
};

struct grpc_incumbents_result_t {
  bool success = false;
  std::string error_message;
  std::vector<grpc_incumbent_entry_t> incumbents;
  int64_t next_index = 0;
  bool job_complete  = false;
};

/**
 * C callback for streaming log lines from Cython (one line at a time).
 * Uses C-compatible ``int`` fields so Cython ``bint`` function pointers match.
 */
using grpc_log_line_callback_t = int (*)(const char* line,
                                         size_t line_len,
                                         int job_complete,
                                         void* user_data);

/**
 * @brief Owning wrapper around grpc_client_t for Cython.
 */
class grpc_python_client_t {
 public:
  grpc_python_client_t(const std::string& host, int port);
  ~grpc_python_client_t();

  grpc_python_client_t(const grpc_python_client_t&)            = delete;
  grpc_python_client_t& operator=(const grpc_python_client_t&) = delete;

  bool connect(std::string& error_out);

  grpc_submit_result_t submit(
    cuopt::mathematical_optimization::io::data_model_view_t<int, double>* data_model,
    cuopt::mathematical_optimization::solver_settings_t<int, double>* settings,
    bool enable_incumbents = false);

  grpc_status_result_t status(const std::string& job_id);

  /**
   * @param timeout_seconds 0 = block on WaitForCompletion; >0 = poll with timeout.
   */
  grpc_status_result_t wait(const std::string& job_id, int timeout_seconds);

  bool cancel(const std::string& job_id, std::string& error_out);

  bool delete_job(const std::string& job_id, std::string& error_out);

  /**
   * @param is_mip When true, fetch a MIP result; otherwise LP.
   */
  grpc_result_outcome_t result(const std::string& job_id, bool is_mip);

  /**
   * @brief Block until the job completes, collecting all solver log lines.
   */
  grpc_logs_result_t fetch_logs(const std::string& job_id, int64_t from_byte = 0);

  /**
   * @brief Stream solver log lines until the job completes.
   *
   * Invokes @p callback for each line. Return false from the callback to stop
   * early. Blocks the calling thread for the lifetime of the stream.
   */
  bool stream_logs(const std::string& job_id,
                   int64_t from_byte,
                   grpc_log_line_callback_t callback,
                   void* user_data);

  /**
   * @brief Poll GetIncumbents until the job completes.
   */
  grpc_incumbents_result_t fetch_incumbents(const std::string& job_id,
                                            int64_t from_index = 0,
                                            int32_t max_count  = 0);

  std::string last_error() const;

 private:
  struct impl_t;
  std::unique_ptr<impl_t> impl_;
};

}  // namespace cuopt::cython
