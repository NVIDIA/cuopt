/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#include <cuopt/grpc/cython_grpc_client.hpp>
#include <cuopt/grpc/grpc_client_env.hpp>

#include <cuopt/mathematical_optimization/cpu_optimization_problem.hpp>
#include <cuopt/mathematical_optimization/io/data_model_view.hpp>
#include <cuopt/mathematical_optimization/optimization_problem_utils.hpp>
#include <cuopt/mathematical_optimization/solver_settings.hpp>

#include "grpc_client.hpp"

#include <chrono>
#include <limits>
#include <thread>

namespace cuopt::cython {

namespace {

grpc_job_status_t map_job_status(cuopt::mathematical_optimization::job_status_t status)
{
  using js = cuopt::mathematical_optimization::job_status_t;
  switch (status) {
    case js::QUEUED: return grpc_job_status_t::QUEUED;
    case js::PROCESSING: return grpc_job_status_t::PROCESSING;
    case js::COMPLETED: return grpc_job_status_t::COMPLETED;
    case js::FAILED: return grpc_job_status_t::FAILED;
    case js::CANCELLED: return grpc_job_status_t::CANCELLED;
    case js::NOT_FOUND: return grpc_job_status_t::NOT_FOUND;
    default: return grpc_job_status_t::NOT_FOUND;
  }
}

grpc_status_result_t map_status_result(
  const cuopt::mathematical_optimization::job_status_result_t& in)
{
  grpc_status_result_t out;
  out.success           = in.success;
  out.error_message     = in.error_message;
  out.status            = map_job_status(in.status);
  out.message           = in.message;
  out.result_size_bytes = in.result_size_bytes;
  return out;
}

bool is_in_flight(grpc_job_status_t status)
{
  return status == grpc_job_status_t::QUEUED || status == grpc_job_status_t::PROCESSING;
}

}  // namespace

struct grpc_python_client_t::impl_t {
  cuopt::mathematical_optimization::grpc_client_t client;
  explicit impl_t(cuopt::mathematical_optimization::grpc_client_config_t config)
    : client(std::move(config))
  {
  }
};

grpc_python_client_t::grpc_python_client_t(const std::string& host, int port)
  : impl_(std::make_unique<impl_t>(
      cuopt::mathematical_optimization::make_grpc_client_config(host, port)))
{
}

grpc_python_client_t::~grpc_python_client_t() = default;

bool grpc_python_client_t::connect(std::string& error_out)
{
  if (!impl_->client.connect()) {
    error_out = impl_->client.get_last_error();
    return false;
  }
  return true;
}

grpc_submit_result_t grpc_python_client_t::submit(
  cuopt::mathematical_optimization::io::data_model_view_t<int, double>* data_model,
  cuopt::mathematical_optimization::solver_settings_t<int, double>* settings,
  bool enable_incumbents)
{
  grpc_submit_result_t out;
  if (data_model == nullptr || settings == nullptr) {
    out.error_message = "data_model and settings must not be null";
    return out;
  }

  cuopt::mathematical_optimization::cpu_optimization_problem_t<int, double> cpu_problem;
  cuopt::mathematical_optimization::populate_from_data_model_view(
    &cpu_problem, data_model, settings, nullptr);

  const bool is_mip =
    cpu_problem.get_problem_category() ==
      cuopt::mathematical_optimization::problem_category_t::MIP ||
    cpu_problem.get_problem_category() == cuopt::mathematical_optimization::problem_category_t::IP;

  if (is_mip) {
    auto sub =
      impl_->client.submit_mip(cpu_problem, settings->get_mip_settings(), enable_incumbents);
    out.success       = sub.success;
    out.error_message = sub.error_message;
    out.job_id        = sub.job_id;
    out.is_mip        = true;
  } else {
    auto sub          = impl_->client.submit_lp(cpu_problem, settings->get_pdlp_settings());
    out.success       = sub.success;
    out.error_message = sub.error_message;
    out.job_id        = sub.job_id;
    out.is_mip        = false;
  }

  return out;
}

grpc_status_result_t grpc_python_client_t::status(const std::string& job_id)
{
  return map_status_result(impl_->client.check_status(job_id));
}

grpc_status_result_t grpc_python_client_t::wait(const std::string& job_id, int timeout_seconds)
{
  if (timeout_seconds < 0) {
    grpc_status_result_t out;
    out.success       = false;
    out.error_message = "timeout_seconds must be non-negative";
    out.status        = grpc_job_status_t::NOT_FOUND;
    return out;
  }

  if (timeout_seconds == 0) { return map_status_result(impl_->client.wait_for_completion(job_id)); }

  const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(timeout_seconds);
  const int poll_ms   = 1000;

  while (std::chrono::steady_clock::now() < deadline) {
    auto st = status(job_id);
    if (!st.success) { return st; }
    if (!is_in_flight(st.status)) { return st; }
    std::this_thread::sleep_for(std::chrono::milliseconds(poll_ms));
  }

  grpc_status_result_t out;
  out.success       = false;
  out.error_message = "Timeout waiting for job completion";
  out.status        = status(job_id).status;
  return out;
}

bool grpc_python_client_t::cancel(const std::string& job_id, std::string& error_out)
{
  auto result = impl_->client.cancel_job(job_id);
  if (!result.success) {
    error_out = result.error_message.empty() ? result.message : result.error_message;
    return false;
  }
  return true;
}

bool grpc_python_client_t::delete_job(const std::string& job_id, std::string& error_out)
{
  if (!impl_->client.delete_job(job_id)) {
    error_out = impl_->client.get_last_error();
    return false;
  }
  return true;
}

grpc_result_outcome_t grpc_python_client_t::result(const std::string& job_id, bool is_mip)
{
  grpc_result_outcome_t out;

  auto st = status(job_id);
  if (!st.success) {
    out.error_message = st.error_message;
    return out;
  }
  if (is_in_flight(st.status)) {
    out.not_ready = true;
    return out;
  }
  if (st.status != grpc_job_status_t::COMPLETED) {
    out.error_message = st.message.empty() ? std::string("Job is not completed") : st.message;
    return out;
  }

  out.solution = std::make_unique<solver_ret_t>();

  if (is_mip) {
    auto remote = impl_->client.get_mip_result<int, double>(job_id);
    if (!remote.success) {
      out.error_message = remote.error_message;
      out.solution.reset();
      return out;
    }
    out.solution->problem_type = cuopt::mathematical_optimization::problem_category_t::MIP;
    out.solution->mip_ret      = remote.solution->to_cpu_mip_ret_t();
  } else {
    auto remote = impl_->client.get_lp_result<int, double>(job_id);
    if (!remote.success) {
      out.error_message = remote.error_message;
      out.solution.reset();
      return out;
    }
    out.solution->problem_type = cuopt::mathematical_optimization::problem_category_t::LP;
    out.solution->lp_ret       = remote.solution->to_cpu_linear_programming_ret_t();
  }

  out.success = true;
  return out;
}

grpc_logs_result_t grpc_python_client_t::fetch_logs(const std::string& job_id, int64_t from_byte)
{
  grpc_logs_result_t out;
  out.success = impl_->client.stream_logs(
    job_id, from_byte, [&out](const std::string& line, bool /*job_complete*/) -> bool {
      if (!line.empty()) { out.lines.push_back(line); }
      return true;
    });
  if (!out.success) { out.error_message = impl_->client.get_last_error(); }
  return out;
}

bool grpc_python_client_t::stream_logs(const std::string& job_id,
                                       int64_t from_byte,
                                       grpc_log_line_callback_t callback,
                                       void* user_data)
{
  if (callback == nullptr) { return false; }
  return impl_->client.stream_logs(
    job_id, from_byte, [callback, user_data](const std::string& line, bool job_complete) -> bool {
      return callback(line.data(), line.size(), job_complete ? 1 : 0, user_data) != 0;
    });
}

grpc_incumbents_result_t grpc_python_client_t::fetch_incumbents(const std::string& job_id,
                                                                int64_t from_index,
                                                                int32_t max_count)
{
  grpc_incumbents_result_t out;
  auto result = impl_->client.get_incumbents(job_id, from_index, max_count);
  if (!result.success) {
    out.error_message = result.error_message;
    return out;
  }
  out.success      = true;
  out.next_index   = result.next_index;
  out.job_complete = result.job_complete;
  for (const auto& inc : result.incumbents) {
    grpc_incumbent_entry_t entry;
    entry.index      = inc.index;
    entry.objective  = inc.objective;
    entry.assignment = inc.assignment;
    out.incumbents.push_back(std::move(entry));
  }
  return out;
}

std::string grpc_python_client_t::last_error() const { return impl_->client.get_last_error(); }

}  // namespace cuopt::cython
