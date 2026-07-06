# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from libc.stdint cimport int64_t
from libcpp.memory cimport unique_ptr
from libcpp.string cimport string
from libcpp.vector cimport vector

from cuopt.linear_programming.data_model.data_model cimport data_model_view_t
from cuopt.linear_programming.solver.solver cimport solver_ret_t
from cuopt.linear_programming.solver_settings.solver_settings cimport (
    solver_settings_t,
)

cdef extern from "cuopt/grpc/cython_grpc_client.hpp" namespace "cuopt::cython":
    ctypedef enum grpc_job_status_t "cuopt::cython::grpc_job_status_t":
        QUEUED "cuopt::cython::grpc_job_status_t::QUEUED"
        PROCESSING "cuopt::cython::grpc_job_status_t::PROCESSING"
        COMPLETED "cuopt::cython::grpc_job_status_t::COMPLETED"
        FAILED "cuopt::cython::grpc_job_status_t::FAILED"
        CANCELLED "cuopt::cython::grpc_job_status_t::CANCELLED"
        NOT_FOUND "cuopt::cython::grpc_job_status_t::NOT_FOUND"

    cdef cppclass grpc_submit_result_t:
        bint success
        string error_message
        string job_id
        bint is_mip

    cdef cppclass grpc_status_result_t:
        bint success
        string error_message
        grpc_job_status_t status
        string message
        long long result_size_bytes

    cdef cppclass grpc_result_outcome_t:
        bint not_ready
        bint success
        string error_message
        unique_ptr[solver_ret_t] solution

    cdef cppclass grpc_logs_result_t:
        bint success
        string error_message
        vector[string] lines

    cdef cppclass grpc_incumbent_entry_t:
        int64_t index
        double objective
        vector[double] assignment

    cdef cppclass grpc_incumbents_result_t:
        bint success
        string error_message
        vector[grpc_incumbent_entry_t] incumbents
        int64_t next_index
        bint job_complete

    ctypedef int (*grpc_log_line_callback_t)(
        const char* line, size_t line_len, int job_complete, void* user_data
    ) noexcept nogil

    cdef cppclass grpc_python_client_t:
        grpc_python_client_t(const string& host, int port) except +
        bint connect(string& error_out)
        grpc_submit_result_t submit(
            data_model_view_t[int, double]* data_model,
            solver_settings_t[int, double]* settings,
            bint enable_incumbents,
        ) except +
        grpc_status_result_t status(const string& job_id) except +
        grpc_status_result_t wait(const string& job_id, int timeout_seconds) except +
        bint cancel(const string& job_id, string& error_out) except +
        bint delete_job(const string& job_id, string& error_out) except +
        grpc_result_outcome_t result(const string& job_id, bint is_mip) except +
        grpc_logs_result_t fetch_logs(const string& job_id, long long from_byte) except +
        bint stream_logs(
            const string& job_id,
            long long from_byte,
            grpc_log_line_callback_t callback,
            void* user_data,
        ) except +
        grpc_incumbents_result_t fetch_incumbents(
            const string& job_id, int64_t from_index, int max_count
        ) except +
        string last_error()
