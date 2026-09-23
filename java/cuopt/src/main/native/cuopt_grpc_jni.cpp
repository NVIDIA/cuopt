/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// JNI bridge onto grpc_python_client_t (the same C++ class the Python Cython bindings wrap --
// see cpp/src/grpc/client/cython_grpc_client.cpp). Despite the name, that class is plain C++
// with no Python dependency, so it's reusable as-is.
//
// LP and MIP, plaintext or TLS, submit/status/wait/cancel/delete/result. Not covered yet: log
// streaming and incumbent callbacks (both need a JNI callback path across native threads, which
// is a separate chunk of work from the request/response calls here).

#include <cuopt/grpc/cython_grpc_client.hpp>
#include <cuopt/mathematical_optimization/io/data_model_view.hpp>
#include <cuopt/mathematical_optimization/mip/solver_solution.hpp>
#include <cuopt/mathematical_optimization/pdlp/solver_solution.hpp>
#include <cuopt/mathematical_optimization/solver_settings.hpp>
#include <cuopt/mathematical_optimization/utilities/cython_solve.hpp>

#include <jni.h>

#include <stdexcept>
#include <string>
#include <vector>

namespace {

using cuopt::cython::grpc_python_client_connect_options_t;
using cuopt::cython::grpc_python_client_t;
using cuopt::cython::grpc_python_tls_mode_t;
using cuopt::mathematical_optimization::solver_settings_t;
using cuopt::mathematical_optimization::io::data_model_view_t;

std::vector<double> get_double_array(JNIEnv* env, jdoubleArray array)
{
  if (array == nullptr) { return {}; }
  const jsize len = env->GetArrayLength(array);
  std::vector<double> out(static_cast<size_t>(len));
  env->GetDoubleArrayRegion(array, 0, len, out.data());
  return out;
}

std::vector<int> get_int_array(JNIEnv* env, jintArray array)
{
  if (array == nullptr) { return {}; }
  const jsize len = env->GetArrayLength(array);
  std::vector<jint> tmp(static_cast<size_t>(len));
  env->GetIntArrayRegion(array, 0, len, tmp.data());
  return std::vector<int>(tmp.begin(), tmp.end());
}

std::vector<char> get_byte_array(JNIEnv* env, jbyteArray array)
{
  if (array == nullptr) { return {}; }
  const jsize len = env->GetArrayLength(array);
  std::vector<jbyte> tmp(static_cast<size_t>(len));
  env->GetByteArrayRegion(array, 0, len, tmp.data());
  return std::vector<char>(tmp.begin(), tmp.end());
}

std::string get_string(JNIEnv* env, jstring value)
{
  if (value == nullptr) { return {}; }
  const char* chars = env->GetStringUTFChars(value, nullptr);
  std::string out(chars);
  env->ReleaseStringUTFChars(value, chars);
  return out;
}

jdoubleArray to_double_array(JNIEnv* env, const std::vector<double>& values)
{
  jdoubleArray result = env->NewDoubleArray(static_cast<jsize>(values.size()));
  env->SetDoubleArrayRegion(result, 0, static_cast<jsize>(values.size()), values.data());
  return result;
}

void throw_runtime_exception(JNIEnv* env, const std::string& message)
{
  jclass cls = env->FindClass("java/lang/RuntimeException");
  if (cls == nullptr) { return; }
  env->ThrowNew(cls, message.c_str());
}

grpc_python_client_t* to_client(jlong handle)
{
  return reinterpret_cast<grpc_python_client_t*>(handle);
}

template <typename F>
auto run(JNIEnv* env, const char* operation, F&& fn) -> decltype(fn())
{
  try {
    return fn();
  } catch (const std::exception& e) {
    throw_runtime_exception(env, std::string(operation) + " failed: " + e.what());
    return decltype(fn()){};
  }
}

// Mirrors createProblem's array convention (see cuopt_jni.cpp) exactly, so callers building a
// Problem for the in-process C API can reuse the same arrays here unchanged. Heap-allocated to
// match cuOptCreateSolverSettings's own construction pattern (see cpp/src/pdlp/cuopt_c.cpp).
std::unique_ptr<data_model_view_t<int, double>> build_data_model(JNIEnv* env,
                                                                 jboolean maximize,
                                                                 jdouble objective_offset,
                                                                 jdoubleArray objective,
                                                                 jintArray row_offsets,
                                                                 jintArray column_indices,
                                                                 jdoubleArray values,
                                                                 jbyteArray row_types,
                                                                 jdoubleArray lower_bounds,
                                                                 jdoubleArray upper_bounds,
                                                                 jbyteArray variable_types,
                                                                 std::vector<double>& obj_out,
                                                                 std::vector<int>& offsets_out,
                                                                 std::vector<int>& cols_out,
                                                                 std::vector<double>& vals_out,
                                                                 std::vector<char>& row_ty_out,
                                                                 std::vector<double>& lbs_out,
                                                                 std::vector<double>& ubs_out,
                                                                 std::vector<char>& var_ty_out)
{
  obj_out     = get_double_array(env, objective);
  offsets_out = get_int_array(env, row_offsets);
  cols_out    = get_int_array(env, column_indices);
  vals_out    = get_double_array(env, values);
  row_ty_out  = get_byte_array(env, row_types);
  lbs_out     = get_double_array(env, lower_bounds);
  ubs_out     = get_double_array(env, upper_bounds);
  var_ty_out  = get_byte_array(env, variable_types);

  auto data_model = std::make_unique<data_model_view_t<int, double>>();
  data_model->set_maximize(maximize != 0);
  data_model->set_objective_offset(static_cast<double>(objective_offset));
  data_model->set_objective_coefficients(obj_out.data(), static_cast<int>(obj_out.size()));
  data_model->set_csr_constraint_matrix(vals_out.data(),
                                        static_cast<int>(vals_out.size()),
                                        cols_out.data(),
                                        static_cast<int>(cols_out.size()),
                                        offsets_out.data(),
                                        static_cast<int>(offsets_out.size()));
  data_model->set_row_types(row_ty_out.data(), static_cast<int>(row_ty_out.size()));
  data_model->set_variable_lower_bounds(lbs_out.data(), static_cast<int>(lbs_out.size()));
  data_model->set_variable_upper_bounds(ubs_out.data(), static_cast<int>(ubs_out.size()));
  data_model->set_variable_types(var_ty_out.data(), static_cast<int>(var_ty_out.size()));
  return data_model;
}

}  // namespace

extern "C" JNIEXPORT jlong JNICALL
Java_com_nvidia_cuopt_mathematicaloptimization_NativeGrpcClient_createClient(JNIEnv* env,
                                                                             jclass,
                                                                             jstring host,
                                                                             jint port)
{
  return run(env, "createClient", [&]() -> jlong {
    auto* client = new grpc_python_client_t(get_string(env, host), static_cast<int>(port));
    return reinterpret_cast<jlong>(client);
  });
}

// tls_mode: 0=ENV (default, respects the usual gRPC/OpenSSL env vars), 1=DISABLED (plaintext),
// 2=EXPLICIT (uses the three PEM strings below). Matches grpc_python_tls_mode_t.
extern "C" JNIEXPORT jlong JNICALL
Java_com_nvidia_cuopt_mathematicaloptimization_NativeGrpcClient_createClientWithTls(
  JNIEnv* env,
  jclass,
  jstring host,
  jint port,
  jint tls_mode,
  jstring tls_root_certs,
  jstring tls_client_cert,
  jstring tls_client_key)
{
  return run(env, "createClientWithTls", [&]() -> jlong {
    grpc_python_client_connect_options_t options;
    options.tls_mode        = static_cast<grpc_python_tls_mode_t>(tls_mode);
    options.tls_root_certs  = get_string(env, tls_root_certs);
    options.tls_client_cert = get_string(env, tls_client_cert);
    options.tls_client_key  = get_string(env, tls_client_key);
    auto* client = new grpc_python_client_t(get_string(env, host), static_cast<int>(port), options);
    return reinterpret_cast<jlong>(client);
  });
}

extern "C" JNIEXPORT void JNICALL
Java_com_nvidia_cuopt_mathematicaloptimization_NativeGrpcClient_destroyClient(JNIEnv*,
                                                                              jclass,
                                                                              jlong handle)
{
  delete to_client(handle);
}

extern "C" JNIEXPORT void JNICALL
Java_com_nvidia_cuopt_mathematicaloptimization_NativeGrpcClient_connect(JNIEnv* env,
                                                                        jclass,
                                                                        jlong handle)
{
  std::string error;
  if (!to_client(handle)->connect(error)) { throw_runtime_exception(env, error); }
}

extern "C" JNIEXPORT jboolean JNICALL
Java_com_nvidia_cuopt_mathematicaloptimization_NativeGrpcClient_ping(JNIEnv* env,
                                                                     jclass,
                                                                     jlong handle,
                                                                     jint timeout_seconds)
{
  std::string error;
  const bool ok = to_client(handle)->ping(error, static_cast<int>(timeout_seconds));
  if (!ok && !error.empty()) { throw_runtime_exception(env, error); }
  return static_cast<jboolean>(ok);
}

// LP vs MIP is decided automatically from variable_types (any 'I' makes it a MIP), exactly like
// grpc_python_client_t::submit() itself -- so one entry point covers both; getLpResult /
// getMipResult below fail fast if called against the wrong kind of job.
extern "C" JNIEXPORT jstring JNICALL
Java_com_nvidia_cuopt_mathematicaloptimization_NativeGrpcClient_submit(JNIEnv* env,
                                                                       jclass,
                                                                       jlong handle,
                                                                       jint num_constraints,
                                                                       jint num_variables,
                                                                       jboolean maximize,
                                                                       jdouble objective_offset,
                                                                       jdoubleArray objective,
                                                                       jintArray row_offsets,
                                                                       jintArray column_indices,
                                                                       jdoubleArray values,
                                                                       jbyteArray row_types,
                                                                       jdoubleArray lower_bounds,
                                                                       jdoubleArray upper_bounds,
                                                                       jbyteArray variable_types,
                                                                       jdouble time_limit_seconds,
                                                                       jboolean enable_incumbents)
{
  return run(env, "submit", [&]() -> jstring {
    (void)num_constraints;
    (void)num_variables;
    std::vector<double> obj, vals, lbs, ubs;
    std::vector<int> offsets, cols;
    std::vector<char> row_ty, var_ty;
    auto data_model = build_data_model(env,
                                       maximize,
                                       objective_offset,
                                       objective,
                                       row_offsets,
                                       column_indices,
                                       values,
                                       row_types,
                                       lower_bounds,
                                       upper_bounds,
                                       variable_types,
                                       obj,
                                       offsets,
                                       cols,
                                       vals,
                                       row_ty,
                                       lbs,
                                       ubs,
                                       var_ty);

    // Heap-allocated to match cuOptCreateSolverSettings's own construction pattern.
    auto settings = std::make_unique<solver_settings_t<int, double>>();
    if (time_limit_seconds > 0) {
      settings->set_parameter_from_string("time_limit", std::to_string(time_limit_seconds));
    }

    auto sub = to_client(handle)->submit(data_model.get(), settings.get(), enable_incumbents != 0);
    if (!sub.success) { throw std::runtime_error(sub.error_message); }
    return env->NewStringUTF(sub.job_id.c_str());
  });
}

extern "C" JNIEXPORT jint JNICALL
Java_com_nvidia_cuopt_mathematicaloptimization_NativeGrpcClient_getStatus(JNIEnv* env,
                                                                          jclass,
                                                                          jlong handle,
                                                                          jstring job_id)
{
  return run(env, "getStatus", [&]() -> jint {
    auto st = to_client(handle)->status(get_string(env, job_id));
    if (!st.success) { throw std::runtime_error(st.error_message); }
    return static_cast<jint>(st.status);
  });
}

// Blocks server-side (long-poll) rather than the client sleeping between getStatus calls.
// timeout_seconds=0 waits indefinitely for a single long-poll response.
extern "C" JNIEXPORT jint JNICALL
Java_com_nvidia_cuopt_mathematicaloptimization_NativeGrpcClient_waitForCompletion(
  JNIEnv* env, jclass, jlong handle, jstring job_id, jint timeout_seconds)
{
  return run(env, "waitForCompletion", [&]() -> jint {
    auto st = to_client(handle)->wait(get_string(env, job_id), static_cast<int>(timeout_seconds));
    if (!st.success) { throw std::runtime_error(st.error_message); }
    return static_cast<jint>(st.status);
  });
}

extern "C" JNIEXPORT void JNICALL
Java_com_nvidia_cuopt_mathematicaloptimization_NativeGrpcClient_cancel(JNIEnv* env,
                                                                       jclass,
                                                                       jlong handle,
                                                                       jstring job_id)
{
  std::string error;
  if (!to_client(handle)->cancel(get_string(env, job_id), error)) {
    throw_runtime_exception(env, error);
  }
}

extern "C" JNIEXPORT void JNICALL
Java_com_nvidia_cuopt_mathematicaloptimization_NativeGrpcClient_deleteJob(JNIEnv* env,
                                                                          jclass,
                                                                          jlong handle,
                                                                          jstring job_id)
{
  std::string error;
  if (!to_client(handle)->delete_job(get_string(env, job_id), error)) {
    throw_runtime_exception(env, error);
  }
}

// Packed as [terminationStatus, objective, mipGap, solutionBound, totalSolveTime,
// var_0, ..., var_{n-1}]. Throws if the job wasn't a MIP submission.
extern "C" JNIEXPORT jdoubleArray JNICALL
Java_com_nvidia_cuopt_mathematicaloptimization_NativeGrpcClient_getMipResult(JNIEnv* env,
                                                                             jclass,
                                                                             jlong handle,
                                                                             jstring job_id)
{
  return run(env, "getMipResult", [&]() -> jdoubleArray {
    auto outcome = to_client(handle)->result(get_string(env, job_id));
    if (outcome.not_ready) { throw std::runtime_error("job is not finished yet"); }
    if (!outcome.success) { throw std::runtime_error(outcome.error_message); }
    if (outcome.solution->problem_type !=
        cuopt::mathematical_optimization::problem_category_t::MIP) {
      throw std::runtime_error("job is not a MIP result");
    }

    const auto& mip = outcome.solution->mip_ret;
    if (mip.is_gpu()) {
      // The remote/host-only path never produces a GPU-backed result; this would only trip if
      // that assumption changes upstream.
      throw std::runtime_error("unexpected GPU-backed result from a remote client");
    }
    const auto& solution = std::get<cuopt::cython::cpu_buffer>(mip.solution_);

    std::vector<double> packed;
    packed.reserve(5 + solution.size());
    packed.push_back(static_cast<double>(mip.termination_status_));
    packed.push_back(mip.objective_);
    packed.push_back(mip.mip_gap_);
    packed.push_back(mip.solution_bound_);
    packed.push_back(mip.total_solve_time_);
    packed.insert(packed.end(), solution.begin(), solution.end());
    return to_double_array(env, packed);
  });
}

// Packed as [terminationStatus, primalObjective, dualObjective, gap, solveTime, numVariables,
// numConstraints, primal_0..primal_{numVariables-1}, dual_0..dual_{numConstraints-1},
// reducedCost_0..reducedCost_{numVariables-1}]. Throws if the job wasn't an LP submission.
extern "C" JNIEXPORT jdoubleArray JNICALL
Java_com_nvidia_cuopt_mathematicaloptimization_NativeGrpcClient_getLpResult(JNIEnv* env,
                                                                            jclass,
                                                                            jlong handle,
                                                                            jstring job_id)
{
  return run(env, "getLpResult", [&]() -> jdoubleArray {
    auto outcome = to_client(handle)->result(get_string(env, job_id));
    if (outcome.not_ready) { throw std::runtime_error("job is not finished yet"); }
    if (!outcome.success) { throw std::runtime_error(outcome.error_message); }
    if (outcome.solution->problem_type !=
        cuopt::mathematical_optimization::problem_category_t::LP) {
      throw std::runtime_error("job is not an LP result");
    }

    const auto& lp = outcome.solution->lp_ret;
    if (lp.is_gpu()) {
      throw std::runtime_error("unexpected GPU-backed result from a remote client");
    }
    const auto& solution =
      std::get<cuopt::cython::linear_programming_ret_t::cpu_solutions_t>(lp.solutions_);

    std::vector<double> packed;
    packed.reserve(7 + solution.primal_solution_.size() + solution.dual_solution_.size() +
                   solution.reduced_cost_.size());
    packed.push_back(static_cast<double>(lp.termination_status_));
    packed.push_back(lp.primal_objective_);
    packed.push_back(lp.dual_objective_);
    packed.push_back(lp.gap_);
    packed.push_back(lp.solve_time_);
    packed.push_back(static_cast<double>(solution.primal_solution_.size()));
    packed.push_back(static_cast<double>(solution.dual_solution_.size()));
    packed.insert(packed.end(), solution.primal_solution_.begin(), solution.primal_solution_.end());
    packed.insert(packed.end(), solution.dual_solution_.begin(), solution.dual_solution_.end());
    packed.insert(packed.end(), solution.reduced_cost_.begin(), solution.reduced_cost_.end());
    return to_double_array(env, packed);
  });
}
