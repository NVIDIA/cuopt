/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <algorithm>
#include <cstring>
#include <exception>
#include <memory>
#include <mutex>
#include <span>
#include <string>
#include <unordered_set>
#include <unordered_map>
#include <vector>

#include <cuopt/linear_programming/cuopt_c.h>
#include <cuopt/linear_programming/optimization_problem_utils.hpp>
#include <cuopt/linear_programming/io/parser.hpp>
#include <pdlp/cuopt_c_internal.hpp>

#include "cuopt_java_native_api.hpp"

#include <jni.h>

namespace {

JavaVM* g_jvm = nullptr;

struct java_callback_context_t {
  jobject callback{nullptr};
  jobject user_data{nullptr};
  int num_variables{0};
};

std::mutex g_callback_mutex;
std::unordered_map<jlong, std::vector<java_callback_context_t*>> g_callback_contexts;

struct java_problem_state_t {
  std::vector<cuopt_float_t> initial_primal_solution;
  std::vector<cuopt_float_t> initial_dual_solution;
};

std::mutex g_problem_state_mutex;
std::unordered_map<jlong, java_problem_state_t> g_problem_states;

// Problems created directly by this JNI module must be destroyed here as well.
// Passing these C++ objects to cuOptDestroyProblem in libcuopt.so crosses the
// shared-library boundary with a private wrapper type.
std::mutex g_jni_owned_problem_mutex;
std::unordered_set<jlong> g_jni_owned_problem_handles;

cuOptOptimizationProblem to_problem(jlong handle)
{
  return reinterpret_cast<cuOptOptimizationProblem>(handle);
}

cuopt::linear_programming::problem_and_stream_view_t* to_problem_view(jlong handle)
{
  return reinterpret_cast<cuopt::linear_programming::problem_and_stream_view_t*>(handle);
}

cuOptSolverSettings to_settings(jlong handle)
{
  return reinterpret_cast<cuOptSolverSettings>(handle);
}

cuOptSolution to_solution(jlong handle)
{
  return reinterpret_cast<cuOptSolution>(handle);
}

jlong from_handle(void* handle)
{
  return reinterpret_cast<jlong>(handle);
}

void remember_jni_owned_problem(void* handle)
{
  std::lock_guard<std::mutex> lock(g_jni_owned_problem_mutex);
  g_jni_owned_problem_handles.insert(from_handle(handle));
}

bool take_jni_owned_problem(jlong handle)
{
  std::lock_guard<std::mutex> lock(g_jni_owned_problem_mutex);
  return g_jni_owned_problem_handles.erase(handle) != 0;
}

std::vector<cuopt_float_t> get_double_array(JNIEnv* env, jdoubleArray array)
{
  if (array == nullptr) { return {}; }
  const jsize len = env->GetArrayLength(array);
  std::vector<jdouble> tmp(static_cast<size_t>(len));
  env->GetDoubleArrayRegion(array, 0, len, tmp.data());
  return std::vector<cuopt_float_t>(tmp.begin(), tmp.end());
}

std::vector<cuopt_int_t> get_int_array(JNIEnv* env, jintArray array)
{
  if (array == nullptr) { return {}; }
  const jsize len = env->GetArrayLength(array);
  std::vector<jint> tmp(static_cast<size_t>(len));
  env->GetIntArrayRegion(array, 0, len, tmp.data());
  return std::vector<cuopt_int_t>(tmp.begin(), tmp.end());
}

std::vector<char> get_byte_array(JNIEnv* env, jbyteArray array)
{
  if (array == nullptr) { return {}; }
  const jsize len = env->GetArrayLength(array);
  std::vector<jbyte> tmp(static_cast<size_t>(len));
  env->GetByteArrayRegion(array, 0, len, tmp.data());
  return std::vector<char>(tmp.begin(), tmp.end());
}

std::string get_string(JNIEnv* env, jstring value);

std::vector<std::string> get_string_array(JNIEnv* env, jobjectArray array)
{
  if (array == nullptr) { return {}; }
  const jsize len = env->GetArrayLength(array);
  std::vector<std::string> values;
  values.reserve(static_cast<size_t>(len));
  for (jsize i = 0; i < len; ++i) {
    values.push_back(get_string(env, static_cast<jstring>(env->GetObjectArrayElement(array, i))));
  }
  return values;
}

jobjectArray to_string_array(JNIEnv* env, const std::vector<std::string>& values)
{
  jclass string_class = env->FindClass("java/lang/String");
  jobjectArray result = env->NewObjectArray(static_cast<jsize>(values.size()), string_class, nullptr);
  for (jsize i = 0; i < static_cast<jsize>(values.size()); ++i) {
    env->SetObjectArrayElement(result, i, env->NewStringUTF(values[static_cast<size_t>(i)].c_str()));
  }
  return result;
}

jdoubleArray to_double_array(JNIEnv* env, const std::vector<cuopt_float_t>& values)
{
  jdoubleArray result = env->NewDoubleArray(static_cast<jsize>(values.size()));
  std::vector<jdouble> tmp(values.begin(), values.end());
  env->SetDoubleArrayRegion(result, 0, static_cast<jsize>(tmp.size()), tmp.data());
  return result;
}

jintArray to_int_array(JNIEnv* env, const std::vector<cuopt_int_t>& values)
{
  jintArray result = env->NewIntArray(static_cast<jsize>(values.size()));
  std::vector<jint> tmp(values.begin(), values.end());
  env->SetIntArrayRegion(result, 0, static_cast<jsize>(tmp.size()), tmp.data());
  return result;
}

jbyteArray to_byte_array(JNIEnv* env, const std::vector<char>& values)
{
  jbyteArray result = env->NewByteArray(static_cast<jsize>(values.size()));
  std::vector<jbyte> tmp(values.begin(), values.end());
  env->SetByteArrayRegion(result, 0, static_cast<jsize>(tmp.size()), tmp.data());
  return result;
}

std::string get_string(JNIEnv* env, jstring value)
{
  if (value == nullptr) { return {}; }
  const char* chars = env->GetStringUTFChars(value, nullptr);
  std::string result(chars);
  env->ReleaseStringUTFChars(value, chars);
  return result;
}

void throw_cuopt_exception(JNIEnv* env, cuopt_int_t status, const std::string& message)
{
  jclass cls = env->FindClass("com/nvidia/cuopt/linearprogramming/CuOptException");
  if (cls == nullptr) { return; }
  jmethodID ctor = env->GetMethodID(cls, "<init>", "(ILjava/lang/String;)V");
  if (ctor == nullptr) { return; }
  jstring msg = env->NewStringUTF(message.c_str());
  jobject ex  = env->NewObject(cls, ctor, static_cast<jint>(status), msg);
  env->Throw(static_cast<jthrowable>(ex));
}

void throw_illegal_state(JNIEnv* env, const std::string& message)
{
  jclass cls = env->FindClass("java/lang/IllegalStateException");
  if (cls == nullptr) { return; }
  env->ThrowNew(cls, message.c_str());
}

bool check_status(JNIEnv* env, cuopt_int_t status, const char* operation)
{
  if (status == CUOPT_SUCCESS) { return true; }
  throw_cuopt_exception(env, status, std::string(operation) + " failed with status " +
                                      std::to_string(status));
  return false;
}

cuopt::linear_programming::lp_solution_interface_t<cuopt_int_t, cuopt_float_t>* to_lp_solution(
  JNIEnv* env, jlong handle, const char* operation)
{
  auto* solution =
    reinterpret_cast<cuopt::linear_programming::solution_and_stream_view_t*>(handle);
  if (solution == nullptr || solution->is_mip || solution->lp_solution_interface_ptr == nullptr) {
    throw_illegal_state(env, std::string(operation) + " is only available for LP solutions");
    return nullptr;
  }
  return solution->lp_solution_interface_ptr;
}

template <typename F>
bool run_problem_operation(JNIEnv* env, const char* operation, F&& operation_fn)
{
  try {
    operation_fn();
    return true;
  } catch (const std::exception& e) {
    throw_cuopt_exception(env,
                          CUOPT_INVALID_ARGUMENT,
                          std::string(operation) + " failed: " + e.what());
    return false;
  }
}

JNIEnv* get_callback_env(bool& detach)
{
  detach = false;
  JNIEnv* env = nullptr;
  if (g_jvm->GetEnv(reinterpret_cast<void**>(&env), JNI_VERSION_1_8) == JNI_OK) { return env; }
  if (g_jvm->AttachCurrentThread(reinterpret_cast<void**>(&env), nullptr) == JNI_OK) {
    detach = true;
    return env;
  }
  return nullptr;
}

void cleanup_callback_contexts(JNIEnv* env, jlong settings_handle)
{
  std::vector<java_callback_context_t*> contexts;
  {
    std::lock_guard<std::mutex> lock(g_callback_mutex);
    auto it = g_callback_contexts.find(settings_handle);
    if (it == g_callback_contexts.end()) { return; }
    contexts = std::move(it->second);
    g_callback_contexts.erase(it);
  }
  for (auto* context : contexts) {
    if (context->callback != nullptr) { env->DeleteGlobalRef(context->callback); }
    if (context->user_data != nullptr) { env->DeleteGlobalRef(context->user_data); }
    delete context;
  }
}

void remember_callback_context(jlong settings_handle, java_callback_context_t* context)
{
  std::lock_guard<std::mutex> lock(g_callback_mutex);
  g_callback_contexts[settings_handle].push_back(context);
}

void mip_get_solution_callback(const cuopt_float_t* solution,
                               const cuopt_float_t* objective_value,
                               const cuopt_float_t* solution_bound,
                               void* user_data)
{
  auto* context = static_cast<java_callback_context_t*>(user_data);
  if (context == nullptr || context->callback == nullptr) { return; }

  bool detach = false;
  JNIEnv* env = get_callback_env(detach);
  if (env == nullptr) { return; }

  jclass cls = env->GetObjectClass(context->callback);
  jmethodID method =
    env->GetMethodID(cls, "onSolution", "([DDDLjava/lang/Object;)V");
  if (method != nullptr) {
    std::vector<cuopt_float_t> values(solution, solution + context->num_variables);
    jdoubleArray solution_array = to_double_array(env, values);
    env->CallVoidMethod(context->callback,
                        method,
                        solution_array,
                        static_cast<jdouble>(*objective_value),
                        static_cast<jdouble>(*solution_bound),
                        context->user_data);
    env->DeleteLocalRef(solution_array);
  }

  if (detach) { g_jvm->DetachCurrentThread(); }
}

void mip_set_solution_callback(cuopt_float_t* solution,
                               cuopt_float_t* objective_value,
                               const cuopt_float_t* solution_bound,
                               void* user_data)
{
  auto* context = static_cast<java_callback_context_t*>(user_data);
  if (context == nullptr || context->callback == nullptr) { return; }

  bool detach = false;
  JNIEnv* env = get_callback_env(detach);
  if (env == nullptr) { return; }

  jclass cls = env->GetObjectClass(context->callback);
  jmethodID method = env->GetMethodID(
    cls, "getSolution", "(DLjava/lang/Object;)Lcom/nvidia/cuopt/linearprogramming/MipCallbackSolution;");
  if (method != nullptr) {
    jobject callback_solution =
      env->CallObjectMethod(context->callback, method, static_cast<jdouble>(*solution_bound), context->user_data);
    if (callback_solution != nullptr) {
      jclass result_cls = env->GetObjectClass(callback_solution);
      jfieldID solution_field = env->GetFieldID(result_cls, "solution", "[D");
      jfieldID objective_field = env->GetFieldID(result_cls, "objectiveValue", "D");
      if (solution_field != nullptr && objective_field != nullptr) {
        auto solution_array =
          static_cast<jdoubleArray>(env->GetObjectField(callback_solution, solution_field));
        const auto values = get_double_array(env, solution_array);
        if (values.size() == static_cast<size_t>(context->num_variables)) {
          std::memcpy(solution, values.data(), values.size() * sizeof(cuopt_float_t));
          *objective_value =
            static_cast<cuopt_float_t>(env->GetDoubleField(callback_solution, objective_field));
        }
      }
    }
  }

  if (detach) { g_jvm->DetachCurrentThread(); }
}

}  // namespace

extern "C" jint JNI_OnLoad(JavaVM* vm, void*)
{
  g_jvm = vm;
  return JNI_VERSION_1_8;
}

extern "C" JNIEXPORT jint JNICALL
Java_com_nvidia_cuopt_linearprogramming_NativeCuOpt_getFloatSize(JNIEnv*, jclass)
{
  return cuOptGetFloatSize();
}

extern "C" JNIEXPORT jobjectArray JNICALL
Java_com_nvidia_cuopt_linearprogramming_NativeCuOpt_getSolverParameterNames(JNIEnv* env, jclass)
{
  cuopt_int_t count = 0;
  if (!check_status(env, cuOptGetNumSolverParameters(&count), "cuOptGetNumSolverParameters")) {
    return nullptr;
  }
  jclass string_class = env->FindClass("java/lang/String");
  jobjectArray result = env->NewObjectArray(count, string_class, nullptr);
  for (cuopt_int_t i = 0; i < count; ++i) {
    char buffer[256] = {};
    if (!check_status(env, cuOptGetSolverParameterName(i, sizeof(buffer), buffer),
                      "cuOptGetSolverParameterName")) {
      return nullptr;
    }
    env->SetObjectArrayElement(result, i, env->NewStringUTF(buffer));
  }
  return result;
}

extern "C" JNIEXPORT jlong JNICALL
Java_com_nvidia_cuopt_linearprogramming_NativeCuOpt_createEmptyProblem(JNIEnv* env, jclass)
{
  try {
    auto problem = std::make_unique<cuopt::linear_programming::problem_and_stream_view_t>(
      cuopt::linear_programming::get_memory_backend_type());
    auto* raw_problem = problem.get();
    remember_jni_owned_problem(raw_problem);
    problem.release();
    return from_handle(raw_problem);
  } catch (const std::exception& e) {
    throw_cuopt_exception(env, CUOPT_RUNTIME_ERROR, std::string("createEmptyProblem failed: ") + e.what());
    return 0;
  }
}

extern "C" JNIEXPORT jlong JNICALL
Java_com_nvidia_cuopt_linearprogramming_NativeCuOpt_parseMpsProblem(JNIEnv* env,
                                                                     jclass,
                                                                     jstring path,
                                                                     jboolean fixed_mps_format)
{
  const auto filename = get_string(env, path);
  auto problem = std::make_unique<cuopt::linear_programming::problem_and_stream_view_t>(
    cuopt::linear_programming::get_memory_backend_type());
  try {
    auto data_model = cuopt::linear_programming::io::read_mps<int, double>(
      filename, static_cast<bool>(fixed_mps_format));
    cuopt::linear_programming::populate_from_mps_data_model(
      problem->get_problem(), data_model);
    auto* raw_problem = problem.get();
    remember_jni_owned_problem(raw_problem);
    problem.release();
    return from_handle(raw_problem);
  } catch (const std::exception& e) {
    const cuopt_int_t status =
      std::string(e.what()).find("Error opening input file") != std::string::npos
        ? CUOPT_MPS_FILE_ERROR
        : CUOPT_MPS_PARSE_ERROR;
    throw_cuopt_exception(env, status, std::string("parseMpsProblem failed: ") + e.what());
    return 0;
  }
}

extern "C" JNIEXPORT jlong JNICALL
Java_com_nvidia_cuopt_linearprogramming_NativeCuOpt_readProblemWithFormat(JNIEnv* env,
                                                                          jclass,
                                                                          jstring path,
                                                                          jboolean fixed_mps_format)
{
  const auto filename = get_string(env, path);
  auto problem = std::make_unique<cuopt::linear_programming::problem_and_stream_view_t>(
    cuopt::linear_programming::get_memory_backend_type());
  try {
    auto data_model = cuopt::linear_programming::io::read<int, double>(
      filename, static_cast<bool>(fixed_mps_format));
    cuopt::linear_programming::populate_from_mps_data_model(
      problem->get_problem(), data_model);
    auto* raw_problem = problem.get();
    remember_jni_owned_problem(raw_problem);
    problem.release();
    return from_handle(raw_problem);
  } catch (const std::exception& e) {
    const cuopt_int_t status =
      std::string(e.what()).find("Error opening input file") != std::string::npos
        ? CUOPT_MPS_FILE_ERROR
        : CUOPT_MPS_PARSE_ERROR;
    throw_cuopt_exception(env, status, std::string("readProblemWithFormat failed: ") + e.what());
    return 0;
  }
}

extern "C" JNIEXPORT jlong JNICALL
Java_com_nvidia_cuopt_linearprogramming_NativeCuOpt_createSolverSettings(JNIEnv* env, jclass)
{
  cuOptSolverSettings settings = nullptr;
  if (!check_status(env, cuOptCreateSolverSettings(&settings), "cuOptCreateSolverSettings")) {
    return 0;
  }
  return from_handle(settings);
}

extern "C" JNIEXPORT void JNICALL
Java_com_nvidia_cuopt_linearprogramming_NativeCuOpt_destroySolverSettings(JNIEnv* env,
                                                                          jclass,
                                                                          jlong handle)
{
  if (handle == 0) { return; }
  cleanup_callback_contexts(env, handle);
  cuopt_java_release_settings_state(to_settings(handle));
  cuOptSolverSettings settings = to_settings(handle);
  cuOptDestroySolverSettings(&settings);
}

extern "C" JNIEXPORT void JNICALL
Java_com_nvidia_cuopt_linearprogramming_NativeCuOpt_setParameter(JNIEnv* env,
                                                                 jclass,
                                                                 jlong handle,
                                                                 jstring name,
                                                                 jstring value)
{
  const auto parameter_name  = get_string(env, name);
  const auto parameter_value = get_string(env, value);
  check_status(env,
               cuOptSetParameter(to_settings(handle), parameter_name.c_str(), parameter_value.c_str()),
               "cuOptSetParameter");
}

extern "C" JNIEXPORT void JNICALL
Java_com_nvidia_cuopt_linearprogramming_NativeCuOpt_setIntegerParameter(JNIEnv* env,
                                                                        jclass,
                                                                        jlong handle,
                                                                        jstring name,
                                                                        jint value)
{
  const auto parameter_name = get_string(env, name);
  check_status(env,
               cuOptSetIntegerParameter(to_settings(handle), parameter_name.c_str(), value),
               "cuOptSetIntegerParameter");
}

extern "C" JNIEXPORT void JNICALL
Java_com_nvidia_cuopt_linearprogramming_NativeCuOpt_setFloatParameter(JNIEnv* env,
                                                                      jclass,
                                                                      jlong handle,
                                                                      jstring name,
                                                                      jdouble value)
{
  const auto parameter_name = get_string(env, name);
  check_status(env,
               cuOptSetFloatParameter(
                 to_settings(handle), parameter_name.c_str(), static_cast<cuopt_float_t>(value)),
               "cuOptSetFloatParameter");
}

extern "C" JNIEXPORT jstring JNICALL
Java_com_nvidia_cuopt_linearprogramming_NativeCuOpt_getParameter(JNIEnv* env,
                                                                 jclass,
                                                                 jlong handle,
                                                                 jstring name)
{
  const auto parameter_name = get_string(env, name);
  char buffer[256]          = {};
  if (!check_status(env,
                    cuOptGetParameter(to_settings(handle), parameter_name.c_str(), sizeof(buffer), buffer),
                    "cuOptGetParameter")) {
    return nullptr;
  }
  return env->NewStringUTF(buffer);
}

extern "C" JNIEXPORT void JNICALL
Java_com_nvidia_cuopt_linearprogramming_NativeCuOpt_loadParametersFromFile(JNIEnv* env,
                                                                          jclass,
                                                                          jlong handle,
                                                                          jstring path)
{
  const auto filename = get_string(env, path);
  check_status(env,
               cuOptLoadParametersFromFile(to_settings(handle), filename.c_str()),
               "cuOptLoadParametersFromFile");
}

extern "C" JNIEXPORT jboolean JNICALL
Java_com_nvidia_cuopt_linearprogramming_NativeCuOpt_dumpParametersToFile(JNIEnv* env,
                                                                          jclass,
                                                                          jlong handle,
                                                                          jstring path,
                                                                          jboolean hyperparameters_only)
{
  const auto filename = get_string(env, path);
  cuopt_int_t dumped_successfully = 0;
  if (!check_status(env,
                    cuOptDumpParametersToFile(to_settings(handle),
                                              filename.c_str(),
                                              hyperparameters_only ? 1 : 0,
                                              &dumped_successfully),
                    "cuOptDumpParametersToFile")) {
    return JNI_FALSE;
  }
  return dumped_successfully != 0 ? JNI_TRUE : JNI_FALSE;
}

extern "C" JNIEXPORT void JNICALL
Java_com_nvidia_cuopt_linearprogramming_NativeCuOpt_setInitialPrimalSolution(JNIEnv* env,
                                                                             jclass,
                                                                             jlong handle,
                                                                             jdoubleArray values)
{
  const auto data = get_double_array(env, values);
  check_status(env,
               cuOptSetInitialPrimalSolution(
                 to_settings(handle), data.data(), static_cast<cuopt_int_t>(data.size())),
               "cuOptSetInitialPrimalSolution");
}

extern "C" JNIEXPORT void JNICALL
Java_com_nvidia_cuopt_linearprogramming_NativeCuOpt_setInitialDualSolution(JNIEnv* env,
                                                                           jclass,
                                                                           jlong handle,
                                                                           jdoubleArray values)
{
  const auto data = get_double_array(env, values);
  check_status(env,
               cuOptSetInitialDualSolution(
                 to_settings(handle), data.data(), static_cast<cuopt_int_t>(data.size())),
               "cuOptSetInitialDualSolution");
}

extern "C" JNIEXPORT void JNICALL
Java_com_nvidia_cuopt_linearprogramming_NativeCuOpt_addMipStart(JNIEnv* env,
                                                                jclass,
                                                                jlong handle,
                                                                jdoubleArray values)
{
  const auto data = get_double_array(env, values);
  check_status(env,
               cuOptAddMIPStart(to_settings(handle), data.data(), static_cast<cuopt_int_t>(data.size())),
               "cuOptAddMIPStart");
}

extern "C" JNIEXPORT void JNICALL
Java_com_nvidia_cuopt_linearprogramming_NativeCuOpt_registerMipGetSolutionCallback(
  JNIEnv* env, jclass, jlong handle, jobject callback, jobject user_data, jint num_variables)
{
  auto* context      = new java_callback_context_t;
  context->callback = env->NewGlobalRef(callback);
  context->user_data = user_data == nullptr ? nullptr : env->NewGlobalRef(user_data);
  context->num_variables = num_variables;
  const auto status =
    cuOptSetMIPGetSolutionCallback(to_settings(handle), mip_get_solution_callback, context);
  if (!check_status(env, status, "cuOptSetMIPGetSolutionCallback")) {
    if (context->callback != nullptr) { env->DeleteGlobalRef(context->callback); }
    if (context->user_data != nullptr) { env->DeleteGlobalRef(context->user_data); }
    delete context;
    return;
  }
  remember_callback_context(handle, context);
}

extern "C" JNIEXPORT void JNICALL
Java_com_nvidia_cuopt_linearprogramming_NativeCuOpt_registerMipSetSolutionCallback(
  JNIEnv* env, jclass, jlong handle, jobject callback, jobject user_data, jint num_variables)
{
  auto* context      = new java_callback_context_t;
  context->callback = env->NewGlobalRef(callback);
  context->user_data = user_data == nullptr ? nullptr : env->NewGlobalRef(user_data);
  context->num_variables = num_variables;
  const auto status =
    cuOptSetMIPSetSolutionCallback(to_settings(handle), mip_set_solution_callback, context);
  if (!check_status(env, status, "cuOptSetMIPSetSolutionCallback")) {
    if (context->callback != nullptr) { env->DeleteGlobalRef(context->callback); }
    if (context->user_data != nullptr) { env->DeleteGlobalRef(context->user_data); }
    delete context;
    return;
  }
  remember_callback_context(handle, context);
}

extern "C" JNIEXPORT void JNICALL
Java_com_nvidia_cuopt_linearprogramming_NativeCuOpt_setPdlpWarmStartData(
  JNIEnv* env,
  jclass,
  jlong handle,
  jdoubleArray current_primal_solution,
  jdoubleArray current_dual_solution,
  jdoubleArray initial_primal_average,
  jdoubleArray initial_dual_average,
  jdoubleArray current_aty,
  jdoubleArray sum_primal_solutions,
  jdoubleArray sum_dual_solutions,
  jdoubleArray last_restart_duality_gap_primal_solution,
  jdoubleArray last_restart_duality_gap_dual_solution,
  jdouble initial_primal_weight,
  jdouble initial_step_size,
  jint total_pdlp_iterations,
  jint total_pdhg_iterations,
  jdouble last_candidate_kkt_score,
  jdouble last_restart_kkt_score,
  jdouble sum_solution_weight,
  jint iterations_since_last_restart)
{
  const auto current_primal = get_double_array(env, current_primal_solution);
  const auto current_dual = get_double_array(env, current_dual_solution);
  const auto initial_primal = get_double_array(env, initial_primal_average);
  const auto initial_dual = get_double_array(env, initial_dual_average);
  const auto aty = get_double_array(env, current_aty);
  const auto sum_primal = get_double_array(env, sum_primal_solutions);
  const auto sum_dual = get_double_array(env, sum_dual_solutions);
  const auto last_primal = get_double_array(env, last_restart_duality_gap_primal_solution);
  const auto last_dual = get_double_array(env, last_restart_duality_gap_dual_solution);

  check_status(
    env,
    cuOptSetPDLPWarmStartData(to_settings(handle),
                              current_primal.data(),
                              current_dual.data(),
                              initial_primal.data(),
                              initial_dual.data(),
                              aty.data(),
                              sum_primal.data(),
                              sum_dual.data(),
                              last_primal.data(),
                              last_dual.data(),
                              static_cast<cuopt_int_t>(current_primal.size()),
                              static_cast<cuopt_int_t>(current_dual.size()),
                              static_cast<cuopt_float_t>(initial_primal_weight),
                              static_cast<cuopt_float_t>(initial_step_size),
                              total_pdlp_iterations,
                              total_pdhg_iterations,
                              static_cast<cuopt_float_t>(last_candidate_kkt_score),
                              static_cast<cuopt_float_t>(last_restart_kkt_score),
                              static_cast<cuopt_float_t>(sum_solution_weight),
                              iterations_since_last_restart),
    "cuOptSetPDLPWarmStartData");
}

extern "C" JNIEXPORT jlong JNICALL
Java_com_nvidia_cuopt_linearprogramming_NativeCuOpt_createProblem(JNIEnv* env,
                                                                  jclass,
                                                                  jint num_constraints,
                                                                  jint num_variables,
                                                                  jint objective_sense,
                                                                  jdouble objective_offset,
                                                                  jdoubleArray objective_coefficients,
                                                                  jintArray row_offsets,
                                                                  jintArray column_indices,
                                                                  jdoubleArray values,
                                                                  jbyteArray constraint_sense,
                                                                  jdoubleArray rhs,
                                                                  jdoubleArray lower_bounds,
                                                                  jdoubleArray upper_bounds,
                                                                  jbyteArray variable_types)
{
  const auto obj    = get_double_array(env, objective_coefficients);
  const auto rows   = get_int_array(env, row_offsets);
  const auto cols   = get_int_array(env, column_indices);
  const auto coeffs = get_double_array(env, values);
  const auto senses = get_byte_array(env, constraint_sense);
  const auto rhs_values = get_double_array(env, rhs);
  const auto lbs = get_double_array(env, lower_bounds);
  const auto ubs = get_double_array(env, upper_bounds);
  const auto types = get_byte_array(env, variable_types);
  cuOptOptimizationProblem problem = nullptr;
  if (!check_status(env,
                    cuOptCreateProblem(num_constraints,
                                       num_variables,
                                       objective_sense,
                                       static_cast<cuopt_float_t>(objective_offset),
                                       obj.data(),
                                       rows.data(),
                                       cols.data(),
                                       coeffs.data(),
                                       senses.data(),
                                       rhs_values.data(),
                                       lbs.data(),
                                       ubs.data(),
                                       types.data(),
                                       &problem),
                    "cuOptCreateProblem")) {
    return 0;
  }
  return from_handle(problem);
}

extern "C" JNIEXPORT jlong JNICALL
Java_com_nvidia_cuopt_linearprogramming_NativeCuOpt_createRangedProblem(
  JNIEnv* env,
  jclass,
  jint num_constraints,
  jint num_variables,
  jint objective_sense,
  jdouble objective_offset,
  jdoubleArray objective_coefficients,
  jintArray row_offsets,
  jintArray column_indices,
  jdoubleArray values,
  jdoubleArray constraint_lower_bounds,
  jdoubleArray constraint_upper_bounds,
  jdoubleArray variable_lower_bounds,
  jdoubleArray variable_upper_bounds,
  jbyteArray variable_types)
{
  const auto obj = get_double_array(env, objective_coefficients);
  const auto rows = get_int_array(env, row_offsets);
  const auto cols = get_int_array(env, column_indices);
  const auto coeffs = get_double_array(env, values);
  const auto clb = get_double_array(env, constraint_lower_bounds);
  const auto cub = get_double_array(env, constraint_upper_bounds);
  const auto vlb = get_double_array(env, variable_lower_bounds);
  const auto vub = get_double_array(env, variable_upper_bounds);
  const auto types = get_byte_array(env, variable_types);
  cuOptOptimizationProblem problem = nullptr;
  if (!check_status(env,
                    cuOptCreateRangedProblem(num_constraints,
                                             num_variables,
                                             objective_sense,
                                             static_cast<cuopt_float_t>(objective_offset),
                                             obj.data(),
                                             rows.data(),
                                             cols.data(),
                                             coeffs.data(),
                                             clb.data(),
                                             cub.data(),
                                             vlb.data(),
                                             vub.data(),
                                             types.data(),
                                             &problem),
                    "cuOptCreateRangedProblem")) {
    return 0;
  }
  return from_handle(problem);
}

extern "C" JNIEXPORT jlong JNICALL
Java_com_nvidia_cuopt_linearprogramming_NativeCuOpt_readProblem(JNIEnv* env, jclass, jstring path)
{
  const auto filename = get_string(env, path);
  cuOptOptimizationProblem problem = nullptr;
  if (!check_status(env, cuOptReadProblem(filename.c_str(), &problem), "cuOptReadProblem")) {
    return 0;
  }
  return from_handle(problem);
}

extern "C" JNIEXPORT void JNICALL
Java_com_nvidia_cuopt_linearprogramming_NativeCuOpt_writeProblem(JNIEnv* env,
                                                                 jclass,
                                                                 jlong handle,
                                                                 jstring path)
{
  const auto filename = get_string(env, path);
  check_status(
    env, cuOptWriteProblem(to_problem(handle), filename.c_str(), CUOPT_FILE_FORMAT_MPS), "cuOptWriteProblem");
}

extern "C" JNIEXPORT void JNICALL
Java_com_nvidia_cuopt_linearprogramming_NativeCuOpt_destroyProblem(JNIEnv*, jclass, jlong handle)
{
  if (handle == 0) { return; }
  {
    std::lock_guard<std::mutex> lock(g_problem_state_mutex);
    g_problem_states.erase(handle);
  }
  if (take_jni_owned_problem(handle)) {
    delete to_problem_view(handle);
    return;
  }
  cuOptOptimizationProblem problem = to_problem(handle);
  cuOptDestroyProblem(&problem);
}

extern "C" JNIEXPORT void JNICALL
Java_com_nvidia_cuopt_linearprogramming_NativeCuOpt_setQuadraticObjective(JNIEnv* env,
                                                                          jclass,
                                                                          jlong handle,
                                                                          jintArray rows,
                                                                          jintArray cols,
                                                                          jdoubleArray coeffs)
{
  const auto row_data = get_int_array(env, rows);
  const auto col_data = get_int_array(env, cols);
  const auto val_data = get_double_array(env, coeffs);
  check_status(env,
               cuOptSetQuadraticObjective(to_problem(handle),
                                           static_cast<cuopt_int_t>(val_data.size()),
                                           row_data.data(),
                                           col_data.data(),
                                           val_data.data()),
               "cuOptSetQuadraticObjective");
}

extern "C" JNIEXPORT void JNICALL
Java_com_nvidia_cuopt_linearprogramming_NativeCuOpt_addQuadraticConstraint(JNIEnv* env,
                                                                           jclass,
                                                                           jlong handle,
                                                                           jintArray rows,
                                                                           jintArray cols,
                                                                           jdoubleArray coeffs,
                                                                           jintArray linear_indices,
                                                                           jdoubleArray linear_coeffs,
                                                                           jbyte sense,
                                                                           jdouble rhs)
{
  const auto row_data = get_int_array(env, rows);
  const auto col_data = get_int_array(env, cols);
  const auto val_data = get_double_array(env, coeffs);
  const auto lin_idx = get_int_array(env, linear_indices);
  const auto lin_coeff = get_double_array(env, linear_coeffs);
  check_status(env,
               cuOptAddQuadraticConstraint(to_problem(handle),
                                            static_cast<cuopt_int_t>(val_data.size()),
                                            row_data.data(),
                                            col_data.data(),
                                            val_data.data(),
                                            static_cast<cuopt_int_t>(lin_coeff.size()),
                                            lin_idx.data(),
                                            lin_coeff.data(),
                                            static_cast<char>(sense),
                                            static_cast<cuopt_float_t>(rhs)),
               "cuOptAddQuadraticConstraint");
}

extern "C" JNIEXPORT jint JNICALL
Java_com_nvidia_cuopt_linearprogramming_NativeCuOpt_getNumVariables(JNIEnv* env,
                                                                    jclass,
                                                                    jlong handle)
{
  cuopt_int_t value = 0;
  check_status(env, cuOptGetNumVariables(to_problem(handle), &value), "cuOptGetNumVariables");
  return value;
}

extern "C" JNIEXPORT jint JNICALL
Java_com_nvidia_cuopt_linearprogramming_NativeCuOpt_getNumConstraints(JNIEnv* env,
                                                                      jclass,
                                                                      jlong handle)
{
  cuopt_int_t value = 0;
  check_status(env, cuOptGetNumConstraints(to_problem(handle), &value), "cuOptGetNumConstraints");
  return value;
}

extern "C" JNIEXPORT jint JNICALL
Java_com_nvidia_cuopt_linearprogramming_NativeCuOpt_getNumNonZeros(JNIEnv* env,
                                                                   jclass,
                                                                   jlong handle)
{
  cuopt_int_t value = 0;
  check_status(env, cuOptGetNumNonZeros(to_problem(handle), &value), "cuOptGetNumNonZeros");
  return value;
}

extern "C" JNIEXPORT jint JNICALL
Java_com_nvidia_cuopt_linearprogramming_NativeCuOpt_getObjectiveSense(JNIEnv* env,
                                                                      jclass,
                                                                      jlong handle)
{
  cuopt_int_t value = 0;
  check_status(env, cuOptGetObjectiveSense(to_problem(handle), &value), "cuOptGetObjectiveSense");
  return value;
}

extern "C" JNIEXPORT jdouble JNICALL
Java_com_nvidia_cuopt_linearprogramming_NativeCuOpt_getObjectiveOffset(JNIEnv* env,
                                                                       jclass,
                                                                       jlong handle)
{
  cuopt_float_t value = 0;
  check_status(env, cuOptGetObjectiveOffset(to_problem(handle), &value), "cuOptGetObjectiveOffset");
  return value;
}

extern "C" JNIEXPORT jdoubleArray JNICALL
Java_com_nvidia_cuopt_linearprogramming_NativeCuOpt_getObjectiveCoefficients(JNIEnv* env,
                                                                             jclass,
                                                                             jlong handle)
{
  const int n = Java_com_nvidia_cuopt_linearprogramming_NativeCuOpt_getNumVariables(env, nullptr, handle);
  std::vector<cuopt_float_t> values(static_cast<size_t>(n));
  if (!check_status(env,
                    cuOptGetObjectiveCoefficients(to_problem(handle), values.data()),
                    "cuOptGetObjectiveCoefficients")) {
    return nullptr;
  }
  return to_double_array(env, values);
}

extern "C" JNIEXPORT jobjectArray JNICALL
Java_com_nvidia_cuopt_linearprogramming_NativeCuOpt_getConstraintMatrix(JNIEnv* env,
                                                                        jclass,
                                                                        jlong handle)
{
  const int rows_size = Java_com_nvidia_cuopt_linearprogramming_NativeCuOpt_getNumConstraints(env, nullptr, handle) + 1;
  const int nnz = Java_com_nvidia_cuopt_linearprogramming_NativeCuOpt_getNumNonZeros(env, nullptr, handle);
  std::vector<cuopt_int_t> rows(static_cast<size_t>(rows_size));
  std::vector<cuopt_int_t> cols(static_cast<size_t>(nnz));
  std::vector<cuopt_float_t> values(static_cast<size_t>(nnz));
  if (!check_status(env,
                    cuOptGetConstraintMatrix(to_problem(handle), rows.data(), cols.data(), values.data()),
                    "cuOptGetConstraintMatrix")) {
    return nullptr;
  }
  jclass object_class = env->FindClass("java/lang/Object");
  jobjectArray result = env->NewObjectArray(3, object_class, nullptr);
  env->SetObjectArrayElement(result, 0, to_int_array(env, rows));
  env->SetObjectArrayElement(result, 1, to_int_array(env, cols));
  env->SetObjectArrayElement(result, 2, to_double_array(env, values));
  return result;
}

extern "C" JNIEXPORT void JNICALL
Java_com_nvidia_cuopt_linearprogramming_NativeCuOpt_setMaximize(JNIEnv* env,
                                                                jclass,
                                                                jlong handle,
                                                                jboolean maximize)
{
  run_problem_operation(env, "setMaximize", [&] {
    to_problem_view(handle)->get_problem()->set_maximize(maximize != JNI_FALSE);
  });
}

extern "C" JNIEXPORT void JNICALL
Java_com_nvidia_cuopt_linearprogramming_NativeCuOpt_setConstraintMatrix(JNIEnv* env,
                                                                        jclass,
                                                                        jlong handle,
                                                                        jdoubleArray values,
                                                                        jintArray indices,
                                                                        jintArray offsets)
{
  const auto h_values = get_double_array(env, values);
  const auto h_indices = get_int_array(env, indices);
  const auto h_offsets = get_int_array(env, offsets);
  run_problem_operation(env, "setConstraintMatrix", [&] {
    to_problem_view(handle)->get_problem()->set_csr_constraint_matrix(
      h_values.data(), h_values.size(), h_indices.data(), h_indices.size(), h_offsets.data(), h_offsets.size());
  });
}

extern "C" JNIEXPORT void JNICALL
Java_com_nvidia_cuopt_linearprogramming_NativeCuOpt_setConstraintBounds(JNIEnv* env,
                                                                        jclass,
                                                                        jlong handle,
                                                                        jdoubleArray values)
{
  const auto h_values = get_double_array(env, values);
  run_problem_operation(env, "setConstraintBounds", [&] {
    to_problem_view(handle)->get_problem()->set_constraint_bounds(h_values.data(), h_values.size());
  });
}

extern "C" JNIEXPORT void JNICALL
Java_com_nvidia_cuopt_linearprogramming_NativeCuOpt_setObjectiveCoefficients(JNIEnv* env,
                                                                            jclass,
                                                                            jlong handle,
                                                                            jdoubleArray values)
{
  const auto h_values = get_double_array(env, values);
  run_problem_operation(env, "setObjectiveCoefficients", [&] {
    to_problem_view(handle)->get_problem()->set_objective_coefficients(h_values.data(), h_values.size());
  });
}

extern "C" JNIEXPORT void JNICALL
Java_com_nvidia_cuopt_linearprogramming_NativeCuOpt_setObjectiveScalingFactor(JNIEnv* env,
                                                                              jclass,
                                                                              jlong handle,
                                                                              jdouble value)
{
  run_problem_operation(env, "setObjectiveScalingFactor", [&] {
    to_problem_view(handle)->get_problem()->set_objective_scaling_factor(value);
  });
}

extern "C" JNIEXPORT jdouble JNICALL
Java_com_nvidia_cuopt_linearprogramming_NativeCuOpt_getObjectiveScalingFactor(JNIEnv* env,
                                                                              jclass,
                                                                              jlong handle)
{
  jdouble value = 0.0;
  if (!run_problem_operation(env, "getObjectiveScalingFactor", [&] {
        value = to_problem_view(handle)->get_problem()->get_objective_scaling_factor();
      })) {
    return 0.0;
  }
  return value;
}

extern "C" JNIEXPORT void JNICALL
Java_com_nvidia_cuopt_linearprogramming_NativeCuOpt_setObjectiveOffset(JNIEnv* env,
                                                                       jclass,
                                                                       jlong handle,
                                                                       jdouble value)
{
  run_problem_operation(env, "setObjectiveOffset", [&] {
    to_problem_view(handle)->get_problem()->set_objective_offset(value);
  });
}

extern "C" JNIEXPORT void JNICALL
Java_com_nvidia_cuopt_linearprogramming_NativeCuOpt_setQuadraticObjectiveMatrix(JNIEnv* env,
                                                                                jclass,
                                                                                jlong handle,
                                                                                jdoubleArray values,
                                                                                jintArray indices,
                                                                                jintArray offsets)
{
  const auto h_values = get_double_array(env, values);
  const auto h_indices = get_int_array(env, indices);
  const auto h_offsets = get_int_array(env, offsets);
  run_problem_operation(env, "setQuadraticObjectiveMatrix", [&] {
    to_problem_view(handle)->get_problem()->set_quadratic_objective_matrix(
      h_values.data(), h_values.size(), h_indices.data(), h_indices.size(), h_offsets.data(), h_offsets.size());
  });
}

extern "C" JNIEXPORT void JNICALL
Java_com_nvidia_cuopt_linearprogramming_NativeCuOpt_setVariableLowerBounds(JNIEnv* env,
                                                                          jclass,
                                                                          jlong handle,
                                                                          jdoubleArray values)
{
  const auto h_values = get_double_array(env, values);
  run_problem_operation(env, "setVariableLowerBounds", [&] {
    to_problem_view(handle)->get_problem()->set_variable_lower_bounds(h_values.data(), h_values.size());
  });
}

extern "C" JNIEXPORT void JNICALL
Java_com_nvidia_cuopt_linearprogramming_NativeCuOpt_setVariableUpperBounds(JNIEnv* env,
                                                                          jclass,
                                                                          jlong handle,
                                                                          jdoubleArray values)
{
  const auto h_values = get_double_array(env, values);
  run_problem_operation(env, "setVariableUpperBounds", [&] {
    to_problem_view(handle)->get_problem()->set_variable_upper_bounds(h_values.data(), h_values.size());
  });
}

extern "C" JNIEXPORT void JNICALL
Java_com_nvidia_cuopt_linearprogramming_NativeCuOpt_setConstraintLowerBounds(JNIEnv* env,
                                                                             jclass,
                                                                             jlong handle,
                                                                             jdoubleArray values)
{
  const auto h_values = get_double_array(env, values);
  run_problem_operation(env, "setConstraintLowerBounds", [&] {
    to_problem_view(handle)->get_problem()->set_constraint_lower_bounds(h_values.data(), h_values.size());
  });
}

extern "C" JNIEXPORT void JNICALL
Java_com_nvidia_cuopt_linearprogramming_NativeCuOpt_setConstraintUpperBounds(JNIEnv* env,
                                                                             jclass,
                                                                             jlong handle,
                                                                             jdoubleArray values)
{
  const auto h_values = get_double_array(env, values);
  run_problem_operation(env, "setConstraintUpperBounds", [&] {
    to_problem_view(handle)->get_problem()->set_constraint_upper_bounds(h_values.data(), h_values.size());
  });
}

extern "C" JNIEXPORT void JNICALL
Java_com_nvidia_cuopt_linearprogramming_NativeCuOpt_setRowTypes(JNIEnv* env,
                                                                jclass,
                                                                jlong handle,
                                                                jbyteArray values)
{
  const auto h_values = get_byte_array(env, values);
  run_problem_operation(env, "setRowTypes", [&] {
    to_problem_view(handle)->get_problem()->set_row_types(h_values.data(), h_values.size());
  });
}

extern "C" JNIEXPORT void JNICALL
Java_com_nvidia_cuopt_linearprogramming_NativeCuOpt_setVariableTypes(JNIEnv* env,
                                                                     jclass,
                                                                     jlong handle,
                                                                     jbyteArray values)
{
  const auto h_values = get_byte_array(env, values);
  run_problem_operation(env, "setVariableTypes", [&] {
    std::vector<cuopt::linear_programming::var_t> variable_types;
    variable_types.reserve(h_values.size());
    for (char value : h_values) {
      variable_types.push_back(cuopt::linear_programming::detail::char_to_var_type(value));
    }
    to_problem_view(handle)->get_problem()->set_variable_types(variable_types.data(), variable_types.size());
  });
}

extern "C" JNIEXPORT void JNICALL
Java_com_nvidia_cuopt_linearprogramming_NativeCuOpt_setVariableNames(JNIEnv* env,
                                                                     jclass,
                                                                     jlong handle,
                                                                     jobjectArray values)
{
  const auto h_values = get_string_array(env, values);
  run_problem_operation(env, "setVariableNames", [&] {
    to_problem_view(handle)->get_problem()->set_variable_names(h_values);
  });
}

extern "C" JNIEXPORT void JNICALL
Java_com_nvidia_cuopt_linearprogramming_NativeCuOpt_setRowNames(JNIEnv* env,
                                                                jclass,
                                                                jlong handle,
                                                                jobjectArray values)
{
  const auto h_values = get_string_array(env, values);
  run_problem_operation(env, "setRowNames", [&] {
    to_problem_view(handle)->get_problem()->set_row_names(h_values);
  });
}

extern "C" JNIEXPORT void JNICALL
Java_com_nvidia_cuopt_linearprogramming_NativeCuOpt_setObjectiveName(JNIEnv* env,
                                                                     jclass,
                                                                     jlong handle,
                                                                     jstring value)
{
  const auto name = get_string(env, value);
  run_problem_operation(env, "setObjectiveName", [&] {
    to_problem_view(handle)->get_problem()->set_objective_name(name);
  });
}

extern "C" JNIEXPORT void JNICALL
Java_com_nvidia_cuopt_linearprogramming_NativeCuOpt_setProblemName(JNIEnv* env,
                                                                   jclass,
                                                                   jlong handle,
                                                                   jstring value)
{
  const auto name = get_string(env, value);
  run_problem_operation(env, "setProblemName", [&] {
    to_problem_view(handle)->get_problem()->set_problem_name(name);
  });
}

extern "C" JNIEXPORT void JNICALL
Java_com_nvidia_cuopt_linearprogramming_NativeCuOpt_setInitialPrimalSolutionOnProblem(JNIEnv* env,
                                                                                      jclass,
                                                                                      jlong handle,
                                                                                      jdoubleArray values)
{
  const auto h_values = get_double_array(env, values);
  run_problem_operation(env, "setInitialPrimalSolutionOnProblem", [&] {
    std::lock_guard<std::mutex> lock(g_problem_state_mutex);
    g_problem_states[handle].initial_primal_solution = h_values;
  });
}

extern "C" JNIEXPORT void JNICALL
Java_com_nvidia_cuopt_linearprogramming_NativeCuOpt_setInitialDualSolutionOnProblem(JNIEnv* env,
                                                                                    jclass,
                                                                                    jlong handle,
                                                                                    jdoubleArray values)
{
  const auto h_values = get_double_array(env, values);
  run_problem_operation(env, "setInitialDualSolutionOnProblem", [&] {
    std::lock_guard<std::mutex> lock(g_problem_state_mutex);
    g_problem_states[handle].initial_dual_solution = h_values;
  });
}

extern "C" JNIEXPORT jdoubleArray JNICALL
Java_com_nvidia_cuopt_linearprogramming_NativeCuOpt_getQuadraticObjectiveValues(JNIEnv* env,
                                                                                jclass,
                                                                                jlong handle)
{
  return to_double_array(env, to_problem_view(handle)->get_problem()->get_quadratic_objective_values());
}

extern "C" JNIEXPORT jintArray JNICALL
Java_com_nvidia_cuopt_linearprogramming_NativeCuOpt_getQuadraticObjectiveIndices(JNIEnv* env,
                                                                                 jclass,
                                                                                 jlong handle)
{
  return to_int_array(env, to_problem_view(handle)->get_problem()->get_quadratic_objective_indices());
}

extern "C" JNIEXPORT jintArray JNICALL
Java_com_nvidia_cuopt_linearprogramming_NativeCuOpt_getQuadraticObjectiveOffsets(JNIEnv* env,
                                                                                 jclass,
                                                                                 jlong handle)
{
  return to_int_array(env, to_problem_view(handle)->get_problem()->get_quadratic_objective_offsets());
}

extern "C" JNIEXPORT jobjectArray JNICALL
Java_com_nvidia_cuopt_linearprogramming_NativeCuOpt_getVariableNames(JNIEnv* env,
                                                                     jclass,
                                                                     jlong handle)
{
  return to_string_array(env, to_problem_view(handle)->get_problem()->get_variable_names());
}

extern "C" JNIEXPORT jobjectArray JNICALL
Java_com_nvidia_cuopt_linearprogramming_NativeCuOpt_getRowNames(JNIEnv* env,
                                                                jclass,
                                                                jlong handle)
{
  return to_string_array(env, to_problem_view(handle)->get_problem()->get_row_names());
}

extern "C" JNIEXPORT jstring JNICALL
Java_com_nvidia_cuopt_linearprogramming_NativeCuOpt_getObjectiveName(JNIEnv* env,
                                                                     jclass,
                                                                     jlong handle)
{
  return env->NewStringUTF(to_problem_view(handle)->get_problem()->get_objective_name().c_str());
}

extern "C" JNIEXPORT jstring JNICALL
Java_com_nvidia_cuopt_linearprogramming_NativeCuOpt_getProblemName(JNIEnv* env,
                                                                   jclass,
                                                                   jlong handle)
{
  return env->NewStringUTF(to_problem_view(handle)->get_problem()->get_problem_name().c_str());
}

extern "C" JNIEXPORT jint JNICALL
Java_com_nvidia_cuopt_linearprogramming_NativeCuOpt_getProblemCategory(JNIEnv* env,
                                                                       jclass,
                                                                       jlong handle)
{
  jint category = 0;
  if (!run_problem_operation(env, "getProblemCategory", [&] {
        category = static_cast<jint>(to_problem_view(handle)->get_problem()->get_problem_category());
      })) {
    return 0;
  }
  return category;
}

extern "C" JNIEXPORT jobjectArray JNICALL
Java_com_nvidia_cuopt_linearprogramming_NativeCuOpt_getQuadraticConstraints(JNIEnv* env,
                                                                            jclass,
                                                                            jlong handle)
{
  const auto& constraints = to_problem_view(handle)->get_problem()->get_quadratic_constraints();
  jclass object_class = env->FindClass("java/lang/Object");
  jobjectArray result = env->NewObjectArray(static_cast<jsize>(constraints.size()), object_class, nullptr);
  for (jsize i = 0; i < static_cast<jsize>(constraints.size()); ++i) {
    const auto& constraint = constraints[static_cast<size_t>(i)];
    jobjectArray entry = env->NewObjectArray(9, object_class, nullptr);
    env->SetObjectArrayElement(entry, 0, to_int_array(env, {constraint.constraint_row_index}));
    env->SetObjectArrayElement(entry, 1, env->NewStringUTF(constraint.constraint_row_name.c_str()));
    env->SetObjectArrayElement(entry, 2, to_byte_array(env, {constraint.constraint_row_type}));
    env->SetObjectArrayElement(entry, 3, to_double_array(env, constraint.linear_values));
    env->SetObjectArrayElement(entry, 4, to_int_array(env, constraint.linear_indices));
    env->SetObjectArrayElement(entry, 5, to_double_array(env, {constraint.rhs_value}));
    env->SetObjectArrayElement(entry, 6, to_int_array(env, constraint.rows));
    env->SetObjectArrayElement(entry, 7, to_int_array(env, constraint.cols));
    env->SetObjectArrayElement(entry, 8, to_double_array(env, constraint.vals));
    env->SetObjectArrayElement(result, i, entry);
    env->DeleteLocalRef(entry);
  }
  return result;
}

extern "C" JNIEXPORT void JNICALL
Java_com_nvidia_cuopt_linearprogramming_NativeCuOpt_clearQuadraticConstraints(JNIEnv* env,
                                                                              jclass,
                                                                              jlong handle)
{
  run_problem_operation(env, "clearQuadraticConstraints", [&] {
    using problem_t = cuopt::linear_programming::optimization_problem_interface_t<int, double>;
    to_problem_view(handle)->get_problem()->set_quadratic_constraints(
      std::vector<problem_t::quadratic_constraint_t>{});
  });
}

#define DEFINE_DOUBLE_PROBLEM_GETTER(JAVA_NAME, C_NAME, COUNT_EXPR)                         \
  extern "C" JNIEXPORT jdoubleArray JNICALL                                                \
    Java_com_nvidia_cuopt_linearprogramming_NativeCuOpt_##JAVA_NAME(                        \
      JNIEnv* env, jclass, jlong handle)                                                     \
  {                                                                                          \
    const int count = (COUNT_EXPR);                                                          \
    std::vector<cuopt_float_t> values(static_cast<size_t>(count));                           \
    if (!check_status(env, C_NAME(to_problem(handle), values.data()), #C_NAME)) { return nullptr; } \
    return to_double_array(env, values);                                                     \
  }

#define DEFINE_BYTE_PROBLEM_GETTER(JAVA_NAME, C_NAME, COUNT_EXPR)                          \
  extern "C" JNIEXPORT jbyteArray JNICALL                                                  \
    Java_com_nvidia_cuopt_linearprogramming_NativeCuOpt_##JAVA_NAME(                       \
      JNIEnv* env, jclass, jlong handle)                                                    \
  {                                                                                         \
    const int count = (COUNT_EXPR);                                                         \
    std::vector<char> values(static_cast<size_t>(count));                                   \
    if (!check_status(env, C_NAME(to_problem(handle), values.data()), #C_NAME)) { return nullptr; } \
    return to_byte_array(env, values);                                                      \
  }

DEFINE_DOUBLE_PROBLEM_GETTER(getConstraintRhs,
                             cuOptGetConstraintRightHandSide,
                             Java_com_nvidia_cuopt_linearprogramming_NativeCuOpt_getNumConstraints(env, nullptr, handle))

extern "C" JNIEXPORT jdoubleArray JNICALL
Java_com_nvidia_cuopt_linearprogramming_NativeCuOpt_getConstraintLowerBounds(JNIEnv* env,
                                                                              jclass,
                                                                              jlong handle)
{
  std::vector<cuopt_float_t> values;
  if (!run_problem_operation(env, "getConstraintLowerBounds", [&] {
        values = to_problem_view(handle)->get_problem()->get_constraint_lower_bounds_host();
      })) {
    return nullptr;
  }
  return to_double_array(env, values);
}

extern "C" JNIEXPORT jdoubleArray JNICALL
Java_com_nvidia_cuopt_linearprogramming_NativeCuOpt_getConstraintUpperBounds(JNIEnv* env,
                                                                              jclass,
                                                                              jlong handle)
{
  std::vector<cuopt_float_t> values;
  if (!run_problem_operation(env, "getConstraintUpperBounds", [&] {
        values = to_problem_view(handle)->get_problem()->get_constraint_upper_bounds_host();
      })) {
    return nullptr;
  }
  return to_double_array(env, values);
}

extern "C" JNIEXPORT jdoubleArray JNICALL
Java_com_nvidia_cuopt_linearprogramming_NativeCuOpt_getVariableLowerBounds(JNIEnv* env,
                                                                            jclass,
                                                                            jlong handle)
{
  std::vector<cuopt_float_t> values;
  if (!run_problem_operation(env, "getVariableLowerBounds", [&] {
        values = to_problem_view(handle)->get_problem()->get_variable_lower_bounds_host();
      })) {
    return nullptr;
  }
  return to_double_array(env, values);
}

extern "C" JNIEXPORT jdoubleArray JNICALL
Java_com_nvidia_cuopt_linearprogramming_NativeCuOpt_getVariableUpperBounds(JNIEnv* env,
                                                                            jclass,
                                                                            jlong handle)
{
  std::vector<cuopt_float_t> values;
  if (!run_problem_operation(env, "getVariableUpperBounds", [&] {
        values = to_problem_view(handle)->get_problem()->get_variable_upper_bounds_host();
      })) {
    return nullptr;
  }
  return to_double_array(env, values);
}

DEFINE_BYTE_PROBLEM_GETTER(getConstraintSense,
                           cuOptGetConstraintSense,
                           Java_com_nvidia_cuopt_linearprogramming_NativeCuOpt_getNumConstraints(env, nullptr, handle))
DEFINE_BYTE_PROBLEM_GETTER(getVariableTypes,
                           cuOptGetVariableTypes,
                           Java_com_nvidia_cuopt_linearprogramming_NativeCuOpt_getNumVariables(env, nullptr, handle))

#undef DEFINE_DOUBLE_PROBLEM_GETTER
#undef DEFINE_BYTE_PROBLEM_GETTER

extern "C" JNIEXPORT jboolean JNICALL
Java_com_nvidia_cuopt_linearprogramming_NativeCuOpt_isMip(JNIEnv* env, jclass, jlong handle)
{
  cuopt_int_t value = 0;
  check_status(env, cuOptIsMIP(to_problem(handle), &value), "cuOptIsMIP");
  return static_cast<jboolean>(value != 0);
}

extern "C" JNIEXPORT jlong JNICALL
Java_com_nvidia_cuopt_linearprogramming_NativeCuOpt_solve(JNIEnv* env,
                                                          jclass,
                                                          jlong problem_handle,
                                                          jlong settings_handle)
{
  cuopt_int_t is_mip = 0;
  if (!check_status(env,
                    cuOptIsMIP(to_problem(problem_handle), &is_mip),
                    "cuOptIsMIP")) {
    return 0;
  }
  java_problem_state_t problem_state;
  {
    std::lock_guard<std::mutex> lock(g_problem_state_mutex);
    auto it = g_problem_states.find(problem_handle);
    if (it != g_problem_states.end()) { problem_state = it->second; }
  }
  if (is_mip != 0) {
    if (!problem_state.initial_primal_solution.empty() &&
        !check_status(env,
                      cuOptAddMIPStart(to_settings(settings_handle),
                                       problem_state.initial_primal_solution.data(),
                                       problem_state.initial_primal_solution.size()),
                      "cuOptAddMIPStart")) {
      return 0;
    }
  } else {
    if (!problem_state.initial_primal_solution.empty() &&
        !check_status(env,
                      cuOptSetInitialPrimalSolution(to_settings(settings_handle),
                                                    problem_state.initial_primal_solution.data(),
                                                    problem_state.initial_primal_solution.size()),
                      "cuOptSetInitialPrimalSolution")) {
      return 0;
    }
    if (!problem_state.initial_dual_solution.empty() &&
        !check_status(env,
                      cuOptSetInitialDualSolution(to_settings(settings_handle),
                                                  problem_state.initial_dual_solution.data(),
                                                  problem_state.initial_dual_solution.size()),
                      "cuOptSetInitialDualSolution")) {
      return 0;
    }
  }
  cuOptSolution solution = nullptr;
  if (!check_status(env,
                    cuOptSolve(to_problem(problem_handle), to_settings(settings_handle), &solution),
                    "cuOptSolve")) {
    return 0;
  }
  return from_handle(solution);
}

extern "C" JNIEXPORT void JNICALL
Java_com_nvidia_cuopt_linearprogramming_NativeCuOpt_destroySolution(JNIEnv*, jclass, jlong handle)
{
  if (handle == 0) { return; }
  cuOptSolution solution = to_solution(handle);
  cuOptDestroySolution(&solution);
}

extern "C" JNIEXPORT jboolean JNICALL
Java_com_nvidia_cuopt_linearprogramming_NativeCuOpt_solutionIsMip(JNIEnv* env,
                                                                  jclass,
                                                                  jlong handle)
{
  cuopt_int_t value = 0;
  check_status(env, cuOptSolutionIsMIP(to_solution(handle), &value), "cuOptSolutionIsMIP");
  return static_cast<jboolean>(value != 0);
}

extern "C" JNIEXPORT jint JNICALL
Java_com_nvidia_cuopt_linearprogramming_NativeCuOpt_getTerminationStatus(JNIEnv* env,
                                                                         jclass,
                                                                         jlong handle)
{
  cuopt_int_t value = 0;
  check_status(env, cuOptGetTerminationStatus(to_solution(handle), &value), "cuOptGetTerminationStatus");
  return value;
}

extern "C" JNIEXPORT jint JNICALL
Java_com_nvidia_cuopt_linearprogramming_NativeCuOpt_getErrorStatus(JNIEnv* env,
                                                                   jclass,
                                                                   jlong handle)
{
  cuopt_int_t value = 0;
  check_status(env, cuOptGetErrorStatus(to_solution(handle), &value), "cuOptGetErrorStatus");
  return value;
}

extern "C" JNIEXPORT jstring JNICALL
Java_com_nvidia_cuopt_linearprogramming_NativeCuOpt_getErrorString(JNIEnv* env,
                                                                   jclass,
                                                                   jlong handle)
{
  char buffer[1024] = {};
  if (!check_status(env, cuOptGetErrorString(to_solution(handle), buffer, sizeof(buffer)), "cuOptGetErrorString")) {
    return nullptr;
  }
  return env->NewStringUTF(buffer);
}

extern "C" JNIEXPORT jdoubleArray JNICALL
Java_com_nvidia_cuopt_linearprogramming_NativeCuOpt_getPrimalSolution(JNIEnv* env,
                                                                      jclass,
                                                                      jlong handle,
                                                                      jint size)
{
  std::vector<cuopt_float_t> values(static_cast<size_t>(size));
  if (!check_status(env, cuOptGetPrimalSolution(to_solution(handle), values.data()), "cuOptGetPrimalSolution")) {
    return nullptr;
  }
  return to_double_array(env, values);
}

extern "C" JNIEXPORT jint JNICALL
Java_com_nvidia_cuopt_linearprogramming_NativeCuOpt_getDualSolutionSize(JNIEnv* env,
                                                                        jclass,
                                                                        jlong handle)
{
  auto* solution = to_lp_solution(env, handle, "getDualSolution");
  if (solution == nullptr) { return 0; }
  return solution->get_dual_solution_size();
}

extern "C" JNIEXPORT jdoubleArray JNICALL
Java_com_nvidia_cuopt_linearprogramming_NativeCuOpt_getDualSolution(JNIEnv* env,
                                                                    jclass,
                                                                    jlong handle,
                                                                    jint)
{
  auto* solution = to_lp_solution(env, handle, "getDualSolution");
  if (solution == nullptr) { return nullptr; }
  try {
    return to_double_array(env, solution->get_dual_solution_host());
  } catch (const std::exception& e) {
    throw_cuopt_exception(
      env, CUOPT_INVALID_ARGUMENT, std::string("getDualSolution failed: ") + e.what());
    return nullptr;
  }
}

extern "C" JNIEXPORT jdoubleArray JNICALL
Java_com_nvidia_cuopt_linearprogramming_NativeCuOpt_getReducedCosts(JNIEnv* env,
                                                                    jclass,
                                                                    jlong handle,
                                                                    jint)
{
  auto* solution = to_lp_solution(env, handle, "getReducedCost");
  if (solution == nullptr) { return nullptr; }
  try {
    return to_double_array(env, solution->get_reduced_cost_host());
  } catch (const std::exception& e) {
    throw_cuopt_exception(
      env, CUOPT_INVALID_ARGUMENT, std::string("getReducedCost failed: ") + e.what());
    return nullptr;
  }
}

extern "C" JNIEXPORT jdouble JNICALL
Java_com_nvidia_cuopt_linearprogramming_NativeCuOpt_getObjectiveValue(JNIEnv* env,
                                                                      jclass,
                                                                      jlong handle)
{
  cuopt_float_t value = 0;
  check_status(env, cuOptGetObjectiveValue(to_solution(handle), &value), "cuOptGetObjectiveValue");
  return value;
}

extern "C" JNIEXPORT jdouble JNICALL
Java_com_nvidia_cuopt_linearprogramming_NativeCuOpt_getDualObjectiveValue(JNIEnv* env,
                                                                          jclass,
                                                                          jlong handle)
{
  auto* solution = to_lp_solution(env, handle, "getDualObjective");
  if (solution == nullptr) { return 0; }
  try {
    return solution->get_dual_objective_value(0);
  } catch (const std::exception& e) {
    throw_cuopt_exception(
      env, CUOPT_INVALID_ARGUMENT, std::string("getDualObjective failed: ") + e.what());
    return 0;
  }
}

extern "C" JNIEXPORT jdouble JNICALL
Java_com_nvidia_cuopt_linearprogramming_NativeCuOpt_getSolveTime(JNIEnv* env,
                                                                 jclass,
                                                                 jlong handle)
{
  cuopt_float_t value = 0;
  check_status(env, cuOptGetSolveTime(to_solution(handle), &value), "cuOptGetSolveTime");
  return value;
}

extern "C" JNIEXPORT jdouble JNICALL
Java_com_nvidia_cuopt_linearprogramming_NativeCuOpt_getMipGap(JNIEnv* env,
                                                              jclass,
                                                              jlong handle)
{
  cuopt_float_t value = 0;
  check_status(env, cuOptGetMIPGap(to_solution(handle), &value), "cuOptGetMIPGap");
  return value;
}

extern "C" JNIEXPORT jdouble JNICALL
Java_com_nvidia_cuopt_linearprogramming_NativeCuOpt_getSolutionBound(JNIEnv* env,
                                                                     jclass,
                                                                     jlong handle)
{
  cuopt_float_t value = 0;
  check_status(env, cuOptGetSolutionBound(to_solution(handle), &value), "cuOptGetSolutionBound");
  return value;
}

extern "C" JNIEXPORT jdoubleArray JNICALL
Java_com_nvidia_cuopt_linearprogramming_NativeCuOpt_getLpStats(JNIEnv* env, jclass, jlong handle)
{
  cuopt_float_t primal = 0;
  cuopt_float_t dual = 0;
  cuopt_float_t gap = 0;
  cuopt_int_t iterations = 0;
  cuopt_int_t solved_by = 0;
  if (!check_status(env,
                    cuOptGetLPSolverStats(to_solution(handle), &primal, &dual, &gap, &iterations, &solved_by),
                    "cuOptGetLPSolverStats")) {
    return nullptr;
  }
  return to_double_array(env, {primal, dual, gap, static_cast<cuopt_float_t>(iterations), static_cast<cuopt_float_t>(solved_by)});
}

extern "C" JNIEXPORT jdoubleArray JNICALL
Java_com_nvidia_cuopt_linearprogramming_NativeCuOpt_getMipStats(JNIEnv* env, jclass, jlong handle)
{
  cuopt_float_t presolve = 0;
  cuopt_float_t max_constraint = 0;
  cuopt_float_t max_int = 0;
  cuopt_float_t max_bound = 0;
  cuopt_int_t nodes = 0;
  cuopt_int_t simplex = 0;
  if (!check_status(env,
                    cuOptGetMIPSolverStats(
                      to_solution(handle), &presolve, &max_constraint, &max_int, &max_bound, &nodes, &simplex),
                    "cuOptGetMIPSolverStats")) {
    return nullptr;
  }
  return to_double_array(env, {presolve, max_constraint, max_int, max_bound, static_cast<cuopt_float_t>(nodes), static_cast<cuopt_float_t>(simplex)});
}

extern "C" JNIEXPORT jboolean JNICALL
Java_com_nvidia_cuopt_linearprogramming_NativeCuOpt_hasPdlpWarmStartData(JNIEnv* env,
                                                                         jclass,
                                                                         jlong handle)
{
  cuopt_int_t value = 0;
  check_status(env, cuOptHasPDLPWarmStartData(to_solution(handle), &value), "cuOptHasPDLPWarmStartData");
  return static_cast<jboolean>(value != 0);
}

extern "C" JNIEXPORT jdoubleArray JNICALL
Java_com_nvidia_cuopt_linearprogramming_NativeCuOpt_getPdlpWarmStartVector(JNIEnv* env,
                                                                           jclass,
                                                                           jlong handle,
                                                                           jint field_id)
{
  cuopt_int_t size = 0;
  if (!check_status(env,
                    cuOptGetPDLPWarmStartVectorSize(to_solution(handle), field_id, &size),
                    "cuOptGetPDLPWarmStartVectorSize")) {
    return nullptr;
  }
  std::vector<cuopt_float_t> values(static_cast<size_t>(size));
  if (size > 0 &&
      !check_status(env,
                    cuOptGetPDLPWarmStartVector(to_solution(handle), field_id, values.data()),
                    "cuOptGetPDLPWarmStartVector")) {
    return nullptr;
  }
  return to_double_array(env, values);
}

extern "C" JNIEXPORT jdouble JNICALL
Java_com_nvidia_cuopt_linearprogramming_NativeCuOpt_getPdlpWarmStartScalar(JNIEnv* env,
                                                                           jclass,
                                                                           jlong handle,
                                                                           jint field_id)
{
  cuopt_float_t value = 0;
  check_status(env,
               cuOptGetPDLPWarmStartScalar(to_solution(handle), field_id, &value),
               "cuOptGetPDLPWarmStartScalar");
  return value;
}

extern "C" JNIEXPORT jint JNICALL
Java_com_nvidia_cuopt_linearprogramming_NativeCuOpt_getPdlpWarmStartInteger(JNIEnv* env,
                                                                            jclass,
                                                                            jlong handle,
                                                                            jint field_id)
{
  cuopt_int_t value = 0;
  check_status(env,
               cuOptGetPDLPWarmStartInteger(to_solution(handle), field_id, &value),
               "cuOptGetPDLPWarmStartInteger");
  return value;
}
