/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <algorithm>
#include <cstring>
#include <exception>
#include <initializer_list>
#include <memory>
#include <mutex>
#include <span>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include <cuopt/mathematical_optimization/cuopt_c.h>
#include <cuopt/mathematical_optimization/io/parser.hpp>
#include <cuopt/mathematical_optimization/optimization_problem_utils.hpp>
#include <pdlp/cuopt_c_internal.hpp>

#include <jni.h>

// These bindings are hand-written JNI so they can run on Java 17, which is still a common
// default. Java 22 and later offer the Foreign Function & Memory API, which calls the C API
// directly and would remove this file; cuVS already takes that route. Raising the floor to 22
// is tracked in #1794.

namespace {

JavaVM* g_jvm = nullptr;

struct java_callback_context_t {
  jobject callback{nullptr};
  jobject user_data{nullptr};
  int num_variables{0};
  // First exception thrown by the Java callback, held as a global ref so solve() can rethrow it
  // from the calling thread. Callbacks may run on a solver-created thread, where a pending
  // exception would otherwise be discarded when that thread is detached.
  jthrowable failure{nullptr};
};

std::mutex g_callback_mutex;
std::unordered_map<jlong, std::vector<java_callback_context_t*>> g_callback_contexts;

// Problems created directly by this JNI module must be destroyed here as well.
// Passing these C++ objects to cuOptDestroyProblem in libcuopt.so crosses the
// shared-library boundary with a private wrapper type.
std::mutex g_jni_owned_problem_mutex;
std::unordered_set<jlong> g_jni_owned_problem_handles;

cuOptOptimizationProblem to_problem(jlong handle)
{
  return reinterpret_cast<cuOptOptimizationProblem>(handle);
}

cuopt::mathematical_optimization::problem_and_stream_view_t* to_problem_view(jlong handle)
{
  return reinterpret_cast<cuopt::mathematical_optimization::problem_and_stream_view_t*>(handle);
}

cuOptSolverSettings to_settings(jlong handle)
{
  return reinterpret_cast<cuOptSolverSettings>(handle);
}

cuOptSolution to_solution(jlong handle) { return reinterpret_cast<cuOptSolution>(handle); }

jlong from_handle(void* handle) { return reinterpret_cast<jlong>(handle); }

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
  jobjectArray result =
    env->NewObjectArray(static_cast<jsize>(values.size()), string_class, nullptr);
  for (jsize i = 0; i < static_cast<jsize>(values.size()); ++i) {
    env->SetObjectArrayElement(
      result, i, env->NewStringUTF(values[static_cast<size_t>(i)].c_str()));
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
  jclass cls = env->FindClass("com/nvidia/cuopt/mathematicaloptimization/CuOptException");
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
  throw_cuopt_exception(
    env, status, std::string(operation) + " failed with status " + std::to_string(status));
  return false;
}

cuopt::mathematical_optimization::lp_solution_interface_t<cuopt_int_t, cuopt_float_t>*
to_lp_solution(JNIEnv* env, jlong handle, const char* operation)
{
  auto* solution =
    reinterpret_cast<cuopt::mathematical_optimization::solution_and_stream_view_t*>(handle);
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
    throw_cuopt_exception(
      env, CUOPT_INVALID_ARGUMENT, std::string(operation) + " failed: " + e.what());
    return false;
  }
}

JNIEnv* get_callback_env(bool& detach)
{
  detach      = false;
  JNIEnv* env = nullptr;
  if (g_jvm->GetEnv(reinterpret_cast<void**>(&env), JNI_VERSION_1_8) == JNI_OK) { return env; }
  if (g_jvm->AttachCurrentThread(reinterpret_cast<void**>(&env), nullptr) == JNI_OK) {
    detach = true;
    return env;
  }
  return nullptr;
}

// Takes any pending exception off the callback thread and stores it on the context. Clearing it
// keeps the remaining JNI calls on this thread well defined; solve() rethrows it afterwards.
void capture_callback_exception(JNIEnv* env, java_callback_context_t* context)
{
  if (env->ExceptionCheck() == JNI_FALSE) { return; }
  jthrowable pending = env->ExceptionOccurred();
  env->ExceptionClear();
  if (pending == nullptr) { return; }
  if (context->failure == nullptr) {
    context->failure = static_cast<jthrowable>(env->NewGlobalRef(pending));
  }
  env->DeleteLocalRef(pending);
}

// Rethrows the first callback exception recorded for this settings handle, if any.
bool rethrow_callback_failure(JNIEnv* env, jlong settings_handle)
{
  jthrowable failure = nullptr;
  {
    std::lock_guard<std::mutex> lock(g_callback_mutex);
    auto it = g_callback_contexts.find(settings_handle);
    if (it == g_callback_contexts.end()) { return false; }
    for (auto* context : it->second) {
      if (context->failure != nullptr) {
        failure          = context->failure;
        context->failure = nullptr;
        break;
      }
    }
  }
  if (failure == nullptr) { return false; }
  env->Throw(failure);
  env->DeleteGlobalRef(failure);
  return true;
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
    if (context->failure != nullptr) { env->DeleteGlobalRef(context->failure); }
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
  if (cls != nullptr) {
    jmethodID method = env->GetMethodID(cls, "onSolution", "([DDDLjava/lang/Object;)V");
    if (method != nullptr) {
      std::vector<cuopt_float_t> values(solution, solution + context->num_variables);
      jdoubleArray solution_array = to_double_array(env, values);
      env->CallVoidMethod(context->callback,
                          method,
                          solution_array,
                          static_cast<jdouble>(*objective_value),
                          static_cast<jdouble>(*solution_bound),
                          context->user_data);
      capture_callback_exception(env, context);
      env->DeleteLocalRef(solution_array);
    }
    env->DeleteLocalRef(cls);
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
  if (cls != nullptr) {
    jmethodID method = env->GetMethodID(
      cls,
      "getSolution",
      "(DLjava/lang/Object;)Lcom/nvidia/cuopt/mathematicaloptimization/MIPCallbackSolution;");
    if (method != nullptr) {
      jobject callback_solution = env->CallObjectMethod(
        context->callback, method, static_cast<jdouble>(*solution_bound), context->user_data);
      capture_callback_exception(env, context);
      if (callback_solution != nullptr) {
        jclass result_cls = env->GetObjectClass(callback_solution);
        if (result_cls != nullptr) {
          jfieldID solution_field  = env->GetFieldID(result_cls, "solution", "[D");
          jfieldID objective_field = env->GetFieldID(result_cls, "objectiveValue", "D");
          if (solution_field != nullptr && objective_field != nullptr) {
            auto solution_array =
              static_cast<jdoubleArray>(env->GetObjectField(callback_solution, solution_field));
            const auto values = get_double_array(env, solution_array);
            if (values.size() == static_cast<size_t>(context->num_variables)) {
              std::memcpy(solution, values.data(), values.size() * sizeof(cuopt_float_t));
              *objective_value =
                static_cast<cuopt_float_t>(env->GetDoubleField(callback_solution, objective_field));
            } else {
              throw_illegal_state(env,
                                  "MIP set-solution callback returned " +
                                    std::to_string(values.size()) + " values for " +
                                    std::to_string(context->num_variables) + " variables");
              capture_callback_exception(env, context);
            }
            if (solution_array != nullptr) { env->DeleteLocalRef(solution_array); }
          }
          env->DeleteLocalRef(result_cls);
        }
        env->DeleteLocalRef(callback_solution);
      }
    }
    env->DeleteLocalRef(cls);
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
Java_com_nvidia_cuopt_mathematicaloptimization_NativeCuOpt_getFloatSize(JNIEnv*, jclass)
{
  return cuOptGetFloatSize();
}

extern "C" JNIEXPORT jlong JNICALL
Java_com_nvidia_cuopt_mathematicaloptimization_NativeCuOpt_readProblemWithFormat(
  JNIEnv* env, jclass, jstring path, jboolean fixed_mps_format)
{
  const auto filename = get_string(env, path);
  auto problem = std::make_unique<cuopt::mathematical_optimization::problem_and_stream_view_t>(
    cuopt::mathematical_optimization::get_memory_backend_type());
  try {
    auto data_model = cuopt::mathematical_optimization::io::read<int, double>(
      filename, static_cast<bool>(fixed_mps_format));
    cuopt::mathematical_optimization::populate_from_mps_data_model(problem->get_problem(),
                                                                   data_model);
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
Java_com_nvidia_cuopt_mathematicaloptimization_NativeCuOpt_createSolverSettings(JNIEnv* env, jclass)
{
  cuOptSolverSettings settings = nullptr;
  if (!check_status(env, cuOptCreateSolverSettings(&settings), "cuOptCreateSolverSettings")) {
    return 0;
  }
  return from_handle(settings);
}

extern "C" JNIEXPORT void JNICALL
Java_com_nvidia_cuopt_mathematicaloptimization_NativeCuOpt_destroySolverSettings(JNIEnv* env,
                                                                                 jclass,
                                                                                 jlong handle)
{
  if (handle == 0) { return; }
  cleanup_callback_contexts(env, handle);
  cuOptSolverSettings settings = to_settings(handle);
  cuOptDestroySolverSettings(&settings);
}

extern "C" JNIEXPORT void JNICALL
Java_com_nvidia_cuopt_mathematicaloptimization_NativeCuOpt_setSetting(
  JNIEnv* env, jclass, jlong handle, jstring name, jstring value)
{
  const auto parameter_name  = get_string(env, name);
  const auto parameter_value = get_string(env, value);
  check_status(
    env,
    cuOptSetParameter(to_settings(handle), parameter_name.c_str(), parameter_value.c_str()),
    "cuOptSetParameter");
}

extern "C" JNIEXPORT void JNICALL
Java_com_nvidia_cuopt_mathematicaloptimization_NativeCuOpt_setIntegerSetting(
  JNIEnv* env, jclass, jlong handle, jstring name, jint value)
{
  const auto parameter_name = get_string(env, name);
  check_status(env,
               cuOptSetIntegerParameter(to_settings(handle), parameter_name.c_str(), value),
               "cuOptSetIntegerParameter");
}

extern "C" JNIEXPORT void JNICALL
Java_com_nvidia_cuopt_mathematicaloptimization_NativeCuOpt_setFloatSetting(
  JNIEnv* env, jclass, jlong handle, jstring name, jdouble value)
{
  const auto parameter_name = get_string(env, name);
  check_status(env,
               cuOptSetFloatParameter(
                 to_settings(handle), parameter_name.c_str(), static_cast<cuopt_float_t>(value)),
               "cuOptSetFloatParameter");
}

extern "C" JNIEXPORT jstring JNICALL
Java_com_nvidia_cuopt_mathematicaloptimization_NativeCuOpt_getSetting(JNIEnv* env,
                                                                      jclass,
                                                                      jlong handle,
                                                                      jstring name)
{
  const auto parameter_name = get_string(env, name);
  char buffer[256]          = {};
  if (!check_status(
        env,
        cuOptGetParameter(to_settings(handle), parameter_name.c_str(), sizeof(buffer), buffer),
        "cuOptGetParameter")) {
    return nullptr;
  }
  return env->NewStringUTF(buffer);
}

extern "C" JNIEXPORT void JNICALL
Java_com_nvidia_cuopt_mathematicaloptimization_NativeCuOpt_addMIPStart(JNIEnv* env,
                                                                       jclass,
                                                                       jlong handle,
                                                                       jdoubleArray values)
{
  const auto data = get_double_array(env, values);
  check_status(
    env,
    cuOptAddMIPStart(to_settings(handle), data.data(), static_cast<cuopt_int_t>(data.size())),
    "cuOptAddMIPStart");
}

extern "C" JNIEXPORT void JNICALL
Java_com_nvidia_cuopt_mathematicaloptimization_NativeCuOpt_setInitialPrimalSolution(
  JNIEnv* env, jclass, jlong handle, jdoubleArray values)
{
  const auto data = get_double_array(env, values);
  check_status(env,
               cuOptSetInitialPrimalSolution(
                 to_settings(handle), data.data(), static_cast<cuopt_int_t>(data.size())),
               "cuOptSetInitialPrimalSolution");
}

extern "C" JNIEXPORT void JNICALL
Java_com_nvidia_cuopt_mathematicaloptimization_NativeCuOpt_setInitialDualSolution(
  JNIEnv* env, jclass, jlong handle, jdoubleArray values)
{
  const auto data = get_double_array(env, values);
  check_status(env,
               cuOptSetInitialDualSolution(
                 to_settings(handle), data.data(), static_cast<cuopt_int_t>(data.size())),
               "cuOptSetInitialDualSolution");
}

extern "C" JNIEXPORT void JNICALL
Java_com_nvidia_cuopt_mathematicaloptimization_NativeCuOpt_registerMIPGetSolutionCallback(
  JNIEnv* env, jclass, jlong handle, jobject callback, jobject user_data, jint num_variables)
{
  auto* context          = new java_callback_context_t;
  context->callback      = env->NewGlobalRef(callback);
  context->user_data     = user_data == nullptr ? nullptr : env->NewGlobalRef(user_data);
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
Java_com_nvidia_cuopt_mathematicaloptimization_NativeCuOpt_registerMIPSetSolutionCallback(
  JNIEnv* env, jclass, jlong handle, jobject callback, jobject user_data, jint num_variables)
{
  auto* context          = new java_callback_context_t;
  context->callback      = env->NewGlobalRef(callback);
  context->user_data     = user_data == nullptr ? nullptr : env->NewGlobalRef(user_data);
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

extern "C" JNIEXPORT jlong JNICALL
Java_com_nvidia_cuopt_mathematicaloptimization_NativeCuOpt_createProblem(
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
  jbyteArray constraint_sense,
  jdoubleArray rhs,
  jdoubleArray lower_bounds,
  jdoubleArray upper_bounds,
  jbyteArray variable_types)
{
  const auto obj                   = get_double_array(env, objective_coefficients);
  const auto rows                  = get_int_array(env, row_offsets);
  const auto cols                  = get_int_array(env, column_indices);
  const auto coeffs                = get_double_array(env, values);
  const auto senses                = get_byte_array(env, constraint_sense);
  const auto rhs_values            = get_double_array(env, rhs);
  const auto lbs                   = get_double_array(env, lower_bounds);
  const auto ubs                   = get_double_array(env, upper_bounds);
  const auto types                 = get_byte_array(env, variable_types);
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

extern "C" JNIEXPORT void JNICALL
Java_com_nvidia_cuopt_mathematicaloptimization_NativeCuOpt_writeProblem(JNIEnv* env,
                                                                        jclass,
                                                                        jlong handle,
                                                                        jstring path)
{
  const auto filename = get_string(env, path);
  check_status(env,
               cuOptWriteProblem(to_problem(handle), filename.c_str(), CUOPT_FILE_FORMAT_MPS),
               "cuOptWriteProblem");
}

extern "C" JNIEXPORT void JNICALL
Java_com_nvidia_cuopt_mathematicaloptimization_NativeCuOpt_destroyProblem(JNIEnv*,
                                                                          jclass,
                                                                          jlong handle)
{
  if (handle == 0) { return; }
  if (take_jni_owned_problem(handle)) {
    delete to_problem_view(handle);
    return;
  }
  cuOptOptimizationProblem problem = to_problem(handle);
  cuOptDestroyProblem(&problem);
}

extern "C" JNIEXPORT void JNICALL
Java_com_nvidia_cuopt_mathematicaloptimization_NativeCuOpt_setQuadraticObjective(
  JNIEnv* env, jclass, jlong handle, jintArray rows, jintArray cols, jdoubleArray coeffs)
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
Java_com_nvidia_cuopt_mathematicaloptimization_NativeCuOpt_addQuadraticConstraint(
  JNIEnv* env,
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
  const auto row_data  = get_int_array(env, rows);
  const auto col_data  = get_int_array(env, cols);
  const auto val_data  = get_double_array(env, coeffs);
  const auto lin_idx   = get_int_array(env, linear_indices);
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
Java_com_nvidia_cuopt_mathematicaloptimization_NativeCuOpt_getNumVariables(JNIEnv* env,
                                                                           jclass,
                                                                           jlong handle)
{
  cuopt_int_t value = 0;
  check_status(env, cuOptGetNumVariables(to_problem(handle), &value), "cuOptGetNumVariables");
  return value;
}

extern "C" JNIEXPORT jint JNICALL
Java_com_nvidia_cuopt_mathematicaloptimization_NativeCuOpt_getNumConstraints(JNIEnv* env,
                                                                             jclass,
                                                                             jlong handle)
{
  cuopt_int_t value = 0;
  check_status(env, cuOptGetNumConstraints(to_problem(handle), &value), "cuOptGetNumConstraints");
  return value;
}

extern "C" JNIEXPORT jint JNICALL
Java_com_nvidia_cuopt_mathematicaloptimization_NativeCuOpt_getNumNonZeros(JNIEnv* env,
                                                                          jclass,
                                                                          jlong handle)
{
  cuopt_int_t value = 0;
  check_status(env, cuOptGetNumNonZeros(to_problem(handle), &value), "cuOptGetNumNonZeros");
  return value;
}

extern "C" JNIEXPORT jint JNICALL
Java_com_nvidia_cuopt_mathematicaloptimization_NativeCuOpt_getObjectiveSense(JNIEnv* env,
                                                                             jclass,
                                                                             jlong handle)
{
  cuopt_int_t value = 0;
  check_status(env, cuOptGetObjectiveSense(to_problem(handle), &value), "cuOptGetObjectiveSense");
  return value;
}

extern "C" JNIEXPORT jdouble JNICALL
Java_com_nvidia_cuopt_mathematicaloptimization_NativeCuOpt_getObjectiveOffset(JNIEnv* env,
                                                                              jclass,
                                                                              jlong handle)
{
  cuopt_float_t value = 0;
  check_status(env, cuOptGetObjectiveOffset(to_problem(handle), &value), "cuOptGetObjectiveOffset");
  return value;
}

extern "C" JNIEXPORT jdoubleArray JNICALL
Java_com_nvidia_cuopt_mathematicaloptimization_NativeCuOpt_getObjectiveCoefficients(JNIEnv* env,
                                                                                    jclass,
                                                                                    jlong handle)
{
  const int n = Java_com_nvidia_cuopt_mathematicaloptimization_NativeCuOpt_getNumVariables(
    env, nullptr, handle);
  std::vector<cuopt_float_t> values(static_cast<size_t>(n));
  if (!check_status(env,
                    cuOptGetObjectiveCoefficients(to_problem(handle), values.data()),
                    "cuOptGetObjectiveCoefficients")) {
    return nullptr;
  }
  return to_double_array(env, values);
}

extern "C" JNIEXPORT jobjectArray JNICALL
Java_com_nvidia_cuopt_mathematicaloptimization_NativeCuOpt_getConstraintMatrix(JNIEnv* env,
                                                                               jclass,
                                                                               jlong handle)
{
  const int rows_size =
    Java_com_nvidia_cuopt_mathematicaloptimization_NativeCuOpt_getNumConstraints(
      env, nullptr, handle) +
    1;
  const int nnz =
    Java_com_nvidia_cuopt_mathematicaloptimization_NativeCuOpt_getNumNonZeros(env, nullptr, handle);
  std::vector<cuopt_int_t> rows(static_cast<size_t>(rows_size));
  std::vector<cuopt_int_t> cols(static_cast<size_t>(nnz));
  std::vector<cuopt_float_t> values(static_cast<size_t>(nnz));
  if (!check_status(
        env,
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
Java_com_nvidia_cuopt_mathematicaloptimization_NativeCuOpt_setVariableNames(JNIEnv* env,
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
Java_com_nvidia_cuopt_mathematicaloptimization_NativeCuOpt_setRowNames(JNIEnv* env,
                                                                       jclass,
                                                                       jlong handle,
                                                                       jobjectArray values)
{
  const auto h_values = get_string_array(env, values);
  run_problem_operation(
    env, "setRowNames", [&] { to_problem_view(handle)->get_problem()->set_row_names(h_values); });
}

extern "C" JNIEXPORT void JNICALL
Java_com_nvidia_cuopt_mathematicaloptimization_NativeCuOpt_setProblemName(JNIEnv* env,
                                                                          jclass,
                                                                          jlong handle,
                                                                          jstring value)
{
  const auto name = get_string(env, value);
  run_problem_operation(
    env, "setProblemName", [&] { to_problem_view(handle)->get_problem()->set_problem_name(name); });
}

extern "C" JNIEXPORT jdoubleArray JNICALL
Java_com_nvidia_cuopt_mathematicaloptimization_NativeCuOpt_getQuadraticObjectiveValues(JNIEnv* env,
                                                                                       jclass,
                                                                                       jlong handle)
{
  return to_double_array(env,
                         to_problem_view(handle)->get_problem()->get_quadratic_objective_values());
}

extern "C" JNIEXPORT jintArray JNICALL
Java_com_nvidia_cuopt_mathematicaloptimization_NativeCuOpt_getQuadraticObjectiveIndices(
  JNIEnv* env, jclass, jlong handle)
{
  return to_int_array(env,
                      to_problem_view(handle)->get_problem()->get_quadratic_objective_indices());
}

extern "C" JNIEXPORT jintArray JNICALL
Java_com_nvidia_cuopt_mathematicaloptimization_NativeCuOpt_getQuadraticObjectiveOffsets(
  JNIEnv* env, jclass, jlong handle)
{
  return to_int_array(env,
                      to_problem_view(handle)->get_problem()->get_quadratic_objective_offsets());
}

extern "C" JNIEXPORT jobjectArray JNICALL
Java_com_nvidia_cuopt_mathematicaloptimization_NativeCuOpt_getVariableNames(JNIEnv* env,
                                                                            jclass,
                                                                            jlong handle)
{
  return to_string_array(env, to_problem_view(handle)->get_problem()->get_variable_names());
}

extern "C" JNIEXPORT jobjectArray JNICALL
Java_com_nvidia_cuopt_mathematicaloptimization_NativeCuOpt_getRowNames(JNIEnv* env,
                                                                       jclass,
                                                                       jlong handle)
{
  return to_string_array(env, to_problem_view(handle)->get_problem()->get_row_names());
}

extern "C" JNIEXPORT jstring JNICALL
Java_com_nvidia_cuopt_mathematicaloptimization_NativeCuOpt_getProblemName(JNIEnv* env,
                                                                          jclass,
                                                                          jlong handle)
{
  return env->NewStringUTF(to_problem_view(handle)->get_problem()->get_problem_name().c_str());
}

extern "C" JNIEXPORT jint JNICALL
Java_com_nvidia_cuopt_mathematicaloptimization_NativeCuOpt_getProblemCategory(JNIEnv* env,
                                                                              jclass,
                                                                              jlong handle)
{
  jint category = 0;
  if (!run_problem_operation(env, "getProblemCategory", [&] {
        category =
          static_cast<jint>(to_problem_view(handle)->get_problem()->get_problem_category());
      })) {
    return 0;
  }
  return category;
}

extern "C" JNIEXPORT jobjectArray JNICALL
Java_com_nvidia_cuopt_mathematicaloptimization_NativeCuOpt_getQuadraticConstraints(JNIEnv* env,
                                                                                   jclass,
                                                                                   jlong handle)
{
  const auto& constraints = to_problem_view(handle)->get_problem()->get_quadratic_constraints();
  jclass object_class     = env->FindClass("java/lang/Object");
  jobjectArray result =
    env->NewObjectArray(static_cast<jsize>(constraints.size()), object_class, nullptr);
  for (jsize i = 0; i < static_cast<jsize>(constraints.size()); ++i) {
    const auto& constraint = constraints[static_cast<size_t>(i)];
    jobjectArray entry     = env->NewObjectArray(9, object_class, nullptr);
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

#define DEFINE_DOUBLE_PROBLEM_GETTER(JAVA_NAME, C_NAME, COUNT_EXPR)               \
  extern "C" JNIEXPORT jdoubleArray JNICALL                                       \
  Java_com_nvidia_cuopt_mathematicaloptimization_NativeCuOpt_##JAVA_NAME(         \
    JNIEnv* env, jclass, jlong handle)                                            \
  {                                                                               \
    const int count = (COUNT_EXPR);                                               \
    std::vector<cuopt_float_t> values(static_cast<size_t>(count));                \
    if (!check_status(env, C_NAME(to_problem(handle), values.data()), #C_NAME)) { \
      return nullptr;                                                             \
    }                                                                             \
    return to_double_array(env, values);                                          \
  }

#define DEFINE_BYTE_PROBLEM_GETTER(JAVA_NAME, C_NAME, COUNT_EXPR)                 \
  extern "C" JNIEXPORT jbyteArray JNICALL                                         \
  Java_com_nvidia_cuopt_mathematicaloptimization_NativeCuOpt_##JAVA_NAME(         \
    JNIEnv* env, jclass, jlong handle)                                            \
  {                                                                               \
    const int count = (COUNT_EXPR);                                               \
    std::vector<char> values(static_cast<size_t>(count));                         \
    if (!check_status(env, C_NAME(to_problem(handle), values.data()), #C_NAME)) { \
      return nullptr;                                                             \
    }                                                                             \
    return to_byte_array(env, values);                                            \
  }

DEFINE_DOUBLE_PROBLEM_GETTER(
  getConstraintRHS,
  cuOptGetConstraintRightHandSide,
  Java_com_nvidia_cuopt_mathematicaloptimization_NativeCuOpt_getNumConstraints(env,
                                                                               nullptr,
                                                                               handle))

DEFINE_DOUBLE_PROBLEM_GETTER(
  getConstraintLowerBounds,
  cuOptGetConstraintLowerBounds,
  Java_com_nvidia_cuopt_mathematicaloptimization_NativeCuOpt_getNumConstraints(env,
                                                                               nullptr,
                                                                               handle))

DEFINE_DOUBLE_PROBLEM_GETTER(
  getConstraintUpperBounds,
  cuOptGetConstraintUpperBounds,
  Java_com_nvidia_cuopt_mathematicaloptimization_NativeCuOpt_getNumConstraints(env,
                                                                               nullptr,
                                                                               handle))

DEFINE_DOUBLE_PROBLEM_GETTER(
  getVariableLowerBounds,
  cuOptGetVariableLowerBounds,
  Java_com_nvidia_cuopt_mathematicaloptimization_NativeCuOpt_getNumVariables(env, nullptr, handle))

DEFINE_DOUBLE_PROBLEM_GETTER(
  getVariableUpperBounds,
  cuOptGetVariableUpperBounds,
  Java_com_nvidia_cuopt_mathematicaloptimization_NativeCuOpt_getNumVariables(env, nullptr, handle))

DEFINE_BYTE_PROBLEM_GETTER(
  getConstraintSense,
  cuOptGetConstraintSense,
  Java_com_nvidia_cuopt_mathematicaloptimization_NativeCuOpt_getNumConstraints(env,
                                                                               nullptr,
                                                                               handle))
DEFINE_BYTE_PROBLEM_GETTER(
  getVariableTypes,
  cuOptGetVariableTypes,
  Java_com_nvidia_cuopt_mathematicaloptimization_NativeCuOpt_getNumVariables(env, nullptr, handle))

#undef DEFINE_DOUBLE_PROBLEM_GETTER
#undef DEFINE_BYTE_PROBLEM_GETTER

extern "C" JNIEXPORT jlong JNICALL Java_com_nvidia_cuopt_mathematicaloptimization_NativeCuOpt_solve(
  JNIEnv* env, jclass, jlong problem_handle, jlong settings_handle)
{
  cuOptSolution solution = nullptr;
  const cuopt_int_t status =
    cuOptSolve(to_problem(problem_handle), to_settings(settings_handle), &solution);

  // A callback that threw takes precedence over the solver's own status: the solve ran against a
  // model the caller did not get to finish describing, so its result is not meaningful.
  if (rethrow_callback_failure(env, settings_handle)) {
    if (solution != nullptr) { cuOptDestroySolution(&solution); }
    return 0;
  }

  if (!check_status(env, status, "cuOptSolve")) { return 0; }
  return from_handle(solution);
}

extern "C" JNIEXPORT void JNICALL
Java_com_nvidia_cuopt_mathematicaloptimization_NativeCuOpt_destroySolution(JNIEnv*,
                                                                           jclass,
                                                                           jlong handle)
{
  if (handle == 0) { return; }
  cuOptSolution solution = to_solution(handle);
  cuOptDestroySolution(&solution);
}

extern "C" JNIEXPORT jint JNICALL
Java_com_nvidia_cuopt_mathematicaloptimization_NativeCuOpt_getTerminationStatus(JNIEnv* env,
                                                                                jclass,
                                                                                jlong handle)
{
  cuopt_int_t value = 0;
  check_status(
    env, cuOptGetTerminationStatus(to_solution(handle), &value), "cuOptGetTerminationStatus");
  return value;
}

extern "C" JNIEXPORT jint JNICALL
Java_com_nvidia_cuopt_mathematicaloptimization_NativeCuOpt_getErrorStatus(JNIEnv* env,
                                                                          jclass,
                                                                          jlong handle)
{
  cuopt_int_t value = 0;
  check_status(env, cuOptGetErrorStatus(to_solution(handle), &value), "cuOptGetErrorStatus");
  return value;
}

extern "C" JNIEXPORT jstring JNICALL
Java_com_nvidia_cuopt_mathematicaloptimization_NativeCuOpt_getErrorString(JNIEnv* env,
                                                                          jclass,
                                                                          jlong handle)
{
  char buffer[1024] = {};
  if (!check_status(env,
                    cuOptGetErrorString(to_solution(handle), buffer, sizeof(buffer)),
                    "cuOptGetErrorString")) {
    return nullptr;
  }
  return env->NewStringUTF(buffer);
}

extern "C" JNIEXPORT jdoubleArray JNICALL
Java_com_nvidia_cuopt_mathematicaloptimization_NativeCuOpt_getPrimalSolution(JNIEnv* env,
                                                                             jclass,
                                                                             jlong handle,
                                                                             jint size)
{
  std::vector<cuopt_float_t> values(static_cast<size_t>(size));
  const cuopt_int_t status = cuOptGetPrimalSolution(to_solution(handle), values.data());
  // The solution carries no primal values when the solve did not produce any, an infeasible
  // problem for instance. This layer always passes a live handle and a correctly sized buffer,
  // so CUOPT_INVALID_ARGUMENT can only mean the values are absent. Report that as an empty
  // array, which is how the Java side already distinguishes "unavailable" from a real result.
  if (status == CUOPT_INVALID_ARGUMENT) { return to_double_array(env, {}); }
  if (!check_status(env, status, "cuOptGetPrimalSolution")) { return nullptr; }
  return to_double_array(env, values);
}

extern "C" JNIEXPORT jint JNICALL
Java_com_nvidia_cuopt_mathematicaloptimization_NativeCuOpt_getDualSolutionSize(JNIEnv* env,
                                                                               jclass,
                                                                               jlong handle)
{
  auto* solution = to_lp_solution(env, handle, "getDualSolution");
  if (solution == nullptr) { return 0; }
  return solution->get_dual_solution_size();
}

extern "C" JNIEXPORT jdoubleArray JNICALL
Java_com_nvidia_cuopt_mathematicaloptimization_NativeCuOpt_getDualSolution(JNIEnv* env,
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
Java_com_nvidia_cuopt_mathematicaloptimization_NativeCuOpt_getReducedCosts(JNIEnv* env,
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
Java_com_nvidia_cuopt_mathematicaloptimization_NativeCuOpt_getObjectiveValue(JNIEnv* env,
                                                                             jclass,
                                                                             jlong handle)
{
  cuopt_float_t value = 0;
  check_status(env, cuOptGetObjectiveValue(to_solution(handle), &value), "cuOptGetObjectiveValue");
  return value;
}

extern "C" JNIEXPORT jdouble JNICALL
Java_com_nvidia_cuopt_mathematicaloptimization_NativeCuOpt_getDualObjectiveValue(JNIEnv* env,
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
Java_com_nvidia_cuopt_mathematicaloptimization_NativeCuOpt_getSolveTime(JNIEnv* env,
                                                                        jclass,
                                                                        jlong handle)
{
  cuopt_float_t value = 0;
  check_status(env, cuOptGetSolveTime(to_solution(handle), &value), "cuOptGetSolveTime");
  return value;
}

extern "C" JNIEXPORT jdouble JNICALL
Java_com_nvidia_cuopt_mathematicaloptimization_NativeCuOpt_getMIPGap(JNIEnv* env,
                                                                     jclass,
                                                                     jlong handle)
{
  cuopt_float_t value = 0;
  check_status(env, cuOptGetMIPGap(to_solution(handle), &value), "cuOptGetMIPGap");
  return value;
}

extern "C" JNIEXPORT jdouble JNICALL
Java_com_nvidia_cuopt_mathematicaloptimization_NativeCuOpt_getSolutionBound(JNIEnv* env,
                                                                            jclass,
                                                                            jlong handle)
{
  cuopt_float_t value = 0;
  check_status(env, cuOptGetSolutionBound(to_solution(handle), &value), "cuOptGetSolutionBound");
  return value;
}

extern "C" JNIEXPORT jint JNICALL
Java_com_nvidia_cuopt_mathematicaloptimization_NativeCuOpt_getSolutionIntAttribute(JNIEnv* env,
                                                                                   jclass,
                                                                                   jlong handle,
                                                                                   jint attribute)
{
  cuopt_int_t value = 0;
  check_status(env,
               cuOptGetSolutionIntAttribute(to_solution(handle), attribute, &value),
               "cuOptGetSolutionIntAttribute");
  return value;
}

extern "C" JNIEXPORT jdouble JNICALL
Java_com_nvidia_cuopt_mathematicaloptimization_NativeCuOpt_getSolutionFloatAttribute(JNIEnv* env,
                                                                                     jclass,
                                                                                     jlong handle,
                                                                                     jint attribute)
{
  cuopt_float_t value = 0;
  check_status(env,
               cuOptGetSolutionFloatAttribute(to_solution(handle), attribute, &value),
               "cuOptGetSolutionFloatAttribute");
  return value;
}
