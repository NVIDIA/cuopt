/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#pragma once

#include <cstddef>
#include <string>
#include <string_view>
#include <type_traits>

#include <cuopt/linear_programming/constants.h>
#include <cuopt/linear_programming/cuopt_c.h>

namespace cuopt {
namespace internals {

class Callback {
 public:
  virtual ~Callback() {}
};

enum class mip_solution_origin_t : uint32_t {
  UNKNOWN                 = CUOPT_MIP_SOLUTION_ORIGIN_UNKNOWN,
  BRANCH_AND_BOUND_NODE   = CUOPT_MIP_SOLUTION_ORIGIN_BRANCH_AND_BOUND,
  BRANCH_AND_BOUND_DIVING = CUOPT_MIP_SOLUTION_ORIGIN_BRANCH_AND_BOUND_DIVING,
  FEASIBILITY_JUMP        = CUOPT_MIP_SOLUTION_ORIGIN_FEASIBILITY_JUMP,
  CPU_FEASIBILITY_JUMP    = CUOPT_MIP_SOLUTION_ORIGIN_CPU_FEASIBILITY_JUMP,
  LOCAL_SEARCH            = CUOPT_MIP_SOLUTION_ORIGIN_LOCAL_SEARCH,
  LP_ROUNDING             = CUOPT_MIP_SOLUTION_ORIGIN_LP_ROUNDING,
  RECOMBINATION           = CUOPT_MIP_SOLUTION_ORIGIN_RECOMBINATION,
  SUB_MIP                 = CUOPT_MIP_SOLUTION_ORIGIN_SUB_MIP,
  USER_INITIAL            = CUOPT_MIP_SOLUTION_ORIGIN_USER_INITIAL,
  USER_INJECTED           = CUOPT_MIP_SOLUTION_ORIGIN_USER_INJECTED,
  RINS                    = CUOPT_MIP_SOLUTION_ORIGIN_RINS,
  PRESOLVE                = CUOPT_MIP_SOLUTION_ORIGIN_PRESOLVE,
};

constexpr const char* mip_solution_origin_to_string(mip_solution_origin_t origin)
{
  switch (origin) {
    case mip_solution_origin_t::UNKNOWN: return "unknown";
    case mip_solution_origin_t::BRANCH_AND_BOUND_NODE: return "branch_and_bound_node";
    case mip_solution_origin_t::BRANCH_AND_BOUND_DIVING: return "branch_and_bound_diving";
    case mip_solution_origin_t::FEASIBILITY_JUMP: return "feasibility_jump";
    case mip_solution_origin_t::CPU_FEASIBILITY_JUMP: return "cpu_feasibility_jump";
    case mip_solution_origin_t::LOCAL_SEARCH: return "local_search";
    case mip_solution_origin_t::LP_ROUNDING: return "lp_rounding";
    case mip_solution_origin_t::RECOMBINATION: return "recombination";
    case mip_solution_origin_t::SUB_MIP: return "sub_mip";
    case mip_solution_origin_t::USER_INITIAL: return "user_initial";
    case mip_solution_origin_t::USER_INJECTED: return "user_injected";
    case mip_solution_origin_t::RINS: return "rins";
    case mip_solution_origin_t::PRESOLVE:
      return "presolve";
      // no default to trigger compiler -Werror
  }
  return "unknown";
}

using mip_solution_callback_info_t = cuOptMIPSolutionCallbackInfo;

// get_solution_ext was added to support passing additional information to the get_solution callback
// without inducing a breaking ABI change
enum class base_solution_callback_type { GET_SOLUTION, GET_SOLUTION_EXT, SET_SOLUTION };

class base_solution_callback_t : public Callback {
 public:
  template <typename T>
  void setup(size_t n_variables_)
  {
    this->isFloat     = std::is_same<T, float>::value;
    this->n_variables = n_variables_;
  }

  void set_user_data(void* input_user_data) { user_data = input_user_data; }
  void* get_user_data() const { return user_data; }

  virtual base_solution_callback_type get_type() const = 0;

 protected:
  bool isFloat       = true;
  size_t n_variables = 0;
  void* user_data    = nullptr;
};

class get_solution_callback_t : public base_solution_callback_t {
 public:
  virtual void get_solution(void* data,
                            void* objective_value,
                            void* solution_bound,
                            void* user_data) = 0;
  base_solution_callback_type get_type() const override
  {
    return base_solution_callback_type::GET_SOLUTION;
  }
};

class get_solution_callback_ext_t : public base_solution_callback_t {
 public:
  virtual void get_solution(void* data,
                            void* objective_value,
                            void* solution_bound,
                            const mip_solution_callback_info_t* callback_info,
                            void* user_data) = 0;
  base_solution_callback_type get_type() const override
  {
    return base_solution_callback_type::GET_SOLUTION_EXT;
  }
};

class set_solution_callback_t : public base_solution_callback_t {
 public:
  virtual void set_solution(void* data,
                            void* objective_value,
                            void* solution_bound,
                            void* user_data) = 0;
  base_solution_callback_type get_type() const override
  {
    return base_solution_callback_type::SET_SOLUTION;
  }
};

}  // namespace internals

namespace linear_programming {

class base_solution_t {
 public:
  virtual ~base_solution_t()  = default;
  virtual bool is_mip() const = 0;
};

template <typename T>
struct parameter_info_t {
  parameter_info_t(
    std::string_view param_name, T* value, T min, T max, T def, const char* description = "")
    : param_name(param_name),
      value_ptr(value),
      min_value(min),
      max_value(max),
      default_value(def),
      description(description)
  {
  }
  std::string param_name;
  T* value_ptr;
  T min_value;
  T max_value;
  T default_value;
  const char* description;
};

template <>
struct parameter_info_t<bool> {
  parameter_info_t(std::string_view name, bool* value, bool def, const char* description = "")
    : param_name(name), value_ptr(value), default_value(def), description(description)
  {
  }
  std::string param_name;
  bool* value_ptr;
  bool default_value;
  const char* description;
};

template <>
struct parameter_info_t<std::string> {
  parameter_info_t(std::string_view name,
                   std::string* value,
                   std::string def,
                   const char* description = "")
    : param_name(name), value_ptr(value), default_value(def), description(description)
  {
  }
  std::string param_name;
  std::string* value_ptr;
  std::string default_value;
  const char* description;
};

/**
 * @brief Enum representing the different presolvers that can be used to solve the
 * linear programming problem.
 *
 * Default: Use the default presolver.
 * None: No presolver.
 * Papilo: Use the Papilo presolver.
 * PSLP: Use the PSLP presolver.
 *
 * @note Default presolver is None.
 */
enum presolver_t : int {
  Default = CUOPT_PRESOLVE_DEFAULT,
  None    = CUOPT_PRESOLVE_OFF,
  Papilo  = CUOPT_PRESOLVE_PAPILO,
  PSLP    = CUOPT_PRESOLVE_PSLP
};

}  // namespace linear_programming
}  // namespace cuopt
