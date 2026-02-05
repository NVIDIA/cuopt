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
namespace cuopt {
namespace internals {

class Callback {
 public:
  virtual ~Callback() {}
};

enum class base_solution_callback_type { GET_SOLUTION, SET_SOLUTION };

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
  parameter_info_t(std::string_view param_name, T* value, T min, T max, T def)
    : param_name(param_name), value_ptr(value), min_value(min), max_value(max), default_value(def)
  {
  }
  std::string param_name;
  T* value_ptr;
  T min_value;
  T max_value;
  T default_value;
};

template <>
struct parameter_info_t<bool> {
  parameter_info_t(std::string_view name, bool* value, bool def)
    : param_name(name), value_ptr(value), default_value(def)
  {
  }
  std::string param_name;
  bool* value_ptr;
  bool default_value;
};

template <>
struct parameter_info_t<std::string> {
  parameter_info_t(std::string_view name, std::string* value, std::string def)
    : param_name(name), value_ptr(value), default_value(def)
  {
  }
  std::string param_name;
  std::string* value_ptr;
  std::string default_value;
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
