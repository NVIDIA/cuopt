/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#include <cstddef>
#include <string>
#include <string_view>
#include <type_traits>

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

  virtual base_solution_callback_type get_type() const = 0;

 protected:
  bool isFloat       = true;
  size_t n_variables = 0;
};

class get_solution_callback_t : public base_solution_callback_t {
 public:
  virtual void get_solution(void* data, void* objective_value) = 0;
  base_solution_callback_type get_type() const override
  {
    return base_solution_callback_type::GET_SOLUTION;
  }
};

class set_solution_callback_t : public base_solution_callback_t {
 public:
  virtual void set_solution(void* data, void* objective_value) = 0;
  base_solution_callback_type get_type() const override
  {
    return base_solution_callback_type::SET_SOLUTION;
  }
};

}  // namespace internals

namespace linear_programming {

class base_solution_t {
 public:
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

}  // namespace linear_programming
}  // namespace cuopt
