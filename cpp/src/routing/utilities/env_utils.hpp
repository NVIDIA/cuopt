/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#pragma once

#include <routing/hyper_params.hpp>

#include <routing/utilities/constants.hpp>

#include <string>

namespace cuopt {
namespace routing {
namespace detail {

// string overload
void inline set_if_env_set(std::string& val, const char* env_var)
{
  const char* str = std::getenv(env_var);
  if (str != NULL) { val = std::string(str); }
}

// float overload
void inline set_if_env_set(float& val, const char* env_var)
{
  const char* str = std::getenv(env_var);
  if (str != NULL) { val = std::stoi(str) / 1000.f; }
}

// i_t overload
void inline set_if_env_set(int& val, const char* env_var)
{
  const char* str = std::getenv(env_var);
  if (str != NULL) { val = std::stoi(str); }
}

// bool overload
void inline set_if_env_set(bool& val, const char* env_var)
{
  const char* str = std::getenv(env_var);
  if (str != NULL) { val = bool(std::stoi(str)); }
}

hyper_params_t inline get_hyper_parameters_from_env()
{
  hyper_params_t hyper_params{};
  return hyper_params;
}

}  // namespace detail
}  // namespace routing
}  // namespace cuopt
