/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#include <cuopt/error.hpp>
#include <cuopt/export.hpp>
#include <cuopt/mathematical_optimization/solver_settings.hpp>
#include <mip_heuristics/mip_constants.hpp>
#include <utilities/logger.hpp>

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <sstream>

namespace cuopt::mathematical_optimization {

template <typename i_t, typename f_t>
void solver_settings_t<i_t, f_t>::set_parameter_from_string(const std::string& name,
                                                            const std::string& value)
{
  registry_.set_parameter_from_string(name, value);
}

template <typename i_t, typename f_t>
template <typename T>
void solver_settings_t<i_t, f_t>::set_parameter(const std::string& name, T value)
{
  registry_.set_parameter(name, value);
}

template <typename i_t, typename f_t>
template <typename T>
T solver_settings_t<i_t, f_t>::get_parameter(const std::string& name) const
{
  return registry_.template get_parameter<T>(name);
}

template <typename i_t, typename f_t>
std::string solver_settings_t<i_t, f_t>::get_parameter_as_string(const std::string& name) const
{
  return registry_.get_parameter_as_string(name);
}

template <typename i_t, typename f_t>
void solver_settings_t<i_t, f_t>::set_mip_callback(internals::base_solution_callback_t* callback,
                                                   void* user_data)
{
  mip_settings.set_mip_callback(callback, user_data);
}

template <typename i_t, typename f_t>
const std::vector<internals::base_solution_callback_t*>
solver_settings_t<i_t, f_t>::get_mip_callbacks() const
{
  return mip_settings.get_mip_callbacks();
}

template <typename i_t, typename f_t>
pdlp_solver_settings_t<i_t, f_t>& solver_settings_t<i_t, f_t>::get_pdlp_settings()
{
  return pdlp_settings;
}

template <typename i_t, typename f_t>
mip_solver_settings_t<i_t, f_t>& solver_settings_t<i_t, f_t>::get_mip_settings()
{
  return mip_settings;
}

template <typename i_t, typename f_t>
const pdlp_warm_start_data_view_t<i_t, f_t>&
solver_settings_t<i_t, f_t>::get_pdlp_warm_start_data_view() const noexcept
{
  return pdlp_settings.get_pdlp_warm_start_data_view();
}

template <typename i_t, typename f_t>
const std::vector<parameter_info_t<f_t>>& solver_settings_t<i_t, f_t>::get_float_parameters() const
{
  return registry_.get_float_parameters();
}

template <typename i_t, typename f_t>
const std::vector<parameter_info_t<i_t>>& solver_settings_t<i_t, f_t>::get_int_parameters() const
{
  return registry_.get_int_parameters();
}

template <typename i_t, typename f_t>
const std::vector<parameter_info_t<bool>>& solver_settings_t<i_t, f_t>::get_bool_parameters() const
{
  return registry_.get_bool_parameters();
}

template <typename i_t, typename f_t>
const std::vector<parameter_info_t<std::string>>&
solver_settings_t<i_t, f_t>::get_string_parameters() const
{
  return registry_.get_string_parameters();
}

template <typename i_t, typename f_t>
const std::vector<std::string> solver_settings_t<i_t, f_t>::get_parameter_names() const
{
  return registry_.get_parameter_names();
}

template <typename i_t, typename f_t>
void solver_settings_t<i_t, f_t>::load_parameters_from_file(const std::string& path)
{
  registry_.load_parameters_from_file(path);
}

template <typename i_t, typename f_t>
bool solver_settings_t<i_t, f_t>::dump_parameters_to_file(const std::string& path,
                                                          bool hyperparameters_only) const
{
  return registry_.dump_parameters_to_file(path, hyperparameters_only);
}

// NOTE: deliberately no `template class solver_settings_t<...>` here.
//
// That would instantiate every member, including the implicitly-defined constructor and
// copy constructor. Those construct a pdlp_solver_settings_t, which holds a
// pdlp_warm_start_data_t by value, whose default ctor lives in a CUDA translation unit --
// so the whole class instantiation drags a CUDA dependency into this CUDA-free library and
// leaves libcuopt_client.so with an undefined symbol. Members are therefore instantiated
// individually below; libcuopt emits the constructors via its own `template class`
// in solver_settings.cu.

#if MIP_INSTANTIATE_FLOAT
template CUOPT_EXPORT void solver_settings_t<int, float>::set_parameter_from_string(
  const std::string&, const std::string&);
template CUOPT_EXPORT std::string solver_settings_t<int, float>::get_parameter_as_string(
  const std::string&) const;
template CUOPT_EXPORT void solver_settings_t<int, float>::set_mip_callback(
  internals::base_solution_callback_t*, void*);
template CUOPT_EXPORT const std::vector<internals::base_solution_callback_t*>
solver_settings_t<int, float>::get_mip_callbacks() const;
template CUOPT_EXPORT pdlp_solver_settings_t<int, float>&
solver_settings_t<int, float>::get_pdlp_settings();
template CUOPT_EXPORT mip_solver_settings_t<int, float>&
solver_settings_t<int, float>::get_mip_settings();
template CUOPT_EXPORT const std::vector<parameter_info_t<float>>&
solver_settings_t<int, float>::get_float_parameters() const;
template CUOPT_EXPORT const std::vector<parameter_info_t<int>>&
solver_settings_t<int, float>::get_int_parameters() const;
template CUOPT_EXPORT const std::vector<parameter_info_t<bool>>&
solver_settings_t<int, float>::get_bool_parameters() const;
template CUOPT_EXPORT const std::vector<std::string>
solver_settings_t<int, float>::get_parameter_names() const;
template CUOPT_EXPORT const std::vector<parameter_info_t<std::string>>&
solver_settings_t<int, float>::get_string_parameters() const;
template CUOPT_EXPORT const pdlp_warm_start_data_view_t<int, float>&
solver_settings_t<int, float>::get_pdlp_warm_start_data_view() const noexcept;
template CUOPT_EXPORT void solver_settings_t<int, float>::load_parameters_from_file(
  const std::string&);
template CUOPT_EXPORT bool solver_settings_t<int, float>::dump_parameters_to_file(
  const std::string&, bool) const;
template CUOPT_EXPORT void solver_settings_t<int, float>::set_parameter(const std::string& name,
                                                                        int value);
template CUOPT_EXPORT void solver_settings_t<int, float>::set_parameter(const std::string& name,
                                                                        float value);
template CUOPT_EXPORT void solver_settings_t<int, float>::set_parameter(const std::string& name,
                                                                        bool value);
template CUOPT_EXPORT int solver_settings_t<int, float>::get_parameter(
  const std::string& name) const;
template CUOPT_EXPORT float solver_settings_t<int, float>::get_parameter(
  const std::string& name) const;
template CUOPT_EXPORT bool solver_settings_t<int, float>::get_parameter(
  const std::string& name) const;
template CUOPT_EXPORT std::string solver_settings_t<int, float>::get_parameter(
  const std::string& name) const;
#endif

#if MIP_INSTANTIATE_DOUBLE
template CUOPT_EXPORT void solver_settings_t<int, double>::set_parameter_from_string(
  const std::string&, const std::string&);
template CUOPT_EXPORT std::string solver_settings_t<int, double>::get_parameter_as_string(
  const std::string&) const;
template CUOPT_EXPORT void solver_settings_t<int, double>::set_mip_callback(
  internals::base_solution_callback_t*, void*);
template CUOPT_EXPORT const std::vector<internals::base_solution_callback_t*>
solver_settings_t<int, double>::get_mip_callbacks() const;
template CUOPT_EXPORT pdlp_solver_settings_t<int, double>&
solver_settings_t<int, double>::get_pdlp_settings();
template CUOPT_EXPORT mip_solver_settings_t<int, double>&
solver_settings_t<int, double>::get_mip_settings();
template CUOPT_EXPORT const std::vector<parameter_info_t<double>>&
solver_settings_t<int, double>::get_float_parameters() const;
template CUOPT_EXPORT const std::vector<parameter_info_t<int>>&
solver_settings_t<int, double>::get_int_parameters() const;
template CUOPT_EXPORT const std::vector<parameter_info_t<bool>>&
solver_settings_t<int, double>::get_bool_parameters() const;
template CUOPT_EXPORT const std::vector<std::string>
solver_settings_t<int, double>::get_parameter_names() const;
template CUOPT_EXPORT const std::vector<parameter_info_t<std::string>>&
solver_settings_t<int, double>::get_string_parameters() const;
template CUOPT_EXPORT const pdlp_warm_start_data_view_t<int, double>&
solver_settings_t<int, double>::get_pdlp_warm_start_data_view() const noexcept;
template CUOPT_EXPORT void solver_settings_t<int, double>::load_parameters_from_file(
  const std::string&);
template CUOPT_EXPORT bool solver_settings_t<int, double>::dump_parameters_to_file(
  const std::string&, bool) const;
template CUOPT_EXPORT void solver_settings_t<int, double>::set_parameter(const std::string& name,
                                                                         int value);
template CUOPT_EXPORT void solver_settings_t<int, double>::set_parameter(const std::string& name,
                                                                         double value);
template CUOPT_EXPORT void solver_settings_t<int, double>::set_parameter(const std::string& name,
                                                                         bool value);
template CUOPT_EXPORT int solver_settings_t<int, double>::get_parameter(
  const std::string& name) const;
template CUOPT_EXPORT double solver_settings_t<int, double>::get_parameter(
  const std::string& name) const;
template CUOPT_EXPORT bool solver_settings_t<int, double>::get_parameter(
  const std::string& name) const;
template CUOPT_EXPORT std::string solver_settings_t<int, double>::get_parameter(
  const std::string& name) const;
#endif

}  // namespace cuopt::mathematical_optimization
