/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

// Warm-start accessors of pdlp_solver_settings_t, split out of solver_settings.cu.
//
// These are trivial `return member_;` getters -- they hand back a reference and emit no
// device code, even where the referent is a GPU type. The gRPC client needs them, so they
// build into the CUDA-free cuopt_client library while the rest of the class (which does
// real thrust/rmm work) stays in solver_settings.cu.
//
// Only these members are instantiated below, deliberately NOT `template class`: the class
// holds a pdlp_warm_start_data_t, so instantiating all of it here would pull in device
// ctor/dtor code that belongs in the CUDA TU.

#include <cuopt/export.hpp>
#include <cuopt/mathematical_optimization/pdlp/solver_settings.hpp>

// Required: the explicit instantiations below are guarded on MIP_INSTANTIATE_* /
// PDLP_INSTANTIATE_*. Without this header those macros are undefined, the guards
// evaluate false, and this TU silently compiles to zero symbols.
#include <mip_heuristics/mip_constants.hpp>

namespace cuopt::mathematical_optimization {

template <typename i_t, typename f_t>
const cpu_pdlp_warm_start_data_t<i_t, f_t>&
pdlp_solver_settings_t<i_t, f_t>::get_cpu_pdlp_warm_start_data() const noexcept
{
  return cpu_pdlp_warm_start_data_;
}

template <typename i_t, typename f_t>
cpu_pdlp_warm_start_data_t<i_t, f_t>&
pdlp_solver_settings_t<i_t, f_t>::get_cpu_pdlp_warm_start_data() noexcept
{
  return cpu_pdlp_warm_start_data_;
}

template <typename i_t, typename f_t>
const pdlp_warm_start_data_view_t<i_t, f_t>&
pdlp_solver_settings_t<i_t, f_t>::get_pdlp_warm_start_data_view() const noexcept
{
  return pdlp_warm_start_data_view_;
}

#if MIP_INSTANTIATE_FLOAT || PDLP_INSTANTIATE_FLOAT
template CUOPT_EXPORT const cpu_pdlp_warm_start_data_t<int, float>&
pdlp_solver_settings_t<int, float>::get_cpu_pdlp_warm_start_data() const noexcept;
template CUOPT_EXPORT cpu_pdlp_warm_start_data_t<int, float>&
pdlp_solver_settings_t<int, float>::get_cpu_pdlp_warm_start_data() noexcept;
template CUOPT_EXPORT const pdlp_warm_start_data_view_t<int, float>&
pdlp_solver_settings_t<int, float>::get_pdlp_warm_start_data_view() const noexcept;
#endif

#if MIP_INSTANTIATE_DOUBLE
template CUOPT_EXPORT const cpu_pdlp_warm_start_data_t<int, double>&
pdlp_solver_settings_t<int, double>::get_cpu_pdlp_warm_start_data() const noexcept;
template CUOPT_EXPORT cpu_pdlp_warm_start_data_t<int, double>&
pdlp_solver_settings_t<int, double>::get_cpu_pdlp_warm_start_data() noexcept;
template CUOPT_EXPORT const pdlp_warm_start_data_view_t<int, double>&
pdlp_solver_settings_t<int, double>::get_pdlp_warm_start_data_view() const noexcept;
#endif

}  // namespace cuopt::mathematical_optimization
