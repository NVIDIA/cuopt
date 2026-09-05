/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

// Device-facing members of solver_settings_t, split out of solver_settings.cu.
//
// Everything else in that class is host-only parameter handling, so the remainder now
// builds as solver_settings.cpp into the CUDA-free cuopt_client library. Only these
// members take a cuda::stream_ref or hand back a device_uvector, so they are the
// only ones that must stay in a CUDA TU inside libcuopt.
//
// The `template class` instantiation in solver_settings.cpp cannot emit these members
// (their definitions are not visible there), so they are instantiated explicitly below.

#include <cuopt/mathematical_optimization/solver_settings.hpp>

#include <rmm/device_uvector.hpp>

#include <cuda/stream_ref>

#include <mip_heuristics/mip_constants.hpp>

namespace cuopt {
namespace CUOPT_EXPORT mathematical_optimization {

template <typename i_t, typename f_t>
void solver_settings_t<i_t, f_t>::set_initial_pdlp_primal_solution(const f_t* solution,
                                                                   i_t size,
                                                                   cuda::stream_ref stream)
{
  pdlp_settings.set_initial_primal_solution(solution, size, stream);
}

template <typename i_t, typename f_t>
void solver_settings_t<i_t, f_t>::set_initial_pdlp_dual_solution(const f_t* solution,
                                                                 i_t size,
                                                                 cuda::stream_ref stream)
{
  pdlp_settings.set_initial_dual_solution(solution, size, stream);
}

template <typename i_t, typename f_t>
void solver_settings_t<i_t, f_t>::set_pdlp_warm_start_data(
  const f_t* current_primal_solution,
  const f_t* current_dual_solution,
  const f_t* initial_primal_average,
  const f_t* initial_dual_average,
  const f_t* current_ATY,
  const f_t* sum_primal_solutions,
  const f_t* sum_dual_solutions,
  const f_t* last_restart_duality_gap_primal_solution,
  const f_t* last_restart_duality_gap_dual_solution,
  i_t primal_size,
  i_t dual_size,
  f_t initial_primal_weight,
  f_t initial_step_size,
  i_t total_pdlp_iterations,
  i_t total_pdhg_iterations,
  f_t last_candidate_kkt_score,
  f_t last_restart_kkt_score,
  f_t sum_solution_weight,
  i_t iterations_since_last_restart)
{
  pdlp_settings.set_pdlp_warm_start_data(current_primal_solution,
                                         current_dual_solution,
                                         initial_primal_average,
                                         initial_dual_average,
                                         current_ATY,
                                         sum_primal_solutions,
                                         sum_dual_solutions,
                                         last_restart_duality_gap_primal_solution,
                                         last_restart_duality_gap_dual_solution,
                                         primal_size,
                                         dual_size,
                                         initial_primal_weight,
                                         initial_step_size,
                                         total_pdlp_iterations,
                                         total_pdhg_iterations,
                                         last_candidate_kkt_score,
                                         last_restart_kkt_score,
                                         sum_solution_weight,
                                         iterations_since_last_restart);
}

template <typename i_t, typename f_t>
const rmm::device_uvector<f_t>& solver_settings_t<i_t, f_t>::get_initial_pdlp_primal_solution()
  const
{
  return pdlp_settings.get_initial_primal_solution();
}

template <typename i_t, typename f_t>
const rmm::device_uvector<f_t>& solver_settings_t<i_t, f_t>::get_initial_pdlp_dual_solution() const
{
  return pdlp_settings.get_initial_dual_solution();
}

template <typename i_t, typename f_t>
void solver_settings_t<i_t, f_t>::add_initial_mip_solution(const f_t* solution,
                                                           i_t size,
                                                           cuda::stream_ref stream)
{
  mip_settings.add_initial_solution(solution, size, stream);
}

#if MIP_INSTANTIATE_FLOAT
template CUOPT_EXPORT void solver_settings_t<int, float>::set_initial_pdlp_primal_solution(
  const float*, int, cuda::stream_ref);
template CUOPT_EXPORT void solver_settings_t<int, float>::set_initial_pdlp_dual_solution(
  const float*, int, cuda::stream_ref);
template CUOPT_EXPORT const rmm::device_uvector<float>&
solver_settings_t<int, float>::get_initial_pdlp_primal_solution() const;
template CUOPT_EXPORT const rmm::device_uvector<float>&
solver_settings_t<int, float>::get_initial_pdlp_dual_solution() const;
template CUOPT_EXPORT void solver_settings_t<int, float>::add_initial_mip_solution(
  const float*, int, cuda::stream_ref);
// The 19-argument host overload. It was moved into this TU with the rest of the block, but
// `template class` in solver_settings.cpp cannot emit it (definition not visible there), so
// without this line the symbol disappears -- and it is the one the Cython layer binds to,
// which takes down every Python test, docs-build and wheel-test job.
template CUOPT_EXPORT void solver_settings_t<int, float>::set_pdlp_warm_start_data(const float*,
                                                                                   const float*,
                                                                                   const float*,
                                                                                   const float*,
                                                                                   const float*,
                                                                                   const float*,
                                                                                   const float*,
                                                                                   const float*,
                                                                                   const float*,
                                                                                   int,
                                                                                   int,
                                                                                   float,
                                                                                   float,
                                                                                   int,
                                                                                   int,
                                                                                   float,
                                                                                   float,
                                                                                   float,
                                                                                   int);
#endif

#if MIP_INSTANTIATE_DOUBLE
template CUOPT_EXPORT void solver_settings_t<int, double>::set_initial_pdlp_primal_solution(
  const double*, int, cuda::stream_ref);
template CUOPT_EXPORT void solver_settings_t<int, double>::set_initial_pdlp_dual_solution(
  const double*, int, cuda::stream_ref);
template CUOPT_EXPORT const rmm::device_uvector<double>&
solver_settings_t<int, double>::get_initial_pdlp_primal_solution() const;
template CUOPT_EXPORT const rmm::device_uvector<double>&
solver_settings_t<int, double>::get_initial_pdlp_dual_solution() const;
template CUOPT_EXPORT void solver_settings_t<int, double>::add_initial_mip_solution(
  const double*, int, cuda::stream_ref);
// The 19-argument host overload. It was moved into this TU with the rest of the block, but
// `template class` in solver_settings.cpp cannot emit it (definition not visible there), so
// without this line the symbol disappears -- and it is the one the Cython layer binds to,
// which takes down every Python test, docs-build and wheel-test job.
template CUOPT_EXPORT void solver_settings_t<int, double>::set_pdlp_warm_start_data(const double*,
                                                                                    const double*,
                                                                                    const double*,
                                                                                    const double*,
                                                                                    const double*,
                                                                                    const double*,
                                                                                    const double*,
                                                                                    const double*,
                                                                                    const double*,
                                                                                    int,
                                                                                    int,
                                                                                    double,
                                                                                    double,
                                                                                    int,
                                                                                    int,
                                                                                    double,
                                                                                    double,
                                                                                    double,
                                                                                    int);
#endif

}  // namespace CUOPT_EXPORT mathematical_optimization
}  // namespace cuopt
