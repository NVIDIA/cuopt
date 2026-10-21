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
// solver_settings.cpp deliberately has no `template class` at all -- that would instantiate
// the constructor, which needs CUDA (see the note there) -- so it cannot emit these members
// either. The `template class` below covers the class as a whole for libcuopt; members
// defined in the CUDA-free TU are instantiated individually there.

#include <cuopt/mathematical_optimization/solver_settings.hpp>

#include <cuda/stream>
#include <rmm/device_uvector.hpp>

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

template <typename i_t, typename f_t>
solver_settings_t<i_t, f_t>::solver_settings_t()
  : pdlp_settings(), mip_settings(), registry_(pdlp_settings, mip_settings)
{
}

#if MIP_INSTANTIATE_FLOAT
// Emits the ctor/dtor/copy for the whole class; solver_settings.cpp deliberately does not,
// because those need CUDA (see the note there).
template class CUOPT_EXPORT solver_settings_t<int, float>;
#endif

#if MIP_INSTANTIATE_DOUBLE
// Emits the ctor/dtor/copy for the whole class; solver_settings.cpp deliberately does not,
// because those need CUDA (see the note there).
template class CUOPT_EXPORT solver_settings_t<int, double>;
#endif

}  // namespace CUOPT_EXPORT mathematical_optimization
}  // namespace cuopt
