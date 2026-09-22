/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#include "escape.hpp"
#include "../internal.hpp"
#include "../problem.hpp"
#include "api.hpp"

namespace cuopt::mathematical_optimization::mip {

template <typename i_t, typename f_t>
void randomize_variable(fj_cpu_climber_t<i_t, f_t>& fj_cpu, i_t var_idx, cuopt::pcgenerator_t& rng)
{
  f_t lb  = std::max(get_lower(fj_cpu.h_var_bounds[var_idx].get()), -1e7);
  f_t ub  = std::min(get_upper(fj_cpu.h_var_bounds[var_idx].get()), 1e7);
  f_t val = lb + (ub - lb) * rng.next_double();
  if (is_integer_var<i_t, f_t>(fj_cpu, var_idx)) {
    lb  = std::ceil(lb);
    ub  = std::floor(ub);
    val = std::round(val);
  }
  val = std::clamp(val, get_lower(fj_cpu.h_var_bounds[var_idx].get()), get_upper(fj_cpu.h_var_bounds[var_idx].get()));

  fj_cpu.h_assignment[var_idx] = val;
}

template <typename i_t, typename f_t>
void perturb(fj_cpu_climber_t<i_t, f_t>& fj_cpu)
{
  CPUFJ_NVTX_RANGE("CPUFJ::perturb");
  if (fj_cpu.feasible_found) {
    cuopt_assert(fj_cpu.h_assignment.size() == fj_cpu.h_best_assignment.size(),
                 "incumbent_assignment span would be invalidated");
    fj_cpu.h_assignment = fj_cpu.h_best_assignment;
  }

  const i_t n_kick = std::max<i_t>(1, fj_cpu.perturb_vars);
  std::vector<i_t> sampled_vars = fj_cpu.problem->h_objective_vars;
  fj_cpu.rng.shuffle(sampled_vars);
  sampled_vars.resize(std::min(sampled_vars.size(), (size_t)n_kick));
  cuopt::pcgenerator_t rng(fj_cpu.settings.seed + fj_cpu.iterations, 0, 0);

  for (auto var_idx : sampled_vars)
    randomize_variable<i_t, f_t>(fj_cpu, var_idx, rng);

  ++fj_cpu.n_lhs_recompute_perturb;
  recompute_slack(fj_cpu);
}

#if MIP_INSTANTIATE_FLOAT
template void perturb<int, float>(fj_cpu_climber_t<int, float>&);
#endif

#if MIP_INSTANTIATE_DOUBLE
template void perturb<int, double>(fj_cpu_climber_t<int, double>&);
#endif

}  // namespace cuopt::mathematical_optimization::mip
