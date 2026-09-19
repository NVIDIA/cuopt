/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#include "escape.hpp"
#include "../audit.hpp"
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
  val = std::clamp(val,
                   get_lower(fj_cpu.h_var_bounds[var_idx].get()),
                   get_upper(fj_cpu.h_var_bounds[var_idx].get()));

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
    if (fj_cpu.shared_incumbent) {
      f_t adopted_objective{};
      if (fj_cpu.shared_incumbent->adopt(
            fj_cpu.h_best_objective, fj_cpu.h_assignment, &adopted_objective)) {
        // A restart must treat the adopted assignment as its own incumbent. Previously only the
        // current assignment changed: the stale local threshold could cause an already-shared
        // solution to be rediscovered/reported and made objective scoring inconsistent until a
        // later full recomputation.
        fj_cpu.h_incumbent_objective = adopted_objective;
        fj_cpu.h_objective_sumcomp   = 0;
        fj_cpu.h_best_objective =
          adopted_objective - fj_cpu.settings.parameters.breakthrough_move_epsilon;
        fj_cpu.h_best_assignment     = fj_cpu.h_assignment;
        fj_cpu.iterations_since_best = 0;
        fj_cpu.perturb_streak        = 0;
      }
      cuopt_func_call(audit_assignment_bounds(fj_cpu, "shared adopt"));
    }
  }

  // select N variables, assign them a random value between their bounds. N grows with consecutive
  // unproductive perturbations, so a lane stuck repeating the same small kick around one basin
  // widens its jump instead of retrying the same neighbourhood.
  const i_t n_kick = std::min<i_t>(fj_cpu.hp.perturb_escalate_cap,
                                   std::max<i_t>(1, fj_cpu.perturb_vars + fj_cpu.perturb_streak));
  const bool categorical_kick = fj_cpu.feasible_found && fj_cpu.continuous_perturb_fraction > 0 &&
                                fj_cpu.problem->card_row_offsets.size() > 1;
  const i_t scalar_kicks          = n_kick - categorical_kick;
  std::vector<i_t> sampled_vars   = fj_cpu.problem->h_objective_vars;
  fj_cpu.rng.shuffle(sampled_vars);
  sampled_vars.resize(std::min(sampled_vars.size(), (size_t)scalar_kicks));
  cuopt::pcgenerator_t rng(fj_cpu.settings.seed + fj_cpu.iterations, 0, 0);

  // Change one selected region directly. Two independent scalar flips would temporarily violate
  // the exact-one equality and make the intended categorical transition difficult to discover.
  if (categorical_kick) {
    const auto& offsets   = fj_cpu.problem->card_row_offsets;
    const auto& variables = fj_cpu.problem->card_variables;
    const i_t group = rng.next_u32() % static_cast<uint32_t>(offsets.size() - 1);
    const i_t begin = offsets[group];
    const i_t width = offsets[group + 1] - begin;
    i_t active      = -1;
    for (i_t q = begin; q < begin + width; ++q) {
      if (fj_cpu.h_assignment[variables[q]] > f_t{0.5}) {
        active = q;
        break;
      }
    }
    if (active >= 0 && width > 1) {
      i_t replacement = begin + rng.next_u32() % static_cast<uint32_t>(width - 1);
      if (replacement >= active) ++replacement;
      fj_cpu.h_assignment[variables[active]]      = f_t{0};
      fj_cpu.h_assignment[variables[replacement]] = f_t{1};
    }
  }

  // Unbounded whole-domain draws destroy the useful scale of a geometric incumbent. Perturb
  // continuous coordinates locally, and let half the lanes bias the draw toward objective descent.
  f_t radius = 0;
  if (fj_cpu.feasible_found && fj_cpu.continuous_perturb_fraction > 0) {
    f_t scale = 1;
    for (i_t variable : fj_cpu.problem->h_objective_vars) {
      const f_t lower = get_lower(fj_cpu.h_var_bounds[variable].get());
      const f_t value = fj_cpu.h_best_assignment[variable];
      scale = std::max(scale, std::abs(value - (std::isfinite(lower) ? lower : f_t{0})));
    }
    radius = fj_cpu.continuous_perturb_fraction * scale;
  }
  for (i_t variable : sampled_vars) {
    if (radius > 0 && !is_integer_var<i_t, f_t>(fj_cpu, variable)) {
      const auto bounds = fj_cpu.h_var_bounds[variable].get();
      const f_t current = fj_cpu.h_assignment[variable];
      f_t lower = std::max(get_lower(bounds), current - radius);
      f_t upper = std::min(get_upper(bounds), current + radius);
      if (fj_cpu.objective_directed_perturb) {
        const f_t coefficient = fj_cpu.problem->h_obj_coeffs[variable];
        if (coefficient > 0) upper = current;
        if (coefficient < 0) lower = current;
      }
      fj_cpu.h_assignment[variable] = lower + (upper - lower) * rng.next_double();
    } else {
      randomize_variable<i_t, f_t>(fj_cpu, variable, rng);
    }
  }

  ++fj_cpu.n_lhs_recompute_perturb;
  ++fj_cpu.perturb_streak;
  recompute_slack(fj_cpu);
  retire_var_best_moves<i_t, f_t>(fj_cpu);
}

template <typename i_t, typename f_t>
void reset_infeasible_checkpoint(fj_cpu_climber_t<i_t, f_t>& fj_cpu)
{
  fj_cpu.h_best_infeasible_assignment.clear();
  fj_cpu.best_infeasible_severity       = std::numeric_limits<f_t>::infinity();
  fj_cpu.checkpoint_severity            = std::numeric_limits<f_t>::infinity();
  fj_cpu.iters_since_infeasible_improve = 0;
}

template <typename i_t, typename f_t>
void restart_from_infeasible_checkpoint(fj_cpu_climber_t<i_t, f_t>& fj_cpu)
{
  cuopt_assert(fj_cpu.h_assignment.size() == fj_cpu.h_best_infeasible_assignment.size(),
               "incumbent_assignment span would be invalidated");
  fj_cpu.h_assignment = fj_cpu.h_best_infeasible_assignment;
  ++fj_cpu.n_lhs_recompute_restart;
  recompute_slack(fj_cpu);
  invalidate_mtm_cache(fj_cpu);
  cuopt_func_call(audit_assignment_bounds(fj_cpu, "checkpoint restore"));
}

template <typename i_t, typename f_t>
void infeasible_kick(fj_cpu_climber_t<i_t, f_t>& fj_cpu)
{
  if (fj_cpu.violated_constraints.empty()) return;
  cuopt::pcgenerator_t rng(fj_cpu.settings.seed + 2654435761u * fj_cpu.iterations, 0, 0);
  bool moved = false;
  if (fj_cpu.use_directed_infeasible_kick) {
    const auto& violated = fj_cpu.violated_constraints.contents;
    const i_t row        = violated[rng.next_u32() % (uint32_t)violated.size()];
    const f_t slack      = fj_cpu.row_state()[row].slack;
    f_t best_error = std::numeric_limits<f_t>::infinity(), best_value = 0;
    i_t best_var            = -1;
    const auto [begin, end] = fj_cpu.range_for_row(row);
    for (i_t p = begin; p < end; ++p) {
      const i_t var   = fj_cpu.h_variables[p];
      const f_t coeff = fj_cpu.h_coefficients[p];
      if (coeff == 0) continue;
      const f_t old     = fj_cpu.h_assignment[var];
      f_t value         = old + slack / coeff;
      const auto bounds = fj_cpu.h_var_bounds[var].get();
      if (is_integer_var<i_t, f_t>(fj_cpu, var))
        value = value > old ? std::ceil(value) : std::floor(value);
      value              = std::clamp(value, get_lower(bounds), get_upper(bounds));
      const f_t progress = -coeff * (value - old);
      if (progress <= fj_cpu.row_tolerance) continue;
      const f_t error = std::fabs(progress + slack);
      if (error < best_error) {
        best_error = error;
        best_var   = var;
        best_value = value;
      }
    }
    if (best_var >= 0) {
      fj_cpu.h_assignment[best_var] = best_value;
      moved                         = true;
    }
  }
  if (!moved) {
    const auto& violated = fj_cpu.violated_constraints.contents;
    for (i_t k = 0; k < std::max<i_t>(1, fj_cpu.infeasible_kick_vars); ++k) {
      const i_t row           = violated[rng.next_u32() % (uint32_t)violated.size()];
      const auto [begin, end] = fj_cpu.range_for_row(row);
      if (begin < end)
        randomize_variable<i_t, f_t>(
          fj_cpu, fj_cpu.h_variables[begin + rng.next_u32() % (uint32_t)(end - begin)], rng);
    }
  }
  ++fj_cpu.n_lhs_recompute_perturb;
  recompute_slack(fj_cpu);
  retire_var_best_moves<i_t, f_t>(fj_cpu);
  invalidate_mtm_cache(fj_cpu);
}

template <typename i_t, typename f_t>
void track_infeasible_checkpoint(fj_cpu_climber_t<i_t, f_t>& fj_cpu)
{
  CPUFJ_NVTX_RANGE("CPUFJ::track_infeasible_checkpoint");
  if (fj_cpu.violated_constraints.empty()) {
    reset_infeasible_checkpoint(fj_cpu);
    return;
  }

  cuopt_func_call(audit_incremental_state(fj_cpu, "checkpoint"));
  const f_t severity = -fj_cpu.total_violations;
  cuopt_assert(severity >= 0, "violation severity should be positive or zero");

  if (severity < fj_cpu.best_infeasible_severity) {
    fj_cpu.best_infeasible_severity       = severity;
    fj_cpu.iters_since_infeasible_improve = 0;
    fj_cpu.restores_since_improvement     = 0;
    if (severity < fj_cpu.checkpoint_severity * fj_cpu.infeasible_checkpoint_refresh_ratio) {
      fj_cpu.h_best_infeasible_assignment = fj_cpu.h_assignment;
      fj_cpu.checkpoint_severity          = severity;
      ++fj_cpu.n_checkpoint_snapshots;
    }
    return;
  }

  // A lane that has never crossed and has exhausted its restores abandons the basin outright.
  if (!fj_cpu.feasible_found) {
    if (fj_cpu.infeasible_kick_interval > 0 && fj_cpu.iters_since_infeasible_improve > 0 &&
        fj_cpu.iters_since_infeasible_improve % fj_cpu.infeasible_kick_interval == 0)
      infeasible_kick(fj_cpu);
    const i_t nnz_scale = 1 + fj_cpu.problem->nnz / fj_cpu.hp.restart_window_nnz_scale;
    const i_t capped =
      nnz_scale < fj_cpu.hp.restart_window_scale_max ? nnz_scale
                                                     : fj_cpu.hp.restart_window_scale_max;
    if (fj_cpu.iters_since_infeasible_improve >=
          fj_cpu.hp.restart_window_multiple * fj_cpu.infeasible_restart_window * capped &&
        fj_cpu.restores_since_improvement >= fj_cpu.infeasible_restart_max_streak) {
      cuopt::pcgenerator_t rng(fj_cpu.settings.seed + fj_cpu.iterations, 0, 0);
      const bool soft = !(fj_cpu.settings.seed & 1) && !fj_cpu.h_best_infeasible_assignment.empty();
      if (soft) fj_cpu.h_assignment = fj_cpu.h_best_infeasible_assignment;
      for (i_t var = 0; var < fj_cpu.problem->n_variables; ++var)
        if (!soft || rng.next_double() < 0.3) randomize_variable<i_t, f_t>(fj_cpu, var, rng);

      ++fj_cpu.n_lhs_recompute_restart;
      recompute_slack(fj_cpu);
      invalidate_mtm_cache(fj_cpu);
      reset_infeasible_checkpoint(fj_cpu);
      fj_cpu.restores_since_improvement = 0;
      cuopt_func_call(audit_assignment_bounds(fj_cpu, "randomized restart"));

      CUOPT_LOG_DEBUG(
        "%sCPUFJ randomized restart at iteration %d", fj_cpu.log_prefix.c_str(), fj_cpu.iterations);
      return;
    }
  }

  if (fj_cpu.restores_since_improvement >= fj_cpu.infeasible_restart_max_streak) return;
  if (++fj_cpu.iters_since_infeasible_improve < fj_cpu.infeasible_restart_window) return;
  if (severity <= fj_cpu.best_infeasible_severity * fj_cpu.infeasible_restart_degrade_ratio) return;
  if (fj_cpu.h_best_infeasible_assignment.empty()) return;

  cuopt_assert(fj_cpu.checkpoint_severity >= fj_cpu.best_infeasible_severity,
               "checkpoint cannot beat the best severity seen");

  restart_from_infeasible_checkpoint(fj_cpu);

  ++fj_cpu.n_checkpoint_restores;
  ++fj_cpu.restores_since_improvement;
  if (fj_cpu.restores_since_improvement > fj_cpu.max_restores_since_improvement)
    fj_cpu.max_restores_since_improvement = fj_cpu.restores_since_improvement;
  fj_cpu.iters_since_infeasible_improve = 0;
}

#if MIP_INSTANTIATE_FLOAT
template void perturb<int, float>(fj_cpu_climber_t<int, float>&);
template void reset_infeasible_checkpoint<int, float>(fj_cpu_climber_t<int, float>&);
template void track_infeasible_checkpoint<int, float>(fj_cpu_climber_t<int, float>&);
#endif

#if MIP_INSTANTIATE_DOUBLE
template void perturb<int, double>(fj_cpu_climber_t<int, double>&);
template void reset_infeasible_checkpoint<int, double>(fj_cpu_climber_t<int, double>&);
template void track_infeasible_checkpoint<int, double>(fj_cpu_climber_t<int, double>&);
#endif

}  // namespace cuopt::mathematical_optimization::mip
