/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#include "structure.hpp"
#include <numeric>
#include <utilities/integer_scaling.hpp>
#include "../climber.hpp"
#include "../internal.hpp"
#include "../problem.hpp"
#include "../search/api.hpp"

namespace cuopt::mathematical_optimization::mip {

template <typename i_t, typename f_t>
double implied_integer_row_scale(const fj_cpu_problem_t<i_t, f_t>& problem,
                                 i_t row_idx,
                                 f_t pivot_coeff)
{
  const i_t begin = problem.offsets[row_idx], end = problem.offsets[row_idx + 1];
  const f_t rhs = problem.cstr_lb[row_idx];
  if (!scaling_bound_finite(rhs)) return 0;
  const double scale = row_int_scale(problem.coefficients.data() + begin,
                                     end - begin,
                                     rhs,
                                     rhs,
                                     end - begin,
                                     std::numeric_limits<int16_t>::max());
  if (scale == 0) return 0;
  const int64_t divisor = std::llround(scale * pivot_coeff);
  if (divisor == 0 || std::llround(scale * rhs) % divisor != 0) return 0;
  for (i_t entry_idx = begin; entry_idx < end; ++entry_idx)
    if (std::llround(scale * problem.coefficients[entry_idx]) % divisor != 0) return 0;
  return scale;
}

template <typename i_t, typename f_t>
void detect_implied_integers(fj_cpu_climber_t<i_t, f_t>& fj_cpu,
                             fj_cpu_problem_t<i_t, f_t>& problem)
{
  const i_t n_variables = problem.n_variables, n_constraints = problem.n_constraints;
  // Snapshot the original types for the initial equality counts.
  std::vector<uint8_t> was_continuous(n_variables);
  bool has_continuous = false;
  for (i_t var_idx = 0; var_idx < n_variables; ++var_idx)
    has_continuous |= was_continuous[var_idx] = problem.h_var_types[var_idx] == var_t::CONTINUOUS;
  if (!has_continuous) return;
  // Equalities with one continuous variable can force it integer; newly integral
  // variables may expose another such equality through the existing transpose.
  std::vector<i_t> n_continuous_in_row(n_constraints, 0), pending_rows;
  for (i_t row_idx = 0; row_idx < n_constraints; ++row_idx) {
    if (!std::isfinite(problem.cstr_lb[row_idx]) ||
        problem.cstr_lb[row_idx] != problem.cstr_ub[row_idx])
      continue;
    for (i_t entry_idx = problem.offsets[row_idx]; entry_idx < problem.offsets[row_idx + 1];
         ++entry_idx)
      n_continuous_in_row[row_idx] += was_continuous[problem.variables[entry_idx]];
    if (n_continuous_in_row[row_idx] == 1) pending_rows.push_back(row_idx);
  }
  i_t n_forced = 0, n_completable = 0;
  std::vector<i_t> changed_vars;
  // A row reaches one unknown at most once. Each newly integral column updates
  // its incident counts once, so even a long chain costs O(nnz + rows + columns).
  for (size_t queue_idx = 0; queue_idx < pending_rows.size(); ++queue_idx) {
    const i_t row_idx = pending_rows[queue_idx];
    if (n_continuous_in_row[row_idx] != 1) continue;
    i_t var_idx     = -1;
    f_t pivot_coeff = 0;
    for (i_t entry_idx = problem.offsets[row_idx]; entry_idx < problem.offsets[row_idx + 1];
         ++entry_idx) {
      if (problem.h_var_types[problem.variables[entry_idx]] == var_t::CONTINUOUS) {
        var_idx     = problem.variables[entry_idx];
        pivot_coeff = problem.coefficients[entry_idx];
        break;
      }
    }
    if (var_idx < 0 || !implied_integer_row_scale(problem, row_idx, pivot_coeff)) continue;
    const auto bounds = fj_cpu.h_var_bounds[var_idx].get();
    bool valid        = scaling_bound_finite(get_lower(bounds)) &&
                 scaling_bound_finite(get_upper(bounds)) &&
                 get_lower(bounds) == std::round(get_lower(bounds)) &&
                 get_upper(bounds) == std::round(get_upper(bounds));
    if (!valid) continue;
    problem.h_var_types[var_idx] = var_t::INTEGER;
    changed_vars.push_back(var_idx);
    ++n_forced;
    for (i_t entry_idx = problem.reverse_offsets[var_idx];
         entry_idx < problem.reverse_offsets[var_idx + 1];
         ++entry_idx) {
      const i_t incident_row = problem.reverse_constraints[entry_idx];
      if (n_continuous_in_row[incident_row] > 0 && --n_continuous_in_row[incident_row] == 1)
        pending_rows.push_back(incident_row);
    }
  }
  // Each pair variable occurs only in this equality, so changing the pair cannot
  // affect other rows. After scaling, the equality fixes their difference to an
  // integer. With zero lower bounds, integral upper bounds and equal positive
  // costs, setting the smaller variable to zero gives an integral optimal completion.
  // Fractional feasible pairs can still exist; integrality here preserves an optimum.
  for (i_t row_idx = 0; row_idx < n_constraints; ++row_idx) {
    if (n_continuous_in_row[row_idx] != 2) continue;
    i_t first_var = -1, second_var = -1;
    f_t pivot_coeff = 0, second_coeff = 0;
    for (i_t entry_idx = problem.offsets[row_idx]; entry_idx < problem.offsets[row_idx + 1];
         ++entry_idx) {
      const i_t var_idx = problem.variables[entry_idx];
      if (problem.h_var_types[var_idx] != var_t::CONTINUOUS) continue;
      if (first_var < 0) {
        first_var   = var_idx;
        pivot_coeff = problem.coefficients[entry_idx];
      } else {
        second_var   = var_idx;
        second_coeff = problem.coefficients[entry_idx];
      }
    }
    if (first_var < 0 || second_var < 0 || first_var == second_var) continue;
    const double scale = implied_integer_row_scale(problem, row_idx, pivot_coeff);
    if (scale == 0 || std::llround(scale * second_coeff) != -std::llround(scale * pivot_coeff))
      continue;
    bool valid = problem.h_obj_coeffs[first_var] > 0 &&
                 std::isfinite(problem.h_obj_coeffs[first_var]) &&
                 problem.h_obj_coeffs[first_var] == problem.h_obj_coeffs[second_var];
    for (i_t var_idx : {first_var, second_var}) {
      const auto bounds = fj_cpu.h_var_bounds[var_idx].get();
      valid &= problem.reverse_offsets[var_idx + 1] - problem.reverse_offsets[var_idx] == 1 &&
               get_lower(bounds) == f_t{0} && get_upper(bounds) >= f_t{0} &&
               scaling_bound_finite(get_upper(bounds)) &&
               get_upper(bounds) == std::round(get_upper(bounds));
    }
    if (!valid) continue;
    problem.h_var_types[first_var] = problem.h_var_types[second_var] = var_t::INTEGER;
    changed_vars.push_back(first_var);
    changed_vars.push_back(second_var);
    n_completable += 2;
  }

  // update classification for the newly-implied-integer vars
  for (i_t var_idx : changed_vars) {
    fj_cpu.h_assignment[var_idx]      = std::round((f_t)fj_cpu.h_assignment[var_idx]);
    fj_cpu.h_best_assignment[var_idx] = std::round((f_t)fj_cpu.h_best_assignment[var_idx]);
    const auto bounds                 = fj_cpu.h_var_bounds[var_idx].get();
    if (get_lower(bounds) == f_t{0} && get_upper(bounds) == f_t{1}) {
      fj_cpu.h_is_binary_variable[var_idx] = 1;
      fj_cpu.h_binary_indices.push_back(var_idx);
    }
  }
  CUOPT_LOG_DEBUG(
    "CPUFJ implied integrality: %d forced, %d integral-completable", n_forced, n_completable);
}

template <typename i_t, typename f_t>
void precompute_problem_features(fj_cpu_climber_t<i_t, f_t>& fj_cpu,
                                 fj_cpu_problem_t<i_t, f_t>& problem)
{
  phase_timer_t timer(fj_cpu.t_features);
  fj_cpu.n_binary_vars  = 0;
  fj_cpu.n_integer_vars = 0;
  for (i_t i = 0; i < (i_t)fj_cpu.h_is_binary_variable.size(); i++) {
    if (fj_cpu.h_is_binary_variable[i]) {
      fj_cpu.n_binary_vars++;
    } else if (problem.h_var_types[i] == var_t::INTEGER) {
      fj_cpu.n_integer_vars++;
    }
  }

  i_t total_nnz = problem.reverse_offsets.back();
  i_t n_vars    = problem.reverse_offsets.size() - 1;
  i_t n_cstrs   = problem.offsets.size() - 1;

  problem.avg_var_degree = (double)total_nnz / n_vars;

  problem.max_var_degree = 0;
  std::vector<i_t> var_degrees(n_vars);
  for (i_t i = 0; i < n_vars; i++) {
    i_t degree             = problem.reverse_offsets[i + 1] - problem.reverse_offsets[i];
    var_degrees[i]         = degree;
    problem.max_var_degree = std::max(problem.max_var_degree, degree);
  }

  i_t equalities = 0;
  for (i_t row = 0; row < n_cstrs; ++row)
    equalities += problem.cstr_lb[row] == problem.cstr_ub[row];
  problem.equality_fraction = n_cstrs ? (double)equalities / n_cstrs : 0.0;
}

template <typename i_t, typename f_t>
void detect_free_equality_singletons(fj_cpu_climber_t<i_t, f_t>& c)
{
  const auto& model = *c.problem;
  const i_t n = model.reverse_offsets.size() - 1, m = model.offsets.size() - 1;
  c.bin_ignore_row.assign(m, 0);
  c.bin_ignore_var.assign(n, 0);
  for (i_t row = 0; row < m; ++row) {
    const f_t rhs = model.cstr_lb[row];
    if (!std::isfinite(rhs) || rhs != model.cstr_ub[row]) continue;
    typename fj_cpu_climber_t<i_t, f_t>::bin_eliminated_row_t rec{row, rhs};
    // A single nonnegative continuous slack can be substituted without fill-in.
    // Keep the original model intact; the encoder applies its bounds and cost.
    i_t singleton = -1;
    for (i_t entry = model.offsets[row]; entry < model.offsets[row + 1]; ++entry) {
      const i_t var = model.variables[entry];
      if (model.h_var_types[var] != var_t::CONTINUOUS) continue;
      const auto bounds = c.h_var_bounds[var].get();
      if (singleton >= 0 || model.reverse_offsets[var + 1] - model.reverse_offsets[var] != 1 ||
          std::abs(model.coefficients[entry]) != f_t{1} || get_lower(bounds) != f_t{0} ||
          std::isfinite(get_upper(bounds)) || !std::isfinite(model.h_obj_coeffs[var])) {
        singleton = -1;
        break;
      }
      singleton = var;
    }
    if (singleton >= 0) {
      c.bin_singletons.emplace_back(row, singleton);
      c.bin_ignore_var[singleton] = 1;
      rec.all.push_back(singleton);
      c.bin_eliminated_rows.push_back(std::move(rec));
      continue;
    }
    bool valid = true, raises = false, lowers = false;
    for (i_t p = model.offsets[row]; p < model.offsets[row + 1]; ++p) {
      const i_t var = model.variables[p];
      const f_t a   = model.coefficients[p];
      if (!a || c.h_is_binary_variable[var]) continue;
      if (c.problem->h_var_types[var] != var_t::CONTINUOUS ||
          model.reverse_offsets[var + 1] - model.reverse_offsets[var] != 1) {
        valid = false;
        break;
      }
      const auto bounds = c.h_var_bounds[var].get();
      const bool up     = (a > 0 && !std::isfinite(get_upper(bounds))) ||
                      (a < 0 && !std::isfinite(get_lower(bounds)));
      const bool down = (a > 0 && !std::isfinite(get_lower(bounds))) ||
                        (a < 0 && !std::isfinite(get_upper(bounds)));
      if (!up && !down) {
        valid = false;
        break;
      }
      rec.all.push_back(var);
      if (up) {
        raises = true;
        rec.positive.push_back(var);
        rec.positive_coeff.push_back(a);
      }
      if (down) {
        lowers = true;
        rec.negative.push_back(var);
        rec.negative_coeff.push_back(a);
      }
    }
    if (!valid || rec.all.empty() || !raises || !lowers) continue;
    c.bin_ignore_row[row] = 1;
    for (i_t var : rec.all)
      c.bin_ignore_var[var] = 1;
    c.bin_eliminated_rows.push_back(std::move(rec));
  }
  c.has_bin_elimination = !c.bin_eliminated_rows.empty();
}

template <typename i_t, typename f_t>
void build_cardinality_index(fj_cpu_climber_t<i_t, f_t>& c, fj_cpu_problem_t<i_t, f_t>& problem)
{
  auto& offsets       = problem.card_row_offsets;
  auto& members       = problem.card_variables;
  auto& cardinalities = problem.card_cardinalities;
  offsets.assign(1, 0);
  members.clear();
  cardinalities.clear();
  problem.card_group_of_variable.assign(problem.n_variables, -1);

  for (i_t row = 0; row < problem.n_constraints; ++row) {
    const f_t lb = problem.cstr_lb[row], ub = problem.cstr_ub[row];
    const i_t begin = problem.offsets[row], end = problem.offsets[row + 1];
    if (!std::isfinite(lb) || !std::isfinite(ub) || std::abs(lb - ub) > 1e-6 || end - begin < 2 ||
        end - begin > 20000)
      continue;

    const f_t common = problem.coefficients[begin];
    if (std::abs(common) <= 1e-6) continue;
    const f_t cardinality = lb / common;
    if (std::abs(cardinality - std::round(cardinality)) > 1e-6 || cardinality < 0 ||
        cardinality > end - begin)
      continue;

    bool valid = true;
    for (i_t p = begin; p < end && valid; ++p) {
      const i_t var = problem.variables[p];
      valid         = c.h_is_binary_variable[var] && std::abs(problem.coefficients[p] - common) <=
                                               1e-6 * std::max<f_t>(1, std::abs(common));
    }
    if (!valid) continue;
    const i_t group = (i_t)offsets.size() - 1;
    for (i_t p = begin; p < end; ++p) {
      const i_t var = problem.variables[p];
      members.push_back(var);
      i_t& owner = problem.card_group_of_variable[var];
      owner      = owner == -1 ? group : (owner == group ? group : -2);
    }
    offsets.push_back((i_t)members.size());
    cardinalities.push_back((i_t)std::llround(cardinality));
  }
}

template <typename i_t, typename f_t>
void certify_epigraph_variables(fj_cpu_climber_t<i_t, f_t>& fj_cpu, i_t n_variables)
{
  fj_cpu.epigraph_push.assign(n_variables, 0);
  fj_cpu.epigraph_vars.clear();

  for (i_t var = 0; var < n_variables; ++var) {
    if (is_integer_var<i_t, f_t>(fj_cpu, var)) continue;
    const f_t obj_coeff = fj_cpu.problem->h_obj_coeffs[var];
    if (obj_coeff == f_t{0}) continue;

    const auto [begin, end] = model_range_for_var<i_t, f_t>(fj_cpu, var);
    if (begin == end) continue;

    // A positive coefficient is minimised by pushing the variable down, so its rows must be the
    // only thing holding it up, and it must be free to rise as far as they demand.
    const bool push_up = obj_coeff > f_t{0};
    const auto bounds  = fj_cpu.h_var_bounds[var].get();
    if (std::isfinite(push_up ? get_upper(bounds) : get_lower(bounds))) continue;

    bool certified = true;
    for (i_t p = begin; p < end && certified; ++p) {
      const i_t row     = fj_cpu.problem->reverse_constraints[p];
      const f_t coeff   = fj_cpu.problem->reverse_coefficients[p];
      const bool has_lb = std::isfinite((f_t)fj_cpu.problem->cstr_lb[row]);
      const bool has_ub = std::isfinite((f_t)fj_cpu.problem->cstr_ub[row]);
      if (coeff == f_t{0}) continue;
      certified = push_up ? ((coeff > 0 && has_lb && !has_ub) || (coeff < 0 && has_ub && !has_lb))
                          : ((coeff > 0 && has_ub && !has_lb) || (coeff < 0 && has_lb && !has_ub));
    }
    if (!certified) continue;

    fj_cpu.epigraph_push[var] = push_up ? 1 : -1;
    fj_cpu.epigraph_vars.push_back(var);
  }
}

template <typename i_t, typename f_t>
void build_one_sided_rows(fj_cpu_climber_t<i_t, f_t>& fj_cpu)
{
  CPUFJ_NVTX_RANGE("CPUFJ::build_one_sided_rows");
  const i_t n_model_rows = fj_cpu.problem->n_constraints;

  i_t n_rows = 0;
  i_t nnz    = 0;
  for (i_t row = 0; row < n_model_rows; ++row) {
    const i_t sides = (i_t)std::isfinite((f_t)fj_cpu.problem->cstr_lb[row]) +
                      (i_t)std::isfinite((f_t)fj_cpu.problem->cstr_ub[row]);
    n_rows += sides;
    nnz += sides * (fj_cpu.problem->offsets[row + 1] - fj_cpu.problem->offsets[row]);
  }
  cuopt_assert(n_rows > 0, "model has no bounded rows");

  auto& fwd_offsets      = fj_cpu.h_offsets;
  auto& fwd_variables    = fj_cpu.h_variables;
  auto& fwd_coefficients = fj_cpu.h_coefficients;
  fwd_offsets.clear();
  fwd_variables.clear();
  fwd_coefficients.clear();
  fwd_offsets.reserve((size_t)n_rows + 1);
  fwd_variables.reserve((size_t)nnz);
  fwd_coefficients.reserve((size_t)nnz);
  fwd_offsets.push_back(0);

  fj_cpu.h_bound.clear();
  fj_cpu.h_bound.reserve((size_t)n_rows);
  fj_cpu.h_row_state.underlying().assign((size_t)n_rows, {});
  fj_cpu.h_row_is_integral.clear();
  fj_cpu.h_row_is_integral.reserve((size_t)n_rows);

  for (i_t row = 0; row < n_model_rows; ++row) {
    const f_t lb           = fj_cpu.problem->cstr_lb[row];
    const f_t ub           = fj_cpu.problem->cstr_ub[row];
    const i_t begin        = fj_cpu.problem->offsets[row];
    const i_t end          = fj_cpu.problem->offsets[row + 1];
    bool integral_activity = true;
    for (i_t k = begin; k < end; ++k) {
      const i_t var   = fj_cpu.problem->variables[k];
      const f_t coeff = fj_cpu.problem->coefficients[k];
      if (coeff != f_t{0} &&
          (fj_cpu.problem->h_var_types[var] != var_t::INTEGER || coeff != std::round(coeff))) {
        integral_activity = false;
        break;
      }
    }

    for (i_t side = 0; side < 2; ++side) {
      const f_t bound = side == 0 ? lb : ub;
      if (!std::isfinite(bound)) continue;
      const f_t sign = side == 0 ? (f_t)-1 : (f_t)1;

      for (i_t k = begin; k < end; ++k) {
        fwd_variables.push_back(fj_cpu.problem->variables[k]);
        fwd_coefficients.push_back(sign * (f_t)fj_cpu.problem->coefficients[k]);
      }
      fwd_offsets.push_back((i_t)fwd_variables.size());

      const i_t r = (i_t)fj_cpu.h_bound.size();
      fj_cpu.h_bound.push_back(sign * bound);
      fj_cpu.h_row_is_integral.push_back(integral_activity && bound == std::round(bound));
      fj_cpu.row_state()[r].weight =
        side == 0 ? fj_cpu.h_initial_left_weights[row] : fj_cpu.h_initial_right_weights[row];
    }
  }
  cuopt_assert((i_t)fj_cpu.h_bound.size() == n_rows, "row count mismatch");
  cuopt_assert((i_t)fj_cpu.h_row_is_integral.size() == n_rows, "row count mismatch");
  cuopt_assert((i_t)fwd_variables.size() == nnz, "nonzero count mismatch");

  fj_cpu.n_rows = n_rows;
  fj_cpu.h_slack_sumcomp.underlying().assign((size_t)n_rows, f_t{0});
  fj_cpu.h_cstr_version.assign((size_t)n_rows, 0);
  fj_cpu.violated_constraints.resize(n_rows);
  fj_cpu.satisfied_constraints.resize(n_rows);
  // Indexed by forward nonzero, so they follow the search rows rather than the model.
  fj_cpu.cached_mtm_moves.assign((size_t)nnz, std::make_pair(f_t{0}, fj_staged_score_t::zero()));
  fj_cpu.cached_mtm_moves_version.assign((size_t)nnz, -1);

  // Counting-sort transpose. Rows are emitted in increasing order, so each variable's slice comes
  // out row-ascending, which the 2-opt merges require.
  const i_t n_variables = fj_cpu.problem->n_variables;
  auto& rev_offsets     = fj_cpu.h_reverse_offsets.underlying();
  rev_offsets.assign((size_t)n_variables + 1, 0);
  for (i_t k = 0; k < nnz; ++k) {
    const i_t var_idx = fwd_variables[k];
    rev_offsets[var_idx + 1]++;
  }
  for (i_t v = 0; v < n_variables; ++v)
    rev_offsets[v + 1] += rev_offsets[v];
  fj_cpu.h_reverse_constraints.resize((size_t)nnz);
  fj_cpu.h_reverse_coefficients.resize((size_t)nnz);
  {
    std::vector<i_t> cursor(rev_offsets.begin(), rev_offsets.begin() + n_variables);
    for (i_t r = 0; r < n_rows; ++r) {
      const i_t row_begin = fwd_offsets[r];
      const i_t row_end   = fwd_offsets[r + 1];
      for (i_t k = row_begin; k < row_end; ++k) {
        const i_t var_idx                   = fwd_variables[k];
        const i_t slot                      = cursor[var_idx]++;
        fj_cpu.h_reverse_constraints[slot]  = r;
        fj_cpu.h_reverse_coefficients[slot] = fwd_coefficients[k];
      }
    }
  }

  recompute_slack(fj_cpu);
  fj_cpu.release_setup_structures();
}

// Eliminate coordinates through exact equalities while retaining each pivot's domain as a row.
// Integer pivots are accepted only when divisibility proves that every lifted value stays integral.
// FJ usually struggles with equality-heavy models since every move may result in equality rows
// being violated and repair having to be applied to many other variables to "compensate". Rewriting
// the problem may help in some cases.
template <typename i_t, typename f_t>
std::unique_ptr<fj_cpu_climber_t<i_t, f_t>> make_equality_reduced_climber(
  fj_cpu_climber_t<i_t, f_t>& c,
  double budget,
  std::vector<fj_equality_substitution_t<i_t, f_t>>& substitutions,
  std::vector<i_t>& retained)
{
  using term_t       = std::pair<i_t, f_t>;
  const auto started = std::chrono::steady_clock::now();
  auto expired       = [&] {
    return c.preemption_flag.load(std::memory_order_relaxed) ||
           std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count() >=
             budget;
  };
  const auto& p = *c.problem;
  std::vector<i_t> equalities;
  for (i_t r = 0; r < p.n_constraints; ++r)
    if (std::isfinite(p.cstr_lb[r]) && p.cstr_lb[r] == p.cstr_ub[r]) equalities.push_back(r);
  if (equalities.empty() || budget <= 0) return nullptr;

  std::vector<std::vector<term_t>> rows(p.n_constraints);
  std::vector<std::vector<i_t>> incidence(p.n_variables);
  std::vector<f_t> lower = p.cstr_lb, upper = p.cstr_ub, objective = p.h_obj_coeffs;
  f_t objective_offset = 0;
  std::vector<uint8_t> eliminated(p.n_variables, 0);
  int64_t nnz = 0;
  for (i_t r = 0; r < p.n_constraints; ++r) {
    auto& row = rows[r];
    for (i_t k = p.offsets[r]; k < p.offsets[r + 1]; ++k)
      row.emplace_back(p.variables[k], p.coefficients[k]);
    std::sort(row.begin(), row.end());
    size_t out = 0;
    for (size_t k = 0; k < row.size();) {
      const i_t v = row[k].first;
      f_t a       = 0;
      do {
        a += row[k++].second;
      } while (k < row.size() && row[k].first == v);
      if (a != 0) {
        row[out++] = {v, a};
        incidence[v].push_back(r);
      }
    }
    row.resize(out);
    nnz += out;
  }
  std::stable_sort(equalities.begin(), equalities.end(), [&](i_t a, i_t b) {
    return rows[a].size() < rows[b].size();
  });

  struct staged_row_t {
    i_t index;
    std::vector<term_t> terms;
    f_t lower;
    f_t upper;
  };
  // Sparse binary scheduling models can encode resource usage as a chain of unit
  // differences. Partial elimination leaves the same equality barrier in place;
  // allow enough fill to expose the cumulative capacity rows on this class only.
  i_t chain_rows = 0;
  if (c.n_binary_vars > p.n_variables / 2 && p.max_var_degree <= 4 &&
      equalities.size() > 0.75 * p.n_constraints) {
    for (i_t r : equalities) {
      if (lower[r] != f_t{0}) continue;
      i_t count = 0;
      f_t sum   = 0;
      bool unit = true;
      for (const auto& [v, a] : rows[r]) {
        if (c.h_is_binary_variable[v]) continue;
        ++count;
        sum += a;
        const auto bounds = c.h_var_bounds[v].get();
        unit &= std::fabs(a) == f_t{1} && objective[v] == f_t{0} && incidence[v].size() <= 2 &&
                std::isfinite(get_lower(bounds)) && std::isfinite(get_upper(bounds));
      }
      if (unit && count == 2 && sum == f_t{0}) ++chain_rows;
    }
  }
  const bool cumulative_chain =
    chain_rows >= 16 && 2 * chain_rows >= p.n_variables - c.n_binary_vars;
  const int64_t fill_limit = (cumulative_chain ? 8 : 2) * std::max<int64_t>(1, nnz);
  for (i_t r : equalities) {
    if (expired()) break;
    const auto& equation = rows[r];
    bool integer_row     = std::isfinite(lower[r]) && lower[r] == std::round(lower[r]) &&
                       std::fabs(lower[r]) <= f_t{1e12};
    int64_t row_gcd = integer_row ? (int64_t)std::fabs(lower[r]) : 0;
    if (integer_row) {
      for (const auto& [v, a] : equation) {
        if (p.h_var_types[v] != var_t::INTEGER || !std::isfinite(a) || a != std::round(a) ||
            std::fabs(a) > f_t{1e12}) {
          integer_row = false;
          break;
        }
        row_gcd = std::gcd(row_gcd, (int64_t)std::fabs(a));
      }
    }

    i_t pivot          = -1;
    f_t divisor        = 0;
    uint64_t best_work = std::numeric_limits<uint64_t>::max();
    for (const auto& [v, a] : equation) {
      const auto bounds = c.h_var_bounds[v].get();
      if (eliminated[v] || c.h_is_binary_variable[v] || get_lower(bounds) == get_upper(bounds))
        continue;
      bool admissible = std::fabs(a) == f_t{1};
      if (p.h_var_types[v] == var_t::INTEGER)
        admissible = integer_row && row_gcd > 0 && std::fabs(a) == (f_t)row_gcd;
      if (!admissible) continue;
      const uint64_t work = (uint64_t)incidence[v].size() * (equation.size() - 1);
      if (work < best_work) {
        pivot     = v;
        divisor   = a;
        best_work = work;
      }
    }
    if (pivot < 0) continue;

    fj_equality_substitution_t<i_t, f_t> sub{pivot, lower[r] / divisor, {}};
    if (!std::isfinite(sub.constant)) continue;
    for (const auto& [v, a] : equation)
      if (v != pivot) sub.terms.emplace_back(v, -a / divisor);
    const auto domain     = c.h_var_bounds[pivot].get();
    const f_t bound_lower = get_lower(domain) - sub.constant;
    const f_t bound_upper = get_upper(domain) - sub.constant;
    if ((std::isfinite(get_lower(domain)) && !std::isfinite(bound_lower)) ||
        (std::isfinite(get_upper(domain)) && !std::isfinite(bound_upper)))
      continue;

    auto affected = incidence[pivot];
    std::sort(affected.begin(), affected.end());
    affected.erase(std::unique(affected.begin(), affected.end()), affected.end());
    std::vector<staged_row_t> staged;
    int64_t next_nnz = nnz - (int64_t)equation.size() + (int64_t)sub.terms.size();
    bool rejected    = false;
    for (i_t row_index : affected) {
      if (row_index == r) continue;
      if (expired()) {
        rejected = true;
        break;
      }
      const auto& old = rows[row_index];
      auto entry      = std::lower_bound(
        old.begin(), old.end(), pivot, [](const term_t& t, i_t v) { return t.first < v; });
      if (entry == old.end() || entry->first != pivot) continue;
      const f_t factor = entry->second;
      staged_row_t replacement{row_index,
                               {},
                               std::fma(-factor, sub.constant, lower[row_index]),
                               std::fma(-factor, sub.constant, upper[row_index])};
      if ((std::isfinite(lower[row_index]) && !std::isfinite(replacement.lower)) ||
          (std::isfinite(upper[row_index]) && !std::isfinite(replacement.upper))) {
        rejected = true;
        break;
      }
      size_t i = 0, j = 0;
      replacement.terms.reserve(old.size() + sub.terms.size());
      while (i < old.size() || j < sub.terms.size()) {
        if (i < old.size() && old[i].first == pivot) {
          ++i;
          continue;
        }
        const i_t v = std::min(i < old.size() ? old[i].first : p.n_variables,
                               j < sub.terms.size() ? sub.terms[j].first : p.n_variables);
        f_t a = 0;
        if (i < old.size() && old[i].first == v) a = old[i++].second;
        if (j < sub.terms.size() && sub.terms[j].first == v)
          a = std::fma(factor, sub.terms[j++].second, a);
        if (!std::isfinite(a) || std::fabs(a) > f_t{1e12}) {
          rejected = true;
          break;
        }
        if (a != 0) replacement.terms.emplace_back(v, a);
      }
      if (rejected) break;
      next_nnz += (int64_t)replacement.terms.size() - (int64_t)old.size();
      if (next_nnz > fill_limit) {
        rejected = true;
        break;
      }
      staged.push_back(std::move(replacement));
    }
    if (rejected) continue;

    for (auto& replacement : staged) {
      auto& old = rows[replacement.index];
      size_t k  = 0;
      for (const auto& [v, a] : replacement.terms) {
        while (k < old.size() && old[k].first < v)
          ++k;
        if (k == old.size() || old[k].first != v) incidence[v].push_back(replacement.index);
      }
      old                      = std::move(replacement.terms);
      lower[replacement.index] = replacement.lower;
      upper[replacement.index] = replacement.upper;
    }
    rows[r]  = sub.terms;
    lower[r] = bound_lower;
    upper[r] = bound_upper;
    for (const auto& [v, a] : sub.terms) {
      objective[v] = std::fma(objective[pivot], a, objective[v]);
      if (!std::isfinite(objective[v])) return nullptr;
    }
    objective_offset = std::fma(objective[pivot], sub.constant, objective_offset);
    if (!std::isfinite(objective_offset)) return nullptr;
    objective[pivot]  = 0;
    eliminated[pivot] = 1;
    incidence[pivot].clear();
    substitutions.push_back(std::move(sub));
    nnz = next_nnz;
  }
  if (substitutions.empty()) return nullptr;

  std::vector<i_t> mapping(p.n_variables, -1), offsets{0}, variables;
  std::vector<f_t> coefficients, row_lower, row_upper, var_lower, var_upper, costs;
  std::vector<var_t> types;
  for (i_t v = 0; v < p.n_variables; ++v) {
    if (eliminated[v]) continue;
    mapping[v] = retained.size();
    retained.push_back(v);
    const auto bounds = c.h_var_bounds[v].get();
    var_lower.push_back(get_lower(bounds));
    var_upper.push_back(get_upper(bounds));
    costs.push_back(objective[v]);
    types.push_back(p.h_var_types[v]);
  }
  for (i_t r = 0; r < p.n_constraints; ++r) {
    if (!std::isfinite(lower[r]) && !std::isfinite(upper[r])) continue;
    if (rows[r].empty()) {
      if (lower[r] > 0 || upper[r] < 0) return nullptr;
      continue;
    }
    for (const auto& [v, a] : rows[r]) {
      cuopt_assert(mapping[v] >= 0, "eliminated column survived substitution");
      variables.push_back(mapping[v]);
      coefficients.push_back(a);
    }
    offsets.push_back(variables.size());
    row_lower.push_back(lower[r]);
    row_upper.push_back(upper[r]);
  }
  if (retained.empty() || row_lower.empty() || coefficients.size() > (size_t)INT32_MAX)
    return nullptr;

  const i_t reduced_nnz  = coefficients.size();
  const i_t reduced_rows = row_lower.size();
  auto child             = init_fj_cpu_from_host_model<i_t, f_t>((i_t)retained.size(),
                                                     reduced_rows,
                                                     reduced_nnz,
                                                     false,
                                                     f_t{1},
                                                     objective_offset,
                                                     std::move(coefficients),
                                                     std::move(variables),
                                                     std::move(offsets),
                                                     std::move(costs),
                                                     std::move(var_lower),
                                                     std::move(var_upper),
                                                     std::move(row_lower),
                                                     std::move(row_upper),
                                                                 {},
                                                                 {},
                                                     std::move(types),
                                                     p.tolerances,
                                                     c.preemption_flag,
                                                     c.settings);
  static_cast<fj_lane_policy_t<i_t, f_t>&>(*child) = static_cast<fj_lane_policy_t<i_t, f_t>&>(c);
  child->use_equality_substitution                 = false;
  child->use_lp_start = child->use_lp_polish = false;
  child->use_bound_prop                      = true;
  child->use_move_batching &= child->n_colors > 0;
  child->log_prefix             = c.log_prefix;
  child->suppress_incumbent_log = true;
  for (i_t j = 0; j < (i_t)retained.size(); ++j) {
    const auto bounds = child->h_var_bounds[j].get();
    f_t value         = c.h_assignment[retained[j]];
    if (is_integer_var(*child, j)) value = std::round(value);
    child->h_assignment[j] = std::clamp(value, get_lower(bounds), get_upper(bounds));
  }
  child->h_best_assignment = child->h_assignment;
  recompute_lhs(*child);
  CUOPT_LOG_DEBUG("%sCPUFJ equality substitution: %zu pivots, %d columns, %d rows, %d nonzeros",
                  c.log_prefix.c_str(),
                  substitutions.size(),
                  (int)retained.size(),
                  (int)reduced_rows,
                  (int)reduced_nnz);
  return child;
}

#if MIP_INSTANTIATE_FLOAT
template void detect_implied_integers<int, float>(fj_cpu_climber_t<int, float>&,
                                                  fj_cpu_problem_t<int, float>&);
template void detect_free_equality_singletons<int, float>(fj_cpu_climber_t<int, float>&);
template void precompute_problem_features<int, float>(fj_cpu_climber_t<int, float>&,
                                                      fj_cpu_problem_t<int, float>&);
template void build_cardinality_index<int, float>(fj_cpu_climber_t<int, float>&,
                                                  fj_cpu_problem_t<int, float>&);
template void certify_epigraph_variables<int, float>(fj_cpu_climber_t<int, float>&, int);
template void build_one_sided_rows<int, float>(fj_cpu_climber_t<int, float>&);
template std::unique_ptr<fj_cpu_climber_t<int, float>> make_equality_reduced_climber<int, float>(
  fj_cpu_climber_t<int, float>&,
  double,
  std::vector<fj_equality_substitution_t<int, float>>&,
  std::vector<int>&);
#endif

#if MIP_INSTANTIATE_DOUBLE
template void detect_implied_integers<int, double>(fj_cpu_climber_t<int, double>&,
                                                   fj_cpu_problem_t<int, double>&);
template void detect_free_equality_singletons<int, double>(fj_cpu_climber_t<int, double>&);
template void precompute_problem_features<int, double>(fj_cpu_climber_t<int, double>&,
                                                       fj_cpu_problem_t<int, double>&);
template void build_cardinality_index<int, double>(fj_cpu_climber_t<int, double>&,
                                                   fj_cpu_problem_t<int, double>&);
template void certify_epigraph_variables<int, double>(fj_cpu_climber_t<int, double>&, int);
template void build_one_sided_rows<int, double>(fj_cpu_climber_t<int, double>&);
template std::unique_ptr<fj_cpu_climber_t<int, double>> make_equality_reduced_climber<int, double>(
  fj_cpu_climber_t<int, double>&,
  double,
  std::vector<fj_equality_substitution_t<int, double>>&,
  std::vector<int>&);
#endif

}  // namespace cuopt::mathematical_optimization::mip
