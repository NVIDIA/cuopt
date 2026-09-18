/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#include "../audit.hpp"
#include "../internal.hpp"
#include "../problem.hpp"
#include "../search/api.hpp"
#include "../setup/bounds.hpp"
#include "starts.hpp"

#include <numeric>

namespace cuopt::mathematical_optimization::mip {

// Minimize residuals in a whitened block of equality directions. Whitening prevents correlated
// rows from overwhelming independent residual directions; all accepted points are still checked
// against the complete, unchanged model.
template <typename i_t, typename f_t>
void apply_affine_equality_start(fj_cpu_climber_t<i_t, f_t>& c, double budget)
{
  phase_timer_t timer(c.t_start);
  const auto started = std::chrono::steady_clock::now();
  auto expired       = [&] {
    return c.preemption_flag.load(std::memory_order_relaxed) ||
           std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count() >=
             budget;
  };
  if (budget <= 0 || c.feasible_found) return;
  const auto& p = *c.problem;
  const i_t n   = p.n_variables;
  if (n <= 0) return;
  const size_t block_size = std::min<size_t>(64, (4u << 20) / (size_t)n);
  if (!block_size) return;
  std::vector<i_t> rows;
  for (i_t r = 0; r < p.n_constraints && rows.size() < block_size; ++r) {
    if (!std::isfinite(p.cstr_lb[r]) || p.cstr_lb[r] != p.cstr_ub[r]) continue;
    for (i_t k = p.offsets[r]; k < p.offsets[r + 1]; ++k) {
      const auto b = c.h_var_bounds[p.variables[k]].get();
      if (get_lower(b) < get_upper(b) && p.coefficients[k] != 0) {
        rows.push_back(r);
        break;
      }
    }
  }
  const size_t q = rows.size();
  if (!q || expired()) return;

  std::vector<f_t> x(n), lower(n), upper(n);
  std::vector<uint8_t> integer(n);
  for (i_t v = 0; v < n; ++v) {
    const auto b = c.h_var_bounds[v].get();
    integer[v]   = is_integer_var(c, v);
    lower[v]     = integer[v] ? std::ceil(get_lower(b)) : get_lower(b);
    upper[v]     = integer[v] ? std::floor(get_upper(b)) : get_upper(b);
    if (lower[v] > upper[v] || !std::isfinite(c.h_assignment[v])) return;
    x[v] = c.h_assignment[v];
  }

  std::vector<f_t> columns((size_t)n * q, 0), scale(q, 0), chol(q * q, 0);
  for (size_t r = 0; r < q; ++r)
    for (i_t k = p.offsets[rows[r]]; k < p.offsets[rows[r] + 1]; ++k) {
      const i_t v = p.variables[k];
      if (lower[v] < upper[v]) columns[(size_t)v * q + r] += p.coefficients[k];
    }
  for (size_t r = 0; r < q; ++r) {
    const auto values = thrust::make_transform_iterator(
      thrust::make_counting_iterator<i_t>(0), [&](i_t v) { return columns[(size_t)v * q + r]; });
    scale[r] = compensated_dot2(values, values, n);
  }
  for (f_t& s : scale) {
    if (!(s > 0) || !std::isfinite(s)) return;
    s = 1 / std::sqrt(s);
  }
  for (i_t v = 0; v < n; ++v) {
    if ((v & 255) == 0 && expired()) return;
    f_t* a = columns.data() + (size_t)v * q;
    for (size_t r = 0; r < q; ++r)
      a[r] *= scale[r];
  }
  for (size_t r = 0; r < q; ++r) {
    const auto row_r = thrust::make_transform_iterator(
      thrust::make_counting_iterator<i_t>(0), [&](i_t v) { return columns[(size_t)v * q + r]; });
    for (size_t s = 0; s <= r; ++s) {
      const auto row_s = thrust::make_transform_iterator(
        thrust::make_counting_iterator<i_t>(0), [&](i_t v) { return columns[(size_t)v * q + s]; });
      chol[r * q + s] = compensated_dot2(row_r, row_s, n);
    }
  }
  for (size_t r = 0; r < q; ++r) {
    chol[r * q + r] += 1e-9;
    for (size_t s = 0; s <= r; ++s) {
      f_t a = chol[r * q + s] - compensated_dot2(chol.data() + r * q, chol.data() + s * q, s);
      if (r == s) {
        if (!(a > 0) || !std::isfinite(a)) return;
        chol[r * q + s] = std::sqrt(a);
      } else {
        chol[r * q + s] = a / chol[s * q + s];
      }
    }
  }
  auto whiten = [&](f_t* a) {
    for (size_t r = 0; r < q; ++r) {
      a[r] -= compensated_dot2(chol.data() + r * q, a, r);
      a[r] /= chol[r * q + r];
    }
  };

  std::vector<f_t> norm(n, 0), gradient(n), error(q), multiplier(q, 0);
  std::vector<i_t> active;
  for (i_t v = 0; v < n; ++v) {
    f_t* a = columns.data() + (size_t)v * q;
    whiten(a);
    norm[v] = compensated_dot2(a, a, q);
    if (!std::isfinite(norm[v])) return;
    if (norm[v] > 0) {
      active.push_back(v);
      x[v] = std::clamp(f_t{0}, lower[v], upper[v]);
    }
  }
  if (active.empty() || expired()) return;

  // Cache only Gram rows actually used by coordinate and exchange moves; a full C'C matrix would
  // be quadratic in the number of columns.
  const size_t cache_size = std::min<size_t>(256, std::max<size_t>(1, (2u << 20) / (size_t)n));
  std::vector<std::vector<f_t>> cache(cache_size);
  std::vector<i_t> cache_column(cache_size, -1), column_slot(n, -1);
  std::vector<uint64_t> age(cache_size, 0);
  uint64_t clock = 0;
  auto gram_row  = [&](i_t u) -> const f_t* {
    i_t slot = column_slot[u];
    if (slot < 0) {
      slot = std::min_element(age.begin(), age.end()) - age.begin();
      if (cache_column[slot] >= 0) column_slot[cache_column[slot]] = -1;
      cache_column[slot] = u;
      column_slot[u]     = slot;
      auto& row          = cache[slot];
      row.assign(n, 0);
      const f_t* a = columns.data() + (size_t)u * q;
      for (i_t v : active) {
        const f_t* b = columns.data() + (size_t)v * q;
        row[v]       = compensated_dot2(a, b, q);
      }
      row[u] = norm[u];
    }
    age[slot] = ++clock;
    return cache[slot].data();
  };
  std::vector<f_t> residual_gradient(n, 0), multiplier_gradient(n, 0);
  auto update_gradient = [&](i_t u, f_t delta) {
    const f_t* gram = gram_row(u);
    for (i_t v : active)
      residual_gradient[v] += delta * gram[v];
  };
  auto refresh = [&] {
    for (size_t r = 0; r < q; ++r) {
      error[r] = (compensated_dot2_csr(p, x, rows[r]) - p.cstr_lb[rows[r]]) * scale[r];
    }
    whiten(error.data());
    for (i_t v : active) {
      const f_t* a           = columns.data() + (size_t)v * q;
      residual_gradient[v]   = compensated_dot2(a, error.data(), q);
      multiplier_gradient[v] = compensated_dot2(a, multiplier.data(), q);
    }
  };

  recompute_lhs(c);
  std::vector<f_t> best(c.h_assignment.begin(), c.h_assignment.end());
  i_t best_count    = c.violated_constraints.size();
  f_t best_severity = -c.total_violations;
  f_t best_metric   = std::numeric_limits<f_t>::infinity();
  cuopt::pcgenerator_t rng((uint64_t)c.settings.seed);
  int stalls = 0;
  refresh();
  for (int iteration = 0; !expired(); ++iteration) {
    if (iteration && iteration % 256 == 0) refresh();
    const f_t metric = compensated_dot2(error.data(), error.data(), error.size());
    if (!std::isfinite(metric)) break;
    if (metric < best_metric) {
      best_metric = metric;
      for (i_t v = 0; v < n; ++v)
        c.h_assignment[v] = (f_t)x[v];
      recompute_lhs(c);
      const i_t count    = c.violated_constraints.size();
      const f_t severity = -c.total_violations;
      if (count < best_count || (count == best_count && severity < best_severity)) {
        best_count    = count;
        best_severity = severity;
        best.assign(c.h_assignment.begin(), c.h_assignment.end());
      }
      if (!count && check_variable_feasibility<i_t, f_t>(c)) {
        best.assign(c.h_assignment.begin(), c.h_assignment.end());
        break;
      }
    }
    if (metric < 1e-26) break;

    i_t chosen       = -1;
    f_t chosen_delta = 0, improvement = -1e-12;
    for (i_t v : active) {
      const f_t g = residual_gradient[v] + multiplier_gradient[v];
      gradient[v] = g;
      f_t delta   = -g / norm[v];
      if (integer[v]) delta = std::round(delta);
      delta          = std::clamp(delta, lower[v] - x[v], upper[v] - x[v]);
      const f_t cost = delta * (2 * g + norm[v] * delta);
      if (std::isfinite(delta) && cost < improvement) {
        improvement  = cost;
        chosen       = v;
        chosen_delta = delta;
      }
    }
    if (chosen >= 0) {
      x[chosen] += chosen_delta;
      const f_t* a = columns.data() + (size_t)chosen * q;
      for (size_t r = 0; r < q; ++r)
        error[r] += chosen_delta * a[r];
      update_gradient(chosen, chosen_delta);
      continue;
    }

    std::vector<i_t> donors;
    for (i_t v : active)
      if (integer[v] && x[v] - 1 >= lower[v]) donors.push_back(v);
    i_t donor = -1, receiver = -1;
    const size_t draws = std::min<size_t>(12, donors.size());
    for (size_t k = 0; k < draws && !expired(); ++k) {
      const size_t j = k + rng.next_u32() % (donors.size() - k);
      std::swap(donors[k], donors[j]);
      const i_t u       = donors[k];
      const f_t* gram   = gram_row(u);
      const f_t removal = norm[u] - 2 * gradient[u];
      for (i_t v : active) {
        if (v == u || !integer[v] || x[v] + 1 > upper[v]) continue;
        const f_t cost = removal + norm[v] + 2 * (gradient[v] - gram[v]);
        if (cost < improvement) {
          improvement = cost;
          donor       = u;
          receiver    = v;
        }
      }
      if (donor >= 0) break;
    }
    if (donor >= 0) {
      x[donor] -= 1;
      x[receiver] += 1;
      const f_t* a = columns.data() + (size_t)donor * q;
      const f_t* b = columns.data() + (size_t)receiver * q;
      for (size_t r = 0; r < q; ++r)
        error[r] += b[r] - a[r];
      update_gradient(donor, -1);
      update_gradient(receiver, 1);
    } else {
      ++stalls;
      for (size_t r = 0; r < q; ++r) {
        multiplier[r] += 0.1 * error[r];
        if (stalls % 100 == 0) multiplier[r] *= 0.5;
      }
      for (i_t v : active) {
        multiplier_gradient[v] += 0.1 * residual_gradient[v];
        if (stalls % 100 == 0) multiplier_gradient[v] *= 0.5;
      }
    }
  }

  std::copy(best.begin(), best.end(), c.h_assignment.begin());
  recompute_lhs(c);
  c.h_best_assignment = c.h_assignment;
  if (c.violated_constraints.empty() && check_variable_feasibility<i_t, f_t>(c)) {
    c.h_best_objective = c.h_incumbent_objective - c.settings.parameters.breakthrough_move_epsilon;
    c.feasible_found   = true;
    report_cpu_incumbent(c);
    share_cpu_incumbent(c);
  }
}

// Matrix-only incumbent construction for thermal unit commitment over a fixed horizon.
//
// Recognition and the final audit read only coefficients, bounds, senses and integrality -- never
// a row or variable name -- so the start survives presolve renaming and reordering. Every structure
// it expects must be proven from the matrix; anything unrecognised declines the whole start rather
// than guessing, and nothing is published until it passes a full original-model audit.
template <typename i_t, typename f_t>
void apply_unit_commitment_start(fj_cpu_climber_t<i_t, f_t>& c)
{
  phase_timer_t timer(c.t_start);
  const auto& p    = *c.problem;
  const i_t n_vars = p.n_variables;
  const i_t n_rows = p.n_constraints;

  constexpr f_t tolerance         = 1e-7;
  constexpr f_t penalty           = 1e6;
  constexpr double start_budget_s = 0.75;
  constexpr size_t min_groups     = 1000;
  constexpr size_t min_products   = 1000;
  constexpr size_t horizon        = 168;

  if (c.n_binary_vars < (i_t)min_groups || c.n_binary_vars == n_vars) return;

  auto is_continuous = [&](i_t v) {
    return p.host_lp && p.host_lp->var_types.size() == (size_t)n_vars
             ? p.host_lp->var_types[v] == simplex::variable_type_t::CONTINUOUS
             : p.h_var_types[v] == var_t::CONTINUOUS;
  };
  auto is_binary         = [&](i_t v) { return c.h_is_binary_variable[v] && !is_continuous(v); };
  auto is_pinned_integer = [&](i_t v) {
    const auto b = c.h_var_bounds[v].get();
    return p.h_var_types[v] == var_t::INTEGER && get_lower(b) == get_upper(b);
  };

  const auto started = std::chrono::steady_clock::now();
  auto expired       = [&] {
    if (c.preemption_flag.load(std::memory_order_relaxed)) return true;
    const auto now = std::chrono::steady_clock::now();
    return std::chrono::duration<double>(now - started).count() > start_budget_s;
  };

  // ---------------------------------------------------------------- variable classification

  std::vector<f_t> base(n_vars);
  std::vector<f_t> x(n_vars);
  std::vector<uint8_t> is_commitment(n_vars);
  std::vector<uint8_t> is_repair_slack(n_vars);

  for (i_t v = 0; v < n_vars; ++v) {
    const auto bounds = c.h_var_bounds[v].get();
    base[v]           = std::clamp(f_t{0}, get_lower(bounds), get_upper(bounds));
    is_commitment[v]  = is_binary(v);

    // A repair slack absorbs a shortfall at a known price: non-negative, penalised in the
    // objective, and entering every one of its rows with a positive coefficient on a row that
    // can only be pushed upwards. Raising it can never break a row it appears in.
    bool slack = is_continuous(v) && p.h_obj_coeffs[v] > 0 && get_lower(bounds) == 0 &&
                 p.reverse_offsets[v] < p.reverse_offsets[v + 1];
    for (i_t q = p.reverse_offsets[v]; slack && q < p.reverse_offsets[v + 1]; ++q) {
      const i_t r = p.reverse_constraints[q];
      slack       = p.reverse_coefficients[q] > 0 &&
              (!std::isfinite(p.cstr_ub[r]) || p.cstr_lb[r] == p.cstr_ub[r]);
    }
    is_repair_slack[v] = slack;
  }

  // ------------------------------------------------------------------------- mode groups
  //
  // A zero-rhs equality holding one negative coefficient and k positive ones of equal magnitude,
  // all binary, defines an aggregate commitment variable over k mutually exclusive operating
  // modes. With k == 1 this degenerates to a plain alias, which is the single-mode case, so one
  // rule covers both formulation styles in this family.

  struct mode_group_t {
    i_t aggregate;
    i_t unit{-1};
    i_t period{-1};
    std::vector<i_t> modes;
  };

  std::vector<mode_group_t> groups;
  std::vector<i_t> group_of(n_vars, -1);

  for (i_t r = 0; r < n_rows; ++r) {
    if (p.cstr_lb[r] != 0 || p.cstr_ub[r] != 0) continue;

    i_t aggregate = -1;
    f_t scale     = 0;
    bool valid    = true;
    std::vector<i_t> modes;

    for (i_t q = p.offsets[r]; q < p.offsets[r + 1]; ++q) {
      const i_t v    = p.variables[q];
      const f_t coef = p.coefficients[q];
      if (coef < 0 && aggregate < 0) {
        aggregate = v;
        scale     = -coef;
      } else if (coef > 0 && is_binary(v)) {
        modes.push_back(v);
      } else {
        valid = false;
        break;
      }
    }
    if (!valid || aggregate < 0 || modes.empty() || group_of[aggregate] >= 0) continue;

    for (i_t q = p.offsets[r]; q < p.offsets[r + 1]; ++q) {
      valid &=
        std::abs(std::abs(p.coefficients[q]) - scale) <= tolerance && group_of[p.variables[q]] < 0;
    }
    if (!valid) continue;

    const i_t g              = (i_t)groups.size();
    group_of[aggregate]      = g;
    is_commitment[aggregate] = 1;
    for (i_t v : modes)
      group_of[v] = g;
    groups.push_back({aggregate, -1, -1, std::move(modes)});
  }
  if (groups.size() < min_groups) return;

  // ------------------------------------------------------- production variables and capacities
  //
  // A two-term zero-rhs row `a*production + b*commitment` with a < 0 < b bounds production by the
  // commitment state: the `<=` form gives production >= (b/-a)*commitment, the `>=` form gives
  // production <= (b/-a)*commitment. Both must be present for the variable to be usable.

  struct production_t {
    i_t var{-1};
    i_t mode{-1};
    i_t period{-1};
    f_t lower_limit{0};
    f_t upper_limit{-1};
    f_t merit{0};
    bool has_lower{false};
  };

  std::vector<production_t> capacity(n_vars);
  std::vector<production_t> products;
  std::vector<i_t> product_of(n_vars, -1);
  std::vector<f_t> mode_minimum(n_vars, 0);

  for (i_t r = 0; r < n_rows; ++r) {
    if (p.offsets[r + 1] - p.offsets[r] != 2) continue;

    i_t production = -1, commitment = -1;
    f_t production_coef = 0, commitment_coef = 0;
    for (i_t q = p.offsets[r]; q < p.offsets[r + 1]; ++q) {
      const i_t v = p.variables[q];
      if (is_commitment[v]) {
        commitment      = v;
        commitment_coef = p.coefficients[q];
      } else if (is_continuous(v)) {
        production      = v;
        production_coef = p.coefficients[q];
      }
    }
    if (production < 0 || commitment < 0) continue;
    if (!(production_coef < 0 && commitment_coef > 0)) continue;

    auto& cap = capacity[production];
    // Two different binaries controlling one production variable is a structure this start does
    // not model; decline rather than pick one.
    if (cap.mode >= 0 && cap.mode != commitment) return;
    cap.var  = production;
    cap.mode = commitment;

    const f_t limit = commitment_coef / -production_coef;
    if (!std::isfinite(p.cstr_lb[r]) && p.cstr_ub[r] == 0) {
      cap.lower_limit = limit;
      cap.has_lower   = true;
    }
    if (!std::isfinite(p.cstr_ub[r]) && p.cstr_lb[r] == 0) { cap.upper_limit = limit; }
  }

  for (i_t v = 0; v < n_vars; ++v) {
    const auto& cap = capacity[v];
    if (!cap.has_lower || cap.upper_limit < cap.lower_limit) continue;

    // A production variable whose controller is not already in a mode group defines a
    // single-mode unit of its own.
    if (group_of[cap.mode] < 0) {
      group_of[cap.mode] = (i_t)groups.size();
      groups.push_back({cap.mode, -1, -1, {cap.mode}});
    }
    capacity[v].merit      = p.h_obj_coeffs[v];
    mode_minimum[cap.mode] = cap.lower_limit;
    product_of[v]          = (i_t)products.size();
    products.push_back(capacity[v]);
  }
  if (products.size() < min_products) return;

  // ------------------------------------------------------------------------- balance rows
  //
  // The per-period demand rows carry every production variable at coefficient exactly +1. Reserve
  // rows carry them at -1 and are therefore excluded by the sign test alone. The objective-sign
  // condition keeps inequality rows that merely happen to sum production out of the period set.

  struct period_t {
    i_t row;
    std::vector<i_t> products;
    std::vector<i_t> groups;
    std::vector<i_t> reserves;
    std::vector<i_t> objective_vars;
  };

  std::vector<period_t> periods;

  for (i_t r = 0; r < n_rows; ++r) {
    if (!std::isfinite(p.cstr_ub[r])) continue;

    std::vector<i_t> entries;
    bool valid = true;
    for (i_t q = p.offsets[r]; q < p.offsets[r + 1]; ++q) {
      const i_t k = product_of[p.variables[q]];
      if (k < 0) continue;
      entries.push_back(k);
      valid &= p.coefficients[q] == 1 &&
               (p.cstr_lb[r] == p.cstr_ub[r] || p.h_obj_coeffs[p.variables[q]] < 0);
    }
    if (!valid || entries.size() < 2) continue;

    const i_t t = (i_t)periods.size();
    periods.push_back({r, entries, {}, {}});
    for (i_t k : entries) {
      // Each production variable belongs to exactly one period, and a unit-period group cannot
      // straddle two. Either violation means the recognition is wrong about this model.
      if (products[k].period >= 0) return;
      products[k].period = t;
      auto& g            = groups[group_of[products[k].mode]];
      if (g.period >= 0 && g.period != t) return;
      g.period = t;
    }
  }
  if (periods.size() != horizon) return;
  for (const auto& prod : products) {
    if (prod.period < 0) return;
  }
  for (i_t g = 0; g < (i_t)groups.size(); ++g) {
    if (groups[g].period >= 0) periods[groups[g].period].groups.push_back(g);
  }
  for (auto& period : periods) {
    for (i_t g : period.groups) {
      period.objective_vars.push_back(groups[g].aggregate);
      period.objective_vars.insert(
        period.objective_vars.end(), groups[g].modes.begin(), groups[g].modes.end());
    }
  }

  // ------------------------------------ units, transition pairs, aliases and reserve rows
  //
  // One pass classifies the remaining row families. Units are the connected components of the
  // commitment variables under rows whose support is entirely commitment variables (plus repair
  // slacks); a component carrying such a row with three or more of them has a genuine
  // multi-period duration coupling, which is what distinguishes this family from a set of
  // independent per-period problems.

  std::vector<i_t> parent(n_vars);
  std::iota(parent.begin(), parent.end(), 0);
  auto find_root = [&](i_t v) {
    while (v != parent[v]) {
      parent[v] = parent[parent[v]];
      v         = parent[v];
    }
    return v;
  };
  auto unite = [&](i_t a, i_t b) { parent[find_root(a)] = find_root(b); };
  for (const auto& g : groups) {
    for (i_t v : g.modes)
      unite(v, g.aggregate);
  }

  struct transition_t {
    i_t row;
    i_t positive;
    i_t negative;
    f_t scale;
  };

  std::vector<transition_t> transitions;
  std::vector<std::pair<i_t, i_t>> aliases;
  std::vector<i_t> repair_slacks;
  for (i_t v = 0; v < n_vars; ++v) {
    if (is_repair_slack[v]) repair_slacks.push_back(v);
  }
  i_t duration_rows = 0;

  for (i_t r = 0; r < n_rows; ++r) {
    i_t first_commitment = -1, n_commitment = 0, n_other = 0, n_pinned = 0;
    i_t positive = -1, negative = -1;
    f_t positive_coef = 0, negative_coef = 0;
    i_t period = -1, n_production = 0;
    bool only_slack_besides_commitment = true;
    bool reserve_shape                 = true;

    for (i_t q = p.offsets[r]; q < p.offsets[r + 1]; ++q) {
      const i_t v    = p.variables[q];
      const f_t coef = p.coefficients[q];

      if (is_commitment[v]) {
        if (first_commitment < 0) first_commitment = v;
        ++n_commitment;
      } else if (is_pinned_integer(v)) {
        if (!is_repair_slack[v]) only_slack_besides_commitment = false;
        ++n_pinned;
      } else {
        if (!is_repair_slack[v]) only_slack_besides_commitment = false;
        ++n_other;
        if (!is_continuous(v)) {
          positive = negative = -2;  // disqualifies the transition and alias shapes below
        } else if (coef > 0 && positive == -1) {
          positive      = v;
          positive_coef = coef;
        } else if (coef < 0 && negative == -1) {
          negative      = v;
          negative_coef = -coef;
        }
      }

      const i_t k = product_of[v];
      if (k >= 0) {
        reserve_shape &= coef == -1 && (period < 0 || period == products[k].period);
        period = products[k].period;
        ++n_production;
      }
    }

    const bool balanced_pair =
      positive >= 0 && negative >= 0 && std::abs(positive_coef - negative_coef) <= tolerance;

    if (only_slack_besides_commitment && n_commitment >= 2) {
      for (i_t q = p.offsets[r]; q < p.offsets[r + 1]; ++q) {
        if (is_commitment[p.variables[q]]) unite(first_commitment, p.variables[q]);
      }
      duration_rows += n_commitment >= 3;
    }
    // Startup/shutdown pair: an equality tying a +a/-a continuous pair to a commitment
    // difference, so the pair is the positive and negative part of that difference.
    if ((n_commitment || n_pinned) && n_other == 2 && balanced_pair &&
        std::isfinite(p.cstr_lb[r]) && p.cstr_lb[r] == p.cstr_ub[r]) {
      transitions.push_back({r, positive, negative, positive_coef});
    }
    if (!n_commitment && !n_pinned && n_other == 2 && balanced_pair && p.cstr_lb[r] == 0 &&
        p.cstr_ub[r] == 0) {
      aliases.emplace_back(positive, negative);
    }
    if (reserve_shape && n_production >= 2 && std::isfinite(p.cstr_lb[r]) &&
        !std::isfinite(p.cstr_ub[r])) {
      periods[period].reserves.push_back(r);
    }
  }

  std::vector<i_t> unit_of(n_vars, -1);
  i_t n_units = 0;
  for (auto& g : groups) {
    const i_t r = find_root(g.aggregate);
    if (unit_of[r] < 0) unit_of[r] = n_units++;
    g.unit = unit_of[r];
  }

  if (!duration_rows || transitions.empty()) return;
  for (const auto& per : periods) {
    if (per.reserves.empty()) return;
  }

  // ------------------------------------------------------------------------ merit ordering

  std::vector<f_t> unit_merit(n_units, 0);
  std::vector<f_t> unit_capacity(n_units, 0);
  std::vector<std::vector<f_t>> unit_merit_coefficients(n_units);
  std::vector<std::vector<f_t>> unit_merit_values(n_units);

  for (auto& prod : products) {
    // Direct-cost models put the marginal cost on production. Epigraph models instead expose
    // c >= k*p + b*u rows; use the cheapest certified line slope for merit ordering.
    if (!(prod.merit > 0)) {
      prod.merit = std::numeric_limits<f_t>::infinity();
      for (i_t q = p.reverse_offsets[prod.var]; q < p.reverse_offsets[prod.var + 1]; ++q) {
        const i_t r = p.reverse_constraints[q];
        if (p.cstr_lb[r] != 0 || std::isfinite(p.cstr_ub[r])) continue;
        if (p.offsets[r + 1] - p.offsets[r] != 3) continue;

        i_t cost_var       = -1;
        f_t cost_coef      = 0;
        bool owns_the_mode = false;
        for (i_t j = p.offsets[r]; j < p.offsets[r + 1]; ++j) {
          const i_t v = p.variables[j];
          if (v == prod.mode) {
            owns_the_mode = true;
          } else if (v != prod.var && is_repair_slack[v]) {
            cost_var  = v;
            cost_coef = p.coefficients[j];
          }
        }
        if (owns_the_mode && cost_var >= 0 && cost_coef > 0) {
          const f_t slope = -p.reverse_coefficients[q] / cost_coef * p.h_obj_coeffs[cost_var];
          prod.merit      = std::min(prod.merit, slope);
        }
      }
      if (!std::isfinite(prod.merit)) prod.merit = 1;
    }

    const i_t u = groups[group_of[prod.mode]].unit;
    unit_merit_coefficients[u].push_back(prod.merit);
    unit_merit_values[u].push_back(prod.upper_limit);
    unit_merit_coefficients[u].push_back(p.h_obj_coeffs[prod.mode]);
    unit_merit_values[u].push_back(f_t{1});
    unit_capacity[u] += prod.upper_limit;
  }
  for (i_t u = 0; u < n_units; ++u) {
    unit_merit[u] = compensated_dot2(unit_merit_coefficients[u].data(),
                                     unit_merit_values[u].data(),
                                     unit_merit_coefficients[u].size());
  }

  std::vector<i_t> order(n_units);
  std::iota(order.begin(), order.end(), 0);
  std::sort(order.begin(), order.end(), [&](i_t a, i_t b) {
    return unit_merit[a] / std::max(f_t{1}, unit_capacity[a]) <
           unit_merit[b] / std::max(f_t{1}, unit_capacity[b]);
  });

  std::vector<std::vector<i_t>> dispatch_order(periods.size());
  for (i_t t = 0; t < (i_t)periods.size(); ++t) {
    dispatch_order[t] = periods[t].products;
    std::stable_sort(dispatch_order[t].begin(), dispatch_order[t].end(), [&](i_t a, i_t b) {
      return products[a].merit < products[b].merit;
    });
  }

  // ---------------------------------------------------------------------------- dispatch

  auto activity = [&](i_t r) { return compensated_dot2_csr(p, x, r); };

  // Fill the period's demand in merit order from the committed units. For a convex separable cost
  // with one balance row and box limits this is the exact optimum of the dispatch subproblem, so
  // the only real decisions left are which units are on and which mode each one runs in.
  auto dispatch = [&](i_t t) {
    auto& per = periods[t];
    for (i_t k : per.products) {
      x[products[k].var] = products[k].lower_limit * x[products[k].mode];
    }

    f_t need = p.cstr_ub[per.row] - activity(per.row);
    for (i_t k : dispatch_order[t]) {
      if (need <= 0) break;
      if (x[products[k].mode] <= 0.5) continue;
      const f_t add = std::min(need, products[k].upper_limit - products[k].lower_limit);
      x[products[k].var] += add;
      need -= add;
    }

    f_t score = penalty * std::max(f_t{0}, -need);  // overgeneration: the row is an equality
    const auto product_merit = thrust::make_transform_iterator(
      per.products.begin(), [&](i_t k) { return products[k].merit; });
    const auto product_value = thrust::make_transform_iterator(
      per.products.begin(), [&](i_t k) { return x[products[k].var]; });
    score += compensated_dot2(product_merit, product_value, per.products.size());
    score += compensated_dot2(
      thrust::make_permutation_iterator(p.h_obj_coeffs.data(), per.objective_vars.data()),
      thrust::make_permutation_iterator(x.data(), per.objective_vars.data()),
      per.objective_vars.size());
    const auto reserve_penalty =
      thrust::make_transform_iterator(per.reserves.begin(), [&](i_t) { return penalty; });
    const auto reserve_deficit = thrust::make_transform_iterator(
      per.reserves.begin(), [&](i_t r) { return std::max(f_t{0}, p.cstr_lb[r] - activity(r)); });
    score += compensated_dot2(reserve_penalty, reserve_deficit, per.reserves.size());
    return score + penalty * std::max(f_t{0}, need);  // unserved demand
  };

  // ------------------------------------------------------------ candidate schedule evaluation

  f_t best_objective = std::numeric_limits<f_t>::infinity();
  bool found         = false;

  auto evaluate = [&](const std::vector<uint8_t>& on) {
    x = base;

    for (const auto& g : groups) {
      i_t chosen     = -1;
      f_t lowest_min = std::numeric_limits<f_t>::infinity();
      for (i_t v : g.modes) {
        x[v] = 0;
        if (!on[g.unit]) continue;
        if (get_upper(c.h_var_bounds[v].get()) < 1) continue;  // unavailable in this period
        if (mode_minimum[v] < lowest_min) {
          lowest_min = mode_minimum[v];
          chosen     = v;
        }
      }
      x[g.aggregate] = chosen >= 0;
      if (chosen >= 0) x[chosen] = 1;
    }

    // Leaving the transition pair at zero is wrong whenever a unit's initial state differs from
    // its committed state, which silently violates the initial-condition rows.
    for (const auto& tr : transitions) {
      x[tr.positive] = x[tr.negative] = 0;
      const f_t delta                 = (p.cstr_lb[tr.row] - activity(tr.row)) / tr.scale;
      x[tr.positive]                  = std::max(f_t{0}, delta);
      x[tr.negative]                  = std::max(f_t{0}, -delta);
    }
    for (int pass = 0; pass < 3; ++pass) {
      for (const auto& a : aliases) {
        x[a.first] = x[a.second] = std::max(x[a.first], x[a.second]);
      }
    }

    f_t score = 0;
    for (i_t t = 0; t < (i_t)periods.size(); ++t) {
      f_t current = dispatch(t);

      // Modes are mutually exclusive alternatives for one committed unit-period. Re-dispatching
      // after a mode swap gives a cheap exact best response against the fixed commitment set.
      for (int pass = 0; pass < 2; ++pass) {
        for (i_t gi : periods[t].groups) {
          const auto& g = groups[gi];
          if (x[g.aggregate] < 0.5 || g.modes.size() < 2) continue;

          i_t chosen = -1;
          for (i_t v : g.modes) {
            if (x[v] > 0.5) chosen = v;
          }
          if (chosen < 0) continue;

          for (i_t v : g.modes) {
            if (v == chosen || get_upper(c.h_var_bounds[v].get()) < 1) continue;
            x[chosen]       = 0;
            x[v]            = 1;
            const f_t trial = dispatch(t);
            if (trial < current - tolerance) {
              chosen  = v;
              current = trial;
            } else {
              x[v]      = 0;
              x[chosen] = 1;
            }
          }
        }
      }
      score += dispatch(t);
    }

    // A repair slack may appear in several rows, so it has to take the largest value any of them
    // demands, not the value one of them does.
    for (int pass = 0; pass < 3; ++pass) {
      for (i_t v : repair_slacks) {
        x[v]         = base[v];
        f_t required = base[v];
        for (i_t q = p.reverse_offsets[v]; q < p.reverse_offsets[v + 1]; ++q) {
          const i_t r = p.reverse_constraints[q];
          if (!std::isfinite(p.cstr_lb[r])) continue;
          const f_t shortfall = p.cstr_lb[r] - activity(r);
          required            = std::max(required, base[v] + shortfall / p.reverse_coefficients[q]);
        }
        x[v] = required;
      }
    }

    // Full audit against the model as the engine holds it: every bound, every integrality
    // restriction, every row. Nothing below this point trusts the construction.
    const f_t objective = compensated_dot2(p.h_obj_coeffs.data(), x.data(), p.h_obj_coeffs.size());
    f_t violation       = 0;
    bool valid          = true;

    for (i_t v = 0; v < n_vars; ++v) {
      const auto bounds = c.h_var_bounds[v].get();
      const f_t excess =
        std::max(f_t{0}, get_lower(bounds) - x[v]) + std::max(f_t{0}, x[v] - get_upper(bounds));
      valid &=
        std::isfinite(x[v]) && excess <= tolerance &&
        (p.h_var_types[v] != var_t::INTEGER || std::abs(x[v] - std::round(x[v])) <= tolerance);
      violation += excess;
    }
    for (i_t r = 0; r < n_rows; ++r) {
      const f_t a      = activity(r);
      const f_t excess = std::max(f_t{0}, p.cstr_lb[r] - a) + std::max(f_t{0}, a - p.cstr_ub[r]);
      valid &= excess <= tolerance;
      violation += excess;
    }

    if (valid && objective < best_objective) {
      std::copy(x.begin(), x.end(), c.h_assignment.begin());
      recompute_lhs(c);
      if (c.violated_constraints.empty() && check_variable_feasibility<i_t, f_t>(c)) {
        best_objective      = objective;
        found               = true;
        c.h_best_assignment = c.h_assignment;
        c.h_best_objective =
          c.h_incumbent_objective - c.settings.parameters.breakthrough_move_epsilon;
        c.feasible_found = true;
        report_cpu_incumbent(c);
        share_cpu_incumbent(c);
      }
    }
    return objective + penalty * violation + score;
  };

  // ------------------------------------------------------------------- commitment search

  const auto anchor = c.h_assignment;
  std::vector<uint8_t> on(n_units, 1);
  std::vector<uint8_t> best_on = on;
  f_t best_score               = std::numeric_limits<f_t>::infinity();

  // Components created by uncoupled periods vary greatly in size. Sweep committed capacity rather
  // than component count so small fragments cannot crowd full-horizon units out of the trial set.
  const f_t total_capacity = std::accumulate(unit_capacity.begin(), unit_capacity.end(), f_t{0});

  constexpr int sweep_steps = 16;
  for (int step = sweep_steps; step >= 0 && !expired(); --step) {
    std::fill(on.begin(), on.end(), 0);
    f_t committed = 0;
    for (i_t j = 0; j < n_units; ++j) {
      const bool take_all = step == sweep_steps;
      if (!take_all && committed >= total_capacity * step / sweep_steps) break;
      on[order[j]] = 1;
      committed += unit_capacity[order[j]];
    }
    const f_t score = evaluate(on);
    if (score < best_score) {
      best_score = score;
      best_on    = on;
    }
  }

  // Once a good threshold is known, expensive capacity is the most valuable to test first.
  on = best_on;
  std::reverse(order.begin(), order.end());
  for (int pass = 0; pass < 2 && !expired(); ++pass) {
    for (i_t u : order) {
      if (expired()) break;
      on[u] ^= 1;
      const f_t score = evaluate(on);
      if (score < best_score - tolerance) {
        best_score = score;
      } else {
        on[u] ^= 1;
      }
    }
  }

  c.h_assignment = found ? c.h_best_assignment : anchor;
  recompute_lhs(c);
}

template <typename i_t, typename f_t>
void apply_structural_completion_start(fj_cpu_climber_t<i_t, f_t>& fj_cpu)
{
  apply_lock_weighted_start<i_t, f_t>(fj_cpu);
  apply_exact_k_start<i_t, f_t>(fj_cpu);
  apply_greedy_covering_start<i_t, f_t>(fj_cpu);
  repair_difficult_anchor<i_t, f_t>(fj_cpu);
}

#if MIP_INSTANTIATE_FLOAT
template void apply_affine_equality_start<int, float>(fj_cpu_climber_t<int, float>&, double);
template void apply_unit_commitment_start<int, float>(fj_cpu_climber_t<int, float>&);
template void apply_structural_completion_start<int, float>(fj_cpu_climber_t<int, float>&);
#endif

#if MIP_INSTANTIATE_DOUBLE
template void apply_affine_equality_start<int, double>(fj_cpu_climber_t<int, double>&, double);
template void apply_unit_commitment_start<int, double>(fj_cpu_climber_t<int, double>&);
template void apply_structural_completion_start<int, double>(fj_cpu_climber_t<int, double>&);
#endif

}  // namespace cuopt::mathematical_optimization::mip
