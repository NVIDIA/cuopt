/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#pragma once

#include <cuopt/mathematical_optimization/mip/heuristics_hyper_params.hpp>

#include <mip_heuristics/logger.hpp>

#include <algorithm>
#include <cmath>
#include <limits>

namespace cuopt::mathematical_optimization::mip {

// Competing hypotheses for how much presolve effort a problem deserves. Each policy maps the
// structural features below onto the same four budgets, so a sweep over policies on a fixed
// instance set isolates the effect of the mapping itself. `legacy` reproduces pre-budget behaviour
// and is the baseline to measure against; `manual` reads the hyper-parameters verbatim so a
// specific point can be pinned from the command line.
enum class presolve_budget_policy_t : int {
  legacy   = 0,
  fixed    = 1,
  size     = 2,
  density  = 3,
  binary   = 4,
  combined = 5,
  manual   = 6,
};

inline const char* presolve_budget_policy_name(int policy)
{
  switch (static_cast<presolve_budget_policy_t>(policy)) {
    case presolve_budget_policy_t::legacy: return "legacy";
    case presolve_budget_policy_t::fixed: return "fixed";
    case presolve_budget_policy_t::size: return "size";
    case presolve_budget_policy_t::density: return "density";
    case presolve_budget_policy_t::binary: return "binary";
    case presolve_budget_policy_t::combined: return "combined";
    case presolve_budget_policy_t::manual: return "manual";
    default: return "unknown";
  }
}

// Dimensions plus cheap structural ratios. Both presolve stages populate this from whatever problem
// representation they hold: Papilo from the original problem before any reduction, the cuOpt
// probing cache from the Papilo-reduced problem. The two therefore see different feature values for
// the same instance, which is intended -- each budget should follow the problem it actually
// operates on.
struct presolve_features_t {
  double n_vars{0};
  double n_cons{0};
  double nnz{0};
  double n_int{0};
  double n_bin{0};
  double max_row_len{0};

  double avg_row_len() const { return n_cons > 0 ? nnz / n_cons : 0.0; }
  double avg_col_len() const { return n_vars > 0 ? nnz / n_vars : 0.0; }
  double density() const { return (n_vars > 0 && n_cons > 0) ? nnz / (n_vars * n_cons) : 0.0; }
  double int_frac() const { return n_vars > 0 ? n_int / n_vars : 0.0; }
  double bin_frac() const { return n_vars > 0 ? n_bin / n_vars : 0.0; }
};

struct presolve_budget_t {
  // <=0 leaves Papilo's own default (unlimited rounds).
  int papilo_max_rounds{-1};
  // <=0 leaves probing.minbadgesize uncapped at max(ncols/2, 32).
  int papilo_max_badgesize{-1};
  // Probing-cache budget in work units: a reproducible count of probing effort, not a time
  // estimate.
  double probing_work_limit{std::numeric_limits<double>::infinity()};
  // Probed variables per step, i.e. the granularity at which the budget can be enforced.
  int probing_step_size{2048};
};

namespace detail {

inline double clamp_d(double v, double lo, double hi) { return std::min(std::max(v, lo), hi); }

inline int clamp_i(double v, int lo, int hi)
{
  return static_cast<int>(clamp_d(std::round(v), lo, hi));
}

}  // namespace detail

template <typename i_t, typename f_t>
presolve_budget_t evaluate_presolve_budget(const mip_heuristics_hyper_params_t<i_t, f_t>& hp,
                                           const presolve_features_t& feat)
{
  using detail::clamp_d;
  using detail::clamp_i;

  presolve_budget_t b{};

  const double nnz   = std::max<double>(feat.nnz, 1.0);
  const double arl   = std::max<double>(feat.avg_row_len(), 1.0);
  const double n_int = std::max<double>(feat.n_int, 1.0);
  const double n_bin = std::max<double>(feat.n_bin, 1.0);
  const double bf    = feat.bin_frac();

  // Lets a single knob scale every derived policy without recompiling: 1.0 at the default of 30.
  const double work_scale = static_cast<double>(hp.cuopt_presolve_work_limit) / 30.0;
  double raw_work         = 30.0;

  switch (static_cast<presolve_budget_policy_t>(hp.presolve_budget_policy)) {
    case presolve_budget_policy_t::legacy:
      b.papilo_max_rounds    = -1;
      b.papilo_max_badgesize = -1;
      b.probing_work_limit   = std::numeric_limits<double>::infinity();
      b.probing_step_size    = 2048;
      return b;

    case presolve_budget_policy_t::fixed:
      b.papilo_max_rounds    = 30;
      b.papilo_max_badgesize = 1024;
      b.probing_work_limit   = 30.0;
      b.probing_step_size    = 512;
      return b;

    case presolve_budget_policy_t::manual:
      b.papilo_max_rounds    = hp.presolve_max_rounds;
      b.papilo_max_badgesize = hp.papilo_probing_max_badgesize;
      b.probing_work_limit   = hp.cuopt_presolve_work_limit;
      b.probing_step_size    = hp.probing_step_size;
      return b;

    // Measured rule. probing.minbadgesize, not the round count, is what drives Papilo's cost, and
    // on wide problems a large badge buys almost nothing: capping it at 32 left the reduced problem
    // bit-identical on square41/supportcase6/nw04/rail507 while cutting presolve 2-26x. On narrow
    // problems the opposite holds (mzzv11, 30n20b8, air05 all reduce measurably worse at 32), so
    // the cap is applied by width only. Rounds are kept non-binding on wide problems -- everything
    // measured saturates well before 50 -- and capped on narrow ones, where mzzv11 keeps growing.
    case presolve_budget_policy_t::size: {
      const bool wide        = feat.n_vars > 2.0e4;
      b.papilo_max_rounds    = wide ? 50 : 20;
      b.papilo_max_badgesize = wide ? 32 : -1;
      raw_work               = 120.0;
      b.probing_step_size    = 512;
      break;
    }

    // A single propagation sweep costs roughly one pass over each touched row, so long rows make
    // every probe more expensive: buy fewer of them, and size the badge so one badge's working
    // limit (~2*nnz in Papilo) stays roughly constant.
    case presolve_budget_policy_t::density:
      b.papilo_max_rounds    = arl <= 10 ? 40 : arl <= 50 ? 20 : 8;
      b.papilo_max_badgesize = clamp_i(2.0e6 / arl, 32, 4096);
      raw_work               = 30.0 * (10.0 / arl);
      b.probing_step_size    = arl <= 50 ? 1024 : 256;
      break;

    // Probing and clique merging only pay off on binaries, so scale with how many there are and how
    // much of the problem they make up.
    case presolve_budget_policy_t::binary:
      b.papilo_max_rounds    = bf >= 0.9 ? 50 : bf >= 0.5 ? 30 : 15;
      b.papilo_max_badgesize = clamp_i(std::max(n_bin / 2.0, 32.0), 32, 1024);
      raw_work               = (2.0 / 3.0) * std::sqrt(n_bin);
      b.probing_step_size    = 512;
      break;

    // Multiplicative over the three effects above, so no single feature can dominate the budget.
    case presolve_budget_policy_t::combined: {
      const double size_f    = clamp_d(1.0e5 / nnz, 0.25, 2.0);
      const double density_f = clamp_d(10.0 / arl, 0.25, 2.0);
      const double binary_f  = clamp_d(0.5 + bf, 0.5, 1.5);
      const double factor    = size_f * density_f * binary_f;
      b.papilo_max_rounds    = clamp_i(30.0 * factor, 5, 60);
      b.papilo_max_badgesize = clamp_i(2.0e6 / arl, 32, 2048);
      raw_work               = 30.0 * factor;
      b.probing_step_size    = clamp_i(512.0 * density_f, 128, 2048);
      break;
    }
  }

  b.probing_work_limit = work_scale * clamp_d(raw_work, 5.0, 120.0);
  return b;
}

// One line per presolve stage carrying the features that went in and the budgets that came out, so
// a sweep can be regressed offline without re-deriving anything from the solver.
inline void log_presolve_budget(const char* stage,
                                int policy,
                                const presolve_features_t& f,
                                const presolve_budget_t& b)
{
  CUOPT_LOG_INFO(
    "PRESOLVE_BUDGET stage=%s policy=%s nvars=%.0f ncons=%.0f nnz=%.0f nint=%.0f nbin=%.0f "
    "arl=%.3f acl=%.3f maxrow=%.0f density=%.3e intfrac=%.3f binfrac=%.3f "
    "rounds=%d badge=%d work=%.3f step=%d",
    stage,
    presolve_budget_policy_name(policy),
    f.n_vars,
    f.n_cons,
    f.nnz,
    f.n_int,
    f.n_bin,
    f.avg_row_len(),
    f.avg_col_len(),
    f.max_row_len,
    f.density(),
    f.int_frac(),
    f.bin_frac(),
    b.papilo_max_rounds,
    b.papilo_max_badgesize,
    b.probing_work_limit,
    b.probing_step_size);
}

}  // namespace cuopt::mathematical_optimization::mip
