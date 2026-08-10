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
#include <limits>

namespace cuopt::mathematical_optimization::mip {

// Work the probing loop charges per unit of effort. These are exact counts rather than timings,
// which is what makes the budget reproducible, and probing_work_scale below is expressed against
// them -- changing either invalidates it.
inline constexpr double probing_probe_work = 0.02;  // per probed variable, host overhead
inline constexpr double probing_iter_work  = 0.01;  // per multi-probe propagation iteration

// Probing work allowed per unit of the cost proxy below; dividing by that proxy is what turns it
// into a per-instance work ceiling. It is a work coefficient, and nothing here converts it to
// seconds. Since the wall cap was removed this is the only bound on probing, which is what the
// value was picked to survive: over 240 instances it stopped every run before the 120s wall could
// fire, worst case 44.7s, while looser scales of 4e8 and 1e9 needed that wall on 2 and 8 instances
// and spent 2-3x the total probing time to do it.
//
// Tight enough to bound time is also tight enough to truncate, and that trade is deliberate:
// probing takes its time from branch and bound, and the truncation measured neutral on solution
// quality (11.78 mean error against 12.01, inside the 0.53 run-to-run noise).
//
// A larger scale is not the way to buy coverage back, and that has now been measured rather than
// inferred: raising it 25% reached only 67 of 240 instances (the rest cannot resolve the change at
// step 128 below), spent 23% more probing time on those, and left their median error delta at
// exactly 0.00. The proxy predicts throughput only to within ~340x and one instance (nw04) pins the
// scale under every reshaping tried, including refitting the exponent to the measured nnz^0.65.
// Beyond that the residual is not explained by any structural feature; it needs throughput measured
// during probing rather than predicted from the problem.
inline constexpr double probing_work_scale = 1.5e8;

// Probed variables between work-budget checks, i.e. the granularity at which the budget can be
// enforced. Work is only folded in at the step barrier, so too large a step runs unbudgeted.
inline constexpr int probing_budget_step_size = 128;

// cuOpt forces probing.minbadgesize to ncols/2 to stop Papilo aborting probing on its own work
// budget, so the first badge overshoots that budget in a single pass and the badge cap is the only
// thing left bounding it. How much one badge costs scales with the nonzeros a probing sweep visits,
// n_bin * avg_col_len, so the clamp applies only where that cost is large enough to matter.
//
// Above this threshold the badge stops paying for itself: square47 (2.7e7) went from 40.6s at badge
// 1024 to 8.1s at 32 with a bit-identical reduced problem. Nearer the threshold the reduction is
// worth more than the time -- triptim1 (3.5e5) removes 669 rows at 1024 against 545 at 32, and
// clamping it regressed the instance -- and the same holds below it for mzzv11, 30n20b8 and air05.
//
// Being exempt still means a bounded badge. Leaving it uncapped hands Papilo ncols/2, which on wide
// problems is enormous and costs most of the presolve budget for nothing: rail01 went to badge
// 58763 and Papilo 12.1s -> 49.2s. 1024 keeps the reduction that mattered on triptim1 and mzzv11.
inline constexpr double papilo_badge_cost_threshold = 5.0e5;
inline constexpr int papilo_badge_clamped           = 32;
inline constexpr int papilo_badge_exempt            = 1024;

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
  int probing_step_size{probing_budget_step_size};
};

// Derives both presolve stages' budgets from the problem's dimensions and structure. Rounds and
// badge accept an explicit override from the hyper-parameters; a negative setting asks for the
// measured rule below, and any other value is passed through, where <=0 removes the cap.
template <typename i_t, typename f_t>
presolve_budget_t evaluate_presolve_budget(const mip_heuristics_hyper_params_t<i_t, f_t>& hp,
                                           const presolve_features_t& feat)
{
  presolve_budget_t b{};

  const double nnz = std::max<double>(feat.nnz, 1.0);
  // Probing candidates are the integers of the problem the probing cache runs on.
  const double n_cand = std::max<double>(feat.n_int, 1.0);
  const double n_bin  = std::max<double>(feat.n_bin, 1.0);
  const double acl    = std::max<double>(feat.avg_col_len(), 1.0);

  // Rounds are uncapped. A round cap looks like a cost limit but is not one: triptim1 is
  // bit-identical at 30 rounds and unlimited, while mzzv11 keeps reducing past 30 (1962 rows
  // against 1576, and 0.07 error against 3.41) for 6s more. None of the Papilo blowups measured
  // came from rounds, so there is nothing for the cap to save; the wall ceiling bounds the cost.
  b.papilo_max_rounds = hp.presolve_max_rounds >= 0 ? (int)hp.presolve_max_rounds : -1;

  const double papilo_probe_cost = n_bin * acl;
  b.papilo_max_badgesize =
    hp.papilo_probing_max_badgesize >= 0
      ? (int)hp.papilo_probing_max_badgesize
      : (papilo_probe_cost > papilo_badge_cost_threshold ? papilo_badge_clamped
                                                         : papilo_badge_exempt);

  // Cost of one probing sweep: every propagation touches the rows of the probed column, so the work
  // a second buys falls off with problem size. nnz + n_cand * avg_col_len tracked that better than
  // nnz, avg_row_len or n_cand alone over 660 measured probing runs, and dividing the scale by it
  // bounds the wall time that a pure coverage target cannot: throughput ranged 1.9 to 689 work
  // units per second, so the same budget was worth 360x more time on one instance than another.
  //
  // The ceiling is the only work bound; there is no coverage target alongside it. A coverage
  // fraction cuts every instance by the same proportion whether or not it is the expensive one,
  // which both obscured which of the two stopped a run and penalised the cheap instances. Dropping
  // it left probing running over a second on 102 instances against 81 before, while runs over ten
  // seconds fell from 34 to 11, since the ceiling truncates by cost rather than uniformly.
  const double probing_cost_proxy = nnz + n_cand * acl;
  b.probing_work_limit            = probing_work_scale / probing_cost_proxy;
  b.probing_step_size             = probing_budget_step_size;
  return b;
}

// One line per presolve stage carrying the features that went in and the budgets that came out, so
// a run can be regressed offline without re-deriving anything from the solver.
inline void log_presolve_budget(const char* stage,
                                const presolve_features_t& f,
                                const presolve_budget_t& b)
{
  CUOPT_LOG_DEBUG(
    "PRESOLVE_BUDGET stage=%s nvars=%.0f ncons=%.0f nnz=%.0f nint=%.0f "
    "nbin=%.0f arl=%.3f acl=%.3f maxrow=%.0f density=%.3e intfrac=%.3f binfrac=%.3f "
    "rounds=%d badge=%d work=%.3f step=%d",
    stage,
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
