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
#include <cstdlib>
#include <exception>
#include <limits>
#include <string>

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

// Benchmark points selected by CUOPT_CONFIG_ID, overriding the policy hyper-parameter so one build
// covers the whole sweep. The 240-instance run conflated two effects -- the Papilo round/badge caps
// and how much of the problem gets probed -- so these span them as a 2x3 factorial: two Papilo
// rules against three probing coverages. Coverage is spaced geometrically rather than evenly
// because truncating probing both won (bab6 at 0.5% of candidates, square41 at 4.3%) and lost
// (30n20b8 at 21.6%, physiciansched3-3 at 9.1%), which puts the crossover somewhere below 10%.
struct presolve_config_t {
  presolve_budget_policy_t papilo_rule;
  double probe_fraction;
};

inline constexpr presolve_config_t presolve_configs[] = {
  {presolve_budget_policy_t::fixed, 1.00},  // 0: Papilo rounds=30/badge=1024, probing unbounded
  {presolve_budget_policy_t::fixed, 0.25},  // 1
  {presolve_budget_policy_t::fixed, 0.05},  // 2
  {presolve_budget_policy_t::size, 1.00},  // 3: Papilo measured wide/narrow rule, probing unbounded
  {presolve_budget_policy_t::size, 0.25},  // 4
  {presolve_budget_policy_t::size, 0.05},  // 5
};
inline constexpr int n_presolve_configs = 6;

// -1 when unset or out of range. Read once; the environment cannot change mid-run.
inline int presolve_config_id()
{
  static const int id = []() -> int {
    const char* raw = std::getenv("CUOPT_CONFIG_ID");
    if (raw == nullptr) { return -1; }
    try {
      const int v = std::stoi(raw);
      if (v < 0 || v >= n_presolve_configs) {
        CUOPT_LOG_WARN("CUOPT_CONFIG_ID=%d is outside [0, %d); ignoring it for presolve budgets",
                       v,
                       n_presolve_configs);
        return -1;
      }
      CUOPT_LOG_INFO("Using presolve budget config %d from CUOPT_CONFIG_ID", v);
      return v;
    } catch (const std::exception& e) {
      CUOPT_LOG_WARN("Failed to parse CUOPT_CONFIG_ID: %s", e.what());
      return -1;
    }
  }();
  return id;
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
  // Coverage the policy asked for, kept only so the log can be compared against what was realised;
  // the two diverge when an instance's probes are dearer than the average the budget assumed.
  double intended_probe_fraction{1.0};
  // The policy that actually ran and the config that selected it, so a log line is attributable
  // even when CUOPT_CONFIG_ID overrode the hyper-parameter.
  int policy{1};
  int config_id{-1};
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

  const double nnz = std::max<double>(feat.nnz, 1.0);
  const double arl = std::max<double>(feat.avg_row_len(), 1.0);
  // Probing candidates are the integers of the problem the probing cache runs on.
  const double n_cand = std::max<double>(feat.n_int, 1.0);
  const double n_bin  = std::max<double>(feat.n_bin, 1.0);
  const double bf     = feat.bin_frac();

  // Probing cost is close to linear in the candidate count -- ~0.055 work units per candidate,
  // measured across the 240-instance benchmark -- and the candidate set is the integers of the
  // reduced problem. A constant budget therefore probes a small problem exhaustively and a large
  // one barely at all: a flat 30 units covered 21% of 30n20b8 but 0.8% of netdiversion, which cost
  // four instances their feasible solution. Policies state the fraction of candidates they want
  // probed instead, converted below with the same cost model the probing loop charges.
  constexpr double avg_iters_per_probe = 3.5;
  const double per_candidate_work =
    (double)hp.probe_host_overhead_work + avg_iters_per_probe * (double)hp.probe_iter_work;
  // Lets a single knob scale every derived policy without recompiling: 1.0 at the default of 30.
  const double work_scale = static_cast<double>(hp.cuopt_presolve_work_limit) / 30.0;
  double probe_fraction   = 1.0;

  // A config id, when set, replaces both the policy and its coverage. Resolved here rather than at
  // each call site so the Papilo stage and the probing stage cannot disagree about which point of
  // the sweep is running.
  const int config  = presolve_config_id();
  const auto policy = config >= 0 ? presolve_configs[config].papilo_rule
                                  : (presolve_budget_policy_t)hp.presolve_budget_policy;
  b.policy          = (int)policy;
  b.config_id       = config;

  switch (policy) {
    case presolve_budget_policy_t::legacy:
      b.papilo_max_rounds    = -1;
      b.papilo_max_badgesize = -1;
      b.probing_work_limit   = std::numeric_limits<double>::infinity();
      b.probing_step_size    = 2048;
      return b;

    // Papilo-only arm: the round and badge caps measured clean against the baseline, so probing is
    // left unbounded here to isolate their effect from the probing budget's.
    case presolve_budget_policy_t::fixed:
      b.papilo_max_rounds    = 30;
      b.papilo_max_badgesize = 1024;
      probe_fraction         = 1.0;
      b.probing_step_size    = 128;
      break;

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
    // The probing fraction is the hypothesis under test here, not a measured value: truncated
    // probing was what won on bab6 (0.5% of candidates), square41 (4.3%) and square47 (2.9%), and
    // what lost on 30n20b8 (21.6%) and physiciansched3-3 (9.1%), so the benchmark has one point per
    // instance and no curve. A quarter sits between those two clusters.
    case presolve_budget_policy_t::size: {
      const bool wide        = feat.n_vars > 2.0e4;
      b.papilo_max_rounds    = wide ? 50 : 20;
      b.papilo_max_badgesize = wide ? 32 : -1;
      probe_fraction         = 0.25;
      b.probing_step_size    = 128;
      break;
    }

    // A single propagation sweep costs roughly one pass over each touched row, so long rows make
    // every probe more expensive: buy fewer of them, and size the badge so one badge's working
    // limit (~2*nnz in Papilo) stays roughly constant.
    case presolve_budget_policy_t::density:
      b.papilo_max_rounds    = arl <= 10 ? 40 : arl <= 50 ? 20 : 8;
      b.papilo_max_badgesize = clamp_i(2.0e6 / arl, 32, 4096);
      probe_fraction         = clamp_d(10.0 / arl, 0.05, 1.0);
      b.probing_step_size    = arl <= 50 ? 256 : 128;
      break;

    // Probing and clique merging only pay off on binaries, so scale with how many there are and how
    // much of the problem they make up.
    case presolve_budget_policy_t::binary:
      b.papilo_max_rounds    = bf >= 0.9 ? 50 : bf >= 0.5 ? 30 : 15;
      b.papilo_max_badgesize = clamp_i(std::max(n_bin / 2.0, 32.0), 32, 1024);
      probe_fraction         = clamp_d(bf, 0.05, 1.0);
      b.probing_step_size    = 128;
      break;

    // Multiplicative over the three effects above, so no single feature can dominate the budget.
    case presolve_budget_policy_t::combined: {
      const double size_f    = clamp_d(1.0e5 / nnz, 0.25, 2.0);
      const double density_f = clamp_d(10.0 / arl, 0.25, 2.0);
      const double binary_f  = clamp_d(0.5 + bf, 0.5, 1.5);
      const double factor    = size_f * density_f * binary_f;
      b.papilo_max_rounds    = clamp_i(30.0 * factor, 5, 60);
      b.papilo_max_badgesize = clamp_i(2.0e6 / arl, 32, 2048);
      probe_fraction         = clamp_d(0.25 * factor, 0.05, 1.0);
      b.probing_step_size    = clamp_i(128.0 * density_f, 64, 512);
      break;
    }
  }

  // A fraction at or above 1 means "probe everything", carried as no budget rather than as a large
  // number: per_candidate_work is an average, so on an instance whose probes are dearer than
  // average a finite budget sized for full coverage would still cut probing short.
  if (config >= 0) { probe_fraction = presolve_configs[config].probe_fraction; }
  probe_fraction *= work_scale;
  b.intended_probe_fraction = std::min(probe_fraction, 1.0);
  b.probing_work_limit      = probe_fraction >= 1.0 ? std::numeric_limits<double>::infinity()
                                                    : probe_fraction * n_cand * per_candidate_work;
  return b;
}

// One line per presolve stage carrying the features that went in and the budgets that came out, so
// a sweep can be regressed offline without re-deriving anything from the solver.
inline void log_presolve_budget(const char* stage,
                                const presolve_features_t& f,
                                const presolve_budget_t& b)
{
  CUOPT_LOG_INFO(
    "PRESOLVE_BUDGET stage=%s config=%d policy=%s nvars=%.0f ncons=%.0f nnz=%.0f nint=%.0f "
    "nbin=%.0f arl=%.3f acl=%.3f maxrow=%.0f density=%.3e intfrac=%.3f binfrac=%.3f "
    "rounds=%d badge=%d work=%.3f step=%d intended_probe_frac=%.4f",
    stage,
    b.config_id,
    presolve_budget_policy_name(b.policy),
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
    b.probing_step_size,
    b.intended_probe_fraction);
}

}  // namespace cuopt::mathematical_optimization::mip
