/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#include "fj_cpu_binary.cuh"
#include "fj_cpu_binary_utils.hpp"

#include "feasibility_jump.cuh"
#include "fj_cpu.cuh"

#include <mip_heuristics/mip_constants.hpp>
#include <mip_heuristics/utils.hpp>

#include <raft/random/rng_device.cuh>

#include <unistd.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <vector>

namespace cuopt::mathematical_optimization::mip {

static const char* fj_binary_reject_name(fj_binary_reject_t reason)
{
  switch (reason) {
    case fj_binary_reject_t::none: return "none";
    case fj_binary_reject_t::empty_problem: return "empty problem";
    case fj_binary_reject_t::non_binary_var: return "non-binary variable";
    case fj_binary_reject_t::fractional_coefficient: return "fractional coefficient";
    case fj_binary_reject_t::coefficient_out_of_range: return "coefficient wider than int16";
    case fj_binary_reject_t::fractional_row_bound: return "fractional row bound";
    case fj_binary_reject_t::row_bound_out_of_range: return "row bound outside int32";
    case fj_binary_reject_t::lhs_headroom: return "row sum|coef| exceeds int32 headroom";
  }
  return "unknown";
}

// work unit proxy. will likely require a lot of tuning
constexpr double fj_bin_bytes_per_nnz = 16.0;
// restarts can help a lot on some smaller combinatorial instances
// 5M leaves a huge margin
constexpr int32_t fj_bin_restart_period     = 5000000;
// DDFW weightin parameters
constexpr int32_t fj_bin_ddfw_transfer      = 1;
constexpr int32_t fj_bin_ddfw_donor_samples = 4;
constexpr int32_t fj_bin_ddfw_escalate_after = 2000;
constexpr int32_t fj_bin_ddfw_escalate_max   = 100;

constexpr int32_t fj_bin_obj_stall_after  = 50;
constexpr int32_t fj_bin_obj_escalate_max = 10;

constexpr int32_t fj_bin_2opt_candidates = 64;

// bounds for the components of the packed score
constexpr int32_t fj_bin_base_limit  = 1 << 16;
constexpr int32_t fj_bin_bonus_limit = 1 << 14;

// tuned values for the argmax tile
// tile size is set relative to L1$ if available
constexpr int32_t fj_bin_argmax_tile_target = 256;
constexpr int32_t fj_bin_argmax_tile_cap_k  = 4;

static int32_t fj_bin_argmax_tile()
{
#ifdef _SC_LEVEL1_DCACHE_SIZE
  long l1 = sysconf(_SC_LEVEL1_DCACHE_SIZE);
#else
  long l1 = 0;
#endif
  if (l1 <= 0) l1 = 32768;  // fallback: 32 KiB, the common x86 L1d
  const int32_t bpv = (int32_t)sizeof(int32_t);
  const int32_t cap = (int32_t)(l1 / (fj_bin_argmax_tile_cap_k * bpv));
  int32_t t         = fj_bin_argmax_tile_target < cap ? fj_bin_argmax_tile_target : cap;
  t &= ~15;  // whole vectors
  return t < 16 ? 16 : t;
}

// the binary-var integer-activities engine.
// sidesteps a lot of the bookkeeping of the general engine by requiring every intermediate result to be integer
template <typename i_t, typename f_t, typename coef_t>
struct fj_bin_engine_t {
  fj_bin_problem_t<coef_t> pb;
  std::vector<int32_t> row_weight;
  std::vector<int32_t> row_slack;

  std::vector<int8_t> assign;
  std::vector<int8_t> best_assign;
  std::shared_ptr<fj_cpu_shared_incumbent_t<i_t, f_t>> shared_incumbent;
  // Staging for an adopted assignment, which arrives as f_t. Sized only when sharing is on.
  std::vector<f_t> adopt_buffer;
  std::vector<int8_t> seed_assign;  // restart target
  std::vector<int32_t> assign_i32;  // gather mirror for the SIMD patch 

  // lowest observed total violation
  int64_t best_infeasible_severity{std::numeric_limits<int64_t>::max()};
  int32_t iters_since_infeasible_improve{0};

  // feasibility component of the score of each var
  std::vector<int64_t> var_score;
  // cached per CSR nnz: last score contribution of variables[k] in its row
  std::vector<int64_t> nnz_score_delta;

  // objective component of the scores
  // inert when in the before-feasibility phase
  std::vector<int64_t> obj_base_score;
  std::vector<int64_t> combined_score;

  fj_bin_tabu_t tabu;

  std::vector<uint8_t> is_violated;
  std::vector<int32_t> violated_list;
  std::vector<int32_t> vpos;
  std::vector<uint8_t> var_bitmap;

  raft::random::PCGenerator rng{0, 0, 0};
  // reusable buffer for the row sampling for moves
  std::vector<int32_t> sample_buf;

  int32_t objective_weight{0};
  int32_t seed_objective_weight{0};
  // Feasible local minima since best_objective last moved, and the value it was last seen at.
  int32_t iterations_at_same_objective{0};
  double last_best_objective{std::numeric_limits<double>::infinity()};
  double incumbent_objective{0};
  double best_objective{std::numeric_limits<double>::infinity()};
  double obj_magnitude{1.0};
  double objective_offset{0};
  int32_t max_weight{1};
  bool feasible_found{false};

  int32_t iters{0};

  int32_t iters_since_best{0};
  int32_t last_restart_iter{0};
  int64_t nnz_touched{0};

  int64_t nnz_patched{0};
  int64_t rows_walked{0};

  int32_t argmax_tile{fj_bin_argmax_tile_target};

  // Settings read at solve entry, where the climber carries populated values.
  int32_t seed{0};
  int32_t tabu_tenure_min{3};
  int32_t tabu_tenure_max{13};
  int32_t perturb_interval{100};
  int32_t mtm_viol_samples{25};

  double breakthrough_margin{1e-4};

  int32_t max_aggregate_base{0};
  int32_t max_aggregate_bonus{0};

  int coefficient_bits() const { return 8 * (int)sizeof(coef_t); }

  void compute_saturation()
  {
    int32_t peak_base = 0, peak_bonus = 0;
    for (int32_t v = 0; v < pb.n_variables; ++v) {
      const int8_t flip = (int8_t)(1 - 2 * assign[v]);
      int32_t agg_base = 0, agg_bonus = 0;
      for (int32_t i = pb.reverse_offsets[v]; i < pb.reverse_offsets[v + 1]; ++i) {
        const int32_t r  = pb.reverse_constraints[i];
        const int32_t os = row_slack[r];
        const int32_t ns = os - (int32_t)pb.reverse_coefficients[i] * flip;
        int32_t base = 0, bonus = 0;
        fj_bin_score_delta_parts(os, ns, row_weight[r], base, bonus);
        agg_base += base;
        agg_bonus += bonus;
      }
      const int32_t abs_base  = agg_base < 0 ? -agg_base : agg_base;
      const int32_t abs_bonus = agg_bonus < 0 ? -agg_bonus : agg_bonus;
      if (abs_base > peak_base) peak_base = abs_base;
      if (abs_bonus > peak_bonus) peak_bonus = abs_bonus;
    }
    max_aggregate_base  = peak_base;
    max_aggregate_bonus = peak_bonus;
  }

  void set_violated(int32_t r)
  {
    if (!is_violated[r]) {
      is_violated[r] = 1;
      vpos[r]        = (int32_t)violated_list.size();
      violated_list.push_back(r);
    }
  }

  void set_satisfied(int32_t r)
  {
    if (is_violated[r]) {
      is_violated[r]     = 0;
      const int32_t p    = vpos[r];
      const int32_t last = violated_list.back();
      violated_list[p]   = last;
      vpos[last]         = p;
      violated_list.pop_back();
      vpos[r] = -1;
    }
  }

  void rebuild_scores()
  {
    std::fill(var_score.begin(), var_score.end(), 0);
    for (int32_t r = 0; r < pb.n_constraints; ++r) {
      const int32_t weight = row_weight[r];
      const int32_t os     = row_slack[r];
      for (int32_t k = pb.offsets[r]; k < pb.offsets[r + 1]; ++k) {
        const int32_t v    = pb.variables[k];
        const int32_t flip = 1 - 2 * assign[v];
        const int32_t ns   = os - (int32_t)pb.coefficients[k] * flip;
        const int64_t p    = fj_bin_packed_score_delta(os, ns, weight);
        nnz_score_delta[k] = p;
        var_score[v] += p;
      }
    }
    nnz_touched += pb.nnz;
  }

  void recompute_slack()
  {
    violated_list.clear();
    std::fill(is_violated.begin(), is_violated.end(), (uint8_t)0);
    for (int32_t r = 0; r < pb.n_constraints; ++r) {
      int32_t lhs = 0;
      for (int32_t k = pb.offsets[r]; k < pb.offsets[r + 1]; ++k)
        lhs += (int32_t)pb.coefficients[k] * assign[pb.variables[k]];
      const int32_t slack = pb.bound[r] - lhs;
      row_slack[r]        = slack;
      if (slack < 0) set_violated(r);
    }
    incumbent_objective = objective_offset;
    for (int32_t v = 0; v < pb.n_variables; ++v)
      incumbent_objective += pb.objective[v] * assign[v];
    nnz_touched += pb.nnz;
    rebuild_scores();
  }

  // Base field of the objective term: the weight, signed by the direction of the gain and scaled by
  // how large that gain is against the model's typical coefficient. Depends only on the variable's
  // own value and the weight, which is what lets a global scan cache it.
  int64_t objective_base(int32_t v, int8_t delta) const
  {
    cuopt_assert(v >= 0 && v < pb.n_variables, "objective base on a column outside engine space");
    const double obj_diff = pb.objective[v] * delta;
    if (obj_diff == 0) return 0;
    cuopt_assert(obj_magnitude > 0, "objective magnitude unit must be positive");
    const double rel = std::fabs(obj_diff) / obj_magnitude;
    const double mult =
      rel < fj_obj_mult_min ? fj_obj_mult_min : (rel > fj_obj_mult_max ? fj_obj_mult_max : rel);
    const double raw = objective_weight * mult;
    cuopt_assert(is_exactly_representable<int32_t>(raw),
                 "scaled objective weight is not an int32");
    const int32_t scaled = (int32_t)std::lround(raw);
    return (int64_t)(obj_diff < 0 ? scaled : -scaled) * fj_bin_score_k;
  }

  int64_t objective_terms(int32_t v, int8_t delta) const
  {
    const double obj_diff = pb.objective[v] * delta;
    int32_t bonus         = 0;
    const bool old_better = incumbent_objective < best_objective;
    const bool new_better = incumbent_objective + obj_diff < best_objective;
    if (!old_better && new_better) {
      bonus += objective_weight;
    } else if (old_better && !new_better) {
      bonus -= objective_weight;
    }
    return objective_base(v, delta) + bonus;
  }

  int64_t flip_objective_base(int32_t v) const
  {
    return objective_base(v, (int8_t)(1 - 2 * assign[v]));
  }

  void update_objective_component()
  {
    for (int32_t v : pb.objective_vars)
      obj_base_score[v] = flip_objective_base(v);
  }

  void set_objective_weight(int32_t new_weight)
  {
    cuopt_assert(new_weight >= 0, "objective weight must be nonnegative");
    objective_weight = new_weight;
    update_objective_component();
  }

  int64_t full_score(int32_t v, int8_t delta) const
  {
    cuopt_assert(v >= 0 && v < pb.n_variables, "score on a column outside engine space");
    if (objective_weight == 0) return var_score[v];
    return var_score[v] + objective_terms(v, delta);
  }

  bool tabu_blocked(int32_t v, bool localmin) const
  {
    cuopt_assert(v >= 0 && v < pb.n_variables, "tabu check on a column outside engine space");
    return tabu.blocked(v, iters, localmin);
  }

  void apply_move(int32_t var, int8_t delta, fj_cpu_climber_t<i_t, f_t>& climber)
  {
    cuopt_assert(var >= 0 && var < pb.n_variables, "move on a column outside engine space");
    const int8_t new_val  = (int8_t)(assign[var] + delta);
    const int8_t new_flip = (int8_t)(1 - 2 * new_val);
    const int32_t ob = pb.reverse_offsets[var], oe = pb.reverse_offsets[var + 1];
    int64_t own_score = 0;

    // walk over rows in tiles, noting which rows require further processing
    // they are handled afterwards
    constexpr int32_t fj_bin_walk_tile = 256;
    int32_t tile_incidence[fj_bin_walk_tile];
    for (int32_t t0 = ob; t0 < oe; t0 += fj_bin_walk_tile) {
      const int32_t t1 = (t0 + fj_bin_walk_tile < oe) ? t0 + fj_bin_walk_tile : oe;
      const int32_t n_tail = fj_bin_walk_rows(row_slack.data(),
                                              pb.reverse_constraints.data(),
                                              pb.reverse_coefficients.data(),
                                              pb.incident_row_cmax.data(),
                                              t0,
                                              t1,
                                              delta,
                                              tile_incidence);
      // handle non-deeply-satisfied rows
      for (int32_t j = 0; j < n_tail; ++j) {
        const int32_t ii        = tile_incidence[j];
        const int32_t r         = pb.reverse_constraints[ii];
        const int32_t weight    = row_weight[r];
        const int32_t skv       = (int32_t)pb.reverse_coefficients[ii];
        const int32_t new_slack = row_slack[r];
        const int32_t old_slack = new_slack + skv * delta;

        // A row can only cross its boundary if the flip moves it by at least the distance to it, so
        // every transition is inside this list and none was lost with the rows the walk absorbed.
        if (new_slack < 0 && old_slack >= 0) {
          set_violated(r);
        } else if (new_slack >= 0 && old_slack < 0) {
          set_satisfied(r);
        }

        // we're in the regime where single flips can affect feasibility. 
        // patch the scores of all incident variables
        const int32_t margin = (int32_t)pb.incident_row_cmax[ii];
        if (!(old_slack < -margin && new_slack < -margin)) {
          const int32_t row_begin = pb.offsets[r], row_end = pb.offsets[r + 1];
          // TODO: check that this may not cause AVX512 powerdown overheads if the AVX2 row/AVX512 row
          // ratio is unbalanced
          fj_bin_patch_row(pb.variables.data(),
                           pb.coefficients.data(),
                           row_begin,
                           row_end,
                           var_score.data(),
                           nnz_score_delta.data(),
                           assign_i32.data(),
                           weight,
                           new_slack,
                           var);
          nnz_touched += row_end - row_begin;
          nnz_patched += row_end - row_begin;
        }

        const int64_t pv = fj_bin_packed_score_delta(new_slack, new_slack - skv * new_flip, weight);
        own_score += pv;
        nnz_score_delta[pb.reverse_to_csr[ii]] = pv;
      }
    }
    nnz_touched += oe - ob;
    rows_walked += oe - ob;

    assign[var]     = new_val;
    assign_i32[var] = new_val;
    var_score[var]  = own_score;
    incumbent_objective += pb.objective[var] * delta;
    if (pb.objective[var] != 0) obj_base_score[var] = flip_objective_base(var);

    // a new best incumbent!
    if (violated_list.empty() && incumbent_objective < best_objective) {
      best_objective   = incumbent_objective;
      best_assign      = assign;
      feasible_found   = true;
      iters_since_best = 0;
      report_incumbent(climber);
    }

    const int32_t tenure =
      tabu_tenure_min + (int32_t)(rng.next_u32() % (uint32_t)(tabu_tenure_max - tabu_tenure_min));
    tabu.on_flip(var, iters, tenure);
  }

  static void uncrush(const fj_cpu_climber_t<i_t, f_t>& climber, ins_vector<f_t>& values)
  {
    for (const auto& rec : climber.bin_eliminated_rows) {
      for (i_t var : rec.all) {
        const auto bounds = climber.h_var_bounds[var];
        values[var]       = isfinite(get_lower(bounds))
                              ? get_lower(bounds)
                              : (isfinite(get_upper(bounds)) ? get_upper(bounds) : f_t{0});
      }
      f_t lhs = 0;
      for (i_t p = climber.problem->offsets[rec.row]; p < climber.problem->offsets[rec.row + 1]; ++p)
        lhs += climber.problem->coefficients[p] * values[climber.problem->variables[p]];
      const f_t residual = rec.rhs - lhs;
      if (residual > 0 && !rec.positive.empty())
        values[rec.positive[0]] += residual / rec.positive_coeff[0];
      else if (residual < 0 && !rec.negative.empty())
        values[rec.negative[0]] += residual / rec.negative_coeff[0];
    }
  }

  void report_incumbent(fj_cpu_climber_t<i_t, f_t>& climber)
  {
    auto& h_assign = climber.h_assignment;
    auto& h_best   = climber.h_best_assignment;
    cuopt_assert(
      (int32_t)h_assign.size() >= pb.n_original && (int32_t)h_best.size() >= pb.n_original,
      "host assignment does not cover the model");
    for (int32_t v = 0; v < pb.n_original; ++v) {
      h_assign[v] = (f_t)pb.var_offset[v];
      h_best[v]   = (f_t)pb.var_offset[v];
    }
    for (int32_t b = 0; b < pb.n_variables; ++b) {
      if (!assign[b]) continue;
      const int32_t v = pb.bit_owner[b];
      h_assign[v] += (f_t)pb.bit_weight[b];
      h_best[v] += (f_t)pb.bit_weight[b];
    }
    if (climber.has_bin_elimination) {
      uncrush(climber, h_assign);
      uncrush(climber, h_best);
    }
    auto objective = [&](const auto& values) {
      f_t result = 0;
      for (i_t var = 0; var < (i_t)climber.problem->h_obj_coeffs.size(); ++var)
        result += climber.problem->h_obj_coeffs[var] * values[var];
      return result;
    };
    const f_t reported = climber.has_bin_elimination ? objective(h_best) : (f_t)best_objective;
    climber.h_incumbent_objective =
      climber.has_bin_elimination ? objective(h_assign) : (f_t)incumbent_objective;
    climber.h_best_objective = reported;
    climber.feasible_found   = true;
    if (shared_incumbent)
      shared_incumbent->publish(reported, climber.get_user_objective(reported), h_best);

    CUOPT_LOG_DEBUG("%sCPUFJ[bin%d] new incumbent: objective %.17g",
                    climber.log_prefix.c_str(),
                    coefficient_bits(),
                    reported);
    if (climber.improvement_callback) {
      const double work_units = climber.work_units_elapsed;
      climber.improvement_callback(reported, h_best, work_units);
    }
  }

  void reweight_constraint(int32_t r, int32_t new_weight)
  {
    if (new_weight == row_weight[r]) return;
    row_weight[r] = new_weight;
    if (new_weight > max_weight) max_weight = new_weight;

    const int32_t row_begin = pb.offsets[r], row_end = pb.offsets[r + 1];
    fj_bin_patch_row(pb.variables.data(),
                     pb.coefficients.data(),
                     row_begin,
                     row_end,
                     var_score.data(),
                     nnz_score_delta.data(),
                     assign_i32.data(),
                     new_weight,
                     row_slack[r],
                     -1);
    nnz_touched += row_end - row_begin;
    nnz_patched += row_end - row_begin;
  }

  // DDFW: every violated row gains weight taken from a satisfied neighbour above the donation
  // floor, so total weight is roughly conserved and differentiation stays local to the hard region.
  // Unit transfers stop moving the landscape on a long stall, so the amount grows with the stall.
  int32_t ddfw_transfer() const
  {
    const int32_t threshold = fj_bin_ddfw_escalate_after;
    if (iters_since_infeasible_improve <= threshold) return fj_bin_ddfw_transfer;
    const int32_t over  = iters_since_infeasible_improve - threshold;
    const int32_t steps = over / threshold + 1;
    const int32_t scale = steps < fj_bin_ddfw_escalate_max ? steps : fj_bin_ddfw_escalate_max;
    return fj_bin_ddfw_transfer * scale;
  }

  // perform DDFW weight updating
  void update_weights()
  {
    const int32_t transfer = ddfw_transfer();
    const int32_t donor_floor = fj_bin_ddfw_init + transfer - 1;

    for (int32_t cf : violated_list) {
      reweight_constraint(cf, row_weight[cf] + transfer);
      const int32_t vo = pb.offsets[cf], ve = pb.offsets[cf + 1];
      if (ve <= vo) continue;
      int32_t best_donor = -1, best_w = donor_floor;
      for (int32_t s = 0; s < fj_bin_ddfw_donor_samples; ++s) {
        const int32_t v  = pb.variables[vo + (int32_t)(rng.next_u32() % (uint32_t)(ve - vo))];
        const int32_t no = pb.reverse_offsets[v], ne = pb.reverse_offsets[v + 1];
        if (ne <= no) continue;
        const int32_t d =
          pb.reverse_constraints[no + (int32_t)(rng.next_u32() % (uint32_t)(ne - no))];
        if (d != cf && !is_violated[d] && row_weight[d] > best_w) {
          best_w     = row_weight[d];
          best_donor = d;
        }
      }
      if (best_donor >= 0) {
        const int32_t donated = row_weight[best_donor] - transfer;
        cuopt_assert(donated >= fj_bin_ddfw_init, "donation broke the weight floor");
        reweight_constraint(best_donor, donated);
      }
    }
    if (violated_list.empty()) {
      if (best_objective < last_best_objective) {
        iterations_at_same_objective = 0;
        last_best_objective          = best_objective;
      } else {
        ++iterations_at_same_objective;
      }
      set_objective_weight(objective_weight + objective_weight_increment());
    }
    track_infeasible_stall();
  }

  // stall-escalation for the objective weight
  int32_t objective_weight_increment() const
  {
    if (iterations_at_same_objective <= fj_bin_obj_stall_after) return 1;
    const int32_t steps =
      1 + (iterations_at_same_objective - fj_bin_obj_stall_after) / fj_bin_obj_stall_after;
    return steps < fj_bin_obj_escalate_max ? steps : fj_bin_obj_escalate_max;
  }

  void reset_infeasible_stall()
  {
    best_infeasible_severity       = std::numeric_limits<int64_t>::max();
    iters_since_infeasible_improve = 0;
  }

  // track the length of the current stall
  void track_infeasible_stall()
  {
    if (violated_list.empty()) {
      reset_infeasible_stall();
      return;
    }

    int64_t severity = 0;
    for (int32_t r : violated_list) {
      cuopt_assert(row_slack[r] < 0, "row in violated_list is not violated");
      severity -= (int64_t)row_slack[r];
    }

    if (severity < best_infeasible_severity) {
      best_infeasible_severity       = severity;
      iters_since_infeasible_improve = 0;
    } else {
      ++iters_since_infeasible_improve;
    }
  }

  // Global argmax over every variable, affordable because var_score is maintained live. While the
  // objective weight is zero the full score is exactly var_score; above zero the sweep runs over
  // var_score plus the cached objective base.
  std::pair<int32_t, int64_t> find_move_global()
  {
    // fastpath if feasibility has never been achieved
    if (objective_weight == 0) {
      // The sweep reads var_score alone; the handful of tabu variables are held at the invalid
      // sentinel across it rather than tested per variable.
      int32_t saved_var[fj_bin_tabu_t::ring_size];
      int64_t saved_score[fj_bin_tabu_t::ring_size];
      const int32_t blocked = tabu.block_tabu(iters, var_score.data(), saved_var, saved_score);

      int32_t v = -1;
      int64_t s = fj_bin_score_invalid;
      fj_bin_argmax(var_score.data(), pb.n_variables, argmax_tile, v, s);

      fj_bin_tabu_t::unblock_tabu(blocked, var_score.data(), saved_var, saved_score);
      return {v, s};
    }
    // path once feasibility has been achieved once
    else {
      // The breakthrough bonus is deliberately absent from the ranking: it depends on
      // incumbent_objective, so no per-variable form of it survives a move, and it occupies the low
      // field where it can only separate variables already tied on the base.
      int64_t* const comb_p = combined_score.data();
      fj_bin_add_scores(var_score.data(), obj_base_score.data(), pb.n_variables, comb_p);

      int32_t saved_var[fj_bin_tabu_t::ring_size];
      int64_t saved_score[fj_bin_tabu_t::ring_size];
      const int32_t blocked = tabu.block_tabu(iters, comb_p, saved_var, saved_score);

      int32_t v = -1;
      int64_t s = fj_bin_score_invalid;
      fj_bin_argmax(comb_p, pb.n_variables, argmax_tile, v, s);

      fj_bin_tabu_t::unblock_tabu(blocked, comb_p, saved_var, saved_score);
      if (v >= 0) s = full_score(v, (int8_t)(1 - 2 * assign[v]));
      return {v, s};
    }
  }

  std::pair<int32_t, int64_t> find_move_in_rows(const std::vector<int32_t>& target_rows,
                                                bool localmin)
  {
    int32_t best_v = -1;
    int64_t best_s = fj_bin_score_invalid;
    for (int32_t r : target_rows) {
      for (int32_t k = pb.offsets[r]; k < pb.offsets[r + 1]; ++k) {
        const int32_t v = pb.variables[k];
        // no collision risks if only a single row is considered
        if (target_rows.size() > 1) {
          if (var_bitmap[v]) continue;
          var_bitmap[v] = 1;
        }
        if (tabu_blocked(v, localmin)) continue;
        const int64_t s = full_score(v, (int8_t)(1 - 2 * assign[v]));
        if (s > best_s) {
          best_s = s;
          best_v = v;
        }
      }
    }
    // Restore the all-zero invariant by revisiting only what was set: the sampled rows hold a few
    // dozen variables against n in the thousands, so this is far cheaper than clearing the array.
    // the var_bitmap array isn't useful when only a single row is considered
    if (target_rows.size() > 1) {
      for (int32_t r : target_rows) {
        for (int32_t k = pb.offsets[r]; k < pb.offsets[r + 1]; ++k)
          var_bitmap[pb.variables[k]] = 0;
      }
    }
    return {best_v, best_s};
  }

  std::pair<int32_t, int64_t> find_move_violated(int32_t sample_size, bool localmin)
  {
    const int32_t n                     = (int32_t)violated_list.size();
    const std::vector<int32_t>* sampled = &violated_list;
    if (n > sample_size) {
      sample_buf.clear();
      for (int32_t i = 0; i < sample_size; ++i) {
        sample_buf.push_back(violated_list[rng.next_u32() % (uint32_t)n]);
      }
      sampled = &sample_buf;
    }
    auto move = find_move_in_rows(*sampled, localmin);

    // Breakthrough moves: once a feasible solution exists, allow objective-driven jumps.
    if (feasible_found && incumbent_objective >= best_objective + breakthrough_margin) {
      for (int32_t v : pb.objective_vars) {
        const double step = (best_objective - incumbent_objective) / pb.objective[v];
        double target =
          pb.objective[v] > 0 ? std::floor(assign[v] + step) : std::ceil(assign[v] + step);
        if (target < 0) target = 0;
        if (target > 1) target = 1;
        if ((int8_t)target == assign[v]) continue;
        if (tabu_blocked(v, false)) continue;
        const int64_t s = full_score(v, (int8_t)((int8_t)target - assign[v]));
        if (s > move.second) move = {v, s};
      }
    }
    return move;
  }

  // True when flipping both variables leaves every row they touch satisfied. Both reverse ranges
  // are row-ascending, so shared rows are handled jointly by merging them.
  bool paired_flip_keeps_feasible(int32_t var1, int8_t delta1, int32_t var2, int8_t delta2) const
  {
    int32_t i = pb.reverse_offsets[var1], ie = pb.reverse_offsets[var1 + 1];
    int32_t j = pb.reverse_offsets[var2], je = pb.reverse_offsets[var2 + 1];

    while (i < ie || j < je) {
      const int32_t r1 = i < ie ? pb.reverse_constraints[i] : INT32_MAX;
      const int32_t r2 = j < je ? pb.reverse_constraints[j] : INT32_MAX;
      const int32_t r  = r1 < r2 ? r1 : r2;

      int32_t change = 0;
      if (r1 == r) change += (int32_t)pb.reverse_coefficients[i++] * delta1;
      if (r2 == r) change += (int32_t)pb.reverse_coefficients[j++] * delta2;
      if (row_slack[r] - change < 0) return false;
    }
    return true;
  }

  // look for objective-improving 2opt flips on the current assignment
  std::pair<std::pair<int32_t, int32_t>, int64_t> find_lift_2opt_move()
  {
    cuopt_assert(violated_list.empty(), "lift moves require a feasible incumbent");

    std::pair<int32_t, int32_t> best_pair = {-1, -1};
    int64_t best_s                        = 0;
    double best_improvement               = 0;
    if (pb.objective_vars.empty()) return {best_pair, best_s};

    const uint32_t n_obj = (uint32_t)pb.objective_vars.size();
    const int32_t n_draws =
      n_obj < (uint32_t)fj_bin_2opt_candidates ? (int32_t)n_obj : fj_bin_2opt_candidates;

    for (int32_t t = 0; t < n_draws; ++t) {
      const int32_t var1  = pb.objective_vars[rng.next_u32() % n_obj];
      const int8_t delta1 = (int8_t)(1 - 2 * assign[var1]);
      if ((double)delta1 * pb.objective[var1] >= 0) continue;
      if (tabu_blocked(var1, false)) continue;

      // Only pairs are useful here: a flip breaking nothing is already the single-flip lift's job,
      // and one breaking several rows cannot be repaired by a single companion.
      int32_t broken = -1;
      bool multiple  = false;
      for (int32_t i = pb.reverse_offsets[var1]; i < pb.reverse_offsets[var1 + 1] && !multiple;
           ++i) {
        const int32_t r = pb.reverse_constraints[i];
        if (row_slack[r] - (int32_t)pb.reverse_coefficients[i] * delta1 < 0) {
          if (broken >= 0)
            multiple = true;
          else
            broken = r;
        }
      }
      if (multiple || broken < 0) continue;

      for (int32_t k = pb.offsets[broken]; k < pb.offsets[broken + 1]; ++k) {
        const int32_t var2 = pb.variables[k];
        if (var2 == var1) continue;

        const int8_t delta2 = (int8_t)(1 - 2 * assign[var2]);
        const double combined =
          (double)delta1 * pb.objective[var1] + (double)delta2 * pb.objective[var2];
        if (combined >= 0) continue;
        if (tabu_blocked(var2, false)) continue;
        if (!paired_flip_keeps_feasible(var1, delta1, var2, delta2)) continue;

        const double improvement = -combined;
        if (improvement > best_improvement) {
          best_improvement = improvement;
          best_s           = 1;  // sign only, never compared against another operator's score
          best_pair        = {var1, var2};
        }
      }
    }
    cuopt_assert((best_pair.first < 0) == (best_improvement <= 0),
                 "pair and score must agree on whether a move was found");
    return {best_pair, best_s};
  }

  std::pair<int32_t, int64_t> find_lift_move() const
  {
    cuopt_assert(violated_list.empty(), "lift moves require a feasible incumbent");

    int32_t best_v          = -1;
    int64_t best_s          = 0;
    double best_improvement = 0;
    for (int32_t v : pb.objective_vars) {
      const int8_t delta = (int8_t)(1 - 2 * assign[v]);
      if ((double)delta * pb.objective[v] >= 0) continue;
      if (tabu_blocked(v, false)) continue;
      // Base field is zero iff the flip breaks no row; K/2 splits it while |bonus| < 2^31.
      if (var_score[v] <= -(fj_bin_score_k / 2)) continue;
      const double improvement = -pb.objective[v] * (double)delta;
      if (improvement > best_improvement) {
        best_improvement = improvement;
        best_s           = 1;
        best_v           = v;
      }
    }
    cuopt_assert((best_v < 0) == (best_improvement <= 0),
                 "move and score must agree on whether a move was found");
    return {best_v, best_s};
  }

  void perturb()
  {
    if (pb.objective_vars.empty()) return;
    if (feasible_found) {
      cuopt_assert((int32_t)best_assign.size() == pb.n_variables, "incumbent size mismatch");
      assign = best_assign;

      if (!pb.encoded && shared_incumbent &&
          shared_incumbent->adopt((f_t)best_objective, adopt_buffer)) {
        for (int32_t v = 0; v < pb.n_variables; ++v)
          assign[v] = (int8_t)(adopt_buffer[pb.bit_owner[v]] >= 0.5 ? 1 : 0);
      }
      for (int32_t v = 0; v < pb.n_variables; ++v)
        assign_i32[v] = assign[v];
    }
    const uint32_t n = (uint32_t)pb.objective_vars.size();
    for (int i = 0; i < 2; ++i) {
      const int32_t v = pb.objective_vars[rng.next_u32() % n];
      assign[v]       = (int8_t)(rng.next_u32() & 1u);
      assign_i32[v]   = assign[v];
    }
    recompute_slack();
    update_objective_component();
  }

  // Restart returns the assignment to the seed the climber was constructed with
  void do_restart()
  {
    assign = seed_assign;
    for (int32_t v = 0; v < pb.n_variables; ++v)
      assign_i32[v] = assign[v];
    for (int32_t r = 0; r < pb.n_constraints; ++r)
      row_weight[r] = pb.initial_weight[r];
    max_weight = fj_bin_ddfw_init;
    set_objective_weight(seed_objective_weight);
    reset_infeasible_stall();
    tabu.clear(iters);
    recompute_slack();
    last_restart_iter = iters;
    // The restarted walk gets a full window before the stall gate can perturb it.
    iters_since_best = 0;
  }

  void init(fj_cpu_climber_t<i_t, f_t>& climber)
  {
    phase_timer_t timer(climber.bin_setup.engine_init);
    const auto& params  = climber.settings.parameters;
    seed                = climber.settings.seed;
    rng                 = raft::random::PCGenerator((uint64_t)seed, 0, 0);
    tabu_tenure_min     = params.tabu_tenure_min;
    tabu_tenure_max     = params.tabu_tenure_max;
    breakthrough_margin = params.breakthrough_move_epsilon;
    perturb_interval    = climber.perturb_interval;
    mtm_viol_samples    = climber.mtm_viol_samples;

    if (tabu_tenure_max <= tabu_tenure_min) tabu_tenure_max = tabu_tenure_min + 1;

    cuopt_assert(tabu_tenure_max <= fj_bin_tabu_t::max_tenure,
                 "tabu tenure exceeds the tabu ring, live entries would be evicted");
    if (tabu_tenure_max > fj_bin_tabu_t::max_tenure) tabu_tenure_max = fj_bin_tabu_t::max_tenure;

    const int32_t n_cols = pb.n_variables, n_rows = pb.n_constraints;
    const auto& h_assign = climber.h_assignment;
    assign.assign(n_cols, 0);
    // we encoded integers as binary vars, crush the assignment
    if (pb.encoded) {
      // Descending weight, so the bit pattern reproduces the start value wherever it is
      // representable: with exact closure that is every integer of the domain.
      std::vector<std::vector<int32_t>> bits_of(pb.n_original);
      for (int32_t b = 0; b < n_cols; ++b)
        bits_of[pb.bit_owner[b]].push_back(b);
      for (int32_t v = 0; v < pb.n_original; ++v) {
        long residual = std::lround((double)h_assign[v] - pb.var_offset[v]);
        if (residual < 0) residual = 0;
        auto& bits = bits_of[v];
        std::sort(bits.begin(), bits.end(), [&](int32_t a, int32_t b) {
          return pb.bit_weight[a] > pb.bit_weight[b];
        });
        for (int32_t b : bits) {
          const long w = std::lround(pb.bit_weight[b]);
          if (w <= residual) {
            assign[b] = 1;
            residual -= w;
          }
        }
        cuopt_assert(residual == 0, "greedy bit encode left the start value unrepresented");
      }
    } else {
      for (int32_t j = 0; j < n_cols; ++j) {
        const double val = (double)h_assign[pb.bit_owner[j]];
        assign[j]        = (int8_t)(val >= 0.5 ? 1 : 0);
      }
    }
    seed_assign      = assign;
    best_assign      = assign;
    shared_incumbent = climber.shared_incumbent;
    if (shared_incumbent) adopt_buffer.assign(pb.n_original, 0);
    reset_infeasible_stall();
    assign_i32.assign(n_cols, 0);
    for (int32_t v = 0; v < n_cols; ++v)
      assign_i32[v] = assign[v];

    row_weight.assign(pb.initial_weight.begin(), pb.initial_weight.end());
    row_slack.assign(n_rows, 0);

    var_score.assign(n_cols, 0);
    nnz_score_delta.assign(pb.nnz + fj_bin_simd_padding, 0);
 
    obj_base_score.assign(n_cols, 0);
    combined_score.assign(n_cols, 0);
    tabu.resize(n_cols);
    is_violated.assign(n_rows, 0);
    vpos.assign(n_rows, -1);
    violated_list.clear();
    var_bitmap.assign(n_cols, 0);

    const int32_t seeded_weight = (int32_t)std::lround(climber.h_objective_weight);
    cuopt_assert(seeded_weight >= 0, "objective weight should be positive or zero");

    double abs_obj_sum = 0;
    for (int32_t v : pb.objective_vars)
      abs_obj_sum += std::fabs(pb.objective[v]);
    obj_magnitude = abs_obj_sum > 0 ? abs_obj_sum / (double)pb.objective_vars.size() : 1.0;
    cuopt_assert(std::isfinite(obj_magnitude) && obj_magnitude > 0,
                 "objective magnitude unit must be finite and positive");

    objective_offset = 0;
    for (int32_t v = 0; v < pb.n_original; ++v)
      objective_offset += pb.orig_objective[v] * pb.var_offset[v];

    argmax_tile = fj_bin_argmax_tile();
    set_objective_weight(seeded_weight > 0 ? seeded_weight : 0);
    seed_objective_weight        = objective_weight;
    max_weight                   = fj_bin_ddfw_init;
    incumbent_objective          = 0;
    best_objective               = std::numeric_limits<double>::infinity();
    last_best_objective          = std::numeric_limits<double>::infinity();
    iterations_at_same_objective = 0;
    feasible_found               = false;
    iters                        = 0;
    iters_since_best             = 0;
    last_restart_iter            = 0;
    recompute_slack();
  }

  void solve(fj_cpu_climber_t<i_t, f_t>& climber, f_t time_limit, double work_unit_limit)
  {
    init(climber);
    if (violated_list.empty()) {
      best_objective = incumbent_objective;
      best_assign    = assign;
      feasible_found = true;
      report_incumbent(climber);
      set_objective_weight(
        std::max(objective_weight, (int32_t)std::lround((double)climber.seed_objective_weight)));
    }

    const auto loop_start = std::chrono::high_resolution_clock::now();
    const auto limit = std::chrono::milliseconds((int64_t)std::floor((double)time_limit * 1000.0));
    const bool bounded_time = std::isfinite((double)time_limit);

    while (!climber.halted && !climber.preemption_flag.load()) {
      if (bounded_time && std::chrono::high_resolution_clock::now() - loop_start > limit) break;
      if (iters >= climber.settings.iteration_limit) break;
      if (iters - last_restart_iter >= fj_bin_restart_period) do_restart();
      tabu.maybe_rebase(iters);

      int32_t move_var                  = -1;
      int64_t score                     = fj_bin_score_invalid;
      std::pair<int32_t, int32_t> pair2 = {-1, -1};
      // look for objective improving moves when feasible
      if (violated_list.empty()) {
        std::tie(move_var, score) = find_lift_move();
        // Pairs are only reachable once no single improving flip preserves feasibility.
        if (score <= 0) {
          int64_t pair_score;
          std::tie(pair2, pair_score) = find_lift_2opt_move();
          if (pair_score > 0) score = pair_score;
        }
      }
      if (pair2.first < 0 && score <= 0) std::tie(move_var, score) = find_move_global();

      bool perturb_now = false;
      if (violated_list.empty() && iters_since_best > perturb_interval) {
        perturb_now = true;
        // Without this the counter stays above the interval and every later iteration perturbs.
        iters_since_best = 0;
      }

      // we found an objective-improving 2opt!
      if (pair2.first >= 0 && !perturb_now) {
        apply_move(pair2.first, (int8_t)(1 - 2 * assign[pair2.first]), climber);
        apply_move(pair2.second, (int8_t)(1 - 2 * assign[pair2.second]), climber);
      } else if (score > 0 && move_var >= 0 && !perturb_now) {
        apply_move(move_var, (int8_t)(1 - 2 * assign[move_var]), climber);
      } else {
        // Local minimum: bump the weights so the landscape moves, then take the best flip in one
        // randomly drawn violated row. Applied whatever its score, which is what breaks the basin.
        update_weights();
        if (perturb_now) perturb();
        std::tie(move_var, score) = find_move_violated(1, true);
        const int32_t var         = move_var >= 0 ? move_var : 0;
        apply_move(var, (int8_t)(1 - 2 * assign[var]), climber);
      }

      if (iters % climber.log_interval == 0) {
        CUOPT_LOG_DEBUG("%sCPUFJ[bin%d] iteration: %d, viol: %zu, best: %g, maxw: %d",
                        climber.log_prefix.c_str(),
                        coefficient_bits(),
                        iters,
                        violated_list.size(),
                        best_objective,
                        max_weight);
      }
      if (iters % climber.diversity_callback_interval == 0 && climber.diversity_callback) {
        auto& h_assign = climber.h_assignment;
        for (int32_t v = 0; v < pb.n_original; ++v)
          h_assign[v] = (f_t)pb.var_offset[v];
        for (int32_t b = 0; b < pb.n_variables; ++b)
          if (assign[b]) h_assign[pb.bit_owner[b]] += (f_t)pb.bit_weight[b];
        climber.diversity_callback((f_t)incumbent_objective, h_assign);
      }

      // Work-unit proxy. nnz_touched is cumulative, reproducing the accumulation shape the general
      // path gets from its cumulative byte counters.
      if (iters % 100 == 0 && iters > 0) {
        const double work =
          (double)nnz_touched * fj_bin_bytes_per_nnz * climber.work_unit_bias / 1e10;
        climber.work_units_elapsed.store(work, std::memory_order_release);
        if (climber.producer_sync != nullptr) climber.producer_sync->notify_progress();
        if (work >= work_unit_limit) break;
      }

      ++iters;
      ++iters_since_best;
    }

    compute_saturation();
    climber.iterations = (i_t)iters;
    CUOPT_LOG_DEBUG(
      "%sCPUFJ[bin%d] done: %d iterations, best %g, max weight %d, aggregate base %d/%d, bonus "
      "%d/%d",
      climber.log_prefix.c_str(),
      coefficient_bits(),
      iters,
      best_objective,
      max_weight,
      max_aggregate_base,
      fj_bin_base_limit,
      max_aggregate_bonus,
      fj_bin_bonus_limit);
    CUOPT_LOG_DEBUG("%sCPUFJ[bin%d] work: nnz_patched %lld, rows_walked %lld",
                    climber.log_prefix.c_str(),
                    coefficient_bits(),
                    (long long)nnz_patched,
                    (long long)rows_walked);
  }
};

template <typename i_t, typename f_t>
bool try_cpufj_binary_solve(fj_cpu_climber_t<i_t, f_t>& climber,
                            f_t time_limit,
                            double work_unit_limit)
{
  static const bool disabled = std::getenv("CUOPT_NO_BINFJ") != nullptr;
  if (disabled || climber.low_latency) return false;

  const fj_bin_scan_t scan = fj_bin_scan(climber, climber.bin_setup);
  if (scan.reject != fj_binary_reject_t::none) {
    if (scan.reject == fj_binary_reject_t::non_binary_var && climber.use_integer_bit_encoding) {
      fj_bin_engine_t<i_t, f_t, int16_t> probe;
      int bits = 0;
      if (fj_bin_encode(climber, probe.pb, bits, climber.bin_setup)) {
        if (bits == 8) {
          fj_bin_engine_t<i_t, f_t, int8_t> engine8;
          int bits8 = 0;
          if (fj_bin_encode(climber, engine8.pb, bits8, climber.bin_setup)) {
            CUOPT_LOG_DEBUG("%sCPUFJ binary fast path enabled (encoded int8): %d bits, %d rows",
                            climber.log_prefix.c_str(),
                            engine8.pb.n_variables,
                            engine8.pb.n_constraints);
            engine8.solve(climber, time_limit, work_unit_limit);
            return true;
          }
        }
        CUOPT_LOG_DEBUG("%sCPUFJ binary fast path enabled (encoded int16): %d bits, %d rows",
                        climber.log_prefix.c_str(),
                        probe.pb.n_variables,
                        probe.pb.n_constraints);
        probe.solve(climber, time_limit, work_unit_limit);
        return true;
      }
    }
    CUOPT_LOG_DEBUG("%sCPUFJ binary fast path declined: %s (row %d, var %d)",
                    climber.log_prefix.c_str(),
                    fj_binary_reject_name(scan.reject),
                    scan.bad_row,
                    scan.bad_var);
    return false;
  }

  auto run = [&](auto& engine) -> bool {
    fj_bin_narrow(climber, scan, engine.pb, climber.bin_setup);
    CUOPT_LOG_DEBUG(
      "%sCPUFJ binary fast path enabled: int%d coefficients, %d rows after one-sided split",
      climber.log_prefix.c_str(),
      scan.coefficient_bits,
      scan.n_split_constraints);
    engine.solve(climber, time_limit, work_unit_limit);
    return true;
  };

  if (scan.coefficient_bits == 8) {
    fj_bin_engine_t<i_t, f_t, int8_t> engine;
    return run(engine);
  }
  fj_bin_engine_t<i_t, f_t, int16_t> engine;
  return run(engine);
}

#if MIP_INSTANTIATE_FLOAT
template bool try_cpufj_binary_solve(fj_cpu_climber_t<int, float>& climber,
                                     float time_limit,
                                     double work_unit_limit);
#endif

#if MIP_INSTANTIATE_DOUBLE
template bool try_cpufj_binary_solve(fj_cpu_climber_t<int, double>& climber,
                                     double time_limit,
                                     double work_unit_limit);
#endif

}  // namespace cuopt::mathematical_optimization::mip
