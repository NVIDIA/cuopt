/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#pragma once

#include "../internal.hpp"

namespace cuopt::mathematical_optimization::mip {

template <typename i_t, typename f_t>
void compute_variable_coloring(fj_cpu_climber_t<i_t, f_t>& fj_cpu)
{
  phase_timer_t timer(fj_cpu.t_coloring);
  const i_t n_vars  = fj_cpu.problem->n_variables;
  const i_t n_cstrs = fj_cpu.problem->n_constraints;

  i_t max_row_length  = 0;
  double clique_edges = 0;
  for (i_t row = 0; row < n_cstrs; ++row) {
    const i_t length = fj_cpu.problem->offsets[row + 1] - fj_cpu.problem->offsets[row];
    max_row_length   = std::max(max_row_length, length);
    if (length > 1) clique_edges += (double)length * (length - 1) / 2.0;
  }
  if (n_vars <= 0 || max_row_length <= 0) return;

  const double class_size    = (double)n_vars / max_row_length;
  const double edges_per_nnz = clique_edges / std::max<double>(1, (double)fj_cpu.problem->nnz);
  if (class_size < fj_cpu.hp.batch_min_class_size ||
      edges_per_nnz > fj_cpu.hp.batch_max_edges_per_nnz) {
    CUOPT_LOG_DEBUG("CPUFJ move batching declined: class size %.2f, clique edges/nnz %.2f",
                    class_size,
                    edges_per_nnz);
    return;
  }

  [[maybe_unused]] const auto started = std::chrono::steady_clock::now();
  fj_cpu.h_var_color.assign(n_vars, -1);
  fj_cpu.n_colors = 0;
  std::vector<i_t> neighbor_stamp(n_vars, -1);
  std::vector<i_t> color_stamp(n_vars, -1);

  for (i_t var = 0; var < n_vars; ++var) {
    const auto [rev_begin, rev_end] = model_range_for_var<i_t, f_t>(fj_cpu, var);
    for (i_t p = rev_begin; p < rev_end; ++p) {
      const auto [begin, end] =
        model_range_for_row<i_t, f_t>(fj_cpu, fj_cpu.problem->reverse_constraints[p]);
      for (i_t k = begin; k < end; ++k) {
        const i_t other = fj_cpu.problem->variables[k];
        if (other == var || neighbor_stamp[other] == var) continue;
        neighbor_stamp[other] = var;
        const i_t taken       = fj_cpu.h_var_color[other];
        if (taken >= 0) color_stamp[taken] = var;
      }
    }

    i_t color = 0;
    while (color < fj_cpu.n_colors && color_stamp[color] == var)
      ++color;
    if (color == fj_cpu.n_colors) ++fj_cpu.n_colors;
    fj_cpu.h_var_color[var] = color;
  }

  fj_cpu.h_var_best_score.assign(n_vars, fj_staged_score_t::invalid());
  fj_cpu.h_var_best_delta.assign(n_vars, f_t{0});
  fj_cpu.h_var_best_stamp.assign(n_vars, 0);
  fj_cpu.h_var_best_rowsum.assign(n_vars, 0);
  fj_cpu.h_var_bucket_stamp.assign(n_vars, 0);
  fj_cpu.batch_size_hist.assign(fj_cpu.hp.batch_hist_bins, 0);
  fj_cpu.h_color_candidates.assign(fj_cpu.n_colors, {});
  fj_cpu.h_color_epoch.assign(fj_cpu.n_colors, 0);
  fj_cpu.var_best_epoch = 1;

  CUOPT_LOG_DEBUG(
    "CPUFJ move batching: %d colours over %d variables in %.3f ms",
    fj_cpu.n_colors,
    n_vars,
    std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - started).count());
}

template <typename i_t, typename f_t>
inline int64_t incident_row_version_sum(fj_cpu_climber_t<i_t, f_t>& fj_cpu, i_t var_idx)
{
  const auto [begin, end] = fj_cpu.range_for_variable(var_idx);
  int64_t sum             = 0;
  for (i_t p = begin; p < end; ++p)
    sum += fj_cpu.h_cstr_version[fj_cpu.h_reverse_constraints[p]];
  return sum;
}

template <typename i_t, typename f_t>
inline void record_var_best_move(fj_cpu_climber_t<i_t, f_t>& fj_cpu,
                                 i_t var_idx,
                                 fj_staged_score_t score,
                                 f_t delta)
{
  if (!fj_cpu.use_move_batching) return;
  if (!(score > fj_staged_score_t::zero())) return;

  const bool current = fj_cpu.h_var_best_stamp[var_idx] == fj_cpu.var_best_epoch;
  if (current && !(score > fj_cpu.h_var_best_score[var_idx])) return;

  fj_cpu.h_var_best_score[var_idx]  = score;
  fj_cpu.h_var_best_delta[var_idx]  = delta;
  fj_cpu.h_var_best_stamp[var_idx]  = fj_cpu.var_best_epoch;
  fj_cpu.h_var_best_rowsum[var_idx] = incident_row_version_sum<i_t, f_t>(fj_cpu, var_idx);

  const i_t color = fj_cpu.h_var_color[var_idx];
  cuopt_assert(color >= 0 && color < fj_cpu.n_colors, "variable has no colour");
  if (fj_cpu.h_color_epoch[color] != fj_cpu.var_best_epoch) {
    fj_cpu.h_color_candidates[color].clear();
    fj_cpu.h_color_epoch[color] = fj_cpu.var_best_epoch;
  }
  if (fj_cpu.h_var_bucket_stamp[var_idx] == fj_cpu.var_best_epoch) return;
  fj_cpu.h_var_bucket_stamp[var_idx] = fj_cpu.var_best_epoch;
  fj_cpu.h_color_candidates[color].push_back(var_idx);
}

template <typename i_t, typename f_t>
void retire_var_best_moves(fj_cpu_climber_t<i_t, f_t>& fj_cpu)
{
  if (!fj_cpu.use_move_batching) return;
  ++fj_cpu.var_best_epoch;
}

template <typename i_t, typename f_t>
void log_batch_distribution(const fj_cpu_climber_t<i_t, f_t>& fj_cpu)
{
  if (fj_cpu.n_batch_attempts == 0) return;

  int32_t smallest = -1;
  int32_t median   = -1;
  int64_t seen     = 0;
  for (size_t bin = 0; bin < fj_cpu.batch_size_hist.size(); ++bin) {
    if (fj_cpu.batch_size_hist[bin] == 0) continue;
    if (smallest < 0) smallest = (int32_t)bin;
    seen += fj_cpu.batch_size_hist[bin];
    if (median < 0 && 2 * seen > fj_cpu.n_batch_attempts) median = (int32_t)bin;
  }

  CUOPT_LOG_DEBUG(
    "%sCPUFJ batch companions: min %d median %d max %lld mean %.3f over %lld attempts, %lld total, "
    "%d colours, batching %s",
    fj_cpu.log_prefix.c_str(),
    smallest,
    median,
    (long long)fj_cpu.max_batch_size,
    (double)fj_cpu.n_batched_moves / (double)fj_cpu.n_batch_attempts,
    (long long)fj_cpu.n_batch_attempts,
    (long long)fj_cpu.n_batched_moves,
    fj_cpu.n_colors,
    fj_cpu.use_move_batching ? "on" : "off");
}

template <typename i_t, typename f_t>
void collect_move_batch(fj_cpu_climber_t<i_t, f_t>& fj_cpu,
                        fj_move_t chosen,
                        std::vector<fj_move_t>& batch)
{
  batch.clear();
  if (!fj_cpu.use_move_batching) return;

  const i_t color = fj_cpu.h_var_color[chosen.var_idx];
  cuopt_assert(color >= 0 && color < fj_cpu.n_colors, "chosen move has no colour");
  if (fj_cpu.h_color_epoch[color] != fj_cpu.var_best_epoch) return;

  for (i_t var_idx : fj_cpu.h_color_candidates[color]) {
    if (var_idx == chosen.var_idx) continue;
    if (fj_cpu.h_var_best_stamp[var_idx] != fj_cpu.var_best_epoch) continue;
    if (!(fj_cpu.h_var_best_score[var_idx] > fj_staged_score_t::zero())) continue;
    if (fj_cpu.h_var_best_rowsum[var_idx] != incident_row_version_sum<i_t, f_t>(fj_cpu, var_idx))
      continue;

    batch.push_back({var_idx, fj_cpu.h_var_best_delta[var_idx]});
    // Invalidated so a second pass over the bucket cannot apply the move twice.
    fj_cpu.h_var_best_stamp[var_idx] = 0;
  }

  ++fj_cpu.n_batch_attempts;
  fj_cpu.n_batched_moves += (int64_t)batch.size();
  ++fj_cpu.batch_size_hist[std::min<size_t>(batch.size(), fj_cpu.batch_size_hist.size() - 1)];
  if ((int64_t)batch.size() > fj_cpu.max_batch_size) fj_cpu.max_batch_size = (int64_t)batch.size();
  if (fj_cpu.n_batch_attempts == fj_cpu.hp.batch_probe_attempts &&
      (double)fj_cpu.n_batched_moves <
        fj_cpu.hp.batch_min_yield * (double)fj_cpu.hp.batch_probe_attempts) {
    fj_cpu.use_move_batching = false;
    CUOPT_LOG_DEBUG("%sCPUFJ move batching off: %lld companions over %lld attempts",
                    fj_cpu.log_prefix.c_str(),
                    (long long)fj_cpu.n_batched_moves,
                    (long long)fj_cpu.n_batch_attempts);
  }
}

}  // namespace cuopt::mathematical_optimization::mip
