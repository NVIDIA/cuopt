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
#include "starts.hpp"

namespace cuopt::mathematical_optimization::mip {

template <typename i_t>
struct pmedian_model_t {
  i_t clients{0}, width{0}, open_count{0};
  std::vector<double> cost, dist;
  double max_coef{0};
  i_t hub{-1};
  std::vector<i_t> facilities;
  std::vector<i_t> assign;
  bool open_when_one{true};
  std::vector<std::vector<i_t>> chain_order, chain_z;
};

template <typename i_t, typename f_t>
static bool recognise_epigraph(fj_cpu_climber_t<i_t, f_t>& c, pmedian_model_t<i_t>& M)
{
  const auto& p = *c.problem;
  const i_t n   = p.n_variables;
  if (c.n_binary_vars != n - 1) return false;
  i_t hub = -1;
  for (i_t v = 0; v < n; ++v) {
    const auto b = c.h_var_bounds[v].get();
    if (!is_integer_var(c, v)) {
      if (hub >= 0 || get_lower(b) < 0 || !std::isfinite(get_lower(b)) || p.h_obj_coeffs[v] <= 0)
        return false;
      hub = v;
    } else if (!c.h_is_binary_variable[v] || get_lower(b) != 0 || get_upper(b) != 1) {
      return false;
    }
    if (!std::isfinite(p.h_obj_coeffs[v]) || p.h_obj_coeffs[v] < 0) return false;
  }
  if (hub < 0) return false;

  std::vector<i_t> controller(n, -1), group(n, -1), facilities;
  std::vector<double> radius(n, 0);
  i_t groups = 0, open_count = 0;
  for (i_t row = 0; row < p.n_constraints; ++row) {
    const i_t begin = p.offsets[row], end = p.offsets[row + 1];
    if (begin == end) return false;
    if (p.cstr_lb[row] == p.cstr_ub[row]) {
      const double rhs = p.cstr_lb[row];
      if (!std::isfinite(rhs) || rhs < 1 || rhs != std::round(rhs) || rhs >= end - begin ||
          (rhs > 1 && !facilities.empty()))
        return false;
      for (i_t q = begin; q < end; ++q) {
        const i_t v = p.variables[q];
        if (v == hub || p.coefficients[q] != 1 || group[v] != -1) return false;
        group[v] = rhs == 1 ? groups : -2;
        if (rhs > 1) facilities.push_back(v);
      }
      if (rhs == 1)
        ++groups;
      else
        open_count = (i_t)rhs;
      continue;
    }
    if (p.cstr_lb[row] != -std::numeric_limits<f_t>::infinity() || p.cstr_ub[row] != 0)
      return false;
    if (end - begin == 1 && p.variables[begin] == hub && p.coefficients[begin] < 0) continue;
    if (end - begin != 2) return false;
    const i_t positive = p.coefficients[begin] > 0 ? begin : begin + 1;
    const i_t negative = positive == begin ? begin + 1 : begin;
    const i_t x = p.variables[positive], y = p.variables[negative];
    if (p.coefficients[positive] <= 0 || p.coefficients[negative] >= 0 || x == hub) return false;
    if (y == hub) {
      radius[x] = std::max(radius[x], -(double)p.coefficients[positive] / p.coefficients[negative]);
    } else {
      if (p.coefficients[positive] != -p.coefficients[negative] || controller[x] >= 0) return false;
      controller[x] = y;
    }
  }
  const i_t width = (i_t)facilities.size();
  if (open_count < 2 || open_count >= width || groups < 2 || (int64_t)groups * width > (1 << 20))
    return false;
  std::vector<i_t> facility_of(n, -1);
  for (i_t j = 0; j < width; ++j) {
    const i_t v = facilities[j];
    if (controller[v] >= 0 || radius[v] != 0 || p.h_obj_coeffs[v] != 0) return false;
    facility_of[v] = j;
  }
  M.assign.assign((size_t)groups * width, -1);
  M.cost.assign((size_t)groups * width, 0.0);
  M.dist.assign((size_t)groups * width, 0.0);
  for (i_t v = 0; v < n; ++v) {
    if (v == hub || facility_of[v] >= 0) continue;
    if (group[v] < 0 || controller[v] < 0 || facility_of[controller[v]] < 0) return false;
    const size_t k = (size_t)group[v] * width + facility_of[controller[v]];
    if (M.assign[k] >= 0 || !std::isfinite(radius[v])) return false;
    M.assign[k] = v;
    M.dist[k]   = radius[v];
    M.cost[k]   = p.h_obj_coeffs[v];
  }
  for (i_t g = 0; g < groups; ++g) {
    double weight = -1;
    for (i_t j = 0; j < width; ++j) {
      const size_t k = (size_t)g * width + j;
      if (M.assign[k] < 0) return false;
      if (M.dist[k] == 0) {
        if (M.cost[k] != 0) return false;
      } else {
        const double ratio = M.cost[k] / M.dist[k];
        if (weight < 0) weight = ratio;
        if (std::abs(ratio - weight) > 1e-10 * std::max(1.0, weight)) return false;
      }
    }
  }
  M.clients       = groups;
  M.width         = width;
  M.open_count    = open_count;
  M.max_coef      = p.h_obj_coeffs[hub];
  M.hub           = hub;
  M.facilities    = std::move(facilities);
  M.open_when_one = true;
  return true;
}

template <typename i_t, typename f_t>
static bool recognise_chain(fj_cpu_climber_t<i_t, f_t>& c, pmedian_model_t<i_t>& M)
{
  const auto& p = *c.problem;
  const i_t n = p.n_variables, m = p.n_constraints;
  if (!c.n_binary_vars || c.n_binary_vars == n) return false;
  i_t card = -1;
  for (i_t row = 0; row < m; ++row) {
    if (p.cstr_lb[row] != p.cstr_ub[row]) continue;
    if (card >= 0) return false;
    card = row;
  }
  if (card < 0) return false;
  const i_t cb = p.offsets[card], ce = p.offsets[card + 1];
  std::vector<i_t> facilities;
  for (i_t q = cb; q < ce; ++q) {
    const i_t v = p.variables[q];
    if (!c.h_is_binary_variable[v] || !is_integer_var(c, v) || p.coefficients[q] != 1) return false;
    facilities.push_back(v);
  }
  const i_t width     = (i_t)facilities.size();
  const double closed = p.cstr_lb[card];
  if (!std::isfinite(closed) || closed != std::round(closed) || closed <= 0 || closed >= width)
    return false;
  const i_t open_count = width - (i_t)closed;
  if (open_count < 2 || open_count >= width) return false;
  std::vector<i_t> facility_of(n, -1);
  for (i_t j = 0; j < width; ++j)
    facility_of[facilities[j]] = j;

  std::vector<i_t> succ_z(n, -1), succ_b(n, -1), head_a(n, -1), head_b(n, -1);
  std::vector<uint8_t> is_target(n, 0), is_head(n, 0);
  for (i_t row = 0; row < m; ++row) {
    if (row == card) continue;
    const i_t begin = p.offsets[row], end = p.offsets[row + 1];
    if (end - begin != 3) return false;
    if (p.cstr_lb[row] != -std::numeric_limits<f_t>::infinity() || p.cstr_ub[row] != 1)
      return false;
    i_t neg = -1, pos[2] = {-1, -1}, np = 0;
    for (i_t q = begin; q < end; ++q) {
      if (p.coefficients[q] == -1) {
        if (neg >= 0) return false;
        neg = p.variables[q];
      } else if (p.coefficients[q] == 1) {
        if (np == 2) return false;
        pos[np++] = p.variables[q];
      } else
        return false;
    }
    if (neg < 0 || np != 2) return false;
    if (is_integer_var(c, neg) || facility_of[neg] >= 0) return false;
    const bool b0 = facility_of[pos[0]] >= 0, b1 = facility_of[pos[1]] >= 0;
    if (b0 && b1) {
      if (is_head[neg]) return false;
      is_head[neg] = 1;
      head_a[neg]  = pos[0];
      head_b[neg]  = pos[1];
    } else if (b0 != b1) {
      const i_t b = b0 ? pos[0] : pos[1];
      const i_t z = b0 ? pos[1] : pos[0];
      if (is_integer_var(c, z) || succ_z[z] >= 0) return false;
      succ_z[z] = neg;
      succ_b[z] = b;
    } else
      return false;
    if (is_target[neg]) return false;
    is_target[neg] = 1;
  }

  std::vector<uint8_t> seen(n, 0);
  M.chain_order.clear();
  M.chain_z.clear();
  for (i_t z = 0; z < n; ++z) {
    if (!is_head[z] || succ_z[z] == z) continue;
    bool is_start = true;
    for (i_t w = 0; w < n && is_start; ++w)
      if (succ_z[w] == z) is_start = false;
    if (!is_start) continue;
    std::vector<i_t> order{head_a[z], head_b[z]}, zs{z};
    i_t cur = z, guard = 0;
    while (succ_z[cur] >= 0) {
      order.push_back(succ_b[cur]);
      cur = succ_z[cur];
      zs.push_back(cur);
      if (++guard > m) return false;
    }
    for (i_t v : zs) {
      if (seen[v]) return false;
      seen[v] = 1;
    }
    for (i_t v : order)
      if (facility_of[v] < 0) return false;
    M.chain_order.push_back(std::move(order));
    M.chain_z.push_back(std::move(zs));
  }
  const i_t clients = (i_t)M.chain_z.size();
  if (clients < 2 || (int64_t)clients * width > (1 << 20)) return false;
  i_t accounted = width;
  for (const auto& zs : M.chain_z)
    accounted += (i_t)zs.size();
  if (accounted != n) return false;

  const double big = (double)width + 1;
  M.cost.assign((size_t)clients * width, big);
  for (i_t ci = 0; ci < clients; ++ci) {
    const auto& order = M.chain_order[ci];
    const auto& zs    = M.chain_z[ci];
    if (order.size() != zs.size() + 1) return false;
    double run = 0;
    for (size_t pos = 0; pos < order.size(); ++pos) {
      if (pos > 0) {
        const double w = p.h_obj_coeffs[zs[pos - 1]];
        if (!std::isfinite(w) || w < 0) return false;
        run += w;
      }
      M.cost[(size_t)ci * width + facility_of[order[pos]]] = run;
    }
  }
  M.dist          = M.cost;
  M.clients       = clients;
  M.width         = width;
  M.open_count    = open_count;
  M.max_coef      = 0;
  M.hub           = -1;
  M.facilities    = std::move(facilities);
  M.open_when_one = false;
  return true;
}

template <typename i_t, typename f_t>
bool apply_pmedian_start(fj_cpu_climber_t<i_t, f_t>& c, double budget)
{
  if (budget <= 0) return false;
  phase_timer_t timer(c.t_start);
  const auto started = std::chrono::steady_clock::now();
  auto expired       = [&] {
    return c.preemption_flag.load(std::memory_order_relaxed) ||
           std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count() >=
             budget;
  };
  const auto& p = *c.problem;
  pmedian_model_t<i_t> M;
  if (!recognise_epigraph(c, M) && !recognise_chain(c, M)) return false;
  if (expired()) return false;

  const i_t G = M.clients, W = M.width, K = M.open_count;
  std::mt19937 rng((uint32_t)c.settings.seed);
  std::vector<i_t> order(W), open_set(K), near(G), second(G);
  std::iota(order.begin(), order.end(), 0);
  std::vector<uint8_t> is_open(W, 0);
  std::vector<f_t> selected_cost(G);
  const auto clients = thrust::make_counting_iterator<i_t>(0);
  const auto one     = thrust::make_transform_iterator(clients, [](i_t) { return f_t{1}; });
  f_t best_objective = std::numeric_limits<f_t>::infinity();
  std::vector<i_t> best_set;

  while (!expired()) {
    std::shuffle(order.begin(), order.end(), rng);
    std::copy(order.begin(), order.begin() + K, open_set.begin());
    std::fill(is_open.begin(), is_open.end(), 0);
    for (i_t f : open_set)
      is_open[f] = 1;
    while (!expired()) {
      f_t maximum = 0;
      for (i_t g = 0; g < G; ++g) {
        const size_t base = (size_t)g * W;
        i_t f1 = -1, f2 = -1;
        for (i_t f : open_set) {
          if (f1 < 0 || M.dist[base + f] < M.dist[base + f1]) {
            f2 = f1;
            f1 = f;
          } else if (f2 < 0 || M.dist[base + f] < M.dist[base + f2])
            f2 = f;
        }
        near[g]          = f1;
        second[g]        = f2;
        selected_cost[g] = M.cost[base + f1];
        maximum          = std::max(maximum, (f_t)M.dist[base + f1]);
      }
      const f_t total   = compensated_dot2(selected_cost.data(), one, selected_cost.size());
      const f_t current = total + (f_t)M.max_coef * maximum;
      if (current < best_objective) {
        best_objective = current;
        best_set       = open_set;
      }
      f_t trial_best = current;
      i_t remove = -1, insert = -1;
      for (i_t i = 0; i < K; ++i) {
        if (expired()) break;
        const i_t r = open_set[i];
        for (i_t f = 0; f < W; ++f) {
          if (is_open[f]) continue;
          f_t mx = 0;
          for (i_t g = 0; g < G; ++g) {
            const size_t base = (size_t)g * W;
            i_t pick          = near[g] == r ? second[g] : near[g];
            if (M.dist[base + f] < M.dist[base + pick]) pick = f;
            selected_cost[g] = M.cost[base + pick];
            mx               = std::max(mx, (f_t)M.dist[base + pick]);
          }
          const f_t sum   = compensated_dot2(selected_cost.data(), one, selected_cost.size());
          const f_t trial = sum + (f_t)M.max_coef * mx;
          if (trial < trial_best - 1e-8) {
            trial_best = trial;
            remove     = i;
            insert     = f;
          }
        }
      }
      if (remove < 0) break;
      is_open[open_set[remove]] = 0;
      is_open[insert]           = 1;
      open_set[remove]          = insert;
    }
  }
  if (best_set.empty()) return false;

  std::vector<f_t> x(p.n_variables, 0);
  std::vector<uint8_t> chosen(W, 0);
  for (i_t f : best_set)
    chosen[f] = 1;
  for (i_t j = 0; j < W; ++j)
    x[M.facilities[j]] = (f_t)((chosen[j] ? 1 : 0) == (M.open_when_one ? 1 : 0) ? 1 : 0);
  if (M.open_when_one) {
    double maximum = get_lower(c.h_var_bounds[M.hub].get());
    for (i_t g = 0; g < G; ++g) {
      const size_t base = (size_t)g * W;
      i_t pick          = -1;
      for (i_t f : best_set)
        if (pick < 0 || M.dist[base + f] < M.dist[base + pick]) pick = f;
      x[M.assign[base + pick]] = 1;
      maximum                  = std::max(maximum, M.dist[base + pick]);
    }
    x[M.hub] = std::nextafter((f_t)maximum, std::numeric_limits<f_t>::infinity());
  } else {
    for (size_t ci = 0; ci < M.chain_z.size(); ++ci) {
      const auto& o = M.chain_order[ci];
      const auto& z = M.chain_z[ci];
      double run    = std::max(0.0, (double)x[o[0]] + (double)x[o[1]] - 1.0);
      x[z[0]]       = (f_t)run;
      for (size_t t = 1; t < z.size(); ++t) {
        run     = std::max(0.0, (double)x[o[t + 1]] + run - 1.0);
        x[z[t]] = (f_t)run;
      }
    }
  }

  return try_commit_start(c, x);
}

#if MIP_INSTANTIATE_FLOAT
template bool apply_pmedian_start<int, float>(fj_cpu_climber_t<int, float>&, double);
#endif

#if MIP_INSTANTIATE_DOUBLE
template bool apply_pmedian_start<int, double>(fj_cpu_climber_t<int, double>&, double);
#endif

}  // namespace cuopt::mathematical_optimization::mip
