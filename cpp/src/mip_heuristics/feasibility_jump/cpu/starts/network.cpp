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

#include <queue>

namespace cuopt::mathematical_optimization::mip {

template <typename i_t, typename f_t>
bool apply_fixed_charge_network_start(fj_cpu_climber_t<i_t, f_t>& c, double budget)
{
  const auto& p = *c.problem;
  if (budget <= 0 || !c.n_binary_vars || c.n_binary_vars == p.n_variables) return false;
  phase_timer_t timer(c.t_start);
  const auto started = std::chrono::steady_clock::now();
  auto expired       = [&] {
    return c.preemption_flag.load(std::memory_order_relaxed) ||
           std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count() >=
             budget;
  };
  constexpr i_t max_nodes_for_reparent = 1024;

  struct arc_t {
    i_t binary, flow, capacity_row, source, target;
    double capacity, fix, unit;
  };
  std::vector<arc_t> arcs;
  std::vector<uint8_t> used(p.n_variables, 0), is_capacity(p.n_constraints, 0);
  std::vector<i_t> arc_of_binary(p.n_variables, -1);
  std::vector<f_t> x(p.n_variables, 0);

  for (i_t u = 0; u < p.n_variables; ++u) {
    if ((u & 1023) == 0 && expired()) return false;
    if (!c.h_is_binary_variable[u] || !is_integer_var(c, u)) continue;
    const auto bounds = c.h_var_bounds[u].get();
    if (get_lower(bounds) != 0 || get_upper(bounds) != 1) return false;
    if (!std::isfinite(p.h_obj_coeffs[u]) || p.h_obj_coeffs[u] < 0) return false;
    i_t row = -1;
    for (i_t q = p.reverse_offsets[u]; q < p.reverse_offsets[u + 1]; ++q) {
      const i_t r = p.reverse_constraints[q];
      if (p.offsets[r + 1] - p.offsets[r] == 2 &&
          p.cstr_lb[r] == -std::numeric_limits<f_t>::infinity() && p.cstr_ub[r] == 0) {
        const i_t begin = p.offsets[r];
        const i_t other = p.variables[begin] == u ? p.variables[begin + 1] : p.variables[begin];
        if (!is_integer_var(c, other)) {
          if (row >= 0) return false;
          row = r;
          continue;
        }
      }
      if (p.reverse_coefficients[q] != 1) return false;
    }
    if (row < 0 || is_capacity[row]) return false;
    const i_t begin = p.offsets[row];
    const i_t uq    = p.variables[begin] == u ? begin : begin + 1;
    const i_t vq    = uq == begin ? begin + 1 : begin;
    const i_t v     = p.variables[vq];
    if (used[v] || p.coefficients[uq] >= 0 || p.coefficients[vq] <= 0) return false;
    if (!std::isfinite(p.h_obj_coeffs[v]) || p.h_obj_coeffs[v] < 0) return false;
    const auto flow_bounds = c.h_var_bounds[v].get();
    const double capacity =
      std::min((double)get_upper(flow_bounds), -(double)p.coefficients[uq] / p.coefficients[vq]);
    if (get_lower(flow_bounds) != 0 || !(capacity > 0) || !std::isfinite(capacity)) return false;
    used[u] = used[v] = is_capacity[row] = 1;
    arc_of_binary[u]                     = (i_t)arcs.size();
    arcs.push_back(
      {u, v, row, -1, -1, capacity, (double)p.h_obj_coeffs[u], (double)p.h_obj_coeffs[v]});
  }
  if (arcs.empty()) return false;
  for (i_t v = 0; v < p.n_variables; ++v) {
    if (used[v]) continue;
    const auto bounds = c.h_var_bounds[v].get();
    if (get_lower(bounds) != get_upper(bounds) || p.reverse_offsets[v] != p.reverse_offsets[v + 1])
      return false;
    x[v] = get_lower(bounds);
  }

  std::vector<i_t> node(p.n_constraints, -1), group_rows;
  std::vector<double> demand;
  double demand_total = 0;
  for (i_t row = 0; row < p.n_constraints; ++row) {
    if (is_capacity[row]) continue;
    bool all_binary = true;
    for (i_t q = p.offsets[row]; q < p.offsets[row + 1] && all_binary; ++q)
      all_binary = c.h_is_binary_variable[p.variables[q]] && is_integer_var(c, p.variables[q]);
    if (all_binary && p.offsets[row + 1] > p.offsets[row]) {
      group_rows.push_back(row);
      continue;
    }
    if (!std::isfinite(p.cstr_lb[row]) || p.cstr_lb[row] != p.cstr_ub[row]) return false;
    node[row] = (i_t)demand.size();
    demand.push_back(p.cstr_lb[row]);
    if (p.cstr_lb[row] > 0) demand_total += p.cstr_lb[row];
  }
  if (!(demand_total > 0) || !std::isfinite(demand_total)) return false;
  const i_t root = (i_t)demand.size();
  demand.push_back(0);

  std::vector<uint8_t> inert(arcs.size(), 0);
  std::vector<std::vector<i_t>> outgoing(demand.size());
  for (i_t e = 0; e < (i_t)arcs.size(); ++e) {
    if ((e & 1023) == 0 && expired()) return false;
    auto& arc = arcs[e];
    for (i_t q = p.reverse_offsets[arc.flow]; q < p.reverse_offsets[arc.flow + 1]; ++q) {
      const i_t row = p.reverse_constraints[q];
      if (row == arc.capacity_row) continue;
      if (node[row] < 0) return false;
      const double coefficient = p.reverse_coefficients[q];
      if (coefficient == -1 && arc.source < 0)
        arc.source = node[row];
      else if (coefficient == 1 && arc.target < 0)
        arc.target = node[row];
      else
        return false;
    }
    if (arc.source < 0 && arc.target < 0) {
      inert[e] = 1;
      continue;
    }
    if (arc.capacity < demand_total) return false;
    if (arc.source < 0) arc.source = root;
    if (arc.target < 0) arc.target = root;
    if (arc.source == arc.target) return false;
    outgoing[arc.source].push_back(e);
  }

  std::vector<uint8_t> needs_parent(demand.size(), 0), forbidden(arcs.size(), 0);
  for (i_t row : group_rows) {
    if (expired()) return false;
    const i_t begin = p.offsets[row], end = p.offsets[row + 1];
    const double lower = p.cstr_lb[row], upper = p.cstr_ub[row];
    if (upper != 1 && upper != 0) return false;
    if (lower != upper && lower != -std::numeric_limits<f_t>::infinity()) return false;
    i_t head = -1;
    for (i_t q = begin; q < end; ++q) {
      if (p.coefficients[q] != 1) return false;
      const i_t e = arc_of_binary[p.variables[q]];
      if (e < 0) return false;
      if (upper == 0) {
        forbidden[e] = 1;
        continue;
      }
      if (inert[e]) continue;
      if (head < 0)
        head = arcs[e].target;
      else if (head != arcs[e].target)
        return false;
    }
    if (upper == 0) continue;
    if (head < 0) return false;
    if (lower == upper) needs_parent[head] = 1;
  }
  for (i_t e = 0; e < (i_t)arcs.size(); ++e)
    if (inert[e] && !forbidden[e]) return false;

  std::vector<i_t> terminals;
  for (i_t v = 0; v < root; ++v)
    if (demand[v] > 0 || needs_parent[v]) terminals.push_back(v);
  if (terminals.empty()) return false;

  std::vector<uint8_t> in_tree(demand.size(), 0);
  std::vector<i_t> parent(demand.size(), -1), predecessor(demand.size());
  std::vector<double> distance(demand.size());
  bool any_source = false;
  for (i_t v = 0; v < root; ++v)
    if (demand[v] < 0) {
      in_tree[v] = 1;
      any_source = true;
    }
  if (!any_source) in_tree[root] = 1;

  auto weight = [&](i_t e) { return arcs[e].fix + arcs[e].unit; };
  while (!expired()) {
    bool complete = true;
    for (i_t v : terminals)
      complete &= (bool)in_tree[v];
    if (complete) break;
    std::fill(distance.begin(), distance.end(), std::numeric_limits<double>::infinity());
    std::fill(predecessor.begin(), predecessor.end(), -1);
    using entry_t = std::pair<double, i_t>;
    std::priority_queue<entry_t, std::vector<entry_t>, std::greater<entry_t>> queue;
    for (i_t v = 0; v < (i_t)demand.size(); ++v)
      if (in_tree[v]) {
        distance[v] = 0;
        queue.emplace(0, v);
      }
    i_t reached = -1;
    while (!queue.empty()) {
      const auto [dist, v] = queue.top();
      queue.pop();
      if (dist != distance[v]) continue;
      if (!in_tree[v] && (demand[v] > 0 || needs_parent[v])) {
        reached = v;
        break;
      }
      if (expired()) return false;
      for (i_t e : outgoing[v]) {
        if (forbidden[e]) continue;
        const double next = dist + weight(e);
        if (next >= distance[arcs[e].target]) continue;
        distance[arcs[e].target]    = next;
        predecessor[arcs[e].target] = e;
        queue.emplace(next, arcs[e].target);
      }
    }
    if (reached < 0) return false;
    for (i_t v = reached; !in_tree[v]; v = arcs[parent[v]].source) {
      parent[v] = predecessor[v];
      if (parent[v] < 0) return false;
      in_tree[v] = 1;
    }
  }
  if (expired()) return false;

  auto subtree_flow = [&](std::vector<double>& flow) {
    flow.assign(demand.size(), 0.0);
    for (i_t t = 0; t < root; ++t) {
      if (demand[t] <= 0) continue;
      i_t guard = 0;
      for (i_t v = t; parent[v] >= 0; v = arcs[parent[v]].source) {
        flow[v] += demand[t];
        if (++guard > (i_t)demand.size()) return false;
      }
    }
    return true;
  };
  auto tree_cost = [&](std::vector<double>& flow) {
    const auto nodes      = thrust::make_counting_iterator<i_t>(0);
    const auto fixed_cost = thrust::make_transform_iterator(
      nodes, [&](i_t v) { return parent[v] >= 0 ? (f_t)arcs[parent[v]].fix : f_t{0}; });
    const auto active = thrust::make_transform_iterator(
      nodes, [&](i_t v) { return parent[v] >= 0 ? f_t{1} : f_t{0}; });
    const auto unit_cost = thrust::make_transform_iterator(
      nodes, [&](i_t v) { return parent[v] >= 0 ? (f_t)arcs[parent[v]].unit : f_t{0}; });
    const auto node_flow =
      thrust::make_transform_iterator(nodes, [&](i_t v) { return (f_t)flow[v]; });
    return compensated_dot2(fixed_cost, active, demand.size()) +
           compensated_dot2(unit_cost, node_flow, demand.size());
  };
  std::vector<double> flow;
  if (!subtree_flow(flow)) return false;

  if ((i_t)demand.size() <= max_nodes_for_reparent) {
    std::vector<std::vector<i_t>> incoming(demand.size());
    for (i_t e = 0; e < (i_t)arcs.size(); ++e)
      if (!inert[e] && !forbidden[e]) incoming[arcs[e].target].push_back(e);
    std::vector<uint8_t> descendant(demand.size(), 0);
    f_t best = tree_cost(flow);
    for (bool improved = true; improved && !expired();) {
      improved = false;
      for (i_t v = 0; v < root && !expired(); ++v) {
        if (parent[v] < 0) continue;
        std::fill(descendant.begin(), descendant.end(), 0);
        descendant[v] = 1;
        for (bool grew = true; grew;) {
          grew = false;
          for (i_t w = 0; w < (i_t)demand.size(); ++w)
            if (!descendant[w] && parent[w] >= 0 && descendant[arcs[parent[w]].source]) {
              descendant[w] = 1;
              grew          = true;
            }
        }
        const i_t original = parent[v];
        i_t choice         = original;
        for (i_t e : incoming[v]) {
          if (e == original || descendant[arcs[e].source] || !in_tree[arcs[e].source]) continue;
          parent[v] = e;
          if (!subtree_flow(flow)) {
            parent[v] = choice;
            continue;
          }
          const f_t trial = tree_cost(flow);
          if (trial < best - 1e-9) {
            best     = trial;
            choice   = e;
            improved = true;
          }
          parent[v] = choice;
        }
        parent[v] = choice;
      }
    }
    if (!subtree_flow(flow)) return false;
  }

  for (i_t v = 0; v < (i_t)demand.size(); ++v) {
    if (parent[v] < 0) continue;
    x[arcs[parent[v]].binary] = 1;
    x[arcs[parent[v]].flow]   = (f_t)flow[v];
  }

  return try_commit_start(c, x);
}

#if MIP_INSTANTIATE_FLOAT
template bool apply_fixed_charge_network_start<int, float>(fj_cpu_climber_t<int, float>&, double);
#endif

#if MIP_INSTANTIATE_DOUBLE
template bool apply_fixed_charge_network_start<int, double>(fj_cpu_climber_t<int, double>&, double);
#endif

}  // namespace cuopt::mathematical_optimization::mip
