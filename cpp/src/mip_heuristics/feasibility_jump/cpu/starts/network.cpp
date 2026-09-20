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
#include "../search/update.hpp"
#include "starts.hpp"

#include <queue>
#include <random>

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

  using arc_t = typename fj_fixed_charge_network_t<i_t, f_t>::arc_t;
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

  // Keep one lane-local perturbation per arc across all shortest-path passes. Models that price
  // both activation and flow benefit from broader tree diversity than pure fixed-charge models.
  const bool costs_both_ends = std::all_of(arcs.begin(), arcs.end(), [](const arc_t& arc) {
    return arc.fix > 0 && arc.unit > 0;
  });
  const double jitter_radius = costs_both_ends ? 0.30 : 0.15;
  std::mt19937 weight_rng(c.settings.seed);
  std::uniform_real_distribution<double> jitter(1.0 - jitter_radius, 1.0 + jitter_radius);
  std::vector<double> arc_weight(arcs.size());
  for (i_t e = 0; e < static_cast<i_t>(arcs.size()); ++e)
    arc_weight[e] = (arcs[e].fix + arcs[e].unit) * jitter(weight_rng);
  auto weight = [&](i_t e) { return arc_weight[e]; };
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

  if (!try_commit_start(c, x)) return false;

  // Preserve the certified basis. Components are fixed by tree exchanges, so disconnected arc
  // endpoints can never become fundamental-cycle candidates later.
  auto& network = c.fixed_charge_network;
  network           = {};
  network.certified = true;
  network.arcs      = arcs;
  network.in_tree.assign(arcs.size(), 0);
  network.adjacency.assign(demand.size(), {});
  for (i_t v = 0; v < (i_t)demand.size(); ++v)
    if (parent[v] >= 0) network.in_tree[parent[v]] = 1;
  for (i_t e = 0; e < (i_t)arcs.size(); ++e) {
    if (inert[e] || forbidden[e]) continue;
    network.adjacency[arcs[e].source].push_back(e);
    network.adjacency[arcs[e].target].push_back(e);
  }

  std::vector<i_t> component(demand.size(), -1), stack;
  for (i_t seed = 0; seed < (i_t)demand.size(); ++seed) {
    if (component[seed] >= 0) continue;
    component[seed] = seed;
    stack.assign(1, seed);
    while (!stack.empty()) {
      const i_t v = stack.back();
      stack.pop_back();
      for (i_t e : network.adjacency[v]) {
        if (!network.in_tree[e]) continue;
        const i_t w = arcs[e].source == v ? arcs[e].target : arcs[e].source;
        if (component[w] >= 0) continue;
        component[w] = seed;
        stack.push_back(w);
      }
    }
  }
  for (i_t e = 0; e < (i_t)arcs.size(); ++e)
    if (!network.in_tree[e] && !inert[e] && !forbidden[e] &&
        component[arcs[e].source] == component[arcs[e].target])
      network.closed_arcs.push_back(e);

  network.path_parent.resize(demand.size());
  network.path_arc.resize(demand.size());
  network.cycle_arcs.reserve(demand.size() + 1);
  network.cycle_signs.reserve(demand.size() + 1);
  network.variable_delta.assign(p.n_variables, f_t{0});
  network.row_touched.assign(p.n_constraints, 0);
  return true;
}

template <typename i_t, typename f_t>
bool try_fundamental_cycle_pivot(fj_cpu_climber_t<i_t, f_t>& c)
{
  auto& network = c.fixed_charge_network;
  if (!c.use_fundamental_cycle_pivot || !network.certified || network.closed_arcs.empty() ||
      !c.feasible_found || !c.violated_constraints.empty())
    return false;

  const auto& p       = *c.problem;
  const size_t slot   = network.next_closed;
  const i_t entering  = network.closed_arcs[slot];
  network.next_closed = (slot + 1) % network.closed_arcs.size();
  if (network.in_tree[entering]) return false;
  const auto& enter = network.arcs[entering];

  // Recover the unique tree path for this entering arc; no alternative move is scanned.
  std::fill(network.path_parent.begin(), network.path_parent.end(), -1);
  std::fill(network.path_arc.begin(), network.path_arc.end(), -1);
  network.stack.clear();
  network.path_parent[enter.target] = enter.target;
  network.stack.push_back(enter.target);
  while (!network.stack.empty() && network.path_parent[enter.source] < 0) {
    const i_t v = network.stack.back();
    network.stack.pop_back();
    for (i_t e : network.adjacency[v]) {
      if (!network.in_tree[e]) continue;
      const auto& arc = network.arcs[e];
      const i_t w     = arc.source == v ? arc.target : arc.source;
      if (network.path_parent[w] >= 0) continue;
      network.path_parent[w] = v;
      network.path_arc[w]    = e;
      network.stack.push_back(w);
    }
  }
  if (network.path_parent[enter.source] < 0) return false;

  network.cycle_arcs.clear();
  network.cycle_signs.clear();
  const f_t bound_tolerance = std::max((f_t)1e-9, p.tolerances.absolute_tolerance);
  const f_t entering_flow   = c.h_assignment[enter.flow];
  int entering_sign;
  if (std::fabs(entering_flow) <= bound_tolerance)
    entering_sign = 1;
  else if (std::fabs(entering_flow - enter.capacity) <= bound_tolerance)
    entering_sign = -1;
  else
    return false;
  network.cycle_arcs.push_back(entering);
  network.cycle_signs.push_back(entering_sign);

  for (i_t v = enter.source; v != enter.target; v = network.path_parent[v]) {
    const i_t e      = network.path_arc[v];
    const i_t parent = network.path_parent[v];
    if (e < 0) return false;
    const auto& arc = network.arcs[e];
    const int sign  = arc.source == parent && arc.target == v ? 1 : -1;
    network.cycle_arcs.push_back(e);
    network.cycle_signs.push_back(entering_sign * sign);
  }

  f_t augmentation = std::numeric_limits<f_t>::infinity();
  for (size_t k = 0; k < network.cycle_arcs.size(); ++k) {
    const auto& arc = network.arcs[network.cycle_arcs[k]];
    const f_t value = c.h_assignment[arc.flow];
    const f_t room  = network.cycle_signs[k] > 0 ? arc.capacity - value : value;
    if (room < -bound_tolerance) return false;
    augmentation = std::min(augmentation, std::max(f_t{0}, room));
  }
  if (!(augmentation > bound_tolerance) || !std::isfinite(augmentation)) return false;

  i_t leaving = -1;
  f_t objective_delta = 0;
  network.touched_variables.clear();
  auto add_delta = [&](i_t variable, f_t delta) {
    if (delta == f_t{0}) return;
    if (network.variable_delta[variable] == f_t{0}) network.touched_variables.push_back(variable);
    network.variable_delta[variable] += delta;
  };

  for (size_t k = 0; k < network.cycle_arcs.size(); ++k) {
    const i_t cycle_arc = network.cycle_arcs[k];
    const auto& arc     = network.arcs[cycle_arc];
    const f_t old_flow  = c.h_assignment[arc.flow];
    f_t new_flow = old_flow + (f_t)network.cycle_signs[k] * augmentation;
    if (std::fabs(new_flow) <= bound_tolerance) new_flow = 0;
    if (std::fabs(new_flow - arc.capacity) <= bound_tolerance) new_flow = arc.capacity;
    if (cycle_arc != entering && (new_flow == f_t{0} || new_flow == arc.capacity) && leaving < 0)
      leaving = cycle_arc;
    add_delta(arc.flow, new_flow - old_flow);

    const bool was_positive = old_flow > bound_tolerance;
    const bool now_positive = new_flow > bound_tolerance;
    const f_t controller     = c.h_assignment[arc.binary];
    if (!was_positive && now_positive && controller < f_t{0.5})
      add_delta(arc.binary, f_t{1} - controller);
    else if (was_positive && !now_positive && controller > f_t{0.5})
      add_delta(arc.binary, -controller);
  }
  if (leaving < 0) {
    for (i_t variable : network.touched_variables) network.variable_delta[variable] = 0;
    return false;
  }
  for (i_t variable : network.touched_variables)
    objective_delta += p.h_obj_coeffs[variable] * network.variable_delta[variable];

  bool accept = objective_delta < f_t{0};
  if (!accept && c.network_temperature > f_t{0}) {
    const size_t sweep = static_cast<size_t>(c.iterations) / network.closed_arcs.size();
    const double temperature =
      c.network_temperature *
      std::max(1.0, static_cast<double>(enter.fix + enter.unit * augmentation)) *
      std::exp(-static_cast<double>(sweep % 16) / 4.0);
    accept = c.rng.next_double() < std::exp(-static_cast<double>(objective_delta) / temperature);
  }
  if (!accept) {
    for (i_t variable : network.touched_variables) network.variable_delta[variable] = 0;
    return false;
  }

  // Validate every touched original row before changing the incremental search state.
  network.touched_rows.clear();
  bool valid = true;
  for (i_t variable : network.touched_variables) {
    const f_t candidate = c.h_assignment[variable] + network.variable_delta[variable];
    if (!std::isfinite(candidate) || !check_variable_within_bounds(c, variable, candidate) ||
        (is_integer_var(c, variable) && !p.is_integer(candidate)))
      valid = false;
    for (i_t q = p.reverse_offsets[variable]; q < p.reverse_offsets[variable + 1]; ++q) {
      const i_t row = p.reverse_constraints[q];
      if (!network.row_touched[row]) {
        network.row_touched[row] = 1;
        network.touched_rows.push_back(row);
      }
    }
  }
  for (i_t row : network.touched_rows) {
    long double activity = 0;
    for (i_t q = p.offsets[row]; q < p.offsets[row + 1]; ++q) {
      const i_t variable = p.variables[q];
      activity += (long double)p.coefficients[q] *
                  (c.h_assignment[variable] + network.variable_delta[variable]);
    }
    if (!std::isfinite((double)activity) ||
        activity < (long double)p.cstr_lb[row] - p.tolerances.absolute_tolerance ||
        activity > (long double)p.cstr_ub[row] + p.tolerances.absolute_tolerance)
      valid = false;
    network.row_touched[row] = 0;
  }
  if (!valid) {
    for (i_t variable : network.touched_variables) network.variable_delta[variable] = 0;
    return false;
  }

  // Apply the dependent cycle as one network move: controllers open before flow changes and close
  // afterward. The complete point has already been checked against the original model.
  for (i_t variable : network.touched_variables)
    if (is_integer_var(c, variable) && network.variable_delta[variable] > 0)
      apply_move(c, variable, network.variable_delta[variable], false);
  for (i_t variable : network.touched_variables)
    if (!is_integer_var(c, variable))
      apply_move(c, variable, network.variable_delta[variable], false);
  for (i_t variable : network.touched_variables)
    if (is_integer_var(c, variable) && network.variable_delta[variable] < 0)
      apply_move(c, variable, network.variable_delta[variable], false);

  for (i_t variable : network.touched_variables) network.variable_delta[variable] = 0;
  network.in_tree[entering] = 1;
  network.in_tree[leaving]  = 0;
  network.closed_arcs[slot] = leaving;
  return true;
}

#if MIP_INSTANTIATE_FLOAT
template bool apply_fixed_charge_network_start<int, float>(fj_cpu_climber_t<int, float>&, double);
template bool try_fundamental_cycle_pivot<int, float>(fj_cpu_climber_t<int, float>&);
#endif

#if MIP_INSTANTIATE_DOUBLE
template bool apply_fixed_charge_network_start<int, double>(fj_cpu_climber_t<int, double>&, double);
template bool try_fundamental_cycle_pivot<int, double>(fj_cpu_climber_t<int, double>&);
#endif

}  // namespace cuopt::mathematical_optimization::mip
