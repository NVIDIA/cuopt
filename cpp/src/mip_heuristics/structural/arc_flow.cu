/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#include "arc_flow.cuh"

#include <mip_heuristics/mip_constants.hpp>
#include <mip_heuristics/utils.cuh>

#include <utilities/copy_helpers.hpp>
#include <utilities/macros.cuh>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <optional>
#include <tuple>
#include <utility>
#include <vector>

namespace cuopt::mathematical_optimization::mip {

namespace {

constexpr int arcflow_paths_supported        = 2;
constexpr int arcflow_max_tokens             = 20000;
constexpr int arcflow_max_col_entries        = 3;
constexpr size_t arcflow_history_bytes_max   = size_t{32} << 20;
constexpr size_t arcflow_candidate_bytes_max = size_t{32} << 20;

enum class row_role_t : uint8_t { flow, cover };

template <typename i_t, typename f_t>
struct arc_t {
  i_t from{-1};
  i_t to{-1};
  i_t label{-1};
  i_t col{-1};
  f_t cost{0};
};

template <typename i_t, typename f_t>
struct arc_flow_model_t {
  i_t n_nodes{0};
  i_t n_labels{0};

  std::vector<f_t> phi;
  std::vector<i_t> path_start;
  std::vector<int64_t> demand;
  std::vector<f_t> displacement;
  std::vector<f_t> slope;
  std::vector<i_t> arc_offset;
  std::vector<arc_t<i_t, f_t>> arcs;

  // A negative terminator_col denotes conservation-row slack.
  std::vector<i_t> terminator_col;
  std::vector<f_t> terminator_cost;
  std::vector<int64_t> terminator_capacity;
};

template <typename i_t, typename f_t>
struct frontier_t {
  std::array<i_t, arcflow_paths_supported> node{};
  f_t cost{0};
};

template <typename i_t>
struct parent_t {
  i_t prev{-1};
  i_t arc{-1};
};

template <typename i_t, typename f_t>
struct candidate_t {
  frontier_t<i_t, f_t> front;
  parent_t<i_t> parent;
};

template <typename i_t>
struct arc_flow_result_t {
  std::vector<i_t> columns;
  bool exact{true};
};

template <typename f_t>
bool is_known(f_t v)
{
  return !std::isnan(v);
}

template <typename i_t, typename f_t>
struct arcflow_profile_t {
  arcflow_profile_t(i_t n_variables = 0, i_t n_constraints = 0)
    : col_entries(n_variables, 0),
      row_min_mag(n_constraints, std::numeric_limits<f_t>::infinity()),
      row_max_mag(n_constraints, 0)
  {
  }

  std::vector<int64_t> col_entries;
  std::vector<f_t> row_min_mag;
  std::vector<f_t> row_max_mag;
};

template <typename i_t, typename f_t>
struct host_problem_t {
  i_t n_variables{0};
  i_t n_constraints{0};
  std::vector<f_t> csr_values;
  std::vector<i_t> csr_cols;
  std::vector<i_t> csr_offsets;
  std::vector<f_t> row_lb;
  std::vector<f_t> row_ub;
  std::vector<f_t> obj;
  std::vector<f_t> var_lb;
  std::vector<f_t> var_ub;
  std::vector<var_t> var_types;
};

bool arcflow_accepts_shape(int64_t n_variables, int64_t n_constraints, int64_t nnz)
{
  if (n_variables <= 0 || n_constraints <= 0) { return false; }
  return nnz > 0 && nnz <= (int64_t)arcflow_max_col_entries * n_variables;
}

// Every row is either two sided or bounded from below only, and both kinds must occur: the first
// become conservation rows and the second covering rows.
template <typename f_t>
bool arcflow_accepts_bounds(const std::vector<f_t>& row_lb, const std::vector<f_t>& row_ub)
{
  int64_t n_flow_candidates  = 0;
  int64_t n_cover_candidates = 0;
  for (size_t r = 0; r < row_lb.size(); ++r) {
    const bool lo_fin = std::isfinite(row_lb[r]);
    const bool hi_fin = std::isfinite(row_ub[r]);
    if (lo_fin && hi_fin) {
      n_flow_candidates++;
    } else if (lo_fin) {
      n_cover_candidates++;
    } else {
      return false;
    }
  }
  return n_flow_candidates > 0 && n_cover_candidates > 0;
}

// The row loop needs every magnitude in the row, so the acceptance pass cannot merge into the
// nonzero pass that builds the profile.
template <typename i_t, typename f_t>
bool arcflow_accepts_profile(const host_problem_t<i_t, f_t>& h,
                             const typename mip_solver_settings_t<i_t, f_t>::tolerances_t& tolerances,
                             arcflow_profile_t<i_t, f_t>& p)
{
  p = arcflow_profile_t<i_t, f_t>(h.n_variables, h.n_constraints);
  for (i_t r = 0; r < h.n_constraints; ++r) {
    for (i_t k = h.csr_offsets[r]; k < h.csr_offsets[r + 1]; ++k) {
      const i_t col = h.csr_cols[k];
      cuopt_assert(col >= 0 && col < h.n_variables, "Column index out of range");
      if (++p.col_entries[col] > arcflow_max_col_entries) { return false; }
      const f_t mag    = std::abs(h.csr_values[k]);
      p.row_min_mag[r] = std::min(p.row_min_mag[r], mag);
      p.row_max_mag[r] = std::max(p.row_max_mag[r], mag);
    }
  }

  f_t cover_demand = 0;
  for (i_t r = 0; r < h.n_constraints; ++r) {
    if (p.row_max_mag[r] == 0 || p.row_min_mag[r] <= tolerances.absolute_tolerance) {
      return false;
    }
    if (p.row_max_mag[r] - p.row_min_mag[r] > tolerances.absolute_tolerance) { return false; }
    if (!std::isfinite(h.row_ub[r])) {
      const f_t demand = h.row_lb[r] / p.row_max_mag[r];
      if (!is_integer<f_t>(demand, tolerances.integrality_tolerance) ||
          demand < 1 - tolerances.absolute_tolerance) {
        return false;
      }
      cover_demand += std::round(demand);
    }
  }
  return cover_demand > 0 && cover_demand <= arcflow_max_tokens;
}

template <typename f_t>
struct row_info_t {
  row_role_t role{row_role_t::cover};
  f_t scale{1};
  f_t lo{0};
  f_t hi{0};
};

template <typename i_t, typename f_t>
bool classify_rows(const host_problem_t<i_t, f_t>& h,
                   const arcflow_profile_t<i_t, f_t>& profile,
                   const typename mip_solver_settings_t<i_t, f_t>::tolerances_t& tolerances,
                   std::vector<row_info_t<f_t>>& rows)
{
  cuopt_assert((i_t)profile.row_min_mag.size() == h.n_constraints, "Size mismatch");
  cuopt_assert((i_t)profile.row_max_mag.size() == h.n_constraints, "Size mismatch");
  rows.assign(h.n_constraints, row_info_t<f_t>{});

  for (i_t r = 0; r < h.n_constraints; ++r) {
    row_info_t<f_t> info;
    info.scale        = profile.row_max_mag[r];
    const f_t lo      = h.row_lb[r] / info.scale;
    const f_t hi      = h.row_ub[r] / info.scale;
    const bool lo_fin = std::isfinite(lo);
    const bool hi_fin = std::isfinite(hi);

    if (lo_fin && hi_fin) {
      info.role = row_role_t::flow;
      info.lo   = lo;
      info.hi   = hi;
    } else if (lo_fin) {
      if (!is_integer<f_t>(lo, tolerances.integrality_tolerance) ||
          lo < 1 - tolerances.absolute_tolerance) {
        return false;
      }
      info.role = row_role_t::cover;
      info.lo   = std::round(lo);
      info.hi   = hi;
    } else {
      return false;
    }
    rows[r] = info;
  }
  return true;
}

// The nonzero flow right-hand sides must all carry one sign.  That sign says which incidence of an
// arc is its tail, and their absolute sum is the number of paths.  Returns 0 when ambiguous.
template <typename i_t, typename f_t>
f_t supply_orientation(const std::vector<row_info_t<f_t>>& rows,
                       const typename mip_solver_settings_t<i_t, f_t>::tolerances_t& tolerances,
                       f_t& total)
{
  const f_t absolute_tolerance = tolerances.absolute_tolerance;
  f_t positive                 = 0;
  f_t negative                 = 0;
  for (const auto& info : rows) {
    if (info.role != row_role_t::flow) { continue; }
    if (info.lo > absolute_tolerance) { positive += info.lo; }
    if (info.hi < -absolute_tolerance) { negative += -info.hi; }
  }
  if (positive > absolute_tolerance && negative > absolute_tolerance) { return 0; }
  if (positive > absolute_tolerance) {
    total = positive;
    return 1;
  }
  if (negative > absolute_tolerance) {
    total = negative;
    return -1;
  }
  return 0;
}

// One incidence per role at most, so a fourth column entry necessarily duplicates one of the three
// and is rejected without counting entries.
template <typename i_t>
struct column_incidence_t {
  i_t tail{-1};
  i_t head{-1};
  i_t label{-1};
};

// A column carrying a single flow incidence is accepted only when that incidence is the arc's
// tail: sources are read from the right-hand side, so the mirror encoding of an explicit
// injection arc is out of scope here.
template <typename i_t, typename f_t>
bool build_structure(const host_problem_t<i_t, f_t>& h,
                     std::vector<row_info_t<f_t>>& rows,
                     const typename mip_solver_settings_t<i_t, f_t>::tolerances_t& tolerances,
                     arc_flow_model_t<i_t, f_t>& model)
{
  f_t supply_total = 0;
  const f_t sign   = supply_orientation<i_t>(rows, tolerances, supply_total);
  if (sign == 0 || !is_integer<f_t>(supply_total, tolerances.integrality_tolerance)) {
    return false;
  }
  if (std::round(supply_total) != arcflow_paths_supported) { return false; }

  // Reorient so a normalized coefficient of +1 always means "the arc leaves this node".
  if (sign < 0) {
    for (auto& info : rows) {
      if (info.role != row_role_t::flow) { continue; }
      const f_t lo = info.lo;
      info.lo      = -info.hi;
      info.hi      = -lo;
    }
  }

  std::vector<i_t> node_of_row(h.n_constraints, -1);
  std::vector<i_t> label_of_row(h.n_constraints, -1);
  for (i_t r = 0; r < h.n_constraints; ++r) {
    if (rows[r].role == row_role_t::flow) {
      node_of_row[r] = model.n_nodes++;
    } else {
      label_of_row[r] = model.n_labels++;
    }
  }
  if (model.n_nodes == 0 || model.n_labels == 0) { return false; }

  model.demand.assign(model.n_labels, 0);
  model.terminator_col.assign(model.n_nodes, -1);
  model.terminator_cost.assign(model.n_nodes, 0);
  model.terminator_capacity.assign(model.n_nodes, 0);

  int64_t total_demand = 0;
  for (i_t r = 0; r < h.n_constraints; ++r) {
    const i_t l = label_of_row[r];
    if (l < 0) { continue; }
    model.demand[l] = std::round(rows[r].lo);
    total_demand += model.demand[l];
  }
  if (total_demand <= 0 || total_demand > arcflow_max_tokens) { return false; }

  // Negative net outflow encodes path termination after singleton-column substitution.
  for (i_t r = 0; r < h.n_constraints; ++r) {
    const i_t v = node_of_row[r];
    if (v < 0) { continue; }
    const auto& info = rows[r];
    if (info.lo > tolerances.absolute_tolerance) {
      if (std::abs(info.lo - info.hi) > tolerances.absolute_tolerance ||
          !is_integer<f_t>(info.lo, tolerances.integrality_tolerance)) {
        return false;
      }
      model.path_start.insert(model.path_start.end(), (size_t)std::round(info.lo), v);
    } else if (std::abs(info.hi) <= tolerances.absolute_tolerance) {
      if (info.lo < -tolerances.absolute_tolerance) {
        if (!is_integer<f_t>(info.lo, tolerances.integrality_tolerance)) { return false; }
        model.terminator_capacity[v] =
          std::min((f_t)arcflow_paths_supported, std::round(-info.lo));
      }
    } else {
      return false;
    }
  }
  if ((i_t)model.path_start.size() != arcflow_paths_supported) { return false; }

  std::vector<column_incidence_t<i_t>> columns(h.n_variables);
  for (i_t r = 0; r < h.n_constraints; ++r) {
    for (i_t k = h.csr_offsets[r]; k < h.csr_offsets[r + 1]; ++k) {
      const i_t j = h.csr_cols[k];
      cuopt_assert(j >= 0 && j < h.n_variables, "Column index out of range");
      auto& column   = columns[j];
      const f_t unit = h.csr_values[k] / rows[r].scale;
      if (rows[r].role == row_role_t::flow) {
        const f_t oriented = unit * sign;
        if (std::abs(oriented - 1) <= tolerances.absolute_tolerance) {
          if (column.tail >= 0) { return false; }
          column.tail = node_of_row[r];
        } else if (std::abs(oriented + 1) <= tolerances.absolute_tolerance) {
          if (column.head >= 0) { return false; }
          column.head = node_of_row[r];
        } else {
          return false;
        }
      } else {
        // A covering incidence is positive irrespective of the flow orientation.
        if (std::abs(unit - 1) > tolerances.absolute_tolerance) { return false; }
        if (column.label >= 0) { return false; }
        column.label = label_of_row[r];
      }
    }
  }

  for (i_t j = 0; j < h.n_variables; ++j) {
    if (h.var_types[j] != var_t::INTEGER) { return false; }
    if (!std::isfinite(h.obj[j])) { return false; }
    if (std::abs(h.var_lb[j]) > tolerances.absolute_tolerance) { return false; }
    const f_t ub = h.var_ub[j];
    if (!std::isfinite(ub) || ub < 1 - tolerances.absolute_tolerance) { return false; }

    const i_t tail  = columns[j].tail;
    const i_t head  = columns[j].head;
    const i_t label = columns[j].label;
    if (tail >= 0 && head >= 0) {
      if (label < 0) { return false; }
      // With this bound the arc capacity can never bind: a use of the arc consumes one of the
      // label's tokens, and there are exactly demand[label] of them.
      if ((f_t)model.demand[label] > ub + tolerances.absolute_tolerance) { return false; }
      model.arcs.push_back(arc_t<i_t, f_t>{tail, head, label, j, h.obj[j]});
    } else if (tail >= 0 && label < 0) {
      // Explicit loss arc: it leaves the node and never arrives, so it ends a path.  A node that
      // already absorbs paths through row slack would need the two capacities apportioned, which
      // no model in this family does, so reject rather than guess.
      if (model.terminator_col[tail] >= 0 || model.terminator_capacity[tail] > 0) { return false; }
      model.terminator_col[tail]  = j;
      model.terminator_cost[tail] = h.obj[j];
      model.terminator_capacity[tail] =
        std::min((f_t)arcflow_paths_supported, std::floor(ub + tolerances.absolute_tolerance));
    } else {
      return false;
    }
  }

  if (model.arcs.empty()) { return false; }

  std::stable_sort(model.arcs.begin(),
                   model.arcs.end(),
                   [](const arc_t<i_t, f_t>& a, const arc_t<i_t, f_t>& b) {
                     if (a.label != b.label) { return a.label < b.label; }
                     if (a.from != b.from) { return a.from < b.from; }
                     return a.cost < b.cost;
                   });
  model.arc_offset.assign(model.n_labels + 1, 0);
  for (const auto& arc : model.arcs) {
    model.arc_offset[arc.label + 1]++;
  }
  for (i_t l = 0; l < model.n_labels; ++l) {
    model.arc_offset[l + 1] += model.arc_offset[l];
    if (model.arc_offset[l] == model.arc_offset[l + 1]) { return false; }
  }
  cuopt_assert(model.arc_offset.back() == (i_t)model.arcs.size(),
               "arc CSR offsets must cover every arc");
  return true;
}

// The potential is recovered from the objective rather than from any index: within a label the
// cost is affine in the potential of the arc's tail, so one label with enough distinct costs fixes
// the potential on every node it touches, and the remaining labels are fitted and extended from
// there.  The result is an affine image of the true potential, which leaves the order that
// consumes it unchanged, since a common positive factor cancels out of every comparison.
template <typename i_t, typename f_t>
bool derive_potential(arc_flow_model_t<i_t, f_t>& model,
                      const typename mip_solver_settings_t<i_t, f_t>::tolerances_t& tolerances,
                      const std::atomic<bool>& preemption_flag)
{
  const i_t n_labels = model.n_labels;

  i_t reference     = -1;
  size_t best_count = 0;
  std::vector<f_t> costs;
  for (i_t l = 0; l < n_labels; ++l) {
    costs.clear();
    for (i_t k = model.arc_offset[l]; k < model.arc_offset[l + 1]; ++k) {
      costs.push_back(model.arcs[k].cost);
    }
    std::sort(costs.begin(), costs.end());
    const size_t distinct = (size_t)(std::unique(costs.begin(), costs.end()) - costs.begin());
    if (distinct > best_count) {
      best_count = distinct;
      reference  = l;
    }
  }
  if (reference < 0 || best_count < 2) { return false; }

  const f_t unknown = std::numeric_limits<f_t>::quiet_NaN();
  model.phi.assign(model.n_nodes, unknown);
  model.slope.assign(n_labels, unknown);
  model.displacement.assign(n_labels, unknown);
  std::vector<f_t> intercept(n_labels, unknown);

  for (i_t k = model.arc_offset[reference]; k < model.arc_offset[reference + 1]; ++k) {
    model.phi[model.arcs[k].from] = model.arcs[k].cost;
  }

  // Each productive round resolves at least one potential, slope, or displacement.
  const long max_rounds = 2L * n_labels + model.n_nodes + 2L;
  long rounds           = 0;
  for (; rounds < max_rounds; ++rounds) {
    if (preemption_flag.load()) { return false; }
    bool progress = false;
    for (i_t l = 0; l < n_labels; ++l) {
      const i_t begin = model.arc_offset[l];
      const i_t end   = model.arc_offset[l + 1];

      if (!is_known(model.slope[l])) {
        i_t lowest  = -1;
        i_t highest = -1;
        for (i_t k = begin; k < end; ++k) {
          const f_t p = model.phi[model.arcs[k].from];
          if (!is_known(p)) { continue; }
          if (lowest < 0 || p < model.phi[model.arcs[lowest].from]) { lowest = k; }
          if (highest < 0 || p > model.phi[model.arcs[highest].from]) { highest = k; }
        }
        if (lowest >= 0 && highest >= 0) {
          const f_t lo_phi = model.phi[model.arcs[lowest].from];
          const f_t hi_phi = model.phi[model.arcs[highest].from];
          if (std::abs(lo_phi - hi_phi) > tolerances.absolute_tolerance) {
            model.slope[l] =
              (model.arcs[highest].cost - model.arcs[lowest].cost) / (hi_phi - lo_phi);
            intercept[l] = model.arcs[lowest].cost - model.slope[l] * lo_phi;
            progress     = true;
          }
        }
      }
      if (is_known(model.slope[l]) && std::abs(model.slope[l]) > tolerances.absolute_tolerance) {
        for (i_t k = begin; k < end; ++k) {
          const i_t from = model.arcs[k].from;
          if (is_known(model.phi[from])) { continue; }
          model.phi[from] = (model.arcs[k].cost - intercept[l]) / model.slope[l];
          progress        = true;
        }
      }

      if (!is_known(model.displacement[l])) {
        for (i_t k = begin; k < end; ++k) {
          const f_t from = model.phi[model.arcs[k].from];
          const f_t to   = model.phi[model.arcs[k].to];
          if (is_known(from) && is_known(to)) {
            model.displacement[l] = to - from;
            progress              = true;
            break;
          }
        }
      }
      if (is_known(model.displacement[l])) {
        for (i_t k = begin; k < end; ++k) {
          const i_t from = model.arcs[k].from;
          const i_t to   = model.arcs[k].to;
          if (is_known(model.phi[from]) && !is_known(model.phi[to])) {
            model.phi[to] = model.phi[from] + model.displacement[l];
            progress      = true;
          } else if (is_known(model.phi[to]) && !is_known(model.phi[from])) {
            model.phi[from] = model.phi[to] - model.displacement[l];
            progress        = true;
          }
        }
      }
    }
    if (!progress) { break; }
  }
  cuopt_assert(rounds < max_rounds, "propagation must reach a fixpoint within its progress bound");

  for (f_t p : model.phi) {
    if (!is_known(p)) { return false; }
  }
  for (f_t p : model.displacement) {
    if (!is_known(p)) { return false; }
  }

  f_t phi_scale = 0;
  for (f_t p : model.phi) {
    phi_scale = std::max(phi_scale, std::abs(p));
  }
  if (phi_scale <= tolerances.absolute_tolerance) { return false; }

  // Orient the potential so displacements are positive.  A consistent potential whose
  // displacements are all strictly positive is exactly what makes the arc graph acyclic: a cycle
  // would need its displacements to sum to zero.
  f_t displacement_sum = 0;
  for (f_t p : model.displacement) {
    displacement_sum += p;
  }
  if (displacement_sum < 0) {
    for (auto& p : model.phi) {
      p = -p;
    }
    for (auto& p : model.displacement) {
      p = -p;
    }
    for (auto& w : model.slope) {
      w = -w;
    }
  }
  for (f_t p : model.displacement) {
    if (p <= tolerances.absolute_tolerance) { return false; }
  }

  // Smith ordering assumes nonnegative job weights.
  for (f_t w : model.slope) {
    if (is_known(w) && w < -tolerances.absolute_tolerance) { return false; }
  }

  // Verify against every arc, not just the two points each fit was built from.  The cost residual
  // matters as much as the potential: the order is read off the fitted slopes, so a label whose
  // costs are not affine in the potential would be ordered on a meaningless quantity.
  for (const auto& arc : model.arcs) {
    const f_t displaced = model.phi[arc.from] + model.displacement[arc.label];
    if (std::abs(model.phi[arc.to] - displaced) > tolerances.absolute_tolerance) { return false; }
    if (!is_known(model.slope[arc.label])) { continue; }
    const f_t predicted = model.slope[arc.label] * model.phi[arc.from] + intercept[arc.label];
    if (std::abs(predicted - arc.cost) > tolerances.absolute_tolerance) { return false; }
  }
  return true;
}

// Weighted shortest processing time orders labels by decreasing slope over displacement.
template <typename i_t, typename f_t>
std::vector<i_t> token_order(const arc_flow_model_t<i_t, f_t>& model,
                             const typename mip_solver_settings_t<i_t, f_t>::tolerances_t& tolerances,
                             bool& all_slopes_identified)
{
  std::vector<f_t> lowest_phi(model.n_labels, 0);
  for (i_t l = 0; l < model.n_labels; ++l) {
    f_t lowest = std::numeric_limits<f_t>::infinity();
    for (i_t k = model.arc_offset[l]; k < model.arc_offset[l + 1]; ++k) {
      lowest = std::min(lowest, model.phi[model.arcs[k].from]);
    }
    lowest_phi[l] = lowest;
  }

  std::vector<i_t> ordered;
  std::vector<i_t> unidentified;
  ordered.reserve(model.n_labels);
  for (i_t l = 0; l < model.n_labels; ++l) {
    if (is_known(model.slope[l])) {
      ordered.push_back(l);
    } else {
      unidentified.push_back(l);
    }
  }

  // Approximate equality is not transitive. Tolerance forms ratio classes before the final sort.
  std::sort(ordered.begin(), ordered.end(), [&](i_t a, i_t b) {
    const _Float128 lhs = (_Float128)model.slope[a] * (_Float128)model.displacement[b];
    const _Float128 rhs = (_Float128)model.slope[b] * (_Float128)model.displacement[a];
    if (lhs != rhs) { return lhs > rhs; }
    return a < b;
  });
  std::vector<i_t> ratio_class(model.n_labels, 0);
  for (size_t position = 1; position < ordered.size(); ++position) {
    const i_t previous  = ordered[position - 1];
    const i_t current   = ordered[position];
    const _Float128 lhs = (_Float128)model.slope[previous] * (_Float128)model.displacement[current];
    const _Float128 rhs = (_Float128)model.slope[current] * (_Float128)model.displacement[previous];
    const _Float128 difference = lhs > rhs ? lhs - rhs : rhs - lhs;
    const bool tied            = difference <= (_Float128)tolerances.absolute_tolerance;
    ratio_class[current]       = ratio_class[previous] + (tied ? 0 : 1);
  }
  std::sort(ordered.begin(), ordered.end(), [&](i_t a, i_t b) {
    if (ratio_class[a] != ratio_class[b]) { return ratio_class[a] < ratio_class[b]; }
    if (lowest_phi[a] != lowest_phi[b]) { return lowest_phi[a] < lowest_phi[b]; }
    if (model.displacement[a] != model.displacement[b]) {
      return model.displacement[a] < model.displacement[b];
    }
    return a < b;
  });

  int64_t total_demand = 0;
  for (int64_t d : model.demand) {
    total_demand += d;
  }
  std::vector<i_t> tokens;
  tokens.reserve((size_t)total_demand);
  for (i_t l : ordered) {
    tokens.insert(tokens.end(), (size_t)model.demand[l], l);
  }

  // Labels without fitted slopes are placed at their first reachable potential.
  all_slopes_identified = unidentified.empty();
  std::sort(unidentified.begin(), unidentified.end(), [&](i_t a, i_t b) {
    if (lowest_phi[a] != lowest_phi[b]) { return lowest_phi[a] < lowest_phi[b]; }
    return a < b;
  });
  for (i_t l : unidentified) {
    f_t consumed = 0;
    size_t slot  = 0;
    while (slot < tokens.size() && consumed < lowest_phi[l]) {
      consumed += model.displacement[tokens[slot]];
      ++slot;
    }
    tokens.insert(tokens.begin() + (ptrdiff_t)slot, (size_t)model.demand[l], l);
  }
  return tokens;
}

template <typename i_t, typename f_t>
std::optional<arc_flow_result_t<i_t>> run_dp(const arc_flow_model_t<i_t, f_t>& model,
                                             const std::vector<i_t>& tokens,
                                             const std::atomic<bool>& preemption_flag)
{
  const i_t n_tokens = tokens.size();
  if (n_tokens == 0) { return std::nullopt; }

  arc_flow_result_t<i_t> result;

  // The reconstruction budget is charged against retained states at each level.
  size_t retained_bytes = 0;

  std::vector<std::vector<parent_t<i_t>>> history;
  history.reserve((size_t)n_tokens);

  frontier_t<i_t, f_t> root;
  for (i_t k = 0; k < arcflow_paths_supported; ++k) {
    root.node[k] = model.path_start[k];
  }
  std::sort(root.node.begin(), root.node.end());
  std::vector<frontier_t<i_t, f_t>> current{root};

  std::vector<candidate_t<i_t, f_t>> candidates;
  std::vector<frontier_t<i_t, f_t>> next;
  std::vector<parent_t<i_t>> parents;
  for (i_t t = 0; t < n_tokens; ++t) {
    if (preemption_flag.load()) { return std::nullopt; }
    const i_t label      = tokens[t];
    const auto arc_begin = model.arcs.begin() + model.arc_offset[label];
    const auto arc_end   = model.arcs.begin() + model.arc_offset[label + 1];

    const size_t candidate_limit = arcflow_candidate_bytes_max / sizeof(candidate_t<i_t, f_t>);
    size_t candidate_count       = 0;
    for (const auto& entry : current) {
      for (i_t k = 0; k < arcflow_paths_supported; ++k) {
        const i_t node   = entry.node[k];
        const auto begin = std::lower_bound(
          arc_begin, arc_end, node, [](const arc_t<i_t, f_t>& a, i_t v) { return a.from < v; });
        const auto end = std::upper_bound(
          begin, arc_end, node, [](i_t v, const arc_t<i_t, f_t>& a) { return v < a.from; });
        const size_t added = end - begin;
        if (added > candidate_limit - candidate_count) { return std::nullopt; }
        candidate_count += added;
      }
    }

    candidates.clear();
    candidates.reserve(candidate_count);
    for (i_t i = 0; i < (i_t)current.size(); ++i) {
      const frontier_t<i_t, f_t>& entry = current[i];
      for (i_t k = 0; k < arcflow_paths_supported; ++k) {
        const i_t node   = entry.node[k];
        const auto begin = std::lower_bound(
          arc_begin, arc_end, node, [](const arc_t<i_t, f_t>& a, i_t v) { return a.from < v; });
        const auto end = std::upper_bound(
          begin, arc_end, node, [](i_t v, const arc_t<i_t, f_t>& a) { return v < a.from; });
        for (auto it = begin; it != end; ++it) {
          candidate_t<i_t, f_t> candidate;
          candidate.front         = entry;
          candidate.front.node[k] = it->to;
          std::sort(candidate.front.node.begin(), candidate.front.node.end());
          candidate.front.cost = entry.cost + it->cost;
          candidate.parent     = parent_t<i_t>{i, (i_t)(it - model.arcs.begin())};
          candidates.push_back(candidate);
        }
      }
    }
    if (candidates.empty()) { return std::nullopt; }

    // A total order makes representative selection independent of enumeration order.
    std::sort(candidates.begin(),
              candidates.end(),
              [](const candidate_t<i_t, f_t>& a, const candidate_t<i_t, f_t>& b) {
                if (a.front.node != b.front.node) { return a.front.node < b.front.node; }
                if (a.front.cost != b.front.cost) { return a.front.cost < b.front.cost; }
                if (a.parent.prev != b.parent.prev) { return a.parent.prev < b.parent.prev; }
                return a.parent.arc < b.parent.arc;
              });
    candidates.erase(
      std::unique(candidates.begin(),
                  candidates.end(),
                  [](const candidate_t<i_t, f_t>& a, const candidate_t<i_t, f_t>& b) {
                    return a.front.node == b.front.node;
                  }),
      candidates.end());

    const size_t remaining =
      arcflow_history_bytes_max > retained_bytes ? arcflow_history_bytes_max - retained_bytes : 0;
    const size_t affordable = std::max<size_t>(remaining / sizeof(parent_t<i_t>), 1);
    if (candidates.size() > affordable) {
      std::stable_sort(candidates.begin(),
                       candidates.end(),
                       [](const candidate_t<i_t, f_t>& a, const candidate_t<i_t, f_t>& b) {
                         if (a.front.cost != b.front.cost) { return a.front.cost < b.front.cost; }
                         return a.front.node < b.front.node;
                       });
      candidates.resize(affordable);
      std::sort(candidates.begin(),
                candidates.end(),
                [](const candidate_t<i_t, f_t>& a, const candidate_t<i_t, f_t>& b) {
                  return a.front.node < b.front.node;
                });
      result.exact = false;
    }

    next.clear();
    parents.clear();
    next.reserve(candidates.size());
    parents.reserve(candidates.size());
    for (const auto& candidate : candidates) {
      next.push_back(candidate.front);
      parents.push_back(candidate.parent);
    }
    retained_bytes += parents.size() * sizeof(parent_t<i_t>);
    history.push_back(std::move(parents));
    current.swap(next);
  }

  i_t best_index = -1;
  f_t best_total = std::numeric_limits<f_t>::infinity();
  for (i_t i = 0; i < (i_t)current.size(); ++i) {
    const frontier_t<i_t, f_t>& entry = current[i];
    f_t total                         = entry.cost;
    bool closable                     = true;
    for (i_t k = 0; k < arcflow_paths_supported && closable; ++k) {
      const i_t node  = entry.node[k];
      int64_t sharing = 0;
      for (i_t q = 0; q < arcflow_paths_supported; ++q) {
        if (entry.node[q] == node) { sharing++; }
      }
      if (model.terminator_capacity[node] < sharing) {
        closable = false;
      } else {
        total += model.terminator_cost[node];
      }
    }
    if (closable && total < best_total) {
      best_total = total;
      best_index = i;
    }
  }
  if (best_index < 0) { return std::nullopt; }

  for (i_t k = 0; k < arcflow_paths_supported; ++k) {
    const i_t col = model.terminator_col[current[best_index].node[k]];
    if (col >= 0) { result.columns.push_back(col); }
  }
  i_t index = best_index;
  for (i_t t = n_tokens; t > 0; --t) {
    const parent_t<i_t>& step = history[t - 1][index];
    cuopt_assert(step.arc >= 0, "every level beyond the root records the arc it consumed");
    result.columns.push_back(model.arcs[step.arc].col);
    index = step.prev;
  }
  cuopt_assert(index == 0, "reconstruction must terminate at the root state");
  return result;
}

}  // namespace

template <typename i_t, typename f_t>
struct arc_flow_t<i_t, f_t>::host_state_t {
  host_state_t(host_problem_t<i_t, f_t>&& problem, arcflow_profile_t<i_t, f_t>&& profile)
    : h(std::move(problem)), profile(std::move(profile))
  {
  }

  host_problem_t<i_t, f_t> h;
  arcflow_profile_t<i_t, f_t> profile;
};

template <typename i_t, typename f_t>
arc_flow_t<i_t, f_t>::arc_flow_t() = default;

template <typename i_t, typename f_t>
arc_flow_t<i_t, f_t>::~arc_flow_t() = default;

template <typename i_t, typename f_t>
bool arc_flow_t<i_t, f_t>::recognize(
  const optimization_problem_t<i_t, f_t>& op_problem,
  const typename mip_solver_settings_t<i_t, f_t>::tolerances_t& tolerances)
{
  const i_t n_variables   = op_problem.get_n_variables();
  const i_t n_constraints = op_problem.get_n_constraints();
  if (!arcflow_accepts_shape(n_variables, n_constraints, op_problem.get_nnz())) { return false; }
  if (op_problem.get_n_integers() != n_variables) { return false; }
  if (!op_problem.get_variable_lower_bounds().is_empty() &&
      (i_t)op_problem.get_variable_lower_bounds().size() != n_variables) {
    return false;
  }
  if ((i_t)op_problem.get_variable_upper_bounds().size() != n_variables) { return false; }

  auto stream = op_problem.get_handle_ptr()->get_stream();

  host_problem_t<i_t, f_t> h;
  h.n_variables   = n_variables;
  h.n_constraints = n_constraints;
  h.row_lb        = cuopt::host_copy(op_problem.get_constraint_lower_bounds(), stream);
  h.row_ub        = cuopt::host_copy(op_problem.get_constraint_upper_bounds(), stream);
  if (!arcflow_accepts_bounds(h.row_lb, h.row_ub)) { return false; }

  h.csr_values  = cuopt::host_copy(op_problem.get_constraint_matrix_values(), stream);
  h.csr_cols    = cuopt::host_copy(op_problem.get_constraint_matrix_indices(), stream);
  h.csr_offsets = cuopt::host_copy(op_problem.get_constraint_matrix_offsets(), stream);
  arcflow_profile_t<i_t, f_t> profile;
  if (!arcflow_accepts_profile(h, tolerances, profile)) { return false; }

  h.obj = cuopt::host_copy(op_problem.get_objective_coefficients(), stream);
  if (op_problem.get_sense()) {
    for (auto& coefficient : h.obj) {
      coefficient = -coefficient;
    }
  }
  h.var_lb.assign(n_variables, f_t{0});
  if (!op_problem.get_variable_lower_bounds().is_empty()) {
    h.var_lb = cuopt::host_copy(op_problem.get_variable_lower_bounds(), stream);
  }
  h.var_ub    = cuopt::host_copy(op_problem.get_variable_upper_bounds(), stream);
  h.var_types = cuopt::host_copy(op_problem.get_variable_types(), stream);
  state_      = std::make_unique<host_state_t>(std::move(h), std::move(profile));
  return true;
}

template <typename i_t, typename f_t>
bool arc_flow_t<i_t, f_t>::recognize(
  const problem_t<i_t, f_t>& problem,
  const typename mip_solver_settings_t<i_t, f_t>::tolerances_t& tolerances)
{
  const i_t n_variables   = problem.n_variables;
  const i_t n_constraints = problem.n_constraints;
  if (!arcflow_accepts_shape(n_variables, n_constraints, problem.nnz)) { return false; }
  if (problem.n_integer_vars != n_variables) { return false; }

  auto stream = problem.handle_ptr->get_stream();

  host_problem_t<i_t, f_t> h;
  h.n_variables   = n_variables;
  h.n_constraints = n_constraints;
  h.csr_values    = cuopt::host_copy(problem.coefficients, stream);
  h.csr_cols      = cuopt::host_copy(problem.variables, stream);
  h.csr_offsets   = cuopt::host_copy(problem.offsets, stream);
  h.row_lb        = cuopt::host_copy(problem.constraint_lower_bounds, stream);
  h.row_ub        = cuopt::host_copy(problem.constraint_upper_bounds, stream);
  if (!arcflow_accepts_bounds(h.row_lb, h.row_ub)) { return false; }

  arcflow_profile_t<i_t, f_t> profile;
  if (!arcflow_accepts_profile(h, tolerances, profile)) { return false; }

  h.obj       = cuopt::host_copy(problem.objective_coefficients, stream);
  h.var_types = cuopt::host_copy(problem.variable_types, stream);
  std::tie(h.var_lb, h.var_ub) =
    cuopt::extract_host_bounds<f_t>(problem.variable_bounds, problem.handle_ptr);
  state_ = std::make_unique<host_state_t>(std::move(h), std::move(profile));
  return true;
}

template <typename i_t, typename f_t>
bool arc_flow_t<i_t, f_t>::solve(
  const typename mip_solver_settings_t<i_t, f_t>::tolerances_t& tolerances,
  std::atomic<bool>& preemption,
  std::vector<f_t>& assignment)
{
  cuopt_assert(state_ != nullptr, "solve called without a successful recognize");
  const auto& h = state_->h;

  std::vector<row_info_t<f_t>> rows;
  if (!classify_rows(h, state_->profile, tolerances, rows)) {
    CUOPT_LOG_DEBUG("[ArcFlow] rejected: rows are not unit incidence after normalization");
    return false;
  }

  arc_flow_model_t<i_t, f_t> model;
  if (!build_structure(h, rows, tolerances, model)) {
    CUOPT_LOG_DEBUG("[ArcFlow] rejected: columns do not match the labelled arc pattern");
    return false;
  }
  if (preemption.load()) { return false; }

  if (!derive_potential(model, tolerances, preemption)) {
    CUOPT_LOG_DEBUG("[ArcFlow] rejected: no consistent potential and affine cost model");
    return false;
  }
  if (preemption.load()) { return false; }

  bool all_slopes_identified = true;
  const auto tokens          = token_order(model, tolerances, all_slopes_identified);
  CUOPT_LOG_DEBUG("[ArcFlow] detected %d nodes, %d labels, %d paths, %zu tokens, ordering %s",
                  (int)model.n_nodes,
                  (int)model.n_labels,
                  arcflow_paths_supported,
                  tokens.size(),
                  all_slopes_identified ? "identified" : "partly by reachability");

  const auto result = run_dp(model, tokens, preemption);
  if (!result.has_value()) {
    CUOPT_LOG_DEBUG("[ArcFlow] no complete path set found in the ordered family");
    return false;
  }
  CUOPT_LOG_DEBUG("[ArcFlow] search %s", result->exact ? "exact" : "beamed by the history budget");

  assignment.assign((size_t)h.n_variables, f_t{0});
  for (i_t col : result->columns) {
    assignment[col] += f_t{1};
  }
  return true;
}

#if MIP_INSTANTIATE_FLOAT
template class arc_flow_t<int, float>;
#endif

#if MIP_INSTANTIATE_DOUBLE
template class arc_flow_t<int, double>;
#endif

}  // namespace cuopt::mathematical_optimization::mip
