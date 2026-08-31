/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#include "arc_flow.cuh"

#include <mip_heuristics/mip_constants.hpp>

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

constexpr int arcflow_paths_supported = 2;
constexpr int arcflow_max_tokens      = 20000;
constexpr int arcflow_max_col_entries = 3;
constexpr size_t arcflow_history_bytes_max = size_t{32} << 20;
constexpr size_t arcflow_candidate_bytes_max = size_t{32} << 20;

// Structural inference reads objective coefficients and matrix entries at f_t precision, so the
// residual a genuine arc-flow model leaves behind is bounded by that precision and not by double.
struct arcflow_tol_t {
  double abs{1e-9};
  double rel{1e-9};
};

template <typename f_t>
arcflow_tol_t structural_tolerance()
{
  const double eps = 8.0 * (double)(std::numeric_limits<f_t>::epsilon());
  return arcflow_tol_t{std::max(1e-9, eps), std::max(1e-9, eps)};
}

enum class row_role_t : uint8_t { flow, cover };

struct arc_t {
  int from{-1};
  int to{-1};
  int label{-1};
  int col{-1};
  double cost{0.0};
};

struct arc_flow_model_t {
  int n_nodes{0};
  int n_labels{0};
  arcflow_tol_t tol;

  std::vector<double> phi;
  std::vector<int> path_start;
  std::vector<int64_t> demand;
  std::vector<double> displacement;
  std::vector<double> slope;
  std::vector<int> arc_offset;
  std::vector<arc_t> arcs;

  // A negative terminator_col denotes conservation-row slack.
  std::vector<int> terminator_col;
  std::vector<double> terminator_cost;
  std::vector<int64_t> terminator_capacity;
};

struct frontier_t {
  std::array<int, arcflow_paths_supported> node{};
  double cost{0.0};
};

struct parent_t {
  int prev{-1};
  int arc{-1};
};

struct candidate_t {
  frontier_t front;
  parent_t parent;
};

struct arc_flow_result_t {
  std::vector<int> columns;
  bool exact{true};
};

bool close_to(double a, double b, double scale, const arcflow_tol_t& tol)
{
  return std::abs(a - b) <= tol.abs + tol.rel * scale;
}

bool is_integral(double v, const arcflow_tol_t& tol)
{
  return std::abs(v - std::round(v)) <= tol.abs + tol.rel * std::abs(v);
}

bool is_known(double v) { return !std::isnan(v); }

struct arcflow_profile_t {
  arcflow_profile_t(int64_t n_variables = 0, int64_t n_constraints = 0)
    : col_entries(n_variables, 0),
      row_min_mag(n_constraints, std::numeric_limits<double>::infinity()),
      row_max_mag(n_constraints, 0.0)
  {
  }

  std::vector<int64_t> col_entries;
  std::vector<double> row_min_mag;
  std::vector<double> row_max_mag;
};

template <typename i_t, typename f_t>
struct host_problem_t {
  i_t n_variables{0};
  i_t n_constraints{0};
  arcflow_tol_t tol;
  std::vector<f_t> csc_values;
  std::vector<i_t> csc_rows;
  std::vector<i_t> csc_offsets;
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
    const bool lo_fin = std::isfinite((double)row_lb[r]);
    const bool hi_fin = std::isfinite((double)row_ub[r]);
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

template <typename i_t, typename f_t>
arcflow_profile_t profile_from_csr(i_t n_variables,
                                   i_t n_constraints,
                                   const std::vector<f_t>& csr_values,
                                   const std::vector<i_t>& csr_cols,
                                   const std::vector<i_t>& csr_offsets)
{
  arcflow_profile_t p(n_variables, n_constraints);
  for (i_t r = 0; r < n_constraints; ++r) {
    for (i_t k = csr_offsets[r]; k < csr_offsets[r + 1]; ++k) {
      const i_t col = csr_cols[k];
      cuopt_assert(col >= 0 && col < n_variables, "Column index out of range");
      const double mag = std::abs((double)csr_values[k]);
      ++p.col_entries[col];
      p.row_min_mag[r] = std::min(p.row_min_mag[r], mag);
      p.row_max_mag[r] = std::max(p.row_max_mag[r], mag);
    }
  }
  return p;
}

template <typename i_t, typename f_t>
arcflow_profile_t profile_from_csc(i_t n_variables,
                                   i_t n_constraints,
                                   const std::vector<f_t>& csc_values,
                                   const std::vector<i_t>& csc_rows,
                                   const std::vector<i_t>& csc_offsets)
{
  arcflow_profile_t p(n_variables, n_constraints);
  for (i_t j = 0; j < n_variables; ++j) {
    p.col_entries[j] = csc_offsets[j + 1] - csc_offsets[j];
    for (i_t k = csc_offsets[j]; k < csc_offsets[j + 1]; ++k) {
      const i_t row = csc_rows[k];
      cuopt_assert(row >= 0 && row < n_constraints, "Row index out of range");
      const double mag   = std::abs((double)csc_values[k]);
      p.row_min_mag[row] = std::min(p.row_min_mag[row], mag);
      p.row_max_mag[row] = std::max(p.row_max_mag[row], mag);
    }
  }
  return p;
}

template <typename f_t>
bool arcflow_accepts_profile(const arcflow_profile_t& p,
                             const std::vector<f_t>& row_lb,
                             const std::vector<f_t>& row_ub,
                             const arcflow_tol_t& tol)
{
  for (const int64_t entries : p.col_entries) {
    if (entries > arcflow_max_col_entries) { return false; }
  }

  double cover_demand = 0.0;
  for (size_t r = 0; r < p.row_max_mag.size(); ++r) {
    if (p.row_max_mag[r] == 0.0 || p.row_min_mag[r] <= tol.abs) { return false; }
    if (!close_to(p.row_min_mag[r], p.row_max_mag[r], p.row_max_mag[r], tol)) { return false; }
    if (!std::isfinite((double)row_ub[r])) {
      const double demand = row_lb[r] / p.row_max_mag[r];
      if (!is_integral(demand, tol) || demand < 1.0 - tol.abs) { return false; }
      cover_demand += std::round(demand);
    }
  }
  return cover_demand > 0.0 && cover_demand <= arcflow_max_tokens;
}

// The model arrives row major and the detector reads it column major, so the transpose is built
// here by counting sort.  Rows stay ascending within a column, which build_structure relies on.
template <typename i_t, typename f_t>
void transpose_into_csc(const std::vector<f_t>& csr_values,
                        const std::vector<i_t>& csr_cols,
                        const std::vector<i_t>& csr_offsets,
                        host_problem_t<i_t, f_t>& h)
{
  const size_t nnz = csr_values.size();
  h.csc_offsets.assign((size_t)h.n_variables + 1, 0);
  for (size_t k = 0; k < nnz; ++k) {
    ++h.csc_offsets[(size_t)csr_cols[k] + 1];
  }
  for (i_t j = 0; j < h.n_variables; ++j) {
    h.csc_offsets[j + 1] += h.csc_offsets[j];
  }
  h.csc_rows.assign(nnz, 0);
  h.csc_values.assign(nnz, f_t{0});
  std::vector<i_t> cursor(h.csc_offsets.begin(), h.csc_offsets.end() - 1);
  for (i_t r = 0; r < h.n_constraints; ++r) {
    for (i_t k = csr_offsets[r]; k < csr_offsets[r + 1]; ++k) {
      const i_t slot     = cursor[csr_cols[k]]++;
      h.csc_rows[slot]   = r;
      h.csc_values[slot] = csr_values[k];
    }
  }
}

struct row_info_t {
  row_role_t role{row_role_t::cover};
  double scale{1.0};
  double lo{0.0};
  double hi{0.0};
};

template <typename i_t, typename f_t>
bool classify_rows(const host_problem_t<i_t, f_t>& h,
                   const arcflow_profile_t& profile,
                   std::vector<row_info_t>& rows)
{
  cuopt_assert((i_t)profile.row_min_mag.size() == h.n_constraints, "Size mismatch");
  cuopt_assert((i_t)profile.row_max_mag.size() == h.n_constraints, "Size mismatch");
  rows.assign(h.n_constraints, row_info_t{});

  for (int r = 0; r < h.n_constraints; ++r) {
    row_info_t info;
    info.scale        = profile.row_max_mag[r];
    const double lo   = h.row_lb[r] / info.scale;
    const double hi   = h.row_ub[r] / info.scale;
    const bool lo_fin = std::isfinite(lo);
    const bool hi_fin = std::isfinite(hi);

    if (lo_fin && hi_fin) {
      info.role = row_role_t::flow;
      info.lo   = lo;
      info.hi   = hi;
    } else if (lo_fin) {
      if (!is_integral(lo, h.tol) || lo < 1.0 - h.tol.abs) { return false; }
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
double supply_orientation(const std::vector<row_info_t>& rows,
                          const arcflow_tol_t& tol,
                          double& total)
{
  double positive = 0.0;
  double negative = 0.0;
  for (const auto& info : rows) {
    if (info.role != row_role_t::flow) { continue; }
    if (info.lo > tol.abs) { positive += info.lo; }
    if (info.hi < -tol.abs) { negative += -info.hi; }
  }
  if (positive > tol.abs && negative > tol.abs) { return 0.0; }
  if (positive > tol.abs) {
    total = positive;
    return 1.0;
  }
  if (negative > tol.abs) {
    total = negative;
    return -1.0;
  }
  return 0.0;
}

// A column carrying a single flow incidence is accepted only when that incidence is the arc's
// tail: sources are read from the right-hand side, so the mirror encoding of an explicit
// injection arc is out of scope here.
template <typename i_t, typename f_t>
bool build_structure(const host_problem_t<i_t, f_t>& h,
                     std::vector<row_info_t>& rows,
                     arc_flow_model_t& model)
{
  model.tol = h.tol;

  double supply_total = 0.0;
  const double sign   = supply_orientation(rows, h.tol, supply_total);
  if (sign == 0.0 || !is_integral(supply_total, h.tol)) { return false; }
  if (std::round(supply_total) != arcflow_paths_supported) { return false; }

  // Reorient so a normalized coefficient of +1 always means "the arc leaves this node".
  if (sign < 0.0) {
    for (auto& info : rows) {
      if (info.role != row_role_t::flow) { continue; }
      const double lo = info.lo;
      info.lo         = -info.hi;
      info.hi         = -lo;
    }
  }

  std::vector<int> node_of_row(h.n_constraints, -1);
  std::vector<int> label_of_row(h.n_constraints, -1);
  for (int r = 0; r < h.n_constraints; ++r) {
    if (rows[r].role == row_role_t::flow) {
      node_of_row[r] = model.n_nodes++;
    } else {
      label_of_row[r] = model.n_labels++;
    }
  }
  if (model.n_nodes == 0 || model.n_labels == 0) { return false; }

  model.demand.assign(model.n_labels, 0);
  model.terminator_col.assign(model.n_nodes, -1);
  model.terminator_cost.assign(model.n_nodes, 0.0);
  model.terminator_capacity.assign(model.n_nodes, 0);

  int64_t total_demand = 0;
  for (int r = 0; r < h.n_constraints; ++r) {
    const int l = label_of_row[r];
    if (l < 0) { continue; }
    model.demand[l] = std::round(rows[r].lo);
    total_demand += model.demand[l];
  }
  if (total_demand <= 0 || total_demand > arcflow_max_tokens) { return false; }

  // Negative net outflow encodes path termination after singleton-column substitution.
  for (int r = 0; r < h.n_constraints; ++r) {
    const int v = node_of_row[r];
    if (v < 0) { continue; }
    const auto& info = rows[r];
    if (info.lo > h.tol.abs) {
      if (!close_to(info.lo, info.hi, std::abs(info.lo), h.tol) || !is_integral(info.lo, h.tol)) {
        return false;
      }
      model.path_start.insert(model.path_start.end(), (size_t)std::round(info.lo), v);
    } else if (std::abs(info.hi) <= h.tol.abs) {
      if (info.lo < -h.tol.abs) {
        if (!is_integral(info.lo, h.tol)) { return false; }
        model.terminator_capacity[v] =
          std::min((double)arcflow_paths_supported, std::round(-info.lo));
      }
    } else {
      return false;
    }
  }
  if ((int)model.path_start.size() != arcflow_paths_supported) { return false; }

  for (int j = 0; j < h.n_variables; ++j) {
    if (h.var_types[j] != var_t::INTEGER) { return false; }
    if (!std::isfinite((double)h.obj[j])) { return false; }
    if (std::abs((double)h.var_lb[j]) > h.tol.abs) { return false; }
    const double ub = h.var_ub[j];
    if (!std::isfinite(ub) || ub < 1.0 - h.tol.abs) { return false; }

    const int begin = h.csc_offsets[j];
    const int end   = h.csc_offsets[j + 1];
    if (end == begin || end - begin > arcflow_max_col_entries) { return false; }

    int tail  = -1;
    int head  = -1;
    int label = -1;
    for (int k = begin; k < end; ++k) {
      const int r       = h.csc_rows[k];
      const double unit = h.csc_values[k] / rows[r].scale;
      if (rows[r].role == row_role_t::flow) {
        const double oriented = unit * sign;
        if (close_to(oriented, 1.0, 1.0, h.tol)) {
          if (tail >= 0) { return false; }
          tail = node_of_row[r];
        } else if (close_to(oriented, -1.0, 1.0, h.tol)) {
          if (head >= 0) { return false; }
          head = node_of_row[r];
        } else {
          return false;
        }
      } else {
        // A covering incidence is positive irrespective of the flow orientation.
        if (!close_to(unit, 1.0, 1.0, h.tol)) { return false; }
        if (label >= 0) { return false; }
        label = label_of_row[r];
      }
    }

    if (tail >= 0 && head >= 0) {
      if (label < 0) { return false; }
      // With this bound the arc capacity can never bind: a use of the arc consumes one of the
      // label's tokens, and there are exactly demand[label] of them.
      if ((double)model.demand[label] > ub + h.tol.abs) { return false; }
      model.arcs.push_back(arc_t{tail, head, label, j, (double)h.obj[j]});
    } else if (tail >= 0 && label < 0) {
      // Explicit loss arc: it leaves the node and never arrives, so it ends a path.  A node that
      // already absorbs paths through row slack would need the two capacities apportioned, which
      // no model in this family does, so reject rather than guess.
      if (model.terminator_col[tail] >= 0 || model.terminator_capacity[tail] > 0) { return false; }
      model.terminator_col[tail]      = j;
      model.terminator_cost[tail]     = h.obj[j];
      model.terminator_capacity[tail] =
        std::min((double)arcflow_paths_supported, std::floor(ub + h.tol.abs));
    } else {
      return false;
    }
  }

  if (model.arcs.empty()) { return false; }

  std::stable_sort(model.arcs.begin(), model.arcs.end(), [](const arc_t& a, const arc_t& b) {
    if (a.label != b.label) { return a.label < b.label; }
    if (a.from != b.from) { return a.from < b.from; }
    return a.cost < b.cost;
  });
  model.arc_offset.assign(model.n_labels + 1, 0);
  for (const auto& arc : model.arcs) {
    model.arc_offset[arc.label + 1]++;
  }
  for (int l = 0; l < model.n_labels; ++l) {
    model.arc_offset[l + 1] += model.arc_offset[l];
    if (model.arc_offset[l] == model.arc_offset[l + 1]) { return false; }
  }
  cuopt_assert(model.arc_offset.back() == (int)model.arcs.size(),
               "arc CSR offsets must cover every arc");
  return true;
}

// The potential is recovered from the objective rather than from any index: within a label the
// cost is affine in the potential of the arc's tail, so one label with enough distinct costs fixes
// the potential on every node it touches, and the remaining labels are fitted and extended from
// there.  The result is an affine image of the true potential, which leaves the order that
// consumes it unchanged, since a common positive factor cancels out of every comparison.
bool derive_potential(arc_flow_model_t& model, const std::atomic<bool>& preemption_flag)
{
  const int n_labels      = model.n_labels;
  const arcflow_tol_t tol = model.tol;

  int reference     = -1;
  size_t best_count = 0;
  std::vector<double> costs;
  for (int l = 0; l < n_labels; ++l) {
    costs.clear();
    for (int k = model.arc_offset[l]; k < model.arc_offset[l + 1]; ++k) {
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

  const double unknown = std::numeric_limits<double>::quiet_NaN();
  model.phi.assign(model.n_nodes, unknown);
  model.slope.assign(n_labels, unknown);
  model.displacement.assign(n_labels, unknown);
  std::vector<double> intercept(n_labels, unknown);

  for (int k = model.arc_offset[reference]; k < model.arc_offset[reference + 1]; ++k) {
    model.phi[model.arcs[k].from] = model.arcs[k].cost;
  }

  // Each productive round resolves at least one potential, slope, or displacement.
  const long max_rounds = 2L * n_labels + model.n_nodes + 2L;
  long rounds           = 0;
  for (; rounds < max_rounds; ++rounds) {
    if (preemption_flag.load()) { return false; }
    bool progress = false;
    for (int l = 0; l < n_labels; ++l) {
      const int begin = model.arc_offset[l];
      const int end   = model.arc_offset[l + 1];

      if (!is_known(model.slope[l])) {
        int lowest  = -1;
        int highest = -1;
        for (int k = begin; k < end; ++k) {
          const double p = model.phi[model.arcs[k].from];
          if (!is_known(p)) { continue; }
          if (lowest < 0 || p < model.phi[model.arcs[lowest].from]) { lowest = k; }
          if (highest < 0 || p > model.phi[model.arcs[highest].from]) { highest = k; }
        }
        if (lowest >= 0 && highest >= 0) {
          const double lo_phi = model.phi[model.arcs[lowest].from];
          const double hi_phi = model.phi[model.arcs[highest].from];
          if (!close_to(lo_phi, hi_phi, std::abs(hi_phi), tol)) {
            model.slope[l] =
              (model.arcs[highest].cost - model.arcs[lowest].cost) / (hi_phi - lo_phi);
            intercept[l] = model.arcs[lowest].cost - model.slope[l] * lo_phi;
            progress     = true;
          }
        }
      }
      if (is_known(model.slope[l]) && std::abs(model.slope[l]) > tol.abs) {
        for (int k = begin; k < end; ++k) {
          const int from = model.arcs[k].from;
          if (is_known(model.phi[from])) { continue; }
          model.phi[from] = (model.arcs[k].cost - intercept[l]) / model.slope[l];
          progress        = true;
        }
      }

      if (!is_known(model.displacement[l])) {
        for (int k = begin; k < end; ++k) {
          const double from = model.phi[model.arcs[k].from];
          const double to   = model.phi[model.arcs[k].to];
          if (is_known(from) && is_known(to)) {
            model.displacement[l] = to - from;
            progress              = true;
            break;
          }
        }
      }
      if (is_known(model.displacement[l])) {
        for (int k = begin; k < end; ++k) {
          const int from = model.arcs[k].from;
          const int to   = model.arcs[k].to;
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

  for (double p : model.phi) {
    if (!is_known(p)) { return false; }
  }
  for (double p : model.displacement) {
    if (!is_known(p)) { return false; }
  }

  double phi_scale = 0.0;
  for (double p : model.phi) {
    phi_scale = std::max(phi_scale, std::abs(p));
  }
  if (phi_scale <= tol.abs) { return false; }

  // Orient the potential so displacements are positive.  A consistent potential whose
  // displacements are all strictly positive is exactly what makes the arc graph acyclic: a cycle
  // would need its displacements to sum to zero.
  double displacement_sum = 0.0;
  for (double p : model.displacement) {
    displacement_sum += p;
  }
  if (displacement_sum < 0.0) {
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
  for (double p : model.displacement) {
    if (p <= tol.abs) { return false; }
  }

  // Smith ordering assumes nonnegative job weights.
  for (double w : model.slope) {
    if (is_known(w) && w < -tol.abs) { return false; }
  }

  // Verify against every arc, not just the two points each fit was built from.  The cost residual
  // matters as much as the potential: the order is read off the fitted slopes, so a label whose
  // costs are not affine in the potential would be ordered on a meaningless quantity.
  double cost_scale = 0.0;
  for (const auto& arc : model.arcs) {
    cost_scale = std::max(cost_scale, std::abs(arc.cost));
  }
  for (const auto& arc : model.arcs) {
    if (!close_to(
          model.phi[arc.to], model.phi[arc.from] + model.displacement[arc.label], phi_scale, tol)) {
      return false;
    }
    if (!is_known(model.slope[arc.label])) { continue; }
    const double predicted = model.slope[arc.label] * model.phi[arc.from] + intercept[arc.label];
    if (!close_to(predicted, arc.cost, cost_scale, tol)) { return false; }
  }
  return true;
}

// Weighted shortest processing time orders labels by decreasing slope over displacement.
std::vector<int> token_order(const arc_flow_model_t& model, bool& all_slopes_identified)
{
  std::vector<double> lowest_phi(model.n_labels, 0.0);
  for (int l = 0; l < model.n_labels; ++l) {
    double lowest = std::numeric_limits<double>::infinity();
    for (int k = model.arc_offset[l]; k < model.arc_offset[l + 1]; ++k) {
      lowest = std::min(lowest, model.phi[model.arcs[k].from]);
    }
    lowest_phi[l] = lowest;
  }

  std::vector<int> ordered;
  std::vector<int> unidentified;
  ordered.reserve(model.n_labels);
  for (int l = 0; l < model.n_labels; ++l) {
    if (is_known(model.slope[l])) {
      ordered.push_back(l);
    } else {
      unidentified.push_back(l);
    }
  }

  const arcflow_tol_t tol = model.tol;
  const auto cross        = [&](int a, int b) {
    return (_Float128)model.slope[a] * (_Float128)model.displacement[b];
  };

  // Approximate equality is not transitive. Tolerance forms ratio classes before the final sort.
  std::sort(ordered.begin(), ordered.end(), [&](int a, int b) {
    const _Float128 lhs = cross(a, b);
    const _Float128 rhs = cross(b, a);
    if (lhs != rhs) { return lhs > rhs; }
    return a < b;
  });
  std::vector<int> ratio_class(model.n_labels, 0);
  for (size_t position = 1; position < ordered.size(); ++position) {
    const int previous             = ordered[position - 1];
    const int current              = ordered[position];
    const _Float128 lhs            = cross(previous, current);
    const _Float128 rhs            = cross(current, previous);
    const _Float128 lhs_magnitude  = lhs < 0 ? -lhs : lhs;
    const _Float128 rhs_magnitude  = rhs < 0 ? -rhs : rhs;
    const _Float128 scale          = std::max(lhs_magnitude, rhs_magnitude);
    const _Float128 difference     = lhs > rhs ? lhs - rhs : rhs - lhs;
    const bool tied                = difference <= (_Float128)tol.abs + (_Float128)tol.rel * scale;
    ratio_class[current]           = ratio_class[previous] + (tied ? 0 : 1);
  }
  std::sort(ordered.begin(), ordered.end(), [&](int a, int b) {
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
  std::vector<int> tokens;
  tokens.reserve((size_t)total_demand);
  for (int l : ordered) {
    tokens.insert(tokens.end(), (size_t)model.demand[l], l);
  }

  // Labels without fitted slopes are placed at their first reachable potential.
  all_slopes_identified = unidentified.empty();
  std::sort(unidentified.begin(), unidentified.end(), [&](int a, int b) {
    if (lowest_phi[a] != lowest_phi[b]) { return lowest_phi[a] < lowest_phi[b]; }
    return a < b;
  });
  for (int l : unidentified) {
    double consumed = 0.0;
    size_t slot     = 0;
    while (slot < tokens.size() && consumed < lowest_phi[l]) {
      consumed += model.displacement[tokens[slot]];
      ++slot;
    }
    tokens.insert(tokens.begin() + (ptrdiff_t)slot, (size_t)model.demand[l], l);
  }
  return tokens;
}

bool state_less(const frontier_t& a, const frontier_t& b)
{
  return std::lexicographical_compare(a.node.begin(), a.node.end(), b.node.begin(), b.node.end());
}

bool state_equal(const frontier_t& a, const frontier_t& b) { return a.node == b.node; }

std::optional<arc_flow_result_t> run_dp(const arc_flow_model_t& model,
                                        const std::vector<int>& tokens,
                                        const std::atomic<bool>& preemption_flag)
{
  const int n_tokens = tokens.size();
  if (n_tokens == 0) { return std::nullopt; }

  arc_flow_result_t result;

  // The reconstruction budget is charged against retained states at each level.
  size_t retained_bytes = 0;

  std::vector<std::vector<parent_t>> history;
  history.reserve((size_t)n_tokens);

  frontier_t root;
  for (int k = 0; k < arcflow_paths_supported; ++k) {
    root.node[k] = model.path_start[k];
  }
  std::sort(root.node.begin(), root.node.end());
  std::vector<frontier_t> current{root};

  std::vector<candidate_t> candidates;
  std::vector<frontier_t> next;
  std::vector<parent_t> parents;
  for (int t = 0; t < n_tokens; ++t) {
    if (preemption_flag.load()) { return std::nullopt; }
    const int label      = tokens[t];
    const auto arc_begin = model.arcs.begin() + model.arc_offset[label];
    const auto arc_end   = model.arcs.begin() + model.arc_offset[label + 1];

    const size_t candidate_limit = arcflow_candidate_bytes_max / sizeof(candidate_t);
    size_t candidate_count       = 0;
    for (const auto& entry : current) {
      for (int k = 0; k < arcflow_paths_supported; ++k) {
        const int node = entry.node[k];
        const auto begin =
          std::lower_bound(arc_begin, arc_end, node, [](const arc_t& a, int v) {
            return a.from < v;
          });
        const auto end = std::upper_bound(begin, arc_end, node, [](int v, const arc_t& a) {
          return v < a.from;
        });
        const size_t added = end - begin;
        if (added > candidate_limit - candidate_count) { return std::nullopt; }
        candidate_count += added;
      }
    }

    candidates.clear();
    candidates.reserve(candidate_count);
    for (int i = 0; i < (int)current.size(); ++i) {
      const frontier_t& entry = current[i];
      for (int k = 0; k < arcflow_paths_supported; ++k) {
        const int node = entry.node[k];
        const auto begin = std::lower_bound(
          arc_begin, arc_end, node, [](const arc_t& a, int v) { return a.from < v; });
        const auto end = std::upper_bound(begin, arc_end, node, [](int v, const arc_t& a) {
          return v < a.from;
        });
        for (auto it = begin; it != end; ++it) {
          candidate_t candidate;
          candidate.front         = entry;
          candidate.front.node[k] = it->to;
          std::sort(candidate.front.node.begin(), candidate.front.node.end());
          candidate.front.cost = entry.cost + it->cost;
          candidate.parent     = parent_t{i, (int)(it - model.arcs.begin())};
          candidates.push_back(candidate);
        }
      }
    }
    if (candidates.empty()) { return std::nullopt; }

    // A total order makes representative selection independent of enumeration order.
    std::sort(candidates.begin(), candidates.end(), [](const candidate_t& a, const candidate_t& b) {
      if (!state_equal(a.front, b.front)) { return state_less(a.front, b.front); }
      if (a.front.cost != b.front.cost) { return a.front.cost < b.front.cost; }
      if (a.parent.prev != b.parent.prev) { return a.parent.prev < b.parent.prev; }
      return a.parent.arc < b.parent.arc;
    });
    candidates.erase(std::unique(candidates.begin(),
                                 candidates.end(),
                                 [](const candidate_t& a, const candidate_t& b) {
                                   return state_equal(a.front, b.front);
                                 }),
                     candidates.end());

    const size_t remaining =
      arcflow_history_bytes_max > retained_bytes ? arcflow_history_bytes_max - retained_bytes : 0;
    const size_t affordable = std::max<size_t>(remaining / sizeof(parent_t), 1);
    if (candidates.size() > affordable) {
      std::stable_sort(
        candidates.begin(), candidates.end(), [](const candidate_t& a, const candidate_t& b) {
          if (a.front.cost != b.front.cost) { return a.front.cost < b.front.cost; }
          return state_less(a.front, b.front);
        });
      candidates.resize(affordable);
      std::sort(
        candidates.begin(), candidates.end(), [](const candidate_t& a, const candidate_t& b) {
          return state_less(a.front, b.front);
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
    retained_bytes += parents.size() * sizeof(parent_t);
    history.push_back(std::move(parents));
    current.swap(next);
  }

  int best_index    = -1;
  double best_total = std::numeric_limits<double>::infinity();
  for (int i = 0; i < (int)current.size(); ++i) {
    const frontier_t& entry = current[i];
    double total            = entry.cost;
    bool closable           = true;
    for (int k = 0; k < arcflow_paths_supported && closable; ++k) {
      const int node  = entry.node[k];
      int64_t sharing = 0;
      for (int q = 0; q < arcflow_paths_supported; ++q) {
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

  for (int k = 0; k < arcflow_paths_supported; ++k) {
    const int col = model.terminator_col[current[best_index].node[k]];
    if (col >= 0) { result.columns.push_back(col); }
  }
  int index = best_index;
  for (int t = n_tokens; t > 0; --t) {
    const parent_t& step = history[t - 1][index];
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
  host_state_t(host_problem_t<i_t, f_t>&& problem, arcflow_profile_t&& profile)
    : h(std::move(problem)), profile(std::move(profile))
  {
  }

  host_problem_t<i_t, f_t> h;
  arcflow_profile_t profile;
};

template <typename i_t, typename f_t>
arc_flow_t<i_t, f_t>::arc_flow_t() = default;

template <typename i_t, typename f_t>
arc_flow_t<i_t, f_t>::~arc_flow_t() = default;

template <typename i_t, typename f_t>
bool arc_flow_t<i_t, f_t>::recognize(const optimization_problem_t<i_t, f_t>& op_problem,
                                     const typename mip_solver_settings_t<i_t, f_t>::tolerances_t&)
{
  const i_t n_variables   = op_problem.get_n_variables();
  const i_t n_constraints = op_problem.get_n_constraints();
  if (!arcflow_accepts_shape(n_variables, n_constraints, op_problem.get_nnz())) { return false; }
  if (op_problem.get_n_integers() != n_variables) { return false; }

  auto stream              = op_problem.get_handle_ptr()->get_stream();
  const auto& d_row_lb     = op_problem.get_constraint_lower_bounds();
  const auto& d_row_ub     = op_problem.get_constraint_upper_bounds();
  const auto& d_values     = op_problem.get_constraint_matrix_values();
  const auto& d_indices    = op_problem.get_constraint_matrix_indices();
  const auto& d_offsets    = op_problem.get_constraint_matrix_offsets();
  const auto& d_obj        = op_problem.get_objective_coefficients();
  const auto& d_var_lb     = op_problem.get_variable_lower_bounds();
  const auto& d_var_ub     = op_problem.get_variable_upper_bounds();
  const auto& d_var_types  = op_problem.get_variable_types();
  if ((i_t)d_row_lb.size() != n_constraints || (i_t)d_row_ub.size() != n_constraints) {
    return false;
  }
  if ((i_t)d_offsets.size() != n_constraints + 1) { return false; }
  if (d_values.size() != d_indices.size()) { return false; }
  cuopt_assert((i_t)d_obj.size() == n_variables, "Size mismatch");
  cuopt_assert((i_t)d_var_types.size() == n_variables, "Size mismatch");
  if (!d_var_lb.is_empty() && (i_t)d_var_lb.size() != n_variables) { return false; }
  if ((i_t)d_var_ub.size() != n_variables) { return false; }

  host_problem_t<i_t, f_t> h;
  h.n_variables   = n_variables;
  h.n_constraints = n_constraints;
  h.tol           = structural_tolerance<f_t>();
  h.row_lb        = cuopt::host_copy(d_row_lb, stream);
  h.row_ub        = cuopt::host_copy(d_row_ub, stream);
  if (!arcflow_accepts_bounds(h.row_lb, h.row_ub)) { return false; }

  const auto values  = cuopt::host_copy(d_values, stream);
  const auto indices = cuopt::host_copy(d_indices, stream);
  const auto offsets = cuopt::host_copy(d_offsets, stream);
  auto profile = profile_from_csr(n_variables, n_constraints, values, indices, offsets);
  if (!arcflow_accepts_profile(profile, h.row_lb, h.row_ub, h.tol)) { return false; }

  h.obj = cuopt::host_copy(d_obj, stream);
  if (op_problem.get_sense()) {
    for (auto& coefficient : h.obj) {
      coefficient = -coefficient;
    }
  }
  h.var_lb.assign(n_variables, f_t{0});
  if (!d_var_lb.is_empty()) { h.var_lb = cuopt::host_copy(d_var_lb, stream); }
  h.var_ub    = cuopt::host_copy(d_var_ub, stream);
  h.var_types = cuopt::host_copy(d_var_types, stream);
  transpose_into_csc<i_t, f_t>(values, indices, offsets, h);
  state_ = std::make_unique<host_state_t>(std::move(h), std::move(profile));
  return true;
}

template <typename i_t, typename f_t>
bool arc_flow_t<i_t, f_t>::recognize(const problem_t<i_t, f_t>& problem,
                                     const typename mip_solver_settings_t<i_t, f_t>::tolerances_t&)
{
  const i_t n_variables   = problem.n_variables;
  const i_t n_constraints = problem.n_constraints;
  if (!arcflow_accepts_shape(n_variables, n_constraints, problem.nnz)) { return false; }
  if (problem.n_integer_vars != n_variables) { return false; }

  auto stream = problem.handle_ptr->get_stream();
  cuopt_assert((i_t)problem.constraint_lower_bounds.size() == n_constraints, "Size mismatch");
  cuopt_assert((i_t)problem.constraint_upper_bounds.size() == n_constraints, "Size mismatch");
  cuopt_assert((i_t)problem.reverse_offsets.size() == n_variables + 1, "Size mismatch");
  cuopt_assert(problem.reverse_coefficients.size() == problem.reverse_constraints.size(),
               "Size mismatch");
  cuopt_assert((i_t)problem.objective_coefficients.size() == n_variables, "Size mismatch");
  cuopt_assert((i_t)problem.variable_types.size() == n_variables, "Size mismatch");

  host_problem_t<i_t, f_t> h;
  h.n_variables   = n_variables;
  h.n_constraints = n_constraints;
  h.tol           = structural_tolerance<f_t>();
  h.csc_values    = cuopt::host_copy(problem.reverse_coefficients, stream);
  h.csc_rows      = cuopt::host_copy(problem.reverse_constraints, stream);
  h.csc_offsets   = cuopt::host_copy(problem.reverse_offsets, stream);
  h.row_lb        = cuopt::host_copy(problem.constraint_lower_bounds, stream);
  h.row_ub        = cuopt::host_copy(problem.constraint_upper_bounds, stream);
  if (!arcflow_accepts_bounds(h.row_lb, h.row_ub)) { return false; }

  auto profile =
    profile_from_csc(n_variables, n_constraints, h.csc_values, h.csc_rows, h.csc_offsets);
  if (!arcflow_accepts_profile(profile, h.row_lb, h.row_ub, h.tol)) { return false; }

  h.obj       = cuopt::host_copy(problem.objective_coefficients, stream);
  h.var_types = cuopt::host_copy(problem.variable_types, stream);
  std::tie(h.var_lb, h.var_ub) =
    cuopt::extract_host_bounds<f_t>(problem.variable_bounds, problem.handle_ptr);
  state_ = std::make_unique<host_state_t>(std::move(h), std::move(profile));
  return true;
}

template <typename i_t, typename f_t>
bool arc_flow_t<i_t, f_t>::solve(
  const typename mip_solver_settings_t<i_t, f_t>::tolerances_t&,
  std::atomic<bool>& preemption,
  std::vector<f_t>& assignment)
{
  cuopt_assert(state_ != nullptr, "solve called without a successful recognize");
  const auto& h = state_->h;

  std::vector<row_info_t> rows;
  if (!classify_rows(h, state_->profile, rows)) {
    CUOPT_LOG_DEBUG("[ArcFlow] rejected: rows are not unit incidence after normalization");
    return false;
  }

  arc_flow_model_t model;
  if (!build_structure(h, rows, model)) {
    CUOPT_LOG_DEBUG("[ArcFlow] rejected: columns do not match the labelled arc pattern");
    return false;
  }
  if (preemption.load()) { return false; }

  if (!derive_potential(model, preemption)) {
    CUOPT_LOG_DEBUG("[ArcFlow] rejected: no consistent potential and affine cost model");
    return false;
  }
  if (preemption.load()) { return false; }

  bool all_slopes_identified = true;
  const auto tokens          = token_order(model, all_slopes_identified);
  CUOPT_LOG_DEBUG("[ArcFlow] detected %d nodes, %d labels, %d paths, %zu tokens, ordering %s",
                  model.n_nodes,
                  model.n_labels,
                  arcflow_paths_supported,
                  tokens.size(),
                  all_slopes_identified ? "identified" : "partly by reachability");

  const auto result = run_dp(model, tokens, preemption);
  if (!result.has_value()) {
    CUOPT_LOG_DEBUG("[ArcFlow] no complete path set found in the ordered family");
    return false;
  }
  CUOPT_LOG_DEBUG(
    "[ArcFlow] search %s", result->exact ? "exact" : "beamed by the history budget");

  assignment.assign((size_t)h.n_variables, f_t{0});
  for (int col : result->columns) {
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
