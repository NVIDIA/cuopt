/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#include <cuopt/mathematical_optimization/io/lp_writer.hpp>

#include <cuopt/mathematical_optimization/io/data_model_view.hpp>
#include <cuopt/mathematical_optimization/io/mps_data_model.hpp>
#include <mps_parser_internal.hpp>
#include <utilities/error.hpp>
#include <utilities/sparse_matrix_helpers.hpp>

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <limits>
#include <memory>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

namespace cuopt::mathematical_optimization::io {

namespace {

// The LP format uses these defaults for a variable that is not otherwise
// constrained: lower bound 0, upper bound +infinity.
template <typename f_t>
constexpr f_t default_lower()
{
  return f_t(0);
}
template <typename f_t>
f_t default_upper()
{
  return std::numeric_limits<f_t>::infinity();
}

}  // namespace

template <typename i_t, typename f_t>
lp_writer_t<i_t, f_t>::lp_writer_t(const data_model_view_t<i_t, f_t>& problem) : problem_(problem)
{
}

template <typename i_t, typename f_t>
data_model_view_t<i_t, f_t> lp_writer_t<i_t, f_t>::create_view(
  const mps_data_model_t<i_t, f_t>& model)
{
  data_model_view_t<i_t, f_t> view;

  view.set_maximize(model.get_sense());

  const auto& A_values  = model.get_constraint_matrix_values();
  const auto& A_indices = model.get_constraint_matrix_indices();
  const auto& A_offsets = model.get_constraint_matrix_offsets();
  if (!A_values.empty()) {
    view.set_csr_constraint_matrix(A_values.data(),
                                   static_cast<i_t>(A_values.size()),
                                   A_indices.data(),
                                   static_cast<i_t>(A_indices.size()),
                                   A_offsets.data(),
                                   static_cast<i_t>(A_offsets.size()));
  }

  const auto& b = model.get_constraint_bounds();
  if (!b.empty()) { view.set_constraint_bounds(b.data(), static_cast<i_t>(b.size())); }

  const auto& c = model.get_objective_coefficients();
  if (!c.empty()) { view.set_objective_coefficients(c.data(), static_cast<i_t>(c.size())); }

  view.set_objective_scaling_factor(model.get_objective_scaling_factor());
  view.set_objective_offset(model.get_objective_offset());

  const auto& lb = model.get_variable_lower_bounds();
  const auto& ub = model.get_variable_upper_bounds();
  if (!lb.empty()) { view.set_variable_lower_bounds(lb.data(), static_cast<i_t>(lb.size())); }
  if (!ub.empty()) { view.set_variable_upper_bounds(ub.data(), static_cast<i_t>(ub.size())); }

  const auto& var_types = model.get_variable_types();
  if (!var_types.empty()) {
    view.set_variable_types(var_types.data(), static_cast<i_t>(var_types.size()));
  }

  const auto& row_types = model.get_row_types();
  if (!row_types.empty()) {
    view.set_row_types(row_types.data(), static_cast<i_t>(row_types.size()));
  }

  const auto& cl = model.get_constraint_lower_bounds();
  const auto& cu = model.get_constraint_upper_bounds();
  if (!cl.empty()) { view.set_constraint_lower_bounds(cl.data(), static_cast<i_t>(cl.size())); }
  if (!cu.empty()) { view.set_constraint_upper_bounds(cu.data(), static_cast<i_t>(cu.size())); }

  view.set_problem_name(model.get_problem_name());
  view.set_objective_name(model.get_objective_name());
  view.set_variable_names(model.get_variable_names());
  view.set_row_names(model.get_row_names());

  const auto& Q_values  = model.get_quadratic_objective_values();
  const auto& Q_indices = model.get_quadratic_objective_indices();
  const auto& Q_offsets = model.get_quadratic_objective_offsets();
  if (!Q_values.empty()) {
    view.set_quadratic_objective_matrix(Q_values.data(),
                                        static_cast<i_t>(Q_values.size()),
                                        Q_indices.data(),
                                        static_cast<i_t>(Q_indices.size()),
                                        Q_offsets.data(),
                                        static_cast<i_t>(Q_offsets.size()));
  }

  if (model.has_quadratic_constraints()) {
    view.set_quadratic_constraints(
      std::vector<typename mps_data_model_t<i_t, f_t>::quadratic_constraint_t>(
        model.get_quadratic_constraints()));
  }

  return view;
}

template <typename i_t, typename f_t>
lp_writer_t<i_t, f_t>::lp_writer_t(const mps_data_model_t<i_t, f_t>& problem)
  : owned_view_(std::make_unique<data_model_view_t<i_t, f_t>>(create_view(problem))),
    problem_(*owned_view_)
{
}

template <typename i_t, typename f_t>
void lp_writer_t<i_t, f_t>::write(const std::string& lp_file_path)
{
  std::ofstream lp_file(lp_file_path);

  mps_parser_expects(lp_file.is_open(),
                     error_type_t::ValidationError,
                     "Error creating output LP file! Given path: %s",
                     lp_file_path.c_str());

  const f_t inf = std::numeric_limits<f_t>::infinity();

  // --- Gather sizes -------------------------------------------------------
  const auto c_span         = problem_.get_objective_coefficients();
  const auto lb_span        = problem_.get_variable_lower_bounds();
  const auto ub_span        = problem_.get_variable_upper_bounds();
  const auto types_span     = problem_.get_variable_types();
  const auto& var_names_ref = problem_.get_variable_names();
  const auto A_values_span  = problem_.get_constraint_matrix_values();
  const auto A_indices_span = problem_.get_constraint_matrix_indices();
  const auto A_offsets_span = problem_.get_constraint_matrix_offsets();
  const auto& row_names_ref = problem_.get_row_names();
  const auto& quadratic_constraints = problem_.get_quadratic_constraints();

  i_t n_variables = 0;
  auto grow       = [&](size_t s) { n_variables = std::max(n_variables, static_cast<i_t>(s)); };
  grow(c_span.size());
  grow(lb_span.size());
  grow(ub_span.size());
  grow(types_span.size());
  grow(var_names_ref.size());
  for (i_t idx : A_indices_span) {
    n_variables = std::max(n_variables, static_cast<i_t>(idx + 1));
  }
  {
    const auto Q_off = problem_.get_quadratic_objective_offsets();
    if (Q_off.size() > 1) { n_variables = std::max(n_variables, static_cast<i_t>(Q_off.size() - 1)); }
    for (i_t idx : problem_.get_quadratic_objective_indices()) {
      n_variables = std::max(n_variables, static_cast<i_t>(idx + 1));
    }
  }
  for (const auto& qc : quadratic_constraints) {
    for (i_t idx : qc.linear_indices)
      n_variables = std::max(n_variables, static_cast<i_t>(idx + 1));
    for (i_t idx : qc.rows)
      n_variables = std::max(n_variables, static_cast<i_t>(idx + 1));
    for (i_t idx : qc.cols)
      n_variables = std::max(n_variables, static_cast<i_t>(idx + 1));
  }

  // --- Local, padded per-variable data ------------------------------------
  std::vector<f_t> c(static_cast<size_t>(n_variables), f_t(0));
  std::vector<f_t> var_lb(static_cast<size_t>(n_variables), default_lower<f_t>());
  std::vector<f_t> var_ub(static_cast<size_t>(n_variables), default_upper<f_t>());
  std::vector<char> var_types(static_cast<size_t>(n_variables), 'C');
  for (size_t j = 0; j < std::min<size_t>(c_span.size(), c.size()); ++j)
    c[j] = c_span[j];
  for (size_t j = 0; j < std::min<size_t>(lb_span.size(), var_lb.size()); ++j)
    var_lb[j] = lb_span[j];
  for (size_t j = 0; j < std::min<size_t>(ub_span.size(), var_ub.size()); ++j)
    var_ub[j] = ub_span[j];
  for (size_t j = 0; j < std::min<size_t>(types_span.size(), var_types.size()); ++j)
    var_types[j] = types_span[j];

  auto var_name = [&](i_t j) -> std::string {
    if (static_cast<size_t>(j) < var_names_ref.size() && !var_names_ref[j].empty())
      return var_names_ref[j];
    return "C" + std::to_string(j);
  };

  // --- Linear constraint bounds -------------------------------------------
  const auto b_span   = problem_.get_constraint_bounds();
  const auto clb_span = problem_.get_constraint_lower_bounds();
  const auto cub_span = problem_.get_constraint_upper_bounds();
  const auto rtype_span = problem_.get_row_types();

  i_t n_constraints = 0;
  if (!b_span.empty())
    n_constraints = static_cast<i_t>(b_span.size());
  else if (!clb_span.empty())
    n_constraints = static_cast<i_t>(clb_span.size());
  else
    n_constraints = static_cast<i_t>(cub_span.size());

  std::vector<f_t> clb(static_cast<size_t>(n_constraints));
  std::vector<f_t> cub(static_cast<size_t>(n_constraints));
  if (clb_span.empty() || cub_span.empty()) {
    // Derive from row types + single-sided b (mirrors mps_writer's fallback).
    for (size_t i = 0; i < static_cast<size_t>(n_constraints); ++i) {
      f_t rhs = i < b_span.size() ? b_span[i] : f_t(0);
      char t  = i < rtype_span.size() ? rtype_span[i] : 'E';
      if (t == 'L') {
        clb[i] = -inf;
        cub[i] = rhs;
      } else if (t == 'G') {
        clb[i] = rhs;
        cub[i] = inf;
      } else {  // 'E'
        clb[i] = rhs;
        cub[i] = rhs;
      }
    }
  } else {
    for (size_t i = 0; i < static_cast<size_t>(n_constraints); ++i) {
      clb[i] = clb_span[i];
      cub[i] = cub_span[i];
    }
  }

  auto row_name = [&](i_t k) -> std::string {
    if (static_cast<size_t>(k) < row_names_ref.size() && !row_names_ref[k].empty())
      return row_names_ref[k];
    return "R" + std::to_string(k);
  };

  // --- Formatting helpers -------------------------------------------------
  const int precision = std::numeric_limits<f_t>::max_digits10;
  auto fmt            = [&](f_t v) -> std::string {
    if (std::isinf(v)) return v > 0 ? "inf" : "-inf";
    std::ostringstream os;
    os << std::setprecision(precision) << v;
    return os.str();
  };

  // Emits a signed algebraic term ("+ <|coeff|> <repr>") with soft line
  // wrapping. Continuation lines start with whitespace followed by the sign
  // token, never with a bare name, so they can never be mistaken for a
  // section header on re-read.
  auto emit_term = [&](f_t coeff, const std::string& repr, int& terms_on_line) {
    const bool neg = coeff < f_t(0);
    const f_t a    = neg ? -coeff : coeff;
    if (terms_on_line > 0 && (terms_on_line % 8) == 0) { lp_file << "\n    "; }
    lp_file << (neg ? " - " : " + ") << fmt(a) << " " << repr;
    ++terms_on_line;
  };

  lp_file << std::setprecision(precision);

  // --- Objective ----------------------------------------------------------
  lp_file << (problem_.get_sense() ? "Maximize\n" : "Minimize\n");
  {
    std::string obj_name =
      problem_.get_objective_name().empty() ? "obj" : problem_.get_objective_name();
    lp_file << " " << obj_name << ":";

    int terms_on_line = 0;
    for (i_t j = 0; j < n_variables; ++j) {
      if (c[j] != f_t(0)) { emit_term(c[j], var_name(j), terms_on_line); }
    }
    // A constant objective term is written directly; read_lp folds it into
    // the objective offset.
    const f_t offset = problem_.get_objective_offset();
    if (std::isfinite(offset) && offset != f_t(0)) {
      lp_file << (offset < f_t(0) ? " - " : " + ") << fmt(std::abs(offset));
    }

    // Quadratic objective: build the symmetric Hessian H = Q + Q^T (matching
    // the MPS writer), then emit its upper triangle inside a '[ ... ] / 2'
    // block. In the LP objective convention a bracket coefficient p on a term
    // contributes 0.5*p to the objective, so for H = Q + Q^T the diagonal
    // coefficient is H[i][i] and the off-diagonal coefficient is 2*H[i][j].
    if (problem_.has_quadratic_objective()) {
      auto Qv = problem_.get_quadratic_objective_values();
      auto Qi = problem_.get_quadratic_objective_indices();
      auto Qo = problem_.get_quadratic_objective_offsets();
      std::vector<f_t> Q_values(Qv.begin(), Qv.end());
      std::vector<i_t> Q_indices(Qi.begin(), Qi.end());
      std::vector<i_t> Q_offsets(Qo.begin(), Qo.end());

      std::vector<f_t> H_values;
      std::vector<i_t> H_indices;
      std::vector<i_t> H_offsets;
      if (problem_.is_Q_symmetrized()) {
        H_values  = std::move(Q_values);
        H_indices = std::move(Q_indices);
        H_offsets = std::move(Q_offsets);
      } else {
        cuopt::symmetrize_csr<i_t, f_t>(
          Q_values, Q_indices, Q_offsets, H_values, H_indices, H_offsets);
      }

      // Collect the upper-triangular entries first so we only open the bracket
      // when there is at least one nonzero quadratic term.
      const i_t n_rows = static_cast<i_t>(H_offsets.size()) > 0
                           ? static_cast<i_t>(H_offsets.size()) - 1
                           : 0;
      std::vector<std::tuple<i_t, i_t, f_t>> upper;
      for (i_t i = 0; i < n_rows; ++i) {
        for (i_t p = H_offsets[i]; p < H_offsets[i + 1]; ++p) {
          const i_t j = H_indices[p];
          const f_t v = H_values[p];
          if (i <= j && v != f_t(0)) { upper.emplace_back(i, j, v); }
        }
      }
      if (!upper.empty()) {
        lp_file << " + [";
        int quad_terms = 0;
        for (const auto& [i, j, v] : upper) {
          if (i == j) {
            emit_term(v, var_name(i) + " ^ 2", quad_terms);
          } else {
            emit_term(f_t(2) * v, var_name(i) + " * " + var_name(j), quad_terms);
          }
        }
        lp_file << " ] / 2";
      }
    }
    lp_file << "\n";
  }

  // --- Constraints --------------------------------------------------------
  lp_file << "Subject To\n";

  // Emits "<name>: <linear terms> <rel> <rhs>" for one linear row. `indices`
  // and `values` hold the row's nonzeros.
  auto write_linear_row = [&](const std::string& name,
                              const std::vector<std::pair<i_t, f_t>>& row,
                              const char* rel,
                              f_t rhs) {
    lp_file << " " << name << ":";
    int terms_on_line = 0;
    for (const auto& [vid, val] : row) {
      if (val != f_t(0)) { emit_term(val, var_name(vid), terms_on_line); }
    }
    lp_file << " " << rel << " " << fmt(rhs) << "\n";
  };

  for (i_t k = 0; k < n_constraints; ++k) {
    std::vector<std::pair<i_t, f_t>> row;
    if (static_cast<size_t>(k) + 1 < A_offsets_span.size()) {
      for (i_t p = A_offsets_span[k]; p < A_offsets_span[k + 1]; ++p) {
        row.emplace_back(A_indices_span[p], A_values_span[p]);
      }
    }

    const f_t lo = clb[k];
    const f_t hi = cub[k];
    if (lo == hi) {
      write_linear_row(row_name(k), row, "=", lo);
    } else if (std::isinf(lo) && lo < 0 && !std::isinf(hi)) {
      write_linear_row(row_name(k), row, "<=", hi);
    } else if (std::isinf(hi) && hi > 0 && !std::isinf(lo)) {
      write_linear_row(row_name(k), row, ">=", lo);
    } else if (!std::isinf(lo) && !std::isinf(hi)) {
      // Range row: the LP format cannot express two finite bounds on a single
      // line, so split it into a '>=' row and a '<=' row.
      write_linear_row(row_name(k) + "_lo", row, ">=", lo);
      write_linear_row(row_name(k) + "_up", row, "<=", hi);
    }
    // (-inf, +inf) is a non-constraining row and is intentionally omitted.
  }

  // Quadratic constraints (QCQP). The linear part is written first, then a
  // '[ ... ]' block (no '/ 2' suffix). Q is stored upper-triangular with the
  // full x^T Q x coefficient per variable pair, which is exactly what the LP
  // constraint-bracket convention expects.
  for (size_t q = 0; q < quadratic_constraints.size(); ++q) {
    typename mps_data_model_t<i_t, f_t>::quadratic_constraint_t qc = quadratic_constraints[q];
    const std::string name =
      qc.constraint_row_name.empty() ? "QC" + std::to_string(q) : qc.constraint_row_name;
    const char* rel = qc.constraint_row_type == 'G'   ? ">="
                      : qc.constraint_row_type == 'E' ? "="
                                                      : "<=";

    lp_file << " " << name << ":";
    int terms_on_line = 0;
    for (size_t t = 0; t < qc.linear_indices.size(); ++t) {
      if (qc.linear_values[t] != f_t(0)) {
        emit_term(qc.linear_values[t], var_name(qc.linear_indices[t]), terms_on_line);
      }
    }

    canonicalize_coo_matrix(qc.rows, qc.cols, qc.vals);
    lp_file << " + [";
    int quad_terms = 0;
    for (size_t p = 0; p < qc.vals.size(); ++p) {
      const i_t i = qc.rows[p];
      const i_t j = qc.cols[p];
      const f_t v = qc.vals[p];
      if (v == f_t(0)) continue;
      if (i == j) {
        emit_term(v, var_name(i) + " ^ 2", quad_terms);
      } else {
        emit_term(v, var_name(i) + " * " + var_name(j), quad_terms);
      }
    }
    lp_file << " ] " << rel << " " << fmt(qc.rhs_value) << "\n";
  }

  // --- Bounds / integrality / semi-continuous -----------------------------
  // Classify variables. Binaries are integers with [0, 1] bounds; those go in
  // the Binaries section (which implies bounds) and get no explicit bound line.
  std::vector<i_t> generals;
  std::vector<i_t> binaries;
  std::vector<i_t> semi_continuous;

  auto is_binary = [&](i_t j) {
    return var_types[j] == 'I' && var_lb[j] == f_t(0) && var_ub[j] == f_t(1);
  };

  std::vector<std::string> bound_lines;
  for (i_t j = 0; j < n_variables; ++j) {
    const char t = var_types[j];
    if (t == 'I') {
      if (is_binary(j)) {
        binaries.push_back(j);
      } else {
        generals.push_back(j);
      }
    } else if (t == 'S') {
      semi_continuous.push_back(j);
    }

    if (is_binary(j)) { continue; }  // bounds implied by the Binaries section

    const f_t lo = var_lb[j];
    const f_t hi = var_ub[j];
    const std::string name = var_name(j);
    std::ostringstream line;

    if (std::isinf(lo) && lo < 0 && std::isinf(hi) && hi > 0) {
      line << " " << name << " free";
    } else if (lo == hi) {
      line << " " << name << " = " << fmt(lo);
    } else {
      bool need_lower = (lo != default_lower<f_t>());
      const bool need_upper = !(std::isinf(hi) && hi > 0);
      // A negative upper bound needs an explicit lower bound, otherwise the
      // default lower of 0 collides with it on re-read (read_lp rejects this).
      if (need_upper && hi < f_t(0) && !need_lower) { need_lower = true; }

      if (need_lower && need_upper) {
        line << " " << fmt(lo) << " <= " << name << " <= " << fmt(hi);
      } else if (need_lower) {
        line << " " << name << " >= " << fmt(lo);
      } else if (need_upper) {
        line << " " << name << " <= " << fmt(hi);
      } else {
        continue;  // default [0, +inf): nothing to emit
      }
    }
    bound_lines.push_back(line.str());
  }

  if (!bound_lines.empty()) {
    lp_file << "Bounds\n";
    for (const auto& l : bound_lines)
      lp_file << l << "\n";
  }

  auto write_name_section = [&](const char* header, const std::vector<i_t>& ids) {
    if (ids.empty()) return;
    lp_file << header << "\n";
    for (i_t j : ids)
      lp_file << " " << var_name(j) << "\n";
  };
  write_name_section("Generals", generals);
  write_name_section("Binaries", binaries);
  write_name_section("Semi-Continuous", semi_continuous);

  lp_file << "End\n";
  lp_file.close();
}

template class lp_writer_t<int, float>;
template class lp_writer_t<int, double>;

}  // namespace cuopt::mathematical_optimization::io
