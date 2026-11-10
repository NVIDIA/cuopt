/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#include <presolve/experimental_rcm_mps.hpp>

#include <cstring>
#include <iostream>

// Host-only model double with the exact mps_data_model getter/setter signatures
// used by the integration adapter. Native library linkage is validated separately.
struct model {
  std::vector<int> offsets{0, 2, 3, 5}, indices{0, 3, 1, 0, 2};
  std::vector<double> values{-0.0, 1.25, -2.5, 3.0, 4.0};
  std::vector<double> c{-0.0, 2.0, -3.0, 4.0}, vl{-1, -2, -3, -4}, vu{1, 2, 3, 4};
  std::vector<double> rl{-1, -2, -3}, ru{1, 2, 3}, rhs{1, 2, 3};
  std::vector<char> rt{'L', 'E', 'G'}, vt{'C', 'C', 'C', 'C'};
  std::vector<std::string> rn{"r0", "r1", "r2"}, vn{"x0", "x1", "x2", "x3"};
  std::vector<double> initial_x, initial_y;
  bool maximize           = true;
  double objective_offset = -42.5, objective_scaling = 3.75;
  std::string objective_name = "obj", problem_name = "native_fixture";

  int get_n_constraints() const { return static_cast<int>(rl.size()); }
  int get_n_variables() const { return static_cast<int>(c.size()); }
  const auto& get_constraint_matrix_values() const { return values; }
  const auto& get_constraint_matrix_indices() const { return indices; }
  const auto& get_constraint_matrix_offsets() const { return offsets; }
  const auto& get_objective_coefficients() const { return c; }
  const auto& get_variable_lower_bounds() const { return vl; }
  const auto& get_variable_upper_bounds() const { return vu; }
  const auto& get_constraint_bounds() const { return rhs; }
  const auto& get_row_types() const { return rt; }
  const auto& get_row_names() const { return rn; }
  const auto& get_variable_types() const { return vt; }
  const auto& get_variable_names() const { return vn; }
  const auto& get_initial_primal_solution() const { return initial_x; }
  const auto& get_initial_dual_solution() const { return initial_y; }

  void set_csr_constraint_matrix(std::span<const double> v,
                                 std::span<const int> i,
                                 std::span<const int> o)
  {
    values.assign(v.begin(), v.end());
    indices.assign(i.begin(), i.end());
    offsets.assign(o.begin(), o.end());
  }
  void set_objective_coefficients(std::span<const double> v) { c.assign(v.begin(), v.end()); }
  void set_variable_lower_bounds(std::span<const double> v) { vl.assign(v.begin(), v.end()); }
  void set_variable_upper_bounds(std::span<const double> v) { vu.assign(v.begin(), v.end()); }
  void set_constraint_lower_bounds(std::span<const double> v) { rl.assign(v.begin(), v.end()); }
  void set_constraint_upper_bounds(std::span<const double> v) { ru.assign(v.begin(), v.end()); }
  void set_constraint_bounds(std::span<const double> v) { rhs.assign(v.begin(), v.end()); }
  void set_row_types(std::span<const char> v) { rt.assign(v.begin(), v.end()); }
  void set_row_names(const std::vector<std::string>& v) { rn = v; }
  void set_variable_types(const std::vector<char>& v) { vt = v; }
  void set_variable_names(const std::vector<std::string>& v) { vn = v; }
};

void require(bool ok, const char* why)
{
  if (!ok) { throw std::runtime_error(why); }
}
bool exact(double a, double b) { return std::memcmp(&a, &b, sizeof(double)) == 0; }

int main()
{
  try {
    int tests = 0;
    const model original;
    auto modified    = original;
    const auto state = lpfeas_ordering::reorder_native_mps(modified, 1, original.rl, original.ru);
    require(state.active && state.excluded_rows == 2 && state.excluded_nnz == 4,
            "Hub threshold changed matrix coverage or stats");
    require(modified.maximize == original.maximize &&
              exact(modified.objective_offset, original.objective_offset) &&
              exact(modified.objective_scaling, original.objective_scaling) &&
              modified.objective_name == original.objective_name &&
              modified.problem_name == original.problem_name,
            "Scalar metadata changed");
    require(modified.values.size() == original.values.size(), "Dropped matrix entries");
    for (std::size_t i = 0; i < state.row_new_to_old.size(); ++i) {
      const int old_i = state.row_new_to_old[i];
      require(modified.rt[i] == original.rt[old_i] && modified.rn[i] == original.rn[old_i] &&
                exact(modified.rhs[i], original.rhs[old_i]) &&
                exact(modified.rl[i], original.rl[old_i]) &&
                exact(modified.ru[i], original.ru[old_i]),
              "Row metadata not permuted");
      for (int k = modified.offsets[i]; k < modified.offsets[i + 1]; ++k) {
        const int old_j = state.column_new_to_old[modified.indices[k]];
        bool found      = false;
        for (int old_k = original.offsets[old_i]; old_k < original.offsets[old_i + 1]; ++old_k) {
          if (original.indices[old_k] == old_j) {
            require(exact(modified.values[k], original.values[old_k]),
                    "Matrix coefficient bits changed");
            found = true;
          }
        }
        require(found, "Matrix permutation is not exact");
      }
    }
    for (std::size_t j = 0; j < state.column_new_to_old.size(); ++j) {
      const int old_j = state.column_new_to_old[j];
      require(modified.vn[j] == original.vn[old_j] && modified.vt[j] == original.vt[old_j] &&
                exact(modified.c[j], original.c[old_j]) &&
                exact(modified.vl[j], original.vl[old_j]) &&
                exact(modified.vu[j], original.vu[old_j]),
              "Column metadata not permuted");
    }
    ++tests;

    // A synthetic native solver result must reach PSLP undo in its original reduced coordinates.
    std::vector<double> x(4), y(3), z(4);
    for (std::size_t j = 0; j < x.size(); ++j) {
      const int old_j = state.column_new_to_old[j];
      x[j]            = original.c[old_j];
      z[j]            = original.vl[old_j];
    }
    for (std::size_t i = 0; i < y.size(); ++i) {
      y[i] = original.rl[state.row_new_to_old[i]];
    }
    state.restore(x, y, z);
    for (std::size_t j = 0; j < x.size(); ++j) {
      require(exact(x[j], original.c[j]) && exact(z[j], original.vl[j]),
              "Inverse x/z changed coordinates");
    }
    for (std::size_t i = 0; i < y.size(); ++i) {
      require(exact(y[i], original.rl[i]), "Inverse y changed coordinates");
    }
    ++tests;

    {
      auto absent = original;
      absent.rhs.clear();
      absent.rt.clear();
      absent.rn.clear();
      absent.vn.clear();
      absent.vt.clear();
      absent.vl.clear();
      absent.vu.clear();
      const auto s = lpfeas_ordering::reorder_native_mps(absent, 1024, original.rl, original.ru);
      require(s.active && absent.rhs.empty() && absent.rt.empty() && absent.rn.empty() &&
                absent.vn.empty(),
              "Absent optional metadata was invented");
      for (double v : absent.vl) {
        require(v == -std::numeric_limits<double>::infinity(),
                "Missing lower bound default changed");
      }
      for (double v : absent.vu) {
        require(v == std::numeric_limits<double>::infinity(),
                "Missing upper bound default changed");
      }
      ++tests;
    }
    {
      auto invalid = original;
      invalid.rn.pop_back();
      bool rejected = false;
      try {
        (void)lpfeas_ordering::reorder_native_mps(invalid, 1024, original.rl, original.ru);
      } catch (const std::runtime_error&) {
        rejected = true;
      }
      require(rejected && invalid.indices == original.indices && invalid.c == original.c,
              "Malformed metadata was accepted or mutated before validation");
      ++tests;
    }
    {
      auto warm      = original;
      warm.initial_x = {0};
      bool rejected  = false;
      try {
        (void)lpfeas_ordering::reorder_native_mps(warm, 1024, original.rl, original.ru);
      } catch (const std::runtime_error&) {
        rejected = true;
      }
      require(rejected, "Warm-start coordinates silently accepted");
      ++tests;
    }
    {
      auto wrong_x = x;
      wrong_x.push_back(7);
      bool rejected = false;
      try {
        state.restore(wrong_x, y, z);
      } catch (const std::runtime_error&) {
        rejected = true;
      }
      require(rejected, "Wrong native postsolve dimensions accepted");
      ++tests;
    }
    {
      lpfeas_ordering::native_rcm_state off;
      const auto saved_x = x, saved_y = y, saved_z = z;
      off.restore(x, y, z);
      require(x == saved_x && y == saved_y && z == saved_z, "Inactive inverse changed vectors");
      auto separate = original;
      const auto other =
        lpfeas_ordering::reorder_native_mps(separate, 1024, original.rl, original.ru);
      require(other.excluded_rows == 0 && state.excluded_rows == 2, "Per-instance state leaked");
      ++tests;
    }
    {
      auto minimize     = original;
      minimize.maximize = false;
      const auto s = lpfeas_ordering::reorder_native_mps(minimize, 1024, original.rl, original.ru);
      require(s.active && !minimize.maximize &&
                exact(minimize.objective_offset, original.objective_offset) &&
                exact(minimize.objective_scaling, original.objective_scaling),
              "Minimization scalar metadata changed");
      ++tests;
    }
    {
      auto empty = original;
      empty.rl.clear();
      empty.ru.clear();
      empty.rhs.clear();
      empty.rt.clear();
      empty.rn.clear();
      empty.c.clear();
      empty.vl.clear();
      empty.vu.clear();
      empty.vn.clear();
      empty.vt.clear();
      empty.values.clear();
      empty.indices.clear();
      empty.offsets = {0};
      const auto s  = lpfeas_ordering::reorder_native_mps(empty, 1024, empty.rl, empty.ru);
      std::vector<double> ex, ey, ez;
      s.restore(ex, ey, ez);
      require(!s.active && s.row_new_to_old.empty() && s.column_new_to_old.empty() && ex.empty(),
              "Presolve-solved empty model was reordered");
      ++tests;
    }
    std::cout << "NATIVE_RCM_INTEGRATION_CPU_TESTS_PASS count=" << tests << '\n';
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "NATIVE_RCM_INTEGRATION_CPU_TESTS_FAIL " << error.what() << '\n';
    return 1;
  }
}
