/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#include <utilities/inline_lp_test_utils.hpp>

#include <cuopt/mathematical_optimization/optimization_problem.hpp>
#include <cuopt/mathematical_optimization/solve.hpp>
#include <cuopt/mathematical_optimization/utilities/internals.hpp>
#include <mip_heuristics/presolve/gf2_presolve.hpp>
#include <mip_heuristics/presolve/third_party_presolve.hpp>

#include <raft/core/handle.hpp>

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <memory>
#include <random>
#include <string>
#include <string_view>
#include <unordered_set>
#include <vector>

namespace cuopt::mathematical_optimization::test {

using mip::gf2_solve;
using mip::gf2_status_t;

namespace {

// A GF(2) system in a form that stays readable as test data: one '0'/'1' string per row.
struct gf2_system_t {
  std::vector<std::string> rows;
  std::vector<int> rhs;

  int n_cols() const { return rows.empty() ? 0 : (int)rows[0].size(); }
  int n_rows() const { return (int)rows.size(); }
};

// Packed without going through gf2_set_bit, so a layout mistake in the solver shows up as a wrong
// result instead of cancelling out against a shared helper.
std::vector<std::vector<uint64_t>> pack(const gf2_system_t& system)
{
  const int n      = system.n_cols();
  const int nwords = (n + 63) / 64;
  std::vector<std::vector<uint64_t>> A(system.n_rows(), std::vector<uint64_t>(nwords, 0));
  for (int r = 0; r < system.n_rows(); ++r) {
    for (int c = 0; c < n; ++c) {
      if (system.rows[r][c] == '1') { A[r][c >> 6] |= uint64_t{1} << (c & 63); }
    }
  }
  return A;
}

std::string describe(const gf2_system_t& system)
{
  std::string out = "system:";
  for (int r = 0; r < system.n_rows(); ++r) {
    out += "\n  [" + system.rows[r] + "] = " + std::to_string(system.rhs[r]);
  }
  return out;
}

bool satisfies(const gf2_system_t& system, const std::vector<int>& x)
{
  for (int r = 0; r < system.n_rows(); ++r) {
    int parity = 0;
    for (int c = 0; c < system.n_cols(); ++c) {
      if (system.rows[r][c] == '1') { parity ^= x[c] & 1; }
    }
    if (parity != (system.rhs[r] & 1)) { return false; }
  }
  return true;
}

// Every x in {0,1}^n satisfying the system, evaluated on the unpacked form.
std::vector<std::vector<int>> all_solutions(const gf2_system_t& system)
{
  const int n = system.n_cols();
  std::vector<std::vector<int>> solutions;
  for (uint64_t assignment = 0; assignment < (uint64_t{1} << n); ++assignment) {
    std::vector<int> x(n);
    for (int c = 0; c < n; ++c) {
      x[c] = (assignment >> c) & 1;
    }
    if (satisfies(system, x)) { solutions.push_back(std::move(x)); }
  }
  return solutions;
}

// Checks gf2_solve against exhaustive enumeration and returns the number of solutions.
//
// Both directions of the determinedness contract matter: reporting a varying column as determined
// would fix a variable the presolver has no right to fix, and reporting a constant column as free
// silently loses a reduction.
size_t check_against_enumeration(const gf2_system_t& system)
{
  const int n = system.n_cols();
  auto A      = pack(system);
  auto b      = system.rhs;
  std::vector<int> x;
  std::vector<uint8_t> determined;
  const gf2_status_t status = gf2_solve(A, n, b, x, determined);

  const auto solutions = all_solutions(system);
  if (solutions.empty()) {
    EXPECT_EQ(status, gf2_status_t::Infeasible) << describe(system);
    return 0;
  }

  EXPECT_EQ(status, gf2_status_t::Feasible) << describe(system);
  EXPECT_EQ((int)x.size(), n) << describe(system);
  EXPECT_EQ((int)determined.size(), n) << describe(system);
  if (status != gf2_status_t::Feasible || (int)x.size() != n || (int)determined.size() != n) {
    return solutions.size();
  }

  EXPECT_TRUE(satisfies(system, x)) << describe(system) << "\nreported x is not a solution";

  for (int c = 0; c < n; ++c) {
    const bool constant =
      std::all_of(solutions.begin(), solutions.end(), [&](const std::vector<int>& solution) {
        return solution[c] == solutions[0][c];
      });
    EXPECT_EQ(determined[c] != 0, constant) << describe(system) << "\ncolumn " << c;
    if (constant) { EXPECT_EQ(x[c], solutions[0][c]) << describe(system) << "\ncolumn " << c; }
  }
  return solutions.size();
}

// Builds a random system. Rows are often XORs of earlier rows: a uniformly random GF(2) matrix is
// almost always full rank, which is the case rank-deficiency handling is least concerned with.
gf2_system_t random_system(std::mt19937& rng)
{
  const int m       = 1 + (int)(rng() % 6);
  const int n       = (int)(rng() % 11);
  const int density = 15 + (int)(rng() % 71);

  gf2_system_t system;
  system.rows.assign(m, std::string(n, '0'));
  system.rhs.assign(m, 0);

  for (int r = 0; r < m; ++r) {
    if (r > 0 && (rng() % 2) == 0) {
      system.rows[r] = system.rows[rng() % r];
      if ((rng() % 2) == 0) {
        const std::string& other = system.rows[rng() % r];
        for (int c = 0; c < n; ++c) {
          system.rows[r][c] = (char)('0' + ((system.rows[r][c] - '0') ^ (other[c] - '0')));
        }
      }
    } else {
      for (int c = 0; c < n; ++c) {
        system.rows[r][c] = ((int)(rng() % 100) < density) ? '1' : '0';
      }
    }
  }

  // Half the systems get a planted solution so the feasible path stays well covered; the rest get
  // a random rhs, which is usually inconsistent once the rows are dependent.
  if ((rng() % 2) == 0) {
    std::vector<int> planted(n);
    for (int c = 0; c < n; ++c) {
      planted[c] = (int)(rng() % 2);
    }
    for (int r = 0; r < m; ++r) {
      int parity = 0;
      for (int c = 0; c < n; ++c) {
        if (system.rows[r][c] == '1') { parity ^= planted[c]; }
      }
      system.rhs[r] = parity;
    }
  } else {
    for (int r = 0; r < m; ++r) {
      system.rhs[r] = (int)(rng() % 2);
    }
  }
  return system;
}

struct gf2_golden_case_t {
  const char* name;
  gf2_system_t system;
  gf2_status_t status;
  // One character per column: '0'/'1' where the column is uniquely determined, '.' where it is
  // not. x is unconstrained on '.' columns, so the test must not pin it there.
  const char* expected;
};

}  // namespace

TEST(gf2_solve, golden_cases)
{
  const std::vector<gf2_golden_case_t> cases = {
    {"identity", {{"100", "010", "001"}, {1, 0, 1}}, gf2_status_t::Feasible, "101"},
    {"full_rank_mixed", {{"110", "011", "001"}, {1, 1, 0}}, gf2_status_t::Feasible, "010"},
    // Duplicate rows: rank 1 of 2, and column 2 appears in no row.
    {"consistent_singular", {{"110", "110"}, {1, 1}}, gf2_status_t::Feasible, "..."},
    {"inconsistent_singular", {{"110", "110"}, {1, 0}}, gf2_status_t::Infeasible, ""},
    // More rows than columns, third row redundant after elimination.
    {"tall_consistent", {{"10", "11", "01"}, {1, 1, 0}}, gf2_status_t::Feasible, "10"},
    {"tall_inconsistent", {{"1", "1"}, {1, 0}}, gf2_status_t::Infeasible, ""},
    // More columns than rows.
    {"fat", {{"100", "011"}, {1, 1}}, gf2_status_t::Feasible, "1.."},
    // Rows carrying only a key variable reach gf2_solve with no columns at all.
    {"no_columns_consistent", {{""}, {0}}, gf2_status_t::Feasible, ""},
    {"no_columns_inconsistent", {{""}, {1}}, gf2_status_t::Infeasible, ""},
    // Column 2 is all zero, as happens for a binary left in the map by a rejected row.
    {"zero_column", {{"100", "010"}, {1, 1}}, gf2_status_t::Feasible, "11."},
    // The free column (1) sits below the determined pivot column (2).
    {"free_col_before_pivot", {{"110", "001"}, {1, 1}}, gf2_status_t::Feasible, "..1"},
    // J - I at even dimension is nonsingular over GF(2); the shape the enlight instances hit.
    {"j_minus_i_4",
     {{"0111", "1011", "1101", "1110"}, {1, 1, 1, 1}},
     gf2_status_t::Feasible,
     "1111"},
  };

  for (const auto& test_case : cases) {
    SCOPED_TRACE(test_case.name);

    auto A = pack(test_case.system);
    auto b = test_case.system.rhs;
    std::vector<int> x;
    std::vector<uint8_t> determined;
    const gf2_status_t status = gf2_solve(A, test_case.system.n_cols(), b, x, determined);

    EXPECT_EQ(status, test_case.status) << describe(test_case.system);

    if (test_case.status == gf2_status_t::Feasible && status == gf2_status_t::Feasible) {
      const std::string expected{test_case.expected};
      ASSERT_EQ(x.size(), expected.size()) << describe(test_case.system);
      ASSERT_EQ(determined.size(), expected.size()) << describe(test_case.system);
      for (size_t c = 0; c < expected.size(); ++c) {
        if (expected[c] == '.') {
          EXPECT_EQ(determined[c], 0) << describe(test_case.system) << "\ncolumn " << c;
        } else {
          EXPECT_NE(determined[c], 0) << describe(test_case.system) << "\ncolumn " << c;
          EXPECT_EQ(x[c], expected[c] - '0') << describe(test_case.system) << "\ncolumn " << c;
        }
      }
    }

    // Double entry: the hand-written expectations above must also agree with enumeration.
    check_against_enumeration(test_case.system);
  }
}

TEST(gf2_solve, matches_enumeration_on_random_systems)
{
  std::mt19937 rng{20260804u};
  int infeasible      = 0;
  int unique          = 0;
  int underdetermined = 0;

  for (int iteration = 0; iteration < 1500; ++iteration) {
    SCOPED_TRACE(iteration);
    const size_t n_solutions = check_against_enumeration(random_system(rng));
    if (n_solutions == 0) {
      infeasible++;
    } else if (n_solutions == 1) {
      unique++;
    } else {
      underdetermined++;
    }
  }

  // A generator drifting into one bucket would gut the test without failing it.
  EXPECT_GT(infeasible, 100);
  EXPECT_GT(unique, 50);
  EXPECT_GT(underdetermined, 200);
}

// Guards against mixing up the row and column spaces, which is the failure mode the m x n
// generalization introduces. determined is a property of the solution set, so it must permute with
// the columns. x must not be compared on undetermined columns: it holds b[pivot_row] there, which
// legitimately depends on which column won the pivot.
TEST(gf2_solve, determinedness_permutes_with_the_columns)
{
  std::mt19937 rng{20260805u};

  for (int iteration = 0; iteration < 300; ++iteration) {
    SCOPED_TRACE(iteration);
    const gf2_system_t system = random_system(rng);
    const int n               = system.n_cols();
    const int m               = system.n_rows();

    auto A = pack(system);
    auto b = system.rhs;
    std::vector<int> x;
    std::vector<uint8_t> determined;
    const gf2_status_t status = gf2_solve(A, n, b, x, determined);

    std::vector<int> row_perm(m);
    std::vector<int> col_perm(n);
    for (int r = 0; r < m; ++r) {
      row_perm[r] = r;
    }
    for (int c = 0; c < n; ++c) {
      col_perm[c] = c;
    }
    std::shuffle(row_perm.begin(), row_perm.end(), rng);
    std::shuffle(col_perm.begin(), col_perm.end(), rng);

    gf2_system_t permuted;
    permuted.rows.assign(m, std::string(n, '0'));
    permuted.rhs.assign(m, 0);
    for (int r = 0; r < m; ++r) {
      for (int c = 0; c < n; ++c) {
        permuted.rows[r][c] = system.rows[row_perm[r]][col_perm[c]];
      }
      permuted.rhs[r] = system.rhs[row_perm[r]];
    }

    auto permuted_A = pack(permuted);
    auto permuted_b = permuted.rhs;
    std::vector<int> permuted_x;
    std::vector<uint8_t> permuted_determined;
    const gf2_status_t permuted_status =
      gf2_solve(permuted_A, n, permuted_b, permuted_x, permuted_determined);

    ASSERT_EQ(status, permuted_status) << describe(system);
    if (status != gf2_status_t::Feasible) { continue; }

    EXPECT_TRUE(satisfies(permuted, permuted_x)) << describe(permuted);
    for (int c = 0; c < n; ++c) {
      EXPECT_EQ(permuted_determined[c] != 0, determined[col_perm[c]] != 0)
        << describe(system) << "\ncolumn " << c;
      if (determined[col_perm[c]]) {
        EXPECT_EQ(permuted_x[c], x[col_perm[c]]) << describe(system) << "\ncolumn " << c;
      }
    }
  }
}

namespace {

mip::third_party_presolve_device_result_t<int, double> run_gf2_presolve(std::string_view lp_text)
{
  const raft::handle_t handle{};
  auto mps_data_model = cuopt::test::parse_inline_lp(lp_text);
  auto op_problem     = mps_data_model_to_optimization_problem(&handle, mps_data_model);
  auto presolver      = std::make_unique<mip::third_party_presolve_t<int, double>>();
  presolver->set_reduction_allowlist(std::unordered_set<std::string>{"gf2presolve"});
  return presolver->apply_presolve_from_op_problem(
    op_problem, problem_category_t::MIP, presolver_t::Papilo, false, 1e-6, 1e-12, 20, 1);
}

}  // namespace

TEST(gf2_presolve, uses_compact_constraint_indices)
{
  constexpr int num_packing_vars = 128;
  constexpr int num_gf2_vars     = 6;
  constexpr int num_key_vars     = 6;
  constexpr int num_packing_rows = 128;
  constexpr int num_key_rows     = 2 * num_key_vars;
  constexpr int num_gf2_rows     = 6;
  constexpr int num_vars         = num_packing_vars + num_gf2_vars + num_key_vars;
  constexpr int num_rows         = num_packing_rows + num_key_rows + num_gf2_rows;
  constexpr int x_offset         = num_packing_vars;
  constexpr int y_offset         = x_offset + num_gf2_vars;

  std::vector<double> values;
  std::vector<int> indices;
  std::vector<int> offsets{0};
  std::vector<double> constraint_lb(num_rows, 1.0);
  std::vector<double> constraint_ub(num_rows, 2.0);

  auto add_entry = [&](int column, double value) {
    indices.push_back(column);
    values.push_back(value);
  };
  auto finish_row = [&] { offsets.push_back(static_cast<int>(values.size())); };

  // A normal binary MIP block keeps the GF2 rows at high raw row indices.
  for (int row = 0; row < num_packing_rows; ++row) {
    std::array columns{row, (row + 1) % num_packing_vars, (row + 2) % num_packing_vars};
    std::sort(columns.begin(), columns.end());
    for (int column : columns) {
      add_entry(column, 1.0);
    }
    finish_row();
  }

  // Keep every GF2 key column non-singleton without forcing it.
  for (int key = 0; key < num_key_vars; ++key) {
    add_entry(3 * key, 1.0);
    add_entry(3 * key + 1, 1.0);
    add_entry(y_offset + key, 1.0);
    finish_row();

    add_entry(3 * key + 1, 1.0);
    add_entry(3 * key + 2, 1.0);
    add_entry(y_offset + key, 1.0);
    finish_row();
  }

  // Over GF(2), this is J-I for even dimension 6, hence nonsingular. Three positive and two
  // negative coefficients per row prevent ordinary bound propagation from fixing the key.
  for (int row = 0; row < num_gf2_rows; ++row) {
    int term = 0;
    for (int col = 0; col < num_gf2_vars; ++col) {
      if (col == row) { continue; }
      add_entry(x_offset + col, term < 3 ? 1.0 : -1.0);
      ++term;
    }
    add_entry(y_offset + row, 2.0);
    finish_row();
    constraint_lb[num_packing_rows + num_key_rows + row] = 1.0;
    constraint_ub[num_packing_rows + num_key_rows + row] = 1.0;
  }

  const raft::handle_t handle_{};
  optimization_problem_t<int, double> problem(&handle_);
  std::vector<double> objective(num_vars, 1.0);
  std::vector<double> variable_lb(num_vars, 0.0);
  std::vector<double> variable_ub(num_vars, 1.0);
  std::vector<var_t> variable_types(num_vars, var_t::INTEGER);
  problem.set_csr_constraint_matrix(
    values.data(), values.size(), indices.data(), indices.size(), offsets.data(), offsets.size());
  problem.set_objective_coefficients(objective.data(), objective.size());
  problem.set_variable_lower_bounds(variable_lb.data(), variable_lb.size());
  problem.set_variable_upper_bounds(variable_ub.data(), variable_ub.size());
  problem.set_variable_types(variable_types.data(), variable_types.size());
  problem.set_constraint_lower_bounds(constraint_lb.data(), constraint_lb.size());
  problem.set_constraint_upper_bounds(constraint_ub.data(), constraint_ub.size());

  auto presolver = std::make_unique<mip::third_party_presolve_t<int, double>>();
  presolver->set_reduction_allowlist(std::unordered_set<std::string>{"gf2presolve"});
  auto result = presolver->apply_presolve_from_op_problem(
    problem, problem_category_t::MIP, presolver_t::Papilo, false, 1e-6, 1e-12, 20, 1);

  EXPECT_EQ(result.status, mip::third_party_presolve_status_t::REDUCED);
}

// Consistent singular: both rows x0 xor x1 = 1. Check we do not return infeasible.
TEST(gf2_presolve, consistent_singular_unchanged)
{
  auto result = run_gf2_presolve(R"LP(
Minimize
  obj: x0 + x1 + k0 + k1
Subject To
  c0: x0 + x1 + 2 k0 = 1
  c1: x0 + x1 + 2 k1 = 1
Binaries
  x0
  x1
  k0
  k1
End
)LP");
  EXPECT_EQ(result.status, mip::third_party_presolve_status_t::UNCHANGED);
}

// Inconsistent singular: x0 xor x1 = 1 and x0 xor x1 = 0.
TEST(gf2_presolve, inconsistent_singular_infeasible)
{
  auto result = run_gf2_presolve(R"LP(
Minimize
  obj: x0 + x1 + k0 + k1
Subject To
  c0: x0 + x1 + 2 k0 = 1
  c1: x0 + x1 + 2 k1 = 0
Binaries
  x0
  x1
  k0
  k1
End
)LP");
  EXPECT_EQ(result.status, mip::third_party_presolve_status_t::INFEASIBLE);
}

// Partially determined: x1 is unique over GF(2); x0, x2 free with x0 xor x2 = 1.
TEST(gf2_presolve, partial_determination_reduces)
{
  auto result = run_gf2_presolve(R"LP(
Minimize
  obj: x0 + x1 + x2 + k0 + k1 + k2
Subject To
  c0: x0 - x2 + 2 k0 = 1
  c1: x1 + 2 k1 = 1
  c2: x0 + x1 - x2 + 2 k2 = 0
Binaries
  x0
  x1
  x2
  k0
  k1
  k2
End
)LP");
  EXPECT_EQ(result.status, mip::third_party_presolve_status_t::REDUCED);
}

// Fat (n > m): x0 fixed; x1 xor x2 free.
TEST(gf2_presolve, more_bins_than_rows_reduces)
{
  auto result = run_gf2_presolve(R"LP(
Minimize
  obj: x0 + x1 + x2 + k0 + k1
Subject To
  c0: x0 + 2 k0 = 1
  c1: x1 + x2 + 2 k1 = 1
Binaries
  x0
  x1
  x2
  k0
  k1
End
)LP");
  EXPECT_EQ(result.status, mip::third_party_presolve_status_t::REDUCED);
}

// Tall consistent (m > n): x0 = 1, x1 = 0 uniquely, third row redundant.
TEST(gf2_presolve, more_rows_than_bins_reduces)
{
  auto result = run_gf2_presolve(R"LP(
Minimize
  obj: x0 + x1 + k0 + k1 + k2
Subject To
  c0: x0 + 2 k0 = 1
  c1: x0 + x1 + 2 k1 = 1
  c2: x1 + 2 k2 = 0
Binaries
  x0
  x1
  k0
  k1
  k2
End
)LP");
  EXPECT_EQ(result.status, mip::third_party_presolve_status_t::OPTIMAL);
}

// Tall inconsistent (m > n): x0 = 1 and x0 = 0.
TEST(gf2_presolve, more_rows_than_bins_infeasible)
{
  auto result = run_gf2_presolve(R"LP(
Minimize
  obj: x0 + k0 + k1
Subject To
  c0: x0 + 2 k0 = 1
  c1: x0 + 2 k1 = 0
Binaries
  x0
  k0
  k1
End
)LP");
  EXPECT_EQ(result.status, mip::third_party_presolve_status_t::INFEASIBLE);
}

// Dual-role: k is key in c0 and a ±1 bin in c1. GF(2) forces k=1; ℤ key recovery wants k=0.
TEST(gf2_presolve, dual_role_key_bin_conflict_infeasible)
{
  auto result = run_gf2_presolve(R"LP(
Minimize
  obj: x0 + k + y
Subject To
  c0: x0 + 2 k = 1
  c1: k + 2 y = 1
Binaries
  x0
  k
  y
End
)LP");
  EXPECT_EQ(result.status, mip::third_party_presolve_status_t::INFEASIBLE);
}

// GF(2)-consistent with x0=x1=1, but key recovery gives k=-1 outside [0,1].
TEST(gf2_presolve, key_out_of_bounds_infeasible)
{
  auto result = run_gf2_presolve(R"LP(
Minimize
  obj: x0 + x1 + a + b + k
Subject To
  c0: x0 + 2 a = 1
  c1: x1 + 2 b = 1
  c2: x0 + x1 + 2 k = 0
Binaries
  x0
  x1
  a
  b
  k
End
)LP");
  EXPECT_EQ(result.status, mip::third_party_presolve_status_t::INFEASIBLE);
}

}  // namespace cuopt::mathematical_optimization::test
