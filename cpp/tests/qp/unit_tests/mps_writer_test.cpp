/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

/**
 * Builds small unconstrained QPs as optimization_problem_t, writes MPS via write_to_mps, and checks
 * QUADOBJ coefficients against the symmetric Hessian H = Q + Q^T stored internally (MPS uses
 * (1/2) x^T H x for the quadratic part).
 */

#include <utilities/common_utils.hpp>

#include <cuopt/linear_programming/optimization_problem.hpp>
#include <cuopt/linear_programming/optimization_problem_interface.hpp>

#include <raft/core/handle.hpp>

#include <gtest/gtest.h>

#include <algorithm>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <limits>
#include <sstream>
#include <string>
#include <tuple>
#include <vector>

namespace {

struct temp_mps_file_guard_t {
  explicit temp_mps_file_guard_t(std::string p) : path(std::move(p)) {}
  std::string path;
  ~temp_mps_file_guard_t()
  {
    if (!path.empty()) {
      std::error_code ec;
      std::filesystem::remove(path, ec);
    }
  }
};

std::string read_entire_file(std::string const& file_path)
{
  std::ifstream in(file_path);
  if (!in) { return {}; }
  std::ostringstream ss;
  ss << in.rdbuf();
  return ss.str();
}

/** Text between the line after "QUADOBJ" and "ENDATA". */
std::string extract_quadobj_body(std::string const& content)
{
  auto pos = content.find("QUADOBJ");
  if (pos == std::string::npos) { return {}; }
  pos = content.find('\n', pos);
  if (pos == std::string::npos) { return {}; }
  ++pos;
  auto const end = content.find("ENDATA", pos);
  if (end == std::string::npos) { return content.substr(pos); }
  return content.substr(pos, end - pos);
}

/** One QUADOBJ line: default variable names are C0, C1, ... */
struct quadobj_entry_t {
  int row{};
  int col{};
  double value{};
};

bool parse_default_c_index(std::string const& name, int& index_out)
{
  if (name.size() < 2 || name[0] != 'C') { return false; }
  try {
    index_out = std::stoi(name.substr(1));
    return true;
  } catch (...) {
    return false;
  }
}

std::vector<quadobj_entry_t> parse_quadobj_entries(std::string const& quad_body)
{
  std::vector<quadobj_entry_t> entries;
  std::istringstream stream(quad_body);
  std::string line;
  while (std::getline(stream, line)) {
    if (line.empty()) { continue; }
    std::istringstream ls(line);
    std::string row_name;
    std::string col_name;
    if (!(ls >> row_name >> col_name)) { continue; }
    std::string tok;
    std::string last;
    while (ls >> tok) {
      last = tok;
    }
    if (last.empty()) { continue; }
    char* endptr   = nullptr;
    double const v = std::strtod(last.c_str(), &endptr);
    if (endptr == last.c_str()) { continue; }
    int row = 0;
    int col = 0;
    if (!parse_default_c_index(row_name, row) || !parse_default_c_index(col_name, col)) {
      continue;
    }
    entries.push_back({row, col, v});
  }
  return entries;
}

void sort_quadobj_entries_lex(std::vector<quadobj_entry_t>& entries)
{
  std::sort(entries.begin(), entries.end(), [](quadobj_entry_t const& a, quadobj_entry_t const& b) {
    return std::tie(a.row, a.col) < std::tie(b.row, b.col);
  });
}

/** Two-variable QP with no constraints (same pattern as no_constraints.cu). */
template <typename Op>
void setup_two_var_unconstrained_qp(Op& op)
{
  double A_values_host[] = {};
  int A_indices_host[]   = {};
  int A_offsets_host[]   = {0};
  op.set_csr_constraint_matrix(A_values_host, 0, A_indices_host, 0, A_offsets_host, 1);

  double lb_host[] = {0.0, 0.0};
  double ub_host[] = {std::numeric_limits<double>::infinity(),
                      std::numeric_limits<double>::infinity()};
  op.set_variable_lower_bounds(lb_host, 2);
  op.set_variable_upper_bounds(ub_host, 2);

  cuopt::linear_programming::var_t const var_types_host[] = {
    cuopt::linear_programming::var_t::CONTINUOUS, cuopt::linear_programming::var_t::CONTINUOUS};
  op.set_variable_types(var_types_host, 2);

  double c_host[] = {0.0, 0.0};
  op.set_objective_coefficients(c_host, 2);
}

/** Three-variable unconstrained QP; same structure as the 2-variable helper. */
template <typename Op>
void setup_three_var_unconstrained_qp(Op& op)
{
  double A_values_host[] = {};
  int A_indices_host[]   = {};
  int A_offsets_host[]   = {0};
  op.set_csr_constraint_matrix(A_values_host, 0, A_indices_host, 0, A_offsets_host, 1);

  double lb_host[] = {0.0, 0.0, 0.0};
  double ub_host[] = {std::numeric_limits<double>::infinity(),
                      std::numeric_limits<double>::infinity(),
                      std::numeric_limits<double>::infinity()};
  op.set_variable_lower_bounds(lb_host, 3);
  op.set_variable_upper_bounds(ub_host, 3);

  cuopt::linear_programming::var_t const var_types_host[] = {
    cuopt::linear_programming::var_t::CONTINUOUS,
    cuopt::linear_programming::var_t::CONTINUOUS,
    cuopt::linear_programming::var_t::CONTINUOUS};
  op.set_variable_types(var_types_host, 3);

  double c_host[] = {0.0, 0.0, 0.0};
  op.set_objective_coefficients(c_host, 3);
}

}  // namespace

namespace cuopt::linear_programming {

TEST(mps_writer_op, write_to_mps_diagonal_qp_quadobj_matches_symmetrized_hessian)
{
  raft::handle_t handle;

  // minimize x1^2 + x2^2  =>  Q = diag(1,1) in CSR; symmetrized H = Q + Q^T has H_ii = 2.
  auto op = optimization_problem_t<int, double>(&handle);
  setup_two_var_unconstrained_qp(op);

  double Q_values_host[] = {1.0, 1.0};
  int Q_indices_host[]   = {0, 1};
  int Q_offsets_host[]   = {0, 1, 2};
  op.set_quadratic_objective_matrix(Q_values_host, 2, Q_indices_host, 2, Q_offsets_host, 3);

  std::string const path = std::string(::testing::TempDir()) + "qp_diag_write.mps";
  temp_mps_file_guard_t guard(path);
  op.write_to_mps(path);

  std::string const content = read_entire_file(path);
  ASSERT_FALSE(content.empty()) << "MPS file was empty or could not be read";

  std::string const quad_body = extract_quadobj_body(content);
  ASSERT_FALSE(quad_body.empty()) << "No QUADOBJ section in:\n" << content;

  auto entries = parse_quadobj_entries(quad_body);
  ASSERT_EQ(entries.size(), 2u)
    << "Expected two upper-triangular QUADOBJ entries for 2x2 diagonal H";
  sort_quadobj_entries_lex(entries);

  EXPECT_EQ(entries[0].row, 0);
  EXPECT_EQ(entries[0].col, 0);
  EXPECT_NEAR(entries[0].value, 2.0, 1e-9);
  EXPECT_EQ(entries[1].row, 1);
  EXPECT_EQ(entries[1].col, 1);
  EXPECT_NEAR(entries[1].value, 2.0, 1e-9);
}

TEST(mps_writer_op, write_to_mps_nonsymmetric_Q_quadobj_matches_Q_plus_Q_transpose)
{
  raft::handle_t handle;

  // Non-symmetric 3x3 Q; zeros at (0,2) and (2,0) omitted from CSR.
  // Dense Q:
  //   [ 1  2  0 ]
  //   [ 3  4  5 ]
  //   [ 0  7  8 ]
  // H = Q + Q^T has upper-triangle nonzeros 2, 5, 8, 12, 16 (H(0,2)=0 is not written to QUADOBJ).
  auto op = optimization_problem_t<int, double>(&handle);
  setup_three_var_unconstrained_qp(op);

  double Q_values_host[] = {1.0, 2.0, 3.0, 4.0, 5.0, 7.0, 8.0};
  int Q_indices_host[]   = {0, 1, 0, 1, 2, 1, 2};
  int Q_offsets_host[]   = {0, 2, 5, 7};
  op.set_quadratic_objective_matrix(Q_values_host, 7, Q_indices_host, 7, Q_offsets_host, 4);

  std::string const path = std::string(::testing::TempDir()) + "qp_nonsym_sparse_write.mps";
  temp_mps_file_guard_t guard(path);
  op.write_to_mps(path);

  std::string const content = read_entire_file(path);
  ASSERT_FALSE(content.empty()) << "MPS file was empty or could not be read";

  std::string const quad_body = extract_quadobj_body(content);
  ASSERT_FALSE(quad_body.empty()) << "No QUADOBJ section in:\n" << content;

  auto entries = parse_quadobj_entries(quad_body);
  ASSERT_EQ(entries.size(), 5u)
    << "Expected five nonzero upper-triangular QUADOBJ entries (H(0,2)=0 skipped)";
  sort_quadobj_entries_lex(entries);

  std::vector<quadobj_entry_t> const expected = {
    {0, 0, 2.0},
    {0, 1, 5.0},
    {1, 1, 8.0},
    {1, 2, 12.0},
    {2, 2, 16.0},
  };
  for (size_t i = 0; i < expected.size(); ++i) {
    EXPECT_EQ(entries[i].row, expected[i].row) << "entry " << i;
    EXPECT_EQ(entries[i].col, expected[i].col) << "entry " << i;
    EXPECT_NEAR(entries[i].value, expected[i].value, 1e-9) << "entry " << i;
  }
}

}  // namespace cuopt::linear_programming
