/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

/**
 * Round-trip tests for the LP writer. Each test parses LP text into a
 * mps_data_model_t (using the trusted read_lp parser), writes it back out with
 * lp_writer_t, re-parses the emitted file, and checks that the two models are
 * structurally equivalent (compared by variable / row name, since LP-format
 * indices are assigned by first appearance).
 */

#include <cuopt/mathematical_optimization/io/lp_writer.hpp>
#include <cuopt/mathematical_optimization/io/mps_data_model.hpp>
#include <cuopt/mathematical_optimization/io/parser.hpp>

#include <gtest/gtest.h>

#include <cmath>
#include <filesystem>
#include <limits>
#include <map>
#include <string>
#include <string_view>
#include <system_error>
#include <unordered_map>
#include <vector>

namespace cuopt::mathematical_optimization::io {

namespace {

constexpr double tol = 1e-9;

struct temp_file_guard_t {
  explicit temp_file_guard_t(std::string p) : path(std::move(p)) {}
  ~temp_file_guard_t()
  {
    if (!path.empty()) {
      std::error_code ec;
      std::filesystem::remove(path, ec);
    }
  }
  std::string path;
};

mps_data_model_t<int, double> read_lp_string(std::string_view content)
{
  return read_lp_from_string<int, double>(content);
}

// Parse -> write -> parse; returns the re-parsed model.
mps_data_model_t<int, double> round_trip(std::string_view lp_text, const std::string& tag)
{
  mps_data_model_t<int, double> original = read_lp_string(lp_text);

  const std::string path = std::string(::testing::TempDir()) + "lp_writer_" + tag + ".lp";
  temp_file_guard_t guard(path);

  lp_writer_t<int, double> writer(original);
  writer.write(path);

  return read_lp<int, double>(path);
}

std::unordered_map<std::string, int> name_to_index(const std::vector<std::string>& names)
{
  std::unordered_map<std::string, int> m;
  for (size_t i = 0; i < names.size(); ++i)
    m[names[i]] = static_cast<int>(i);
  return m;
}

// Compares model `a` (reference) and model `b` (re-parsed) by name.
void expect_equivalent(std::string_view lp_text, const std::string& tag)
{
  mps_data_model_t<int, double> a = read_lp_string(lp_text);
  mps_data_model_t<int, double> b = round_trip(lp_text, tag);

  EXPECT_EQ(a.get_sense(), b.get_sense());
  EXPECT_NEAR(a.get_objective_offset(), b.get_objective_offset(), tol);

  const auto& an = a.get_variable_names();
  const auto& bn = b.get_variable_names();
  ASSERT_EQ(an.size(), bn.size()) << "variable count differs for " << tag;
  auto b_idx = name_to_index(bn);

  const auto& ac = a.get_objective_coefficients();
  const auto& bc = b.get_objective_coefficients();
  const auto& alb = a.get_variable_lower_bounds();
  const auto& blb = b.get_variable_lower_bounds();
  const auto& aub = a.get_variable_upper_bounds();
  const auto& bub = b.get_variable_upper_bounds();
  const auto& at  = a.get_variable_types();
  const auto& bt  = b.get_variable_types();

  for (size_t i = 0; i < an.size(); ++i) {
    ASSERT_TRUE(b_idx.count(an[i])) << "variable '" << an[i] << "' missing after round trip";
    int j = b_idx[an[i]];
    EXPECT_NEAR(ac[i], bc[j], tol) << "objective coeff for " << an[i];
    EXPECT_NEAR(alb[i], blb[j], tol) << "lower bound for " << an[i];
    EXPECT_NEAR(aub[i], bub[j], tol) << "upper bound for " << an[i];
    EXPECT_EQ(at[i], bt[j]) << "type for " << an[i];
  }

  // Compare linear constraints by row name. Build (row name -> map<var name, coeff>) and
  // (row name -> [lb, ub]) for each model.
  auto build_rows = [](const mps_data_model_t<int, double>& m) {
    const auto& rn      = m.get_row_names();
    const auto& names   = m.get_variable_names();
    const auto& offsets = m.get_constraint_matrix_offsets();
    const auto& indices = m.get_constraint_matrix_indices();
    const auto& values  = m.get_constraint_matrix_values();
    const auto& clb     = m.get_constraint_lower_bounds();
    const auto& cub     = m.get_constraint_upper_bounds();
    std::map<std::string, std::map<std::string, double>> coeffs;
    std::map<std::string, std::pair<double, double>> bounds;
    for (size_t r = 0; r < rn.size(); ++r) {
      auto& row = coeffs[rn[r]];
      for (int p = offsets[r]; p < offsets[r + 1]; ++p)
        row[names[indices[p]]] += values[p];
      bounds[rn[r]] = {clb[r], cub[r]};
    }
    return std::make_pair(coeffs, bounds);
  };

  auto [a_coeffs, a_bounds] = build_rows(a);
  auto [b_coeffs, b_bounds] = build_rows(b);
  ASSERT_EQ(a_coeffs.size(), b_coeffs.size()) << "linear row count differs for " << tag;
  for (const auto& [rname, arow] : a_coeffs) {
    ASSERT_TRUE(b_coeffs.count(rname)) << "row '" << rname << "' missing after round trip";
    const auto& brow = b_coeffs[rname];
    EXPECT_EQ(arow.size(), brow.size()) << "nnz differs for row " << rname;
    for (const auto& [vname, coeff] : arow) {
      ASSERT_TRUE(brow.count(vname)) << "var " << vname << " missing in row " << rname;
      EXPECT_NEAR(coeff, brow.at(vname), tol) << "coeff for " << vname << " in row " << rname;
    }
    auto compare_bound = [](double x, double y) {
      if (std::isinf(x) || std::isinf(y)) return (std::isinf(x) && std::isinf(y) && (x > 0) == (y > 0));
      return std::abs(x - y) < tol;
    };
    EXPECT_TRUE(compare_bound(a_bounds[rname].first, b_bounds[rname].first)) << "clb " << rname;
    EXPECT_TRUE(compare_bound(a_bounds[rname].second, b_bounds[rname].second)) << "cub " << rname;
  }

  EXPECT_EQ(a.has_quadratic_objective(), b.has_quadratic_objective());
  EXPECT_EQ(a.get_quadratic_constraints().size(), b.get_quadratic_constraints().size());
}

}  // namespace

TEST(lp_writer, simple_lp_round_trip)
{
  expect_equivalent(R"LP(
Minimize
 obj: 3 x + 2 y - z
Subject To
 c1: x + y <= 10
 c2: x - z >= -4
 c3: 2 x + y = 6
Bounds
 0 <= x <= 8
 y >= 1
 -5 <= z <= 5
End
)LP",
                    "simple_lp");
}

TEST(lp_writer, maximize_and_offset_round_trip)
{
  expect_equivalent(R"LP(
Maximize
 obj: 2 a + 3 b + 7
Subject To
 r1: a + b <= 4
End
)LP",
                    "max_offset");
}

TEST(lp_writer, mip_generals_binaries_round_trip)
{
  expect_equivalent(R"LP(
Minimize
 obj: x + y + 2 z
Subject To
 c1: x + y + z <= 5
Bounds
 0 <= y <= 10
Generals
 y
Binaries
 x
 z
End
)LP",
                    "mip");
}

TEST(lp_writer, free_and_fixed_bounds_round_trip)
{
  expect_equivalent(R"LP(
Minimize
 obj: p + q + r
Subject To
 c1: p + q + r >= 1
Bounds
 p free
 q = 3
 r <= 9
End
)LP",
                    "bounds");
}

TEST(lp_writer, quadratic_objective_round_trip)
{
  // min 0.5*(2 x^2 + 4 x*y + 6 y^2) + x
  expect_equivalent(R"LP(
Minimize
 obj: x + [ 2 x ^ 2 + 4 x * y + 6 y ^ 2 ] / 2
Subject To
 c1: x + y >= 1
End
)LP",
                    "qp_obj");
}

TEST(lp_writer, quadratic_constraint_round_trip)
{
  expect_equivalent(R"LP(
Minimize
 obj: x + y
Subject To
 lin: x + y <= 10
 qc: x + [ x ^ 2 + y ^ 2 ] <= 4
End
)LP",
                    "qcqp");
}

}  // namespace cuopt::mathematical_optimization::io
