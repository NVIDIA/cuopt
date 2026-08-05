/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

/**
 * LP writer tests. Primary coverage is exact string output for a range of
 * input problems (specified as LP text for convenience, then re-parsed and
 * rewritten). A couple of round-trip checks are retained for quadratic cases.
 */

#include <cuopt/mathematical_optimization/io/lp_writer.hpp>
#include <cuopt/mathematical_optimization/io/mps_data_model.hpp>
#include <cuopt/mathematical_optimization/io/parser.hpp>

#include <gtest/gtest.h>

#include <filesystem>
#include <fstream>
#include <limits>
#include <sstream>
#include <string>
#include <string_view>
#include <system_error>
#include <vector>

namespace cuopt::mathematical_optimization::io {

namespace {

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

std::string read_file(const std::string& path)
{
  std::ifstream in(path);
  std::ostringstream ss;
  ss << in.rdbuf();
  return ss.str();
}

std::string write_lp_to_string(const mps_data_model_t<int, double>& model, const std::string& tag)
{
  const std::string path = std::string(::testing::TempDir()) + "lp_writer_" + tag + ".lp";
  temp_file_guard_t guard(path);
  lp_writer_t<int, double> writer(model);
  writer.write(path);
  return read_file(path);
}

std::string rewrite_lp_text(std::string_view lp_text, const std::string& tag)
{
  return write_lp_to_string(read_lp_from_string<int, double>(lp_text), tag);
}

}  // namespace

TEST(lp_writer, simple_lp_exact_output)
{
  // Named rows/vars, mixed constraint senses, non-default bounds.
  const std::string out = rewrite_lp_text(R"LP(
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

  EXPECT_EQ(out,
            "Minimize\n"
            " obj: + 3 x + 2 y - 1 z\n"
            "Subject To\n"
            " c1: + 1 x + 1 y <= 10\n"
            " c2: + 1 x - 1 z >= -4\n"
            " c3: + 2 x + 1 y = 6\n"
            "Bounds\n"
            " x <= 8\n"
            " y >= 1\n"
            " -5 <= z <= 5\n"
            "End\n");
}

TEST(lp_writer, maximize_and_offset_exact_output)
{
  const std::string out = rewrite_lp_text(R"LP(
Maximize
 obj: 2 a + 3 b + 7
Subject To
 r1: a + b <= 4
End
)LP",
                                          "max_offset");

  EXPECT_EQ(out,
            "Maximize\n"
            " obj: + 2 a + 3 b + 7\n"
            "Subject To\n"
            " r1: + 1 a + 1 b <= 4\n"
            "End\n");
}

TEST(lp_writer, unnamed_constraints_omit_names)
{
  // No row names in the model → writer must not invent or print names.
  mps_data_model_t<int, double> model;
  const std::vector<double> c{1.0, 1.0};
  const std::vector<double> lb{0.0, 0.0};
  const std::vector<double> ub{std::numeric_limits<double>::infinity(),
                               std::numeric_limits<double>::infinity()};
  const std::vector<std::string> var_names{"x", "y"};
  const std::vector<double> A_values{1.0, 1.0};
  const std::vector<int> A_indices{0, 1};
  const std::vector<int> A_offsets{0, 2};
  const std::vector<double> clb{-std::numeric_limits<double>::infinity()};
  const std::vector<double> cub{5.0};
  const std::vector<char> row_types{'L'};

  model.set_objective_coefficients(c);
  model.set_maximize(false);
  model.set_variable_lower_bounds(lb);
  model.set_variable_upper_bounds(ub);
  model.set_variable_names(var_names);
  model.set_csr_constraint_matrix(A_values, A_indices, A_offsets);
  model.set_constraint_lower_bounds(clb);
  model.set_constraint_upper_bounds(cub);
  model.set_row_types(row_types);

  EXPECT_EQ(write_lp_to_string(model, "unnamed"),
            "Minimize\n"
            " obj: + 1 x + 1 y\n"
            "Subject To\n"
            "  + 1 x + 1 y <= 5\n"
            "End\n");
}

TEST(lp_writer, mip_generals_binaries_exact_output)
{
  const std::string out = rewrite_lp_text(R"LP(
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

  EXPECT_EQ(out,
            "Minimize\n"
            " obj: + 1 x + 1 y + 2 z\n"
            "Subject To\n"
            " c1: + 1 x + 1 y + 1 z <= 5\n"
            "Bounds\n"
            " y <= 10\n"
            "Generals\n"
            " y\n"
            "Binaries\n"
            " x\n"
            " z\n"
            "End\n");
}

TEST(lp_writer, free_and_fixed_bounds_exact_output)
{
  const std::string out = rewrite_lp_text(R"LP(
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

  EXPECT_EQ(out,
            "Minimize\n"
            " obj: + 1 p + 1 q + 1 r\n"
            "Subject To\n"
            " c1: + 1 p + 1 q + 1 r >= 1\n"
            "Bounds\n"
            " p free\n"
            " q = 3\n"
            " r <= 9\n"
            "End\n");
}

TEST(lp_writer, quadratic_objective_exact_output)
{
  // min 0.5*(2 x^2 + 4 x*y + 6 y^2) + x  →  written with H = Q+Q^T convention
  const std::string out = rewrite_lp_text(R"LP(
Minimize
 obj: x + [ 2 x ^ 2 + 4 x * y + 6 y ^ 2 ] / 2
Subject To
 c1: x + y >= 1
End
)LP",
                                          "qp_obj");

  EXPECT_EQ(out,
            "Minimize\n"
            " obj: + 1 x + [ + 2 x ^ 2 + 4 x * y + 6 y ^ 2 ] / 2\n"
            "Subject To\n"
            " c1: + 1 x + 1 y >= 1\n"
            "End\n");
}

TEST(lp_writer, quadratic_constraint_round_trip)
{
  // Retain one round-trip for QCQP content (parser/writer quadratic path).
  const auto original = read_lp_from_string<int, double>(R"LP(
Minimize
 obj: x + y
Subject To
 lin: x + y <= 10
 qc: x + [ x ^ 2 + y ^ 2 ] <= 4
End
)LP");
  const auto rewritten =
    read_lp_from_string<int, double>(write_lp_to_string(original, "qcqp_rt"));

  EXPECT_EQ(original.get_sense(), rewritten.get_sense());
  EXPECT_EQ(original.get_quadratic_constraints().size(),
            rewritten.get_quadratic_constraints().size());
  ASSERT_EQ(rewritten.get_quadratic_constraints().size(), 1u);
  EXPECT_EQ(rewritten.get_quadratic_constraints()[0].constraint_row_name, "qc");
  EXPECT_NEAR(rewritten.get_quadratic_constraints()[0].rhs_value, 4.0, 1e-9);
}

}  // namespace cuopt::mathematical_optimization::io
