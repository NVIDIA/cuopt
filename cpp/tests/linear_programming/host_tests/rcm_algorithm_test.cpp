/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#include <presolve/experimental_rcm_algorithm.hpp>

#include <cstdint>
#include <cstring>
#include <iostream>

void require(bool condition, const char* message)
{
  if (!condition) { throw std::runtime_error(message); }
}

bool same_bits(double a, double b)
{
  std::uint64_t x, y;
  std::memcpy(&x, &a, sizeof(x));
  std::memcpy(&y, &b, sizeof(y));
  return x == y;
}

struct fixture {
  std::vector<int> Ap, Ai;
  std::vector<double> Ax, c, lhs, rhs, lbs, ubs;
  lpfeas_ordering::problem_view view() const
  {
    return {lhs.size(),
            c.size(),
            Ax.size(),
            Ap.data(),
            Ai.data(),
            Ax.data(),
            c.data(),
            lhs.data(),
            rhs.data(),
            lbs.data(),
            ubs.data()};
  }
};

fixture make_fixture(std::size_t m, std::size_t n)
{
  fixture f;
  f.Ap.resize(m + 1, 0);
  f.c.resize(n);
  f.lbs.resize(n);
  f.ubs.resize(n);
  f.lhs.resize(m);
  f.rhs.resize(m);
  for (std::size_t j = 0; j < n; ++j) {
    f.c[j]   = j == 0 ? -0.0 : double(j) / 8;
    f.lbs[j] = j % 3 ? -double(j + 1) : -std::numeric_limits<double>::infinity();
    f.ubs[j] = j % 3 ? double(j + 3) : std::numeric_limits<double>::infinity();
  }
  for (std::size_t i = 0; i < m; ++i) {
    f.lhs[i] = i % 2 ? double(i) : -std::numeric_limits<double>::infinity();
    f.rhs[i] = double(i + 1);
  }
  return f;
}

void exact_check(const lpfeas_ordering::problem_view& original,
                 const lpfeas_ordering::reordered_problem& reordered)
{
  const auto p = reordered.view();
  require(p.m == original.m && p.n == original.n && p.nnz == original.nnz, "Shape changed");
  auto rows = reordered.row_new_to_old, cols = reordered.column_new_to_old;
  std::sort(rows.begin(), rows.end());
  std::sort(cols.begin(), cols.end());
  for (std::size_t i = 0; i < p.m; ++i) {
    require(rows[i] == int(i), "Row map not bijective");
    const auto old_i = reordered.row_new_to_old[i];
    require(reordered.row_old_to_new[old_i] == int(i), "Wrong row inverse");
    require(same_bits(p.lhs[i], original.lhs[old_i]) && same_bits(p.rhs[i], original.rhs[old_i]),
            "Row bounds changed");
    require(p.Ap[i + 1] - p.Ap[i] == original.Ap[old_i + 1] - original.Ap[old_i],
            "Row degree changed");
    for (int k = p.Ap[i]; k < p.Ap[i + 1]; ++k) {
      const auto old_j = reordered.column_new_to_old[p.Ai[k]];
      const auto begin = original.Ai + original.Ap[old_i];
      const auto end   = original.Ai + original.Ap[old_i + 1];
      const auto found = std::lower_bound(begin, end, old_j);
      require(found != end && *found == old_j, "Coordinate changed");
      require(same_bits(p.Ax[k], original.Ax[found - original.Ai]), "Coefficient bits changed");
    }
  }
  std::vector<double> x(std::max(std::size_t(1), p.n)), y(std::max(std::size_t(1), p.m)),
    z(x.size());
  std::vector<double> expected_x(x.size()), expected_y(y.size()), expected_z(z.size());
  for (std::size_t j = 0; j < p.n; ++j) {
    require(cols[j] == int(j), "Column map not bijective");
    const auto old_j = reordered.column_new_to_old[j];
    require(reordered.column_old_to_new[old_j] == int(j), "Wrong column inverse");
    require(same_bits(p.c[j], original.c[old_j]) && same_bits(p.lbs[j], original.lbs[old_j]) &&
              same_bits(p.ubs[j], original.ubs[old_j]),
            "Objective or variable bounds changed");
    x[j]              = old_j ? double(old_j) / 16 : -0.0;
    z[j]              = -double(old_j + 3) / 8;
    expected_x[old_j] = x[j];
    expected_z[old_j] = z[j];
  }
  for (std::size_t i = 0; i < p.m; ++i) {
    const auto old_i  = reordered.row_new_to_old[i];
    y[i]              = old_i ? -double(old_i) / 4 : -0.0;
    expected_y[old_i] = y[i];
  }
  reordered.restore_solution(x, y, z);
  for (std::size_t j = 0; j < x.size(); ++j) {
    require(same_bits(x[j], expected_x[j]) && same_bits(z[j], expected_z[j]),
            "Primal/reduced-cost inverse changed bits");
  }
  for (std::size_t i = 0; i < y.size(); ++i) {
    require(same_bits(y[i], expected_y[i]), "Dual inverse changed bits");
  }
}

int main()
{
  try {
    int tests = 0;
    {
      struct saved_environment {
        const char* name;
        bool present;
        std::string value;
        explicit saved_environment(const char* key)
          : name(key), present(std::getenv(key) != nullptr), value(present ? std::getenv(key) : "")
        {
        }
        ~saved_environment()
        {
          if (present) {
            setenv(name, value.c_str(), 1);
          } else {
            unsetenv(name);
          }
        }
      } mode("LPFEAS_REORDER"), threshold("LPFEAS_REORDER_HUB_THRESHOLD");
      unsetenv(mode.name);
      unsetenv(threshold.name);
      require(!lpfeas_ordering::from_environment().enabled &&
                lpfeas_ordering::from_environment().hub_threshold == 1024,
              "Wrong defaults");
      setenv(mode.name, "rcm", 1);
      setenv(threshold.name, "4096", 1);
      require(lpfeas_ordering::from_environment().enabled &&
                lpfeas_ordering::from_environment().hub_threshold == 4096,
              "Wrong explicit options");
      for (const char* invalid : {"", "bad", " RCM"}) {
        setenv(mode.name, invalid, 1);
        bool rejected = false;
        try {
          (void)lpfeas_ordering::from_environment();
        } catch (const std::exception&) {
          rejected = true;
        }
        require(rejected, "Invalid mode accepted");
      }
      setenv(mode.name, "off", 1);
      for (const char* invalid : {"", "0", "-1", " 1024", "1024x"}) {
        setenv(threshold.name, invalid, 1);
        bool rejected = false;
        try {
          (void)lpfeas_ordering::from_environment();
        } catch (const std::exception&) {
          rejected = true;
        }
        require(rejected, "Invalid threshold accepted");
      }
      ++tests;
    }
    for (auto dimensions : {std::pair<std::size_t, std::size_t>{0, 0}, {0, 5}, {4, 0}}) {
      auto f = make_fixture(dimensions.first, dimensions.second);
      exact_check(f.view(), lpfeas_ordering::reorder(f.view(), 1024));
      ++tests;
    }
    auto f            = make_fixture(5, 7);
    f.Ap              = {0, 3, 5, 5, 10, 12};
    f.Ai              = {0, 3, 5, 1, 4, 0, 1, 2, 4, 5, 3, 5};
    f.Ax              = {-0.0, 1.25, -2.5, 3.0, -4.0, 0.5, -1.0, 2.0, 8.0, -16.0, 0.25, -0.125};
    const auto before = f.Ax;
    for (std::size_t threshold : {2, 4, 1024}) {
      const auto result = lpfeas_ordering::reorder(f.view(), threshold);
      exact_check(f.view(), result);
      const auto again = lpfeas_ordering::reorder(f.view(), threshold);
      require(result.row_new_to_old == again.row_new_to_old &&
                result.column_new_to_old == again.column_new_to_old,
              "Ordering is nondeterministic");
      for (std::size_t k = 0; k < before.size(); ++k) {
        require(same_bits(before[k], f.Ax[k]), "Input mutated");
      }
      if (threshold == 4) {
        require(result.excluded_rows == 1 && result.excluded_nnz == 5, "Hub statistics wrong");
      }
      ++tests;
    }
    {
      auto invalid  = f;
      invalid.Ai[1] = invalid.Ai[0];
      bool rejected = false;
      try {
        (void)lpfeas_ordering::reorder(invalid.view(), 1024);
      } catch (const std::runtime_error&) {
        rejected = true;
      }
      require(rejected, "Duplicate CSR was not rejected");
      ++tests;
    }
    {
      const auto result = lpfeas_ordering::reorder(f.view(), 1024);
      std::vector<double> wrong;
      bool rejected = false;
      try {
        result.restore_solution(wrong, wrong, wrong);
      } catch (const std::runtime_error&) {
        rejected = true;
      }
      require(rejected, "Wrong solution shape was not rejected");
      ++tests;
    }
    std::cout << "ORDERING_CPU_TESTS_PASS count=" << tests << '\n';
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "ORDERING_CPU_TESTS_FAIL " << error.what() << '\n';
    return 1;
  }
}
