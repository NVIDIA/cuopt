/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */
#include <cassert>
#include "original_checks.hpp"

static bool flag(const Json& j, const std::string& key)
{
  return j.out.str().find("\"" + key + "\":true") != std::string::npos;
}
static Model base(int sense = 1)
{
  Model a;
  a.m      = 1;
  a.n      = 2;
  a.nnz    = 2;
  a.sense  = sense;
  a.offset = 7;
  a.ap     = {0, 2};
  a.ai     = {0, 1};
  a.ax     = {1, 1};
  a.c      = {1, 2};
  a.lhs    = {1};
  a.rhs    = {3};
  a.lb     = {0, 0};
  a.ub     = {2, 2};
  a.types  = {'I', 'C'};
  return a;
}
static Json run(Model a, std::vector<double> x, std::vector<double> y, std::vector<double> z)
{
  signed_minimum_diagnostics(a, y, z);
  Json j;
  validate(a, Solution{x.data(), y.data(), z.data(), size_t(a.n), size_t(a.m)}, j, 1e-4, 1e-6);
  return j;
}
int main()
{
  // min x+2y+7 subject to1<=x+y<=3,0<=x,y<=2; optimum(1,0).
  auto a       = base();
  auto minimum = run(a, {1, 0}, {1}, {0, 1});
  assert(flag(minimum, "original_native_formula_all_checks"));
  assert(flag(minimum, "mixed_inf_diagnostic_checks"));
  assert(minimum.out.str().find("\"primal_objective\":8") != std::string::npos);
  // max same model; optimum(1,2), native upper-bound multipliers positive.
  auto maximum = run(base(-1), {1, 2}, {1}, {0, 1});
  assert(flag(maximum, "original_native_formula_all_checks"));
  assert(flag(maximum, "mixed_inf_diagnostic_checks"));
  assert(maximum.out.str().find("\"primal_objective\":12") != std::string::npos);
  // Ranged row violation and wrong upper-side sign independently fail.
  assert(!flag(run(a, {2, 2}, {1}, {0, 1}), "original_native_formula_all_checks"));
  assert(!flag(run(base(-1), {1, 2}, {-1}, {0, -1}), "mixed_inf_diagnostic_checks"));
  assert(!flag(run(a, {1, 0}, {0}, {0, 0}), "original_native_formula_all_checks"));
  assert(
    !flag(run(a, {1, 0}, {1}, {0, std::numeric_limits<double>::infinity()}), "validation_finite"));
  assert(
    !flag(run(a, {std::numeric_limits<double>::quiet_NaN(), 0}, {1}, {0, 1}), "validation_finite"));
  // Canonicalization preserves row/column coordinates, merges exact duplicates
  // and drops a cancelled duplicate while retaining objective/bounds/types.
  a.ap             = {0, 4};
  a.ai             = {1, 0, 1, 0};
  a.ax             = {2, 3, -1, -2};
  a.nnz            = 4;
  const auto old_c = a.c, old_lhs = a.lhs, old_rhs = a.rhs, old_lb = a.lb, old_ub = a.ub;
  const auto old_types = a.types;
  a.canonicalize();
  assert(a.ap == std::vector<int>({0, 2}) && a.ai == std::vector<int>({0, 1}));
  assert(a.ax == std::vector<double>({1, 1}) && a.nnz == 2);
  assert(a.c == old_c && a.lhs == old_lhs && a.rhs == old_rhs && a.lb == old_lb && a.ub == old_ub &&
         a.types == old_types);
  a.ap  = {0, 2};
  a.ai  = {0, 0};
  a.ax  = {1, -1};
  a.nnz = 2;
  a.canonicalize();
  assert(a.nnz == 0 && a.ap == std::vector<int>({0, 0}));
  // Empty model and dimension mismatch are explicit, never dereferenced.
  Model empty;
  empty.ap = {0};
  empty.canonicalize();
  auto zero = run(empty, {}, {}, {});
  assert(flag(zero, "validation_finite") && flag(zero, "original_native_formula_all_checks"));
  bool rejected = false;
  try {
    Json j;
    auto original = base();
    validate(original, Solution{nullptr, nullptr, nullptr, 0, 0}, j, 1e-4, 1e-6);
  } catch (const std::runtime_error&) {
    rejected = true;
  }
  assert(rejected);
  // Invalid CSR is rejected before the native API is called.
  a        = base();
  a.ai[0]  = 3;
  rejected = false;
  try {
    a.canonicalize();
  } catch (const std::runtime_error&) {
    rejected = true;
  }
  assert(rejected);
  std::cout << "12 original-space CPU check groups passed\n";
}
