/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#include <presolve/experimental_equivalent_probe_registry.inc>

#include <array>
#include <future>
#include <iostream>

std::vector<double> probe_from_other_translation_unit(const void*, std::vector<double>);

namespace ep = lpfeas_equivalent_probe;

template <class F>
void must_throw(F&& operation)
{
  bool threw = false;
  try {
    operation();
  } catch (const std::runtime_error&) {
    threw = true;
  }
  if (!threw) { throw std::runtime_error("Expected failure did not occur"); }
}

void require(bool condition)
{
  if (!condition) { throw std::runtime_error("CPU equivalent-probe fixture failed"); }
}

std::array<double, 3> gram(const double a[3][3], const std::vector<double>& z)
{
  std::array<double, 3> atz{}, out{};
  for (int j = 0; j < 3; ++j) {
    for (int i = 0; i < 3; ++i) {
      atz[j] += a[i][j] * z[i];
    }
  }
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) {
      out[i] += a[i][j] * atz[j];
    }
  }
  return out;
}

int main()
{
  unsetenv("LPFEAS_REORDER");
  unsetenv("LPFEAS_INIT_DIAGNOSTICS");
  unsetenv("LPFEAS_EQUIVALENT_PROBE");
  unsetenv("LPFEAS_EQUIVALENT_PROBE_DUMP_DIR");
  unsetenv("LPFEAS_EQUIVALENT_PROBE_DUMP_SCALED");
  int owner_a, owner_b, problem_a, problem_b, unknown;
  require(!ep::enabled());
  require(!ep::diagnostics_enabled());
  setenv("LPFEAS_REORDER", "rcm", 1);
  require(ep::enabled());
  require(!ep::diagnostics_enabled());
  setenv("LPFEAS_EQUIVALENT_PROBE", "0", 1);
  must_throw([] { ep::enabled(); });
  unsetenv("LPFEAS_EQUIVALENT_PROBE");
  setenv("LPFEAS_INIT_DIAGNOSTICS", "1", 1);
  require(ep::diagnostics_enabled());
  unsetenv("LPFEAS_INIT_DIAGNOSTICS");
  setenv("LPFEAS_EQUIVALENT_PROBE_DUMP_DIR", "unused-probe-fixture", 1);
  require(ep::diagnostics_enabled());
  unsetenv("LPFEAS_EQUIVALENT_PROBE_DUMP_DIR");
  setenv("LPFEAS_REORDER", "off", 1);
  require(!ep::enabled());
  setenv("LPFEAS_EQUIVALENT_PROBE", "0", 1);
  require(!ep::enabled());
  unsetenv("LPFEAS_REORDER");
  setenv("LPFEAS_EQUIVALENT_PROBE", "1", 1);
  require(ep::enabled());
  setenv("LPFEAS_EQUIVALENT_PROBE", "", 1);
  must_throw([] { ep::enabled(); });
  unsetenv("LPFEAS_EQUIVALENT_PROBE");
  must_throw([&] { ep::require_context(&problem_a); });
  must_throw([&] { ep::publish(&owner_a, 3, 3, false, {}, {}); });

  const std::vector<int> row_map{2, 0, 1}, column_map{1, 2, 0};
  const std::vector<double> original{1.0, 2.0, 4.0};
  std::shared_ptr<const ep::context> retained;
  {
    ep::scoped_context scope(&owner_a);
    must_throw([&] { ep::scoped_context duplicate(&owner_a); });
    must_throw([&] { scope.bind_problem(&problem_a, 3, 3); });
    must_throw([&] { ep::publish(&owner_a, 3, 3, true, {0, 0, 2}, column_map); });
    must_throw([&] { ep::publish(&owner_a, 3, 3, true, {-1, 0, 2}, column_map); });
    must_throw([&] { ep::publish(&owner_a, 3, 3, true, {3, 0, 2}, column_map); });
    must_throw([&] { ep::publish(&owner_a, 3, 3, true, {0, 2}, column_map); });
    ep::publish(&owner_a, 3, 3, true, row_map, column_map);
    must_throw([&] { ep::publish(&owner_a, 3, 3, true, row_map, column_map); });
    must_throw([&] { scope.bind_problem(&problem_a, 2, 3); });
    scope.bind_problem(&problem_a, 3, 3);
    retained = ep::require_context(&problem_a);
    must_throw([&] { ep::require_context(&unknown); });
    require(probe_from_other_translation_unit(&problem_a, original) ==
            std::vector<double>({4, 1, 2}));

    ep::scoped_context other(&owner_b);
    ep::publish(&owner_b, 3, 3, false, {}, {});
    must_throw([&] { other.bind_problem(&problem_a, 3, 3); });
    other.bind_problem(&problem_b, 3, 3);
    auto worker_a =
      std::async(std::launch::async, probe_from_other_translation_unit, &problem_a, original);
    auto worker_b =
      std::async(std::launch::async, probe_from_other_translation_unit, &problem_b, original);
    require(worker_a.get() == std::vector<double>({4, 1, 2}));
    require(worker_b.get() == original);
    must_throw([&] {
      auto wrong = std::vector<double>{1};
      ep::permute_probe(*retained, wrong);
    });

    // With exact small integer operations, P A Q^T (P A Q^T)^T P z = P A A^T z.
    const double matrix[3][3] = {{2, -1, 0}, {3, 0, 4}, {0, 5, 1}};
    double reordered[3][3];
    for (int i = 0; i < 3; ++i) {
      for (int j = 0; j < 3; ++j) {
        reordered[i][j] = matrix[row_map[i]][column_map[j]];
      }
    }
    const auto old_gram = gram(matrix, original);
    const auto new_gram = gram(reordered, probe_from_other_translation_unit(&problem_a, original));
    for (int i = 0; i < 3; ++i) {
      require(new_gram[i] == old_gram[row_map[i]]);
    }
  }
  must_throw([&] { ep::require_context(&problem_a); });
  must_throw([&] { ep::require_context(&problem_b); });
  auto still_owned = original;
  ep::permute_probe(*retained, still_owned);
  require(still_owned == std::vector<double>({4, 1, 2}));
  try {
    ep::scoped_context scope(&owner_a);
    ep::publish(&owner_a, 3, 3, false, {}, {});
    scope.bind_problem(&problem_a, 3, 3);
    throw 7;
  } catch (int) {
  }
  must_throw([&] { ep::require_context(&problem_a); });

  // Byte preservation covers signed zero, subnormal, infinities, and distinct NaN payloads.
  const std::array<std::uint64_t, 8> raw = {0x0000000000000000ULL,
                                            0x8000000000000000ULL,
                                            0x0000000000000001ULL,
                                            0x7ff0000000000000ULL,
                                            0xfff0000000000000ULL,
                                            0x7ff8000000000011ULL,
                                            0x7ff0000000000022ULL,
                                            0x3ff0000000000000ULL};
  std::vector<double> values(raw.size());
  std::memcpy(values.data(), raw.data(), sizeof(raw));
  ep::context bits{nullptr, raw.size(), 0, true, {7, 6, 5, 4, 3, 2, 1, 0}, {}, {}};
  ep::permute_probe(bits, values);
  for (std::size_t i = 0; i < values.size(); ++i) {
    std::uint64_t result;
    std::memcpy(&result, &values[i], sizeof(result));
    require(result == raw[raw.size() - i - 1]);
  }
  std::cout << "PASS: map direction, Gram equivalence, exact scalar bytes, identity, invalid maps, "
               "two-TU lookup, concurrent isolation, lifetime and exception cleanup\n";
}
