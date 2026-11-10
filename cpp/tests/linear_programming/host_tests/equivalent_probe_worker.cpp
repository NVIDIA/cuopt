/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#include <presolve/experimental_equivalent_probe.hpp>

std::vector<double> probe_from_other_translation_unit(const void* problem,
                                                      std::vector<double> probe)
{
  auto state = lpfeas_equivalent_probe::require_context(problem);
  lpfeas_equivalent_probe::permute_probe(*state, probe);
  return probe;
}
