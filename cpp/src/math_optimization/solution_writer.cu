/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#include <raft/core/nvtx.hpp>
#include <utilities/logger.hpp>
#include "solution_writer.hpp"

#include <mip_heuristics/mip_constants.hpp>

#include <fstream>

namespace cuopt::linear_programming {

template <typename f_t>
void solution_writer_t::write_solution_to_sol_file(const std::string& filename,
                                                   const std::string& status,
                                                   const f_t objective_value,
                                                   const std::vector<std::string>& variable_names,
                                                   const std::vector<f_t>& variable_values)
{
  raft::common::nvtx::range fun_scope("write final solution to .sol file");
  std::ofstream file(filename.data());

  if (!file.is_open()) {
    CUOPT_LOG_ERROR("Could not open file: %s for solution output", filename.data());
    return;
  }

  file.precision(std::numeric_limits<f_t>::max_digits10 + 1);

  file << "# Status: " << status << std::endl;

  if (status != "Infeasible") {
    file << "# Objective value: " << objective_value << std::endl;
    for (size_t i = 0; i < variable_names.size(); ++i) {
      file << variable_names[i] << " " << variable_values[i] << std::endl;
    }
  }
}

#if MIP_INSTANTIATE_FLOAT || PDLP_INSTANTIATE_FLOAT
template void solution_writer_t::write_solution_to_sol_file<float>(
  const std::string& filename,
  const std::string& status,
  const float objective_value,
  const std::vector<std::string>& variable_names,
  const std::vector<float>& variable_values);
#endif

#if MIP_INSTANTIATE_DOUBLE
template void solution_writer_t::write_solution_to_sol_file<double>(
  const std::string& filename,
  const std::string& status,
  const double objective_value,
  const std::vector<std::string>& variable_names,
  const std::vector<double>& variable_values);
#endif

}  // namespace cuopt::linear_programming
