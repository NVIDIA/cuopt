/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved. SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <raft/common/nvtx.hpp>
#include <utilities/logger.hpp>
#include "solution_writer.hpp"

#include <fstream>

namespace cuopt::linear_programming {

void solution_writer_t::write_solution_to_sol_file(const std::string& filename,
                                                   const std::string& status,
                                                   const double objective_value,
                                                   const std::vector<std::string>& variable_names,
                                                   const std::vector<double>& variable_values)
{
  raft::common::nvtx::range fun_scope("write final solution to .sol file");
  std::ofstream file(filename.data());

  if (!file.is_open()) {
    CUOPT_LOG_ERROR("Could not open file: %s for solution output", filename.data());
    return;
  }

  file.precision(std::numeric_limits<double>::max_digits10 + 1);

  file << "# Status: " << status << std::endl;

  if (status != "Infeasible") {
    file << "# Objective value: " << objective_value << std::endl;
    for (size_t i = 0; i < variable_names.size(); ++i) {
      file << variable_names[i] << " " << variable_values[i] << std::endl;
    }
  }
}

}  // namespace cuopt::linear_programming
