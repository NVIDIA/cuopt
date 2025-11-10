/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#pragma once

#include <mps_parser/mps_data_model.hpp>

#include <memory>

namespace cuopt {
namespace cython {

std::unique_ptr<cuopt::mps_parser::mps_data_model_t<int, double>> call_parse_mps(
  const std::string& mps_file_path, bool fixed_mps_format);

}  // namespace cython
}  // namespace cuopt
