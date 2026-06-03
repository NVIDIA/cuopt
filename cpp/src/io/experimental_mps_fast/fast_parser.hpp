// SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights
// reserved. SPDX-License-Identifier: Apache-2.0

#pragma once

#include "file_reader.hpp"

#include <cuopt/linear_programming/io/mps_data_model.hpp>

#include <cstddef>
#include <string>

namespace mps_fast {

template <typename i_t, typename f_t>
cuopt::linear_programming::io::mps_data_model_t<i_t, f_t> parse_mps_fast_file(
  const std::string& path, FileReadMethod read_method = FileReadMethod::Read);

}  // namespace mps_fast
