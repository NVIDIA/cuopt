// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "file_reader.hpp"

#include <cuopt/mathematical_optimization/io/mps_data_model.hpp>

#include <cstddef>
#include <string>

namespace cuopt::mathematical_optimization::io::detail {

template <typename i_t, typename f_t>
using parser_model_t = mps_data_model_t<i_t, f_t>;

template <typename i_t, typename f_t>
parser_model_t<i_t, f_t> parse_mps_fast_file(const std::string& path,
                                             FileReadMethod read_method = FileReadMethod::Read);

}  // namespace cuopt::mathematical_optimization::io::detail
