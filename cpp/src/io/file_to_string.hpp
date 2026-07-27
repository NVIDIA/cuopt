/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#pragma once

#include <string>
#include <vector>

namespace cuopt::mathematical_optimization::io {

// Reads `file` into a buffer and appends a trailing '\0'.
//
// The dispatcher looks at the extension:
//   - ".bz2" → libbz2 (dlopen'd at runtime), if MPS_PARSER_WITH_BZIP2.
//   - ".gz"  → libz   (dlopen'd at runtime), if MPS_PARSER_WITH_ZLIB.
//   - ".lz4" → liblz4 (dlopen'd at runtime), if MPS_PARSER_WITH_LZ4.
//   - otherwise → plain fopen.
// The returned buffer's size includes the null terminator.
std::vector<char> file_to_string(const std::string& file);

}  // namespace cuopt::mathematical_optimization::io
