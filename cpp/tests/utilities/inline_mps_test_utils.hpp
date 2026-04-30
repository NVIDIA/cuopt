/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#pragma once

#include <mps_parser/parser.hpp>

#include <string_view>

namespace cuopt::test::inline_mps {

static inline const char sc_standard_mps[] = R"(NAME sc_standard
ROWS
 N  OBJ
 G  cover
COLUMNS
    x         OBJ       3
    x         cover     1
    y         OBJ       2
    y         cover     1
RHS
    RHS1      cover     4
BOUNDS
 LO BND1      x         2
 SC BND1      x         10
 LO BND1      y         -10
 UP BND1      y         10
ENDATA
)";

static inline const char sc_no_ub_mps[] = R"(NAME sc_no_ub
ROWS
 N  OBJ
 G  cover
COLUMNS
    x         OBJ       3
    x         cover     1
    y         OBJ       2
    y         cover     1
RHS
    RHS1      cover     4
BOUNDS
 LO BND1      x         2
 SC BND1      x         1e+30
 LO BND1      y         -10
 UP BND1      y         10
ENDATA
)";

static inline const char sc_lb_zero_mps[] = R"(NAME sc_lb_zero
ROWS
 N  OBJ
 G  cover
COLUMNS
    x         OBJ       3
    x         cover     1
    y         OBJ       2
    y         cover     1
RHS
    RHS1      cover     4
BOUNDS
 SC BND1      x         10
 LO BND1      y         -10
 UP BND1      y         10
ENDATA
)";

static inline const char sc_inferred_ub_mps[] = R"(NAME sc_inferred_ub
ROWS
 N  OBJ
 L  cap
COLUMNS
    x         OBJ       -1
    x         cap       1
    y         cap       1
RHS
    RHS1      cap       4
BOUNDS
 LO BND1      x         2
 SC BND1      x         1e+30
 UP BND1      y         10
ENDATA
)";

static inline const char sc_missing_upper_mps[] = R"(NAME sc_missing_upper
ROWS
 N  OBJ
 G  cover
COLUMNS
    x         OBJ       3
    x         cover     1
RHS
    RHS1      cover     4
BOUNDS
 LO BND1      x         2
 SC BND1      x
ENDATA
)";

inline cuopt::mps_parser::mps_data_model_t<int, double> parse_inline_mps(std::string_view mps_text)
{
  return cuopt::mps_parser::parse_mps_from_string<int, double>(mps_text, false);
}

}  // namespace cuopt::test::inline_mps
