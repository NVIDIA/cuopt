/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#pragma once

#if defined(__GNUC__) || defined(__clang__)
#define MPS_PARSER_EXPORT __attribute__((visibility("default")))
#else
#define MPS_PARSER_EXPORT
#endif
