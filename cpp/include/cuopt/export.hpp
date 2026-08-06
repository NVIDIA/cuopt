/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#pragma once

#if defined(__GNUC__) || defined(__clang__)
#define CUOPT_EXPORT __attribute__((visibility("default")))
#else
#define CUOPT_EXPORT
#endif
