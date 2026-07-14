/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#pragma once

#if defined(__GNUC__) || defined(__clang__)
#define CUOPT_EXPORT __attribute__((visibility("default")))
// Marks internal symbols that are exported solely for test access.
// These are NOT part of the stable public API and may change without notice.
#define CUOPT_INTERNAL_EXPORT __attribute__((visibility("default")))
#else
#define CUOPT_EXPORT
#define CUOPT_INTERNAL_EXPORT
#endif
