/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#include <utilities/seed_generator.cuh>

int64_t cuopt::seed_generator::base_seed_ = 0;
std::atomic<int64_t> cuopt::seed_generator::epoch_{0};
