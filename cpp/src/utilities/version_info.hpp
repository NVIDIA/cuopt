/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */
#pragma once

namespace cuopt {
// Prints devices [0, num_devices). Defaults to the first visible device.
void print_version_info(int num_devices = 1);
}  // namespace cuopt
