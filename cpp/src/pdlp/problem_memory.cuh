/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#pragma once

#include <rmm/device_uvector.hpp>

namespace cuopt::mathematical_optimization::pdlp {

template <typename T>
void release_workspace(rmm::device_uvector<T>& workspace)
{
  workspace.resize(0, workspace.stream());
  workspace.shrink_to_fit(workspace.stream());
}

}  // namespace cuopt::mathematical_optimization::pdlp
