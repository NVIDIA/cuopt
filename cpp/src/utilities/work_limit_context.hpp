/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved. SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#pragma once

#include <algorithm>
#include <string>

#include <mip/logger.hpp>

#include "timer.hpp"
#include "work_unit_scheduler.hpp"

namespace cuopt {

struct work_limit_context_t {
  double global_work_units_elapsed{0.0};
  double total_sync_time{0.0};  // Total time spent waiting at sync barriers (seconds)
  bool deterministic{false};
  work_unit_scheduler_t* scheduler{nullptr};
  std::string name;

  work_limit_context_t(const std::string& name) : name(name) {}

  void record_work_sync_on_horizon(double work)
  {
    if (!deterministic) return;
    global_work_units_elapsed += work;
    if (scheduler) { scheduler->on_work_recorded(*this, global_work_units_elapsed); }
  }
};

}  // namespace cuopt
