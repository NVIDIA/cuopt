// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <cuopt/routing/data_model_view.hpp>
#include <cuopt/routing/solve.hpp>

#include <raft/core/handle.hpp>
#include <rmm/device_uvector.hpp>

#include <cmath>
#include <cstdio>
#include <vector>

// Four locations on a line, one vehicle: the best tour costs 1 + 1 + 1 + 3.
int main()
{
  raft::handle_t handle;
  auto stream             = handle.get_stream();
  std::vector<float> cost = {0, 1, 2, 3, 1, 0, 1, 2, 2, 1, 0, 1, 3, 2, 1, 0};
  rmm::device_uvector<float> d_cost(cost.size(), stream);
  raft::update_device(d_cost.data(), cost.data(), cost.size(), stream);
  handle.sync_stream();

  cuopt::routing::data_model_view_t<int, float> model(&handle, 4, 1);
  model.add_cost_matrix(d_cost.data());
  cuopt::routing::solver_settings_t<int, float> settings;
  settings.set_time_limit(1);

  auto solution = cuopt::routing::solve(model, settings);
  std::printf(
    "%s objective=%f\n", solution.get_status_string().c_str(), solution.get_total_objective());
  return solution.get_status() == cuopt::routing::solution_status_t::SUCCESS &&
             std::fabs(solution.get_total_objective() - 6.0) < 1e-3
           ? 0
           : 1;
}
