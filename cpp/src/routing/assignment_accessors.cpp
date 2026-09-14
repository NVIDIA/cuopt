/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

// Host-only members of assignment_t, split out of assignment.cu so they can build into the
// CUDA-free cuopt_client. They read scalar members only and never touch the device_uvectors
// the rest of the class holds, so compiling them here emits no CUDA runtime dependency.
//
// The routing gRPC solution mapper needs exactly these; keeping them in the CUDA translation
// unit is what forced that mapper into a component linked against the routing engine.

#include <cuopt/error.hpp>
#include <cuopt/export.hpp>
#include <cuopt/routing/assignment.hpp>

#include <map>
#include <string>

namespace cuopt {
namespace routing {

template <typename i_t>
double assignment_t<i_t>::get_total_objective() const
{
  return total_objective_value_;
}

template <typename i_t>
const std::map<objective_t, double>& assignment_t<i_t>::get_objectives() const noexcept
{
  return objective_values_;
}

template <typename i_t>
i_t assignment_t<i_t>::get_vehicle_count() const
{
  return vehicle_count_;
}

template <typename i_t>
std::string assignment_t<i_t>::get_status_string() const noexcept
{
  return solution_string_;
}

template <typename i_t>
solution_status_t assignment_t<i_t>::get_status() const
{
  return status_;
}

template <typename i_t>
cuopt::logic_error assignment_t<i_t>::get_error_status() const noexcept
{
  return error_status_;
}

// Instantiated per member rather than with `template class`: that would also instantiate the
// device-facing members defined in assignment.cu and pull CUDA into this translation unit.
template CUOPT_EXPORT double assignment_t<int>::get_total_objective() const;
template CUOPT_EXPORT const std::map<objective_t, double>& assignment_t<int>::get_objectives()
  const noexcept;
template CUOPT_EXPORT int assignment_t<int>::get_vehicle_count() const;
template CUOPT_EXPORT std::string assignment_t<int>::get_status_string() const noexcept;
template CUOPT_EXPORT solution_status_t assignment_t<int>::get_status() const;
template CUOPT_EXPORT cuopt::logic_error assignment_t<int>::get_error_status() const noexcept;

}  // namespace routing
}  // namespace cuopt
