/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#include "regret_kernels.cuh"
#include "vehicle_assignment.cuh"

namespace cuopt {
namespace routing {
namespace detail {

template <typename i_t, typename f_t, request_t REQUEST>
auto compute_route_costs(solution_t<i_t, f_t, REQUEST>& sol,
                         move_candidates_t<i_t, f_t>& move_candidates,
                         vehicle_assignment_t<i_t, f_t, REQUEST>& vehicle_assignment)
{
  auto n_blocks      = sol.get_n_routes() * sol.problem_ptr->get_num_buckets();
  auto constexpr TPB = 128;
  auto shmem         = sol.check_routes_can_insert_and_get_sh_size(0);
  bool is_set        = set_shmem_of_kernel(compute_route_costs_kernel<i_t, f_t, REQUEST>, shmem);
  if (!is_set) { return false; }

  compute_route_costs_kernel<i_t, f_t, REQUEST>
    <<<n_blocks, TPB, shmem, sol.sol_handle->get_stream()>>>(
      sol.view(), move_candidates.view(), vehicle_assignment.view());
  RAFT_CHECK_CUDA(sol.sol_handle->get_stream());
  return true;
}

template <typename i_t, typename f_t, request_t REQUEST>
auto compute_route_cost_differences(solution_t<i_t, f_t, REQUEST>& sol,
                                    vehicle_assignment_t<i_t, f_t, REQUEST>& vehicle_assignment)
{
  auto n_blocks      = sol.get_n_routes() * (vehicle_assignment.get_k_regrets() - 1);
  auto constexpr TPB = min_bucket_entries;
  auto shmem = sizeof(double) * (std::max(sol.problem_ptr->get_num_buckets(), min_bucket_entries) +
                                 min_bucket_entries) +
               sizeof(i_t) * min_bucket_entries;
  bool is_set =
    set_shmem_of_kernel(compute_route_cost_differences_kernel<i_t, f_t, REQUEST, TPB>, shmem);
  if (!is_set) { return false; }

  compute_route_cost_differences_kernel<i_t, f_t, REQUEST, TPB>
    <<<n_blocks, TPB, shmem, sol.sol_handle->get_stream()>>>(sol.view(), vehicle_assignment.view());
  RAFT_CHECK_CUDA(sol.sol_handle->get_stream());
  return true;
}

template <typename i_t, typename f_t, request_t REQUEST>
auto compute_route_vehicle_assignments(solution_t<i_t, f_t, REQUEST>& sol,
                                       vehicle_assignment_t<i_t, f_t, REQUEST>& vehicle_assignment)
{
  auto n_blocks      = sol.get_n_routes() * (vehicle_assignment.get_k_regrets() - 1);
  auto constexpr TPB = 128;
  auto shmem         = sizeof(double) * (TPB / warp_size);
  bool is_set =
    set_shmem_of_kernel(compute_route_vehicle_assignments_kernel<i_t, f_t, REQUEST>, shmem);
  if (!is_set) { return false; }

  compute_route_vehicle_assignments_kernel<i_t, f_t, REQUEST>
    <<<n_blocks, TPB, shmem, sol.sol_handle->get_stream()>>>(sol.view(), vehicle_assignment.view());
  RAFT_CHECK_CUDA(sol.sol_handle->get_stream());
  return true;
}

template <typename i_t, typename f_t, request_t REQUEST>
auto update_assignment(solution_t<i_t, f_t, REQUEST>& sol,
                       move_candidates_t<i_t, f_t>& move_candidates,
                       vehicle_assignment_t<i_t, f_t, REQUEST>& vehicle_assignment)
{
  auto constexpr TPB = 128;
  auto shmem         = sol.check_routes_can_insert_and_get_sh_size(0);
  bool is_set        = set_shmem_of_kernel(update_assignment_kernel<i_t, f_t, REQUEST>, shmem);
  if (!is_set) { return false; }

  auto k_iter = vehicle_assignment.get_k_regrets() - 1;
  update_assignment_kernel<i_t, f_t, REQUEST><<<k_iter, TPB, shmem, sol.sol_handle->get_stream()>>>(
    sol.view(), move_candidates.view(), vehicle_assignment.view());
  RAFT_CHECK_CUDA(sol.sol_handle->get_stream());
  return true;
}

template <typename i_t, typename f_t, request_t REQUEST>
void reset_vehicle_availability(solution_t<i_t, f_t, REQUEST>& sol,
                                vehicle_assignment_t<i_t, f_t, REQUEST>& vehicle_assignment)
{
  auto constexpr TPB = 128;
  async_fill(vehicle_assignment.vehicle_availability, -1, sol.sol_handle->get_stream());
  auto k_iter = vehicle_assignment.get_k_regrets() - 1;
  reset_vehicle_availability_kernel<i_t, f_t, REQUEST>
    <<<k_iter, TPB, 0, sol.sol_handle->get_stream()>>>(sol.view(), vehicle_assignment.view());
  RAFT_CHECK_CUDA(sol.sol_handle->get_stream());
}

template <typename i_t, typename f_t, request_t REQUEST>
void reset(solution_t<i_t, f_t, REQUEST>& sol,
           vehicle_assignment_t<i_t, f_t, REQUEST>& vehicle_assignment)
{
  auto problem_buckets      = sol.problem_ptr->get_vehicle_buckets();
  auto vehicle_availability = sol.problem_ptr->fleet_info_h.vehicle_availability;
  vehicle_assignment.vehicle_buckets.resize(sol.problem_ptr->get_fleet_size(),
                                            sol.sol_handle->get_stream());
  std::vector<i_t> bucket_offsets(sol.problem_ptr->get_num_buckets() + 1);
  bucket_offsets[0] = 0;
  auto offset       = 0;
  for (size_t i = 0; i < vehicle_availability.size(); ++i) {
    raft::copy(vehicle_assignment.vehicle_buckets.data() + offset,
               problem_buckets[i].data(),
               vehicle_availability[i],
               sol.sol_handle->get_stream());
    offset += vehicle_availability[i];
    bucket_offsets[i + 1] = offset;
  }

  cuopt::device_copy(
    vehicle_assignment.bucket_offsets, bucket_offsets, sol.sol_handle->get_stream());
  reset_vehicle_availability(sol, vehicle_assignment);
}

template <typename i_t, typename f_t, request_t REQUEST>
bool compute_assignment(solution_t<i_t, f_t, REQUEST>& sol,
                        move_candidates_t<i_t, f_t>& move_candidates,
                        vehicle_assignment_t<i_t, f_t, REQUEST>& vehicle_assignment)
{
  for (i_t i = 0; i < sol.get_n_routes(); ++i) {
    if (!compute_route_cost_differences(sol, vehicle_assignment)) { return false; }
    if (!compute_route_vehicle_assignments(sol, vehicle_assignment)) { return false; }
    if (!update_assignment(sol, move_candidates, vehicle_assignment)) { return false; }
  }
  return true;
}

template <typename i_t, typename f_t, request_t REQUEST>
auto find_best_assignment(solution_t<i_t, f_t, REQUEST>& sol,
                          vehicle_assignment_t<i_t, f_t, REQUEST>& vehicle_assignment)
{
  auto constexpr TPB   = 32;
  auto constexpr shmem = 0;
  bool is_set          = set_shmem_of_kernel(find_best_assignment_kernel<i_t, f_t, REQUEST>, shmem);
  if (!is_set) { return false; }
  find_best_assignment_kernel<i_t, f_t, REQUEST>
    <<<1, TPB, shmem, sol.sol_handle->get_stream()>>>(sol.view(), vehicle_assignment.view());
  RAFT_CHECK_CUDA(sol.sol_handle->get_stream());
  return true;
}

template <typename i_t, typename f_t, request_t REQUEST>
auto update_solution(solution_t<i_t, f_t, REQUEST>& sol,
                     move_candidates_t<i_t, f_t>& move_candidates,
                     vehicle_assignment_t<i_t, f_t, REQUEST>& vehicle_assignment)
{
  reset_vehicle_availability(sol, vehicle_assignment);

  auto constexpr TPB = 128;
  auto shmem         = sol.check_routes_can_insert_and_get_sh_size(0);
  bool is_set        = set_shmem_of_kernel(update_solution_kernel<i_t, f_t, REQUEST>, shmem);
  if (!is_set) { return false; }
  update_solution_kernel<i_t, f_t, REQUEST>
    <<<sol.get_n_routes(), TPB, shmem, sol.sol_handle->get_stream()>>>(
      sol.view(), move_candidates.view(), vehicle_assignment.view());
  RAFT_CHECK_CUDA(sol.sol_handle->get_stream());

  sol.compute_cost();
  sol.sol_handle->sync_stream();
  return true;
}

template <typename i_t, typename f_t, request_t REQUEST>
bool run_vehicle_assignment(solution_t<i_t, f_t, REQUEST>& sol,
                            move_candidates_t<i_t, f_t>& move_candidates,
                            vehicle_assignment_t<i_t, f_t, REQUEST>& vehicle_assignment)
{
  raft::common::nvtx::range fun_scope("vehicle_assignment_heuristic");
  sol.global_runtime_checks(false, false, "vehicle_assignment_checks_start");

  [[maybe_unused]] double cost_before = 0., cost_after = 0.;
  cuopt_func_call(sol.compute_cost());
  cuopt_func_call(cost_before =
                    sol.get_cost(move_candidates.include_objective, move_candidates.weights));

  vehicle_assignment.resize(
    sol.get_n_routes(), sol.problem_ptr->get_num_buckets(), sol.sol_handle->get_stream());

  async_fill(vehicle_assignment.assignments, -1, sol.sol_handle->get_stream());
  async_fill(vehicle_assignment.assignment_costs, 0.0, sol.sol_handle->get_stream());
  async_fill(vehicle_assignment.run_sort, 1, sol.sol_handle->get_stream());
  reset(sol, vehicle_assignment);

  if (!compute_route_costs(sol, move_candidates, vehicle_assignment)) { return false; }
  if (!compute_assignment(sol, move_candidates, vehicle_assignment)) { return false; }
  if (!find_best_assignment(sol, vehicle_assignment)) { return false; }

  // On rare cases the assignment can be worsening when OX found a more optimal split.
  if ((sol.get_cost(move_candidates.include_objective, move_candidates.weights) -
       vehicle_assignment.best_cost.value(sol.sol_handle->get_stream())) < EPSILON) {
    return false;
  }

  if (!update_solution(sol, move_candidates, vehicle_assignment)) { return false; }

  [[maybe_unused]] double best_cost;
  cuopt_func_call(best_cost = vehicle_assignment.best_cost.value(sol.sol_handle->get_stream()));
  cuopt_assert(abs(best_cost - sol.get_cost(move_candidates.include_objective,
                                            move_candidates.weights)) < EPSILON,
               "Mismatch between computed and solution cost");
  cuopt_func_call(sol.check_cost_coherence(move_candidates.weights));
  cuopt_func_call(cost_after =
                    sol.get_cost(move_candidates.include_objective, move_candidates.weights));
  cuopt_assert(cost_before - cost_after >= EPSILON, "Cost should improve!");
  sol.global_runtime_checks(false, false, "vehicle_assignment_checks_end");
  return true;
}

template bool
cuopt::routing::detail::run_vehicle_assignment<int, float, (cuopt::routing::request_t)0>(
  cuopt::routing::detail::solution_t<int, float, (cuopt::routing::request_t)0>&,
  cuopt::routing::detail::move_candidates_t<int, float>&,
  cuopt::routing::detail::vehicle_assignment_t<int, float, (cuopt::routing::request_t)0>&);
template bool
cuopt::routing::detail::run_vehicle_assignment<int, float, (cuopt::routing::request_t)1>(
  cuopt::routing::detail::solution_t<int, float, (cuopt::routing::request_t)1>&,
  cuopt::routing::detail::move_candidates_t<int, float>&,
  cuopt::routing::detail::vehicle_assignment_t<int, float, (cuopt::routing::request_t)1>&);

}  // namespace detail
}  // namespace routing
}  // namespace cuopt
