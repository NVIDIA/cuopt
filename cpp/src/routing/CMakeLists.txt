# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

set(ROUTING_SRC_FILES
  ${CMAKE_CURRENT_SOURCE_DIR}/local_search/compute_insertions.cu
  ${CMAKE_CURRENT_SOURCE_DIR}/ges/squeeze.cu
  ${CMAKE_CURRENT_SOURCE_DIR}/local_search/sliding_window.cu
  ${CMAKE_CURRENT_SOURCE_DIR}/local_search/sliding_tsp.cu
  ${CMAKE_CURRENT_SOURCE_DIR}/ges/lexicographic_search/lexicographic_search.cu
  ${CMAKE_CURRENT_SOURCE_DIR}/assignment.cu
  ${CMAKE_CURRENT_SOURCE_DIR}/data_model_view.cu
  ${CMAKE_CURRENT_SOURCE_DIR}/generator/generator.cu
  ${CMAKE_CURRENT_SOURCE_DIR}/ges_solver.cu
  ${CMAKE_CURRENT_SOURCE_DIR}/adapters/adapted_modifier.cu
  ${CMAKE_CURRENT_SOURCE_DIR}/adapters/adapted_generator.cu
  ${CMAKE_CURRENT_SOURCE_DIR}/crossovers/optimal_eax_cycles.cu
  ${CMAKE_CURRENT_SOURCE_DIR}/distance_engine/waypoint_matrix.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/ges/guided_ejection_search.cu
  ${CMAKE_CURRENT_SOURCE_DIR}/ges/compute_fragment_ejections.cu
  ${CMAKE_CURRENT_SOURCE_DIR}/ges/execute_insertion.cu
  ${CMAKE_CURRENT_SOURCE_DIR}/ges/eject_until_feasible.cu
  ${CMAKE_CURRENT_SOURCE_DIR}/local_search/breaks_insertion.cu
  ${CMAKE_CURRENT_SOURCE_DIR}/local_search/compute_compatible.cu
  ${CMAKE_CURRENT_SOURCE_DIR}/local_search/cycle_finder/cycle_finder.cu
  ${CMAKE_CURRENT_SOURCE_DIR}/local_search/fill_gpu_graph.cu
  ${CMAKE_CURRENT_SOURCE_DIR}/local_search/local_search.cu
  ${CMAKE_CURRENT_SOURCE_DIR}/local_search/perform_moves.cu
  ${CMAKE_CURRENT_SOURCE_DIR}/local_search/random_cross.cu
  ${CMAKE_CURRENT_SOURCE_DIR}/local_search/prize_collection.cu
  ${CMAKE_CURRENT_SOURCE_DIR}/local_search/two_opt.cu
  ${CMAKE_CURRENT_SOURCE_DIR}/local_search/hvrp/vehicle_assignment.cu
  ${CMAKE_CURRENT_SOURCE_DIR}/local_search/vrp/nodes_to_search.cu
  ${CMAKE_CURRENT_SOURCE_DIR}/local_search/vrp/vrp_execute.cu
  ${CMAKE_CURRENT_SOURCE_DIR}/local_search/vrp/vrp_search.cu
  ${CMAKE_CURRENT_SOURCE_DIR}/problem/problem.cu
  ${CMAKE_CURRENT_SOURCE_DIR}/solution/solution.cu
  ${CMAKE_CURRENT_SOURCE_DIR}/util_kernels/compute_backward_forward.cu
  ${CMAKE_CURRENT_SOURCE_DIR}/util_kernels/set_initial_nodes.cu
  ${CMAKE_CURRENT_SOURCE_DIR}/util_kernels/runtime_checks.cu
  ${CMAKE_CURRENT_SOURCE_DIR}/ges/lexicographic_search/brute_force_lexico.cu
  ${CMAKE_CURRENT_SOURCE_DIR}/fleet_info.cu
  ${CMAKE_CURRENT_SOURCE_DIR}/order_info.cu
  ${CMAKE_CURRENT_SOURCE_DIR}/fleet_order_constraints.cu
  ${CMAKE_CURRENT_SOURCE_DIR}/fleet_order_info.cu
  ${CMAKE_CURRENT_SOURCE_DIR}/solve.cu
  ${CMAKE_CURRENT_SOURCE_DIR}/solver.cu
  ${CMAKE_CURRENT_SOURCE_DIR}/solver_settings.cu
  ${CMAKE_CURRENT_SOURCE_DIR}/utilities/check_input.cu
  ${CMAKE_CURRENT_SOURCE_DIR}/utilities/cython.cu)

set(CUOPT_SRC_FILES ${CUOPT_SRC_FILES} ${ROUTING_SRC_FILES} PARENT_SCOPE)
