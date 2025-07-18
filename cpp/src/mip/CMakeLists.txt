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

set(MIP_SRC_FILES ${CMAKE_CURRENT_SOURCE_DIR}/problem/problem.cu)

list(PREPEND
  MIP_SRC_FILES
  ${CMAKE_CURRENT_SOURCE_DIR}/problem/write_mps.cu
  ${CMAKE_CURRENT_SOURCE_DIR}/solve.cu
  ${CMAKE_CURRENT_SOURCE_DIR}/solver.cu
  ${CMAKE_CURRENT_SOURCE_DIR}/solver_settings.cu
  ${CMAKE_CURRENT_SOURCE_DIR}/solver_solution.cu
  ${CMAKE_CURRENT_SOURCE_DIR}/diversity/assignment_hash_map.cu
  ${CMAKE_CURRENT_SOURCE_DIR}/diversity/diversity_manager.cu
  ${CMAKE_CURRENT_SOURCE_DIR}/diversity/multi_armed_bandit.cu
  ${CMAKE_CURRENT_SOURCE_DIR}/diversity/population.cu
  ${CMAKE_CURRENT_SOURCE_DIR}/relaxed_lp/relaxed_lp.cu
  ${CMAKE_CURRENT_SOURCE_DIR}/local_search/local_search.cu
  ${CMAKE_CURRENT_SOURCE_DIR}/local_search/rounding/bounds_repair.cu
  ${CMAKE_CURRENT_SOURCE_DIR}/local_search/rounding/constraint_prop.cu
  ${CMAKE_CURRENT_SOURCE_DIR}/local_search/rounding/lb_bounds_repair.cu
  ${CMAKE_CURRENT_SOURCE_DIR}/local_search/rounding/lb_constraint_prop.cu
  ${CMAKE_CURRENT_SOURCE_DIR}/local_search/rounding/simple_rounding.cu
  ${CMAKE_CURRENT_SOURCE_DIR}/local_search/feasibility_pump/feasibility_pump.cu
  ${CMAKE_CURRENT_SOURCE_DIR}/local_search/line_segment_search/line_segment_search.cu
  ${CMAKE_CURRENT_SOURCE_DIR}/presolve/bounds_presolve.cu
  ${CMAKE_CURRENT_SOURCE_DIR}/presolve/bounds_update_data.cu
  ${CMAKE_CURRENT_SOURCE_DIR}/presolve/conditional_bound_strengthening.cu
  ${CMAKE_CURRENT_SOURCE_DIR}/presolve/lb_probing_cache.cu
  ${CMAKE_CURRENT_SOURCE_DIR}/presolve/load_balanced_bounds_presolve.cu
  ${CMAKE_CURRENT_SOURCE_DIR}/presolve/multi_probe.cu
  ${CMAKE_CURRENT_SOURCE_DIR}/presolve/probing_cache.cu
  ${CMAKE_CURRENT_SOURCE_DIR}/presolve/trivial_presolve.cu
  ${CMAKE_CURRENT_SOURCE_DIR}/problem/load_balanced_problem.cu
  ${CMAKE_CURRENT_SOURCE_DIR}/feasibility_jump/feasibility_jump.cu
  ${CMAKE_CURRENT_SOURCE_DIR}/feasibility_jump/feasibility_jump_kernels.cu
  ${CMAKE_CURRENT_SOURCE_DIR}/solution/solution.cu)

set(CUOPT_SRC_FILES ${CUOPT_SRC_FILES}
  ${MIP_SRC_FILES} PARENT_SCOPE)
