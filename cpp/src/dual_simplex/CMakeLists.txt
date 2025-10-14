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

set(DUAL_SIMPLEX_SRC_FILES
  ${CMAKE_CURRENT_SOURCE_DIR}/cusparse_view.cu
  ${CMAKE_CURRENT_SOURCE_DIR}/barrier.cu
  ${CMAKE_CURRENT_SOURCE_DIR}/basis_solves.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/basis_updates.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/bound_flipping_ratio_test.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/branch_and_bound.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/crossover.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/folding.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/initial_basis.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/mip_node.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/phase1.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/phase2.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/presolve.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/primal.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/pseudo_costs.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/right_looking_lu.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/scaling.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/singletons.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/solve.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/sparse_matrix.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/device_sparse_matrix.cu
  ${CMAKE_CURRENT_SOURCE_DIR}/sparse_vector.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/tic_toc.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/triangle_solve.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/vector_math.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/pinned_host_allocator.cu
  )

# Uncomment to enable debug info
#set_source_files_properties(${DUAL_SIMPLEX_SRC_FILES} DIRECTORY ${CMAKE_SOURCE_DIR} PROPERTIES COMPILE_OPTIONS "-g1")

set(CUOPT_SRC_FILES ${CUOPT_SRC_FILES}
  ${DUAL_SIMPLEX_SRC_FILES} PARENT_SCOPE)
