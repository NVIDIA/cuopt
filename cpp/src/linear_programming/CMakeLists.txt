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

# Core LP files always included
set(LP_CORE_FILES
  ${CMAKE_CURRENT_SOURCE_DIR}/solver_settings.cu
  ${CMAKE_CURRENT_SOURCE_DIR}/optimization_problem.cu
  ${CMAKE_CURRENT_SOURCE_DIR}/utilities/problem_checking.cu
  ${CMAKE_CURRENT_SOURCE_DIR}/solve.cu
  ${CMAKE_CURRENT_SOURCE_DIR}/pdlp.cu
  ${CMAKE_CURRENT_SOURCE_DIR}/pdhg.cu
  ${CMAKE_CURRENT_SOURCE_DIR}/solver_solution.cu
  ${CMAKE_CURRENT_SOURCE_DIR}/saddle_point.cu
  ${CMAKE_CURRENT_SOURCE_DIR}/cusparse_view.cu
  ${CMAKE_CURRENT_SOURCE_DIR}/pdlp_hyper_params.cu
  ${CMAKE_CURRENT_SOURCE_DIR}/pdlp_warm_start_data.cu
  ${CMAKE_CURRENT_SOURCE_DIR}/initial_scaling_strategy/initial_scaling.cu
  ${CMAKE_CURRENT_SOURCE_DIR}/step_size_strategy/adaptive_step_size_strategy.cu
  ${CMAKE_CURRENT_SOURCE_DIR}/restart_strategy/pdlp_restart_strategy.cu
  ${CMAKE_CURRENT_SOURCE_DIR}/restart_strategy/weighted_average_solution.cu
  ${CMAKE_CURRENT_SOURCE_DIR}/restart_strategy/localized_duality_gap_container.cu
  ${CMAKE_CURRENT_SOURCE_DIR}/termination_strategy/termination_strategy.cu
  ${CMAKE_CURRENT_SOURCE_DIR}/termination_strategy/infeasibility_information.cu
  ${CMAKE_CURRENT_SOURCE_DIR}/termination_strategy/convergence_information.cu
)

# C and Python adapter files
set(LP_ADAPTER_FILES
  ${CMAKE_CURRENT_SOURCE_DIR}/utilities/cython_solve.cu
  ${CMAKE_CURRENT_SOURCE_DIR}/cuopt_c.cpp
)

# Choose which files to include based on build mode
if(SKIP_C_PYTHON_ADAPTERS)
  set(LP_SRC_FILES ${LP_CORE_FILES})
else()
  set(LP_SRC_FILES ${LP_CORE_FILES} ${LP_ADAPTER_FILES})
endif()

set(CUOPT_SRC_FILES ${CUOPT_SRC_FILES} ${LP_SRC_FILES} PARENT_SCOPE)
