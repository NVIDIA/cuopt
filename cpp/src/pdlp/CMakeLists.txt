# cmake-format: off
# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# cmake-format: on

# Core LP files always included
set(LP_CORE_FILES
  ${CMAKE_CURRENT_SOURCE_DIR}/solver_settings.cu
  ${CMAKE_CURRENT_SOURCE_DIR}/optimization_problem.cu
  ${CMAKE_CURRENT_SOURCE_DIR}/gpu_optimization_problem.cu
  ${CMAKE_CURRENT_SOURCE_DIR}/cpu_optimization_problem.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/backend_selection.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/utilities/problem_checking.cu
  ${CMAKE_CURRENT_SOURCE_DIR}/solve.cu
  ${CMAKE_CURRENT_SOURCE_DIR}/solve_remote.cu
  ${CMAKE_CURRENT_SOURCE_DIR}/pdlp.cu
  ${CMAKE_CURRENT_SOURCE_DIR}/pdhg.cu
  ${CMAKE_CURRENT_SOURCE_DIR}/solver_solution.cu
  ${CMAKE_CURRENT_SOURCE_DIR}/solution_conversion.cu
  ${CMAKE_CURRENT_SOURCE_DIR}/saddle_point.cu
  ${CMAKE_CURRENT_SOURCE_DIR}/cusparse_view.cu
  ${CMAKE_CURRENT_SOURCE_DIR}/pdlp_warm_start_data.cu
  ${CMAKE_CURRENT_SOURCE_DIR}/cpu_pdlp_warm_start_data.cu
  ${CMAKE_CURRENT_SOURCE_DIR}/initial_scaling_strategy/initial_scaling.cu
  ${CMAKE_CURRENT_SOURCE_DIR}/step_size_strategy/adaptive_step_size_strategy.cu
  ${CMAKE_CURRENT_SOURCE_DIR}/restart_strategy/pdlp_restart_strategy.cu
  ${CMAKE_CURRENT_SOURCE_DIR}/restart_strategy/weighted_average_solution.cu
  ${CMAKE_CURRENT_SOURCE_DIR}/restart_strategy/localized_duality_gap_container.cu
  ${CMAKE_CURRENT_SOURCE_DIR}/termination_strategy/termination_strategy.cu
  ${CMAKE_CURRENT_SOURCE_DIR}/termination_strategy/infeasibility_information.cu
  ${CMAKE_CURRENT_SOURCE_DIR}/termination_strategy/convergence_information.cu
  ${CMAKE_CURRENT_SOURCE_DIR}/optimal_batch_size_handler/optimal_batch_size_handler.cu
  ${CMAKE_CURRENT_SOURCE_DIR}/utilities/ping_pong_graph.cu
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
