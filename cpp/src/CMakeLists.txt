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

set(UTIL_SRC_FILES ${CMAKE_CURRENT_SOURCE_DIR}/utilities/seed_generator.cu
                   ${CMAKE_CURRENT_SOURCE_DIR}/utilities/logger_helper.cpp)

add_subdirectory(linear_programming)
add_subdirectory(math_optimization)
add_subdirectory(mip)
# add_subdirectory(routing)
add_subdirectory(dual_simplex)

set(CUOPT_SRC_FILES ${CUOPT_SRC_FILES} ${UTIL_SRC_FILES} PARENT_SCOPE)
