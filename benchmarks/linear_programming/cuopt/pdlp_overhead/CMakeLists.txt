# cmake-format: off
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# cmake-format: on

find_package(Threads REQUIRED)

add_executable(cuopt_pdlp_overhead driver.cpp)
set_target_properties(cuopt_pdlp_overhead PROPERTIES
    CXX_SCAN_FOR_MODULES OFF
    ENABLE_EXPORTS ON
)
target_compile_features(cuopt_pdlp_overhead PRIVATE cxx_std_20)
target_compile_definitions(cuopt_pdlp_overhead PRIVATE
    CCCL_DISABLE_PDL
    THRUST_DISABLE_ABI_NAMESPACE
    THRUST_IGNORE_ABI_NAMESPACE_ERROR
)
target_compile_options(cuopt_pdlp_overhead PRIVATE
    "$<$<COMPILE_LANGUAGE:CXX>:${CUOPT_CXX_FLAGS}>"
)
target_link_libraries(cuopt_pdlp_overhead PRIVATE
    cuopt_mathopt
    cuopt_client
    rmm::rmm
    CUDA::cudart
    CUDA::cuda_driver
    Threads::Threads
    ${CMAKE_DL_LIBS}
    $<$<PLATFORM_ID:Linux>:m>
)
# Export the executable's RMM resource symbols and allow build-tree library selection.
# CUDA::cuda_driver supplies the link dependency; no stub directory is added to RPATH.
target_link_options(cuopt_pdlp_overhead PRIVATE
    "$<$<PLATFORM_ID:Linux>:-Wl,--exclude-libs,ALL>"
    "$<$<PLATFORM_ID:Linux>:-Wl,--enable-new-dtags>"
)

add_executable(cuopt_pdlp_overhead_checks checks_test.cpp)
set_target_properties(cuopt_pdlp_overhead_checks PROPERTIES CXX_SCAN_FOR_MODULES OFF)
target_compile_features(cuopt_pdlp_overhead_checks PRIVATE cxx_std_20)
# These CPU fixtures use assert(), including in Release benchmark builds.
target_compile_options(cuopt_pdlp_overhead_checks PRIVATE
    "$<$<COMPILE_LANG_AND_ID:CXX,GNU,Clang,AppleClang>:-UNDEBUG>"
)
if(BUILD_TESTS)
    add_test(NAME cuopt_pdlp_overhead_checks COMMAND cuopt_pdlp_overhead_checks)
endif()
