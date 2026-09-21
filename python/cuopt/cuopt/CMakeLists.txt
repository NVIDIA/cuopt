# cmake-format: off
# SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# cmake-format: on

add_subdirectory(linear_programming/data_model)
add_subdirectory(linear_programming/io)
add_subdirectory(linear_programming/solver)
add_subdirectory(linear_programming/solver_settings)
# The gRPC extension wraps cython_grpc_client, which is only compiled into cuopt::client
# when SKIP_GRPC_BUILD is off. The target exists either way, so ask the exported flags.
# It needs the VRP arm too: one module covers both, and it calls submit_vrp/result_vrp,
# which live in cython_grpc_client_vrp.cpp and are dropped with the routing engine.
if(CUOPT_HAS_GRPC AND CUOPT_HAS_GRPC_ROUTING)
    add_subdirectory(grpc)
else()
    message(STATUS "cuOpt built without gRPC or without its VRP arm -- skipping the gRPC client extension module")
endif()

# These two link cuopt::routing, which a SKIP_ROUTING_BUILD install does not provide.
# Tested on the found package rather than on SKIP_ROUTING_BUILD: this build only does
# find_package(cuopt) and never sees that option, so the variable would be empty here
# and the guard would always pass.
if(TARGET cuopt::routing)
    add_subdirectory(distance_engine)
    add_subdirectory(routing)
else()
    message(STATUS "cuopt::routing not found -- skipping the routing and distance_engine extension modules")
endif()

# No sources here -- every extension module is created in a subdirectory, and each links
# the component it actually resolves symbols from. Anything added here would link the
# cuopt umbrella and pull in both engines, so add it to the owning subdirectory instead.
set(cython_sources)
set(linked_libraries cuopt::cuopt)

rapids_cython_create_modules(SOURCE_FILES "${cython_sources}" LINKED_LIBRARIES "${linked_libraries}" ASSOCIATED_TARGETS cuopt
                                                                               CXX)
