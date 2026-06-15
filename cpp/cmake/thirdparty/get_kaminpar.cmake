# cmake-format: off
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# cmake-format: on

# Multi-threaded graph partitioner for distributed PDLP.
# Uses rapids_cpm_find so a system / conda / .deb install of KaMinPar (which ships a
# CMake config package exporting KaMinPar::KaMinPar) is used when available, and
# otherwise the pinned source is cloned and built via CPM. KaMinPar depends on TBB,
# which cuOpt already requires (see find_package(TBB) for papilo).
function(find_and_configure_kaminpar)
    set(oneValueArgs VERSION PINNED_TAG)
    cmake_parse_arguments(PKG "" "${oneValueArgs}" "" ${ARGN})

    # NOTE: KaMinPar is intentionally NOT added to cuopt's BUILD/INSTALL export sets.
    # It is a from-source static dependency that is fully embedded into libcuopt.so and
    # never installed (INSTALL_KAMINPAR OFF below). Registering it in cuopt-exports would
    # both break export generation ("target KaMinPar is not in any export set") and emit a
    # bogus find_dependency(KaMinPar) into the installed cuopt config. It is linked by file
    # in cpp/CMakeLists.txt (mirroring PSLP) so it stays out of cuopt's export interface.
    rapids_cpm_find(KaMinPar ${PKG_VERSION}
        GLOBAL_TARGETS KaMinPar::KaMinPar
        CPM_ARGS
        GIT_REPOSITORY https://github.com/KaHIP/KaMinPar.git
        GIT_TAG ${PKG_PINNED_TAG}
        EXCLUDE_FROM_ALL
        OPTIONS
            "KAMINPAR_BUILD_APPS OFF"
            "KAMINPAR_BUILD_TOOLS OFF"
            "KAMINPAR_BUILD_TESTS OFF"
            "KAMINPAR_BUILD_BENCHMARKS OFF"
            "KAMINPAR_BUILD_EXAMPLES OFF"
            "KAMINPAR_BUILD_DISTRIBUTED OFF"
            # Timers use global state and force single-threaded use of the library
            # interface; disable so cuOpt can call the partitioner freely.
            "KAMINPAR_ENABLE_TIMERS OFF"
            # Avoid an extra hard dependency on Google Sparsehash.
            "KAMINPAR_BUILD_WITH_SPARSEHASH OFF"
            # cuOpt's TBB is discovered via a legacy find that only exposes TBB::tbb
            # (no TBB::tbbmalloc target); disable KaMinPar's optional tbbmalloc use.
            "KAMINPAR_ENABLE_TBB_MALLOC OFF"
            # Large LP constraint graphs can exceed 2^31 directed edges.
            "KAMINPAR_64BIT_EDGE_IDS ON"
            "INSTALL_KAMINPAR OFF"
    )

    if(KaMinPar_ADDED)
        message(VERBOSE "CUOPT: Using KaMinPar located in ${KaMinPar_SOURCE_DIR}")
        # KaMinPar's public header pulls in <tbb/global_control.h>. On older TBB releases
        # that header is gated behind TBB_PREVIEW_GLOBAL_CONTROL (KaMinPar upstream assumes a
        # newer oneTBB and never defines it). Define it on KaMinParCommon PUBLIC so it
        # propagates to all KaMinPar translation units (KaMinPar links KaMinParCommon PUBLIC).
        # Harmless on newer oneTBB where global_control is no longer a preview feature.
        if(TARGET KaMinParCommon)
            target_compile_definitions(KaMinParCommon PUBLIC TBB_PREVIEW_GLOBAL_CONTROL)
        endif()
    else()
        message(VERBOSE "CUOPT: Using KaMinPar located in ${KaMinPar_DIR}")
    endif()
endfunction()

find_and_configure_kaminpar(VERSION 3.7.3 PINNED_TAG v3.7.3)
