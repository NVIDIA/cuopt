# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Makefile

LIBCUOPT_BUILD_DIR ?= cpp/build
VERBOSE_FLAG ?=
PARALLEL_LEVEL ?=

.PHONY: all ninja-build

all: ninja-build

ninja-build:
	cmake --build $(LIBCUOPT_BUILD_DIR) $(VERBOSE_FLAG) $(if $(PARALLEL_LEVEL),-j$(PARALLEL_LEVEL),)
