# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Makefile for cuOpt C API MIP Examples
#
# Usage:
#   make all              - Build all examples (auto-discovered)
#   make <example_name>   - Build specific example (without .c extension)
#   make clean            - Remove all built executables
#
# Required variables (pass on command line):
#   make INCLUDE_PATH=/path/to/cuopt/include LIBCUOPT_LIBRARY_PATH=/path/to/cuopt/lib

# Compiler
CC = gcc

# Compiler flags
CFLAGS = -I$(INCLUDE_PATH) -Wall -Wextra
LDFLAGS = -L$(LIBCUOPT_LIBRARY_PATH) -lcuopt -Wl,-rpath,$(LIBCUOPT_LIBRARY_PATH)

# Automatically discover all C source files in current directory
SOURCES = $(wildcard *.c)
TARGETS = $(SOURCES:.c=)

.PHONY: all clean check-paths

all: check-paths $(TARGETS)

%: %.c check-paths
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS)

clean:
	rm -f $(TARGETS)

check-paths:
	@if [ -z "$(INCLUDE_PATH)" ]; then \
		echo "ERROR: INCLUDE_PATH is not set"; \
		echo "Usage: make INCLUDE_PATH=/path/to/cuopt/include LIBCUOPT_LIBRARY_PATH=/path/to/cuopt/lib"; \
		exit 1; \
	fi
	@if [ -z "$(LIBCUOPT_LIBRARY_PATH)" ]; then \
		echo "ERROR: LIBCUOPT_LIBRARY_PATH is not set"; \
		echo "Usage: make INCLUDE_PATH=/path/to/cuopt/include LIBCUOPT_LIBRARY_PATH=/path/to/cuopt/lib"; \
		exit 1; \
	fi
