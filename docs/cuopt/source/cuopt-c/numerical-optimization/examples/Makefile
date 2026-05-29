# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Makefile for cuOpt C API LP/MILP Examples
#
# This Makefile automatically discovers all .c files in the current directory
# and builds corresponding executables.
#
# Usage:
#   make all              - Build all examples (auto-discovered)
#   make <example_name>   - Build specific example (without .c extension)
#   make clean            - Remove all built executables
#   make help             - Show help message
#
# Prerequisites:
#   - cuOpt C library installed or built locally
#   - Set INCLUDE_PATH and LIBCUOPT_LIBRARY_PATH (see below)

# Compiler
CC = gcc

# Set include and library paths
# REQUIRED: Set these paths either:
#   1. As environment variables
#   2. On command line: make INCLUDE_PATH=... LIBCUOPT_LIBRARY_PATH=...
#   3. Uncomment and edit below for your system:
#
# INCLUDE_PATH = /path/to/cuopt/include
# LIBCUOPT_LIBRARY_PATH = /path/to/cuopt/lib
#
# Example for local build:
# INCLUDE_PATH = ../../../cpp/include
# LIBCUOPT_LIBRARY_PATH = ../../../cpp/build
#
# Example for conda install:
# INCLUDE_PATH = $(CONDA_PREFIX)/include
# LIBCUOPT_LIBRARY_PATH = $(CONDA_PREFIX)/lib
#
# Note: The test script automatically finds and passes these paths

# Compiler flags
CFLAGS = -I$(INCLUDE_PATH) -Wall -Wextra
LDFLAGS = -L$(LIBCUOPT_LIBRARY_PATH) -lcuopt -Wl,-rpath,$(LIBCUOPT_LIBRARY_PATH)

# Automatically discover all C source files in current directory
SOURCES = $(wildcard *.c)
# Generate list of executables (remove .c extension)
TARGETS = $(SOURCES:.c=)

.PHONY: all clean help list

# Default target: build all examples
all: check-paths $(TARGETS)
	@echo "==================================="
	@echo "All examples built successfully!"
	@echo "Built $(words $(TARGETS)) executables:"
	@for target in $(TARGETS); do echo "  - $$target"; done
	@echo "==================================="

# Pattern rule: build any executable from corresponding .c file
%: %.c
	@echo "Building $@ from $<..."
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS)
	@echo "✓ Built: $@"

# Clean all built executables
clean:
	@echo "Cleaning up..."
	@if [ -n "$(TARGETS)" ]; then \
		rm -f $(TARGETS); \
		echo "Removed $(words $(TARGETS)) executables"; \
	else \
		echo "No executables to clean"; \
	fi
	@echo "Clean complete"

# Check if required paths are set
check-paths:
	@if [ -z "$(INCLUDE_PATH)" ]; then \
		echo "ERROR: INCLUDE_PATH is not set"; \
		echo ""; \
		echo "Please set it using one of these methods:"; \
		echo "  1. Environment: export INCLUDE_PATH=/path/to/cuopt/include"; \
		echo "  2. Command line: make INCLUDE_PATH=/path/to/cuopt/include LIBCUOPT_LIBRARY_PATH=/path/to/lib"; \
		echo "  3. Edit Makefile and uncomment/set the path variables"; \
		echo ""; \
		exit 1; \
	fi
	@if [ -z "$(LIBCUOPT_LIBRARY_PATH)" ]; then \
		echo "ERROR: LIBCUOPT_LIBRARY_PATH is not set"; \
		echo ""; \
		echo "Please set it using one of these methods:"; \
		echo "  1. Environment: export LIBCUOPT_LIBRARY_PATH=/path/to/cuopt/lib"; \
		echo "  2. Command line: make LIBCUOPT_LIBRARY_PATH=/path/to/lib"; \
		echo "  3. Edit Makefile and uncomment/set the path variables"; \
		echo ""; \
		exit 1; \
	fi

# List all discovered C files and targets
list:
	@echo "Discovered C source files:"
	@for src in $(SOURCES); do echo "  - $$src"; done
	@echo ""
	@echo "Will build these executables:"
	@for target in $(TARGETS); do echo "  - $$target"; done
	@echo ""
	@echo "Total: $(words $(SOURCES)) source file(s)"

# Show help
help:
	@echo "cuOpt C API LP/MILP Examples Makefile"
	@echo "======================================"
	@echo ""
	@echo "This Makefile automatically discovers all .c files and builds them."
	@echo ""
	@echo "Targets:"
	@echo "  all          - Build all examples (default)"
	@echo "  <name>       - Build specific example (e.g., 'make simple_lp_example')"
	@echo "  clean        - Remove all built executables"
	@echo "  list         - Show all discovered source files and targets"
	@echo "  help         - Show this help message"
	@echo ""
	@echo "Current Configuration:"
	@echo "  INCLUDE_PATH: $(INCLUDE_PATH)"
	@echo "  LIBCUOPT_LIBRARY_PATH: $(LIBCUOPT_LIBRARY_PATH)"
	@echo ""
	@echo "Discovered Files:"
	@echo "  C sources: $(words $(SOURCES))"
	@echo "  Targets: $(words $(TARGETS))"
	@echo ""
	@echo "To set paths:"
	@echo "  Method 1 - Environment variables:"
	@echo "    export INCLUDE_PATH=/path/to/include"
	@echo "    export LIBCUOPT_LIBRARY_PATH=/path/to/lib"
	@echo "    make"
	@echo ""
	@echo "  Method 2 - Command line:"
	@echo "    make INCLUDE_PATH=/path/to/include LIBCUOPT_LIBRARY_PATH=/path/to/lib"
	@echo ""
	@echo "  Method 3 - Edit this Makefile:"
	@echo "    Uncomment and set INCLUDE_PATH and LIBCUOPT_LIBRARY_PATH at the top"
