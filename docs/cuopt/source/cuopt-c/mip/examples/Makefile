# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#   make INCLUDE_PATH=/path/to/libcuopt_mathopt/include LIBCUOPT_LIBRARY_PATH=/path/to/libcuopt_mathopt/lib \
#        CLIENT_INCLUDE_PATH=/path/to/libcuopt_client/include CLIENT_LIBRARY_PATH=/path/to/libcuopt_client/lib

# Compiler
CC = gcc

# Compiler flags
CFLAGS = -I$(INCLUDE_PATH) -I$(CLIENT_INCLUDE_PATH) -Wall -Wextra
LDFLAGS = -L$(LIBCUOPT_LIBRARY_PATH) -L$(CLIENT_LIBRARY_PATH) -Wl,--no-as-needed -lcuopt_mathopt -lcuopt_client \
          -Wl,-rpath,$(LIBCUOPT_LIBRARY_PATH) -Wl,-rpath,$(CLIENT_LIBRARY_PATH)

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
	@if [ -z "$(INCLUDE_PATH)" ] || [ -z "$(LIBCUOPT_LIBRARY_PATH)" ] || \
	    [ -z "$(CLIENT_INCLUDE_PATH)" ] || [ -z "$(CLIENT_LIBRARY_PATH)" ]; then \
		echo "ERROR: one or more required paths are not set"; \
		echo "Usage: make INCLUDE_PATH=/path/to/libcuopt_mathopt/include LIBCUOPT_LIBRARY_PATH=/path/to/libcuopt_mathopt/lib \\"; \
		echo "            CLIENT_INCLUDE_PATH=/path/to/libcuopt_client/include CLIENT_LIBRARY_PATH=/path/to/libcuopt_client/lib"; \
		exit 1; \
	fi
