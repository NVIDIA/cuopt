# Makefile for cuOpt C API LP/MILP Examples
#
# Usage:
#   make all              - Build all examples
#   make simple_lp        - Build simple LP example
#   make mps_file         - Build MPS file example
#   make simple_milp      - Build simple MILP example
#   make milp_mps         - Build MILP MPS file example
#   make clean            - Remove built executables
#
# Prerequisites:
#   - cuOpt C library installed or built locally
#   - Set INCLUDE_PATH and LIBCUOPT_LIBRARY_PATH appropriately

# Compiler
CC = gcc

# Find include and library paths (or set them manually)
# If you have built locally, set these to your build directories:
# INCLUDE_PATH = ../../../cpp/include
# LIBCUOPT_LIBRARY_PATH = ../../../cpp/build

# Otherwise, try to find them
INCLUDE_PATH ?= $(shell find / -name "cuopt_c.h" -path "*/linear_programming/*" -printf "%h\n" 2>/dev/null | sed 's/\/linear_programming//' | head -1)
LIBCUOPT_LIBRARY_PATH ?= $(shell find / -name "libcuopt.so" 2>/dev/null | head -1 | xargs dirname)

# Compiler flags
CFLAGS = -I$(INCLUDE_PATH) -Wall -Wextra
LDFLAGS = -L$(LIBCUOPT_LIBRARY_PATH) -lcuopt -Wl,-rpath,$(LIBCUOPT_LIBRARY_PATH)

# Targets
TARGETS = simple_lp_example mps_file_example simple_milp_example milp_mps_example

.PHONY: all clean help

all: $(TARGETS)

simple_lp_example: simple_lp_example.c
	@echo "Building simple LP example..."
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS)
	@echo "Built: $@"

mps_file_example: mps_file_example.c
	@echo "Building MPS file example..."
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS)
	@echo "Built: $@"

simple_milp_example: simple_milp_example.c
	@echo "Building simple MILP example..."
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS)
	@echo "Built: $@"

milp_mps_example: milp_mps_example.c
	@echo "Building MILP MPS file example..."
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS)
	@echo "Built: $@"

clean:
	@echo "Cleaning up..."
	rm -f $(TARGETS)
	@echo "Clean complete"

help:
	@echo "cuOpt C API LP/MILP Examples Makefile"
	@echo ""
	@echo "Targets:"
	@echo "  all          - Build all examples (default)"
	@echo "  simple_lp    - Build simple LP example"
	@echo "  mps_file     - Build MPS file example"
	@echo "  simple_milp  - Build simple MILP example"
	@echo "  milp_mps     - Build MILP MPS file example"
	@echo "  clean        - Remove built executables"
	@echo "  help         - Show this help message"
	@echo ""
	@echo "Current Paths:"
	@echo "  INCLUDE_PATH: $(INCLUDE_PATH)"
	@echo "  LIBCUOPT_LIBRARY_PATH: $(LIBCUOPT_LIBRARY_PATH)"
	@echo ""
	@echo "To override paths, set them as environment variables:"
	@echo "  make INCLUDE_PATH=/path/to/include LIBCUOPT_LIBRARY_PATH=/path/to/lib"

