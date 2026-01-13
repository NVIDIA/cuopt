#!/bin/bash
# Build the C API test program
#
# Usage:
#   ./build_c_test.sh
#
# Prerequisites:
#   - Activate conda environment with cuopt installed
#   - CONDA_PREFIX must be set

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

if [ -z "$CONDA_PREFIX" ]; then
    echo "Error: CONDA_PREFIX not set. Activate a conda environment first."
    exit 1
fi

echo "Building test_c_api..."
gcc -I "$CONDA_PREFIX/include" \
    -L "$CONDA_PREFIX/lib" \
    -Wl,-rpath,"$CONDA_PREFIX/lib" \
    -o "$SCRIPT_DIR/test_c_api" \
    "$SCRIPT_DIR/test_c_api.c" \
    -lcuopt

echo "Built: $SCRIPT_DIR/test_c_api"
echo ""
echo "Usage:"
echo "  # Local solve:"
echo "  $SCRIPT_DIR/test_c_api /path/to/problem.mps"
echo ""
echo "  # Remote solve:"
echo "  CUOPT_REMOTE_HOST=localhost CUOPT_REMOTE_PORT=9090 $SCRIPT_DIR/test_c_api /path/to/problem.mps"
