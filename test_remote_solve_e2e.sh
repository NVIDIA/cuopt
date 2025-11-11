#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

# End-to-end test of remote solve: Start server, run client, stop server

set -e

# Check if in correct conda environment
if [ "$CONDA_DEFAULT_ENV" != "cuopt_dev_2510_12" ]; then
    echo "Error: Please run this script from the cuopt_dev_2510_12 conda environment"
    echo "Run: conda activate cuopt_dev_2510_12"
    exit 1
fi

PORT=9999
SERVER_PID=""
SERVER_LOG="server_e2e.log"

# Cleanup function
cleanup() {
    echo ""
    echo "=== Cleanup ==="
    if [ -n "$SERVER_PID" ]; then
        echo "Stopping server (PID: $SERVER_PID)..."
        kill $SERVER_PID 2>/dev/null || true
        wait $SERVER_PID 2>/dev/null || true
        echo "Server stopped"
    fi
}

# Set trap for cleanup on exit
trap cleanup EXIT INT TERM

echo "=========================================================="
echo "cuOpt Remote Solve - End-to-End Test"
echo "=========================================================="
echo ""

# Start server in background
echo "=== Starting Server ==="
echo "Command: cpp/build/cuopt_remote_server $PORT"
cpp/build/cuopt_remote_server $PORT > $SERVER_LOG 2>&1 &
SERVER_PID=$!

echo "Server started (PID: $SERVER_PID) on port $PORT"
echo "Server log: $SERVER_LOG"
echo ""

# Wait for server to be ready
echo "Waiting for server to be ready..."
sleep 3

# Check if server is still running
if ! kill -0 $SERVER_PID 2>/dev/null; then
    echo "❌ Server failed to start!"
    echo "Server log contents:"
    cat $SERVER_LOG
    exit 1
fi

echo "✅ Server is running"
echo ""

# Set environment variables for client
export CUOPT_REMOTE_HOST="127.0.0.1"
export CUOPT_REMOTE_PORT="$PORT"

echo "=== Running Client ==="
echo "Environment variables:"
echo "  CUOPT_REMOTE_HOST=$CUOPT_REMOTE_HOST"
echo "  CUOPT_REMOTE_PORT=$CUOPT_REMOTE_PORT"
echo ""

# Run Python client with timeout
echo "Executing: timeout 60 python test_remote_client.py"
echo ""
echo "========== CLIENT OUTPUT =========="
timeout 60 python test_remote_client.py 2>&1 | grep -v "CuPy" || echo "Client timed out or failed"
CLIENT_EXIT=$?
echo "==================================="
echo ""

# Give server time to finish processing
sleep 2

# Show server log
echo "========== SERVER LOG =========="
cat $SERVER_LOG | grep -v "^$"
echo "================================"
echo ""

if [ $CLIENT_EXIT -eq 0 ]; then
    echo "=========================================================="
    echo "✅ End-to-End Test Completed Successfully!"
    echo "=========================================================="
    exit 0
elif [ $CLIENT_EXIT -eq 124 ]; then
    echo "=========================================================="
    echo "❌ Client timed out after 60 seconds"
    echo "=========================================================="
    exit 1
else
    echo "=========================================================="
    echo "❌ Client failed with exit code: $CLIENT_EXIT"
    echo "=========================================================="
    exit 1
fi
