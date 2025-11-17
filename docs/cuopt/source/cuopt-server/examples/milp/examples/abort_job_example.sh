#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# TEST_SKIP: This is a template file requiring manual configuration
#
# Abort Running MILP Job CLI Example
#
# This example demonstrates how to abort a running or queued job
# using the cuopt_sh CLI tool.
#
# Usage: Replace <UUID> with the actual UUID from a running job
#
# Requirements:
#   - cuOpt server running on localhost:5000
#   - cuopt_sh CLI tool installed
#

# Set server connection details
export ip="localhost"
export port=5000

# Replace this with the actual UUID from a job you want to abort
UUID="<UUID_THAT_WE_GOT>"

echo "Aborting job with UUID: $UUID"

# Abort the job
# -d: Delete/abort
# -r: If running
# -q: If queued
cuopt_sh -d -r -q $UUID -i $ip -p $port

echo "Job abort request sent"
echo ""
echo "Note: This is a template. To use:"
echo "1. Start a MILP job and note its UUID from the response"
echo "2. Replace <UUID_THAT_WE_GOT> in this script with that UUID"
echo "3. Run this script while the job is running or queued"
