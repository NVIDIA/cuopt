#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BRANCHES_FILE="$SCRIPT_DIR/sonar-branches.txt"
WORK_DIR="/tmp/sonar-analysis-$(date +%Y%m%d-%H%M%S)"

# Conda environment file to use for building
# Adjust this path based on your CUDA version and architecture
# Available: all_cuda-129_arch-{x86_64,aarch64}.yaml, all_cuda-131_arch-{x86_64,aarch64}.yaml
ARCH=$(uname -m)
CONDA_ENV_FILE="conda/environments/all_cuda-131_arch-${ARCH}.yaml"

# SonarQube Configuration
# The token should be set via environment variable SONAR_TOKEN for security
if [ -z "$SONAR_TOKEN" ]; then
  echo "ERROR: SONAR_TOKEN environment variable is not set"
  echo "Please set it with: export SONAR_TOKEN=your_sonarqube_token"
  HAD_FAILURES=1
  exit 1
fi

# Repository URL
REPO_URL="git@github.com:NVIDIA/cuopt.git"

echo "Repository URL: $REPO_URL"
echo "Working directory: $WORK_DIR"

# Create working directory and logs directory
mkdir -p "$WORK_DIR"
LOG_DIR="$WORK_DIR/logs"
mkdir -p "$LOG_DIR"

# Persistent log directory for failures
PERSISTENT_LOG_DIR="/var/log/sonarqube/runs"
mkdir -p "$PERSISTENT_LOG_DIR" 2>/dev/null || PERSISTENT_LOG_DIR="$HOME/.sonarqube/logs/runs"
mkdir -p "$PERSISTENT_LOG_DIR"

# Track if we had failures (for cleanup decision)
HAD_FAILURES=0

# Cleanup function
cleanup() {
  echo ""
  if [ "$HAD_FAILURES" -eq 0 ] && [ ${#failed_branches[@]} -eq 0 ]; then
    echo "All branches succeeded - cleaning up working directory: $WORK_DIR"
    rm -rf "$WORK_DIR"
  else
    # Preserve logs on failure
    RUN_ID=$(date +%Y%m%d-%H%M%S)
    SAVED_LOG_DIR="$PERSISTENT_LOG_DIR/failed-run-$RUN_ID"
    echo "Failures detected - preserving logs to: $SAVED_LOG_DIR"
    mkdir -p "$SAVED_LOG_DIR"
    cp -r "$LOG_DIR"/* "$SAVED_LOG_DIR/" 2>/dev/null || true
    echo "Logs saved. Check: $SAVED_LOG_DIR"
    echo "Cleaning up working directory: $WORK_DIR"
    rm -rf "$WORK_DIR"
  fi
}

# Register cleanup on exit
trap cleanup EXIT

# Check if branches file exists
if [ ! -f "$BRANCHES_FILE" ]; then
  echo "ERROR: Branches file not found: $BRANCHES_FILE"
  HAD_FAILURES=1
  exit 1
fi

# Read and validate branches
branches=()
while IFS= read -r branch || [ -n "$branch" ]; do
  # Skip comments and empty lines
  [[ "$branch" =~ ^#.*$ ]] && continue
  [[ -z "${branch// }" ]] && continue

  # Trim whitespace and add to array
  branch=$(echo "$branch" | xargs)
  branches+=("$branch")
done < "$BRANCHES_FILE"

# Fail if no branches found
if [ ${#branches[@]} -eq 0 ]; then
  echo "ERROR: No branches configured in $BRANCHES_FILE"
  echo "Please add at least one branch to the file."
  HAD_FAILURES=1
  exit 1
fi

echo "Found ${#branches[@]} branch(es) to process: ${branches[*]}"
echo "Host: $(hostname)"
echo "Start time: $(date)"

# Track success/failure
successful_branches=()
failed_branches=()

# Process each branch
for branch in "${branches[@]}"; do
  echo "=========================================="
  echo "Processing branch: $branch"
  echo "=========================================="

  # Create a safe directory name from branch name
  safe_branch_name="${branch//\//_}"
  clone_dir="$WORK_DIR/$safe_branch_name"

  # Clone the specific branch
  echo "Cloning branch: $branch into $clone_dir"
  git clone --single-branch --branch "$branch" --depth 1 "$REPO_URL" "$clone_dir" 2>&1 | tee "$LOG_DIR/clone_${safe_branch_name}.log"
  if [ "${PIPESTATUS[0]}" -ne 0 ]; then
    echo "ERROR: Failed to clone branch: $branch"
    failed_branches+=("$branch (clone failed)")
    continue
  fi

  # Change to cloned directory
  if ! cd "$clone_dir"; then
    echo "ERROR: Failed to change directory to: $clone_dir"
    failed_branches+=("$branch (cd failed)")
    continue
  fi

  # Setup conda environment, build, and analyze
  echo "Setting up conda environment for: $branch"

  # Create a unique conda environment name for this branch
  conda_env_name="cuopt_sonar_${safe_branch_name}"

  # Create conda environment
  mamba env create -n "$conda_env_name" -f "$CONDA_ENV_FILE" 2>&1 | tee "$LOG_DIR/conda_create_${safe_branch_name}.log"
  if [ "${PIPESTATUS[0]}" -ne 0 ]; then
    echo "ERROR: Conda environment creation failed for branch: $branch. Check logs at $LOG_DIR/conda_create_${safe_branch_name}.log"
    failed_branches+=("$branch (conda env creation failed)")
    cd "$WORK_DIR" || echo "WARNING: Failed to cd to $WORK_DIR"
    rm -rf "$clone_dir"
    continue
  fi

  # Activate conda environment and run build + analysis in a subshell
  echo "Building and analyzing branch: $branch in conda environment: $conda_env_name"

  if ! bash -c "
    set -e
    source \$(conda info --base)/etc/profile.d/conda.sh
    conda activate $conda_env_name

    echo 'Conda environment activated: $conda_env_name'
    echo 'Python version:' \$(python --version)

    # Build the project
    echo 'Building project...'
    ./build.sh 2>&1 | tee '$LOG_DIR/build_${safe_branch_name}.log'
    if [ \${PIPESTATUS[0]} -ne 0 ]; then
      echo 'Build failed'
      exit 1
    fi

    # Run SonarQube analysis
    # Note: SONAR_TOKEN is read from environment automatically by sonar-scanner
    echo 'Running SonarQube analysis...'
    sonar-scanner \
      -Dsonar.branch.name='$branch' \
      2>&1 | tee '$LOG_DIR/sonar_${safe_branch_name}.log'
    if [ \${PIPESTATUS[0]} -ne 0 ]; then
      echo 'SonarQube analysis failed'
      exit 1
    fi

    echo 'Build and analysis completed successfully'
  "; then
    echo "ERROR: Build or analysis failed for branch: $branch"
    if grep -q "Build failed" "$LOG_DIR/build_${safe_branch_name}.log" 2>/dev/null; then
      failed_branches+=("$branch (build failed)")
    else
      failed_branches+=("$branch (sonar analysis failed)")
    fi

    # Clean up conda environment
    conda env remove -n "$conda_env_name" -y 2>/dev/null || true
    cd "$WORK_DIR" || echo "WARNING: Failed to cd to $WORK_DIR"
    rm -rf "$clone_dir"
    continue
  fi

  # Clean up conda environment after successful analysis
  echo "Cleaning up conda environment: $conda_env_name"
  conda env remove -n "$conda_env_name" -y 2>/dev/null || true

  successful_branches+=("$branch")
  echo "✓ Successfully completed analysis for: $branch"
  echo "Progress: ${#successful_branches[@]} succeeded, ${#failed_branches[@]} failed out of ${#branches[@]} total"

  # Clean up clone directory after successful analysis
  echo "Cleaning up clone directory for: $branch"
  cd "$WORK_DIR" || echo "WARNING: Failed to cd to $WORK_DIR"
  rm -rf "$clone_dir"
done

# Final summary
echo "=========================================="
echo "SonarQube Analysis Complete"
echo "=========================================="
echo "Total branches: ${#branches[@]}"
echo "Successful: ${#successful_branches[@]}"
echo "Failed: ${#failed_branches[@]}"
echo ""

if [ ${#successful_branches[@]} -gt 0 ]; then
  echo "✓ Successful branches:"
  for branch in "${successful_branches[@]}"; do
    echo "  - $branch"
  done
  echo ""
fi

if [ ${#failed_branches[@]} -gt 0 ]; then
  echo "✗ Failed branches:"
  for branch in "${failed_branches[@]}"; do
    echo "  - $branch"
  done
  echo ""
fi

echo "End time: $(date)"
echo "=========================================="

# Exit with error if any branches failed
if [ ${#failed_branches[@]} -gt 0 ]; then
  echo "ERROR: ${#failed_branches[@]} branch(es) failed analysis"
  HAD_FAILURES=1
  exit 1
fi

echo "All branches processed successfully!"
exit 0
