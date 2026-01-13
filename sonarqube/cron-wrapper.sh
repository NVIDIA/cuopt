#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Cron wrapper script for SonarQube analysis
# This script clones/pulls the cuopt repository and runs SonarQube analysis

set -e

# Configuration
REPO_URL="git@github.com:rgsl888prabhu/cuopt_public.git"
REPO_BRANCH="enable_sonar_cube_for_cuopt"
WORK_DIR="/tmp/cuopt-sonar-cron"
LOG_DIR="/var/log/sonarqube"

# Create log directory if it doesn't exist
mkdir -p "$LOG_DIR"

# Log file with timestamp
LOG_FILE="$LOG_DIR/sonar-cron-$(date +%Y%m%d-%H%M%S).log"

# Function to log messages
log() {
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG_FILE"
}

log "========================================"
log "SonarQube Cron Job Started"
log "========================================"
log "Repository: $REPO_URL"
log "Branch: $REPO_BRANCH"
log "Work Directory: $WORK_DIR"
log "Log File: $LOG_FILE"

# Check if SONAR_TOKEN is set
if [ -z "$SONAR_TOKEN" ]; then
  log "ERROR: SONAR_TOKEN environment variable is not set"
  log "Please set it in crontab or source a secrets file"
  exit 1
fi

# Clone or update repository
if [ -d "$WORK_DIR" ]; then
  log "Repository directory exists, pulling latest changes..."
  cd "$WORK_DIR"
  
  # Check if it's a git repository
  if ! git rev-parse --git-dir > /dev/null 2>&1; then
    log "ERROR: $WORK_DIR exists but is not a git repository"
    log "Removing and will re-clone..."
    cd /tmp
    rm -rf "$WORK_DIR"
  else
    # Ensure we're on the correct branch
    current_branch=$(git rev-parse --abbrev-ref HEAD)
    if [ "$current_branch" != "$REPO_BRANCH" ]; then
      log "Switching from branch $current_branch to $REPO_BRANCH"
      if ! git fetch origin; then
        log "ERROR: Failed to fetch from origin"
        exit 1
      fi
      if ! git checkout "$REPO_BRANCH"; then
        log "ERROR: Failed to checkout branch $REPO_BRANCH"
        exit 1
      fi
    fi
    
    # Pull latest changes
    log "Pulling latest changes for branch: $REPO_BRANCH"
    if ! git pull origin "$REPO_BRANCH"; then
      log "ERROR: Failed to pull latest changes"
      exit 1
    fi
    
    log "Successfully updated repository"
  fi
fi

# Clone if directory doesn't exist
if [ ! -d "$WORK_DIR" ]; then
  log "Cloning repository for the first time..."
  if ! git clone --branch "$REPO_BRANCH" "$REPO_URL" "$WORK_DIR"; then
    log "ERROR: Failed to clone repository"
    exit 1
  fi
  log "Successfully cloned repository"
fi

# Change to repository directory
cd "$WORK_DIR"

# Show current commit
CURRENT_COMMIT=$(git rev-parse --short HEAD)
COMMIT_MSG=$(git log -1 --pretty=%B)
log "Current commit: $CURRENT_COMMIT"
log "Commit message: $COMMIT_MSG"

# Check if sonarqube directory exists
if [ ! -d "sonarqube" ]; then
  log "ERROR: sonarqube directory not found in repository"
  exit 1
fi

# Check if run-sonar-analysis.sh exists
if [ ! -f "sonarqube/run-sonar-analysis.sh" ]; then
  log "ERROR: sonarqube/run-sonar-analysis.sh not found"
  exit 1
fi

# Make script executable
chmod +x sonarqube/run-sonar-analysis.sh

# Run SonarQube analysis
log "========================================"
log "Starting SonarQube Analysis"
log "========================================"

if ./sonarqube/run-sonar-analysis.sh 2>&1 | tee -a "$LOG_FILE"; then
  log "========================================"
  log "SonarQube Analysis Completed Successfully"
  log "========================================"
  exit 0
else
  EXIT_CODE=$?
  log "========================================"
  log "SonarQube Analysis Failed (Exit Code: $EXIT_CODE)"
  log "========================================"
  exit $EXIT_CODE
fi
