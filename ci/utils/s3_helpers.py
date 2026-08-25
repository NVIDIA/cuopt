#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Shared S3 helper functions for cuOpt CI scripts.

Maps CUOPT_AWS_* credentials to standard AWS env vars and provides
download / upload / list wrappers around the aws CLI.
"""

import os
import subprocess
import sys


def s3_env():
    """Build env dict for AWS CLI calls using CUOPT-specific credentials.

    The cuOpt S3 bucket requires explicit CUOPT_AWS_* static credentials.
    Role-based credentials from aws-actions/configure-aws-credentials do not
    have access.  We override AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY with
    the CUOPT_* values and unset AWS_SESSION_TOKEN to avoid mixing with
    role-based session tokens (matching the pattern in datasets/*.sh).
    """
    env = os.environ.copy()
    if os.environ.get("CUOPT_AWS_ACCESS_KEY_ID"):
        env["AWS_ACCESS_KEY_ID"] = os.environ["CUOPT_AWS_ACCESS_KEY_ID"]
    if os.environ.get("CUOPT_AWS_SECRET_ACCESS_KEY"):
        env["AWS_SECRET_ACCESS_KEY"] = os.environ[
            "CUOPT_AWS_SECRET_ACCESS_KEY"
        ]
    # Unset session token to avoid mixing role-based tokens with static keys
    env.pop("AWS_SESSION_TOKEN", None)
    if os.environ.get("CUOPT_AWS_REGION"):
        env["AWS_DEFAULT_REGION"] = os.environ["CUOPT_AWS_REGION"]
    elif "AWS_DEFAULT_REGION" not in env:
        env["AWS_DEFAULT_REGION"] = "us-east-1"
    return env


def s3_download(s3_uri, local_path):
    """Download a file from S3. Returns True on success, False on any error."""
    env = s3_env()
    try:
        subprocess.run(
            ["aws", "s3", "cp", s3_uri, local_path],
            env=env,
            check=True,
            capture_output=True,
            text=True,
        )
        print(f"Downloaded {s3_uri}")
        return True
    except FileNotFoundError:
        print(
            "WARNING: aws CLI not found, skipping S3 download", file=sys.stderr
        )
        return False
    except subprocess.CalledProcessError as exc:
        print(
            f"WARNING: S3 download failed (first run?): {exc.stderr.strip()}",
            file=sys.stderr,
        )
        return False


def s3_upload(local_path, s3_uri):
    """Upload a file to S3. Returns True on success."""
    env = s3_env()
    try:
        subprocess.run(
            ["aws", "s3", "cp", local_path, s3_uri],
            env=env,
            check=True,
            capture_output=True,
            text=True,
        )
        print(f"Uploaded {local_path} to {s3_uri}")
        return True
    except FileNotFoundError:
        print(
            "WARNING: aws CLI not found, skipping S3 upload", file=sys.stderr
        )
        return False
    except subprocess.CalledProcessError as exc:
        print(
            f"WARNING: S3 upload failed: {exc.stderr.strip()}", file=sys.stderr
        )
        return False


def s3_list(s3_prefix):
    """List objects under an S3 prefix (recursive). Returns list of S3 URIs."""
    env = s3_env()
    # Extract bucket and prefix from s3_prefix for reconstructing full URIs
    # s3_prefix looks like "s3://bucket/path/to/prefix/"
    try:
        result = subprocess.run(
            ["aws", "s3", "ls", "--recursive", s3_prefix],
            env=env,
            check=True,
            capture_output=True,
            text=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError) as exc:
        print(f"WARNING: S3 ls failed: {exc}", file=sys.stderr)
        return []

    # --recursive output format: "2026-04-16 12:00:00  1234 path/to/file.json"
    # We need to reconstruct full S3 URIs from the key paths
    # Parse bucket from s3_prefix
    if not s3_prefix.startswith("s3://"):
        return []
    without_scheme = s3_prefix[5:]  # remove "s3://"
    bucket = without_scheme.split("/")[0]
    base_uri = f"s3://{bucket}/"

    uris = []
    for line in result.stdout.strip().splitlines():
        parts = line.split(None, 3)  # date, time, size, key
        if len(parts) == 4:
            uris.append(f"{base_uri}{parts[3]}")
    return uris
