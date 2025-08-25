#!/usr/bin/env python3

# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""
Script to update versions1.json with new version from VERSION file.
This script reads the current version from the VERSION file and adds it to versions1.json
if it doesn't already exist, setting it as the latest and preferred version.
"""

import json
import os
import sys
from pathlib import Path


def update_versions(version_file_path, versions_file_path):
    """Update the versions list with the new version."""
    # Read VERSION file
    with open(version_file_path, "r") as f:
        version = f.read().strip()
    if not version:
        raise ValueError("VERSION file is empty")

    # Read versions1.json file
    with open(versions_file_path, "r") as f:
        versions = json.load(f)

    # Check if version already exists
    for version_entry in versions:
        if version_entry.get("version") == version:
            print(f"Version {version} already exists in versions1.json")
            return False

    # Remove "latest" and "preferred" from existing entries
    for version_entry in versions:
        if version_entry.get("name") == "latest":
            version_entry.pop("name", None)
        if version_entry.get("preferred"):
            version_entry.pop("preferred", None)

    # Create new entry for the current version
    new_entry = {
        "version": version,
        "url": f"../{version}/",
        "name": "latest",
        "preferred": True,
    }

    # Add new entry at the beginning (most recent first)
    versions.insert(0, new_entry)

    # Write updated versions back to file
    with open(versions_file_path, "w") as f:
        json.dump(versions, f, indent=2)

    return True


def main():
    """Main function to update versions1.json."""
    # Get the repository root directory (assuming script is run from repo root)
    repo_root = Path.cwd()

    # Hard-coded file paths
    version_file_path = repo_root / "VERSION"
    versions_file_path = (
        repo_root / "docs" / "cuopt" / "source" / "versions1.json"
    )

    # Update versions
    update_versions(version_file_path, versions_file_path)


if __name__ == "__main__":
    main()
