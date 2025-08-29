#!/usr/bin/env python3

# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


"""
Notebook Command Extractor

This script extracts pip install and other shell commands from Jupyter notebooks
and can optionally execute them. It's designed to be used by the nbtest.sh script.
"""

import argparse
import json
import subprocess
import sys
from typing import List, Tuple


def extract_pip_commands(notebook_path: str) -> List[str]:
    """Extract pip install commands from a Jupyter notebook."""
    try:
        with open(notebook_path, "r") as f:
            notebook = json.load(f)

        pip_commands = []
        for cell in notebook.get("cells", []):
            if cell.get("cell_type") == "code":
                source = "".join(cell.get("source", []))
                lines = source.split("\n")
                for line in lines:
                    line = line.strip()
                    if line.startswith("!pip install") or line.startswith(
                        "pip install"
                    ):
                        # Clean up the line but preserve quotes
                        clean_line = line.strip()
                        if clean_line:
                            pip_commands.append(clean_line)

        return pip_commands

    except Exception as e:
        print(f"Error parsing notebook: {e}", file=sys.stderr)
        return []


def extract_shell_commands(notebook_path: str) -> List[str]:
    """Extract other shell commands from a Jupyter notebook."""
    try:
        with open(notebook_path, "r") as f:
            notebook = json.load(f)

        shell_commands = []
        allowed_commands = [
            "wget",
            "curl",
            "git",
            "python",
            "cd",
            "mkdir",
            "rm",
            "cp",
            "mv",
            "unzip",
            "tar",
        ]

        for cell in notebook.get("cells", []):
            if cell.get("cell_type") == "code":
                source = "".join(cell.get("source", []))
                lines = source.split("\n")
                for line in lines:
                    line = line.strip()
                    if line.startswith("!"):
                        # Check if it's a shell command we want to execute
                        cmd = (
                            line[1:].strip().split()[0]
                            if line[1:].strip()
                            else ""
                        )
                        if cmd in allowed_commands:
                            shell_commands.append(line)

        return shell_commands

    except Exception as e:
        print(f"Error parsing notebook: {e}", file=sys.stderr)
        return []


def execute_pip_command(cmd: str, verbose: bool = False) -> bool:
    """Execute a pip install command."""
    if verbose:
        print(f"Processing command: '{cmd}'")

    # Remove the ! prefix if present for execution
    exec_cmd = cmd.lstrip("!").strip()

    if verbose:
        print(f"DEBUG: Original command: '{cmd}'")
        print(f"DEBUG: Cleaned command: '{exec_cmd}'")
        print(f"DEBUG: Command length: {len(exec_cmd)}")
        print(
            f"DEBUG: Command contains 'numpy': {'YES' if 'numpy' in exec_cmd else 'NO'}"
        )
        print(f"Executing: {exec_cmd}")

    # Add --pre to exec_cmd if not already present
    if exec_cmd.startswith("pip install") and "--pre" not in exec_cmd:
        exec_cmd += " --pre --extra-index-url https://pypi.anaconda.org/rapidsai-nightly/simple"

    if verbose:
        print(f"Final command: {exec_cmd}")

    try:
        # Execute pip install commands
        if exec_cmd.startswith("pip install"):
            # Use shell=True for pip install to handle quoted arguments properly
            # This is safe since we're only executing pip install commands
            result = subprocess.run(
                exec_cmd, shell=True, capture_output=True, text=True
            )
            if result.returncode == 0:
                if verbose:
                    print(f"✓ Successfully executed: {cmd}")
                return True
            else:
                if verbose:
                    print(f"✗ Failed to execute: {cmd}")
                    print(f"Error: {result.stderr}")
                return False
        else:
            if verbose:
                print(f"✗ Invalid pip install command format: {exec_cmd}")
            return False
    except Exception as e:
        if verbose:
            print(f"✗ Exception executing {cmd}: {e}")
        return False


def execute_shell_command(cmd: str, verbose: bool = False) -> bool:
    """Execute a shell command."""
    if verbose:
        print(f"Processing command: '{cmd}'")

    # Remove the ! prefix for execution
    exec_cmd = cmd.lstrip("!").strip()

    if verbose:
        print(f"DEBUG: Original command: '{cmd}'")
        print(f"DEBUG: Cleaned command: '{exec_cmd}'")
        print(f"DEBUG: Command length: {len(exec_cmd)}")
        print(f"Executing: {exec_cmd}")

    # Skip potentially dangerous commands
    dangerous_commands = ["chmod", "chown", "sudo", "su"]
    if any(exec_cmd.startswith(dangerous) for dangerous in dangerous_commands):
        if verbose:
            print(f"⚠ Skipping potentially dangerous command: {cmd}")
        return False

    try:
        if verbose:
            print("Executing shell command...")

        result = subprocess.run(
            exec_cmd, shell=True, capture_output=True, text=True
        )
        if result.returncode == 0:
            if verbose:
                print(f"✓ Successfully executed: {cmd}")
            return True
        else:
            if verbose:
                print(f"✗ Failed to execute: {cmd}")
                print(f"Error: {result.stderr}")
            return False
    except Exception as e:
        if verbose:
            print(f"✗ Exception executing {cmd}: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Extract and optionally execute commands from Jupyter notebooks"
    )
    parser.add_argument("notebook_path", help="Path to the Jupyter notebook")
    parser.add_argument(
        "--extract-only",
        action="store_true",
        help="Only extract commands, do not execute",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Verbose output"
    )
    parser.add_argument(
        "--output-format",
        choices=["json", "text"],
        default="text",
        help="Output format for extracted commands",
    )

    args = parser.parse_args()

    # Extract commands
    pip_commands = extract_pip_commands(args.notebook_path)
    shell_commands = extract_shell_commands(args.notebook_path)

    print(f"Pip commands: {pip_commands}")
    print(f"Shell commands: {shell_commands}")

    if args.output_format == "json":
        # Output as JSON for shell script processing
        output = {
            "pip_commands": pip_commands,
            "shell_commands": shell_commands,
        }
        print(json.dumps(output))
    else:
        # Output as text (default)
        if pip_commands:
            print("PIP_COMMANDS:")
            for cmd in pip_commands:
                print(cmd)

        if shell_commands:
            print("SHELL_COMMANDS:")
            for cmd in shell_commands:
                print(cmd)

    # Execute commands if not extract-only mode
    if not args.extract_only:
        success_count = 0
        total_count = 0

        if pip_commands:
            print(f"\nExecuting {len(pip_commands)} pip install commands...")
            for cmd in pip_commands:
                if execute_pip_command(cmd, args.verbose):
                    success_count += 1
                total_count += 1

        if shell_commands:
            print(f"\nExecuting {len(shell_commands)} shell commands...")
            for cmd in shell_commands:
                if execute_shell_command(cmd, args.verbose):
                    success_count += 1
                total_count += 1

        if total_count > 0:
            print(
                f"\nExecution summary: {success_count}/{total_count} commands succeeded"
            )
            return 0 if success_count == total_count else 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
