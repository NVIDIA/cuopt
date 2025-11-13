# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Aborting a Running MILP Job Example

This example demonstrates how to abort a running or queued job on the cuOpt server.
This is useful when:
- A solve is taking too long
- You want to cancel a queued job
- You need to free up server resources

Requirements:
    - cuOpt server running (default: localhost:5000)
    - cuopt_sh_client package installed

Usage:
    Replace <UUID_THAT_WE_GOT> with the actual UUID returned by the solver
    when you submitted a job.
"""

from cuopt_sh_client import CuOptServiceSelfHostClient


def main():
    """Run the abort job example."""
    # This is a UUID that is returned by the solver while the solver is trying
    # to find solution so users can come back and check the status or query for results.
    job_uuid = "<UUID_THAT_WE_GOT>"

    print(f"Attempting to abort job: {job_uuid}")

    # If cuOpt is not running on localhost:5000, edit ip and port parameters
    cuopt_service_client = CuOptServiceSelfHostClient(
        ip="localhost", port=5000
    )

    # Delete the job if it is still queued or running
    # Parameters:
    #   - job_uuid: The UUID of the job to abort
    #   - running=True: Abort if the job is currently running
    #   - queued=True: Abort if the job is in the queue
    #   - cached=False: Don't delete from cache (only abort active/queued jobs)
    response = cuopt_service_client.delete(
        job_uuid, running=True, queued=True, cached=False
    )

    print(f"Response: {response}")


if __name__ == "__main__":
    # Example usage - in practice, you would get the UUID from a previous job submission
    print("This is a template example.")
    print("To use this:")
    print("1. Submit a MILP job and get its UUID")
    print("2. Replace '<UUID_THAT_WE_GOT>' in this file with that UUID")
    print("3. Run this script to abort the job")

    # Uncomment the line below and add your actual UUID to run
    # main()
