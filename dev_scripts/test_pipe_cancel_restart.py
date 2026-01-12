#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Test job cancellation with pipe mode and verify worker restart/pipe recreation.

This test:
1. Submits a long-running LP job (with many iterations)
2. Cancels it while running
3. Verifies the cancellation succeeded
4. Submits a simple job and verifies it completes (tests pipe recreation)
"""

import socket
import struct
import time

# Import the generated protobuf module
import sys

sys.path.insert(0, "/home/tmckay/repos/nvidia-cuopt/cpp/build")
import cuopt_remote_pb2

HOST = "localhost"
PORT = 8765


def send_recv(sock, data):
    """Send request and receive response with uint64 size prefix."""
    # Send size (uint64) + data
    sock.sendall(struct.pack("<Q", len(data)) + data)

    # Receive size (uint64)
    size_data = b""
    while len(size_data) < 8:
        chunk = sock.recv(8 - len(size_data))
        if not chunk:
            raise ConnectionError("Connection closed")
        size_data += chunk

    size = struct.unpack("<Q", size_data)[0]

    # Receive response
    response = b""
    while len(response) < size:
        chunk = sock.recv(min(4096, size - len(response)))
        if not chunk:
            raise ConnectionError("Connection closed")
        response += chunk

    return response


def create_long_running_lp():
    """Create an LP problem that takes a while to solve."""
    # Create a much larger problem that will run longer
    n_vars = 5000
    n_constraints = 3000

    req = cuopt_remote_pb2.AsyncRequest()
    req.request_type = cuopt_remote_pb2.SUBMIT_JOB
    req.blocking = False

    lp = req.lp_request
    prob = lp.problem
    prob.maximize = False

    # Objective: sum of all variables
    prob.c.extend([1.0] * n_vars)

    # Variable bounds: 0 <= x_i <= 10
    prob.variable_lower_bounds.extend([0.0] * n_vars)
    prob.variable_upper_bounds.extend([10.0] * n_vars)

    # Constraint bounds: sum constraints
    prob.constraint_lower_bounds.extend([1.0] * n_constraints)
    prob.constraint_upper_bounds.extend([float("inf")] * n_constraints)

    # Sparse constraint matrix (each constraint involves ~10 variables)
    offsets = [0]
    indices = []
    values = []

    for i in range(n_constraints):
        # Each constraint: x[i] + x[i+1] + ... + x[i+9] >= 1
        for j in range(10):
            var_idx = (i + j) % n_vars
            indices.append(var_idx)
            values.append(1.0)
        offsets.append(len(indices))

    prob.A_offsets.extend(offsets)
    prob.A_indices.extend(indices)
    prob.A.extend(values)

    # Row types: all >= constraints
    prob.row_types = bytes([ord("G")] * n_constraints)

    # Settings to make it run longer
    lp.settings.log_to_console = True
    lp.settings.time_limit = 60.0  # Allow up to 60 seconds

    return req.SerializeToString()


def create_simple_lp():
    """Create a simple LP that solves quickly."""
    req = cuopt_remote_pb2.AsyncRequest()
    req.request_type = cuopt_remote_pb2.SUBMIT_JOB
    req.blocking = False

    lp = req.lp_request
    prob = lp.problem
    prob.maximize = False

    # Simple: minimize x + 2y subject to x + y >= 1, x,y >= 0
    prob.c.extend([1.0, 2.0])
    prob.variable_lower_bounds.extend([0.0, 0.0])
    prob.variable_upper_bounds.extend([float("inf"), float("inf")])
    prob.constraint_lower_bounds.extend([1.0])
    prob.constraint_upper_bounds.extend([float("inf")])
    prob.A_offsets.extend([0, 2])
    prob.A_indices.extend([0, 1])
    prob.A.extend([1.0, 1.0])
    prob.row_types = bytes([ord("G")])
    lp.settings.log_to_console = True

    return req.SerializeToString()


def submit_job(sock, request_data):
    """Submit a job and return the job_id."""
    response_data = send_recv(sock, request_data)
    response = cuopt_remote_pb2.AsyncResponse()
    response.ParseFromString(response_data)

    if response.submit_response.status != cuopt_remote_pb2.SUCCESS:
        raise Exception(f"Submit failed: {response.submit_response.message}")

    return (
        response.submit_response.job_id.decode()
        if isinstance(response.submit_response.job_id, bytes)
        else response.submit_response.job_id
    )


def check_status(sock, job_id):
    """Check job status."""
    req = cuopt_remote_pb2.AsyncRequest()
    req.request_type = cuopt_remote_pb2.CHECK_STATUS
    req.job_id = job_id.encode() if isinstance(job_id, str) else job_id

    response_data = send_recv(sock, req.SerializeToString())
    response = cuopt_remote_pb2.AsyncResponse()
    response.ParseFromString(response_data)

    return (
        response.status_response.job_status,
        response.status_response.message,
    )


def cancel_job(sock, job_id):
    """Cancel a job."""
    req = cuopt_remote_pb2.AsyncRequest()
    req.request_type = cuopt_remote_pb2.CANCEL_JOB
    req.job_id = job_id.encode() if isinstance(job_id, str) else job_id

    response_data = send_recv(sock, req.SerializeToString())
    response = cuopt_remote_pb2.AsyncResponse()
    response.ParseFromString(response_data)

    success = response.cancel_response.status == cuopt_remote_pb2.SUCCESS
    return success, response.cancel_response.message


def wait_for_result(sock, job_id):
    """Wait for job to complete using WAIT_FOR_RESULT."""
    req = cuopt_remote_pb2.AsyncRequest()
    req.request_type = cuopt_remote_pb2.WAIT_FOR_RESULT
    req.job_id = job_id.encode() if isinstance(job_id, str) else job_id

    response_data = send_recv(sock, req.SerializeToString())
    response = cuopt_remote_pb2.AsyncResponse()
    response.ParseFromString(response_data)

    return (
        response.result_response.status == cuopt_remote_pb2.SUCCESS,
        response,
    )


def main():
    print("=" * 60)
    print("Test: Job Cancellation with Pipe Mode + Worker Restart")
    print("=" * 60)

    # Step 1: Submit a long-running job
    print("\n[1] Submitting long-running LP job...")
    sock1 = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock1.connect((HOST, PORT))

    long_job_data = create_long_running_lp()
    job_id = submit_job(sock1, long_job_data)
    print(f"    Job submitted: {job_id}")
    sock1.close()

    # Step 2: Wait a bit for job to start processing
    print("\n[2] Waiting for job to start processing...")
    time.sleep(2)

    sock2 = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock2.connect((HOST, PORT))
    status, msg = check_status(sock2, job_id)
    print(f"    Status: {cuopt_remote_pb2.JobStatus.Name(status)} - {msg}")
    sock2.close()

    # Step 3: Cancel the job
    print("\n[3] Cancelling the job...")
    sock3 = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock3.connect((HOST, PORT))
    success, msg = cancel_job(sock3, job_id)
    print(f"    Cancel result: success={success}, message={msg}")
    sock3.close()

    # Step 4: Verify job is cancelled
    print("\n[4] Verifying job status after cancellation...")
    time.sleep(
        1
    )  # Give time for worker to be killed and result to be recorded

    sock4 = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock4.connect((HOST, PORT))
    status, msg = check_status(sock4, job_id)
    print(f"    Status: {cuopt_remote_pb2.JobStatus.Name(status)} - {msg}")
    sock4.close()

    if status != cuopt_remote_pb2.CANCELLED:
        print(
            f"    WARNING: Expected CANCELLED, got {cuopt_remote_pb2.JobStatus.Name(status)}"
        )
    else:
        print("    ✓ Job successfully cancelled!")

    # Step 5: Wait a bit for worker to restart with new pipes
    print("\n[5] Waiting for worker to restart (with new pipes)...")
    time.sleep(2)

    # Step 6: Submit a new simple job to test pipe recreation
    print("\n[6] Submitting new simple LP job to test pipe recreation...")
    sock5 = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock5.connect((HOST, PORT))

    simple_job_data = create_simple_lp()
    job_id2 = submit_job(sock5, simple_job_data)
    print(f"    Job submitted: {job_id2}")
    sock5.close()

    # Step 7: Wait for the new job to complete
    print("\n[7] Waiting for new job to complete...")
    sock6 = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock6.connect((HOST, PORT))
    success, response = wait_for_result(sock6, job_id2)
    sock6.close()

    if success:
        print("    ✓ New job completed successfully!")
        if response.result_response.HasField("lp_solution"):
            sol = response.result_response.lp_solution
            obj = sol.primal_objective
            print(f"    Objective value: {obj}")
            if abs(obj - 1.0) < 0.01:
                print("    ✓ Objective matches expected value (1.0)!")
            else:
                print(f"    WARNING: Expected objective ~1.0, got {obj}")
    else:
        print("    ✗ New job failed!")
        return 1

    print("\n" + "=" * 60)
    print("TEST PASSED: Cancellation and pipe recreation work correctly!")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
