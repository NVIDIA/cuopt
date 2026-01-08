# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Remote solve utilities for cuOpt.

This module provides functions for interacting with a remote cuopt_remote_server,
including job management operations like cancellation.
"""

import os
import socket
import struct
from dataclasses import dataclass
from enum import IntEnum
from typing import Optional

# Try to import the protobuf module - may not be available in all environments
try:
    import sys

    # Add build directory for development
    build_path = os.path.join(
        os.path.dirname(__file__), "..", "..", "..", "..", "..", "cpp", "build"
    )
    if os.path.exists(build_path):
        sys.path.insert(0, os.path.abspath(build_path))
    import cuopt_remote_pb2 as pb

    _HAS_PROTOBUF = True
except ImportError:
    _HAS_PROTOBUF = False


class JobStatus(IntEnum):
    """Status of a remote job."""

    QUEUED = 0  # Job is waiting in queue
    PROCESSING = 1  # Job is being processed by a worker
    COMPLETED = 2  # Job completed successfully
    FAILED = 3  # Job failed with an error
    NOT_FOUND = 4  # Job ID not found on server
    CANCELLED = 5  # Job was cancelled


@dataclass
class CancelResult:
    """Result of a cancel job request."""

    success: bool
    message: str
    job_status: JobStatus


def get_remote_config() -> Optional[tuple]:
    """Get remote server configuration from environment variables.

    Returns
    -------
    tuple or None
        (host, port) tuple if CUOPT_REMOTE_HOST and CUOPT_REMOTE_PORT are set,
        None otherwise.
    """
    host = os.environ.get("CUOPT_REMOTE_HOST", "")
    port = os.environ.get("CUOPT_REMOTE_PORT", "")

    if host and port:
        try:
            return (host, int(port))
        except ValueError:
            return None
    return None


def cancel_job(
    job_id: str, host: Optional[str] = None, port: Optional[int] = None
) -> CancelResult:
    """Cancel a job on a remote cuopt_remote_server.

    This function can cancel jobs that are queued (waiting for a worker) or
    currently running. For running jobs, the worker process is killed and
    automatically restarted by the server.

    Parameters
    ----------
    job_id : str
        The job ID to cancel (e.g., "job_1234567890abcdef")
    host : str, optional
        Server hostname. If not provided, uses CUOPT_REMOTE_HOST environment variable.
    port : int, optional
        Server port. If not provided, uses CUOPT_REMOTE_PORT environment variable.

    Returns
    -------
    CancelResult
        Result containing success status, message, and job status after cancellation.

    Raises
    ------
    RuntimeError
        If protobuf module is not available or connection fails.
    ValueError
        If host/port are not provided and environment variables are not set.

    Examples
    --------
    >>> # Using environment variables
    >>> import os
    >>> os.environ['CUOPT_REMOTE_HOST'] = 'localhost'
    >>> os.environ['CUOPT_REMOTE_PORT'] = '9090'
    >>> result = cancel_job("job_1234567890abcdef")
    >>> print(result.success, result.message)

    >>> # Explicitly specifying host and port
    >>> result = cancel_job("job_1234567890abcdef", host="192.168.1.100", port=9090)
    """
    if not _HAS_PROTOBUF:
        raise RuntimeError(
            "Protobuf module not available. Please install protobuf or ensure "
            "cuopt_remote_pb2.py is in the Python path."
        )

    # Get host/port from parameters or environment
    if host is None or port is None:
        config = get_remote_config()
        if config is None:
            raise ValueError(
                "Host and port must be provided or set via CUOPT_REMOTE_HOST "
                "and CUOPT_REMOTE_PORT environment variables."
            )
        if host is None:
            host = config[0]
        if port is None:
            port = config[1]

    # Create cancel request
    request = pb.AsyncRequest()
    request.request_type = pb.CANCEL_JOB
    request.job_id = job_id

    try:
        # Connect to server
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(30.0)  # 30 second timeout
        sock.connect((host, port))

        # Send request (length-prefixed)
        data = request.SerializeToString()
        sock.sendall(struct.pack("<I", len(data)))
        sock.sendall(data)

        # Receive response
        size_data = sock.recv(4)
        if len(size_data) < 4:
            sock.close()
            return CancelResult(
                success=False,
                message="Failed to receive response size",
                job_status=JobStatus.NOT_FOUND,
            )

        size = struct.unpack("<I", size_data)[0]
        response_data = b""
        while len(response_data) < size:
            chunk = sock.recv(size - len(response_data))
            if not chunk:
                break
            response_data += chunk

        sock.close()

        # Parse response
        response = pb.AsyncResponse()
        response.ParseFromString(response_data)

        cancel_resp = response.cancel_response

        # Map protobuf status to our enum
        status_map = {
            pb.QUEUED: JobStatus.QUEUED,
            pb.PROCESSING: JobStatus.PROCESSING,
            pb.COMPLETED: JobStatus.COMPLETED,
            pb.FAILED: JobStatus.FAILED,
            pb.NOT_FOUND: JobStatus.NOT_FOUND,
            pb.CANCELLED: JobStatus.CANCELLED,
        }
        job_status = status_map.get(
            cancel_resp.job_status, JobStatus.NOT_FOUND
        )

        return CancelResult(
            success=(cancel_resp.status == pb.SUCCESS),
            message=cancel_resp.message,
            job_status=job_status,
        )

    except socket.timeout:
        return CancelResult(
            success=False,
            message="Connection timed out",
            job_status=JobStatus.NOT_FOUND,
        )
    except socket.error as e:
        return CancelResult(
            success=False,
            message=f"Connection error: {e}",
            job_status=JobStatus.NOT_FOUND,
        )
    except Exception as e:
        return CancelResult(
            success=False,
            message=f"Error: {e}",
            job_status=JobStatus.NOT_FOUND,
        )
