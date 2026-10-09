/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
package com.nvidia.cuopt.mathematicaloptimization;

/** Mirrors cuopt::cython::grpc_job_status_t (cpp/include/cuopt/grpc/cython_grpc_client.hpp). */
public enum GrpcJobStatus {
  QUEUED,
  PROCESSING,
  COMPLETED,
  FAILED,
  CANCELLED,
  NOT_FOUND;

  static GrpcJobStatus fromOrdinal(int ordinal) {
    GrpcJobStatus[] values = values();
    if (ordinal < 0 || ordinal >= values.length) {
      throw new IllegalArgumentException("Unknown gRPC job status ordinal: " + ordinal);
    }
    return values[ordinal];
  }

  boolean isInFlight() {
    return this == QUEUED || this == PROCESSING;
  }
}
