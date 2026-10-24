/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
package com.nvidia.cuopt.mathematicaloptimization;

/**
 * Raw JNI surface onto {@code cuopt::cython::grpc_python_client_t} -- the same C++ class the
 * Python bindings wrap, including its automatic chunked upload/download for problems too large
 * for a single protobuf message. See {@link GrpcClient} for the ergonomic wrapper.
 */
final class NativeGrpcClient {
  static {
    NativeLibraryLoader.load();
  }

  private NativeGrpcClient() {}

  static native long createClient(String host, int port);

  /** tlsMode: 0=ENV, 1=DISABLED, 2=EXPLICIT (matches grpc_python_tls_mode_t). */
  static native long createClientWithTls(
      String host,
      int port,
      int tlsMode,
      String tlsRootCerts,
      String tlsClientCert,
      String tlsClientKey);

  static native void destroyClient(long handle);

  static native void connect(long handle);

  static native boolean ping(long handle, int timeoutSeconds);

  /** LP vs MIP is decided automatically from variableTypes (any 'I' makes it a MIP). */
  static native String submit(
      long handle,
      int numConstraints,
      int numVariables,
      boolean maximize,
      double objectiveOffset,
      double[] objectiveCoefficients,
      int[] rowOffsets,
      int[] columnIndices,
      double[] values,
      byte[] rowTypes,
      double[] constraintBounds,
      double[] variableLowerBounds,
      double[] variableUpperBounds,
      byte[] variableTypes,
      double timeLimitSeconds,
      boolean enableIncumbents);

  static native int getStatus(long handle, String jobId);

  /** Server-side long-poll; timeoutSeconds=0 waits indefinitely for one long-poll response. */
  static native int waitForCompletion(long handle, String jobId, int timeoutSeconds);

  static native void cancel(long handle, String jobId);

  static native void deleteJob(long handle, String jobId);

  /** Packed as [terminationStatus, objective, mipGap, solutionBound, totalSolveTime, var...]. */
  static native double[] getMipResult(long handle, String jobId);

  /**
   * Packed as [terminationStatus, primalObjective, dualObjective, gap, solveTime, numVariables,
   * numConstraints, primal..., dual..., reducedCost...].
   */
  static native double[] getLpResult(long handle, String jobId);
}
