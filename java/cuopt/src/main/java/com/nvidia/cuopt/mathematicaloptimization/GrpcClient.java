/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
package com.nvidia.cuopt.mathematicaloptimization;

/**
 * Java client for a remote cuOpt gRPC server, wrapping the same C++ {@code grpc_client_t} the
 * Python client wraps -- including its automatic chunked upload/download for problems too large
 * for a single protobuf message (this is what avoids the protobuf/int 2GB limit that a plain
 * REST client can hit on very large problems).
 *
 * <p>Covers LP and MIP submit/status/wait/cancel/delete/result, plaintext or TLS. Not covered
 * yet: log streaming and incumbent callbacks, which need a JNI callback path across native
 * threads -- a separate piece of work from the request/response calls here.
 *
 * <p>Not wired into the Maven-packaged classifier jar or CI yet -- see java/cuopt/README.md for
 * how to build java/cuopt/src/main/native/cuopt_grpc_jni.cpp locally against an existing
 * libcuopt build.
 */
public final class GrpcClient implements AutoCloseable {
  private long handle;

  public GrpcClient(String host, int port) {
    this.handle = NativeGrpcClient.createClient(host, port);
  }

  private GrpcClient(long handle) {
    this.handle = handle;
  }

  /** tlsMode: 0=ENV (respects the usual gRPC/OpenSSL env vars), 1=DISABLED, 2=EXPLICIT. */
  public static GrpcClient withTls(
      String host,
      int port,
      int tlsMode,
      String tlsRootCerts,
      String tlsClientCert,
      String tlsClientKey) {
    return new GrpcClient(
        NativeGrpcClient.createClientWithTls(
            host, port, tlsMode, tlsRootCerts, tlsClientCert, tlsClientKey));
  }

  public void connect() {
    NativeGrpcClient.connect(handle);
  }

  public boolean ping(int timeoutSeconds) {
    return NativeGrpcClient.ping(handle, timeoutSeconds);
  }

  /**
   * Submits a {@link Problem} built by the regular Java API (e.g. {@link Problem#read}) --
   * mirrors the Python client's {@code submit()}, which also takes the high-level problem object
   * directly rather than raw arrays. Rejects quadratic problems: the gRPC wire protocol supports
   * QP, but this client doesn't wire it through yet.
   */
  public String submit(Problem problem, double timeLimitSeconds) {
    return submit(problem, timeLimitSeconds, /* enableIncumbents= */ false);
  }

  public String submit(Problem problem, double timeLimitSeconds, boolean enableIncumbents) {
    if (problem.getQuadraticObjectiveMatrix() != null || !problem.getQuadraticConstraints().isEmpty()) {
      throw new UnsupportedOperationException(
          "GrpcClient does not support quadratic problems (QP/QCQP) yet");
    }
    return submit(new ProblemAdapter(problem), timeLimitSeconds, enableIncumbents);
  }

  /** LP vs MIP is decided automatically from the problem's variable types (any 'I' -> MIP). */
  public String submit(RawProblem problem, double timeLimitSeconds) {
    return submit(problem, timeLimitSeconds, /* enableIncumbents= */ false);
  }

  public String submit(RawProblem problem, double timeLimitSeconds, boolean enableIncumbents) {
    return NativeGrpcClient.submit(
        handle,
        problem.getNumConstraints(),
        problem.getNumVariables(),
        problem.isMaximize(),
        problem.getObjectiveOffset(),
        problem.getObjectiveCoefficients(),
        problem.getRowOffsets(),
        problem.getColumnIndices(),
        problem.getValues(),
        problem.getRowTypes(),
        problem.getConstraintBounds(),
        problem.getVariableLowerBounds(),
        problem.getVariableUpperBounds(),
        problem.getVariableTypes(),
        timeLimitSeconds,
        enableIncumbents);
  }

  public GrpcJobStatus getStatus(String jobId) {
    return GrpcJobStatus.fromOrdinal(NativeGrpcClient.getStatus(handle, jobId));
  }

  /**
   * Blocks server-side (long-poll) until the job leaves QUEUED/PROCESSING or timeoutSeconds
   * elapses. Prefer this over a client-side sleep loop around {@link #getStatus}: it returns as
   * soon as the job finishes instead of waiting for the next poll interval, and it doesn't spend
   * a request every interval just to find out nothing has changed yet.
   */
  public GrpcJobStatus waitForCompletion(String jobId, int timeoutSeconds) {
    return GrpcJobStatus.fromOrdinal(
        NativeGrpcClient.waitForCompletion(handle, jobId, timeoutSeconds));
  }

  public void cancel(String jobId) {
    NativeGrpcClient.cancel(handle, jobId);
  }

  public void deleteJob(String jobId) {
    NativeGrpcClient.deleteJob(handle, jobId);
  }

  public GrpcMipResult getMipResult(String jobId) {
    return GrpcMipResult.fromPacked(NativeGrpcClient.getMipResult(handle, jobId));
  }

  public GrpcLpResult getLpResult(String jobId) {
    return GrpcLpResult.fromPacked(NativeGrpcClient.getLpResult(handle, jobId));
  }

  @Override
  public void close() {
    if (handle != 0) {
      NativeGrpcClient.destroyClient(handle);
      handle = 0;
    }
  }

  /** Plain data holder for the fields submit() needs -- distinct from the full {@link Problem}
   * object model, which does not expose raw CSR arrays. */
  public interface RawProblem {
    int getNumConstraints();

    int getNumVariables();

    boolean isMaximize();

    double getObjectiveOffset();

    double[] getObjectiveCoefficients();

    int[] getRowOffsets();

    int[] getColumnIndices();

    double[] getValues();

    byte[] getRowTypes();

    /** The 'b' / right-hand-side value for each row, matching getRowTypes()'s sense. */
    double[] getConstraintBounds();

    double[] getVariableLowerBounds();

    double[] getVariableUpperBounds();

    byte[] getVariableTypes();
  }

  /** Adapts a {@link Problem} to {@link RawProblem} by reading its existing getters -- no new
   * C++ or JNI, since Problem already exposes the CSR matrix and per-variable/constraint bounds
   * and types that submit() needs. */
  private static final class ProblemAdapter implements RawProblem {
    private final Problem problem;

    ProblemAdapter(Problem problem) {
      this.problem = problem;
    }

    @Override
    public int getNumConstraints() {
      return problem.getNumConstraints();
    }

    @Override
    public int getNumVariables() {
      return problem.getNumVariables();
    }

    @Override
    public boolean isMaximize() {
      return problem.getObjectiveSense() == ObjectiveSense.MAXIMIZE;
    }

    @Override
    public double getObjectiveOffset() {
      return problem.getObjectiveConstant();
    }

    @Override
    public double[] getObjectiveCoefficients() {
      int n = problem.getNumVariables();
      double[] coefficients = new double[n];
      for (int i = 0; i < n; i++) {
        coefficients[i] = problem.getVariable(i).getObjectiveCoefficient();
      }
      return coefficients;
    }

    @Override
    public int[] getRowOffsets() {
      return problem.getConstraintMatrix().getRowOffsets();
    }

    @Override
    public int[] getColumnIndices() {
      return problem.getConstraintMatrix().getColumnIndices();
    }

    @Override
    public double[] getValues() {
      return problem.getConstraintMatrix().getValues();
    }

    @Override
    public byte[] getRowTypes() {
      int n = problem.getNumConstraints();
      byte[] rowTypes = new byte[n];
      for (int i = 0; i < n; i++) {
        rowTypes[i] = problem.getConstraint(i).getSense().nativeValue();
      }
      return rowTypes;
    }

    @Override
    public double[] getConstraintBounds() {
      int n = problem.getNumConstraints();
      double[] bounds = new double[n];
      for (int i = 0; i < n; i++) {
        bounds[i] = problem.getConstraint(i).getRHS();
      }
      return bounds;
    }

    @Override
    public double[] getVariableLowerBounds() {
      int n = problem.getNumVariables();
      double[] bounds = new double[n];
      for (int i = 0; i < n; i++) {
        bounds[i] = problem.getVariable(i).getLowerBound();
      }
      return bounds;
    }

    @Override
    public double[] getVariableUpperBounds() {
      int n = problem.getNumVariables();
      double[] bounds = new double[n];
      for (int i = 0; i < n; i++) {
        bounds[i] = problem.getVariable(i).getUpperBound();
      }
      return bounds;
    }

    @Override
    public byte[] getVariableTypes() {
      int n = problem.getNumVariables();
      byte[] types = new byte[n];
      for (int i = 0; i < n; i++) {
        types[i] = problem.getVariable(i).getVariableType().nativeValue();
      }
      return types;
    }
  }
}
