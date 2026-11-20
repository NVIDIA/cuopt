/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
package com.nvidia.cuopt.mathematicaloptimization;

/**
 * Runnable demo, not a JUnit test: submits the same small MIP two ways -- built directly via the
 * regular Java API, and read from an MPS file written to disk -- against a real cuopt_grpc_server.
 *
 * <pre>{@code
 * cpp/build/cuopt_grpc_server --port 5001 &
 * mvn -q test-compile
 * mvn -q exec:java -Dexec.mainClass=com.nvidia.cuopt.mathematicaloptimization.GrpcClientExample \
 *     -Dexec.classpathScope=test -Dcuopt.native.dir=<path to libcuopt_jni.so's directory> \
 *     -Dexec.args="localhost 5001"
 * }</pre>
 */
public final class GrpcClientExample {
  private GrpcClientExample() {}

  public static void main(String[] args) throws Exception {
    if (args.length != 2) {
      System.err.println("Usage: GrpcClientExample <host> <port>");
      System.exit(2);
    }
    String host = args[0];
    int port = Integer.parseInt(args[1]);

    submitBuiltViaApi(host, port);
    submitReadFromMpsFile(host, port);
  }

  // maximize x + y  s.t.  x + y <= 4,  x, y integer in [0, 4]. Optimal objective is 4.
  private static Problem smallMip(String name) {
    Problem problem = new Problem(name);
    Variable x = problem.addVariable(0.0, 4.0, 1.0, VariableType.INTEGER, "x");
    Variable y = problem.addVariable(0.0, 4.0, 1.0, VariableType.INTEGER, "y");
    problem.addConstraint(LinearExpression.of(x).plus(y).le(4.0), "c0");
    problem.setObjective(LinearExpression.of(x).plus(y), ObjectiveSense.MAXIMIZE);
    return problem;
  }

  // Does not close 'problem' -- ownership stays with the caller, matching Problem's own
  // AutoCloseable contract.
  private static void submitAndPrint(String label, String host, int port, Problem problem)
      throws InterruptedException {
    System.out.println("=== " + label + " ===");
    try (GrpcClient client = new GrpcClient(host, port)) {
      client.connect();
      String jobId = client.submit(problem, /* timeLimitSeconds= */ 30.0);
      System.out.println("submitted job " + jobId);

      GrpcJobStatus status = client.waitForCompletion(jobId, /* timeoutSeconds= */ 60);
      System.out.println("status: " + status);

      GrpcMipResult result = client.getMipResult(jobId);
      System.out.println("objective: " + result.getObjective());
      double[] solution = result.getSolution();
      System.out.println("x=" + solution[0] + " y=" + solution[1]);
    }
  }

  private static void submitBuiltViaApi(String host, int port) throws InterruptedException {
    try (Problem problem = smallMip("api-built")) {
      submitAndPrint("built via the regular Java API", host, port, problem);
    }
  }

  private static void submitReadFromMpsFile(String host, int port) throws Exception {
    java.nio.file.Path file = java.nio.file.Files.createTempFile("cuopt-grpc-example-", ".mps");
    try {
      try (Problem source = smallMip("mps-source")) {
        source.write(file.toString());
      }
      System.out.println("wrote " + file);

      try (Problem problem = Problem.read(file.toString())) {
        submitAndPrint("read from MPS file", host, port, problem);
      }
    } finally {
      java.nio.file.Files.deleteIfExists(file);
    }
  }
}
