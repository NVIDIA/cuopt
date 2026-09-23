/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
package com.nvidia.cuopt.mathematicaloptimization;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.Assumptions;

/**
 * End-to-end test against a real cuopt_grpc_server. There is no in-process fixture for it (unlike
 * the Python bindings' grpc_server_fixtures.py), so this needs one started separately:
 *
 * <pre>{@code
 * cpp/build/cuopt_grpc_server --port 5001 &
 * mvn test -Dtest=GrpcClientIntegrationTest -Dcuopt.grpc.test.host=localhost -Dcuopt.grpc.test.port=5001
 * }</pre>
 *
 * Skips (does not fail) when {@code cuopt.grpc.test.port} is unset, which is why it isn't wired
 * into the default `mvn test` run yet -- see java/cuopt/README.md's gRPC client section.
 */
final class GrpcClientIntegrationTest {

  private static void assumeServerConfigured() {
    Assumptions.assumeTrue(
        System.getProperty("cuopt.grpc.test.port") != null,
        "cuopt.grpc.test.port is unset: no gRPC server configured for this run");
  }

  private static GrpcClient newConnectedClient() {
    String host = System.getProperty("cuopt.grpc.test.host", "localhost");
    int port = Integer.parseInt(System.getProperty("cuopt.grpc.test.port"));
    GrpcClient client = new GrpcClient(host, port);
    client.connect();
    return client;
  }

  // Tiny MIP: maximize x + y s.t. x + y <= 4, x, y integer in [0, 4]. Optimal objective is 4.
  private static final class SmallMip implements GrpcClient.RawProblem {
    @Override
    public int getNumConstraints() {
      return 1;
    }

    @Override
    public int getNumVariables() {
      return 2;
    }

    @Override
    public boolean isMaximize() {
      return true;
    }

    @Override
    public double getObjectiveOffset() {
      return 0.0;
    }

    @Override
    public double[] getObjectiveCoefficients() {
      return new double[] {1.0, 1.0};
    }

    @Override
    public int[] getRowOffsets() {
      return new int[] {0, 2};
    }

    @Override
    public int[] getColumnIndices() {
      return new int[] {0, 1};
    }

    @Override
    public double[] getValues() {
      return new double[] {1.0, 1.0};
    }

    @Override
    public byte[] getRowTypes() {
      return new byte[] {'L'};
    }

    @Override
    public double[] getVariableLowerBounds() {
      return new double[] {0.0, 0.0};
    }

    @Override
    public double[] getVariableUpperBounds() {
      return new double[] {4.0, 4.0};
    }

    @Override
    public byte[] getVariableTypes() {
      return new byte[] {'I', 'I'};
    }
  }

  @Test
  void connectAndPingSucceed() {
    NativeTestSupport.assumeNativeLibrary();
    assumeServerConfigured();
    try (GrpcClient client = newConnectedClient()) {
      assertTrue(client.ping(5));
    }
  }

  @Test
  void submitMipAndReadResult() {
    NativeTestSupport.assumeNativeLibrary();
    assumeServerConfigured();
    try (GrpcClient client = newConnectedClient()) {
      String jobId = client.submit(new SmallMip(), /* timeLimitSeconds= */ 30.0);
      GrpcJobStatus status = client.waitForCompletion(jobId, /* timeoutSeconds= */ 60);
      assertEquals(GrpcJobStatus.COMPLETED, status);

      GrpcMipResult result = client.getMipResult(jobId);
      assertEquals(4.0, result.getObjective(), 1e-6);
      assertEquals(2, result.getSolution().length);
    }
  }
}
