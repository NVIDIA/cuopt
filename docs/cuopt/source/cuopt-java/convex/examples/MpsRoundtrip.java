/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
import com.nvidia.cuopt.mathematicaloptimization.*;

public class MpsRoundtrip {
  public static void main(String[] args) throws Exception {
    try (Problem problem = Problem.read("sample.mps")) {
      System.out.println("Variables: " + problem.getNumVariables());
      problem.write("roundtrip.mps");
    }
  }
}
