/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
package com.nvidia.cuopt.optimization;

import static com.nvidia.cuopt.cuOpt.*;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import org.junit.jupiter.api.Test;

class MilpSolverIT {

    @Test
    void simple_knapsack_binary_milp() {
        // 0-1 knapsack:
        //   maximize Σ vᵢ·xᵢ
        //   s.t.     Σ wᵢ·xᵢ <= capacity
        //            xᵢ in {0, 1}
        double[] values = {8.0, 4.0, 12.0, 3.0, 9.0};
        double[] weights = {5.0, 3.0, 8.0, 2.0, 6.0};
        double capacity = 15.0;
        int n = values.length;

        try (var problem = new Problem("knapsack");
             var settings = new SolverSettings()
                 .setTimeLimit(30.0)
                 .setRelativeMipGap(0.01)) {

            Variable[] x = new Variable[n];
            for (int i = 0; i < n; i++) {
                x[i] = problem.addVariable(0, 1, INTEGER, "x" + i);
            }

            LinearExpr cap = new LinearExpr().addTerms(weights, x);
            problem.addConstraint(cap, LESS_EQUAL, capacity, "capacity");

            LinearExpr obj = new LinearExpr().addTerms(values, x);
            problem.setObjective(obj, MAXIMIZE);

            problem.solve(settings);

            assertTrue(problem.isMIP());
            // Termination should be OPTIMAL, FEASIBLE_FOUND, or a limit.
            TerminationStatus s = problem.status();
            assertTrue(s == TerminationStatus.OPTIMAL
                    || s == TerminationStatus.FEASIBLE_FOUND
                    || s == TerminationStatus.PRIMAL_FEASIBLE,
                "MIP should find a feasible solution, got " + s);

            // Values must be near 0 or 1.
            for (int i = 0; i < n; i++) {
                double v = x[i].value();
                assertTrue(v < 0.5 || v > 0.5,
                    "x[" + i + "] should be near 0 or 1, was " + v);
            }

            // MILP stats should be present.
            assertTrue(problem.mipStats().isPresent());
        }
    }
}
