/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
package com.nvidia.cuopt.linear_programming;

import static com.nvidia.cuopt.cuOpt.*;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.atomic.AtomicInteger;
import org.junit.jupiter.api.Test;

class MipCallbackIT {

    /** Captures one incumbent delivered by the solver. */
    private record Incumbent(double[] solution, double objective, double bound) {}

    /**
     * Same 5-item knapsack as {@link MilpSolverIT}. Registers a Get callback
     * and asserts that at least one incumbent was delivered, that array
     * lengths match the problem dimensions, and that the final objective
     * reported via {@link Problem#objectiveValue()} agrees with the last
     * incumbent's objective.
     */
    @Test
    void get_callback_receives_incumbents() {
        double[] values = {8.0, 4.0, 12.0, 3.0, 9.0};
        double[] weights = {5.0, 3.0, 8.0, 2.0, 6.0};
        double capacity = 15.0;
        int n = values.length;

        List<Incumbent> incumbents = new ArrayList<>();

        try (var problem = new Problem("knapsack-cb");
             var settings = new SolverSettings()
                 .setTimeLimit(30.0)
                 .setRelativeMipGap(0.01)) {

            settings.setMIPGetSolutionCallback((sol, obj, bound) -> {
                // Snapshot — solver may overwrite the underlying buffer after we return.
                incumbents.add(new Incumbent(sol.clone(), obj, bound));
            });

            Variable[] x = new Variable[n];
            for (int i = 0; i < n; i++) {
                x[i] = problem.addVariable(0, 1, INTEGER, "x" + i);
            }
            problem.addConstraint(new LinearExpr().addTerms(weights, x),
                                  LESS_EQUAL, capacity, "capacity");
            problem.setObjective(new LinearExpr().addTerms(values, x), MAXIMIZE);

            problem.solve(settings);

            assertTrue(problem.isMIP());
            assertTrue(!incumbents.isEmpty(),
                "Get callback should have been invoked at least once");

            for (Incumbent inc : incumbents) {
                assertEquals(n, inc.solution().length,
                    "Each incumbent's primal array must match numVariables");
            }

            // The final solver-reported objective should match the best incumbent's.
            double bestSeen = Double.NEGATIVE_INFINITY;
            for (Incumbent inc : incumbents) {
                if (inc.objective() > bestSeen) bestSeen = inc.objective();
            }
            assertEquals(problem.objectiveValue(), bestSeen, 1e-6,
                "Final objectiveValue() should match the best incumbent observed");
        }
    }

    /**
     * Registers a Set callback that injects a known-feasible solution
     * (all-zero, which trivially satisfies the capacity constraint).
     * Confirms the callback was invoked at least once.
     */
    @Test
    void set_callback_is_invoked() {
        double[] values = {8.0, 4.0, 12.0, 3.0, 9.0};
        double[] weights = {5.0, 3.0, 8.0, 2.0, 6.0};
        double capacity = 15.0;
        int n = values.length;

        AtomicInteger setInvocations = new AtomicInteger();

        try (var problem = new Problem("knapsack-set-cb");
             var settings = new SolverSettings()
                 .setTimeLimit(30.0)
                 .setRelativeMipGap(0.01)) {

            settings.setMIPSetSolutionCallback((outSol, outObj, bound) -> {
                setInvocations.incrementAndGet();
                // All-zero is trivially feasible for this knapsack (no items chosen).
                java.util.Arrays.fill(outSol, 0.0);
                outObj[0] = 0.0;
            });

            Variable[] x = new Variable[n];
            for (int i = 0; i < n; i++) {
                x[i] = problem.addVariable(0, 1, INTEGER, "x" + i);
            }
            problem.addConstraint(new LinearExpr().addTerms(weights, x),
                                  LESS_EQUAL, capacity, "capacity");
            problem.setObjective(new LinearExpr().addTerms(values, x), MAXIMIZE);

            problem.solve(settings);

            assertTrue(setInvocations.get() >= 1,
                "Set callback should have been invoked at least once, got " + setInvocations.get());
        }
    }
}
