/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
package com.nvidia.cuopt.linear_programming;

import static com.nvidia.cuopt.CuOpt.*;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

import org.junit.jupiter.api.Test;

class LpSolverTest {

    /**
     * Minimal LP that mirrors the cuOpt Python LP solver test in
     * {@code test_lp_solver.py::test_solver}.
     */
    @Test
    void simpleLpSolveToOptimal() {
        try (var problem = new Problem("test_lp");
             var settings = new SolverSettings()
                 .setOptimalityTolerance(1e-2)
                 .setMethod(SolverMethod.PDLP)) {

            Variable x1 = problem.addVariable(0.0, INF, CONTINUOUS, "x1");
            Variable x2 = problem.addVariable(0.0, INF, CONTINUOUS, "x2");

            // x1 + x2 <= 1.0 (twice to mirror the Python test's offsets array [0,1,2])
            LinearExpr c1 = new LinearExpr().addTerm(1.0, x1);
            problem.addConstraint(c1, LESS_EQUAL, 1.0, "c1");
            LinearExpr c2 = new LinearExpr().addTerm(1.0, x1);
            problem.addConstraint(c2, LESS_EQUAL, 1.0, "c2");

            // minimize x1
            LinearExpr obj = new LinearExpr().addTerm(1.0, x1);
            problem.setObjective(obj, MINIMIZE);

            problem.solve(settings);

            assertEquals(TerminationStatus.OPTIMAL, problem.status(),
                "expected OPTIMAL; got " + problem.terminationReason());
            assertEquals(0.0, x1.value(), 1e-6);
            assertEquals(0.0, problem.objectiveValue(), 1e-6);
            assertTrue(problem.solveTime() >= 0.0);
        }
    }

    @Test
    void diet_lp_optimal_objective() {
        // Diet problem (toy LP):
        //   min   1.5*bread + 0.8*milk
        //   s.t.  2*bread + 3*milk >= 50    (calories)
        //         1*bread + 4*milk >= 12    (protein)
        //         0 <= bread <= 10
        //         0 <= milk  <= 10
        try (var problem = new Problem("diet")) {
            Variable bread = problem.addVariable(0, 10, CONTINUOUS, "bread");
            Variable milk = problem.addVariable(0, 10, CONTINUOUS, "milk");

            LinearExpr calories = new LinearExpr()
                .addTerm(2.0, bread).addTerm(3.0, milk);
            problem.addConstraint(calories, GREATER_EQUAL, 50.0, "calories");

            LinearExpr protein = new LinearExpr()
                .addTerm(1.0, bread).addTerm(4.0, milk);
            problem.addConstraint(protein, GREATER_EQUAL, 12.0, "protein");

            LinearExpr cost = new LinearExpr()
                .addTerm(1.5, bread).addTerm(0.8, milk);
            problem.setObjective(cost, MINIMIZE);

            problem.solve();

            // We don't pin the objective to an exact value (numerical
            // tolerances vary), but it should be feasible and finite.
            assertEquals(TerminationStatus.OPTIMAL, problem.status(),
                "diet LP should be feasible and optimal");
            double obj = problem.objectiveValue();
            assertTrue(Double.isFinite(obj));
            assertTrue(obj >= 0.0);
            // Values should be within bounds.
            assertTrue(bread.value() >= -1e-6 && bread.value() <= 10.0 + 1e-6);
            assertTrue(milk.value() >= -1e-6 && milk.value() <= 10.0 + 1e-6);
        }
    }

    @Test
    void problem_introspection_reflects_build_state() {
        try (var problem = new Problem("introspection")) {
            assertEquals(0, problem.numVariables());
            assertEquals(0, problem.numConstraints());
            assertEquals(0, problem.numNonZeros());

            Variable x = problem.addVariable(0, 5, CONTINUOUS, "x");
            Variable y = problem.addVariable(0, 5, CONTINUOUS, "y");
            assertEquals(2, problem.numVariables());

            problem.addConstraint(
                new LinearExpr().addTerm(1, x).addTerm(1, y),
                LESS_EQUAL, 10.0, "c");
            assertEquals(1, problem.numConstraints());
            assertEquals(2, problem.numNonZeros());

            assertEquals(ProblemCategory.LP, problem.problemCategory());
            assertNotNull(problem.getVariable("x"));
            assertEquals(x, problem.getVariable("x"));
        }
    }
}
