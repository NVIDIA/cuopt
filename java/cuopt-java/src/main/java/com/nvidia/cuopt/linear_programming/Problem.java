/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
package com.nvidia.cuopt.linear_programming;

import com.nvidia.cuopt.CuOptException;
import com.nvidia.cuopt.spi.CuOptProvider;
import com.nvidia.cuopt.spi.SolveResult;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Optional;

/**
 * An LP / MILP / QP optimization problem.
 *
 * <p>{@code Problem} owns all post-solve state. Variables and
 * Constraints are lightweight handles that delegate back to the
 * Problem for value queries. There is no separate {@code Solution}
 * class — primal values are accessed via {@link Variable#value()},
 * duals via {@link Constraint#dualValue()}, and solver-level info
 * (status, objective, stats) via accessors on this class.
 *
 * <pre>{@code
 *   import static com.nvidia.cuopt.CuOpt.*;
 *
 *   try (var problem = new Problem("diet")) {
 *       Variable x = problem.addVariable(0, INF, CONTINUOUS, "x");
 *       Variable y = problem.addVariable(0, INF, CONTINUOUS, "y");
 *
 *       LinearExpr expr = new LinearExpr().addTerm(2, x).addTerm(3, y);
 *       problem.addConstraint(expr, LESS_EQUAL, 100, "c1");
 *
 *       LinearExpr obj = new LinearExpr().addTerm(1, x).addTerm(1, y);
 *       problem.setObjective(obj, MAXIMIZE);
 *
 *       problem.solve();
 *
 *       if (problem.status() == TerminationStatus.OPTIMAL) {
 *           System.out.println("Optimal: " + problem.objectiveValue());
 *           System.out.println("x = " + x.value());
 *           System.out.println("y = " + y.value());
 *       }
 *   }
 * }</pre>
 */
public final class Problem implements AutoCloseable {

    // ── build-time state ─────────────────────────────────────────
    private final String name;
    private final List<Variable> variables = new ArrayList<>();
    private final List<Constraint> constraints = new ArrayList<>();
    private final List<LinearExpr> constraintExpressions = new ArrayList<>();
    private final Map<String, Variable> variablesByName = new HashMap<>();
    private final Map<String, Constraint> constraintsByName = new HashMap<>();

    private LinearExpr linearObjective;
    private QuadraticExpr quadraticObjective;
    private Sense objectiveSense = Sense.MINIMIZE;
    private double objectiveOffset = 0.0;

    // ── post-solve state (populated by solve()) ──────────────────
    private boolean solved = false;
    private double[] primalSolution;
    private double[] dualSolution;
    private double[] reducedCost;
    private double[] slack;
    private TerminationStatus status;
    private String terminationReason;
    private ErrorStatus errorStatus;
    private String errorMessage;
    private double objectiveValue;
    private double dualObjectiveValue;
    private double solveTime;
    private SolverMethod solvedBy;
    private LpStats lpStats;
    private MilpStats milpStats;

    private boolean closed = false;

    public Problem() {
        this("");
    }

    public Problem(String name) {
        this.name = name == null ? "" : name;
    }

    public String name() {
        return name;
    }

    // ── building the problem ─────────────────────────────────────

    public Variable addVariable(double lb, double ub, VType vtype, String name) {
        checkNotSolved();
        int index = variables.size();
        Variable v = new Variable(this, index, lb, ub, Objects.requireNonNull(vtype), name);
        variables.add(v);
        if (name != null && !name.isEmpty()) {
            variablesByName.put(name, v);
        }
        return v;
    }

    public Variable addVariable(double lb, double ub, VType vtype) {
        return addVariable(lb, ub, vtype, "");
    }

    public Variable addVariable(double lb, double ub) {
        return addVariable(lb, ub, VType.CONTINUOUS, "");
    }

    /** Adds a continuous variable with bounds {@code [0, +inf)}. */
    public Variable addVariable() {
        return addVariable(0.0, Double.POSITIVE_INFINITY, VType.CONTINUOUS, "");
    }

    public Constraint addConstraint(LinearExpr lhs, CType sense, double rhs, String name) {
        checkNotSolved();
        Objects.requireNonNull(lhs, "lhs");
        Objects.requireNonNull(sense, "sense");
        if (lhs.owner() != null && lhs.owner() != this) {
            throw new IllegalArgumentException("LinearExpr uses variables from a different Problem");
        }
        int index = constraints.size();
        // The constraint constant becomes part of the RHS: lhs + c <= rhs  ≡  lhs <= rhs - c
        double adjustedRhs = rhs - lhs.constant();
        Constraint c = new Constraint(this, index, sense, adjustedRhs, name);
        constraints.add(c);
        constraintExpressions.add(snapshotLinear(lhs));
        if (name != null && !name.isEmpty()) {
            constraintsByName.put(name, c);
        }
        return c;
    }

    public Constraint addConstraint(LinearExpr lhs, CType sense, double rhs) {
        return addConstraint(lhs, sense, rhs, "");
    }

    public Constraint addConstraint(LinearExpr lhs, CType sense, Variable rhsVar, String name) {
        // lhs <op> rhsVar  ≡  (lhs - rhsVar) <op> 0
        LinearExpr combined = snapshotLinear(lhs);
        combined.addTerm(-1.0, rhsVar);
        return addConstraint(combined, sense, 0.0, name);
    }

    public Constraint addConstraint(LinearExpr lhs, CType sense, LinearExpr rhs, String name) {
        // lhs <op> rhs  ≡  (lhs - rhs) <op> 0
        LinearExpr combined = snapshotLinear(lhs);
        for (Map.Entry<Variable, Double> e : rhs.terms().entrySet()) {
            combined.addTerm(-e.getValue(), e.getKey());
        }
        combined.addConstant(-rhs.constant());
        return addConstraint(combined, sense, 0.0, name);
    }

    public void setObjective(LinearExpr expr, Sense sense) {
        checkNotSolved();
        Objects.requireNonNull(expr, "expr");
        Objects.requireNonNull(sense, "sense");
        if (expr.owner() != null && expr.owner() != this) {
            throw new IllegalArgumentException("LinearExpr uses variables from a different Problem");
        }
        this.linearObjective = snapshotLinear(expr);
        this.quadraticObjective = null;
        this.objectiveSense = sense;
        this.objectiveOffset = expr.constant();
    }

    public void setObjective(QuadraticExpr expr, Sense sense) {
        checkNotSolved();
        Objects.requireNonNull(expr, "expr");
        Objects.requireNonNull(sense, "sense");
        if (expr.owner() != null && expr.owner() != this) {
            throw new IllegalArgumentException("QuadraticExpr uses variables from a different Problem");
        }
        this.quadraticObjective = expr;
        this.linearObjective = null;
        this.objectiveSense = sense;
        this.objectiveOffset = expr.linearPart().constant();
    }

    // ── introspection ────────────────────────────────────────────

    public List<Variable> getVariables() {
        return java.util.Collections.unmodifiableList(variables);
    }

    public List<Constraint> getConstraints() {
        return java.util.Collections.unmodifiableList(constraints);
    }

    public Variable getVariable(int index) {
        return variables.get(index);
    }

    public Variable getVariable(String name) {
        return variablesByName.get(name);
    }

    public Constraint getConstraint(int index) {
        return constraints.get(index);
    }

    public Constraint getConstraint(String name) {
        return constraintsByName.get(name);
    }

    public int numVariables() {
        return variables.size();
    }

    public int numConstraints() {
        return constraints.size();
    }

    public int numNonZeros() {
        int n = 0;
        for (LinearExpr e : constraintExpressions) n += e.numTerms();
        return n;
    }

    public boolean isMip() {
        for (Variable v : variables) {
            if (v.variableType() != VType.CONTINUOUS) return true;
        }
        return false;
    }

    public boolean isQp() {
        return quadraticObjective != null && quadraticObjective.numQuadraticTerms() > 0;
    }

    // ── solve ────────────────────────────────────────────────────

    public void solve() {
        solve(null);
    }

    public void solve(SolverSettings settings) {
        checkOpen();
        SolveResult result = CuOptProvider.instance().solveProblem(this, settings);
        populateFromResult(result);
        this.solved = true;
    }

    // ── post-solve accessors ─────────────────────────────────────

    public TerminationStatus status() {
        checkSolved();
        return status;
    }

    public String terminationReason() {
        checkSolved();
        return terminationReason;
    }

    public ErrorStatus errorStatus() {
        checkSolved();
        return errorStatus;
    }

    public Optional<String> errorMessage() {
        checkSolved();
        return Optional.ofNullable(errorMessage);
    }

    public double objectiveValue() {
        checkSolved();
        return objectiveValue;
    }

    public double dualObjectiveValue() {
        checkSolved();
        return dualObjectiveValue;
    }

    public double solveTime() {
        checkSolved();
        return solveTime;
    }

    public SolverMethod solvedBy() {
        checkSolved();
        return solvedBy;
    }

    public ProblemCategory problemCategory() {
        if (isQp()) return ProblemCategory.QP;
        if (!isMip()) return ProblemCategory.LP;
        // MIP vs IP: IP if all variables are integer
        for (Variable v : variables) {
            if (v.variableType() == VType.CONTINUOUS) return ProblemCategory.MIP;
        }
        return ProblemCategory.IP;
    }

    public LpStats lpStats() {
        checkSolved();
        return lpStats;
    }

    public Optional<MilpStats> milpStats() {
        checkSolved();
        return Optional.ofNullable(milpStats);
    }

    public boolean isSolved() {
        return solved;
    }

    // ── delegation targets used by Variable / Constraint ─────────

    double primalValueOf(int index) {
        checkSolved();
        return primalSolution[index];
    }

    double dualValueOf(int index) {
        checkSolved();
        if (dualSolution == null) return Double.NaN;
        return dualSolution[index];
    }

    double reducedCostOf(int index) {
        checkSolved();
        if (reducedCost == null) return Double.NaN;
        return reducedCost[index];
    }

    double slackOf(int index) {
        checkSolved();
        if (slack == null) return Double.NaN;
        return slack[index];
    }

    // ── lifecycle ────────────────────────────────────────────────

    @Override
    public void close() {
        if (closed) return;
        closed = true;
        // Native handles are owned by the SPI implementation (per-instance Arena
        // inside ProblemImpl). Closing happens there.
    }

    // ── package-private accessors used by ProblemImpl when building
    //    native problem ────────────────────────────────────────────

    LinearExpr linearObjective() {
        return linearObjective;
    }

    QuadraticExpr quadraticObjective() {
        return quadraticObjective;
    }

    Sense objectiveSense() {
        return objectiveSense;
    }

    double objectiveOffset() {
        return objectiveOffset;
    }

    List<LinearExpr> constraintExpressions() {
        return java.util.Collections.unmodifiableList(constraintExpressions);
    }

    // ── helpers ──────────────────────────────────────────────────

    private void checkOpen() {
        if (closed) throw new CuOptException("Problem has been closed");
    }

    private void checkSolved() {
        checkOpen();
        if (!solved) throw new CuOptException("Problem has not been solved");
    }

    private void checkNotSolved() {
        checkOpen();
        if (solved) throw new CuOptException("Problem has already been solved; create a new Problem to re-build");
    }

    private void populateFromResult(SolveResult r) {
        this.primalSolution = r.primalSolution();
        this.dualSolution = r.dualSolution();
        this.reducedCost = r.reducedCost();
        this.slack = r.slack();
        this.status = r.terminationStatus();
        this.terminationReason = r.terminationReason();
        this.errorStatus = r.errorStatus();
        this.errorMessage = r.errorMessage();
        this.objectiveValue = r.objectiveValue();
        this.dualObjectiveValue = r.dualObjectiveValue();
        this.solveTime = r.solveTime();
        this.solvedBy = r.solvedBy();
        this.lpStats = r.lpStats();
        this.milpStats = r.milpStats();
    }

    // LinearExpr copy used at addConstraint time so the user's
    // subsequent mutation of the expression doesn't change the
    // constraint we already accepted.
    private LinearExpr snapshotLinear(LinearExpr src) {
        LinearExpr copy = new LinearExpr();
        for (Map.Entry<Variable, Double> e : src.terms().entrySet()) {
            copy.addTerm(e.getValue(), e.getKey());
        }
        copy.addConstant(src.constant());
        return copy;
    }
}
