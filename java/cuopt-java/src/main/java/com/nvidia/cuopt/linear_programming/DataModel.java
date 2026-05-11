/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
package com.nvidia.cuopt.linear_programming;

import com.nvidia.cuopt.CuOptException;
import com.nvidia.cuopt.spi.CuOptProvider;
import com.nvidia.cuopt.spi.SolveResult;
import java.util.Optional;

/**
 * Low-level escape-hatch API matching cuOpt's Python
 * {@code data_model.DataModel}. Use this when you already have
 * CSR-format constraint matrices (e.g., from MPS parsing or
 * programmatic generation) and don't want the {@link Problem} modeling
 * layer.
 *
 * <p>All setters return {@code this} for chaining. Bulk-array
 * arguments are not defensively copied — do not mutate the supplied
 * arrays before {@code Solver.solve(dm)} is called.
 *
 * <pre>{@code
 *   try (var dm = new DataModel()) {
 *       dm.setCsrConstraintMatrix(values, indices, offsets)
 *         .setObjectiveCoefficients(c)
 *         .setVariableLowerBounds(lb)
 *         .setVariableUpperBounds(ub)
 *         .setRowTypes(new CType[]{CType.LE, CType.GE});
 *
 *       Solver.solve(dm);
 *
 *       System.out.println(dm.status());
 *       System.out.println(java.util.Arrays.toString(dm.primalSolution()));
 *   }
 * }</pre>
 */
public final class DataModel implements AutoCloseable {

    // ── problem definition ───────────────────────────────────────
    private boolean maximize = false;
    private double objectiveOffset = 0.0;
    private double[] objectiveCoefficients;
    private double[] constraintMatrixValues;
    private int[] constraintMatrixIndices;
    private int[] constraintMatrixOffsets;
    private double[] quadraticObjectiveValues;
    private int[] quadraticObjectiveIndices;
    private int[] quadraticObjectiveOffsets;
    private double[] variableLowerBounds;
    private double[] variableUpperBounds;
    private double[] constraintLowerBounds;
    private double[] constraintUpperBounds;
    private double[] constraintBounds;       // for one-sided (single rhs)
    private CType[] rowTypes;
    private VType[] variableTypes;
    private String problemName = "";
    private String[] variableNames;
    private String[] rowNames;
    private String objectiveName = "";

    // ── post-solve state ─────────────────────────────────────────
    private boolean solved = false;
    private double[] primalSolution;
    private double[] dualSolution;
    private double[] reducedCost;
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

    public DataModel() {}

    // ── builder-style setters (chainable) ────────────────────────

    public DataModel setMaximize(boolean maximize) {
        this.maximize = maximize;
        return this;
    }

    public DataModel setObjectiveOffset(double offset) {
        this.objectiveOffset = offset;
        return this;
    }

    public DataModel setObjectiveCoefficients(double[] c) {
        this.objectiveCoefficients = c;
        return this;
    }

    public DataModel setCsrConstraintMatrix(double[] values, int[] indices, int[] offsets) {
        this.constraintMatrixValues = values;
        this.constraintMatrixIndices = indices;
        this.constraintMatrixOffsets = offsets;
        return this;
    }

    public DataModel setQuadraticObjectiveMatrix(double[] values, int[] indices, int[] offsets) {
        this.quadraticObjectiveValues = values;
        this.quadraticObjectiveIndices = indices;
        this.quadraticObjectiveOffsets = offsets;
        return this;
    }

    public DataModel setVariableLowerBounds(double[] lb) {
        this.variableLowerBounds = lb;
        return this;
    }

    public DataModel setVariableUpperBounds(double[] ub) {
        this.variableUpperBounds = ub;
        return this;
    }

    public DataModel setConstraintLowerBounds(double[] lb) {
        this.constraintLowerBounds = lb;
        return this;
    }

    public DataModel setConstraintUpperBounds(double[] ub) {
        this.constraintUpperBounds = ub;
        return this;
    }

    /** Sets one-sided rhs (use together with {@link #setRowTypes(CType[])}). */
    public DataModel setConstraintBounds(double[] b) {
        this.constraintBounds = b;
        return this;
    }

    public DataModel setRowTypes(CType[] types) {
        this.rowTypes = types;
        return this;
    }

    public DataModel setVariableTypes(VType[] types) {
        this.variableTypes = types;
        return this;
    }

    public DataModel setProblemName(String name) {
        this.problemName = name == null ? "" : name;
        return this;
    }

    public DataModel setVariableNames(String[] names) {
        this.variableNames = names;
        return this;
    }

    public DataModel setRowNames(String[] names) {
        this.rowNames = names;
        return this;
    }

    public DataModel setObjectiveName(String name) {
        this.objectiveName = name == null ? "" : name;
        return this;
    }

    // ── post-solve accessors ─────────────────────────────────────

    public double[] primalSolution() {
        checkSolved();
        return primalSolution;
    }

    public double[] dualSolution() {
        checkSolved();
        return dualSolution;
    }

    public double[] reducedCost() {
        checkSolved();
        return reducedCost;
    }

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

    // ── solve entry point (used by Solver.solve(dm)) ─────────────

    void solveInternal(SolverSettings settings) {
        if (closed) throw new CuOptException("DataModel has been closed");
        SolveResult r = CuOptProvider.instance().solveDataModel(this, settings);
        this.primalSolution = r.primalSolution();
        this.dualSolution = r.dualSolution();
        this.reducedCost = r.reducedCost();
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
        this.solved = true;
    }

    // ── internal accessors used by the FFM implementation
    //    (public for cross-package access from internal/; not part of
    //    the stable user-facing API — do not rely on these in user code) ─

    public boolean maximize() { return maximize; }
    public double objectiveOffset() { return objectiveOffset; }
    public double[] objectiveCoefficients() { return objectiveCoefficients; }
    public double[] constraintMatrixValues() { return constraintMatrixValues; }
    public int[] constraintMatrixIndices() { return constraintMatrixIndices; }
    public int[] constraintMatrixOffsets() { return constraintMatrixOffsets; }
    public double[] quadraticObjectiveValues() { return quadraticObjectiveValues; }
    public int[] quadraticObjectiveIndices() { return quadraticObjectiveIndices; }
    public int[] quadraticObjectiveOffsets() { return quadraticObjectiveOffsets; }
    public double[] variableLowerBounds() { return variableLowerBounds; }
    public double[] variableUpperBounds() { return variableUpperBounds; }
    public double[] constraintLowerBounds() { return constraintLowerBounds; }
    public double[] constraintUpperBounds() { return constraintUpperBounds; }
    public double[] constraintBoundsArray() { return constraintBounds; }
    public CType[] rowTypes() { return rowTypes; }
    public VType[] variableTypes() { return variableTypes; }
    public String problemName() { return problemName; }

    @Override
    public void close() {
        closed = true;
    }

    private void checkSolved() {
        if (closed) throw new CuOptException("DataModel has been closed");
        if (!solved) throw new CuOptException("DataModel has not been solved");
    }
}
