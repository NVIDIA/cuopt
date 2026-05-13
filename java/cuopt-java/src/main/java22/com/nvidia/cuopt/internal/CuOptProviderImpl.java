/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
package com.nvidia.cuopt.internal;

import com.nvidia.cuopt.CuOptException;
import com.nvidia.cuopt.internal.panama.cuOptMIPGetSolutionCallback;
import com.nvidia.cuopt.internal.panama.cuOptMIPSetSolutionCallback;
import com.nvidia.cuopt.internal.panama.cuopt_c_h;
import com.nvidia.cuopt.linear_programming.CType;
import com.nvidia.cuopt.linear_programming.Constraint;
import com.nvidia.cuopt.linear_programming.ErrorStatus;
import com.nvidia.cuopt.linear_programming.LinearExpr;
import com.nvidia.cuopt.linear_programming.LpStats;
import com.nvidia.cuopt.linear_programming.MIPGetSolutionCallback;
import com.nvidia.cuopt.linear_programming.MIPSetSolutionCallback;
import com.nvidia.cuopt.linear_programming.MIPStats;
import com.nvidia.cuopt.linear_programming.Problem;
import com.nvidia.cuopt.linear_programming.QuadraticExpr;
import com.nvidia.cuopt.linear_programming.SolverMethod;
import com.nvidia.cuopt.linear_programming.SolverSettings;
import com.nvidia.cuopt.linear_programming.TerminationStatus;
import com.nvidia.cuopt.linear_programming.VType;
import com.nvidia.cuopt.linear_programming.Variable;
import com.nvidia.cuopt.spi.CuOptProvider;
import com.nvidia.cuopt.spi.SolveResult;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.nio.charset.StandardCharsets;
import java.util.List;

/**
 * FFM implementation of {@link CuOptProvider}. Registered via
 * {@code META-INF/services/com.nvidia.cuopt.spi.CuOptProvider}.
 *
 * <p>Lives in the Java 22 multi-release layer; loaded only on JVMs at
 * Java 22 or higher.
 *
 * <p>Native handles cross the SPI boundary as {@code long} (raw
 * addresses); this class reconstructs {@code MemorySegment} from the
 * addresses on each call.
 */
public final class CuOptProviderImpl implements CuOptProvider {

    public CuOptProviderImpl() {
        NativeLibraryLoader.ensureLoaded();
    }

    // ── library-level ────────────────────────────────────────────

    @Override
    public String getVersion() {
        try (Arena arena = Arena.ofConfined()) {
            MemorySegment major = arena.allocate(ValueLayout.JAVA_INT);
            MemorySegment minor = arena.allocate(ValueLayout.JAVA_INT);
            MemorySegment patch = arena.allocate(ValueLayout.JAVA_INT);

            int rc = cuopt_c_h.cuOptGetVersion(major, minor, patch);
            checkRc(rc, "cuOptGetVersion");
            return major.get(ValueLayout.JAVA_INT, 0)
                + "." + minor.get(ValueLayout.JAVA_INT, 0)
                + "." + patch.get(ValueLayout.JAVA_INT, 0);
        }
    }

    // ── SolverSettings lifecycle ─────────────────────────────────

    @Override
    public long createSolverSettings() {
        try (Arena local = Arena.ofConfined()) {
            MemorySegment outPtr = local.allocate(ValueLayout.ADDRESS);
            int rc = cuopt_c_h.cuOptCreateSolverSettings(outPtr);
            checkRc(rc, "cuOptCreateSolverSettings");
            return outPtr.get(ValueLayout.ADDRESS, 0).address();
        }
    }

    @Override
    public void destroySolverSettings(long handle) {
        if (handle == 0L) return;
        try (Arena local = Arena.ofConfined()) {
            MemorySegment ptr = local.allocate(ValueLayout.ADDRESS);
            ptr.set(ValueLayout.ADDRESS, 0, MemorySegment.ofAddress(handle));
            cuopt_c_h.cuOptDestroySolverSettings(ptr);
        }
    }

    @Override
    public void setSolverIntegerParameter(long handle, String name, long value) {
        try (Arena local = Arena.ofConfined()) {
            MemorySegment nameSeg = cString(local, name);
            int rc = cuopt_c_h.cuOptSetIntegerParameter(
                MemorySegment.ofAddress(handle), nameSeg, (int) value);
            checkRc(rc, "cuOptSetIntegerParameter(" + name + ")");
        }
    }

    @Override
    public void setSolverFloatParameter(long handle, String name, double value) {
        try (Arena local = Arena.ofConfined()) {
            MemorySegment nameSeg = cString(local, name);
            int rc = cuopt_c_h.cuOptSetFloatParameter(
                MemorySegment.ofAddress(handle), nameSeg, value);
            checkRc(rc, "cuOptSetFloatParameter(" + name + ")");
        }
    }

    @Override
    public void setSolverStringParameter(long handle, String name, String value) {
        try (Arena local = Arena.ofConfined()) {
            MemorySegment nameSeg = cString(local, name);
            MemorySegment valueSeg = cString(local, value);
            int rc = cuopt_c_h.cuOptSetParameter(
                MemorySegment.ofAddress(handle), nameSeg, valueSeg);
            checkRc(rc, "cuOptSetParameter(" + name + ")");
        }
    }

    // ── solve entry points ───────────────────────────────────────

    @Override
    public SolveResult solveProblem(Problem problem, SolverSettings settings) {
        try (Arena arena = Arena.ofConfined()) {
            ProblemBuild build = buildNativeProblem(arena, problem);
            return solveAndExtract(arena, problem, build, settings);
        }
    }


    // ── native problem construction from Problem (modeling API) ──

    private static final class ProblemBuild {
        long problemHandle;
        int numVariables;
        int numConstraints;

        ProblemBuild(long handle, int numVariables, int numConstraints) {
            this.problemHandle = handle;
            this.numVariables = numVariables;
            this.numConstraints = numConstraints;
        }
    }

    private ProblemBuild buildNativeProblem(Arena arena, Problem problem) {
        int numV = problem.numVariables();
        int numC = problem.numConstraints();

        // Variable bounds + types
        double[] varLb = new double[numV];
        double[] varUb = new double[numV];
        byte[] varTypes = new byte[numV];
        boolean anyBinary = false;
        for (int i = 0; i < numV; i++) {
            Variable v = problem.getVariable(i);
            varLb[i] = v.lowerBound();
            varUb[i] = v.upperBound();
            varTypes[i] = (byte) v.variableType().code();
            if (v.variableType() == VType.BINARY) anyBinary = true;
        }
        // BINARY is INTEGER with bounds clamped to [0, 1]. The user may
        // already have set these bounds, but enforce here too.
        if (anyBinary) {
            for (int i = 0; i < numV; i++) {
                if (problem.getVariable(i).variableType() == VType.BINARY) {
                    if (Double.isNaN(varLb[i]) || varLb[i] < 0.0) varLb[i] = 0.0;
                    if (Double.isNaN(varUb[i]) || varUb[i] > 1.0) varUb[i] = 1.0;
                }
            }
        }

        // Build constraint CSR from per-constraint LinearExprs.
        List<LinearExpr> rows = problem.constraintExpressions();
        CSRBuilder.CSR ccsr = CSRBuilder.buildConstraintCSR(rows);

        // Constraint sense + rhs from the Constraint metadata.
        byte[] sense = new byte[numC];
        double[] rhs = new double[numC];
        for (int i = 0; i < numC; i++) {
            Constraint c = problem.getConstraint(i);
            sense[i] = (byte) c.sense().code();
            rhs[i] = c.rhs();
        }

        // Objective coefficients from linear / linear-part-of-quadratic.
        double[] objCoeffs = new double[numV];
        LinearExpr objLinear = problem.linearObjective() != null
            ? problem.linearObjective()
            : problem.quadraticObjective() != null
                ? problem.quadraticObjective().linearPart()
                : null;
        if (objLinear != null) {
            for (var e : objLinear.terms().entrySet()) {
                objCoeffs[e.getKey().index()] += e.getValue();
            }
        }

        // Allocate and copy into native segments.
        MemorySegment objCoeffsSeg = doubleArray(arena, objCoeffs);
        MemorySegment rowOffsetsSeg = intArray(arena, ccsr.rowOffsets);
        MemorySegment colIndicesSeg = intArray(arena, ccsr.colIndices);
        MemorySegment ccoeffsSeg = doubleArray(arena, ccsr.values);
        MemorySegment senseSeg = byteArray(arena, sense);
        MemorySegment rhsSeg = doubleArray(arena, rhs);
        MemorySegment lbSeg = doubleArray(arena, varLb);
        MemorySegment ubSeg = doubleArray(arena, varUb);
        MemorySegment varTypesSeg = byteArray(arena, varTypes);
        MemorySegment problemPtr = arena.allocate(ValueLayout.ADDRESS);

        int sense_code = problem.objectiveSense().code();
        double offset = problem.objectiveOffset();

        int rc;
        if (problem.isQP() && problem.quadraticObjective() != null) {
            QuadraticExpr q = problem.quadraticObjective();
            CSRBuilder.CSR qcsr = CSRBuilder.buildQuadraticCSR(q, numV);
            MemorySegment qRowOff = intArray(arena, qcsr.rowOffsets);
            MemorySegment qColIdx = intArray(arena, qcsr.colIndices);
            MemorySegment qVals = doubleArray(arena, qcsr.values);
            rc = cuopt_c_h.cuOptCreateQuadraticProblem(
                numC, numV, sense_code, offset, objCoeffsSeg,
                qRowOff, qColIdx, qVals,
                rowOffsetsSeg, colIndicesSeg, ccoeffsSeg,
                senseSeg, rhsSeg, lbSeg, ubSeg, problemPtr);
            checkRc(rc, "cuOptCreateQuadraticProblem");
        } else {
            rc = cuopt_c_h.cuOptCreateProblem(
                numC, numV, sense_code, offset, objCoeffsSeg,
                rowOffsetsSeg, colIndicesSeg, ccoeffsSeg,
                senseSeg, rhsSeg, lbSeg, ubSeg, varTypesSeg, problemPtr);
            checkRc(rc, "cuOptCreateProblem");
        }

        long handle = problemPtr.get(ValueLayout.ADDRESS, 0).address();
        return new ProblemBuild(handle, numV, numC);
    }


    // ── solve + extract ─────────────────────────────────────────

    private SolveResult solveAndExtract(Arena arena, Problem problem,
                                        ProblemBuild build, SolverSettings settings) {
        return solveAndExtract(arena, build, settings, build.numConstraints, build.numVariables,
            problem);
    }


    private SolveResult solveAndExtract(Arena arena, ProblemBuild build,
                                        SolverSettings settings, int numC, int numV,
                                        Problem problemForSlack) {
        long settingsHandle = settings != null ? settings.nativeHandle() : createSolverSettings();
        boolean ownSettingsHandle = (settings == null);
        try {
            registerMipCallbacks(settings, settingsHandle, numV, arena);

            MemorySegment solPtr = arena.allocate(ValueLayout.ADDRESS);
            int rc = cuopt_c_h.cuOptSolve(
                MemorySegment.ofAddress(build.problemHandle),
                MemorySegment.ofAddress(settingsHandle),
                solPtr);
            checkRc(rc, "cuOptSolve");
            MemorySegment solutionHandle = solPtr.get(ValueLayout.ADDRESS, 0);

            try {
                // Termination + error
                MemorySegment outInt = arena.allocate(ValueLayout.JAVA_INT);
                cuopt_c_h.cuOptGetTerminationStatus(solutionHandle, outInt);
                TerminationStatus term = TerminationStatus.fromCode(outInt.get(ValueLayout.JAVA_INT, 0));

                cuopt_c_h.cuOptGetErrorStatus(solutionHandle, outInt);
                ErrorStatus err = ErrorStatus.fromCode(outInt.get(ValueLayout.JAVA_INT, 0));

                String errMsg = null;
                if (err != ErrorStatus.SUCCESS) {
                    MemorySegment buf = arena.allocate(1024);
                    cuopt_c_h.cuOptGetErrorString(solutionHandle, buf, 1024);
                    errMsg = buf.getString(0);
                    if (errMsg != null && errMsg.isEmpty()) errMsg = null;
                }

                // Primal / dual / reduced cost
                double[] primal = new double[numV];
                if (numV > 0) {
                    MemorySegment primalSeg = arena.allocate(
                        ValueLayout.JAVA_DOUBLE.byteSize() * numV);
                    cuopt_c_h.cuOptGetPrimalSolution(solutionHandle, primalSeg);
                    MemorySegment.copy(primalSeg, ValueLayout.JAVA_DOUBLE, 0, primal, 0, numV);
                }

                double[] dual = null;
                double[] reduced = null;
                // cuOptIsMIP determines whether dual is meaningful.
                MemorySegment isMipPtr = arena.allocate(ValueLayout.JAVA_INT);
                cuopt_c_h.cuOptIsMIP(MemorySegment.ofAddress(build.problemHandle), isMipPtr);
                boolean isMip = isMipPtr.get(ValueLayout.JAVA_INT, 0) != 0;

                if (!isMip && numC > 0) {
                    dual = new double[numC];
                    MemorySegment dualSeg = arena.allocate(
                        ValueLayout.JAVA_DOUBLE.byteSize() * numC);
                    cuopt_c_h.cuOptGetDualSolution(solutionHandle, dualSeg);
                    MemorySegment.copy(dualSeg, ValueLayout.JAVA_DOUBLE, 0, dual, 0, numC);
                }

                if (!isMip && numV > 0) {
                    reduced = new double[numV];
                    MemorySegment rcSeg = arena.allocate(
                        ValueLayout.JAVA_DOUBLE.byteSize() * numV);
                    cuopt_c_h.cuOptGetReducedCosts(solutionHandle, rcSeg);
                    MemorySegment.copy(rcSeg, ValueLayout.JAVA_DOUBLE, 0, reduced, 0, numV);
                }

                // Slack (computed Java-side from primal + constraint expressions)
                double[] slack = (problemForSlack != null && numC > 0)
                    ? computeSlackFromExpressions(problemForSlack, primal) : null;

                // Objective values + time
                MemorySegment outDbl = arena.allocate(ValueLayout.JAVA_DOUBLE);
                cuopt_c_h.cuOptGetObjectiveValue(solutionHandle, outDbl);
                double objVal = outDbl.get(ValueLayout.JAVA_DOUBLE, 0);

                double dualObjVal = Double.NaN;
                if (!isMip) {
                    cuopt_c_h.cuOptGetDualObjectiveValue(solutionHandle, outDbl);
                    dualObjVal = outDbl.get(ValueLayout.JAVA_DOUBLE, 0);
                }

                cuopt_c_h.cuOptGetSolveTime(solutionHandle, outDbl);
                double solveTime = outDbl.get(ValueLayout.JAVA_DOUBLE, 0);

                // Stats — LP and MIP. We populate what's readily available;
                // missing fields are NaN / -1.
                LpStats lpStats = new LpStats(
                    Double.NaN, Double.NaN, objVal, dualObjVal, -1L, Double.NaN);

                MIPStats mipStats = null;
                if (isMip) {
                    cuopt_c_h.cuOptGetMIPGap(solutionHandle, outDbl);
                    double mipGap = outDbl.get(ValueLayout.JAVA_DOUBLE, 0);
                    cuopt_c_h.cuOptGetSolutionBound(solutionHandle, outDbl);
                    double bound = outDbl.get(ValueLayout.JAVA_DOUBLE, 0);
                    mipStats = new MIPStats(mipGap, bound, -1L, -1L, Double.NaN, Double.NaN);
                }

                return new SolveResult(
                    primal, dual, reduced, slack,
                    term, term.name(), err, errMsg,
                    objVal, dualObjVal, solveTime,
                    SolverMethod.UNSET,
                    lpStats, mipStats);
            } finally {
                MemorySegment solDestroyPtr = arena.allocate(ValueLayout.ADDRESS);
                solDestroyPtr.set(ValueLayout.ADDRESS, 0, solutionHandle);
                cuopt_c_h.cuOptDestroySolution(solDestroyPtr);
            }
        } finally {
            // Destroy the problem handle we built.
            MemorySegment probDestroyPtr = arena.allocate(ValueLayout.ADDRESS);
            probDestroyPtr.set(ValueLayout.ADDRESS, 0,
                MemorySegment.ofAddress(build.problemHandle));
            cuopt_c_h.cuOptDestroyProblem(probDestroyPtr);
            if (ownSettingsHandle) destroySolverSettings(settingsHandle);
        }
    }

    // ── helpers ──────────────────────────────────────────────────

    /**
     * Wires MIP user callbacks from {@code settings} (if any) to the native
     * solver via FFM upcall stubs allocated in {@code arena}.
     *
     * <p>The Get callback is always registered when {@code settings} is non-null —
     * the trampoline reads {@code settings.getMIPGetSolutionCallback()} at
     * invocation time and no-ops if it's null. This is safe: registering a
     * Get callback has no solver side effects.
     *
     * <p>The Set callback is registered only when the user has provided one,
     * because registering it disables presolve (per the C API contract). If
     * the user clears the Set callback after a previous solve registered one,
     * the stale native registration persists; users should construct a new
     * {@link SolverSettings} instead.
     */
    private static void registerMipCallbacks(
            SolverSettings settings, long settingsHandle, int numV, Arena arena) {
        if (settings == null) return;
        final int finalNumV = numV;
        final SolverSettings finalSettings = settings;

        cuOptMIPGetSolutionCallback.Function getTrampoline = (solPtr, objPtr, boundPtr, userData) -> {
            MIPGetSolutionCallback cb = finalSettings.getMIPGetSolutionCallback();
            if (cb == null) return;
            try {
                double[] solution = (finalNumV > 0)
                    ? solPtr.reinterpret(ValueLayout.JAVA_DOUBLE.byteSize() * finalNumV)
                          .toArray(ValueLayout.JAVA_DOUBLE)
                    : new double[0];
                double objVal = objPtr.reinterpret(ValueLayout.JAVA_DOUBLE.byteSize())
                                      .get(ValueLayout.JAVA_DOUBLE, 0);
                double boundVal = boundPtr.reinterpret(ValueLayout.JAVA_DOUBLE.byteSize())
                                          .get(ValueLayout.JAVA_DOUBLE, 0);
                cb.onSolution(solution, objVal, boundVal);
            } catch (Throwable ignore) {
                // Never propagate exceptions across the FFM upcall boundary.
            }
        };
        MemorySegment getStub = cuOptMIPGetSolutionCallback.allocate(getTrampoline, arena);
        int rcGet = cuopt_c_h.cuOptSetMIPGetSolutionCallback(
            MemorySegment.ofAddress(settingsHandle), getStub, MemorySegment.NULL);
        checkRc(rcGet, "cuOptSetMIPGetSolutionCallback");

        MIPSetSolutionCallback setCb = settings.getMIPSetSolutionCallback();
        if (setCb != null) {
            final MIPSetSolutionCallback finalSetCb = setCb;
            cuOptMIPSetSolutionCallback.Function setTrampoline = (solPtr, objPtr, boundPtr, userData) -> {
                try {
                    double[] outSolution = new double[finalNumV];
                    double[] outObjective = new double[1];
                    double boundVal = boundPtr.reinterpret(ValueLayout.JAVA_DOUBLE.byteSize())
                                              .get(ValueLayout.JAVA_DOUBLE, 0);
                    finalSetCb.provideSolution(outSolution, outObjective, boundVal);
                    if (finalNumV > 0) {
                        MemorySegment.copy(outSolution, 0,
                            solPtr.reinterpret(ValueLayout.JAVA_DOUBLE.byteSize() * finalNumV),
                            ValueLayout.JAVA_DOUBLE, 0, finalNumV);
                    }
                    objPtr.reinterpret(ValueLayout.JAVA_DOUBLE.byteSize())
                          .set(ValueLayout.JAVA_DOUBLE, 0, outObjective[0]);
                } catch (Throwable ignore) {
                    // Never propagate exceptions across the FFM upcall boundary.
                }
            };
            MemorySegment setStub = cuOptMIPSetSolutionCallback.allocate(setTrampoline, arena);
            int rcSet = cuopt_c_h.cuOptSetMIPSetSolutionCallback(
                MemorySegment.ofAddress(settingsHandle), setStub, MemorySegment.NULL);
            checkRc(rcSet, "cuOptSetMIPSetSolutionCallback");
        }
    }

    private static double[] computeSlackFromExpressions(Problem p, double[] primal) {
        List<LinearExpr> rows = p.constraintExpressions();
        double[] slack = new double[rows.size()];
        for (int i = 0; i < rows.size(); i++) {
            double lhs = 0.0;
            for (var e : rows.get(i).terms().entrySet()) {
                lhs += primal[e.getKey().index()] * e.getValue();
            }
            Constraint c = p.getConstraint(i);
            // Slack convention: rhs - lhs (positive for satisfied <= constraints).
            slack[i] = c.rhs() - lhs;
        }
        return slack;
    }

    private static MemorySegment cString(Arena arena, String s) {
        if (s == null) return MemorySegment.NULL;
        byte[] bytes = s.getBytes(StandardCharsets.UTF_8);
        MemorySegment seg = arena.allocate(bytes.length + 1);
        MemorySegment.copy(bytes, 0, seg, ValueLayout.JAVA_BYTE, 0, bytes.length);
        seg.set(ValueLayout.JAVA_BYTE, bytes.length, (byte) 0);
        return seg;
    }

    private static MemorySegment doubleArray(Arena arena, double[] arr) {
        if (arr == null || arr.length == 0) return MemorySegment.NULL;
        MemorySegment seg = arena.allocate(ValueLayout.JAVA_DOUBLE.byteSize() * arr.length);
        MemorySegment.copy(arr, 0, seg, ValueLayout.JAVA_DOUBLE, 0, arr.length);
        return seg;
    }

    private static MemorySegment intArray(Arena arena, int[] arr) {
        if (arr == null || arr.length == 0) return MemorySegment.NULL;
        MemorySegment seg = arena.allocate(ValueLayout.JAVA_INT.byteSize() * arr.length);
        MemorySegment.copy(arr, 0, seg, ValueLayout.JAVA_INT, 0, arr.length);
        return seg;
    }

    private static MemorySegment byteArray(Arena arena, byte[] arr) {
        if (arr == null || arr.length == 0) return MemorySegment.NULL;
        MemorySegment seg = arena.allocate(arr.length);
        MemorySegment.copy(arr, 0, seg, ValueLayout.JAVA_BYTE, 0, arr.length);
        return seg;
    }

    private static byte[] ctypesToBytes(CType[] types, int n) {
        byte[] out = new byte[n];
        if (types == null) {
            java.util.Arrays.fill(out, (byte) 'L');
        } else {
            for (int i = 0; i < n; i++) out[i] = (byte) types[i].code();
        }
        return out;
    }


    private static void checkRc(int rc, String op) {
        if (rc != 0) {
            ErrorStatus err = ErrorStatus.fromCode(rc);
            throw new CuOptException(op + " returned non-zero status " + rc + " (" + err + ")");
        }
    }
}
