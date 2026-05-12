/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
package com.nvidia.cuopt.linear_programming;

import com.nvidia.cuopt.CuOpt;
import com.nvidia.cuopt.CuOptException;
import com.nvidia.cuopt.spi.CuOptProvider;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.Objects;

/**
 * Solver tuning settings. {@link AutoCloseable} — owns a native
 * {@code cuOptSolverSettings} handle.
 *
 * <p>Top-frequency parameters have typed setters with IDE autocomplete.
 * Anything else falls through to {@link #setParameter(String, Object)}
 * for forward compatibility with new C-API parameters.
 *
 * <pre>{@code
 *   try (var settings = new SolverSettings()
 *           .setTimeLimit(60)
 *           .setOptimalityTolerance(1e-6)
 *           .setMethod(SolverMethod.PDLP)) {
 *       // ...
 *   }
 * }</pre>
 */
public final class SolverSettings implements AutoCloseable {

    // We hold parameter values in a Java-side map and replay them onto
    // the native handle lazily (just before solve, or eagerly on each
    // setParameter call). For now we replay eagerly via the SPI.
    private final Map<String, Object> parameters = new LinkedHashMap<>();
    private long nativeHandle;     // address of the native cuOptSolverSettings
    private boolean closed = false;

    // MIP user callbacks. The Java 22 FFM impl reads these at solve time
    // and binds them to per-solve Arena-scoped upcall stubs.
    private MIPGetSolutionCallback mipGetCallback;
    private MIPSetSolutionCallback mipSetCallback;

    public SolverSettings() {
        this.nativeHandle = CuOptProvider.instance().createSolverSettings();
    }

    // ── typed setters (top-frequency parameters) ─────────────────

    public SolverSettings setTimeLimit(double seconds) {
        return setFloatParameter(CuOpt.TIME_LIMIT, seconds);
    }

    public SolverSettings setIterationLimit(long iterations) {
        return setIntegerParameter(CuOpt.ITERATION_LIMIT, iterations);
    }

    public SolverSettings setOptimalityTolerance(double eps) {
        // Cuopt's "optimality tolerance" is the relative gap tolerance.
        return setFloatParameter(CuOpt.RELATIVE_GAP_TOLERANCE, eps);
    }

    public SolverSettings setMethod(SolverMethod method) {
        return setIntegerParameter(CuOpt.METHOD, method.code());
    }

    public SolverSettings setPdlpSolverMode(PdlpSolverMode mode) {
        return setIntegerParameter(CuOpt.PDLP_SOLVER_MODE, mode.code());
    }

    public SolverSettings setRelativeMipGap(double gap) {
        return setFloatParameter(CuOpt.MIP_RELATIVE_GAP, gap);
    }

    public SolverSettings setAbsoluteMipGap(double gap) {
        return setFloatParameter(CuOpt.MIP_ABSOLUTE_GAP, gap);
    }

    public SolverSettings setLogToConsole(boolean enabled) {
        return setIntegerParameter(CuOpt.LOG_TO_CONSOLE, enabled ? 1 : 0);
    }

    public SolverSettings setNumCpuThreads(int n) {
        return setIntegerParameter(CuOpt.NUM_CPU_THREADS, n);
    }

    public SolverSettings setRandomSeed(long seed) {
        return setIntegerParameter(CuOpt.RANDOM_SEED, seed);
    }

    // ── MIP user callbacks ───────────────────────────────────────

    /**
     * Registers a callback to receive incumbent MIP solutions. The
     * callback is invoked by the solver each time a new incumbent is
     * found. Note: only meaningful for MIP solves.
     *
     * <p>Pass {@code null} to clear a previously-set callback.
     *
     * @see MIPGetSolutionCallback
     */
    public SolverSettings setMIPGetSolutionCallback(MIPGetSolutionCallback cb) {
        checkOpen();
        this.mipGetCallback = cb;
        return this;
    }

    /**
     * Registers a callback to inject candidate MIP solutions.
     * <b>Registering this callback disables presolve.</b>
     *
     * <p>Pass {@code null} to clear a previously-set callback. Note: once
     * a non-null callback has been registered and used in a solve,
     * clearing it on the same {@code SolverSettings} and reusing for
     * another solve is not supported (the previous native registration
     * persists). Create a fresh {@link SolverSettings} instead.
     *
     * @see MIPSetSolutionCallback
     */
    public SolverSettings setMIPSetSolutionCallback(MIPSetSolutionCallback cb) {
        checkOpen();
        this.mipSetCallback = cb;
        return this;
    }

    /** Returns the registered MIP get-solution callback, or {@code null}. */
    public MIPGetSolutionCallback getMIPGetSolutionCallback() {
        return mipGetCallback;
    }

    /** Returns the registered MIP set-solution callback, or {@code null}. */
    public MIPSetSolutionCallback getMIPSetSolutionCallback() {
        return mipSetCallback;
    }

    // ── generic escape hatch ─────────────────────────────────────

    /**
     * Sets a parameter by name. Dispatches to the appropriate typed C-side
     * setter based on the value's runtime type:
     *
     * <ul>
     *   <li>{@code Integer}, {@code Long}, {@code Boolean}, or an enum
     *       with an {@code int code()} method → integer parameter</li>
     *   <li>{@code Float}, {@code Double} → float parameter</li>
     *   <li>{@code String} or anything else → string parameter
     *       (via {@code Object.toString()})</li>
     * </ul>
     */
    public SolverSettings setParameter(String name, Object value) {
        Objects.requireNonNull(name, "name");
        Objects.requireNonNull(value, "value");
        checkOpen();
        parameters.put(name, value);

        if (value instanceof Boolean b) {
            setIntegerParameter(name, b ? 1 : 0);
        } else if (value instanceof Integer i) {
            setIntegerParameter(name, i.longValue());
        } else if (value instanceof Long l) {
            setIntegerParameter(name, l);
        } else if (value instanceof Float f) {
            setFloatParameter(name, f.doubleValue());
        } else if (value instanceof Double d) {
            setFloatParameter(name, d);
        } else if (value instanceof SolverMethod sm) {
            setIntegerParameter(name, sm.code());
        } else if (value instanceof PdlpSolverMode pm) {
            setIntegerParameter(name, pm.code());
        } else if (value instanceof Enum<?> e) {
            // Best-effort: try to find a code() method; fallback to ordinal.
            try {
                java.lang.reflect.Method m = e.getClass().getMethod("code");
                Object code = m.invoke(e);
                if (code instanceof Integer ci) {
                    setIntegerParameter(name, ci.longValue());
                } else if (code instanceof Long cl) {
                    setIntegerParameter(name, cl);
                } else {
                    setIntegerParameter(name, ((Number) code).longValue());
                }
            } catch (ReflectiveOperationException reflectFail) {
                setIntegerParameter(name, e.ordinal());
            }
        } else {
            setStringParameter(name, value.toString());
        }
        return this;
    }

    public SolverSettings setIntegerParameter(String name, long value) {
        checkOpen();
        parameters.put(name, value);
        CuOptProvider.instance().setSolverIntegerParameter(nativeHandle, name, value);
        return this;
    }

    public SolverSettings setFloatParameter(String name, double value) {
        checkOpen();
        parameters.put(name, value);
        CuOptProvider.instance().setSolverFloatParameter(nativeHandle, name, value);
        return this;
    }

    public SolverSettings setStringParameter(String name, String value) {
        checkOpen();
        parameters.put(name, value);
        CuOptProvider.instance().setSolverStringParameter(nativeHandle, name, value);
        return this;
    }

    /** Returns the last-set value for a parameter, or {@code null} if never set. */
    public Object getParameter(String name) {
        return parameters.get(name);
    }

    // ── internal accessor used by the FFM implementation
    //    (public for cross-package access; not for user code) ────

    public long nativeHandle() {
        checkOpen();
        return nativeHandle;
    }

    private void checkOpen() {
        if (closed) {
            throw new CuOptException("SolverSettings has been closed");
        }
    }

    @Override
    public synchronized void close() {
        if (closed) return;
        closed = true;
        CuOptProvider.instance().destroySolverSettings(nativeHandle);
        nativeHandle = 0L;
    }
}
