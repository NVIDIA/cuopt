/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
package com.nvidia.cuopt;

import com.nvidia.cuopt.optimization.CType;
import com.nvidia.cuopt.optimization.Sense;
import com.nvidia.cuopt.optimization.VType;

/**
 * Constants holder for cuopt-java. Designed for static import:
 *
 * <pre>{@code
 *     import static com.nvidia.cuopt.cuOpt.*;
 *
 *     Variable x = problem.addVariable(0, INF, CONTINUOUS, "x");
 *     problem.addConstraint(expr, LESS_EQUAL, 100, "c1");
 *     problem.setObjective(obj, MAXIMIZE);
 * }</pre>
 *
 * <p>Parameter name string constants (mirrors of the C-side
 * {@code CUOPT_*} preprocessor macros from {@code constants.h}) are
 * grouped at the bottom. They're used with
 * {@link com.nvidia.cuopt.optimization.SolverSettings#setParameter(String, Object)}.
 */
public final class cuOpt {

    private cuOpt() {}

    // ── numeric constants ────────────────────────────────────────
    /** Positive infinity, for unbounded variable / constraint sides. */
    public static final double INF = Double.POSITIVE_INFINITY;
    /** Negative infinity. */
    public static final double NEG_INF = Double.NEGATIVE_INFINITY;

    // ── enum re-exports (static-importable shortcuts) ────────────
    public static final VType CONTINUOUS = VType.CONTINUOUS;
    public static final VType INTEGER = VType.INTEGER;
    public static final VType BINARY = VType.BINARY;

    public static final CType LESS_EQUAL = CType.LE;
    public static final CType GREATER_EQUAL = CType.GE;
    public static final CType EQUAL = CType.EQ;

    public static final Sense MINIMIZE = Sense.MINIMIZE;
    public static final Sense MAXIMIZE = Sense.MAXIMIZE;

    // ── solver parameter name strings (subset; the rest are
    //    available via setParameter(name, value) with raw strings) ─
    public static final String TIME_LIMIT = "time_limit";
    public static final String ITERATION_LIMIT = "iteration_limit";
    public static final String WORK_LIMIT = "work_limit";
    public static final String METHOD = "method";
    public static final String PDLP_SOLVER_MODE = "pdlp_solver_mode";

    public static final String ABSOLUTE_PRIMAL_TOLERANCE = "absolute_primal_tolerance";
    public static final String RELATIVE_PRIMAL_TOLERANCE = "relative_primal_tolerance";
    public static final String ABSOLUTE_DUAL_TOLERANCE = "absolute_dual_tolerance";
    public static final String RELATIVE_DUAL_TOLERANCE = "relative_dual_tolerance";
    public static final String ABSOLUTE_GAP_TOLERANCE = "absolute_gap_tolerance";
    public static final String RELATIVE_GAP_TOLERANCE = "relative_gap_tolerance";

    public static final String MIP_ABSOLUTE_GAP = "mip_absolute_gap";
    public static final String MIP_RELATIVE_GAP = "mip_relative_gap";
    public static final String MIP_INTEGRALITY_TOLERANCE = "mip_integrality_tolerance";

    public static final String LOG_TO_CONSOLE = "log_to_console";
    public static final String LOG_FILE = "log_file";
    public static final String NUM_CPU_THREADS = "num_cpu_threads";
    public static final String NUM_GPUS = "num_gpus";
    public static final String RANDOM_SEED = "random_seed";
    public static final String CROSSOVER = "crossover";
    public static final String PRESOLVE = "presolve";
    public static final String INFEASIBILITY_DETECTION = "infeasibility_detection";
}
