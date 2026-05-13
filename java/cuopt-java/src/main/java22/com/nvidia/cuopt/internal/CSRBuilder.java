/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
package com.nvidia.cuopt.internal;

import com.nvidia.cuopt.linear_programming.LinearExpr;
import com.nvidia.cuopt.linear_programming.QuadraticExpr;
import com.nvidia.cuopt.linear_programming.Variable;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Map;

/**
 * Builds CSR (compressed sparse row) representations from
 * {@link LinearExpr} / {@link QuadraticExpr} for the FFM bridge.
 *
 * <p>Output arrays are ready to copy into native {@code MemorySegment}s.
 */
final class CSRBuilder {

    private CSRBuilder() {}

    /** Output of CSR construction for the linear constraint matrix. */
    static final class CSR {
        final int[] rowOffsets;
        final int[] colIndices;
        final double[] values;

        CSR(int[] rowOffsets, int[] colIndices, double[] values) {
            this.rowOffsets = rowOffsets;
            this.colIndices = colIndices;
            this.values = values;
        }
    }

    /**
     * Builds CSR for the constraint matrix. One row per constraint
     * expression, terms sorted by column (variable index) within each
     * row. Empty constraints produce a row with no non-zeros.
     */
    static CSR buildConstraintCSR(List<LinearExpr> rows) {
        int numRows = rows.size();
        int[] rowOffsets = new int[numRows + 1];

        // First pass: count non-zeros to size the value/column arrays.
        int totalNnz = 0;
        for (LinearExpr e : rows) {
            totalNnz += e.numTerms();
        }
        int[] colIndices = new int[totalNnz];
        double[] values = new double[totalNnz];

        int writeIdx = 0;
        for (int i = 0; i < numRows; i++) {
            rowOffsets[i] = writeIdx;
            LinearExpr row = rows.get(i);
            // Materialize sorted-by-column entries.
            List<int[]> indexBox = new ArrayList<>(row.numTerms());
            List<Double> valueBox = new ArrayList<>(row.numTerms());
            for (Map.Entry<Variable, Double> e : row.terms().entrySet()) {
                indexBox.add(new int[]{e.getKey().index()});
                valueBox.add(e.getValue());
            }
            // Sort by column index ascending.
            Integer[] sortIdx = new Integer[row.numTerms()];
            for (int k = 0; k < sortIdx.length; k++) sortIdx[k] = k;
            int[][] indexFinal = indexBox.toArray(new int[0][]);
            Double[] valueFinal = valueBox.toArray(new Double[0]);
            java.util.Arrays.sort(sortIdx,
                (a, b) -> Integer.compare(indexFinal[a][0], indexFinal[b][0]));

            for (int k = 0; k < sortIdx.length; k++) {
                colIndices[writeIdx] = indexFinal[sortIdx[k]][0];
                values[writeIdx] = valueFinal[sortIdx[k]];
                writeIdx++;
            }
        }
        rowOffsets[numRows] = writeIdx;
        return new CSR(rowOffsets, colIndices, values);
    }

    /**
     * Builds CSR for the quadratic-objective Q matrix, indexed by
     * variable (rows = variables, cols = variables). The user-supplied
     * quadratic terms may be upper-triangular or full; we forward them
     * as given. cuOpt is expected to accept either form and symmetrize
     * internally.
     *
     * <p>Returns CSR with {@code numVariables + 1} row offsets.
     */
    static CSR buildQuadraticCSR(QuadraticExpr qexpr, int numVariables) {
        int n = qexpr.numQuadraticTerms();
        int[] rowOffsets = new int[numVariables + 1];

        // Bucket entries by row (var1.index()), sort each bucket by col.
        List<int[]> entriesByRow = new ArrayList<>(numVariables);
        for (int i = 0; i < numVariables; i++) entriesByRow.add(null);

        // Collect (row, col, val) triples.
        int[] rowsRaw = new int[n];
        int[] colsRaw = new int[n];
        double[] valsRaw = new double[n];
        for (int i = 0; i < n; i++) {
            rowsRaw[i] = qexpr.quadVar1(i).index();
            colsRaw[i] = qexpr.quadVar2(i).index();
            valsRaw[i] = qexpr.quadCoeff(i);
        }

        // Count entries per row.
        int[] rowCount = new int[numVariables];
        for (int i = 0; i < n; i++) rowCount[rowsRaw[i]]++;

        rowOffsets[0] = 0;
        for (int i = 0; i < numVariables; i++) rowOffsets[i + 1] = rowOffsets[i] + rowCount[i];

        int[] colIndices = new int[n];
        double[] values = new double[n];
        int[] cursor = new int[numVariables];
        for (int i = 0; i < n; i++) {
            int r = rowsRaw[i];
            int pos = rowOffsets[r] + cursor[r]++;
            colIndices[pos] = colsRaw[i];
            values[pos] = valsRaw[i];
        }

        // Sort each row by column index.
        for (int r = 0; r < numVariables; r++) {
            int start = rowOffsets[r];
            int end = rowOffsets[r + 1];
            sortRow(colIndices, values, start, end);
        }
        return new CSR(rowOffsets, colIndices, values);
    }

    private static void sortRow(int[] cols, double[] vals, int start, int end) {
        // Simple insertion sort; row sizes in OR matrices are typically small.
        for (int i = start + 1; i < end; i++) {
            int col = cols[i];
            double val = vals[i];
            int j = i - 1;
            while (j >= start && cols[j] > col) {
                cols[j + 1] = cols[j];
                vals[j + 1] = vals[j];
                j--;
            }
            cols[j + 1] = col;
            vals[j + 1] = val;
        }
    }
}
