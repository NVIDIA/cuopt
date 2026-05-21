#!/usr/bin/env python3
"""Solve dumped augmented KKT system with symmetric LDL factorization (NumPy only)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np


def read_matrix_market(path: Path) -> np.ndarray:
    with path.open(encoding="utf-8") as f:
        line = f.readline()
        while line.startswith("%"):
            line = f.readline()
        m, n, nz = map(int, line.split())
        if m != n:
            raise ValueError(f"Expected square matrix, got {m}x{n}")
        A = np.zeros((m, n), dtype=np.float64)
        for _ in range(nz):
            line = f.readline()
            if not line:
                break
            i, j, v = line.split()
            A[int(i) - 1, int(j) - 1] = float(v)
    return A


def load_rhs(path: Path) -> np.ndarray:
    data = np.loadtxt(path, dtype=np.float64)
    if data.ndim == 0:
        return np.array([data], dtype=np.float64)
    n = int(data[0])
    b = data[1:]
    if b.size != n:
        raise ValueError(f"{path}: length {b.size} != header n={n}")
    return b


def load_vector_file(path: Path) -> np.ndarray:
    return load_rhs(path)


def ldl_factor(A: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Symmetric LDL: A = L D L^T, L unit lower, D diagonal."""
    n = A.shape[0]
    L = np.eye(n, dtype=np.float64)
    D = np.zeros(n, dtype=np.float64)
    for j in range(n):
        if j > 0:
            D[j] = A[j, j] - np.dot(L[j, :j] * D[:j], L[j, :j])
        else:
            D[j] = A[j, j]
        if abs(D[j]) < 1e-30:
            raise np.linalg.LinAlgError(f"zero pivot at j={j}, D[j]={D[j]}")
        for i in range(j + 1, n):
            if j > 0:
                L[i, j] = (A[i, j] - np.dot(L[i, :j] * D[:j], L[j, :j])) / D[j]
            else:
                L[i, j] = A[i, j] / D[j]
    return L, D


def ldl_solve(L: np.ndarray, D: np.ndarray, b: np.ndarray) -> np.ndarray:
    n = b.size
    y = np.empty(n, dtype=np.float64)
    for i in range(n):
        y[i] = b[i] - L[i, :i] @ y[:i]
    z = y / D
    x = np.empty(n, dtype=np.float64)
    for i in range(n - 1, -1, -1):
        x[i] = z[i] - L[i + 1 :, i] @ x[i + 1 :]
    return x


def solve_ldl(A: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, dict]:
    asym = float(np.max(np.abs(A - A.T)))
    A_sym = 0.5 * (A + A.T)
    L, D = ldl_factor(A_sym)
    x = ldl_solve(L, D, b)
    return x, {"n": A.shape[0], "max_asymmetry": asym}


def write_solution(path: Path, x: np.ndarray) -> None:
    with path.open("w", encoding="utf-8") as f:
        f.write(f"{x.size}\n")
        for val in x:
            f.write(f"{val:.16e}\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--matrix",
        type=Path,
        default=Path("cuopt_barrier_aug_initial_matrix.mtx"),
        help="Matrix Market file (e.g. cuopt_barrier_aug_initial_matrix.mtx)",
    )
    parser.add_argument(
        "--rhs",
        type=Path,
        default=Path("cuopt_barrier_aug_initial_rhs.txt"),
        help="Dumped RHS (first line is n, then n values)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("cuopt_barrier_aug_initial_ldl_x.txt"),
        help="Where to write the Python LDL solution",
    )
    parser.add_argument(
        "--reference",
        type=Path,
        default=Path("cuopt_barrier_aug_initial_x.txt"),
        help="cuOpt LDL solution before GMRES IR (from CUOPT_BARRIER_AUG_DUMP)",
    )
    args = parser.parse_args()

    A = read_matrix_market(args.matrix)
    b = load_rhs(args.rhs)
    if b.size != A.shape[0]:
        print(f"Dimension mismatch: A is {A.shape[0]}, b is {b.size}", file=sys.stderr)
        return 1

    x, info = solve_ldl(A, b)
    write_solution(args.out, x)

    r = b - A @ x
    print(f"Matrix: {args.matrix}")
    print(f"RHS:    {args.rhs}")
    print(f"n={info['n']}, max|A-A^T|={info['max_asymmetry']:.3e}")
    print(f"LDL solution written to {args.out}")
    print(f"||Ax - b||_inf = {np.max(np.abs(r)):.6e}")
    print(f"||Ax - b||_2   = {np.linalg.norm(r):.6e}")

    if args.reference.is_file():
        x_ref = load_vector_file(args.reference)
        diff = x - x_ref
        print(f"Reference: {args.reference}")
        print(f"||x_ldl - x_chol||_inf = {np.max(np.abs(diff)):.6e}")
        print(f"||x_ldl - x_chol||_2   = {np.linalg.norm(diff):.6e}")
        r_ref = np.max(np.abs(b - A @ x_ref))
        print(f"||A x_chol - b||_inf (reference residual) = {r_ref:.6e}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
