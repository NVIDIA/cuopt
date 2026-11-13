# Final original-model residual logging

The final summary evaluates the restored solution on the original model, after
solver unscaling and any outer presolve recovery. It follows the HPR postsolve
convention for box violations, RHS normalization, and projected dual residuals.
It is a diagnostic: it does not change returned vectors, API statistics, or status.

The linear/quadratic model is

\[
 \min s\left(c^Tx+\tfrac12x^THx+o\right),\qquad
 l_A\le Ax\le u_A,\quad l\le x\le u.
\]

For LP, \(H=0\). The quadratic setter stores \(H=Q+Q^T\), so the gradient
is \(Hx+c\), without an additional factor of two. Residuals use the coefficient
units of `c` and `H`, consistent with the returned duals; objective factor `s` and
offset `o` are not inserted into these residuals.

## Primal feasibility

Let \(\Pi_{[a,b]}\) denote componentwise projection onto a box. Compute

\[
 v_A=Ax-\Pi_{[l_A,u_A]}(Ax),\qquad v_x=x-\Pi_{[l,u]}(x).
\]

Equivalently, the absolute row violation is
\(\max(0,l_{A,i}-(Ax)_i,(Ax)_i-u_{A,i})\). This handles equalities,
one-sided bounds, and two-sided bounds. Variable violations use the same rule.
The log prints \(\|v_A\|_2\) and \(\|v_x\|_2\) separately.

The conceptual RHS uses one entry per original row:

\[
 b_i=\max\left(
  \begin{cases}|l_{A,i}|&l_{A,i}\text{ finite}\\0&\text{otherwise}\end{cases},
  \begin{cases}|u_{A,i}|&u_{A,i}\text{ finite}\\0&\text{otherwise}\end{cases}\right).
\]

Thus equality endpoints are counted once, and a free row contributes zero.
The HPR-style relative primal residual is

\[
 r_p^{\rm rel}=\frac{\max(\|v_A\|_2,\|v_x\|_2)}{1+\|b\|_2}.
\]

In particular, the two violation norms are not concatenated into a single L2 norm.
Both use the RHS normalization, including the variable-bound term.

## Dual feasibility diagnostic

For minimization, an interval \([a,b]\) admits a positive multiplier only when
its lower endpoint is finite, and a negative multiplier only when its upper
endpoint is finite. Its admissible domain is \(\mathbb R\), \([0,\infty)\),
\(( -\infty,0]\), or \(\{0\}\) for two finite bounds, lower-only, upper-only,
or a free interval, respectively. For maximization these sign domains are reversed.

Project **copies** of the returned multipliers onto those domains, obtaining
\(\hat y,\hat z\). The diagnostic printed as **projected y/z** is

\[
 r_d=Hx+c-A^T\hat y-\hat z,\qquad
 r_d^{\rm rel}=\frac{\|r_d\|_2}{1+\|c\|_2}.
\]

The log also prints

\[
 \sqrt{\|y-\hat y\|_2^2+\|z-\hat z\|_2^2}.
\]

A small projected residual with a large projection distance does not certify the
returned unprojected dual solution. This summary does not check complementarity
or replace an objective-gap test.

## Final status and elapsed time

After recovery and diagnostics, a separate line reports the returned status,
the original-model objective, the solver iteration count, and wall-clock time:

```text
Final Status: Optimal  Objective: 1.50000000e+00  Iterations: 10  Time: 0.300s
```

The numbers above illustrate the format. Each solve measures its own time.
For an `Optimal` result, the displayed objective is independently evaluated as

\[
 P_{\rm log}=s\left(c^T x+\tfrac12 x^T Hx+o\right)
\]

using the restored original-model primal vector. This includes the objective
scale and offset, and uses the original coefficients for either optimization
sense. For every other status, `Objective` is `nan`, even if a feasible iterate
exists. This is a display rule; it does not erase the API objective or change
the termination status. The iteration count comes from the returned solver
statistics; `-1` means the solver did not provide a count.

Time is measured with a separate monotonic timer from entry into the single-GPU
LP/QP solve to the final log line. It includes initialization inside that call,
presolve, solving, unscaling, postsolve, final quality evaluation, and any solution
file output before that line. The model stream is synchronized before sampling
the timer. Input parsing and model construction before the call are excluded,
as are destruction/return overhead after the line. Time is rounded to milliseconds.
The solver's time-limit timer and API `solve_time` field are unchanged.

This line is also printed for presolve-terminal and crossing-bound returns.
Exceptions continue to use the existing error diagnostics. It is separate from
Barrier's existing per-iteration output, whose frequency is unchanged.

## Execution and availability

- `Unscale` identifies completion of solver scaling recovery.
- `Postsolve` identifies restoration of original variables/rows when it runs.
- A model solved entirely by presolve has a postsolve line and a final summary;
  it has no solver-unscaling line because there were no solver iterations.
- The final summary is separate from the solver's earlier iteration statistics.
- Missing dual recovery, wrong vector sizes, or non-finite dual values produce an
  explicit unavailable message for dual statistics; primal violations can still be printed.
- Infeasibility rays are not treated as solution iterates. Quadratic constraints
  require a separate nonlinear evaluator and are explicitly marked unavailable.
- Diagnostics are skipped for internal MIP/batch solves and when all logging is disabled.
  This change covers the single-GPU LP/QP entry points; distributed PDLP reporting
  is unchanged.

The original sparse matrix stays on device for `Ax` and `A^T y_hat`. Only vectors
are copied to the host for final norms. Norms use scaled accumulation (`hypot`),
and non-finite activities cannot become a zero violation through comparisons.
