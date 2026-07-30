# Budgeting a solver stage, and attributing a benchmark regression to one

Applies to any bounded stage that consumes a shared time budget — presolve, probing, cut
generation, root LP, heuristics.

## A work counter buys reproducibility, not a time bound

A deterministic work counter (units of effort charged by a cost model) makes a stage's
*coverage* machine-independent. It does not bound how long that coverage takes, because the
work-per-second a stage achieves varies with problem structure. Measure the spread before
assuming otherwise: log realised `work / wall` per run and look at the range. When it spans
orders of magnitude, a single work limit is worth wildly different amounts of time per
instance, and the slow end will consume the whole solve.

Two limits, two jobs — keep both:

- **Work limit** — makes coverage reproducible across machines.
- **Wall ceiling** — keeps the stage from starving whatever runs after it.

## Size the limit to conclude *before* the ceiling

A stage that hits its wall ceiling is the worst case, not the safe one: the effort is spent
and the partial result is discarded. Prefer a limit the stage finishes inside, leaving the
ceiling as a backstop that never fires. Verify by checking that the stage's `hit_limit` /
`timed_out` flag is clear on the instances that used to be truncated.

To bound wall time structurally, divide a scale constant by a **cost proxy** built from
dimensions the stage's inner loop actually touches. Rank candidate proxies by how nearly
`throughput * proxy` is constant across measured runs, and pick the scale so the slowest
measured run still finishes with margin. The proxy will be too noisy to *predict* time; it
only needs to *bound* it, so tune it against the worst case and accept that typical instances
get less than they could use.

## Verify which knob costs, rather than the one that looks like it should

Iteration or round counts read like cost limits and often are not: a stage that converges
before its round cap is not bounded by it at all. The knob that drives cost is usually the one
controlling work *per* round — batch or candidate-set size — and it may sit behind a
third-party default.

Establish whether a cap binds by **running the stage with it removed and diffing the output**,
not by reading a flag. Flags answer narrower questions than they appear to: a `hit_tlim` /
`timed_out` field reports the *time* limit only, so it stays clear on a stage the round cap
stopped early, and reading it as "no limit bound" silently exonerates the wrong knob. Unless a
counter records the cap's own trigger, the A/B is the only sound check.

When a round cap does bind, prefer deleting it over tuning it. It is a poor cost limit in both
directions — where it binds it truncates a stage mid-convergence and costs output, and where it
does not it saves nothing — while the wall ceiling bounds cost directly and only fires when
time is genuinely short.

A knob can also be free on some instances and a genuine quality lever on others. Sweep it and
record both cost and the stage's own output (reduced dimensions, cuts kept) — identical output
at lower cost means the knob is free; changed output means there is a real trade-off, which
must then be settled end-to-end rather than on the stage's own metric.

## More effort in a stage can be worse end-to-end

A stage's own metric improving is not evidence the solve improves. Effort spent there is taken
from everything downstream, so compare final objective and gap, not reduction counts. Watch
for a stage consuming a large share of the budget alongside a missing dual bound (infinite
gap): that pairing means the stage starved the root relaxation.

## Attributing a regression before tuning

Run these checks in order; each can end the investigation:

0. **Is the regression larger than the instance's own noise?** Use the spread between repeats
   of the *same* build as a per-instance noise band and drop everything inside it. This is not
   a formality: on one 240-instance run it removed 18 of 26 apparent regressions, and four of
   the remaining eight turned out to have a bit-identical reduced problem, leaving four real
   ones. Tuning against noise is worse than not tuning, because the change looks justified.
   Instances with wide bands cannot be settled by single runs at all — carry them to the full
   set as competing configs rather than picking a winner from one measurement.
1. **Did the limit bind?** Split instances by the `budget_exhausted` flag and compare the mean
   error delta per group. If the regressed set is not the bound set, the limit is not the
   cause.
2. **Did the stage starve a later one?** Correlate stage wall time as a fraction of the budget
   against instances that lost their dual bound.
3. **Control: remove the limit on the failing instance.** If the failure reproduces with the
   stage fully unbounded, the limit is exonerated — look at whichever stage the time actually
   went to. This is the cheapest way to avoid tuning a limit that was never responsible.

Expect a binding limit to be **bidirectional**: truncating a stage wins on instances where its
effort was wasted and loses where it was load-bearing. A near-zero aggregate delta can hide
large per-instance swings in both directions, so judge by the split, never by the mean alone.

## Log one parseable line per stage

Emit the features the budget was derived from, the budget that came out, and what was actually
spent, on a single greppable line per stage. That makes a sweep regressable offline without
re-deriving anything from the solver, and it is what makes the attribution checks above
one-liners instead of re-runs. Include whether the limit bound and which of several limits
bound, since "the budget was 300 and it spent 302" is the fact that distinguishes a limit that
shaped the stage from one that stopped it.
