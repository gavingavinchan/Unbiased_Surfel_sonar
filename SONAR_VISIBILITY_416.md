# 416 visibility acceleration

Unchanged model: MODEL_ORACLE_409.md, including 16×16 midpoints, detached visibility,
area_r4, half-open tangent support, exact rational fallback and reciprocal coincidence.
The byte-preserved oracle is fork/gaussian_renderer/ray_depth_oracle_409.py (base
8fc5ed8e43c897a07ecdf84f34a09384c91d4735). No center-admission, material, aperture,
sampling or emission rule changes. The 100 m ambiguity-policy failures remain.

## Conservative angular candidates
Let n=u×v. A blocker support point is C+a U+b V, where
U=(v×n)/|n|², V=(n×u)/|n|², -h0≤a<h0, -h1≤b<h1.
A closed enclosing ball has radius squared h0²|U|²+h1²|V|²+2h0h1|U·V|.
Use the maximum of this radius and the constructed source rectangle radius using
u,v directly, so decoded tangent nonorthogonality does not shrink either support.
Inflate by 4096 eps64 (|C|_1+r+1) max(1,kappa)², kappa=|u||v|/|u×v|.
Nonfinite/degenerate/ill-conditioned (kappa>1e4), origin-containing or horizon
uncertain balls use the whole sphere (no pair rejection). Finite normal-range
arithmetic is the inherited contract; this is not a physical uncertainty bound.

If r<d=|C|, every ray through the ball lies within alpha=asin(r/d) of C/d.
Its chord radius is sqrt(2)*(r/d)/sqrt(1+sqrt(1-(r/d)²)). By the triangle
inequality, overlapping ray supports require |dir_i-dir_j|≤chord_i+chord_j.
Reject only when dir_i·dir_j < 1-(chord_i+chord_j)²/2 - 1e-11.
The outward dot margin additionally encloses float64 dot/normalization/cap
arithmetic, including cancellation as r/d→0. The radius inflation is deliberately
loose compared with the inherited gamma128 predicate envelope. There is no range
or range-cell test; a blocker many bins nearer remains a candidate. Negative rho
blockers may be omitted because the oracle explicitly requires rho>0.

A candidate list is rebuilt on every call from current detached geometry. There
is no stale cache and no geometry-dependent list retained across optimizer steps.
Source chunks of 256 build the list without retaining a dense N² allocation.

## Batched predicates and exact fallback
Evaluate the unchanged homogeneous predicates on sparse directed pairs in batches
of at most 2048×256 sample tests (configurable 1..8192 pairs). Source self ties,
409 symmetric plane coincidence and half-open inequalities are retained. Float64
arithmetic grouping may differ from the oracle's GEMM; uncertain signs use the
same Fraction construction, exact support/depth and exact plane test. Integer
sums count ties/near hits before averaging weights, avoiding reduction-order
changes. SONAR_VISIBILITY_ACCEL=oracle executes the preserved implementation.
SONAR_VISIBILITY_KERNEL=compiled additionally fuses only the floating filter with
torch.compile; this is optional and separately compared, never the acoustic depositor.

Diagnostics for coincidence pairs refer only to retained candidates and are not
identical to the oracle's global all-pairs diagnostic count. Unrelated nonoverlap
planes cannot affect a visibility fraction. The global min parallel gap is not
computed; it remains null, explicitly labelled. No diagnostic may imply material
or moving-shadow gradients that the oracle lacks.

## Bounded execution
Every broad-phase source chunk, pair chunk and 64 exact-point calls checks a
context-local guard, a project-local stop-file (if configured), available host
memory ≥4 GiB and memory PSI avg10≤1. The worker's context guard also checks manual
pause/signals/disk. On a hold, discard the incomplete render, atomically save the
last completed optimizer update and exit, freeing allocations. This is local
restart support, not installation or verification of a native project controller.
Actual unchecked intervals, release and nonzero-step resume are measured separately.
