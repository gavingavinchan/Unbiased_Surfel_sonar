# One frozen interior family, before derivative scoring

The authoritative protocol is `fork/tests/fixtures/screen_center_176.json`.
Exactly 16 arms: two raw covariances × two mappings × normal/forced LinAlgError
factorization × scalar/batch actual AST producers. Repeat in float32 and float64.
Raw covariances are [[.08,.01],[.01,.12]] and [[.14,-.02],[-.02,.18]].
Raw means are (.25,-.35), (.4,.2). The maps are R(20°)diag(1.1,.9) and
R(-25°)diag(.9,1.1), with translation target minus B times raw mean, yielding
mathematical mapped means (7.25,7.75) and (8.25,8.75). Finite arithmetic producer
outputs are frozen explicitly, rather than assumed exact. Observations are
(7,8),(8,8) and (8,9),(9,9); weights +.7,-.3, opacity .7, image 32×32.

The unchanged conditioner floors the first raw covariance's low eigenvalue at .1.
That is part of this prescribed family, not a reason to replace the covariance.
No covariance derivatives are requested; the full mean chain is nonzero.

Original172 finite differences and tolerances are preserved verbatim: float32
mean .004; row-major matrix [.002,.002,.004,.002,.002,.004,5e-5,5e-5,.0005];
float64 mean1e-5; matrix [1e-6,1e-6,1e-6,1e-6,1e-6,1e-6,1e-8,1e-8,1e-7].
Both matrix and mean norm-relative FD limits are1e-3/1e-6. CPU equation/closed
form vs autograd entry tolerances are atol=rtol2e-6/1e-12; compiled comparator
retains atol3e-5/rtol3e-4, forward2e-6/2e-5. Wrong full-chain sign must fail
the native gradient tolerance in every arm. Exact-zero entries0,1,3,4 remain
visible entry-wise; no unstable entry-relative ratio hides a zero.

`interior_endpoints_176.json` freezes every concrete packed base and all22 +/-
endpoints for all32 dtype/arms before any derivatives. `interior_bounds_176.json`
contains outward-rounded interval bounds from those values, with coordinate
boxes ±2e-5/±2e-12. `scripts/eligibility.py` derives them without autograd.
Every possible clamped tile rectangle is (0,0,1,1), even though affine predicate
and radius may change. If any analytic eligibility condition fails, STOP before
derivative scoring. No family search, step/tolerance tuning, or renderer edit.

172's manifest is preserved byte-for-byte as `screen_center_172_negative.json`;
its saved scores/hash and all32 failed allocation signatures are negative
eligibility evidence, never mandatory derivative successes for this family.
