# Frozen interior screen-center result


## 2026-10-20 — Interior-tile screen-chain fixtures (Codex)
Exact PR6 unchanged renderer; frozen16 arms in each precision before scoring.
Every arm passes conservative interval eligibility and actual tile stability.
Original float32 matrix FD criterion fails12/16 (max0.001978794578462839>0.001),
float64 passes16/16 (max1.9571069575453928e-8); both mean chains pass16/16.
Independent quotient VJP agrees with autograd and CPU native equations; wrong
full-chain sign rejected16/16 per precision. Preserved172 negative tile signatures.
Strict CPU372pass/12fail/15skip, all318 inherited passes preserved. This branch
is negative numerical validation, not native acceptance. No renderer/backward/
conditioning/footprint or mandatory native method edits, no successor staged.
Frozen input/bounds commit fcb2c6f946f11261985e782bb33181237a89eab6 precedes scores.

Full reproducible evidence is in sonar-recon/experiments/176-surfel-interior-tile-gradient-fixtures. Native coverage remains unexecuted.
