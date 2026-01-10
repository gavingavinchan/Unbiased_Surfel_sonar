# Design Decisions Log

This document tracks design decisions, the reasoning behind them, and alternatives considered.

---

## Decision 001: Scale Factor Learning Strategy

**Date**: 2025-01-09

**Status**: Proposed

### Problem Statement

The scale factor and surfel positions are **coupled** - both can adjust to minimize reconstruction loss. With a single frame (or insufficient viewpoint diversity), this creates an **identifiability problem** similar to credit assignment in reinforcement learning.

Specifically:
- Scale factor `s` affects where the sonar origin is positioned
- Surfel positions `P` determine what gets rendered
- For any scale `s`, surfels can adjust positions to compensate
- The loss can be minimized without learning the "true" scale

This coupling means joint optimization may not converge to the correct scale.

### Selected Approach: Curriculum Learning (Scale-First)

**Strategy**:
```
Stage 1: Fix surfels, learn scale factor only
Stage 2: Fix scale factor, learn surfels
Stage 3: (Optional) Joint fine-tuning with reduced scale LR
```

**Reasoning**:
1. Learning scale first with fixed surfels forces the scale to explain the range discrepancy
2. Initial surfels (from backward projection) provide a reasonable geometry estimate
3. Once scale is stable, surfels can refine to match the correctly-scaled projection
4. Avoids the degenerate solution where surfels drift to compensate for wrong scale

**Why not fix surfels first?**
- Initial surfels from backward projection use **unscaled** poses
- If we fix these "wrong" positions and try to learn scale, the scale would need to compensate for initialization errors
- Better to let scale converge first, then let surfels adjust to the correct scale

### Alternative Approaches Considered

| Approach | Description | Why Not Selected |
|----------|-------------|------------------|
| **Joint optimization** | Learn both simultaneously with same LR | Identifiability problem; scale may not converge |
| **Timescale separation** | Different learning rates (scale slower) | Still coupled; may need very careful LR tuning |
| **Multi-frame constraint** | Use many frames to provide geometric constraint | Good complement but doesn't solve single-frame debugging |
| **Range distribution matching** | Match mean/median range statistics | Indirect signal; may not be strong enough |
| **Scale regularization/prior** | Soft constraint toward expected scale | Requires knowing expected scale a priori |
| **Anchor constraints** | Known real-world distances | Requires manual measurement; not always available |
| **Fix surfels first, then scale** | Opposite order curriculum | Surfels initialized with wrong scale; scale would learn to compensate for bad init |

### Implementation Plan

```python
# Stage 1: Learn scale only (N iterations)
for iter in range(scale_only_iters):
    with torch.no_grad():
        # Freeze surfel gradients
        gaussians.freeze()

    render_pkg = render_sonar(..., scale_factor=scale_factor)
    loss = compute_loss(render_pkg, gt)
    loss.backward()
    scale_optimizer.step()

# Stage 2: Learn surfels only (M iterations)
scale_factor.freeze()  # or set requires_grad=False
for iter in range(surfel_only_iters):
    render_pkg = render_sonar(..., scale_factor=scale_factor)
    loss = compute_loss(render_pkg, gt)
    loss.backward()
    gaussians.optimizer.step()

# Stage 3 (optional): Joint fine-tuning
scale_factor.unfreeze()  # reduced LR
for iter in range(finetune_iters):
    # Both update, scale with lower LR
    ...
```

### Open Questions

1. How many iterations for Stage 1 vs Stage 2?
2. Should Stage 3 joint fine-tuning be included?
3. What's the convergence criterion for Stage 1 (scale stability)?

### References

- Related to actor-critic learning rate separation in RL
- Similar to coordinate descent optimization

---

## Decision 002: [Template for Future Decisions]

**Date**: YYYY-MM-DD

**Status**: Proposed / Accepted / Superseded

### Problem Statement

[Describe the problem or question that needs a decision]

### Selected Approach

[Describe the chosen solution]

**Reasoning**:
[Why this approach was selected]

### Alternative Approaches Considered

| Approach | Description | Why Not Selected |
|----------|-------------|------------------|
| ... | ... | ... |

### Implementation Plan

[Code snippets or steps]

### Open Questions

[Remaining uncertainties]

---

*Document maintained as part of the Sonar 2DGS project*
