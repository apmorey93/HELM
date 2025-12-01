# Expected Results for HELM Experiments

This document defines success criteria for all three experiments. Use these targets to validate that your implementation is working correctly and producing meaningful results.

---

## Experiment 1: Spectral Cliff

### Primary Metric: Jacobian Frobenius Norm vs Horizon

**Success Pattern:**
- **Flat JEPA**: Exponential growth (linear in log-space)
  - T=5 → ||J||_F ≈ 4-8
  - T=20 → ||J||_F ≈ 50-150
  - T=50 → ||J||_F ≈ 10³-10⁵ (explosion!)
  - T=100 → ||J||_F > 10⁶ or NaN
  
- **H-JEPA**: Saturates to bounded value
  - T=5 → ||J||_F ≈ 3-5
  - T=20 → ||J||_F ≈ 8-15
  - T=50 → ||J||_F ≈ 12-25 (saturation)
  - T=100 → ||J||_F ≈ 15-30 (stays bounded!)

**Success Criteria:**
✅ At T=100, H-JEPA norm is < 50 while Flat > 1000 (20x+ difference)
✅ Flat's log(norm) grows linearly with T
✅ H-JEPA's log(norm) flattens/saturates

### Secondary Metric: GD Convergence Iterations

**Success Pattern:**
- **Flat JEPA**: Exponential growth or failure
  - T=10 → ~50 iters
  - T=20 → ~200 iters
  - T=50 → fails (>1000 iters or diverges)
  
- **H-JEPA**: Roughly constant
  - T=10 → ~20 iters
  - T=20 → ~25 iters
  - T=50 → ~35 iters
  - T=100 → ~40 iters

**Success Criteria:**
✅ H-JEPA converges in <60 iters for all T ≤ 100
✅ Flat fails (no convergence) for T > 30

---

## Experiment 2: Funnel Heatmap

### Primary Metric: Basin of Attraction Coverage

**Success Values:**
- **True Energy**: Not applicable (rugged landscape)
- **Flat Model**: 12-20% coverage
- **HELM Model**: 60-80% coverage

**Success Criteria:**
✅ HELM coverage ≥ 60%
✅ HELM/Flat ratio ≥ 4x
✅ Visual inspection shows:
  - Flat basin heatmap: fragmented, many spurious attractors
  - HELM basin heatmap: single large blue region (smooth funnel)

### Contour Plot Visual Checks

**True Energy:**
- Quadratic bowl with sinusoidal ripples
- ~10-20 local minima visible
- Global minimum at (0, 0)

**Flat Model:**
- Fits the ruggedness (overfits noise)
- Many small basins
- High-frequency features preserved

**HELM Model:**
- Smooth, nearly convex bowl
- Sinusoidal noise mollified away
- Optimum may be slightly offset (~0.1-0.2) but still in correct region

**Success Criteria:**
✅ HELM contours are visually smoother than Flat
✅ HELM has ≤2 spurious local minima (Flat has 8+)

---

## Experiment 3: Planning Efficiency (MiniGrid)

### Primary Metric: Success Rate vs Horizon

**Success Values (% of episodes reaching goal):**

| Horizon T | Flat GD | HELM | CEM | Target (HELM) |
|-----------|---------|------|-----|---------------|
| 10        | 40-50% | 70-80% | 75-85% | >65% |
| 20        | 15-25% | 65-75% | 70-80% | >60% |
| 30        | 5-10% | 55-65% | 65-75% | >50% |
| 50        | 0-5% | 45-60% | 60-70% | >40% |
| 80        | 0% | 30-50% | 50-65% | >30% |

**Success Criteria:**
✅ HELM maintains >50% success at T=30
✅ Flat GD fails (<10%) for T > 30
✅ HELM within 10% of CEM performance (CEM is gold standard)

### Computational Cost (if measured)

**GD Steps to Plan:**
- Flat: grows exponentially or diverges
- HELM: stays ~30-60 steps regardless of T
- CEM: fixed (num_samples × num_iters = ~1280 rollouts)

**Success Criteria:**
✅ HELM uses <80 GD steps for all T ≤ 50
✅ HELM is faster than CEM in wall-clock time (fewer model evaluations)

---

## Minimum Viable Results (Workshop Quality)

If your results are slightly weaker than targets above, the experiments are still publishable if:

- **Exp 1**: H-JEPA norm is 2x+ lower than Flat at T=50 (any bounded vs unbounded gap)
- **Exp 2**: Basin coverage >40% (vs Flat's ~15%)
- **Exp 3**: HELM success >30% at T=50 (vs Flat's 0%)

---

## Red Flags (Implementation Bugs)

**Exp 1:**
❌ H-JEPA norm also explodes exponentially → spectral regularization not working
❌ Both models have similar norms → model too small or λ_spec too large

**Exp 2:**
❌ HELM basin coverage <30% → HF invariance loss not working
❌ HELM contours still rugged → λ_hf too small or σ_noise too large

**Exp 3:**
❌ All planners fail (0% success) → world model not trained properly
❌ HELM worse than Flat → coarse-to-fine initialization is harmful
❌ CEM also fails → goal encoding is wrong

---

## Debugging Tips

**If Exp 1 fails:**
1. Check that `compute_spectral_regularization()` is actually called during HELM training
2. Verify λ_spectral is in [1e-4, 1e-2] range
3. Ensure gradients flow through model outputs (requires_grad=True)

**If Exp 2 fails:**
1. Verify HF noise σ is small (0.01-0.1 range)
2. Check λ_hf is ≥ 0.5
3. Ensure perturbations are applied during training

**If Exp 3 fails:**
1. **Most common**: goal encoding is random → train encoder separately first
2. World model MSE should be <0.01 after training
3. Gumbel-Softmax τ should be ≥ 0.5 (too low causes gradients to vanish)

---

Use these criteria to validate your implementation before writing up results!
