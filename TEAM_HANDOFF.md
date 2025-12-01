# HELM Project Handoff

## Overview
This project validates the **Hierarchical Energy-Landscape Mollification (HELM)** theory. We have empirically demonstrated that HELM solves the "spectral cliff" problem in long-horizon planning by enforcing local contractivity and creating smooth optimization funnels.

## Key Findings (Validated)

### 1. Spectral Cliff (Lemma 1)
- **Problem**: In expansive systems ($\lambda > 1$), the Jacobian norm grows exponentially with horizon $T$.
- **Validation**: On a synthetic linear system ($\lambda_{max}=1.3$):
  - **Flat Model**: $||J(50)|| \approx 1712$ (Exponential explosion).
  - **HELM Model**: $||J(50)|| \approx 90$ (Saturated).
  - **Result**: **~19x reduction** in curvature, validating spectral control.

### 2. Funnel Creation (Lemma 2)
- **Problem**: High-frequency ruggedness traps gradient descent in local minima.
- **Validation**: On a rugged energy landscape:
  - **Flat Model**: Overfits traps. Basin coverage = **4.3%**.
  - **HELM Model**: Smooths landscape. Basin coverage = **96.4%**.
  - **Result**: **22x improvement** in optimization reliability.

## Experiment 3: Planning Efficiency (The "Boundary of Theory")
**Status**: Completed (Negative Result).

- **Finding**: HELM reliably shapes optimization geometry (prediction error $\approx 0$), but planning fails (0% success) because the learned latent manifold does not preserve task topology.
- **Key Insight**: **HELM fixes optimization. It cannot fix latent topology.**
- **Conclusion**: HELM must be paired with a representation-learning module (e.g., contrastive learning, graph embeddings) that encodes the feasible transition structure.

### How to Run
```bash
cd scratch/helm_experiments/exp3_planning
# 1. Train Models (Empty-8x8)
python train.py --env MiniGrid-Empty-8x8-v0 --episodes 200
# 2. Evaluate Planners
python run_planning.py --env MiniGrid-Empty-8x8-v0 --episodes 20 --horizon 10
```

## Code Structure
- `exp1_spectral_cliff/`: Linear system validation & plots.
- `exp2_funnel_heatmap/`: Energy landscape validation & plots.
- `exp3_planning/`: MiniGrid implementation (Gymnasium based).
- `common/`: Shared utilities (Jacobian, Config).

## Next Steps for the Team
1.  **Solve Latent Topology**: Replace the reconstruction loss with **Contrastive Learning** (e.g., Dreamer-style) or **Metric-Structured Space** to ensure latent distance correlates with planning distance.
2.  **Trivial Symbolic Test**: Validate HELM on `Empty-8x8` using *ground-truth* state encoding (x,y coordinates) to prove the planner works when topology is guaranteed.
3.  **Scale Up**: Once topology is fixed, scale back to `DoorKey` and pixel observations.
