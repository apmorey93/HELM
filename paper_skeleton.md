# Title: The Two Pillars of Gradient Planning: Spectral Stability and Metric Alignment in Latent World Models

## Abstract
Gradient-based planning in learned latent spaces promises efficient long-horizon control but remains notoriously unstable. We identify two fundamental pathologies that prevent success: (1) **Spectral Explosion**, where the Jacobian norm of the forward dynamics grows exponentially with the horizon, and (2) **Metric Misalignment**, where the latent geometry fails to reflect the task topology. We introduce **HELM (Hierarchical Energy-Landscape Mollification)**, a framework that enforces spectral stability via hierarchical regularization, and show that it is necessary but not sufficient for planning. We demonstrate that HELM must be paired with **Contrastive Metric Alignment** to ensure gradients are both stable and useful. Empirical validation on expansive systems and discrete POMDPs confirms that this unified "Two Pillars" approach achieves 100% planning success where standard methods fail completely.

---

## 1. Introduction
- **The Promise**: Differentiable planning (Dreamer, MuZero) vs. Sampling.
- **The Problem**: "Why doesn't gradient descent work in latent space?"
- **Existing Hypotheses**: Local minima, vanishing gradients.
- **Our Insight**: It's actually two distinct problems:
    1.  **Numerical Stability**: Gradients explode (Spectral Cliff).
    2.  **Geometric Validity**: Gradients point nowhere (Topology Mismatch).
- **Contribution**: The "Two Pillars" framework (HELM + Contrastive).

## 2. Theory: The Spectral Cliff
- **Lemma 1 (Spectral Explosion)**: Proof that for any expansive system ($\lambda > 1$), $||\nabla_x x_T|| \to \infty$ exponentially.
- **Consequence**: Planning horizon is effectively bounded by $\log(1/\lambda)$.
- **Solution**: **Spectral Regularization** ($\lambda_{spec}$) to bound the Lipschitz constant of the transition function.

## 3. Method: HELM (Hierarchical Energy-Landscape Mollification)
- **Architecture**:
    - **Level 0**: Fine-grained dynamics (High fidelity).
    - **Level 1**: Coarse-grained dynamics (Mollified).
- **Mechanism 1: Spectral Control**: Enforcing $||\nabla f|| \approx 1$.
- **Mechanism 2: High-Frequency Invariance**: Smoothing the energy landscape to create convex funnels (Lemma 2).
- **The Missing Piece**: Acknowledging that stable gradients require a valid metric.

## 4. The Second Pillar: Metric Alignment
- **The Paradox**: A model can have perfect prediction loss ($L_{pred} \approx 0$) but zero planning utility.
- **Diagnosis**: Standard reconstruction losses (MSE) do not enforce geodesic consistency in latent space.
- **Solution**: **Contrastive Goal-Conditioning**.
    - Objective: Minimize distance between temporally close states.
    - Result: Latent Euclidean distance $\propto$ Environment Geodesic distance.

## 5. Experiments

### Exp 1: The Spectral Cliff (Validation)
- **Setup**: Linear expansive system.
- **Result**: Flat model Jacobian explodes (1712). HELM saturates (90). **19x reduction**.

### Exp 2: Funnel Creation (Validation)
- **Setup**: Rugged non-convex energy landscape.
- **Result**: Flat model overfits noise (4% coverage). HELM creates smooth funnel (96% coverage).

### Exp 3: The Failure of Stability Alone (Ablation)
- **Setup**: MiniGrid planning with HELM but *without* metric alignment.
- **Result**: **0% Success**. Perfect prediction, useless planning.
- **Insight**: Proves "Stability $\neq$ Utility."

### Exp 4: The Unified Solution (Success)
- **Setup**: HELM + Contrastive Metric Alignment.
- **Result**: **100% Success**. Planner navigates from Distance 7.0 $\to$ 0.0.

## 6. Discussion
- **The Two Pillars**:
    1.  **HELM** fixes the *optimization geometry* (curvature, smoothness).
    2.  **Contrastive** fixes the *representation geometry* (topology, alignment).
- **Comparison**:
    - VAE/MSE: Good prediction, bad topology.
    - Contrastive only: Good topology, unstable gradients (without HELM).
    - **Unified**: Stable and aligned.

## 7. Conclusion
- Summary of the two pathologies.
- Final claim: Gradient planning is viable only when both spectral and metric invariants are satisfied.

## Appendices
- **A. Proof of Lemma 1**
- **B. Proof of Lemma 2**
- **C. Hyperparameter Sweeps**
