# Experiment 3: Planning Efficiency (MiniGrid)

This experiment validates Theorem 1 by demonstrating that HELM's hierarchical world models enable more efficient planning in long-horizon tasks compared to flat baselines.

## Setup
```bash
pip install gymnasium minigrid torch numpy
```

## Running the Experiment

1. **Train World Models**
   Collects data (random policy) and trains both Flat and HELM models.
   ```bash
   python train.py
   ```
   - Collects 200 episodes from `MiniGrid-DoorKey-8x8-v0`.
   - Trains Flat model (Encoder + TransitionL0).
   - Trains HELM model (Encoder + TransitionL0 + TransitionL1) with Spectral and HF regularization.
   - Saves `flat_model.pt` and `helm_model.pt`.

2. **Evaluate Planners**
   Runs planning evaluation using Flat-GD, HELM-GD (Coarse-to-Fine), and CEM.
   ```bash
   python run_planning.py
   ```
   - Evaluates each planner on 20 episodes.
   - Reports Success Rate and Average Steps.

## Expected Results
- **Flat-GD**: Low success rate (<10%) due to vanishing/exploding gradients over long horizons (T=20).
- **HELM-GD**: Higher success rate (>30%) due to coarse-to-fine planning and better conditioned optimization landscape.
- **CEM**: High success rate but requires significantly more model evaluations (expensive).

## Implementation Details
- **Environment**: `MiniGrid-DoorKey-8x8-v0` with `FullyObsWrapper` (Symbolic 8x8x3).
- **Models**:
    - Encoder: CNN (8x8 -> 4x4 -> Flat -> Latent).
    - TransitionL0: MLP residual dynamics ($s_t, a_t \to s_{t+1}$).
    - TransitionL1: MLP coarse dynamics ($s_t, u_t \to s_{t+4}$).
- **Planners**:
    - Flat-GD: Gumbel-Softmax optimization on action logits.
    - HELM-GD: 2-stage optimization (Coarse $u$ -> Upsample -> Fine $a$).
    - CEM: Cross-Entropy Method (sampling-based).
