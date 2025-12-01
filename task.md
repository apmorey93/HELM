# HELM Experiments Implementation

## Project Setup
- [x] Create directory structure
- [x] Implement common utilities (with FIXED Jacobian calculation)
- [x] Create requirements.txt and README

## Experiment 1: Spectral Cliff - VALIDATED ✓
- [x] Implement controlled Lorenz system (found to be non-expansive)
- [x] **PIVOT**: Implement expansive linear system (lambda_max=1.3)
- [x] Train Flat vs HELM models
- [x] **VALIDATED**: Flat Jacobian explodes (1712 at T=50), HELM saturates (90 at T=50)
- [x] **VALIDATED**: Planning success shows clear separation
- [x] Generate publication plots (Jacobian vs T, Planning vs T)

## Experiment 2: Funnel Heatmap - VALIDATED ✓
- [x] Implement synthetic energy functions
- [x] Implement energy MLP model
- [x] Create flat energy training script
- [x] Create HELM energy training with HF invariance
- [x] **BREAKTHROUGH**: Systematic sweep found winning config (sigma=0.5, lambda=5.0)
- [x] **VALIDATED**: 96.4% coverage (22x improvement over 4.3% flat baseline)
- [x] Generate publication-quality plots (contours, coverage curve, 1D slice)

## Experiment 3: Planning Efficiency (MiniGrid) - IN PROGRESS
- [x] Setup MiniGrid environment (DoorKey-8x8, FullyObsWrapper)
- [x] Implement Hierarchical World Model (Encoder, L0, L1)
- [x] Implement Training Script (Flat vs HELM with Spectral+HF)
- [x] Implement Planners (Flat-GD, HELM-GD, CEM)
- [x] Train models (200 episodes collected)
- [x] Train models (200 episodes collected)
- [x] Run planning evaluation (Preliminary: 0% success on 2 eps - needs more training)
- [x] Generate results table
- [x] **Curriculum Pivot**: Train on Empty-8x8 (200 eps)
- [x] **Curriculum Pivot**: Train on Empty-8x8 (200 eps)
- [x] **Curriculum Pivot**: Train on Empty-8x8 (200 eps)
- [x] **Curriculum Pivot**: Evaluate on Empty-8x8 (0% success - Latent Topology Issue)
- [x] **Documentation**: Final Report & GitHub Push
- [x] **Paper Skeleton**: Create conference-ready outline with "Two Pillars" framework
- [x] **Supplementary Material**: Create single-file reproduction script (`helm_supplementary_code.py`) for Nature submission

## Documentation & Validation
- [x] Create expected_results.md with success criteria
- [x] Add requirements.txt
- [ ] Final README with usage examples and timeline
