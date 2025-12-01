"""
Visualization and basin analysis for Experiment 2.

Generates contour plots and basin of attraction heatmaps.
"""

import torch
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from common.config import Exp2Config
from common.viz_utils import plot_energy_contours, plot_basin_heatmap
from energy_functions import true_energy, evaluate_on_grid
from models import EnergyMLP


def gradient_descent_on_energy(energy_model, a_init, lr=0.1, steps=100, device='cuda'):
    """
    Run gradient descent to minimize energy from initial point.
    
    Args:
        energy_model: EnergyMLP or function
        a_init: (batch, 2) initial points
        lr: learning rate
        steps: GD steps
        device: torch device
        
    Returns:
        (batch, 2) final points after GD
    """
    a = a_init.clone().detach().requires_grad_(True)
    optimizer = torch.optim.SGD([a], lr=lr)
    
    for _ in range(steps):
        optimizer.zero_grad()
        E = energy_model(a).sum()  # sum over batch for backward
        E.backward()
        optimizer.step()
    
    return a.detach()


def compute_basin_coverage(energy_model, a_true, config, device='cuda'):
    """
    Compute basin of attraction coverage by running GD from grid of init points.
    
    Args:
        energy_model: trained EnergyMLP
        a_true: (2,) true optimum location
        config: Exp2Config
        device: torch device
        
    Returns:
        X, Y: meshgrids
        dist_grid: (H, W) distance to optimum after GD
        coverage: fraction of points that converge close to optimum
    """
    print("Computing basin of attraction...")
    
    grid_res = config.basin_grid_resolution
    x = torch.linspace(-config.domain_range, config.domain_range, grid_res, device=device)
    y = torch.linspace(-config.domain_range, config.domain_range, grid_res, device=device)
    
    X, Y = torch.meshgrid(x, y, indexing='xy')
    grid = torch.stack([X, Y], dim=-1)  # (grid_res, grid_res, 2)
    
    # Flatten for batch processing
    grid_flat = grid.reshape(-1, 2)  # (grid_res^2, 2)
    
    # Run GD from each point
    finals = gradient_descent_on_energy(
        energy_model,
        grid_flat,
        lr=config.gd_lr_basin,
        steps=config.gd_steps_basin,
        device=device
    )  # (grid_res^2, 2)
    
    # Compute distance to true optimum
    a_true_t = torch.tensor(a_true, device=device, dtype=torch.float32)
    dist = torch.norm(finals - a_true_t, dim=-1)  # (grid_res^2,)
    
    # Reshape to grid
    dist_grid = dist.reshape(grid_res, grid_res)
    
    # Coverage metric: fraction within threshold
    threshold = 0.5
    coverage = (dist < threshold).float().mean().item()
    
    print(f"Basin coverage (within {threshold} of optimum): {coverage*100:.1f}%")
    
    return X, Y, dist_grid, coverage


def visualize_experiment_2(config=None, device='cuda'):
    """
    Generate all visualizations for Experiment 2.
    
    Creates:
        1. True energy contours
        2. Flat model contours
        3. HELM model contours
        4. Basin heatmaps (Flat vs HELM)
    """
    if config is None:
        config = Exp2Config()
    
    print("\n" + "="*60)
    print(" Experiment 2: Visualization & Analysis ".center(60, "="))
    print("="*60 + "\n")
    
    os.makedirs('plots', exist_ok=True)
    
    # Load models
    flat_model = EnergyMLP().to(device)
    helm_model = EnergyMLP().to(device)
    
    flat_model.load_state_dict(torch.load('checkpoints/flat_energy.pt'))
    helm_model.load_state_dict(torch.load('checkpoints/helm_energy.pt'))
    
    flat_model.eval()
    helm_model.eval()
    
    # ===== CONTOUR PLOTS =====
    print("Generating contour plots...")
    
    # True energy
    def E_true(a):
        return true_energy(a, config)
    
    X, Y, Z_true = evaluate_on_grid(E_true, resolution=config.grid_resolution, device=device)
    plot_energy_contours(
        (X, Y), Z_true,
        title='True Energy (Convex + Rugged)',
        save_path='plots/exp2_true_energy.png'
    )
    
    # Flat model
    with torch.no_grad():
        X_f, Y_f, Z_flat = evaluate_on_grid(
            flat_model, resolution=config.grid_resolution, device=device
        )
    plot_energy_contours(
        (X_f, Y_f), Z_flat,
        title='Flat Model (Overfits Ruggedness)',
        save_path='plots/exp2_flat_energy.png'
    )
    
    # HELM model
    with torch.no_grad():
        X_h, Y_h, Z_helm = evaluate_on_grid(
            helm_model, resolution=config.grid_resolution, device=device
        )
    plot_energy_contours(
        (X_h, Y_h), Z_helm,
        title='HELM Model (Smoothed Funnel)',
        save_path='plots/exp2_helm_energy.png'
    )
    
    # ===== BASIN OF ATTRACTION ANALYSIS =====
    print("\nAnalyzing basins of attraction...")
    
    a_true = config.a_optimum
    
    # Flat model basin
    X_b, Y_b, dist_flat, coverage_flat = compute_basin_coverage(
        flat_model, a_true, config, device
    )
    plot_basin_heatmap(
        (X_b, Y_b), dist_flat,
        title=f'Flat Model Basin (Coverage: {coverage_flat*100:.1f}%)',
        save_path='plots/exp2_flat_basin.png'
    )
    
    # HELM model basin
    _, _, dist_helm, coverage_helm = compute_basin_coverage(
        helm_model, a_true, config, device
    )
    plot_basin_heatmap(
        (X_b, Y_b), dist_helm,
        title=f'HELM Model Basin (Coverage: {coverage_helm*100:.1f}%)',
        save_path='plots/exp2_helm_basin.png'
    )
    
    # ===== SUMMARY =====
    print("\n" + "="*60)
    print(" SUMMARY ".center(60, "="))
    print("="*60)
    print(f"\nBasin Coverage:")
    print(f"  Flat Model: {coverage_flat*100:.1f}%")
    print(f"  HELM Model: {coverage_helm*100:.1f}%")
    print(f"  Improvement: {coverage_helm/coverage_flat:.2f}x")
    
    print("\nPlots saved to plots/:")
    print("  - exp2_true_energy.png")
    print("  - exp2_flat_energy.png")
    print("  - exp2_helm_energy.png")
    print("  - exp2_flat_basin.png")
    print("  - exp2_helm_basin.png")
    
    return {
        'coverage_flat': coverage_flat,
        'coverage_helm': coverage_helm
    }


if __name__ == '__main__':
    config = Exp2Config()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    results = visualize_experiment_2(config, device)
