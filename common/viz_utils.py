"""
Visualization utilities for HELM experiments.
"""

import matplotlib.pyplot as plt
import numpy as np
import torch


def plot_spectral_cliff(results_flat, results_helm, save_path='spectral_cliff.png'):
    """
    Plot spectral norm vs horizon for Experiment 1.
    
    Args:
        results_flat: list of (T, sigma) tuples for flat model
        results_helm: list of (T, sigma) tuples for HELM model
        save_path: where to save figure
        
    Expected pattern:
        - Flat: exponential growth (linear in log-space)
        - HELM: saturation (flattens)
    """
    T_flat, sigma_flat = zip(*results_flat)
    T_helm, sigma_helm = zip(*results_helm)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Left panel: log(sigma) vs T
    ax1.plot(T_flat, np.log(sigma_flat), 'o-', label='Flat JEPA', linewidth=2, markersize=8)
    ax1.plot(T_helm, np.log(sigma_helm), 's-', label='H-JEPA', linewidth=2, markersize=8)
    ax1.set_xlabel('Planning Horizon T', fontsize=12)
    ax1.set_ylabel('log(||J||_F)', fontsize=12)
    ax1.set_title('Spectral Norm vs Horizon', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    
    # Right panel: sigma vs T (linear scale for visualization)
    ax2.semilogy(T_flat, sigma_flat, 'o-', label='Flat JEPA', linewidth=2, markersize=8)
    ax2.semilogy(T_helm, sigma_helm, 's-', label='H-JEPA', linewidth=2, markersize=8)
    ax2.set_xlabel('Planning Horizon T', fontsize=12)
    ax2.set_ylabel('||J||_F (log scale)', fontsize=12)
    ax2.set_title('Spectral Explosion (Flat) vs Saturation (HELM)', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Saved spectral cliff plot to {save_path}")
    plt.close()


def plot_convergence_rate(results_flat, results_helm, save_path='convergence_rate.png'):
    """
    Plot GD convergence iterations vs horizon.
    
    Expected pattern:
        - Flat: exponential growth or failure
        - HELM: roughly constant
    """
    T_flat, iters_flat = zip(*results_flat)
    T_helm, iters_helm = zip(*results_helm)
    
    plt.figure(figsize=(8, 6))
    plt.plot(T_flat, iters_flat, 'o-', label='Flat JEPA', linewidth=2, markersize=8)
    plt.plot(T_helm, iters_helm, 's-', label='H-JEPA', linewidth=2, markersize=8)
    plt.xlabel('Planning Horizon T', fontsize=12)
    plt.ylabel('GD Iterations to Converge', fontsize=12)
    plt.title('Convergence Rate vs Horizon', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Saved convergence rate plot to {save_path}")
    plt.close()


def plot_energy_contours(grid_coords, energy_values, title='Energy Landscape', 
                         save_path='energy_contours.png', vmin=None, vmax=None):
    """
    Plot 2D energy landscape contours.
    
    Args:
        grid_coords: (X, Y) meshgrid tensors or numpy arrays
        energy_values: (H, W) energy values
        title: plot title
        save_path: where to save
        vmin, vmax: color scale limits (for comparison across plots)
    """
    X, Y = grid_coords
    if isinstance(X, torch.Tensor):
        X = X.cpu().numpy()
        Y = Y.cpu().numpy()
    if isinstance(energy_values, torch.Tensor):
        energy_values = energy_values.cpu().numpy()
    
    plt.figure(figsize=(8, 7))
    
    # Contour filled plot
    contour = plt.contourf(X, Y, energy_values, levels=50, cmap='viridis', 
                           vmin=vmin, vmax=vmax)
    plt.colorbar(contour, label='Energy')
    
    # Optionally add contour lines
    plt.contour(X, Y, energy_values, levels=10, colors='white', alpha=0.3, linewidths=0.5)
    
    plt.xlabel('a₁', fontsize=12)
    plt.ylabel('a₂', fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.axis('equal')
    
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Saved energy contours to {save_path}")
    plt.close()


def plot_basin_heatmap(grid_coords, distance_to_optimum, title='Basin of Attraction',
                       save_path='basin_heatmap.png'):
    """
    Plot basin of attraction heatmap showing convergence behavior.
    
    Args:
        grid_coords: (X, Y) meshgrid
        distance_to_optimum: (H, W) final distance after GD
        title: plot title
        save_path: where to save
    """
    X, Y = grid_coords
    if isinstance(X, torch.Tensor):
        X = X.cpu().numpy()
        Y = Y.cpu().numpy()
    if isinstance(distance_to_optimum, torch.Tensor):
        distance_to_optimum = distance_to_optimum.cpu().numpy()
    
    plt.figure(figsize=(8, 7))
    
    # Use reversed colormap: blue = good (converged), red = bad (diverged)
    im = plt.imshow(distance_to_optimum, origin='lower', 
                    extent=[X.min(), X.max(), Y.min(), Y.max()],
                    cmap='coolwarm', vmin=0, vmax=3.0)
    plt.colorbar(im, label='Distance to Global Optimum')
    
    plt.xlabel('a₁', fontsize=12)
    plt.ylabel('a₂', fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Saved basin heatmap to {save_path}")
    plt.close()


def plot_planning_success_rate(results_dict, save_path='planning_success.png'):
    """
    Plot success rate vs horizon for different planning methods.
    
    Args:
        results_dict: {method_name: [(T, success_rate), ...]}
        save_path: where to save
    """
    plt.figure(figsize=(10, 6))
    
    markers = ['o', 's', '^', 'd', 'v']
    for idx, (method, results) in enumerate(results_dict.items()):
        T_vals, success_rates = zip(*results)
        plt.plot(T_vals, success_rates, f'{markers[idx % len(markers)]}-', 
                label=method, linewidth=2, markersize=8)
    
    plt.xlabel('Planning Horizon T', fontsize=12)
    plt.ylabel('Success Rate (%)', fontsize=12)
    plt.title('Planning Success Rate vs Horizon', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.ylim(0, 105)
    
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Saved planning success rate plot to {save_path}")
    plt.close()


def plot_ablation_study(variant_names, success_rates, gd_steps, 
                        save_path='ablation_study.png'):
    """
    Plot ablation study results showing contribution of each component.
    
    Args:
        variant_names: list of variant names
        success_rates: list of success rates (%)
        gd_steps: list of average GD steps
        save_path: where to save
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    x = np.arange(len(variant_names))
    width = 0.6
    
    # Left: success rate
    bars1 = ax1.bar(x, success_rates, width, color=['#ff6b6b', '#f9ca24', '#6c5ce7', '#00b894'])
    ax1.set_ylabel('Success Rate (%) @ T=50', fontsize=12)
    ax1.set_title('Component Contribution: Success Rate', fontsize=14, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(variant_names, rotation=15, ha='right')
    ax1.grid(True, alpha=0.3, axis='y')
    ax1.set_ylim(0, 100)
    
    # Add value labels on bars
    for bar, val in zip(bars1, success_rates):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 2,
                f'{val:.1f}%', ha='center', va='bottom', fontsize=10)
    
    # Right: GD steps
    bars2 = ax2.bar(x, gd_steps, width, color=['#ff6b6b', '#f9ca24', '#6c5ce7', '#00b894'])
    ax2.set_ylabel('Avg GD Steps @ T=50', fontsize=12)
    ax2.set_title('Component Contribution: Efficiency', fontsize=14, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(variant_names, rotation=15, ha='right')
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar, val in zip(bars2, gd_steps):
        if val < 999:  # N/A values shown as 999
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 2,
                    f'{val:.0f}', ha='center', va='bottom', fontsize=10)
        else:
            ax2.text(bar.get_x() + bar.get_width()/2., 5,
                    'N/A', ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Saved ablation study plot to {save_path}")
    plt.close()
