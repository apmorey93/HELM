"""
Generate Figure 1: Comparison of Energy Landscapes (Flat vs HELM).
Matches the specific caption requested by the user:
(Left) Standard 'Flat' JEPA landscape.
(Right) HELM Level 1 landscape.
"""
import torch
import numpy as np
import matplotlib.pyplot as plt
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from common.config import Exp2Config
from energy_functions import true_energy
from models import EnergyMLP

def generate_figure1(device='cpu'):
    print("Generating Figure 1 (Flat vs HELM)...")
    
    config = Exp2Config()
    
    # Load models
    flat_model = EnergyMLP().to(device)
    helm_model = EnergyMLP().to(device)
    
    flat_path = 'checkpoints/flat_energy.pt'
    helm_path = 'checkpoints/helm_best_sigma0.5_lambda5.0.pt'
    
    if not os.path.exists(flat_path) or not os.path.exists(helm_path):
        print(f"Error: Checkpoints not found. Expected {flat_path} and {helm_path}")
        return

    flat_model.load_state_dict(torch.load(flat_path, map_location=device))
    helm_model.load_state_dict(torch.load(helm_path, map_location=device))
    
    flat_model.eval()
    helm_model.eval()
    
    # Grid
    res = 200
    x = torch.linspace(-3, 3, res, device=device)
    y = torch.linspace(-3, 3, res, device=device)
    X, Y = torch.meshgrid(x, y, indexing='xy')
    grid = torch.stack([X, Y], dim=-1)
    
    # Evaluate
    with torch.no_grad():
        Z_flat = flat_model(grid).cpu().numpy()
        Z_helm = helm_model(grid).cpu().numpy()
    
    X_np = X.cpu().numpy()
    Y_np = Y.cpu().numpy()
    
    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Shared color scale
    vmin = min(Z_flat.min(), Z_helm.min())
    vmax = max(Z_flat.max(), Z_helm.max())
    
    # Flat model
    im1 = axes[0].contourf(X_np, Y_np, Z_flat, levels=50, cmap='viridis', vmin=vmin, vmax=vmax)
    axes[0].contour(X_np, Y_np, Z_flat, levels=10, colors='white', alpha=0.3, linewidths=0.5)
    axes[0].set_title('Standard "Flat" JEPA Landscape', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('$a_1$', fontsize=12)
    axes[0].set_ylabel('$a_2$', fontsize=12)
    axes[0].plot(0, 0, 'r*', markersize=15, label='Global Opt')
    axes[0].legend()
    
    # HELM model
    im2 = axes[1].contourf(X_np, Y_np, Z_helm, levels=50, cmap='viridis', vmin=vmin, vmax=vmax)
    axes[1].contour(X_np, Y_np, Z_helm, levels=10, colors='white', alpha=0.3, linewidths=0.5)
    axes[1].set_title('HELM Level 1 Landscape', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('$a_1$', fontsize=12)
    axes[1].set_ylabel('$a_2$', fontsize=12)
    axes[1].plot(0, 0, 'r*', markersize=15)
    
    # Shared colorbar
    fig.colorbar(im2, ax=axes, label='Energy', fraction=0.03, pad=0.04)
    
    plt.tight_layout()
    output_path = 'final_plots/figure1_flat_vs_helm.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()

if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    os.makedirs('final_plots', exist_ok=True)
    generate_figure1(device)
