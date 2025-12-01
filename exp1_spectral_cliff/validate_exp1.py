"""
Experiment 1 Validation - Following Exact Spec

Two metrics to validate Lemma 1:
1. Jacobian norm vs horizon T: Flat explodes (linear in log), HELM saturates
2. Planning success vs horizon T: Flat collapses, HELM degrades gracefully

No HF here - pure spectral regularization test.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from common.config import Exp1Config
from common.jacobian_utils import frobenius_norm_jacobian
from models import FlatWorldModel, HELMWorldModel


def measure_jacobian_vs_horizon(model, config, device='cpu', n_samples=5):
    """
    Metric 1: Measure ||J(T)||_F for increasing horizons.
    
    J = ∂s_T/∂u where u is action sequence.
    Average over n_samples random initializations.
    """
    T_list = [5, 10, 20, 50, 100, 200]
    
    results = []
    
    for T in T_list:
        norms_T = []
        
        for _ in range(n_samples):
            # Sample initial state and actions
            s0 = torch.randn(1, 3, device=device)
            actions = torch.randn(1, T, 1, device=device, requires_grad=True)
            
            # Define function a -> s_T
            # Model takes (s0, actions) where actions is (batch, T, d_a)
            def f(a):
                return model(s0, a)
            
            # Compute Frobenius norm
            J_norm = frobenius_norm_jacobian(f, actions)
            norms_T.append(J_norm)
        
        avg_norm = np.mean(norms_T)
        std_norm = np.std(norms_T)
        
        results.append((T, avg_norm, std_norm))
        print(f"  T={T:3d}: ||J||_F = {avg_norm:8.2f} ± {std_norm:6.2f}")
    
    return results


def planning_success_vs_horizon(model, config, device='cpu', n_trials=20):
    """
    Metric 2: Test planning stability across horizons.
    
    For each T:
    - Sample random target s_T
    - Optimize action sequence u with GD
    - Measure success rate (reaching target within threshold)
    """
    T_list = [10, 20, 50, 100]
    
    results = []
    
    for T in T_list:
        successes = 0
        total_final_loss = 0.0
        
        for trial in range(n_trials):
            # Random initial state
            s0 = torch.randn(1, 3, device=device)
            
            # Generate target by running model with random actions
            with torch.no_grad():
                rand_actions = torch.randn(1, T, 1, device=device) * 0.1
                s_target = model(s0, rand_actions)
            
            # Initialize action sequence
            actions = torch.zeros(1, T, 1, device=device, requires_grad=True)
            optimizer = torch.optim.Adam([actions], lr=0.01)
            
            # GD optimization
            n_steps = 200
            for step in range(n_steps):
                optimizer.zero_grad()
                
                # Predict final state
                s_pred = model(s0, actions)
                
                # Loss
                loss = 0.5 * ((s_pred - s_target) ** 2).sum()
                loss.backward()
                optimizer.step()
                
                # Early stopping
                if loss.item() < 0.05:
                    break
            
            final_loss = loss.item()
            total_final_loss += final_loss
            
            if final_loss < 0.1:  # Success threshold
                successes += 1
        
        success_rate = successes / n_trials * 100
        avg_loss = total_final_loss / n_trials
        
        results.append((T, success_rate, avg_loss))
        print(f"  T={T:3d}: Success={success_rate:5.1f}%, Avg Loss={avg_loss:.4f}")
    
    return results


def plot_results(results_flat, results_helm, save_dir='final_plots'):
    """Generate both metric plots side-by-side."""
    
    # Unpack Jacobian results
    T_jac, norms_flat, stds_flat = zip(*results_flat['jacobian'])
    _, norms_helm, stds_helm = zip(*results_helm['jacobian'])
    
    # Unpack planning results
    T_plan, sr_flat, loss_flat = zip(*results_flat['planning'])
    _, sr_helm, loss_helm = zip(*results_helm['planning'])
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    
    # Left: Jacobian norm vs T (log scale)
    ax1 = axes[0]
    ax1.plot(T_jac, np.log(norms_flat), 'o-', label='Flat (No Regularization)', 
             linewidth=2.5, markersize=10, color='#E74C3C')
    ax1.plot(T_jac, np.log(norms_helm), 's-', label='HELM (Spectral Penalty)', 
             linewidth=2.5, markersize=10, color='#2ECC71')
    
    ax1.set_xlabel('Horizon T', fontsize=13)
    ax1.set_ylabel('log(||J||_F)', fontsize=13)
    ax1.set_title('Jacobian Norm Growth (Spectral Cliff)', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=12, loc='upper left')
    ax1.grid(True, alpha=0.3)
    
    # Add annotation
    ratio_final = norms_flat[-1] / norms_helm[-1]
    ax1.text(0.98, 0.05, f'Separation at T={T_jac[-1]}: {ratio_final:.1f}x', 
             transform=ax1.transAxes, fontsize=11, 
             verticalalignment='bottom', horizontalalignment='right',
             bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.6))
    
    # Right: Planning success vs T
    ax2 = axes[1]
    ax2.plot(T_plan, sr_flat, 'o-', label='Flat', 
             linewidth=2.5, markersize=10, color='#E74C3C')
    ax2.plot(T_plan, sr_helm, 's-', label='HELM', 
             linewidth=2.5, markersize=10, color='#2ECC71')
    
    ax2.set_xlabel('Horizon T', fontsize=13)
    ax2.set_ylabel('Success Rate (%)', fontsize=13)
    ax2.set_title('Planning Stability vs Horizon', fontsize=14, fontweight='bold')
    ax2.set_ylim(0, 105)
    ax2.legend(fontsize=12)
    ax2.grid(True, alpha=0.3)
    
    # Add annotation
    ax2.text(0.05, 0.95, 'Flat: Collapses at long T\nHELM: Degrades gracefully', 
             transform=ax2.transAxes, fontsize=11, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.6))
    
    plt.tight_layout()
    
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, 'exp1_spectral_cliff_validation.png')
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    print(f"\nSaved: {save_path}")
    plt.close()


def run_validation():
    """Main validation pipeline."""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}\n")
    
    config = Exp1Config()
    
    print("="*70)
    print(" EXPERIMENT 1: SPECTRAL CLIFF VALIDATION ".center(70, "="))
    print("="*70)
    
    # Load models
    flat_path = 'checkpoints/flat_world_model.pt'
    helm_path = 'checkpoints/helm_world_model.pt'
    
    if not os.path.exists(flat_path) or not os.path.exists(helm_path):
        print("\nERROR: Models not found!")
        print("Run: python train_flat.py && python train_helm.py")
        return
    
    print("\n>> Loading models...")
    flat_model = FlatWorldModel(d_s=3, d_a=1, hidden_dim=64).to(device)
    helm_model = HELMWorldModel(d_s=3, d_a=1, hidden_dim=64).to(device)
    
    flat_model.load_state_dict(torch.load(flat_path, map_location=device))
    helm_model.load_state_dict(torch.load(helm_path, map_location=device))
    
    flat_model.eval()
    helm_model.eval()
    
    # Metric 1: Jacobian norms
    print("\n" + "="*70)
    print(" METRIC 1: JACOBIAN NORM VS HORIZON ".center(70, "="))
    print("="*70)
    
    print("\n>>> Flat Model:")
    jac_flat = measure_jacobian_vs_horizon(flat_model, config, device, n_samples=5)
    
    print("\n>>> HELM Model:")
    jac_helm = measure_jacobian_vs_horizon(helm_model, config, device, n_samples=5)
    
    # Summary
    print("\n" + "-"*70)
    print("JACOBIAN SUMMARY:")
    print(f"{'T':>5} | {'||J||_F Flat':>15} | {'||J||_F HELM':>15} | {'Ratio (F/H)':>15}")
    print("-"*70)
    
    for (T_f, n_f, _), (T_h, n_h, _) in zip(jac_flat, jac_helm):
        ratio = n_f / n_h
        print(f"{T_f:>5} | {n_f:>15.2f} | {n_h:>15.2f} | {ratio:>15.2f}x")
    
    # Metric 2: Planning stability
    print("\n" + "="*70)
    print(" METRIC 2: PLANNING SUCCESS VS HORIZON ".center(70, "="))
    print("="*70)
    
    print("\n>>> Flat Model:")
    plan_flat = planning_success_vs_horizon(flat_model, config, device, n_trials=20)
    
    print("\n>>> HELM Model:")
    plan_helm = planning_success_vs_horizon(helm_model, config, device, n_trials=20)
    
    # Summary
    print("\n" + "-"*70)
    print("PLANNING SUMMARY:")
    print(f"{'T':>5} | {'Success Flat':>15} | {'Success HELM':>15} | {'Improvement':>15}")
    print("-"*70)
    
    for (T_f, sr_f, _), (T_h, sr_h, _) in zip(plan_flat, plan_helm):
        improvement = sr_h - sr_f
        print(f"{T_f:>5} | {sr_f:>14.1f}% | {sr_h:>14.1f}% | {improvement:>+14.1f}%")
    
    # Verdict
    print("\n" + "="*70)
    print(" VERDICT ".center(70, "="))
    print("="*70)
    
    final_ratio = jac_flat[-1][1] / jac_helm[-1][1]
    
    if final_ratio > 5.0:
        print(f"\nPASS: Clear spectral cliff detected!")
        print(f"  At T=200: Flat={jac_flat[-1][1]:.1f}, HELM={jac_helm[-1][1]:.1f}")
        print(f"  Separation: {final_ratio:.1f}x")
    else:
        print(f"\nWARNING: Weak separation ({final_ratio:.1f}x)")
        print(f"  Consider increasing lambda_spectral")
    
    print(f"\nPlanning at T=100:")
    print(f"  Flat: {plan_flat[-1][1]:.1f}% success")
    print(f"  HELM: {plan_helm[-1][1]:.1f}% success")
    
    # Generate plots
    results_all = {
        'flat': {'jacobian': jac_flat, 'planning': plan_flat},
        'helm': {'jacobian': jac_helm, 'planning': plan_helm}
    }
    
    plot_results(results_all['flat'], results_all['helm'])
    
    print("\n" + "="*70)
    print(" EXP 1 VALIDATION COMPLETE ".center(70, "="))
    print("="*70)


if __name__ == '__main__':
    run_validation()
