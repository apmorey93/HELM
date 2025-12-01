"""
Linear system validation for Jacobian calculation.

CRITICAL TEST: Validate spectral norm calculation on simple linear dynamics
BEFORE trusting it on chaotic Lorenz system.

Theory:
- Expansive dynamics (λ > 1): ||J|| grows exponentially with T
- Contractive dynamics (λ < 1): ||J|| saturates

If this test fails, the Jacobian machinery is broken.
"""

import torch
import sys
import os
import matplotlib.pyplot as plt
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from common.jacobian_utils import frobenius_norm_jacobian


def linear_rollout_expansive(s0, actions, T, expansion_factor=1.1, device='cpu'):
    """
    Linear expansive dynamics: s_{t+1} = λ * s_t + B * a_t
    
    Args:
        s0: (batch, d_s) initial state
        actions: (batch, T, d_a) action sequence
        T: horizon
        expansion_factor: λ > 1 for expansion
        device: torch device
        
    Returns:
        s_T: (batch, d_s) final state
    """
    d_s = s0.shape[-1]
    d_a = actions.shape[-1]
    
    A = expansion_factor * torch.eye(d_s, device=device)
    B = torch.eye(d_s, d_a, device=device) if d_s == d_a else torch.randn(d_s, d_a, device=device) * 0.1
    
    s = s0
    for t in range(T):
        s = s @ A.T + actions[:, t, :] @ B.T
    
    return s


def linear_rollout_contractive(s0, actions, T, contraction_factor=0.9, device='cpu'):
    """
    Linear contractive dynamics: s_{t+1} = λ * s_t + B * a_t
    
    Args:
        s0: (batch, d_s) initial state
        actions: (batch, T, d_a) action sequence
        T: horizon
        contraction_factor: λ < 1 for contraction
        device: torch device
        
    Returns:
        s_T: (batch, d_s) final state
    """
    d_s = s0.shape[-1]
    d_a = actions.shape[-1]
    
    A = contraction_factor * torch.eye(d_s, device=device)
    B = torch.eye(d_s, d_a, device=device) if d_s == d_a else torch.randn(d_s, d_a, device=device) * 0.1
    
    s = s0
    for t in range(T):
        s = s @ A.T + actions[:, t, :] @ B.T
    
    return s


def test_linear_system_jacobian(horizons, expansion_factor=1.1, contraction_factor=0.9, device='cpu'):
    """
    Test Jacobian norm on linear systems with known behavior.
    
    Expected:
    - Expansive: log ||J|| grows linearly with slope ≈ log(λ)
    - Contractive: log ||J|| saturates
    
    Args:
        horizons: list of T values
        expansion_factor: λ for expansive system
        contraction_factor: λ for contractive system
        device: torch device
        
    Returns:
        results_exp, results_contr: (T, ||J||_F) tuples
    """
    print("="*70)
    print(" LINEAR SYSTEM VALIDATION TEST ".center(70, "="))
    print("="*70)
    print(f"\nExpansive factor: {expansion_factor} (should grow)")
    print(f"Contractive factor: {contraction_factor} (should saturate)\n")
    
    d_s = 3
    d_a = 1
    s0 = torch.randn(1, d_s, device=device)
    
    results_exp = []
    results_contr = []
    
    for T in horizons:
        print(f"--- Horizon T={T} ---")
        
        # Random action sequence
        actions = torch.randn(1, T, d_a, device=device)
        
        # Expansive system
        def f_exp(a):
            return linear_rollout_expansive(s0, a, T, expansion_factor, device)
        
        sigma_exp = frobenius_norm_jacobian(f_exp, actions)
        results_exp.append((T, sigma_exp))
        
        # Contractive system
        def f_contr(a):
            return linear_rollout_contractive(s0, a, T, contraction_factor, device)
        
        sigma_contr = frobenius_norm_jacobian(f_contr, actions)
        results_contr.append((T, sigma_contr))
        
        print(f"Expansive:   ||J||_F = {sigma_exp:.4f}, log = {np.log(sigma_exp):.4f}")
        print(f"Contractive: ||J||_F = {sigma_contr:.4f}, log = {np.log(sigma_contr):.4f}")
        
        # Theoretical prediction for expansive (rough approximation)
        # ||J|| ≈ λ^T for simple case
        theoretical_exp = expansion_factor ** T
        print(f"Theoretical (exp): {theoretical_exp:.4f}\n")
    
    return results_exp, results_contr


def plot_linear_validation(results_exp, results_contr, expansion_factor, contraction_factor, save_path='linear_validation.png'):
    """Plot results and check against theory."""
    T_exp, sigma_exp = zip(*results_exp)
    T_contr, sigma_contr = zip(*results_contr)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Left: log(||J||) vs T
    ax1.plot(T_exp, np.log(sigma_exp), 'o-', label=f'Expansive (λ={expansion_factor})', 
             linewidth=2, markersize=8, color='red')
    ax1.plot(T_contr, np.log(sigma_contr), 's-', label=f'Contractive (λ={contraction_factor})', 
             linewidth=2, markersize=8, color='blue')
    
    # Theoretical line for expansive (log(λ^T) = T*log(λ))
    T_array = np.array(T_exp)
    theoretical_log = T_array * np.log(expansion_factor)
    ax1.plot(T_array, theoretical_log, '--', label='Theory: T·log(λ)', 
             color='red', alpha=0.5, linewidth=2)
    
    ax1.set_xlabel('Horizon T', fontsize=12)
    ax1.set_ylabel('log(||J||_F)', fontsize=12)
    ax1.set_title('Linear System: Spectral Norm Growth', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    
    # Right: ||J|| vs T (linear plot)
    ax2.plot(T_exp, sigma_exp, 'o-', label=f'Expansive (λ={expansion_factor})', 
             linewidth=2, markersize=8, color='red')
    ax2.plot(T_contr, sigma_contr, 's-', label=f'Contractive (λ={contraction_factor})', 
             linewidth=2, markersize=8, color='blue')
    
    ax2.set_xlabel('Horizon T', fontsize=12)
    ax2.set_ylabel('||J||_F', fontsize=12)
    ax2.set_title('Spectral Norm (Linear Scale)', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\nSaved plot to {save_path}")
    plt.close()
    
    # Validation checks
    print("\n" + "="*70)
    print(" VALIDATION CHECKS ".center(70, "="))
    print("="*70)
    
    # Check 1: Expansive system grows
    growth_ratio = sigma_exp[-1] / sigma_exp[0]
    print(f"\n1. Expansive Growth Check:")
    print(f"   ||J|| ratio (T={T_exp[-1]}/T={T_exp[0]}): {growth_ratio:.2f}")
    print(f"   Expected: > 10 for significant growth")
    if growth_ratio > 10:
        print("   ✓ PASS: Expansive system shows growth")
    else:
        print("   ✗ FAIL: Growth insufficient - Jacobian calculation may be broken!")
    
    # Check 2: Contractive system saturates
    late_growth = sigma_contr[-1] / sigma_contr[-2] if len(sigma_contr) > 1 else 1.0
    print(f"\n2. Contractive Saturation Check:")
    print(f"   Last step growth: {late_growth:.3f}")
    print(f"   Expected: < 1.2 for saturation")
    if late_growth < 1.2:
        print("   ✓ PASS: Contractive system saturates")
    else:
        print("   ✗ FAIL: Still growing - not saturating!")
    
    # Check 3: Slope consistency for expansive
    log_sigmas = np.log(sigma_exp)
    T_array = np.array(T_exp)
    slope = (log_sigmas[-1] - log_sigmas[0]) / (T_array[-1] - T_array[0])
    theoretical_slope = np.log(expansion_factor)
    print(f"\n3. Slope Consistency Check (Expansive):")
    print(f"   Measured slope: {slope:.4f}")
    print(f"   Theoretical slope (log λ): {theoretical_slope:.4f}")
    print(f"   Ratio: {slope/theoretical_slope:.2f}")
    if 0.5 < slope/theoretical_slope < 2.0:
        print("   ✓ PASS: Slope within 2x of theory")
    else:
        print("   ⚠ WARNING: Slope mismatch - check implementation")


if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}\n")
    
    # Test horizons
    horizons = [5, 10, 20, 50, 100]
    
    # Run validation
    results_exp, results_contr = test_linear_system_jacobian(
        horizons, 
        expansion_factor=1.1,
        contraction_factor=0.9,
        device=device
    )
    
    # Plot and validate
    os.makedirs('plots', exist_ok=True)
    plot_linear_validation(
        results_exp, 
        results_contr,
        expansion_factor=1.1,
        contraction_factor=0.9,
        save_path='plots/linear_validation.png'
    )
    
    print("\n" + "="*70)
    print(" GATE CHECK ".center(70, "="))
    print("="*70)
    print("\nBefore trusting Lorenz results:")
    print("1. Check that ALL validation checks above pass")
    print("2. Visually inspect plots/linear_validation.png:")
    print("   - Red line should be straight upward")
    print("   - Blue line should flatten/saturate")
    print("3. Only if linear system behaves correctly, trust Lorenz\n")
