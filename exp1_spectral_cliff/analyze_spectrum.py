"""
Spectral analysis tools for Experiment 1.

Measures Jacobian norm as function of horizon T and tests
convergence rate of gradient-based planning.
"""

import torch
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from common.jacobian_utils import frobenius_norm_jacobian
from common.config import Exp1Config
from dataset import generate_controlled_lorenz_trajectories
from models import FlatWorldModel, HELMWorldModel


def analyze_spectral_norm(model, T, num_samples=64, device='cuda', config=None):
    """
    Measure Frobenius norm of Jacobian ||∂s_T/∂a||_F.
    
    Args:
        model: trained world model
        T: horizon
        num_samples: number of test samples
        device: torch device
        config: Exp1Config
        
    Returns:
        mean spectral norm over samples
    """
    if config is None:
        config = Exp1Config()
    
    print(f"Analyzing spectral norm for T={T}...")
    
    # Generate test data
    states, actions = generate_controlled_lorenz_trajectories(
        num_traj=num_samples,
        T=T,
        dt=config.dt,
        sigma=config.sigma,
        rho=config.rho,
        beta=config.beta,
        control_coupling=config.control_coupling,
        device=device
    )
    
    s0 = states[:, 0]
    
    # Measure Jacobian norm for each sample
    spectral_norms = []
    
    for i in range(num_samples):
        s0_i = s0[i:i+1]  # (1, 3)
        a_i = actions[i:i+1]  # (1, T, 1)
        
        # Define function f: a -> s_T
        def f(a_seq):
            return model(s0_i, a_seq)
        
        # Compute ||∂f/∂a||_F
        sigma = frobenius_norm_jacobian(f, a_i)
        spectral_norms.append(sigma)
        
        if (i + 1) % 10 == 0:
            print(f"  Sample {i+1}/{num_samples}: ||J||_F = {sigma:.3f}")
    
    mean_sigma = sum(spectral_norms) / len(spectral_norms)
    std_sigma = (sum((x - mean_sigma)**2 for x in spectral_norms) / len(spectral_norms)) ** 0.5
    
    print(f"T={T}: mean ||J||_F = {mean_sigma:.3f} ± {std_sigma:.3f}")
    
    return mean_sigma, std_sigma


def test_convergence_rate(model, T, num_tests=20, max_iters=1000, 
                          lr=0.01, threshold=1e-4, device='cuda', config=None):
    """
    Test gradient descent convergence rate for planning.
    
    Setup: Given target s_T*, initialize random s_0, run GD to minimize
    ||model(s_0, T) - s_T*||^2. Count iterations to converge.
    
    Args:
        model: trained world model
        T: horizon
        num_tests: number of test cases
        max_iters: maximum GD iterations
        lr: learning rate
        threshold: convergence threshold
        device: torch device
        config: Exp1Config
        
    Returns:
        mean convergence iterations (or max_iters if failed)
    """
    if config is None:
        config = Exp1Config()
    
    print(f"Testing convergence rate for T={T}...")
    
    # Generate target states
    states, actions_gt = generate_controlled_lorenz_trajectories(
        num_traj=num_tests,
        T=T,
        device=device
    )
    s_T_targets = states[:, -1]  # (num_tests, 3)
    
    convergence_iters = []
    
    for i in range(num_tests):
        s_T_target = s_T_targets[i:i+1]
        
        # Initialize random s_0
        s0 = torch.randn(1, 3, device=device, requires_grad=True)
        
        # Fixed random action sequence (we optimize s0, not actions)
        actions = torch.randn(1, T, 1, device=device)
        
        optimizer = torch.optim.SGD([s0], lr=lr)
        
        converged_at = max_iters
        
        for step in range(max_iters):
            optimizer.zero_grad()
            
            s_T_pred = model(s0, actions)
            loss = (s_T_pred - s_T_target).pow(2).sum()
            
            if loss.item() < threshold:
                converged_at = step
                break
            
            loss.backward()
            optimizer.step()
        
        convergence_iters.append(converged_at)
        
        if (i + 1) % 5 == 0:
            print(f"  Test {i+1}/{num_tests}: converged in {converged_at} iters")
    
    mean_iters = sum(convergence_iters) / len(convergence_iters)
    success_rate = sum(1 for x in convergence_iters if x < max_iters) / len(convergence_iters)
    
    print(f"T={T}: mean convergence = {mean_iters:.1f} iters, success = {success_rate*100:.1f}%")
    
    return mean_iters, success_rate


def main():
    """Run spectral analysis for both Flat and HELM models."""
    config = Exp1Config()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}\n")
    
    results_spectral_flat = []
    results_spectral_helm = []
    results_convergence_flat = []
    results_convergence_helm = []
    
    for T in config.horizons:
        print(f"\n{'='*60}")
        print(f"Analyzing Horizon T={T}")
        print(f"{'='*60}\n")
        
        # Load models
        flat_model = FlatWorldModel(d_s=3, d_a=1).to(device)
        helm_model = HELMWorldModel(d_s=3, d_a=1).to(device)
        
        flat_path = f'checkpoints/flat_model_T{T}.pt'
        helm_path = f'checkpoints/helm_model_T{T}.pt'
        
        if os.path.exists(flat_path) and os.path.exists(helm_path):
            flat_model.load_state_dict(torch.load(flat_path))
            helm_model.load_state_dict(torch.load(helm_path))
            
            flat_model.eval()
            helm_model.eval()
            
            # Spectral norm analysis
            print("\n--- Spectral Norm Analysis ---")
            mean_flat, std_flat = analyze_spectral_norm(
                flat_model, T, num_samples=config.num_test_samples, device=device, config=config
            )
            mean_helm, std_helm = analyze_spectral_norm(
                helm_model, T, num_samples=config.num_test_samples, device=device, config=config
            )
            
            results_spectral_flat.append((T, mean_flat, std_flat))
            results_spectral_helm.append((T, mean_helm, std_helm))
            
            # Convergence rate analysis
            print("\n--- Convergence Rate Analysis ---")
            iters_flat, success_flat = test_convergence_rate(
                flat_model, T, num_tests=20, device=device, config=config
            )
            iters_helm, success_helm = test_convergence_rate(
                helm_model, T, num_tests=20, device=device, config=config
            )
            
            results_convergence_flat.append((T, iters_flat))
            results_convergence_helm.append((T, iters_helm))
        else:
            print(f"Models not found for T={T}. Please train first.")
    
    # Print summary
    print(f"\n{'='*60}")
    print("SUMMARY: Spectral Norm Results")
    print(f"{'='*60}")
    print(f"{'T':>5} | {'Flat ||J||_F':>15} | {'HELM ||J||_F':>15} | {'Ratio':>10}")
    print("-" * 60)
    for (T_f, mean_f, std_f), (T_h, mean_h, std_h) in zip(results_spectral_flat, results_spectral_helm):
        ratio = mean_f / mean_h if mean_h > 0 else float('inf')
        print(f"{T_f:>5} | {mean_f:>12.3f} | {mean_h:>12.3f} | {ratio:>10.2f}x")
    
    print(f"\n{'='*60}")
    print("SUMMARY: Convergence Rate Results")
    print(f"{'='*60}")
    print(f"{'T':>5} | {'Flat (iters)':>15} | {'HELM (iters)':>15}")
    print("-" * 60)
    for (T_f, iters_f), (T_h, iters_h) in zip(results_convergence_flat, results_convergence_helm):
        print(f"{T_f:>5} | {iters_f:>12.1f} | {iters_h:>12.1f}")
    
    # Save results
    torch.save({
        'spectral_flat': results_spectral_flat,
        'spectral_helm': results_spectral_helm,
        'convergence_flat': results_convergence_flat,
        'convergence_helm': results_convergence_helm
    }, 'analysis_results.pt')
    
    print("\nResults saved to analysis_results.pt")
    
    return results_spectral_flat, results_spectral_helm, results_convergence_flat, results_convergence_helm


if __name__ == '__main__':
    main()
