"""
Systematic HF hyperparameter sweep with strong regularization.

Testing 3 configs to see if stronger HF actually enables smoothing:
1. Mildly stronger: sigma=0.2, lambda=5.0
2. Moderately strong: sigma=0.5, lambda=5.0  
3. Aggressive: sigma=0.5, lambda=10.0

For each, measure:
- MSE and HF loss progression
- Final variance
- Basin coverage
"""

import torch
import torch.nn as nn
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from common.config import Exp2Config
from energy_functions import true_energy
from models import EnergyMLP


def train_helm_with_logging(sigma, lambda_hf, num_steps=5000, device='cpu'):
    """Train HELM with detailed logging of loss components."""
    print(f"\n{'='*70}")
    print(f" Training: sigma={sigma}, lambda_HF={lambda_hf} ".center(70, "="))
    print(f"{'='*70}\n")
    
    config = Exp2Config()
    config.hf_noise_sigma = sigma
    config.lambda_hf = lambda_hf
    config.num_training_steps = num_steps
    
    model = EnergyMLP(hidden_dim=128).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    loss_fn = nn.MSELoss()
    
    # Track loss progression
    mse_history = []
    hf_history = []
    total_history = []
    
    for step in range(num_steps):
        # Sample points
        a = (torch.rand(config.batch_size, 2, device=device) - 0.5) * 6.0
        
        with torch.no_grad():
            E_true = true_energy(a, config)
        
        E_pred = model(a)
        mse_loss = loss_fn(E_pred, E_true)
        
        # HF invariance
        delta = torch.randn_like(a) * sigma
        E_pred_perturbed = model(a + delta)
        hf_loss = ((E_pred - E_pred_perturbed) ** 2).mean()
        
        # Combined
        total_loss = mse_loss + lambda_hf * hf_loss
        
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        
        # Log
        mse_history.append(mse_loss.item())
        hf_history.append(hf_loss.item())
        total_history.append(total_loss.item())
        
        # Print progress
        if (step + 1) % 1000 == 0 or step == 0:
            print(f"Step {step+1}/{num_steps}:")
            print(f"  MSE:   {mse_loss.item():.6f}")
            print(f"  HF:    {hf_loss.item():.6f}")
            print(f"  Total: {total_loss.item():.6f}")
            print(f"  HF contribution: {(lambda_hf * hf_loss.item()) / total_loss.item() * 100:.1f}%")
    
    print(f"\nFinal losses:")
    print(f"  MSE:   {mse_history[-1]:.6f}")
    print(f"  HF:    {hf_history[-1]:.6f}")
    print(f"  Total: {total_history[-1]:.6f}")
    
    return model, mse_history, hf_history


def evaluate_model(model, config, device='cpu'):
    """Evaluate variance and basin coverage."""
    # Variance
    a_samples = (torch.rand(5000, 2, device=device) - 0.5) * 6.0
    with torch.no_grad():
        E_pred = model(a_samples).cpu().numpy()
    variance = E_pred.var()
    
    # Basin coverage - run GD on model
    x = torch.linspace(-3, 3, 50, device=device)
    y = torch.linspace(-3, 3, 50, device=device)
    X, Y = torch.meshgrid(x, y, indexing='xy')
    grid = torch.stack([X, Y], dim=-1).reshape(-1, 2)
    
    # Clone and enable grad
    a = grid.clone().detach()
    a.requires_grad = True
    optimizer = torch.optim.SGD([a], lr=0.1)
    
    # Put model in eval mode
    model.eval()
    
    for _ in range(100):
        optimizer.zero_grad()
        E = model(a)
        E.sum().backward()
        optimizer.step()
    
    a_true = torch.tensor(config.a_optimum, device=device, dtype=torch.float32)
    dist = torch.norm(a.detach() - a_true, dim=-1)
    coverage = (dist < 0.5).float().mean().item() * 100
    
    return variance, coverage


def run_sweep():
    """Run systematic sweep."""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}\n")
    
    config = Exp2Config()
    
    # Baseline: Flat model (already trained)
    print("="*70)
    print(" BASELINE: Flat Model ".center(70, "="))
    print("="*70)
    
    flat_model = EnergyMLP().to(device)
    flat_model.load_state_dict(torch.load('checkpoints/flat_energy.pt', map_location=device))
    flat_model.eval()
    
    var_flat, cov_flat = evaluate_model(flat_model, config, device)
    print(f"\nFlat model:")
    print(f"  Variance: {var_flat:.4f}")
    print(f"  Coverage: {cov_flat:.1f}%")
    
    # Test configs
    configs = [
        {'sigma': 0.2, 'lambda': 5.0, 'name': 'Mildly stronger'},
        {'sigma': 0.5, 'lambda': 5.0, 'name': 'Moderately strong'},
        {'sigma': 0.5, 'lambda': 10.0, 'name': 'Aggressive'},
    ]
    
    results = []
    
    for cfg in configs:
        print(f"\n\n{'='*70}")
        print(f" CONFIG: {cfg['name']} ".center(70, "="))
        print(f"{'='*70}")
        
        model, mse_hist, hf_hist = train_helm_with_logging(
            cfg['sigma'], cfg['lambda'], num_steps=5000, device=device
        )
        
        print(f"\nEvaluating...")
        var_helm, cov_helm = evaluate_model(model, config, device)
        
        result = {
            'name': cfg['name'],
            'sigma': cfg['sigma'],
            'lambda': cfg['lambda'],
            'mse_final': mse_hist[-1],
            'hf_final': hf_hist[-1],
            'variance': var_helm,
            'coverage': cov_helm,
            'var_ratio': var_helm / var_flat,
            'cov_improvement': cov_helm / cov_flat
        }
        
        results.append(result)
        
        print(f"\nResults:")
        print(f"  Variance:    {var_helm:.4f} (vs flat {var_flat:.4f})")
        print(f"  Var ratio:   {var_helm/var_flat:.3f}")
        print(f"  Coverage:    {cov_helm:.1f}% (vs flat {cov_flat:.1f}%)")
        print(f"  Improvement: {cov_helm/cov_flat:.2f}x")
        
        # Save best model
        if cov_helm > cov_flat * 1.5:  # If 1.5x better
            torch.save(model.state_dict(), f'checkpoints/helm_best_sigma{cfg["sigma"]}_lambda{cfg["lambda"]}.pt')
            print(f"  >> Saved as potential winner!")
    
    # Final summary
    print("\n\n" + "="*70)
    print(" FINAL SUMMARY ".center(70, "="))
    print("="*70)
    
    print(f"\nBaseline (Flat):")
    print(f"  Variance: {var_flat:.4f}")
    print(f"  Coverage: {cov_flat:.1f}%")
    
    print(f"\n{'Config':<20} | {'Var':>8} | {'Var/Flat':>10} | {'Cov':>8} | {'Improve':>10}")
    print("-" * 70)
    
    for r in results:
        print(f"{r['name']:<20} | {r['variance']:>8.4f} | {r['var_ratio']:>10.3f} | {r['coverage']:>7.1f}% | {r['cov_improvement']:>9.2f}x")
    
    # Verdict
    best = max(results, key=lambda x: x['coverage'])
    
    print("\n" + "-" * 70)
    print(f"Best config: {best['name']}")
    print(f"  Coverage: {best['coverage']:.1f}% ({best['cov_improvement']:.2f}x improvement)")
    print(f"  Variance: {best['variance']:.4f} ({best['var_ratio']:.3f} of flat)")
    
    if best['cov_improvement'] < 1.5:
        print("\nVERDICT: HF regularization is NOT working effectively.")
        print("Even aggressive hyperparams don't move the needle >1.5x.")
        print("Next step: Try explicit smoothed target training.")
    elif best['var_ratio'] < 0.3:
        print("\nVERDICT: Over-smoothed. Variance collapsed.")
        print("Model is learning near-constant function. Reduce lambda or sigma.")
    else:
        print("\nVERDICT: HF CAN work with these hyperparams!")
        print(f"Retrain with sigma={best['sigma']}, lambda={best['lambda']} for full 5000 steps.")
    
    return results


if __name__ == '__main__':
    results = run_sweep()
