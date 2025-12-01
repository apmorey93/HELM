"""
CRITICAL DIAGNOSTICS for Experiment 2 Failure.

Current status: HELM coverage 4.5% vs Flat 4.3% = FAILURE
Before touching HELM hyperparams, validate the basin test itself works.

Step 0: Can GD even recover the true minimum?
Step 1: Variance diagnostic - is HELM smoothing or flattening?
Step 2: 1D slice - visual confirmation of smoothing
Step 3: Systematic HF sweep
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from common.config import Exp2Config
from energy_functions import true_energy, quadratic_U
from models import EnergyMLP


def gradient_descent_test(energy_fn, a_init, lr=0.1, steps=100, device='cpu'):
    """Run GD and return final point."""
    a = a_init.clone().detach().requires_grad_(True)
    optimizer = torch.optim.SGD([a], lr=lr)
    
    for _ in range(steps):
        optimizer.zero_grad()
        E = energy_fn(a).sum()
        E.backward()
        optimizer.step()
    
    return a.detach()


def basin_coverage_test(energy_fn, a_true, grid_res=50, lr=0.1, steps=100, 
                        threshold=0.5, device='cpu'):
    """Compute basin coverage for any energy function."""
    x = torch.linspace(-3, 3, grid_res, device=device)
    y = torch.linspace(-3, 3, grid_res, device=device)
    
    X, Y = torch.meshgrid(x, y, indexing='xy')
    grid = torch.stack([X, Y], dim=-1).reshape(-1, 2)
    
    # Run GD from each point
    finals = gradient_descent_test(energy_fn, grid, lr=lr, steps=steps, device=device)
    
    # Distance to true optimum
    a_true_t = torch.tensor(a_true, device=device, dtype=torch.float32)
    dist = torch.norm(finals - a_true_t, dim=-1)
    
    coverage = (dist < threshold).float().mean().item()
    
    return coverage, dist.reshape(grid_res, grid_res)


def step0_sanity_check_basin_test(config, device='cpu'):
    """
    STEP 0: Verify GD can recover minimum on known functions.
    
    Test on:
    1. U(a) - pure quadratic (should get ~100% coverage)
    2. E_true(a) - quadratic + sin noise (coverage should drop but be reasonable)
    """
    print("\n" + "="*70)
    print(" STEP 0: BASIN TEST SANITY CHECK ".center(70, "="))
    print("="*70)
    
    a_opt = config.a_optimum
    
    # Test 1: Pure quadratic U(a)
    print("\n1. Testing on PURE QUADRATIC U(a)...")
    print("   Expected: ~80-100% coverage (smooth convex bowl)")
    
    Q = torch.diag(torch.tensor(config.Q_eigenvalues, device=device, dtype=torch.float32))
    a0 = torch.tensor(config.a_optimum, device=device, dtype=torch.float32)
    
    def U_fn(a):
        return quadratic_U(a, a0, Q)
    
    coverage_U, _ = basin_coverage_test(
        U_fn, a_opt, grid_res=50, lr=0.1, steps=100, device=device
    )
    
    print(f"   Result: {coverage_U*100:.1f}%")
    
    if coverage_U < 0.5:
        print("   !!! CRITICAL FAILURE !!!")
        print("   GD cannot even solve a pure quadratic.")
        print("   Diagnosis: lr too high/low, steps too few, or GD is broken")
        print("   FIX THIS BEFORE TOUCHING HELM")
        return False
    else:
        print("   OK: GD works on convex problems")
    
    # Test 2: True energy E(a) = U(a) + psi(a)
    print("\n2. Testing on TRUE ENERGY E_true(a) = U(a) + sin_noise...")
    print("   Expected: Lower than U(a) due to local minima, but >10%")
    
    def E_true_fn(a):
        return true_energy(a, config)
    
    coverage_true, _ = basin_coverage_test(
        E_true_fn, a_opt, grid_res=50, lr=0.1, steps=100, device=device
    )
    
    print(f"   Result: {coverage_true*100:.1f}%")
    
    if coverage_true < 0.05:
        print("   WARNING: Coverage very low (<5%)")
        print("   Either:")
        print("   - Sin traps are too strong (reduce amplitudes)")
        print("   - GD hyperparams need tuning (try lr=0.05, steps=200)")
    else:
        print("   OK: GD works despite ruggedness")
    
    # Summary
    print("\n" + "-"*70)
    print("STEP 0 SUMMARY:")
    print(f"  U(a) coverage:     {coverage_U*100:.1f}%")
    print(f"  E_true(a) coverage: {coverage_true*100:.1f}%")
    print(f"  Flat model (prev): 4.3%")
    print(f"  HELM model (prev): 4.5%")
    
    if coverage_U < 0.5:
        print("\n  >> GD is broken. Fix lr/steps first.")
        return False
    elif coverage_true < 0.03:
        print("\n  >> GD works but energy is too hard. Consider easier landscape.")
        return False
    elif coverage_true < 0.1:
        print("\n  >> GD barely works. Models at 4% means they're not learning E_true well.")
        return True
    else:
        print("\n  >> Huge gap: True energy gets {:.1f}%, models get 4%".format(coverage_true*100))
        print("  >> Models are NOT approximating E_true properly!")
        return True


def step1_variance_diagnostic(flat_model, helm_model, config, device='cpu'):
    """
    STEP 1: Check if HELM is smoothing or just flattening.
    
    Compute Var[E(a)] over random samples:
    - If Var(HELM) << Var(flat): over-smoothed to mush
    - If Var(HELM) ~ Var(flat): not smoothing enough
    - Sweet spot: Var(HELM) slightly < Var(flat)
    """
    print("\n" + "="*70)
    print(" STEP 1: VARIANCE DIAGNOSTIC ".center(70, "="))
    print("="*70)
    
    # Sample random actions
    num_samples = 5000
    a_samples = (torch.rand(num_samples, 2, device=device) - 0.5) * 6.0
    
    with torch.no_grad():
        E_true = true_energy(a_samples, config).cpu().numpy()
        E_flat = flat_model(a_samples).cpu().numpy()
        E_helm = helm_model(a_samples).cpu().numpy()
    
    var_true = E_true.var()
    var_flat = E_flat.var()
    var_helm = E_helm.var()
    
    print(f"\nVariance over {num_samples} random samples:")
    print(f"  True energy:  {var_true:.4f}")
    print(f"  Flat model:   {var_flat:.4f}")
    print(f"  HELM model:   {var_helm:.4f}")
    print(f"\nRatios:")
    print(f"  Var(Flat) / Var(True):  {var_flat/var_true:.3f}")
    print(f"  Var(HELM) / Var(True):  {var_helm/var_true:.3f}")
    print(f"  Var(HELM) / Var(Flat):  {var_helm/var_flat:.3f}")
    
    # Diagnosis
    print("\nDiagnosis:")
    if var_helm / var_flat < 0.5:
        print("  >> HELM variance MUCH lower than Flat")
        print("  >> Risk: Over-smoothed into near-constant function")
        print("  >> Action: Reduce lambda_HF or sigma")
    elif var_helm / var_flat > 0.9:
        print("  >> HELM variance SAME as Flat")
        print("  >> HF loss is not smoothing effectively")
        print("  >> Action: Increase lambda_HF or sigma")
    else:
        print("  >> HELM variance slightly lower than Flat (good range)")
        print("  >> Smoothing is happening, but maybe not enough for basin")


def step2_1d_slice_plot(flat_model, helm_model, config, device='cpu'):
    """
    STEP 2: 1D slice through action space.
    
    Plot E(a) along line from off-optimum to optimum.
    Should show:
    - True/Flat: sinusoidal wiggles
    - HELM: wiggles attenuated if smoothing works
    """
    print("\n" + "="*70)
    print(" STEP 2: 1D SLICE VISUALIZATION ".center(70, "="))
    print("="*70)
    
    a_opt = torch.tensor(config.a_optimum, device=device, dtype=torch.float32)
    
    # Direction: from (-2, -2) to optimum
    a_start = torch.tensor([-2.0, -2.0], device=device)
    direction = a_opt - a_start
    
    # Parameterize line
    t_vals = torch.linspace(0, 1.5, 200, device=device)
    a_line = a_start.unsqueeze(0) + t_vals.unsqueeze(1) * direction.unsqueeze(0)
    
    with torch.no_grad():
        E_true = true_energy(a_line, config).cpu().numpy()
        E_flat = flat_model(a_line).cpu().numpy()
        E_helm = helm_model(a_line).cpu().numpy()
    
    plt.figure(figsize=(10, 6))
    plt.plot(t_vals.cpu(), E_true, label='True Energy', linewidth=2, alpha=0.7)
    plt.plot(t_vals.cpu(), E_flat, label='Flat Model', linewidth=2, alpha=0.7)
    plt.plot(t_vals.cpu(), E_helm, label='HELM Model', linewidth=2, alpha=0.7)
    
    plt.axvline(x=1.0, color='black', linestyle='--', alpha=0.5, label='Optimum (t=1)')
    plt.xlabel('t (0=start, 1=optimum)', fontsize=12)
    plt.ylabel('Energy E(a)', fontsize=12)
    plt.title('1D Slice: Start to Optimum', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    os.makedirs('diagnostics', exist_ok=True)
    plt.savefig('diagnostics/1d_slice.png', dpi=150)
    print("\nSaved 1D slice to diagnostics/1d_slice.png")
    plt.close()
    
    # Compute smoothness metric: std of derivative
    deriv_true = np.diff(E_true)
    deriv_flat = np.diff(E_flat)
    deriv_helm = np.diff(E_helm)
    
    print(f"\nSmoothness (std of derivative):")
    print(f"  True: {deriv_true.std():.4f}")
    print(f"  Flat: {deriv_flat.std():.4f}")
    print(f"  HELM: {deriv_helm.std():.4f}")
    
    if deriv_helm.std() < deriv_flat.std() * 0.8:
        print("  >> HELM is smoother than Flat (good!)")
    else:
        print("  >> HELM smoothness same as Flat (HF not working!)")


def step3_hf_sweep(config, device='cpu'):
    """
    STEP 3: Systematic HF hyperparameter sweep.
    
    Test combinations of (sigma, lambda_HF) and measure:
    - MSE
    - HF loss
    - Variance
    - Basin coverage
    """
    print("\n" + "="*70)
    print(" STEP 3: HF HYPERPARAMETER SWEEP ".center(70, "="))
    print("="*70)
    
    from train_helm import train_helm_energy
    
    # Sweep configs
    sigma_vals = [0.2, 0.5]
    lambda_vals = [1.0, 5.0, 10.0]
    
    print(f"\nTesting {len(sigma_vals) * len(lambda_vals)} configs:")
    print(f"  sigma: {sigma_vals}")
    print(f"  lambda_HF: {lambda_vals}")
    print("\nRunning abbreviated training (2000 steps each)...\n")
    
    results = []
    
    for sigma in sigma_vals:
        for lambda_hf in lambda_vals:
            print(f"\n--- Config: sigma={sigma}, lambda_HF={lambda_hf} ---")
            
            # Modify config
            config_test = Exp2Config()
            config_test.hf_noise_sigma = sigma
            config_test.lambda_hf = lambda_hf
            config_test.num_training_steps = 2000  # Abbreviated
            
            # Train
            model = train_helm_energy(config_test, device)
            
            # Evaluate
            a_samples = (torch.rand(1000, 2, device=device) - 0.5) * 6.0
            with torch.no_grad():
                E_pred = model(a_samples).cpu().numpy()
            
            var_pred = E_pred.var()
            
            # Basin coverage
            def energy_fn(a):
                with torch.no_grad():
                    return model(a)
            
            coverage, _ = basin_coverage_test(
                energy_fn, config.a_optimum, grid_res=50, device=device
            )
            
            results.append({
                'sigma': sigma,
                'lambda': lambda_hf,
                'variance': var_pred,
                'coverage': coverage * 100
            })
            
            print(f"  Variance: {var_pred:.4f}")
            print(f"  Coverage: {coverage*100:.1f}%")
    
    # Summary table
    print("\n" + "="*70)
    print("SWEEP RESULTS:")
    print("-"*70)
    print(f"{'sigma':>8} | {'lambda':>8} | {'Var':>10} | {'Coverage':>10}")
    print("-"*70)
    for r in results:
        print(f"{r['sigma']:>8.1f} | {r['lambda']:>8.1f} | {r['variance']:>10.4f} | {r['coverage']:>9.1f}%")
    
    # Find best
    best = max(results, key=lambda x: x['coverage'])
    print("-"*70)
    print(f"Best config: sigma={best['sigma']}, lambda={best['lambda']}")
    print(f"  Coverage: {best['coverage']:.1f}%")
    
    if best['coverage'] < 10:
        print("\nCONCLUSION: HF regularization is NOT working.")
        print("Even best config gives <10% coverage.")
        print("Next step: Try explicit smoothed targets (Step 4 in plan)")
    else:
        print(f"\nCONCLUSION: HF CAN work with right hyperparams!")
        print(f"Retrain with sigma={best['sigma']}, lambda={best['lambda']}")


def run_all_diagnostics():
    """Run complete diagnostic suite."""
    config = Exp2Config()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # STEP 0: Sanity check
    gd_ok = step0_sanity_check_basin_test(config, device)
    
    if not gd_ok:
        print("\n!!! STOP: Fix GD hyperparams before continuing !!!")
        return
    
    # Load trained models
    print("\nLoading trained Flat and HELM models...")
    flat_model = EnergyMLP().to(device)
    helm_model = EnergyMLP().to(device)
    
    flat_model.load_state_dict(torch.load('checkpoints/flat_energy.pt', map_location=device))
    helm_model.load_state_dict(torch.load('checkpoints/helm_energy.pt', map_location=device))
    
    flat_model.eval()
    helm_model.eval()
    
    # STEP 1: Variance
    step1_variance_diagnostic(flat_model, helm_model, config, device)
    
    # STEP 2: 1D slice
    step2_1d_slice_plot(flat_model, helm_model, config, device)
    
    # STEP 3: HF sweep
    step3_hf_sweep(config, device)


if __name__ == '__main__':
    run_all_diagnostics()
