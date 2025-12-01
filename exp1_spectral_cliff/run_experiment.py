"""
Main runner for Experiment 1: Spectral Cliff.

Trains models, runs analysis, and generates plots.
"""

import torch
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from common.config import Exp1Config
from common.viz_utils import plot_spectral_cliff, plot_convergence_rate
from train_flat import train_flat_model
from train_helm import train_helm_model
from analyze_spectrum import analyze_spectral_norm, test_convergence_rate
from models import FlatWorldModel, HELMWorldModel


def run_experiment_1(train=True, analyze=True, plot=True):
    """
    Run complete Experiment 1 pipeline.
    
    Args:
        train: whether to train models (or load from checkpoints)
        analyze: whether to run spectral analysis
        plot: whether to generate plots
    """
    config = Exp1Config()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print("="*70)
    print(" EXPERIMENT 1: SPECTRAL CLIFF ".center(70, "="))
    print("="*70)
    print(f"\nDevice: {device}")
    print(f"Horizons: {config.horizons}\n")
    
    os.makedirs('checkpoints', exist_ok=True)
    os.makedirs('plots', exist_ok=True)
    
    # ===== TRAINING =====
    if train:
        print("\n" + "="*70)
        print(" PHASE 1: TRAINING MODELS ".center(70, "="))
        print("="*70)
        
        for T in config.horizons:
            # Train flat model
            flat_model = train_flat_model(T, config, device)
            torch.save(flat_model.state_dict(), f'checkpoints/flat_model_T{T}.pt')
            
            # Train HELM model
            helm_model = train_helm_model(T, config, device)
            torch.save(helm_model.state_dict(), f'checkpoints/helm_model_T{T}.pt')
    
    # ===== ANALYSIS =====
    if analyze:
        print("\n" + "="*70)
        print(" PHASE 2: SPECTRAL ANALYSIS ".center(70, "="))
        print("="*70)
        
        results_spectral_flat = []
        results_spectral_helm = []
        results_convergence_flat = []
        results_convergence_helm = []
        
        for T in config.horizons:
            print(f"\n--- Horizon T={T} ---\n")
            
            # Load models
            flat_model = FlatWorldModel(d_s=3, d_a=1).to(device)
            helm_model = HELMWorldModel(d_s=3, d_a=1).to(device)
            
            flat_model.load_state_dict(torch.load(f'checkpoints/flat_model_T{T}.pt'))
            helm_model.load_state_dict(torch.load(f'checkpoints/helm_model_T{T}.pt'))
            
            flat_model.eval()
            helm_model.eval()
            
            # Spectral norm
            print("Spectral Norm Analysis:")
            mean_flat, std_flat = analyze_spectral_norm(
                flat_model, T, config.num_test_samples, device, config
            )
            mean_helm, std_helm = analyze_spectral_norm(
                helm_model, T, config.num_test_samples, device, config
            )
            
            results_spectral_flat.append((T, mean_flat))
            results_spectral_helm.append((T, mean_helm))
            
            # Convergence rate
            print("\nConvergence Rate Analysis:")
            iters_flat, _ = test_convergence_rate(flat_model, T, 20, device=device, config=config)
            iters_helm, _ = test_convergence_rate(helm_model, T, 20, device=device, config=config)
            
            results_convergence_flat.append((T, iters_flat))
            results_convergence_helm.append((T, iters_helm))
        
        # Save results
        results = {
            'spectral_flat': results_spectral_flat,
            'spectral_helm': results_spectral_helm,
            'convergence_flat': results_convergence_flat,
            'convergence_helm': results_convergence_helm
        }
        torch.save(results, 'analysis_results.pt')
        print("\nResults saved to analysis_results.pt")
    else:
        # Load existing results
        results = torch.load('analysis_results.pt')
        results_spectral_flat = results['spectral_flat']
        results_spectral_helm = results['spectral_helm']
        results_convergence_flat = results['convergence_flat']
        results_convergence_helm = results['convergence_helm']
    
    # ===== PLOTTING =====
    if plot:
        print("\n" + "="*70)
        print(" PHASE 3: GENERATING PLOTS ".center(70, "="))
        print("="*70 + "\n")
        
        # Plot spectral cliff
        plot_spectral_cliff(
            results_spectral_flat,
            results_spectral_helm,
            save_path='plots/exp1_spectral_cliff.png'
        )
        
        # Plot convergence rate
        plot_convergence_rate(
            results_convergence_flat,
            results_convergence_helm,
            save_path='plots/exp1_convergence_rate.png'
        )
        
        print("\nAll plots saved to plots/ directory")
    
    # ===== SUMMARY =====
    print("\n" + "="*70)
    print(" SUMMARY ".center(70, "="))
    print("="*70)
    
    print(f"\n{'T':>5} | {'Flat ||J||_F':>15} | {'HELM ||J||_F':>15} | {'Reduction':>12}")
    print("-" * 60)
    for (T_f, sigma_f), (T_h, sigma_h) in zip(results_spectral_flat, results_spectral_helm):
        ratio = sigma_f / sigma_h if sigma_h > 0 else float('inf')
        print(f"{T_f:>5} | {sigma_f:>12.3f} | {sigma_h:>12.3f} | {ratio:>10.2f}x")
    
    print("\n" + "="*70)
    print(" EXPERIMENT 1 COMPLETE! ".center(70, "="))
    print("="*70 + "\n")


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Run Experiment 1: Spectral Cliff')
    parser.add_argument('--no-train', action='store_true', help='Skip training (load from checkpoints)')
    parser.add_argument('--no-analyze', action='store_true', help='Skip analysis (load results)')
    parser.add_argument('--no-plot', action='store_true', help='Skip plotting')
    
    args = parser.parse_args()
    
    run_experiment_1(
        train=not args.no_train,
        analyze=not args.no_analyze,
        plot=not args.no_plot
    )
