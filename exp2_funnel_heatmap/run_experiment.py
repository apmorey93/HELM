"""
Main runner for Experiment 2: Funnel Heatmap.

Trains energy models and generates visualizations.
"""

import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from common.config import Exp2Config
from train_flat import train_flat_energy
from train_helm import train_helm_energy
from visualize_heatmap import visualize_experiment_2


def run_experiment_2(train=True, visualize=True):
    """
    Run complete Experiment 2 pipeline.
    
    Args:
        train: whether to train models
        visualize: whether to generate plots
    """
    config = Exp2Config()
    device = 'cuda' if __import__('torch').cuda.is_available() else 'cpu'
    
    print("="*70)
    print(" EXPERIMENT 2: FUNNEL HEATMAP ".center(70, "="))
    print("="*70)
    print(f"\nDevice: {device}\n")
    
    os.makedirs('checkpoints', exist_ok=True)
    os.makedirs('plots', exist_ok=True)
    
    # ===== TRAINING =====
    if train:
        print("\n" + "="*70)
        print(" PHASE 1: TRAINING MODELS ".center(70, "="))
        print("="*70)
        
        # Train flat model
        flat_model = train_flat_energy(config, device)
        __import__('torch').save(flat_model.state_dict(), 'checkpoints/flat_energy.pt')
        print("Saved flat model\n")
        
        # Train HELM model
        helm_model = train_helm_energy(config, device)
        __import__('torch').save(helm_model.state_dict(), 'checkpoints/helm_energy.pt')
        print("Saved HELM model\n")
    
    # ===== VISUALIZATION =====
    if visualize:
        print("\n" + "="*70)
        print(" PHASE 2: VISUALIZATION & ANALYSIS ".center(70, "="))
        print("="*70)
        
        results = visualize_experiment_2(config, device)
        
        print("\n" + "="*70)
        print(" EXPERIMENT 2 COMPLETE! ".center(70, "="))
        print("="*70)
        
        return results


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Run Experiment 2: Funnel Heatmap')
    parser.add_argument('--no-train', action='store_true', help='Skip training')
    parser.add_argument('--no-viz', action='store_true', help='Skip visualization')
    
    args = parser.parse_args()
    
    run_experiment_2(train=not args.no_train, visualize=not args.no_viz)
