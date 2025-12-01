"""
Training script for HELM energy model (Experiment 2).

WITH high-frequency invariance => smooth energy landscape.
"""

import torch
import torch.nn as nn
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from common.config import Exp2Config
from energy_functions import true_energy
from models import EnergyMLP


def train_helm_energy(config=None, device='cuda'):
    """
    Train HELM energy model with HF invariance regularization.
    
    HF invariance loss: E(a) ≈ E(a + δ) for small δ
    This encourages the model to smooth out high-frequency noise.
    
    Args:
        config: Exp2Config
        device: torch device
        
    Returns:
        trained EnergyMLP
    """
    if config is None:
        config = Exp2Config()
    
    print("\n" + "="*60)
    print(" Training HELM Energy Model (HF Invariance) ".center(60, "="))
    print("="*60 + "\n")
    
    # Initialize model
    model = EnergyMLP(hidden_dim=128).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    loss_fn = nn.MSELoss()
    
    # Training loop
    print(f"Training for {config.num_training_steps} steps...")
    print(f"lambda_HF = {config.lambda_hf}, sigma_noise = {config.hf_noise_sigma}\n")
    
    for step in range(config.num_training_steps):
        # Sample random points
        a = (torch.rand(config.batch_size, 2, device=device) - 0.5) * 2 * config.domain_range
        
        # Ground truth
        with torch.no_grad():
            E_true = true_energy(a, config)
        
        # Main prediction
        E_pred = model(a)
        mse_loss = loss_fn(E_pred, E_true)
        
        # HF invariance: perturb input slightly
        delta = torch.randn_like(a) * config.hf_noise_sigma
        a_perturbed = a + delta
        E_pred_perturbed = model(a_perturbed)
        
        # Encourage invariance: E(a) ≈ E(a + δ)
        hf_loss = ((E_pred - E_pred_perturbed) ** 2).mean()
        
        # Combined loss
        loss = mse_loss + config.lambda_hf * hf_loss
        
        # Update
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if (step + 1) % 500 == 0 or step == 0:
            print(f"Step {step+1}/{config.num_training_steps}: "
                  f"loss={loss.item():.6f}, mse={mse_loss.item():.6f}, hf={hf_loss.item():.6f}")
    
    print(f"\nTraining complete!")
    print(f"Final - MSE: {mse_loss.item():.6f}, HF: {hf_loss.item():.6f}")
    return model


if __name__ == '__main__':
    config = Exp2Config()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    model = train_helm_energy(config, device)
    
    # Save model
    os.makedirs('checkpoints', exist_ok=True)
    torch.save(model.state_dict(), 'checkpoints/helm_energy.pt')
    print("Saved model to checkpoints/helm_energy.pt")
