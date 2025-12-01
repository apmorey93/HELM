"""
Training script for Flat energy model (Experiment 2).

Learns to fit true energy (including ruggedness).
"""

import torch
import torch.nn as nn
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from common.config import Exp2Config
from energy_functions import true_energy
from models import EnergyMLP


def train_flat_energy(config=None, device='cuda'):
    """
    Train flat energy model on true energy landscape.
    
    Args:
        config: Exp2Config
        device: torch device
        
    Returns:
        trained EnergyMLP
    """
    if config is None:
        config = Exp2Config()
    
    print("\n" + "="*60)
    print(" Training Flat Energy Model ".center(60, "="))
    print("="*60 + "\n")
    
    # Initialize model
    model = EnergyMLP(hidden_dim=128).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    loss_fn = nn.MSELoss()
    
    # Training loop
    print(f"Training for {config.num_training_steps} steps...")
    
    for step in range(config.num_training_steps):
        # Sample random points in domain
        a = (torch.rand(config.batch_size, 2, device=device) - 0.5) * 2 * config.domain_range
        
        # Ground truth energy
        with torch.no_grad():
            E_true = true_energy(a, config)
        
        # Predict
        E_pred = model(a)
        
        # Loss
        loss = loss_fn(E_pred, E_true)
        
        # Update
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if (step + 1) % 500 == 0 or step == 0:
            print(f"Step {step+1}/{config.num_training_steps}: loss={loss.item():.6f}")
    
    print(f"\nTraining complete! Final loss: {loss.item():.6f}")
    return model


if __name__ == '__main__':
    config = Exp2Config()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    model = train_flat_energy(config, device)
    
    # Save model
    os.makedirs('checkpoints', exist_ok=True)
    torch.save(model.state_dict(), 'checkpoints/flat_energy.pt')
    print("Saved model to checkpoints/flat_energy.pt")
