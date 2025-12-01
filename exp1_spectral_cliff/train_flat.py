"""
Training script for Flat world model (Experiment 1).

No regularization => spectral explosion expected.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from common.config import Exp1Config
from dataset import generate_controlled_lorenz_trajectories
from models import FlatWorldModel


def train_flat_model(T, config=None, device='cuda'):
    """
    Train flat world model for given horizon T.
    
    Args:
        T: planning horizon
        config: Exp1Config instance
        device: torch device
        
    Returns:
        trained FlatWorldModel
    """
    if config is None:
        config = Exp1Config()
    
    print(f"\n{'='*60}")
    print(f"Training Flat Model for horizon T={T}")
    print(f"{'='*60}")
    
    # Generate training data
    print(f"Generating {config.num_trajectories} training trajectories...")
    states, actions = generate_controlled_lorenz_trajectories(
        num_traj=config.num_trajectories,
        T=T,
        dt=config.dt,
        sigma=config.sigma,
        rho=config.rho,
        beta=config.beta,
        control_coupling=config.control_coupling,
        device=device
    )
    
    s0 = states[:, 0]  # (N, 3)
    sT = states[:, -1]  # (N, 3)
    
    # Create dataloader
    dataset = TensorDataset(s0, actions, sT)
    dataloader = DataLoader(
        dataset, 
        batch_size=config.batch_size, 
        shuffle=True,
        drop_last=True
    )
    
    # Initialize model
    model = FlatWorldModel(d_s=3, d_a=1).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    loss_fn = nn.MSELoss()
    
    # Training loop
    print(f"Training for {config.epochs} epochs...")
    for epoch in range(config.epochs):
        total_loss = 0.0
        num_batches = 0
        
        for batch_s0, batch_actions, batch_sT in dataloader:
            # Forward pass
            pred_sT = model(batch_s0, batch_actions)
            loss = loss_fn(pred_sT, batch_sT)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        avg_loss = total_loss / num_batches
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"Epoch {epoch+1}/{config.epochs}: loss={avg_loss:.6f}")
    
    print(f"Training complete! Final loss: {avg_loss:.6f}")
    return model


def main():
    """Train flat models for all horizons."""
    config = Exp1Config()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}\n")
    
    models = {}
    
    for T in config.horizons:
        model = train_flat_model(T, config, device)
        models[T] = model
        
        # Save model
        os.makedirs('checkpoints', exist_ok=True)
        save_path = f'checkpoints/flat_model_T{T}.pt'
        torch.save(model.state_dict(), save_path)
        print(f"Saved model to {save_path}\n")
    
    print("All flat models trained successfully!")
    return models


if __name__ == '__main__':
    main()
