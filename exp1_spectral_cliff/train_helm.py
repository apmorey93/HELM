"""
Training script for HELM world model (Experiment 1).

WITH spectral regularization => bounded gradients expected.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from common.config import Exp1Config
from dataset import generate_controlled_lorenz_trajectories
from models import HELMWorldModel, compute_spectral_regularization


def train_helm_model(T, config=None, device='cuda'):
    """
    Train HELM world model with spectral regularization for given horizon T.
    
    Args:
        T: planning horizon
        config: Exp1Config instance
        device: torch device
        
    Returns:
        trained HELMWorldModel
    """
    if config is None:
        config = Exp1Config()
    
    print(f"\n{'='*60}")
    print(f"Training HELM Model for horizon T={T}")
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
    
    s0 = states[:, 0]
    sT = states[:, -1]
    
    # Create dataloader
    dataset = TensorDataset(s0, actions, sT)
    dataloader = DataLoader(
        dataset, 
        batch_size=config.batch_size, 
        shuffle=True,
        drop_last=True
    )
    
    # Initialize model
    model = HELMWorldModel(d_s=3, d_a=1).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    loss_fn = nn.MSELoss()
    
    # Training loop
    print(f"Training for {config.epochs} epochs with lambda_spec={config.lambda_spectral}...")
    for epoch in range(config.epochs):
        total_mse = 0.0
        total_spec = 0.0
        total_loss = 0.0
        num_batches = 0
        
        for batch_s0, batch_actions, batch_sT in dataloader:
            # CRITICAL: Enable gradients for spectral regularization
            batch_s0 = batch_s0.detach().requires_grad_(True)
            batch_actions = batch_actions.detach().requires_grad_(True)
            
            # Forward pass
            pred_sT = model(batch_s0, batch_actions)
            mse_loss = loss_fn(pred_sT, batch_sT)
            
            # Spectral regularization: penalize large Jacobian
            spec_penalty = compute_spectral_regularization(
                model, batch_s0, batch_actions
            )
            
            # Combined loss
            loss = mse_loss + config.lambda_spectral * spec_penalty
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_mse += mse_loss.item()
            total_spec += spec_penalty.item()
            total_loss += loss.item()
            num_batches += 1
        
        avg_mse = total_mse / num_batches
        avg_spec = total_spec / num_batches
        avg_loss = total_loss / num_batches
        
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"Epoch {epoch+1}/{config.epochs}: "
                  f"loss={avg_loss:.6f}, mse={avg_mse:.6f}, spec={avg_spec:.6f}")
    
    print(f"Training complete!")
    print(f"Final - MSE: {avg_mse:.6f}, Spectral: {avg_spec:.6f}")
    return model


def main():
    """Train HELM models for all horizons."""
    config = Exp1Config()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}\n")
    
    models = {}
    
    for T in config.horizons:
        model = train_helm_model(T, config, device)
        models[T] = model
        
        # Save model
        os.makedirs('checkpoints', exist_ok=True)
        save_path = f'checkpoints/helm_model_T{T}.pt'
        torch.save(model.state_dict(), save_path)
        print(f"Saved model to {save_path}\n")
    
    print("All HELM models trained successfully!")
    return models


if __name__ == '__main__':
    main()
