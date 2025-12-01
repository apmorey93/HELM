"""
Fast training version for Exp 1 - reduced dataset for quick iteration.

Changes from original:
- 1000 trajectories instead of 10000
- 10 epochs instead of 50
- Goal: Get models trained in ~2-3 minutes to validate spectral cliff

This is for validation. Once we confirm the effect, can do full training.
"""

import torch
import torch.nn as nn
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from common.config import Exp1Config
from dataset import generate_controlled_lorenz_trajectories
from models import FlatWorldModel, HELMWorldModel, compute_spectral_regularization


def fast_train_flat(config, device='cpu'):
    """Fast training for Flat model."""
    print("\n" + "="*70)
    print(" FAST TRAINING: Flat Model ".center(70, "="))
    print("="*70)
    
    # Reduced data
    print("\nGenerating 1000 training trajectories (fast mode)...")
    dataset = generate_controlled_lorenz_trajectories(
        n_trajectories=1000,
        T=config.trajectory_length,
        config=config
    )
    
    model = FlatWorldModel(config.d_state, config.d_action, config.d_hidden).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    loss_fn = nn.MSELoss()
    
    states, actions, next_states = dataset
    states = states.to(device)
    actions = actions.to(device)
    next_states = next_states.to(device)
    
    print(f"Training for 10 epochs (fast mode)...")
    
    for epoch in range(10):  # Reduced from 50
        total_loss = 0.0
        n_batches = len(states) // config.batch_size
        
        for i in range(n_batches):
            start_idx = i * config.batch_size
            end_idx = start_idx + config.batch_size
            
            s_batch = states[start_idx:end_idx]
            a_batch = actions[start_idx:end_idx]
            s_next_batch = next_states[start_idx:end_idx]
            
            optimizer.zero_grad()
            s_pred = model.step(s_batch, a_batch)
            loss = loss_fn(s_pred, s_next_batch)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / n_batches
        if (epoch + 1) % 2 == 0:
            print(f"Epoch {epoch+1}/10: loss={avg_loss:.6f}")
    
    # Save
    os.makedirs('checkpoints', exist_ok=True)
    torch.save(model.state_dict(), 'checkpoints/flat_world_model.pt')
    print(f"\nSaved model. Final loss: {avg_loss:.6f}")
    
    return model


def fast_train_helm(config, device='cpu'):
    """Fast training for HELM model."""
    print("\n" + "="*70)
    print(" FAST TRAINING: HELM Model ".center(70, "="))
    print("="*70)
    
    # Reduced data
    print("\nGenerating 1000 training trajectories (fast mode)...")
    dataset = generate_controlled_lorenz_trajectories(
        n_trajectories=1000,
        T=config.trajectory_length,
        config=config
    )
    
    model = HELMWorldModel(config.d_state, config.d_action, config.d_hidden).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    loss_fn = nn.MSELoss()
    
    states, actions, next_states = dataset
    states = states.to(device)
    actions = actions.to(device)
    next_states = next_states.to(device)
    
    print(f"Training for 10 epochs with lambda_spec={config.lambda_spectral}...")
    
    for epoch in range(10):  # Reduced from 50
        total_mse = 0.0
        total_spec = 0.0
        n_batches = len(states) // config.batch_size
        
        for i in range(n_batches):
            start_idx = i * config.batch_size
            end_idx = start_idx + config.batch_size
            
            s_batch = states[start_idx:end_idx]
            a_batch = actions[start_idx:end_idx]
            s_next_batch = next_states[start_idx:end_idx]
            
            optimizer.zero_grad()
            
            # MSE
            s_pred = model.step(s_batch, a_batch)
            mse_loss = loss_fn(s_pred, s_next_batch)
            
            # Spectral penalty
            spec_loss = compute_spectral_regularization(model, s_batch, a_batch, horizon=5)
            
            # Combined
            loss = mse_loss + config.lambda_spectral * spec_loss
            loss.backward()
            optimizer.step()
            
            total_mse += mse_loss.item()
            total_spec += spec_loss.item()
        
        avg_mse = total_mse / n_batches
        avg_spec = total_spec / n_batches
        
        if (epoch + 1) % 2 == 0:
            spec_contrib = (config.lambda_spectral * avg_spec) / (avg_mse + config.lambda_spectral * avg_spec) * 100
            print(f"Epoch {epoch+1}/10: mse={avg_mse:.6f}, spec={avg_spec:.6f}, spec%={spec_contrib:.1f}%")
    
    # Save
    os.makedirs('checkpoints', exist_ok=True)
    torch.save(model.state_dict(), 'checkpoints/helm_world_model.pt')
    print(f"\nSaved model. Final - MSE: {avg_mse:.6f}, Spec: {avg_spec:.6f}")
    
    return model


if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}\n")
    
    config = Exp1Config()
    
    # Train both models
    flat_model = fast_train_flat(config, device)
    helm_model = fast_train_helm(config, device)
    
    print("\n" + "="*70)
    print(" FAST TRAINING COMPLETE ".center(70, "="))
    print("="*70)
    print("\nRun: python validate_exp1.py")
