"""
Diagnostic script for Exp 3 World Models.
Measures 1-step prediction error and rollout error vs horizon.
"""
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from train import collect_data, prepare_batch, make_env
from models import Encoder, TransitionL0, TransitionL1
import os

def evaluate_model_errors(encoder, f0, data_l0, device='cpu', max_horizon=20):
    encoder.eval()
    f0.eval()
    
    # 1. One-step Error
    total_mse = 0
    count = 0
    
    # Use a subset for speed
    indices = np.random.choice(len(data_l0), min(1000, len(data_l0)), replace=False)
    batch = [data_l0[i] for i in indices]
    o_t, a_t, o_next = prepare_batch(batch, device)
    
    with torch.no_grad():
        s_t = encoder(o_t)
        s_next_true = encoder(o_next)
        s_next_pred = f0(s_t, a_t)
        mse = F.mse_loss(s_next_pred, s_next_true)
        
    print(f"1-step MSE: {mse.item():.6f}")
    
    # 2. Rollout Error
    # We need full episodes for rollout, not just transitions.
    # So we'll collect new episodes and roll them out.
    print("Measuring rollout error...")
    rollout_errors = {k: [] for k in range(1, max_horizon+1)}
    
    env = make_env()
    num_eval_eps = 10
    
    for ep in range(num_eval_eps):
        obs, _ = env.reset()
        done = False
        
        # Collect full trajectory
        traj_obs = [obs]
        traj_actions = []
        
        while not done and len(traj_actions) < max_horizon + 5:
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            traj_obs.append(obs)
            traj_actions.append(action)
            
        # Perform rollouts from each step
        T = len(traj_actions)
        for t in range(T - max_horizon):
            # Start state
            o_start = traj_obs[t]
            
            # Encode start
            o_tensor = torch.tensor(o_start, dtype=torch.float32).permute(2, 0, 1) / 255.0
            o_tensor = o_tensor.unsqueeze(0).to(device)
            with torch.no_grad():
                s_curr = encoder(o_tensor)
            
            # Rollout
            for k in range(1, max_horizon + 1):
                # Action at t+k-1
                a_idx = traj_actions[t + k - 1]
                a_tensor = F.one_hot(torch.tensor([a_idx], device=device), num_classes=7).float()
                
                with torch.no_grad():
                    s_curr = f0(s_curr, a_tensor)
                    
                # True state at t+k
                o_true = traj_obs[t + k]
                o_true_tensor = torch.tensor(o_true, dtype=torch.float32).permute(2, 0, 1) / 255.0
                o_true_tensor = o_true_tensor.unsqueeze(0).to(device)
                with torch.no_grad():
                    s_true = encoder(o_true_tensor)
                
                # Error
                err = F.mse_loss(s_curr, s_true).item()
                rollout_errors[k].append(err)
                
    # Average errors
    avg_errors = {k: np.mean(v) for k, v in rollout_errors.items() if v}
    return avg_errors

if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Load Data (reuse collect_data)
    print("Collecting validation data...")
    data_l0, _ = collect_data(num_episodes=20)
    
    # Load Models
    print("Loading Flat Model...")
    flat_enc = Encoder().to(device)
    flat_f0 = TransitionL0().to(device)
    flat_ckpt = torch.load('flat_model.pt', map_location=device)
    flat_enc.load_state_dict(flat_ckpt['encoder'])
    flat_f0.load_state_dict(flat_ckpt['f0'])
    
    print("Evaluating Flat Model...")
    flat_errors = evaluate_model_errors(flat_enc, flat_f0, data_l0, device)
    
    print("Loading HELM Model...")
    helm_enc = Encoder().to(device)
    helm_f0 = TransitionL0().to(device)
    helm_ckpt = torch.load('helm_model.pt', map_location=device)
    helm_enc.load_state_dict(helm_ckpt['encoder'])
    helm_f0.load_state_dict(helm_ckpt['f0'])
    
    print("Evaluating HELM Model (L0)...")
    helm_errors = evaluate_model_errors(helm_enc, helm_f0, data_l0, device)
    
    # Print Summary
    print("\n" + "="*40)
    print(" ROLLOUT ERRORS (MSE) ".center(40, "="))
    print("="*40)
    print(f"{'Horizon':<10} | {'Flat':<12} | {'HELM':<12}")
    print("-" * 40)
    for k in [1, 5, 10, 20]:
        if k in flat_errors:
            print(f"{k:<10} | {flat_errors[k]:<12.6f} | {helm_errors[k]:<12.6f}")
