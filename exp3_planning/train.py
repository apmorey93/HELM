"""
Environment setup and Training script for Exp 3.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import gymnasium as gym
from minigrid.wrappers import FullyObsWrapper, ImgObsWrapper
from models import Encoder, TransitionL0, TransitionL1
import os

# ==============================================================================
# 1. ENVIRONMENT
# ==============================================================================

def make_env(env_name="MiniGrid-DoorKey-8x8-v0"):
    env = gym.make(env_name)
    env = FullyObsWrapper(env) # Fully observable symbolic (8x8x3)
    env = ImgObsWrapper(env) # Get rid of 'mission' dict
    return env

def collect_data(num_episodes=500, stride=4, env_name="MiniGrid-DoorKey-8x8-v0"):
    print(f"Collecting {num_episodes} episodes from {env_name}...")
    env = make_env(env_name)
    
    data_l0 = [] # (o_t, a_t, o_t+1)
    data_l1 = [] # (o_t, u_t, o_t+4)
    
    for ep in range(num_episodes):
        obs, _ = env.reset()
        done = False
        episode_obs = [obs]
        episode_actions = []
        
        if (ep+1) % 1 == 0:
            print(f"Collected {ep+1}/{num_episodes} episodes", flush=True)
            
        while not done:
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            episode_obs.append(obs)
            episode_actions.append(action)
            
        # Process episode for L0
        T = len(episode_actions)
        for t in range(T):
            o_t = episode_obs[t]
            a_t = episode_actions[t]
            o_next = episode_obs[t+1]
            data_l0.append((o_t, a_t, o_next))
            
        # Process episode for L1
        for t in range(0, T - stride + 1):
            o_t = episode_obs[t]
            o_next4 = episode_obs[t+stride]
            
            # Average action one-hot
            actions_window = episode_actions[t:t+stride]
            u_avg = np.zeros(env.action_space.n)
            for a in actions_window:
                u_avg[a] += 1
            u_avg /= stride
            
            data_l1.append((o_t, u_avg, o_next4))
            
    print(f"Collected {len(data_l0)} L0 samples, {len(data_l1)} L1 samples")
    return data_l0, data_l1

def prepare_batch(batch_data, device):
    obs_t, actions, obs_next = zip(*batch_data)
    
    # Convert obs to tensor (B, 3, H, W)
    obs_t = torch.tensor(np.array(obs_t), dtype=torch.float32).permute(0, 3, 1, 2) / 255.0
    obs_next = torch.tensor(np.array(obs_next), dtype=torch.float32).permute(0, 3, 1, 2) / 255.0
    
    # Actions are either integers (L0) or float vectors (L1)
    if isinstance(actions[0], (int, np.integer)):
        # One-hot encode L0 actions
        actions = torch.tensor(actions, dtype=torch.long)
        actions = F.one_hot(actions, num_classes=7).float()
    else:
        # L1 actions are already vectors
        actions = torch.tensor(np.array(actions), dtype=torch.float32)
        
    return obs_t.to(device), actions.to(device), obs_next.to(device)


# ==============================================================================
# 2. TRAINING
# ==============================================================================

def train_flat(data_l0, device='cpu'):
    print("\n" + "="*50)
    print(" TRAINING FLAT MODEL ".center(50, "="))
    print("="*50)
    
    encoder = Encoder().to(device)
    f0 = TransitionL0().to(device)
    optimizer = torch.optim.Adam(list(encoder.parameters()) + list(f0.parameters()), lr=3e-4)
    
    batch_size = 128
    epochs = 30
    
    for epoch in range(epochs):
        indices = np.random.permutation(len(data_l0))
        total_loss = 0
        
        for i in range(0, len(data_l0), batch_size):
            batch_idx = indices[i:i+batch_size]
            batch = [data_l0[k] for k in batch_idx]
            o_t, a_t, o_next = prepare_batch(batch, device)
            
            optimizer.zero_grad()
            
            s_t = encoder(o_t)
            s_next_true = encoder(o_next).detach() # Detach target encoder
            s_next_pred = f0(s_t, a_t)
            
            loss = F.mse_loss(s_next_pred, s_next_true)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
        if (epoch+1) % 5 == 0:
            print(f"Epoch {epoch+1}: loss={total_loss / (len(data_l0)//batch_size):.6f}")
            
    return encoder, f0

def train_helm(data_l0, data_l1, device='cpu'):
    print("\n" + "="*50)
    print(" TRAINING HELM MODEL ".center(50, "="))
    print("="*50)
    
    encoder = Encoder().to(device)
    f0 = TransitionL0().to(device)
    f1 = TransitionL1().to(device)
    
    params = list(encoder.parameters()) + list(f0.parameters()) + list(f1.parameters())
    optimizer = torch.optim.Adam(params, lr=3e-4)
    
    batch_size = 128
    epochs = 30
    
    lambda_spec = 1e-3
    lambda_hf = 5.0
    
    # We iterate through both datasets. Since they have different sizes, we'll just loop 
    # based on L0 size and sample L1 randomly
    
    n_batches = len(data_l0) // batch_size
    
    for epoch in range(epochs):
        indices_l0 = np.random.permutation(len(data_l0))
        total_l0 = 0
        total_l1 = 0
        total_spec = 0
        total_hf = 0
        
        for i in range(n_batches):
            # L0 Batch
            idx0 = indices_l0[i*batch_size : (i+1)*batch_size]
            batch0 = [data_l0[k] for k in idx0]
            o0_t, a0_t, o0_next = prepare_batch(batch0, device)
            
            # L1 Batch (random sample)
            idx1 = np.random.choice(len(data_l1), batch_size)
            batch1 = [data_l1[k] for k in idx1]
            o1_t, u1_t, o1_next4 = prepare_batch(batch1, device)
            
            optimizer.zero_grad()
            
            # --- L0 Loss ---
            s0_t = encoder(o0_t)
            s0_next_true = encoder(o0_next).detach()
            s0_next_pred = f0(s0_t, a0_t)
            loss_l0 = F.mse_loss(s0_next_pred, s0_next_true)
            
            # --- L1 Loss ---
            s1_t = encoder(o1_t)
            s1_next4_true = encoder(o1_next4).detach()
            s1_next4_pred = f1(s1_t, u1_t)
            loss_l1 = F.mse_loss(s1_next4_pred, s1_next4_true)
            
            # --- Spectral Reg (on f1) ---
            s_spec = s1_t.detach().clone().requires_grad_(True)
            u_spec = u1_t.detach().clone().requires_grad_(True)
            s_out_spec = f1(s_spec, u_spec)
            
            grad_s = torch.autograd.grad(s_out_spec.sum(), s_spec, create_graph=True)[0]
            loss_spec = grad_s.pow(2).mean()
            
            # --- HF Invariance (on f1) ---
            noise_sigma = 0.3
            delta = torch.randn_like(u1_t) * noise_sigma
            delta = delta - delta.mean(dim=-1, keepdim=True) # Zero-sum
            u_pert = (u1_t + delta).clamp(0.0, 1.0)
            u_pert = u_pert / (u_pert.sum(dim=-1, keepdim=True) + 1e-8) # Re-normalize
            
            s_pert = f1(s1_t, u_pert)
            loss_hf = F.mse_loss(s1_next4_pred, s_pert) # Compare to unperturbed prediction
            
            # Total Loss
            loss = loss_l0 + loss_l1 + lambda_spec * loss_spec + lambda_hf * loss_hf
            loss.backward()
            optimizer.step()
            
            total_l0 += loss_l0.item()
            total_l1 += loss_l1.item()
            total_spec += loss_spec.item()
            total_hf += loss_hf.item()
            
        if (epoch+1) % 5 == 0:
            print(f"Epoch {epoch+1}: L0={total_l0/n_batches:.4f}, L1={total_l1/n_batches:.4f}, Spec={total_spec/n_batches:.4f}, HF={total_hf/n_batches:.4f}")
            
    return encoder, f0, f1

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, default="MiniGrid-DoorKey-8x8-v0", help="Environment name")
    parser.add_argument("--episodes", type=int, default=200, help="Number of episodes")
    args = parser.parse_args()
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    print(f"Environment: {args.env}")
    
    # 1. Collect Data
    print("Starting data collection...")
    data_l0, data_l1 = collect_data(num_episodes=args.episodes, env_name=args.env)
    
    # 2. Train Flat
    flat_enc, flat_f0 = train_flat(data_l0, device)
    torch.save({'encoder': flat_enc.state_dict(), 'f0': flat_f0.state_dict()}, 'flat_model.pt')
    
    # 3. Train HELM
    helm_enc, helm_f0, helm_f1 = train_helm(data_l0, data_l1, device)
    torch.save({
        'encoder': helm_enc.state_dict(), 
        'f0': helm_f0.state_dict(),
        'f1': helm_f1.state_dict()
    }, 'helm_model.pt')
    
    print("\nTraining Complete!")
