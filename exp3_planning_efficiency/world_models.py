"""
World models for Experiment 3.

Encoder: CNN observation -> latent state
Transition: latent state + action -> next latent state
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CNNEncoder(nn.Module):
    """
    CNN encoder for MiniGrid observations.
    
    Maps (C, H, W) image to d_latent dimensional latent state.
    """
    
    def __init__(self, input_channels=3, d_latent=64, channels=[32, 64, 64]):
        super().__init__()
        
        self.conv_layers = nn.ModuleList()
        
        in_ch = input_channels
        for out_ch in channels:
            self.conv_layers.append(
                nn.Sequential(
                    nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1),
                    nn.ReLU(),
                    nn.MaxPool2d(2)
                )
            )
            in_ch = out_ch
        
        # Adaptive pooling to fixed size
        self.adaptive_pool = nn.AdaptiveAvgPool2d((2, 2))
        
        # Final FC to latent
        self.fc = nn.Linear(channels[-1] * 2 * 2, d_latent)
    
    def forward(self, obs):
        """
        Args:
            obs: (batch, C, H, W) observations
            
        Returns:
            (batch, d_latent) latent states
        """
        x = obs
        
        for conv in self.conv_layers:
            x = conv(x)
        
        x = self.adaptive_pool(x)
        x = x.reshape(x.size(0), -1)  # flatten
        
        latent = self.fc(x)
        return latent


class TransitionModel(nn.Module):
    """
    Transition model: (s_t, a_t) -> s_{t+1}
    
    For discrete actions, we use one-hot encoding.
    """
    
    def __init__(self, d_latent=64, n_actions=7, hidden_dim=128):
        super().__init__()
        self.d_latent = d_latent
        self.n_actions = n_actions
        
        self.net = nn.Sequential(
            nn.Linear(d_latent + n_actions, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, d_latent)
        )
    
    def forward(self, s, a):
        """
        Args:
            s: (batch, d_latent) current latent state
            a: (batch, n_actions) one-hot actions OR soft actions
            
        Returns:
            (batch, d_latent) next latent state
        """
        x = torch.cat([s, a], dim=-1)
        s_next = self.net(x)
        return s_next


class WorldModel(nn.Module):
    """
    Combined world model: encoder + transition.
    
    Can rollout latent dynamics for planning.
    """
    
    def __init__(self, encoder, transition):
        super().__init__()
        self.encoder = encoder
        self.transition = transition
    
    def encode(self, obs):
        """Encode observation to latent."""
        return self.encoder(obs)
    
    def step(self, s, a):
        """Single latent transition."""
        return self.transition(s, a)
    
    def rollout(self, obs, actions):
        """
        Rollout latent dynamics for planning.
        
        Args:
            obs: (batch, C, H, W) initial observations
            actions: (batch, T, n_actions) action sequence (one-hot or soft)
            
        Returns:
            s_final: (batch, d_latent) final latent state
        """
        s = self.encoder(obs)  # (batch, d_latent)
        
        T = actions.shape[1]
        for t in range(T):
            a_t = actions[:, t, :]  # (batch, n_actions)
            s = self.transition(s, a_t)
        
        return s


def train_world_model(encoder, transition, episodes, config, device='cuda'):
    """
    Train world model on collected episodes.
    
    Loss: MSE between predicted and true next latent states.
    
    Args:
        encoder: CNNEncoder
        transition: TransitionModel
        episodes: list of episode dicts
        config: Exp3Config
        device: torch device
        
    Returns:
        trained encoder, transition
    """
    print("\n" + "="*60)
    print(" Training World Model ".center(60, "="))
    print("="*60 + "\n")
    
    encoder = encoder.to(device)
    transition = transition.to(device)
    
    optimizer = torch.optim.Adam(
        list(encoder.parameters()) + list(transition.parameters()),
        lr=config.learning_rate
    )
    
    loss_fn = nn.MSELoss()
    
    # Prepare training data
    all_obs = []
    all_next_obs = []
    all_actions = []
    
    for ep in episodes:
        obs_seq = ep['observations']  # (T+1, C, H, W)
        act_seq = ep['actions']  # (T,)
        
        for t in range(len(act_seq)):
            all_obs.append(obs_seq[t])
            all_next_obs.append(obs_seq[t+1])
            all_actions.append(act_seq[t])
    
    all_obs = torch.stack(all_obs).to(device)  # (N, C, H, W)
    all_next_obs = torch.stack(all_next_obs).to(device)
    all_actions = torch.tensor(all_actions, device=device)  # (N,)
    
    # One-hot encode actions
    all_actions_onehot = F.one_hot(
        all_actions.long(), num_classes=transition.n_actions
    ).float()  # (N, n_actions)
    
    print(f"Training on {len(all_obs)} transitions...")
    
    # Training loop
    batch_size = 64
    num_batches = len(all_obs) // batch_size
    
    for epoch in range(20):  # Simplified: fixed epochs
        total_loss = 0.0
        
        # Shuffle data
        perm = torch.randperm(len(all_obs))
        
        for i in range(num_batches):
            idx = perm[i*batch_size:(i+1)*batch_size]
            
            obs_batch = all_obs[idx]
            next_obs_batch = all_next_obs[idx]
            actions_batch = all_actions_onehot[idx]
            
            # Encode
            s_t = encoder(obs_batch)
            s_next_true = encoder(next_obs_batch).detach()  # stop gradient
            
            # Predict
            s_next_pred = transition(s_t, actions_batch)
            
            # Loss
            loss = loss_fn(s_next_pred, s_next_true)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / num_batches
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"Epoch {epoch+1}/20: loss={avg_loss:.6f}")
    
    print(f"\nWorld model training complete! Final loss: {avg_loss:.6f}")
    
    return encoder, transition


if __name__ == '__main__':
    # Test models
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    encoder = CNNEncoder(input_channels=3, d_latent=64).to(device)
    transition = TransitionModel(d_latent=64, n_actions=7).to(device)
    
    # Test forward pass
    obs = torch.randn(4, 3, 7, 7, device=device)  # MiniGrid default size
    latent = encoder(obs)
    print(f"Encoder output: {latent.shape}")
    
    actions = F.one_hot(torch.tensor([0, 1, 2, 3], device=device), num_classes=7).float()
    next_latent = transition(latent, actions)
    print(f"Transition output: {next_latent.shape}")
    
    # Test rollout
    world_model = WorldModel(encoder, transition)
    action_seq = F.one_hot(
        torch.randint(0, 7, (4, 10), device=device), num_classes=7
    ).float()
    final_latent = world_model.rollout(obs, action_seq)
    print(f"Rollout final latent: {final_latent.shape}")
    
    print("\nWorld model test passed ✓")
