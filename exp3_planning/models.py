"""
World Models for Experiment 3 (MiniGrid Planning).
"""
import torch
import torch.nn as nn

class Encoder(nn.Module):
    """
    Encodes 8x8x3 observations into latent state.
    """
    def __init__(self, d_latent=64):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 16, 3, stride=1, padding=1), nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1), nn.ReLU(),  # 8x8 -> 4x4
        )
        self.fc = nn.Sequential(
            nn.Linear(32 * 4 * 4, 128), nn.ReLU(),
            nn.Linear(128, d_latent),
        )

    def forward(self, obs):  
        # obs: (B, 3, H, W) float in [0,1]
        x = self.conv(obs)
        x = x.reshape(x.size(0), -1)
        return self.fc(x)


class TransitionL0(nn.Module):
    """
    Level-0 Transition: s_t, a_t -> s_{t+1}
    Standard flat dynamics.
    """
    def __init__(self, d_latent=64, n_actions=7):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_latent + n_actions, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU(),
            nn.Linear(128, d_latent),
        )

    def forward(self, s, a):  
        # s: (B, d_latent), a: (B, n_actions)
        x = torch.cat([s, a], dim=-1)
        return s + self.net(x)  # residual


class TransitionL1(nn.Module):
    """
    Level-1 Transition: s_t, u_t -> s_{t+4}
    Coarse dynamics with stride 4.
    u_t is the average action distribution over 4 steps.
    """
    def __init__(self, d_latent=64, n_actions=7):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_latent + n_actions, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU(),
            nn.Linear(128, d_latent),
        )

    def forward(self, s, u):
        # s: (B, d_latent), u: (B, n_actions)
        x = torch.cat([s, u], dim=-1)
        return s + self.net(x)
