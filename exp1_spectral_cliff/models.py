"""
World models for Experiment 1: Flat JEPA vs HELM.

Both models predict s_T given s_0 and action sequence.
HELM version enforces spectral boundedness via regularization.
"""

import torch
import torch.nn as nn


class FlatWorldModel(nn.Module):
    """
    Flat JEPA-style predictor: GRU mapping (s_0, actions) -> s_T.
    
    No spectral regularization => Jacobian norm explodes with horizon.
    """
    
    def __init__(self, d_s=3, d_a=1, hidden_dim=64):
        super().__init__()
        self.d_s = d_s
        self.d_a = d_a
        self.hidden_dim = hidden_dim
        
        # GRU processes action sequence
        self.rnn = nn.GRU(
            input_size=d_a, 
            hidden_size=hidden_dim, 
            batch_first=True
        )
        
        # Combine s_0 with GRU output to predict s_T
        self.fc_combine = nn.Sequential(
            nn.Linear(d_s + hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh()
        )
        
        self.fc_out = nn.Linear(hidden_dim, d_s)
    
    def forward(self, s0, actions):
        """
        Args:
            s0: (batch, d_s) initial state
            actions: (batch, T, d_a) action sequence
            
        Returns:
            s_T: (batch, d_s) predicted final state
        """
        batch_size = s0.shape[0]
        T = actions.shape[1]
        
        # Process action sequence with GRU
        h0 = torch.zeros(1, batch_size, self.hidden_dim, device=actions.device)
        rnn_out, _ = self.rnn(actions, h0)  # (batch, T, hidden)
        
        # Use final RNN state
        h_T = rnn_out[:, -1, :]  # (batch, hidden)
        
        # Combine with initial state
        combined = torch.cat([s0, h_T], dim=-1)  # (batch, d_s + hidden)
        features = self.fc_combine(combined)  # (batch, hidden)
        
        # Predict final state
        s_T = self.fc_out(features)  # (batch, d_s)
        
        return s_T


class HELMWorldModel(nn.Module):
    """
    HELM world model with same architecture as Flat, but trained with
    spectral regularization to bound ||∂s_T/∂a||.
    
    The regularization is applied during training, not in the forward pass.
    """
    
    def __init__(self, d_s=3, d_a=1, hidden_dim=64):
        super().__init__()
        self.d_s = d_s
        self.d_a = d_a
        self.hidden_dim = hidden_dim
        
        # Same architecture as Flat
        self.rnn = nn.GRU(
            input_size=d_a, 
            hidden_size=hidden_dim, 
            batch_first=True
        )
        
        self.fc_combine = nn.Sequential(
            nn.Linear(d_s + hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh()
        )
        
        self.fc_out = nn.Linear(hidden_dim, d_s)
    
    def forward(self, s0, actions):
        """Same forward pass as Flat model."""
        batch_size = s0.shape[0]
        T = actions.shape[1]
        
        h0 = torch.zeros(1, batch_size, self.hidden_dim, device=actions.device)
        rnn_out, _ = self.rnn(actions, h0)
        h_T = rnn_out[:, -1, :]
        
        combined = torch.cat([s0, h_T], dim=-1)
        features = self.fc_combine(combined)
        s_T = self.fc_out(features)
        
        return s_T


def compute_spectral_regularization(model, s0, actions):
    """
    Compute spectral regularization term for HELM training.
    
    Penalizes ||∂s_T/∂s_0||_F^2 + ||∂s_T/∂a||_F^2
    to enforce bounded Jacobian.
    
    Args:
        model: HELMWorldModel instance
        s0: (batch, d_s) initial states (requires_grad=True)
        actions: (batch, T, d_a) actions (requires_grad=True)
        
    Returns:
        scalar tensor: spectral penalty
    """
    pred_sT = model(s0, actions)
    
    # Gradient wrt s0
    grad_s0 = torch.autograd.grad(
        pred_sT.sum(), 
        s0,
        retain_graph=True,
        create_graph=True,
        allow_unused=False
    )[0]  # (batch, d_s)
    
    # Gradient wrt actions
    grad_a = torch.autograd.grad(
        pred_sT.sum(), 
        actions,
        retain_graph=True,
        create_graph=True,
        allow_unused=False
    )[0]  # (batch, T, d_a)
    
    # Frobenius norm squared (sum of squared gradients)
    spec_penalty = (
        grad_s0.pow(2).mean() +  # average over batch
        grad_a.pow(2).mean()
    )
    
    return spec_penalty


if __name__ == '__main__':
    # Test models
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    batch_size = 4
    T = 20
    d_s = 3
    d_a = 1
    
    s0 = torch.randn(batch_size, d_s, device=device)
    actions = torch.randn(batch_size, T, d_a, device=device)
    
    # Flat model
    flat_model = FlatWorldModel(d_s=d_s, d_a=d_a).to(device)
    sT_flat = flat_model(s0, actions)
    print(f"Flat model output shape: {sT_flat.shape}")
    
    # HELM model
    helm_model = HELMWorldModel(d_s=d_s, d_a=d_a).to(device)
    sT_helm = helm_model(s0, actions)
    print(f"HELM model output shape: {sT_helm.shape}")
    
    # Test spectral regularization
    s0.requires_grad_(True)
    actions.requires_grad_(True)
    spec_reg = compute_spectral_regularization(helm_model, s0, actions)
    print(f"Spectral regularization: {spec_reg.item():.4f}")
    
    print("\nModels initialized successfully ✓")
