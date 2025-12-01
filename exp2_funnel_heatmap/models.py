"""
Energy MLP model for Experiment 2.

Learns to approximate E(a): R^2 -> R.
"""

import torch
import torch.nn as nn


class EnergyMLP(nn.Module):
    """
    MLP that maps 2D action to scalar energy.
    
    Used for both Flat (overfits ruggedness) and HELM (smooths it).
    """
    
    def __init__(self, input_dim=2, hidden_dim=128, num_layers=3):
        super().__init__()
        
        layers = []
        
        # Input layer
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.Tanh())
        
        # Hidden layers
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.Tanh())
        
        # Output layer (scalar energy)
        layers.append(nn.Linear(hidden_dim, 1))
        
        self.net = nn.Sequential(*layers)
    
    def forward(self, a):
        """
        Args:
            a: (..., 2) batch of 2D actions
            
        Returns:
            (...,) energy values (scalar per input)
        """
        output = self.net(a)  # (..., 1)
        return output.squeeze(-1)  # (...,)


if __name__ == '__main__':
    # Test model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    model = EnergyMLP(hidden_dim=128).to(device)
    
    # Test with batch
    batch_size = 16
    a = torch.randn(batch_size, 2, device=device)
    E = model(a)
    
    print(f"Input shape: {a.shape}")
    print(f"Output shape: {E.shape}")
    print(f"Output range: [{E.min().item():.3f}, {E.max().item():.3f}]")
    
    # Test with 2D grid (for contour plotting)
    X = torch.linspace(-3, 3, 50, device=device)
    Y = torch.linspace(-3, 3, 50, device=device)
    XX, YY = torch.meshgrid(X, Y, indexing='xy')
    grid = torch.stack([XX, YY], dim=-1)  # (50, 50, 2)
    
    E_grid = model(grid)  # (50, 50)
    print(f"\nGrid output shape: {E_grid.shape}")
    
    print("\nModel test passed ✓")
