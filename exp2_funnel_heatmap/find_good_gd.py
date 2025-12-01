"""
Quick fix: Test different GD hyperparams to find what actually works.
"""

import torch
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from common.config import Exp2Config
from energy_functions import quadratic_U

config = Exp2Config()
device = 'cpu'

Q = torch.diag(torch.tensor(config.Q_eigenvalues, device=device, dtype=torch.float32))
a0 = torch.tensor(config.a_optimum, device=device, dtype=torch.float32)

def U_fn(a):
    return quadratic_U(a, a0, Q)

def test_gd_hyperparams(lr, steps):
    """Test GD convergence with given hyperparams."""
    # Grid of starting points
    x = torch.linspace(-3, 3, 50, device=device)
    y = torch.linspace(-3, 3, 50, device=device)
    X, Y = torch.meshgrid(x, y, indexing='xy')
    grid = torch.stack([X, Y], dim=-1).reshape(-1, 2)
    
    # Run GD
    a = grid.clone().requires_grad_(True)
    optimizer = torch.optim.SGD([a], lr=lr)
    
    for _ in range(steps):
        optimizer.zero_grad()
        E = U_fn(a).sum()
        E.backward()
        optimizer.step()
    
    # Check convergence
    dist = torch.norm(a.detach() - a0, dim=-1)
    coverage = (dist < 0.5).float().mean().item()
    
    return coverage * 100

print("Testing different GD hyperparams on pure quadratic U(a)...\n")
print(f"{'lr':>8} | {'steps':>8} | {'Coverage':>10}")
print("-" * 35)

for lr in [0.01, 0.05, 0.1, 0.2, 0.5]:
    for steps in [50, 100, 200, 500]:
        cov = test_gd_hyperparams(lr, steps)
        print(f"{lr:>8.2f} | {steps:>8} | {cov:>9.1f}%")
        
        if cov > 80:
            print(f"  ^^^ GOOD: lr={lr}, steps={steps} achieves >80% coverage")

print("\nUse the hyperparams with >80% coverage for basin tests.")
