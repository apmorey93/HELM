"""
Synthetic 2D energy functions for Experiment 2.

Energy = U(a) + ψ(a) where:
- U(a) is convex quadratic (smooth signal)
- ψ(a) is sinusoidal noise (high-frequency ruggedness)
"""

import torch


def quadratic_U(a, a0, Q):
    """
    Quadratic bowl: U(a) = 0.5 * (a - a0)^T Q (a - a0)
    
    Args:
        a: (..., 2) action points
        a0: (2,) optimum location
        Q: (2, 2) positive definite matrix
        
    Returns:
        (...,) energy values
    """
    diff = a - a0  # (..., 2)
    # Quadratic form: diff^T Q diff
    energy = 0.5 * torch.sum(diff @ Q * diff, dim=-1)
    return energy


def sinusoidal_noise(a, alphas, omegas):
    """
    Sinusoidal ruggedness: ψ(a) = Σ_i α_i sin(ω_i^T a)
    
    Args:
        a: (..., 2) action points
        alphas: list of amplitudes
        omegas: list of (2,) frequency vectors (as tensors)
        
    Returns:
        (...,) noise values
    """
    total = torch.zeros(a.shape[:-1], device=a.device, dtype=a.dtype)
    
    for alpha, omega in zip(alphas, omegas):
        omega = omega.to(a.device)
        phase = torch.sum(a * omega, dim=-1)  # (...,)
        total = total + alpha * torch.sin(phase)
    
    return total


def true_energy(a, config=None):
    """
    Complete energy function: E(a) = U(a) + ψ(a)
    
    Args:
        a: (..., 2) action points
        config: Exp2Config (optional)
        
    Returns:
        (...,) total energy
    """
    if config is None:
        # Default parameters
        from common.config import Exp2Config
        config = Exp2Config()
    
    # Convert config lists to tensors
    a0 = torch.tensor(config.a_optimum, device=a.device, dtype=a.dtype)
    
    # Q matrix from eigenvalues (for simplicity, diagonal)
    Q = torch.diag(torch.tensor(config.Q_eigenvalues, device=a.device, dtype=a.dtype))
    
    # Omega vectors
    omegas = [torch.tensor(freq, device=a.device, dtype=a.dtype) 
              for freq in config.sinusoid_freqs]
    alphas = config.sinusoid_amplitudes
    
    # Compute components
    U = quadratic_U(a, a0, Q)
    psi = sinusoidal_noise(a, alphas, omegas)
    
    return U + psi


def evaluate_on_grid(energy_fn, x_range=(-3, 3), y_range=(-3, 3), 
                     resolution=200, device='cpu'):
    """
    Evaluate energy function on 2D grid.
    
    Args:
        energy_fn: function taking (N, 2) -> (N,)
        x_range, y_range: domain bounds
        resolution: grid resolution
        device: torch device
        
    Returns:
        X, Y: (resolution, resolution) meshgrids
        Z: (resolution, resolution) energy values
    """
    x = torch.linspace(x_range[0], x_range[1], resolution, device=device)
    y = torch.linspace(y_range[0], y_range[1], resolution, device=device)
    
    X, Y = torch.meshgrid(x, y, indexing='xy')
    grid = torch.stack([X, Y], dim=-1)  # (resolution, resolution, 2)
    
    # Evaluate energy
    Z = energy_fn(grid)  # (resolution, resolution)
    
    return X, Y, Z


if __name__ == '__main__':
    from common.config import Exp2Config
    import matplotlib.pyplot as plt
    
    config = Exp2Config()
    device = 'cpu'
    
    # Create energy function
    def E(a):
        return true_energy(a, config)
    
    # Evaluate on grid
    X, Y, Z = evaluate_on_grid(E, resolution=200, device=device)
    
    # Plot
    plt.figure(figsize=(10, 8))
    plt.contourf(X.numpy(), Y.numpy(), Z.numpy(), levels=50, cmap='viridis')
    plt.colorbar(label='Energy')
    plt.contour(X.numpy(), Y.numpy(), Z.numpy(), levels=10, colors='white', alpha=0.3)
    plt.xlabel('a₁')
    plt.ylabel('a₂')
    plt.title('True Energy Landscape (Convex + Rugged)')
    plt.axis('equal')
    plt.tight_layout()
    plt.savefig('test_energy.png', dpi=150)
    print("Saved test energy plot to test_energy.png")
    
    # Check optimum
    print(f"\nOptimum: {config.a_optimum}")
    a_opt = torch.tensor([config.a_optimum], dtype=torch.float32)
    E_opt = E(a_opt)
    print(f"Energy at optimum: {E_opt.item():.4f}")
    
    # Check nearby points
    a_nearby = a_opt + torch.randn(10, 2) * 0.5
    E_nearby = E(a_nearby)
    print(f"Energy nearby (mean): {E_nearby.mean().item():.4f}")
    print(f"Should be higher due to ruggedness ✓" if E_nearby.mean() > E_opt else "ISSUE")
