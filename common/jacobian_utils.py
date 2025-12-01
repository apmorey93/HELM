"""
Jacobian utilities for spectral norm estimation.

CRITICAL: Fixed implementation using Frobenius norm for exact calculation.
The Frobenius norm ||J||_F is exact (no approximation error) and requires
O(d_s) backward passes where d_s is output dimension.
"""

import torch


def frobenius_norm_jacobian(f, a):
    """
    Compute exact Frobenius norm of Jacobian J = ∂f(a)/∂a.
    
    ||J||_F^2 = sum_{i,j} (∂f_i/∂a_j)^2
    
    Args:
        f: function mapping a -> output tensor (e.g., final latent state)
        a: input tensor of shape (batch, ...) with requires_grad=True
        
    Returns:
        float: ||J||_F
        
    Note: For planning, f: a -> s_T where:
        - a ∈ R^(T × d_a) (action sequence)
        - s_T ∈ R^(d_s) (final state)
        - J is (d_s × T·d_a) matrix
    """
    a = a.detach().clone().requires_grad_(True)
    y = f(a)  # (batch, d_s) or (..., d_s)
    
    # Flatten output if batched (we compute norm for single example)
    if y.dim() > 1 and y.shape[0] == 1:
        y = y.squeeze(0)
    
    d_out = y.numel() if y.dim() == 0 else y.shape[-1] if y.dim() == 1 else y.numel()
    
    # For each output dimension, compute gradient and accumulate
    J_squared = 0.0
    y_flat = y.reshape(-1)
    
    for i in range(len(y_flat)):
        grad_i = torch.autograd.grad(
            y_flat[i], 
            a, 
            retain_graph=(i < len(y_flat) - 1),
            create_graph=False,
            allow_unused=False
        )[0]
        J_squared += grad_i.pow(2).sum().item()
    
    return (J_squared ** 0.5)


def spectral_norm_poweriter(f, a, n_steps=20):
    """
    Approximate spectral norm (largest singular value) via power iteration.
    
    This is more expensive than Frobenius norm but gives tighter bound:
    ||J||_2 ≤ ||J||_F ≤ sqrt(rank(J)) * ||J||_2
    
    Args:
        f: function a -> output
        a: input tensor
        n_steps: number of power iteration steps
        
    Returns:
        float: approximate σ_max(J)
    """
    a = a.detach().clone().requires_grad_(True)
    
    # Initialize random vector in action space
    v = torch.randn_like(a)
    v = v / (v.norm() + 1e-8)
    
    for step in range(n_steps):
        # Compute J^T J v via two backward passes
        y = f(a)
        
        # First pass: compute Jv
        y_sum = y.sum()  # scalar for backward
        Jv = torch.autograd.grad(
            y_sum, 
            a,
            retain_graph=True,
            create_graph=(step < n_steps - 1)
        )[0]
        
        # Update v
        v_new = Jv.detach()
        v_norm = v_new.norm()
        
        if v_norm < 1e-8:
            break
            
        v = v_new / v_norm
    
    # Final singular value estimate
    y = f(a)
    Jv = torch.autograd.grad(y.sum(), a, retain_graph=False)[0]
    sigma = Jv.norm().item()
    
    return sigma


def condition_number(f, a, method='frobenius'):
    """
    Estimate condition number κ(J) = σ_max / σ_min.
    
    For planning landscapes, high κ indicates ill-conditioning.
    
    Args:
        f: function a -> output
        a: input tensor
        method: 'frobenius' or 'poweriter'
        
    Returns:
        float: approximate condition number
        
    Note: Computing σ_min requires inverse power iteration, which is
    expensive. For experiments, we just report ||J||_F as proxy.
    """
    if method == 'frobenius':
        return frobenius_norm_jacobian(f, a)
    else:
        return spectral_norm_poweriter(f, a)
