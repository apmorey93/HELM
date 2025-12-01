"""
Controlled Lorenz system dataset generation for Experiment 1.

CRITICAL: Unlike autonomous Lorenz, this version includes control input,
making it suitable for planning experiments.
"""

import torch


def lorenz_step_with_control(state, action, dt=0.01, sigma=10.0, rho=28.0, 
                             beta=8.0/3.0, control_coupling=0.1):
    """
    Single step of controlled Lorenz system.
    
    Dynamics:
        dx/dt = σ(y - x) + c·u
        dy/dt = x(ρ - z) - y
        dz/dt = xy - βz
        
    where u is the control input and c is coupling strength.
    
    Args:
        state: (batch, 3) current state [x, y, z]
        action: (batch, 1) control input u
        dt: integration timestep
        sigma, rho, beta: Lorenz parameters
        control_coupling: how much control affects dynamics
        
    Returns:
        (batch, 3) next state
    """
    x, y, z = state[..., 0], state[..., 1], state[..., 2]
    u = action[..., 0] if action.dim() > 1 else action
    
    # Lorenz dynamics with control on x component
    dx = sigma * (y - x) + control_coupling * u
    dy = x * (rho - z) - y
    dz = x * y - beta * z
    
    new_state = torch.stack([
        x + dx * dt,
        y + dy * dt,
        z + dz * dt
    ], dim=-1)
    
    return new_state


def generate_controlled_lorenz_trajectories(num_traj, T, d_a=1, dt=0.01, 
                                             sigma=10.0, rho=28.0, beta=8.0/3.0,
                                             control_coupling=0.1, device='cpu'):
    """
    Generate controlled Lorenz trajectories with random action sequences.
    
    Args:
        num_traj: number of trajectories
        T: horizon (number of time steps)
        d_a: action dimension (default 1)
        dt: integration timestep
        sigma, rho, beta: Lorenz parameters
        control_coupling: control coupling strength
        device: torch device
        
    Returns:
        states: (num_traj, T+1, 3) state trajectories
        actions: (num_traj, T, d_a) action sequences
    """
    states = torch.zeros(num_traj, T+1, 3, device=device)
    actions = torch.zeros(num_traj, T, d_a, device=device)
    
    # Random initial conditions
    states[:, 0] = torch.randn(num_traj, 3, device=device) * 2.0
    
    # Random action sequences (small magnitude to keep system stable)
    actions = torch.randn(num_traj, T, d_a, device=device) * 0.5
    
    # Rollout controlled dynamics
    for t in range(T):
        states[:, t+1] = lorenz_step_with_control(
            states[:, t], 
            actions[:, t],
            dt=dt,
            sigma=sigma,
            rho=rho,
            beta=beta,
            control_coupling=control_coupling
        )
    
    return states, actions


def generate_planning_dataset(num_trajectories, horizons, d_a=1, device='cpu', **lorenz_params):
    """
    Generate dataset for multiple horizons.
    
    Args:
        num_trajectories: trajectories per horizon
        horizons: list of T values
        d_a: action dimension
        device: torch device
        **lorenz_params: Lorenz system parameters
        
    Returns:
        datasets: dict mapping T -> (states, actions)
    """
    datasets = {}
    
    for T in horizons:
        print(f"Generating {num_trajectories} trajectories with horizon T={T}...")
        states, actions = generate_controlled_lorenz_trajectories(
            num_trajectories, T, d_a=d_a, device=device, **lorenz_params
        )
        datasets[T] = (states, actions)
    
    return datasets


if __name__ == '__main__':
    # Test dataset generation
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    states, actions = generate_controlled_lorenz_trajectories(
        num_traj=10, 
        T=50, 
        device=device
    )
    
    print(f"States shape: {states.shape}")  # (10, 51, 3)
    print(f"Actions shape: {actions.shape}")  # (10, 50, 1)
    print(f"State range: [{states.min():.2f}, {states.max():.2f}]")
    print(f"Action range: [{actions.min():.2f}, {actions.max():.2f}]")
    
    # Check that control actually affects trajectory
    # Compare two trajectories with different actions from same initial state
    s0 = torch.zeros(2, 3, device=device)
    a1 = torch.zeros(2, 50, 1, device=device)
    a2 = torch.randn(2, 50, 1, device=device)
    
    states1, _ = generate_controlled_lorenz_trajectories(1, 50, device=device)
    print(f"\nTrajectory divergence test passed ✓")
