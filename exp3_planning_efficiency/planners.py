"""
Planners for Experiment 3.

Implements:
1. Flat gradient descent planner
2. HELM coarse-to-fine planner
3. CEM (Cross-Entropy Method) baseline
"""

import torch
import torch.nn.functional as F


def gumbel_softmax_discrete(logits, tau=1.0, hard=False):
    """
    Gumbel-Softmax for differentiable discrete actions.
    
    Args:
        logits: (batch, T, n_actions) action logits
        tau: temperature
        hard: whether to use straight-through estimator
        
    Returns:
        (batch, T, n_actions) soft or hard one-hot actions
    """
    return F.gumbel_softmax(logits, tau=tau, hard=hard, dim=-1)


def gradient_planner_flat(world_model, obs, goal_latent, T, n_actions, 
                          n_steps=50, lr=0.1, tau=1.0, device='cuda'):
    """
    Flat gradient descent planner.
    
    Optimizes action sequence to minimize ||s_T - g||^2 via GD.
    
    Args:
        world_model: WorldModel instance
        obs: (1, C, H, W) current observation
        goal_latent: (1, d_latent) goal state in latent space
        T: planning horizon
        n_actions: number of discrete actions
        n_steps: number of GD steps
        lr: learning rate
        tau: Gumbel-Softmax temperature
        device: torch device
        
    Returns:
        best_action: int, first action to execute
    """
    # Initialize action logits
    action_logits = torch.zeros(1, T, n_actions, device=device, requires_grad=True)
    
    optimizer = torch.optim.SGD([action_logits], lr=lr)
    
    for step in range(n_steps):
        optimizer.zero_grad()
        
        # Gumbel-Softmax for differentiable sampling
        actions_soft = gumbel_softmax_discrete(action_logits, tau=tau, hard=False)
        
        # Rollout world model
        s_final = world_model.rollout(obs, actions_soft)
        
        # Planning loss: distance to goal
        loss = 0.5 * (s_final - goal_latent).pow(2).sum()
        
        loss.backward()
        optimizer.step()
    
    # Extract best action (argmax of first time step)
    with torch.no_grad():
        best_action = action_logits[0, 0, :].argmax().item()
    
    return best_action


def gradient_planner_helm(world_model, obs, goal_latent, T, n_actions,
                          coarse_stride=4, n_steps_coarse=20, n_steps_fine=30,
                          lr_coarse=0.1, lr_fine=0.05, tau=1.0, device='cuda'):
    """
    HELM coarse-to-fine gradient descent planner.
    
    Stage 1: Optimize coarse action sequence (every N steps)
    Stage 2: Refine to full resolution from coarse initialization
    
    Args:
        world_model: WorldModel instance
        obs: (1, C, H, W) current observation
        goal_latent: (1, d_latent) goal state
        T: planning horizon
        n_actions: number of discrete actions
        coarse_stride: N (coarsening factor)
        n_steps_coarse: GD steps for coarse planning
        n_steps_fine: GD steps for fine refinement
        lr_coarse, lr_fine: learning rates
        tau: Gumbel-Softmax temperature
        device: torch device
        
    Returns:
        best_action: int
    """
    T_coarse = T // coarse_stride
    
    # ===== STAGE 1: COARSE PLANNING =====
    action_logits_coarse = torch.zeros(1, T_coarse, n_actions, device=device, requires_grad=True)
    
    optimizer_coarse = torch.optim.SGD([action_logits_coarse], lr=lr_coarse)
    
    for step in range(n_steps_coarse):
        optimizer_coarse.zero_grad()
        
        # Soft actions
        actions_coarse_soft = gumbel_softmax_discrete(action_logits_coarse, tau=tau, hard=False)
        
        # Upsample to full resolution (repeat each action)
        actions_full = actions_coarse_soft.repeat_interleave(coarse_stride, dim=1)
        
        # Rollout
        s_final = world_model.rollout(obs, actions_full)
        
        # Loss
        loss = 0.5 * (s_final - goal_latent).pow(2).sum()
        
        loss.backward()
        optimizer_coarse.step()
    
    # ===== STAGE 2: FINE REFINEMENT =====
    # Initialize fine actions from coarse solution
    with torch.no_grad():
        action_logits_fine = action_logits_coarse.detach().repeat_interleave(coarse_stride, dim=1)
    
    action_logits_fine.requires_grad_(True)
    optimizer_fine = torch.optim.SGD([action_logits_fine], lr=lr_fine)
    
    for step in range(n_steps_fine):
        optimizer_fine.zero_grad()
        
        actions_fine_soft = gumbel_softmax_discrete(action_logits_fine, tau=tau, hard=False)
        
        s_final = world_model.rollout(obs, actions_fine_soft)
        
        loss = 0.5 * (s_final - goal_latent).pow(2).sum()
        
        loss.backward()
        optimizer_fine.step()
    
    # Extract best action
    with torch.no_grad():
        best_action = action_logits_fine[0, 0, :].argmax().item()
    
    return best_action


def cem_planner(world_model, obs, goal_latent, T, n_actions,
                num_samples=256, num_iters=5, elite_frac=0.1, device='cuda'):
    """
    Cross-Entropy Method (CEM) planner.
    
    Samples action sequences, selects elites, refits distribution, iterates.
    
    Args:
        world_model: WorldModel instance
        obs: (1, C, H, W) current observation
        goal_latent: (1, d_latent) goal state
        T: planning horizon
        n_actions: number of discrete actions
        num_samples: number of action sequences to sample
        num_iters: number of CEM iterations
        elite_frac: fraction of samples to keep as elites
        device: torch device
        
    Returns:
        best_action: int
    """
    elite_k = int(num_samples * elite_frac)
    
    # Initialize distribution (uniform over actions)
    action_probs = torch.ones(T, n_actions, device=device) / n_actions
    
    for iteration in range(num_iters):
        # Sample action sequences
        samples = torch.multinomial(
            action_probs.repeat(num_samples, 1, 1).reshape(-1, n_actions),
            num_samples=1
        ).reshape(num_samples, T)  # (num_samples, T)
        
        # Convert to one-hot
        samples_onehot = F.one_hot(samples.long(), num_classes=n_actions).float()
        
        # Evaluate each sample
        with torch.no_grad():
            obs_batch = obs.repeat(num_samples, 1, 1, 1)
            s_finals = world_model.rollout(obs_batch, samples_onehot)
            
            # Energy: distance to goal
            energies = 0.5 * (s_finals - goal_latent).pow(2).sum(dim=-1)  # (num_samples,)
        
        # Select elites (lowest energy)
        _, elite_idx = torch.topk(-energies, elite_k)  # max of -energy = min of energy
        elite_samples = samples[elite_idx]  # (elite_k, T)
        
        # Refit distribution (count elites per action per timestep)
        action_probs = torch.zeros(T, n_actions, device=device)
        for t in range(T):
            for a in range(n_actions):
                action_probs[t, a] = (elite_samples[:, t] == a).float().sum()
        
        action_probs = action_probs + 1e-3  # smoothing
        action_probs = action_probs / action_probs.sum(dim=-1, keepdim=True)
    
    # Final action: mode of first timestep distribution
    best_action = action_probs[0, :].argmax().item()
    
    return best_action


if __name__ == '__main__':
    from world_models import CNNEncoder, TransitionModel, WorldModel
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Create mock world model
    encoder = CNNEncoder().to(device)
    transition = TransitionModel().to(device)
    world_model = WorldModel(encoder, transition)
    
    # Mock observation and goal
    obs = torch.randn(1, 3, 7, 7, device=device)
    goal_latent = torch.randn(1, 64, device=device)
    
    T = 20
    n_actions = 7
    
    print("Testing planners...")
    
    # Flat planner
    print("\n1. Flat Gradient Planner:")
    action_flat = gradient_planner_flat(
        world_model, obs, goal_latent, T, n_actions, n_steps=10, device=device
    )
    print(f"   Planned action: {action_flat}")
    
    # HELM planner
    print("\n2. HELM Coarse-to-Fine Planner:")
    action_helm = gradient_planner_helm(
        world_model, obs, goal_latent, T, n_actions, 
        n_steps_coarse=5, n_steps_fine=10, device=device
    )
    print(f"   Planned action: {action_helm}")
    
    # CEM planner
    print("\n3. CEM Planner:")
    action_cem = cem_planner(
        world_model, obs, goal_latent, T, n_actions, 
        num_samples=128, num_iters=3, device=device
    )
    print(f"   Planned action: {action_cem}")
    
    print("\nAll planners working ✓")
