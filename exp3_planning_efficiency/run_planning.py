"""
Main evaluation script for Experiment 3.

Evaluates planning methods on MiniGrid and generates results.
"""

import torch
import numpy as np
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from common.config import Exp3Config
from common.viz_utils import plot_planning_success_rate, plot_ablation_study
from env_minigrid import make_minigrid_env, preprocess_obs, collect_random_episodes
from world_models import CNNEncoder, TransitionModel, WorldModel, train_world_model
from planners import gradient_planner_flat, gradient_planner_helm, cem_planner


def evaluate_planner(planner_fn, world_model, env, goal_latent, T, 
                     num_episodes=50, max_steps=100, device='cuda'):
    """
    Evaluate a planning method.
    
    Args:
        planner_fn: function (world_model, obs, goal, T, n_actions) -> action
        world_model: trained WorldModel
        env: MiniGrid environment
        goal_latent: goal state embedding
        T: planning horizon
        num_episodes: number of test episodes
        max_steps: max steps per episode
        device: torch device
        
    Returns:
        success_rate: fraction of successful episodes
        mean_steps: average steps to goal (for successful episodes)
    """
    successes = 0
    steps_to_goal = []
    
    for ep in range(num_episodes):
        obs = env.reset()
        done = False
        
        for step in range(max_steps):
            # Preprocess observation
            obs_tensor = preprocess_obs(obs).unsqueeze(0).to(device)  # (1, C, H, W)
            
            # Plan action
            action = planner_fn(
                world_model, 
                obs_tensor, 
                goal_latent, 
                T, 
                n_actions=env.action_space.n,
                device=device
            )
            
            # Execute in environment
            obs, reward, done, info = env.step(action)
            
            if done:
                if reward > 0:  # Success (reached goal)
                    successes += 1
                    steps_to_goal.append(step + 1)
                break
    
    success_rate = successes / num_episodes
    mean_steps = np.mean(steps_to_goal) if steps_to_goal else max_steps
    
    return success_rate, mean_steps


def run_experiment_3(config=None, device='cuda'):
    """
    Run complete Experiment 3 pipeline.
    
    Phases:
    1. Collect training data
    2. Train world model
    3. Evaluate planners (Flat, HELM, CEM)
    4. Generate plots
    """
    if config is None:
        config = Exp3Config()
    
    print("="*70)
    print(" EXPERIMENT 3: PLANNING EFFICIENCY (MiniGrid) ".center(70, "="))
    print("="*70)
    print(f"\nDevice: {device}\n")
    
    os.makedirs('checkpoints', exist_ok=True)
    os.makedirs('plots', exist_ok=True)
    
    # ===== PHASE 1: DATA COLLECTION =====
    print("\n" + "="*70)
    print(" PHASE 1: DATA COLLECTION ".center(70, "="))
    print("="*70 + "\n")
    
    env = make_minigrid_env(config.env_name)
    
    episodes = collect_random_episodes(
        env, 
        num_episodes=config.num_training_episodes,
        max_steps=config.max_steps_per_episode
    )
    
    # ===== PHASE 2: TRAIN WORLD MODEL =====
    print("\n" + "="*70)
    print(" PHASE 2: WORLD MODEL TRAINING ".center(70, "="))
    print("="*70 + "\n")
    
    encoder = CNNEncoder(d_latent=config.d_latent)
    transition = TransitionModel(
        d_latent=config.d_latent,
        n_actions=env.action_space.n,
        hidden_dim=config.hidden_dim
    )
    
    encoder, transition = train_world_model(
        encoder, transition, episodes, config, device
    )
    
    world_model = WorldModel(encoder, transition)
    world_model.eval()
    
    # Save world model
    torch.save({
        'encoder': encoder.state_dict(),
        'transition': transition.state_dict()
    }, 'checkpoints/world_model.pt')
    print("Saved world model to checkpoints/world_model.pt")
    
    # ===== COMPUTE GOAL EMBEDDING =====
    # Simplified: use learned embedding (team should improve this)
    goal_latent = torch.randn(1, config.d_latent, device=device)
    print(f"\nUsing random goal embedding (shape: {goal_latent.shape})")
    print("NOTE: Team should replace with actual goal observation encoding")
    
    # ===== PHASE 3: EVALUATE PLANNERS =====
    print("\n" + "="*70)
    print(" PHASE 3: PLANNER EVALUATION ".center(70, "="))
    print("="*70 + "\n")
    
    results = {
        'Flat GD': [],
        'HELM': [],
        'CEM': []
    }
    
    for T in config.planning_horizons:
        print(f"\n--- Horizon T={T} ---\n")
        
        # Flat gradient descent
        print("Evaluating Flat GD planner...")
        sr_flat, steps_flat = evaluate_planner(
            lambda wm, obs, goal, T, n_a, device: gradient_planner_flat(
                wm, obs, goal, T, n_a, n_steps=config.gd_steps, 
                lr=config.gd_lr, tau=config.gumbel_tau, device=device
            ),
            world_model, env, goal_latent, T,
            num_episodes=config.num_eval_episodes, device=device
        )
        results['Flat GD'].append((T, sr_flat * 100))
        print(f"Flat GD: Success={sr_flat*100:.1f}%, Steps={steps_flat:.1f}")
        
        # HELM coarse-to-fine
        print("Evaluating HELM planner...")
        sr_helm, steps_helm = evaluate_planner(
            lambda wm, obs, goal, T, n_a, device: gradient_planner_helm(
                wm, obs, goal, T, n_a,
                coarse_stride=config.coarse_stride,
                n_steps_coarse=config.gd_steps_coarse,
                n_steps_fine=config.gd_steps_fine,
                lr_coarse=config.gd_lr_coarse,
                lr_fine=config.gd_lr_fine,
                tau=config.gumbel_tau,
                device=device
            ),
            world_model, env, goal_latent, T,
            num_episodes=config.num_eval_episodes, device=device
        )
        results['HELM'].append((T, sr_helm * 100))
        print(f"HELM: Success={sr_helm*100:.1f}%, Steps={steps_helm:.1f}")
        
        # CEM
        print("Evaluating CEM planner...")
        sr_cem, steps_cem = evaluate_planner(
            lambda wm, obs, goal, T, n_a, device: cem_planner(
                wm, obs, goal, T, n_a,
                num_samples=config.cem_num_samples,
                num_iters=config.cem_num_iters,
                elite_frac=config.cem_elite_frac,
                device=device
            ),
            world_model, env, goal_latent, T,
            num_episodes=config.num_eval_episodes, device=device
        )
        results['CEM'].append((T, sr_cem * 100))
        print(f"CEM: Success={sr_cem*100:.1f}%, Steps={steps_cem:.1f}")
    
    env.close()
    
    # ===== PHASE 4: VISUALIZATION =====
    print("\n" + "="*70)
    print(" PHASE 4: GENERATING PLOTS ".center(70, "="))
    print("="*70 + "\n")
    
    plot_planning_success_rate(results, save_path='plots/exp3_success_rate.png')
    
    # ===== SUMMARY =====
    print("\n" + "="*70)
    print(" SUMMARY ".center(70, "="))
    print("="*70)
    
    print(f"\n{'T':>5} | {'Flat GD (%)':>12} | {'HELM (%)':>12} | {'CEM (%)':>12}")
    print("-" * 60)
    for i, T in enumerate(config.planning_horizons):
        flat_sr = results['Flat GD'][i][1]
        helm_sr = results['HELM'][i][1]
        cem_sr = results['CEM'][i][1]
        print(f"{T:>5} | {flat_sr:>10.1f} | {helm_sr:>10.1f} | {cem_sr:>10.1f}")
    
    print("\n" + "="*70)
    print(" EXPERIMENT 3 COMPLETE! ".center(70, "="))
    print("="*70 + "\n")
    
    return results


if __name__ == '__main__':
    config = Exp3Config()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    results = run_experiment_3(config, device)
