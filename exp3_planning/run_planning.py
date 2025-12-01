"""
Main execution script for Experiment 3.
"""
import torch
import numpy as np
import gymnasium as gym
from minigrid.wrappers import FullyObsWrapper, ImgObsWrapper # Use FULL obs
from models import Encoder, TransitionL0, TransitionL1
from planner import FlatPlanner, HELMPlanner, CEMPlanner
import os

def make_env_full(env_name="MiniGrid-DoorKey-8x8-v0"):
    # Use Fully Observable wrapper as implied by "fully observable RGB"
    env = gym.make(env_name)
    env = FullyObsWrapper(env) 
    env = ImgObsWrapper(env)
    return env

def get_goal_latent(encoder, env, device):
    """
    Hack to get the goal latent state.
    We place the agent at the goal position and encode the observation.
    """
    env.reset()
    
    # Find goal pos
    goal_pos = None
    for i in range(env.unwrapped.grid.width):
        for j in range(env.unwrapped.grid.height):
            cell = env.unwrapped.grid.get(i, j)
            if cell and cell.type == 'goal':
                goal_pos = (i, j)
                break
    
    if goal_pos is None:
        # Fallback (e.g. Empty-8x8)
        goal_pos = (env.unwrapped.grid.width-2, env.unwrapped.grid.height-2)
        
    # Place agent at goal
    env.unwrapped.agent_pos = goal_pos
    env.unwrapped.agent_dir = 0 # Facing right
    
    # Generate Symbolic observation manually
    obs = env.unwrapped.grid.encode()
    
    # Preprocess
    obs_tensor = torch.tensor(obs, dtype=torch.float32).permute(2, 0, 1) / 255.0
    obs_tensor = obs_tensor.unsqueeze(0).to(device)
    
    with torch.no_grad():
        g = encoder(obs_tensor)
        
    return g

def evaluate_planner(name, planner, env_maker, goal_latent, num_episodes=20, device='cpu', max_steps=30, env_name="MiniGrid-DoorKey-8x8-v0"):
    print(f"\nEvaluating {name}...")
    successes = 0
    total_steps = 0
    
    for ep in range(num_episodes):
        env = env_maker(env_name)
        obs, _ = env.reset()
        done = False
        steps = 0
        
        while not done and steps < max_steps:
            # Preprocess obs
            obs_tensor = torch.tensor(obs, dtype=torch.float32).permute(2, 0, 1) / 255.0
            obs_tensor = obs_tensor.unsqueeze(0).to(device)
            
            # Plan
            action = planner.plan(obs_tensor, goal_latent)
            
            # Execute
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            steps += 1
            
            if done and reward > 0:
                successes += 1
                break
        
        print(f"  Ep {ep+1}/{num_episodes}: {'Success' if done and reward > 0 else 'Fail'} (Steps: {steps})", flush=True)
        total_steps += steps
        
    success_rate = successes / num_episodes * 100
    avg_steps = total_steps / num_episodes
    return success_rate, avg_steps

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, default="MiniGrid-DoorKey-8x8-v0", help="Environment name")
    parser.add_argument("--episodes", type=int, default=20, help="Number of eval episodes")
    parser.add_argument("--horizon", type=int, default=30, help="Max steps per episode")
    args = parser.parse_args()
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    print(f"Environment: {args.env}")
    print(f"Horizon: {args.horizon}")
    
    # Load Models
    print("Loading models...", flush=True)
    
    # Flat
    flat_enc = Encoder().to(device)
    flat_f0 = TransitionL0().to(device)
    flat_ckpt = torch.load('flat_model.pt', map_location=device)
    flat_enc.load_state_dict(flat_ckpt['encoder'])
    flat_f0.load_state_dict(flat_ckpt['f0'])
    flat_enc.eval()
    flat_f0.eval()
    
    # HELM
    helm_enc = Encoder().to(device)
    helm_f0 = TransitionL0().to(device)
    helm_f1 = TransitionL1().to(device)
    helm_ckpt = torch.load('helm_model.pt', map_location=device)
    helm_enc.load_state_dict(helm_ckpt['encoder'])
    helm_f0.load_state_dict(helm_ckpt['f0'])
    helm_f1.load_state_dict(helm_ckpt['f1'])
    helm_enc.eval()
    helm_f0.eval()
    helm_f1.eval()
    
    # Setup Env and Goal
    print("Getting goal latent...", flush=True)
    env = make_env_full(args.env)
    
    goal_flat = get_goal_latent(flat_enc, make_env_full(args.env), device)
    goal_helm = get_goal_latent(helm_enc, make_env_full(args.env), device)
    print("Goal latents obtained.", flush=True)
    
    # Initialize Planners
    flat_planner = FlatPlanner(flat_enc, flat_f0, device=device)
    helm_planner = HELMPlanner(helm_enc, helm_f0, helm_f1, device=device)
    cem_planner = CEMPlanner(flat_enc, flat_f0, device=device) # CEM uses Flat model
    
    # Evaluate
    results = {}
    
    # 1. Flat GD
    sr, steps = evaluate_planner("Flat-GD", flat_planner, make_env_full, goal_flat, num_episodes=args.episodes, device=device, max_steps=args.horizon, env_name=args.env)
    results['Flat-GD'] = (sr, steps)
    
    # 2. HELM GD
    sr, steps = evaluate_planner("HELM-GD", helm_planner, make_env_full, goal_helm, num_episodes=args.episodes, device=device, max_steps=args.horizon, env_name=args.env)
    results['HELM-GD'] = (sr, steps)
    
    # 3. CEM (Baseline)
    sr, steps = evaluate_planner("CEM", cem_planner, make_env_full, goal_flat, num_episodes=args.episodes, device=device, max_steps=args.horizon, env_name=args.env)
    results['CEM'] = (sr, steps)
    
    # Print Summary
    print("\n" + "="*50)
    print(" EXPERIMENT 3 RESULTS ".center(50, "="))
    print("="*50)
    print(f"{'Planner':<15} | {'Success %':<10} | {'Avg Steps':<10}")
    print("-" * 40)
    for name, (sr, steps) in results.items():
        print(f"{name:<15} | {sr:<10.1f} | {steps:<10.1f}")
