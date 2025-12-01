"""
MiniGrid environment wrapper for Experiment 3.

Provides clean interface for world model training and planning.
"""

import gym
import numpy as np
import torch


def make_minigrid_env(env_name='MiniGrid-KeyCorridorS3R3-v0', seed=0):
    """
    Create MiniGrid environment.
    
    Args:
        env_name: MiniGrid environment name
        seed: random seed
        
    Returns:
        gym environment
    """
    try:
        # Try new minigrid API
        import minigrid
        env = gym.make(env_name)
    except:
        # Fallback to old gym_minigrid
        import gym_minigrid
        env = gym.make(env_name)
    
    env.seed(seed)
    return env


def preprocess_obs(obs):
    """
    Preprocess MiniGrid observation for neural network.
    
    Args:
        obs: MiniGrid observation dict
        
    Returns:
        tensor: (C, H, W) preprocessed image
    """
    # MiniGrid obs is dict with 'image' key
    if isinstance(obs, dict):
        img = obs['image']
    else:
        img = obs
    
    # Convert to tensor and normalize
    # MiniGrid uses (H, W, C) format with values 0-255
    img_tensor = torch.from_numpy(img).float() / 255.0
    
    # Rearrange to (C, H, W)
    img_tensor = img_tensor.permute(2, 0, 1)
    
    return img_tensor


def collect_random_episodes(env, num_episodes=100, max_steps=100):
    """
    Collect episodes using random policy for world model training.
    
    Args:
        env: MiniGrid environment
        num_episodes: number of episodes to collect
        max_steps: max steps per episode
        
    Returns:
        episodes: list of (observations, actions, rewards, dones)
    """
    episodes = []
    
    print(f"Collecting {num_episodes} random episodes...")
    
    for ep in range(num_episodes):
        obs = env.reset()
        
        observations = [preprocess_obs(obs)]
        actions = []
        rewards = []
        dones = []
        
        for step in range(max_steps):
            action = env.action_space.sample()
            obs, reward, done, info = env.step(action)
            
            observations.append(preprocess_obs(obs))
            actions.append(action)
            rewards.append(reward)
            dones.append(done)
            
            if done:
                break
        
        episodes.append({
            'observations': torch.stack(observations),  # (T+1, C, H, W)
            'actions': torch.tensor(actions),  # (T,)
            'rewards': torch.tensor(rewards),  # (T,)
            'dones': torch.tensor(dones)  # (T,)
        })
        
        if (ep + 1) % 20 == 0:
            print(f"  Collected {ep+1}/{num_episodes} episodes")
    
    print("Data collection complete!")
    return episodes


def get_goal_observation(env):
    """
    Get goal observation for encoding goal state.
    
    For KeyCorridor, this is the observation when agent reaches goal.
    In practice, we can either:
    1. Play episode to goal and record obs
    2. Use learned goal embedding (trainable parameter)
    
    Args:
        env: MiniGrid environment
        
    Returns:
        goal_obs: (C, H, W) goal observation
    """
    # Simple approach: reset and step towards goal
    # For real implementation, you'd want to manually complete the task
    # or use a pretrained policy
    
    # Placeholder: return current obs (team should improve this)
    obs = env.reset()
    return preprocess_obs(obs)


if __name__ == '__main__':
    # Test environment
    env = make_minigrid_env()
    
    print(f"Environment: {env}")
    print(f"Action space: {env.action_space}")
    print(f"Observation space: {env.observation_space}")
    
    # Test episode collection
    episodes = collect_random_episodes(env, num_episodes=5, max_steps=50)
    
    print(f"\nCollected {len(episodes)} episodes")
    print(f"First episode:")
    print(f"  Observations: {episodes[0]['observations'].shape}")
    print(f"  Actions: {episodes[0]['actions'].shape}")
    print(f"  Rewards: {episodes[0]['rewards'].sum():.2f}")
    
    env.close()
