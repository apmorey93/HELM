import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

# Mock MiniGrid
class MockEnv:
    """Simple 2D GridNav proxy for MiniGrid"""
    def __init__(self):
        self.action_space = type('obj', (object,), {'n': 4})
        # Discrete grid state
        self.size = 8
        self.pos = np.array([0, 0])
        self.goal = np.array([7, 7])
    
    def reset(self):
        self.pos = np.array([0, 0])
        return self._get_obs()
    
    def _get_obs(self):
        # Return one-hot encoding of position as observation
        obs = np.zeros((self.size, self.size))
        obs[self.pos[0], self.pos[1]] = 1
        return obs.flatten()
    
    def step(self, action):
        # 0: up, 1: down, 2: left, 3: right
        # Note: MiniGrid coords are (col, row) usually, but let's stick to simple matrix indexing
        # Let's say 0: x+1 (down), 1: x-1 (up), 2: y-1 (left), 3: y+1 (right)
        # Wait, user code:
        # 0: pos[1] += 1 (Right/Down depending on axis)
        # Let's assume standard matrix indexing (row, col)
        # 0: pos[1] += 1 -> Right
        # 1: pos[1] -= 1 -> Left
        # 2: pos[0] -= 1 -> Up
        # 3: pos[0] += 1 -> Down
        
        if action == 0: self.pos[1] = min(self.pos[1] + 1, self.size-1)
        elif action == 1: self.pos[1] = max(self.pos[1] - 1, 0)
        elif action == 2: self.pos[0] = max(self.pos[0] - 1, 0)
        elif action == 3: self.pos[0] = min(self.pos[0] + 1, self.size-1)
        
        dist = np.linalg.norm(self.pos - self.goal)
        reward = -dist # Simple distance reward
        done = np.array_equal(self.pos, self.goal)
        return self._get_obs(), reward, done, {}

# --- 1. Contrastive World Model ---
class ContrastiveWorldModel(nn.Module):
    def __init__(self, input_dim, action_dim, hidden_dim=64, latent_dim=16):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim) # Latent state s
        )
        # Transition: s_t, a_t -> s_t+1
        self.transition = nn.Sequential(
            nn.Linear(latent_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim)
        )
        
    def forward(self, obs, a_onehot):
        s = self.encoder(obs)
        x = torch.cat([s, a_onehot], dim=-1)
        return s + self.transition(x)

# --- 2. Training with Contrastive Shaping ---
def train_contrastive_world_model(model, env, episodes=500):
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    print(f"Training on {episodes} episodes...")
    
    for ep in range(episodes):
        obs = env.reset()
        done = False
        trajectory = []
        
        # Collect trajectory
        while not done and len(trajectory) < 50:
            action = np.random.randint(0, 4)
            next_obs, r, done, _ = env.step(action)
            
            a_onehot = np.zeros(4)
            a_onehot[action] = 1
            
            trajectory.append((obs, a_onehot, next_obs))
            obs = next_obs
            
        # Contrastive Update
        if len(trajectory) < 2: continue
            
        batch = trajectory
        obs_batch = torch.tensor(np.array([x[0] for x in batch]), dtype=torch.float32)
        act_batch = torch.tensor(np.array([x[1] for x in batch]), dtype=torch.float32)
        next_batch = torch.tensor(np.array([x[2] for x in batch]), dtype=torch.float32)
        
        # 1. Transition Loss (Predict next state)
        s_t = model.encoder(obs_batch)
        s_next_pred = model.transition(torch.cat([s_t, act_batch], dim=-1)) + s_t
        s_next_true = model.encoder(next_batch)
        trans_loss = F.mse_loss(s_next_pred, s_next_true)
        
        # 2. Topology Loss (Contrastive)
        # InfoNCE: maximize similarity of (s_t, s_t+1), minimize others
        # We use s_t and s_next_true as positive pairs
        
        # Normalize embeddings for cosine similarity
        s_t_norm = F.normalize(s_t, dim=-1)
        s_next_norm = F.normalize(s_next_true, dim=-1)
        
        # Similarity matrix (B, B)
        sim_matrix = torch.matmul(s_t_norm, s_next_norm.T) 
        
        # Temperature
        tau = 0.1
        sim_matrix = sim_matrix / tau
        
        labels = torch.arange(len(batch)).to(s_t.device)
        contrastive_loss = F.cross_entropy(sim_matrix, labels)
        
        loss = trans_loss + 1.0 * contrastive_loss
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if (ep+1) % 100 == 0:
            print(f"Ep {ep+1}: Loss={loss.item():.4f} (Trans={trans_loss.item():.4f}, Cont={contrastive_loss.item():.4f})")
        
    return model

# --- 3. Gradient Planner ---
def gradient_planner(model, s0, goal_emb, T, action_dim, lr=0.05, steps=1000):
    # Learnable logits
    logits = torch.zeros(1, T, action_dim, requires_grad=True)
    optimizer = optim.Adam([logits], lr=lr)
    
    for i in range(steps):
        optimizer.zero_grad()
        # Anneal temperature
        tau = max(0.1, 1.0 - (i / steps))
        a_soft = F.gumbel_softmax(logits, tau=tau, hard=False)
        
        curr_s = s0
        for t in range(T):
            # Transition in latent space
            # a_soft is (1, T, action_dim) -> slice (1, action_dim)
            action_t = a_soft[:, t, :]
            x = torch.cat([curr_s, action_t], dim=-1)
            curr_s = curr_s + model.transition(x)
            
        # Energy: Latent Euclidean Distance to Goal
        loss = torch.sum((curr_s - goal_emb)**2)
        
        loss.backward()
        optimizer.step()
        
    return logits.argmax(dim=-1).squeeze(0)

if __name__ == "__main__":
    env = MockEnv()
    # Input dim = 8*8 = 64
    model = ContrastiveWorldModel(64, 4)
    
    print("Training World Model with Contrastive Shaping...")
    model = train_contrastive_world_model(model, env, episodes=3000)
    
    # Planning Test
    print("\nPlanning...")
    env.reset()
    # Detach to prevent backprop through encoder weights and graph retention issues
    s0 = model.encoder(torch.tensor(env._get_obs(), dtype=torch.float32).unsqueeze(0)).detach()
    
    # Hack: encode the goal state
    env.pos = env.goal
    g_emb = model.encoder(torch.tensor(env._get_obs(), dtype=torch.float32).unsqueeze(0)).detach()
    env.reset() # Reset back to start
    
    # Freeze model for planning
    for p in model.parameters():
        p.requires_grad = False
    
    # Optimal path is 14 steps (7 right, 7 down)
    # Give it plenty of slack T=25
    plan = gradient_planner(model, s0, g_emb, T=25, action_dim=4, steps=1000)
    
    print("Plan found:", plan.tolist())
    
    # Execute Plan
    print("\nExecuting Plan...")
    curr_pos = env.pos.copy()
    print(f"Start: {curr_pos}")
    
    success = False
    for i, a in enumerate(plan):
        obs, r, done, _ = env.step(a.item())
        print(f"Step {i+1}: Action {a.item()} -> Pos {env.pos}")
        if done:
            success = True
            break
            
    if success:
        print("\nSUCCESS! Goal Reached.")
    else:
        print("\nFAILURE. Did not reach goal.")
        print(f"Final Distance: {np.linalg.norm(env.pos - env.goal)}")
