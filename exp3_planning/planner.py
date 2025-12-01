"""
Planners for Experiment 3.
"""
import torch
import torch.nn.functional as F
import numpy as np

class FlatPlanner:
    def __init__(self, encoder, f0, device='cpu', T_plan=20, n_actions=7):
        self.encoder = encoder
        self.f0 = f0
        self.device = device
        self.T_plan = T_plan
        self.n_actions = n_actions
        
    def plan(self, obs, goal_latent):
        # Encode current state
        with torch.no_grad():
            s_t = self.encoder(obs)
            
        # Decision variable: Logits
        A_logits = torch.zeros(self.T_plan, self.n_actions, requires_grad=True, device=self.device)
        optimizer = torch.optim.Adam([A_logits], lr=0.1)
        
        for _ in range(64): # n_gd_steps
            optimizer.zero_grad()
            a_soft = F.gumbel_softmax(A_logits, tau=1.0, hard=False)
            
            s = s_t
            for t in range(self.T_plan):
                s = self.f0(s, a_soft[t:t+1])
                
            loss = 0.5 * (s - goal_latent).pow(2).sum()
            loss.backward()
            optimizer.step()
            
        # Return first action
        with torch.no_grad():
            best_action = A_logits[0].argmax().item()
        return best_action


class HELMPlanner:
    def __init__(self, encoder, f0, f1, device='cpu', T_plan=20, n_actions=7, stride=4):
        self.encoder = encoder
        self.f0 = f0
        self.f1 = f1
        self.device = device
        self.T_plan = T_plan
        self.n_actions = n_actions
        self.stride = stride
        self.T1 = T_plan // stride
        
    def plan(self, obs, goal_latent):
        with torch.no_grad():
            s_t = self.encoder(obs)
            
        # --- Stage 1: Coarse Planning (f1) ---
        U_logits = torch.zeros(self.T1, self.n_actions, requires_grad=True, device=self.device)
        opt1 = torch.optim.Adam([U_logits], lr=0.1)
        
        for _ in range(32): # n_gd1_steps
            opt1.zero_grad()
            u_soft = F.gumbel_softmax(U_logits, tau=1.0, hard=False)
            
            s = s_t
            for k in range(self.T1):
                s = self.f1(s, u_soft[k:k+1])
                
            loss = 0.5 * (s - goal_latent).pow(2).sum()
            loss.backward()
            opt1.step()
            
        # --- Upsample ---
        # Repeat each macro-action N times
        A_init_logits = U_logits.repeat_interleave(self.stride, dim=0)
        A_logits = A_init_logits.clone().detach().requires_grad_(True)
        
        # --- Stage 2: Fine Refinement (f0) ---
        opt2 = torch.optim.Adam([A_logits], lr=0.05)
        
        for _ in range(32): # n_gd2_steps
            opt2.zero_grad()
            a_soft = F.gumbel_softmax(A_logits, tau=1.0, hard=False)
            
            s = s_t
            for t in range(self.T_plan):
                s = self.f0(s, a_soft[t:t+1])
                
            loss = 0.5 * (s - goal_latent).pow(2).sum()
            loss.backward()
            opt2.step()
            
        with torch.no_grad():
            best_action = A_logits[0].argmax().item()
        return best_action


class CEMPlanner:
    def __init__(self, encoder, f0, device='cpu', T_plan=20, n_actions=7):
        self.encoder = encoder
        self.f0 = f0
        self.device = device
        self.T_plan = T_plan
        self.n_actions = n_actions
        
    def plan(self, obs, goal_latent):
        with torch.no_grad():
            s_t = self.encoder(obs)
            
        # CEM Params
        pop_size = 64
        elite_frac = 0.2
        n_iters = 5
        n_elite = int(pop_size * elite_frac)
        
        mean = torch.zeros(self.T_plan, self.n_actions, device=self.device)
        std = torch.ones_like(mean) * 1.0
        
        for _ in range(n_iters):
            # Sample logits: (pop_size, T, n_actions)
            noise = torch.randn(pop_size, self.T_plan, self.n_actions, device=self.device)
            logits = mean.unsqueeze(0) + std.unsqueeze(0) * noise
            a_soft = F.softmax(logits, dim=-1)
            
            # Evaluate population
            energies = []
            for i in range(pop_size):
                s = s_t
                # Unroll f0
                for t in range(self.T_plan):
                    # Slice batch i, time t
                    # a_soft[i, t:t+1] is (1, n_actions)
                    s = self.f0(s, a_soft[i, t:t+1])
                    
                E = 0.5 * (s - goal_latent).pow(2).sum()
                energies.append(E.item())
                
            # Select elites
            energies = np.array(energies)
            elite_idx = energies.argsort()[:n_elite]
            elite_logits = logits[elite_idx] # (n_elite, T, n_actions)
            
            # Update distribution
            mean = elite_logits.mean(dim=0)
            std = elite_logits.std(dim=0) + 1e-4
            
        best_action = mean[0].argmax().item()
        return best_action
