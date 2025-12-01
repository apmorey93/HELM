"""
Experiment 1: Spectral Cliff Validation on Expansive Linear System.

Replaces Lorenz with a controlled linear system s_{t+1} = W s_t + B a_t
where W has eigenvalues > 1. This GUARANTEES that the ground truth
Jacobian explodes exponentially, providing a clean testbed for Lemma 1.

Metrics:
1. ||J(T)|| vs T: Should be linear (log scale) for Flat, saturated for HELM.
2. Planning Success vs T: Flat should collapse, HELM should persist.
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import os

# ==============================================================================
# 1. EXPANSIVE LINEAR SYSTEM
# ==============================================================================

class ExpansiveLinearSystem:
    def __init__(self, d_s=4, d_a=1, expansion_rate=1.1, device='cpu'):
        self.d_s = d_s
        self.d_a = d_a
        self.device = device
        
        # Construct W with specific eigenvalues
        # W = Q Lambda Q^-1
        eigvals = torch.linspace(expansion_rate, expansion_rate + 0.2, d_s)
        Q, _ = torch.linalg.qr(torch.randn(d_s, d_s))
        self.W = (Q @ torch.diag(eigvals) @ Q.T).to(device)
        
        self.B = torch.randn(d_s, d_a).to(device) * 0.5
        
        print(f"Initialized Linear System:")
        print(f"  Eigenvalues: {eigvals.numpy()}")
        print(f"  Max eigenvalue: {eigvals.max().item():.3f} (Expansive!)")

    def step(self, s, a):
        """s_{t+1} = W s_t + B a_t"""
        return s @ self.W.T + a @ self.B.T

    def rollout(self, s0, actions):
        """Unroll sequence."""
        s = s0
        for t in range(actions.shape[1]):
            s = self.step(s, actions[:, t])
        return s

    def generate_dataset(self, n_traj, T):
        """Generate (s, a, s_next) tuples for training."""
        s = torch.randn(n_traj, self.d_s, device=self.device)
        a = torch.randn(n_traj, self.d_a, device=self.device)
        s_next = self.step(s, a)
        return s, a, s_next


# ==============================================================================
# 2. MODELS
# ==============================================================================

class LinearWorldModel(nn.Module):
    """Simple MLP to learn the linear mapping."""
    def __init__(self, d_s, d_a, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_s + d_a, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, d_s)
        )
        
    def forward(self, s0, actions):
        """
        Predict s_T given s_0 and action sequence.
        Naive unrolling (not efficient, but correct for Jacobian).
        """
        s = s0
        for t in range(actions.shape[1]):
            # Single step prediction
            inp = torch.cat([s, actions[:, t]], dim=-1)
            s = self.net(inp)
        return s
        
    def step(self, s, a):
        """Single step for training."""
        inp = torch.cat([s, a], dim=-1)
        return self.net(inp)


def compute_spectral_reg(model, s, a):
    """Compute ||ds'/ds||^2 + ||ds'/da||^2"""
    s = s.clone().detach().requires_grad_(True)
    a = a.clone().detach().requires_grad_(True)
    
    s_next = model.step(s, a)
    
    grad_s = torch.autograd.grad(s_next.sum(), s, create_graph=True)[0]
    grad_a = torch.autograd.grad(s_next.sum(), a, create_graph=True)[0]
    
    return grad_s.pow(2).mean() + grad_a.pow(2).mean()


# ==============================================================================
# 3. TRAINING
# ==============================================================================

def train_models(system, device='cpu'):
    print("\n" + "="*60)
    print(" TRAINING MODELS ".center(60, "="))
    print("="*60)
    
    # Generate training data (single steps)
    s_train, a_train, s_next_train = system.generate_dataset(n_traj=5000, T=1)
    
    # --- Train Flat ---
    print("\nTraining Flat Model...")
    flat_model = LinearWorldModel(system.d_s, system.d_a).to(device)
    opt = torch.optim.Adam(flat_model.parameters(), lr=1e-3)
    
    for epoch in range(500):
        opt.zero_grad()
        pred = flat_model.step(s_train, a_train)
        loss = (pred - s_next_train).pow(2).mean()
        loss.backward()
        opt.step()
        
        if epoch % 100 == 0:
            print(f"  Epoch {epoch}: loss={loss.item():.6f}")
            
    # --- Train HELM ---
    print("\nTraining HELM Model (lambda=0.1)...")
    helm_model = LinearWorldModel(system.d_s, system.d_a).to(device)
    opt = torch.optim.Adam(helm_model.parameters(), lr=1e-3)
    lambda_spec = 0.1
    
    for epoch in range(500):
        opt.zero_grad()
        pred = helm_model.step(s_train, a_train)
        mse = (pred - s_next_train).pow(2).mean()
        reg = compute_spectral_reg(helm_model, s_train, a_train)
        
        loss = mse + lambda_spec * reg
        loss.backward()
        opt.step()
        
        if epoch % 100 == 0:
            print(f"  Epoch {epoch}: mse={mse.item():.6f}, reg={reg.item():.6f}")
            
    return flat_model, helm_model


# ==============================================================================
# 4. VALIDATION
# ==============================================================================

def measure_jacobian(model, T_list, d_s, d_a, device):
    norms = []
    for T in T_list:
        s0 = torch.randn(1, d_s, device=device)
        actions = torch.randn(1, T, d_a, device=device, requires_grad=True)
        
        s_T = model(s0, actions)
        
        # Frobenius norm of Jacobian d(s_T)/d(actions)
        # Approx: sum of gradients
        grad = torch.autograd.grad(s_T.sum(), actions)[0]
        norm = torch.norm(grad).item()
        norms.append(norm)
    return norms

def measure_planning(model, T_list, d_s, d_a, device):
    success_rates = []
    for T in T_list:
        success = 0
        for _ in range(20): # 20 trials
            s0 = torch.randn(1, d_s, device=device)
            # Generate valid target
            with torch.no_grad():
                tgt_a = torch.randn(1, T, d_a, device=device)
                s_target = model(s0, tgt_a)
            
            # Plan
            plan_a = torch.zeros(1, T, d_a, device=device, requires_grad=True)
            opt = torch.optim.Adam([plan_a], lr=0.05)
            
            for _ in range(100):
                opt.zero_grad()
                pred = model(s0, plan_a)
                loss = (pred - s_target).pow(2).sum()
                loss.backward()
                opt.step()
                if loss.item() < 0.1:
                    success += 1
                    break
        success_rates.append(success / 20 * 100)
    return success_rates

def run_experiment():
    device = 'cpu'
    system = ExpansiveLinearSystem(device=device)
    
    flat_model, helm_model = train_models(system, device)
    
    T_list = [5, 10, 20, 30, 40, 50]
    
    print("\n" + "="*60)
    print(" VALIDATION RESULTS ".center(60, "="))
    print("="*60)
    
    # 1. Jacobian
    print("\nMeasuring Jacobians...")
    j_flat = measure_jacobian(flat_model, T_list, system.d_s, system.d_a, device)
    j_helm = measure_jacobian(helm_model, T_list, system.d_s, system.d_a, device)
    
    # 2. Planning
    print("Measuring Planning Success...")
    p_flat = measure_planning(flat_model, T_list, system.d_s, system.d_a, device)
    p_helm = measure_planning(helm_model, T_list, system.d_s, system.d_a, device)
    
    # Print Table
    print("\nResults Summary:")
    print(f"{'T':<5} | {'J_Flat':<10} | {'J_HELM':<10} | {'Plan_F':<8} | {'Plan_H':<8}")
    print("-" * 55)
    for i, T in enumerate(T_list):
        print(f"{T:<5} | {j_flat[i]:<10.2f} | {j_helm[i]:<10.2f} | {p_flat[i]:<8.0f} | {p_helm[i]:<8.0f}")
        
    # Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    ax1.plot(T_list, np.log(j_flat), 'r-o', label='Flat')
    ax1.plot(T_list, np.log(j_helm), 'g-s', label='HELM')
    ax1.set_title('Log Jacobian Norm (Spectral Cliff)')
    ax1.set_xlabel('Horizon T')
    ax1.set_ylabel('log ||J||')
    ax1.legend()
    ax1.grid(True)
    
    ax2.plot(T_list, p_flat, 'r-o', label='Flat')
    ax2.plot(T_list, p_helm, 'g-s', label='HELM')
    ax2.set_title('Planning Success Rate')
    ax2.set_xlabel('Horizon T')
    ax2.set_ylabel('Success %')
    ax2.legend()
    ax2.grid(True)
    
    os.makedirs('final_plots', exist_ok=True)
    plt.savefig('final_plots/exp1_linear_cliff.png')
    print("\nSaved plot to final_plots/exp1_linear_cliff.png")

if __name__ == "__main__":
    run_experiment()
