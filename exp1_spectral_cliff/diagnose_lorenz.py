"""
Diagnostic script: Check if Lorenz environment actually explodes vs Model.
"""
import torch
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dataset import lorenz_step_with_control
from models import FlatWorldModel

def get_env_jacobian_norm(T=50, eps=1e-4):
    """Finite difference Jacobian of the actual environment."""
    s0 = torch.randn(1, 3)
    actions = torch.randn(1, T, 1) * 0.5
    
    # Base rollout
    s = s0.clone()
    for t in range(T):
        s = lorenz_step_with_control(s, actions[:, t])
    base_sT = s
    
    # Perturb each action
    grads = []
    for t in range(T):
        a_pert = actions.clone()
        a_pert[:, t] += eps
        
        s = s0.clone()
        for step in range(T):
            s = lorenz_step_with_control(s, a_pert[:, step])
        
        diff = (s - base_sT) / eps
        grads.append(diff.flatten())
    
    J = torch.stack(grads)
    return torch.norm(J).item()

def get_model_jacobian_norm(model, T=50):
    """Autograd Jacobian of the trained model."""
    s0 = torch.randn(1, 3)
    actions = torch.randn(1, T, 1, requires_grad=True)
    
    s_pred = model(s0, actions)
    
    # Compute full Jacobian norm via autograd
    # Approximate with vector-Jacobian product for speed if needed, 
    # but for diagnostic we can just do sum() grad for a rough scale check
    # or proper Frobenius norm
    
    grad_a = torch.autograd.grad(s_pred.sum(), actions, create_graph=False)[0]
    return torch.norm(grad_a).item() # This is a lower bound proxy, but sufficient for scale

if __name__ == "__main__":
    print("Checking Jacobian scales (T=50)...")
    
    # 1. Environment
    env_norm = get_env_jacobian_norm(T=50)
    print(f"Environment Jacobian Norm: {env_norm:.4f}")
    
    # 2. Model
    try:
        model = FlatWorldModel(d_s=3, d_a=1, hidden_dim=64)
        model.load_state_dict(torch.load('checkpoints/flat_world_model.pt'))
        model.eval()
        
        model_norm = get_model_jacobian_norm(model, T=50)
        print(f"Model Jacobian Norm:       {model_norm:.4f}")
        
        ratio = env_norm / (model_norm + 1e-6)
        print(f"Ratio (Env/Model):         {ratio:.1f}x")
        
        if ratio > 10.0:
            print("\nDIAGNOSIS CONFIRMED: Model is stabilizing chaos.")
        else:
            print("\nDIAGNOSIS: Model matches environment (or env doesn't explode).")
            
    except Exception as e:
        print(f"Could not load model: {e}")
