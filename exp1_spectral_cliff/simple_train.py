"""Ultra simple training for Exp1 - one-step prediction."""
import torch
import torch.nn as nn
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from common.config import Exp1Config
from dataset import generate_controlled_lorenz_trajectories  
from models import FlatWorldModel, HELMWorldModel, compute_spectral_regularization

device = 'cpu'
config = Exp1Config()

# Generate dataset
print("Generating 1000 Lorenz trajectories...")
all_states, all_actions = generate_controlled_lorenz_trajectories(num_traj=1000, T=5, d_a=1, device=device)

# Create single-step transitions
s_t = all_states[:, :-1, :].reshape(-1, 3)  # 5000 x 3
s_next = all_states[:, 1:, :].reshape(-1,3)
a_t = all_actions.reshape(-1, 1, 1)  # 5000 x 1 x 1 (need time dim for model)

print(f"Dataset: {len(s_t)} transitions")

######## FLAT MODEL ########  
print("\n" + "="*70)
print("TRAINING FLAT MODEL")
print("="*70)

flat = FlatWorldModel(d_s=3, d_a=1, hidden_dim=64).to(device)
opt = torch.optim.Adam(flat.parameters(), lr=1e-3)

for epoch in range(10):
    total_loss = 0
    for i in range(0, len(s_t), 64):
        s = s_t[i:i+64]
        a = a_t[i:i+64]
        s_tgt = s_next[i:i+64]
        
        opt.zero_grad()
        s_pred = flat(s, a)  # Forward takes (s0, actions_seq)
        loss = (s_pred - s_tgt).pow(2).mean()
        loss.backward()
        opt.step()
        total_loss += loss.item()
    
    if (epoch+1) % 2 == 0:
        print(f"Epoch {epoch+1}: loss={total_loss/(len(s_t)//64):.6f}")

os.makedirs('checkpoints', exist_ok=True)
torch.save(flat.state_dict(), 'checkpoints/flat_world_model.pt')
print("Saved flat_world_model.pt\n")

######## HELM MODEL ########
print("="*70)
print("TRAINING HELM MODEL")
print("="*70)

helm = HELMWorldModel(d_s=3, d_a=1, hidden_dim=64).to(device)
opt2 = torch.optim.Adam(helm.parameters(), lr=1e-3)

for epoch in range(10):
    total_mse = 0
    total_spec = 0
    for i in range(0, len(s_t), 64):
        s = s_t[i:i+64].clone()
        a = a_t[i:i+64].clone()
        s_tgt = s_next[i:i+64]
        
        s.requires_grad_(True)
        a.requires_grad_(True)
        
        opt2.zero_grad()
        s_pred = helm(s, a)
        mse = (s_pred - s_tgt).pow(2).mean()
        spec = compute_spectral_regularization(helm, s, a)
        
        loss = mse + config.lambda_spectral * spec
        loss.backward()
        opt2.step()
        
        total_mse += mse.item()
        total_spec += spec.item()
    
    if (epoch+1) % 2 == 0:
        avg_mse = total_mse / (len(s_t)//64)
        avg_spec = total_spec / (len(s_t)//64)
        spec_pct = config.lambda_spectral * avg_spec / (avg_mse + config.lambda_spectral * avg_spec) * 100
        print(f"Epoch {epoch+1}: mse={avg_mse:.6f}, spec={avg_spec:.6f}, spec%={spec_pct:.1f}%")

torch.save(helm.state_dict(), 'checkpoints/helm_world_model.pt')
print("Saved helm_world_model.pt\n")

print("="*70)
print("DONE - Now run: python validate_exp1.py")
print("="*70)
