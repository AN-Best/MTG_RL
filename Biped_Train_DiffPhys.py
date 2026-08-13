import os
import glob
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import urllib.request

from Biped_Simulation import MuscleTorqueGenerator
from SymPy_Skeleton import skeleton_dynamics

# Define device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device for DiffPhys training: {device}")

# -----------------------------------------------------------------------------
# 1. Deterministic Policy Network
# -----------------------------------------------------------------------------
class DifferentiablePolicy(nn.Module):
    def __init__(self, obs_dim=17, action_dim=6):
        super().__init__()
        # Multi-layer perceptron for joint targets
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 128),
            nn.Tanh(),
            nn.Linear(128, 128),
            nn.Tanh(),
            nn.Linear(128, action_dim)
        )
        
        # Initialize final layer to zero so policy starts by outputting nominal actions
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)
        
        # Scaling limits (limits match Gym environment)
        self.register_buffer("limits_low", torch.tensor([-0.5, -2.0, -0.5, -0.5, -2.0, -0.5], dtype=torch.float32))
        self.register_buffer("limits_high", torch.tensor([1.5, 0.0, 0.5, 1.5, 0.0, 0.5], dtype=torch.float32))

    def forward(self, obs):
        # Output a small residual joint angle offset in [-0.3, 0.3] radians
        return 0.3 * torch.tanh(self.net(obs))

# -----------------------------------------------------------------------------
# 2. Reference Motion Loader
# -----------------------------------------------------------------------------
def load_reference_motion(num_samples=10000):
    url = "https://raw.githubusercontent.com/opensim-org/opensim-models/master/Pipelines/Gait2392_Simbody/subject01_walk1_ik.mot"
    local_path = "subject01_walk1_ik.mot"
    
    if not os.path.exists(local_path):
        print(f"Downloading reference motion from {url}...")
        urllib.request.urlretrieve(url, local_path)
        
    try:
        with open(local_path, "r") as f:
            lines = f.readlines()
        data_start = 0
        for idx, line in enumerate(lines):
            if line.strip() == "endheader":
                data_start = idx + 1
                break
                
        col_headers = lines[data_start].strip().split()
        data_lines = [list(map(float, line.strip().split())) for line in lines[data_start+1:] if line.strip()]
        data = np.array(data_lines)
        
        # Crop to the optimal periodic gait cycle (0.4s to 1.6s)
        idx_start = np.where(np.abs(data[:, 0] - 0.40) < 1e-4)[0][0]
        idx_end = np.where(np.abs(data[:, 0] - 1.60) < 1e-4)[0][0]
        data = data[idx_start:idx_end+1]
        
        times = data[:, 0]
        conv = np.pi / 180.0
        
        # Extract coordinates
        q_lh_raw = data[:, col_headers.index("hip_flexion_l")] * conv
        q_lk_raw = data[:, col_headers.index("knee_angle_l")] * conv
        q_la_raw = data[:, col_headers.index("ankle_angle_l")] * conv
        q_rh_raw = data[:, col_headers.index("hip_flexion_r")] * conv
        q_rk_raw = data[:, col_headers.index("knee_angle_r")] * conv
        q_ra_raw = data[:, col_headers.index("ankle_angle_r")] * conv
        
        q_tr_raw = np.zeros_like(times)
        
        # Mirror treadmill pelvis coordinates into forward overground walking
        cropped_data = data[idx_start:idx_end+1]
        
        times = cropped_data[:, 0]
        
        # Extract columns using exact OpenSim Gait2392 header indices
        # 0: time, 1: pelvis_tilt, 4: pelvis_tx, 5: pelvis_ty
        # 7: hip_flexion_r, 10: knee_angle_r, 11: ankle_angle_r
        # 14: hip_flexion_l, 17: knee_angle_l, 18: ankle_angle_l
        pelvis_tx_raw = cropped_data[:, 4]
        pelvis_ty_raw = cropped_data[:, 5] - 0.1070  # Retarget pelvis height down by 10.7cm (compensates for 1.5cm weight sag)
        pelvis_tilt_raw = -np.radians(cropped_data[:, 1])
        
        hip_flex_r_raw = np.radians(cropped_data[:, 7])
        knee_ang_r_raw = np.radians(cropped_data[:, 10]) - np.where(hip_flex_r_raw > 0.05, 0.08, 0.0)
        ankle_ang_r_raw = np.radians(cropped_data[:, 11]) + 0.03
        
        hip_flex_l_raw = np.radians(cropped_data[:, 14])
        knee_ang_l_raw = np.radians(cropped_data[:, 17]) - np.where(hip_flex_l_raw > 0.05, 0.08, 0.0)
        ankle_ang_l_raw = np.radians(cropped_data[:, 18]) + 0.03
        
        # Build continuous dataset of size `num_samples`
        cycle_duration = times[-1] - times[0]
        cycle_distance = pelvis_tx_raw[-1] - pelvis_tx_raw[0]
        target_t = np.linspace(times[0], times[0] + cycle_duration * (num_samples / len(times)), num_samples)
        mod_t = times[0] + np.mod(target_t - times[0], cycle_duration)
        cycle_count = np.floor((target_t - times[0]) / cycle_duration)
        
        # Interpolate coordinates
        q_x = np.interp(mod_t, times, pelvis_tx_raw) + cycle_distance * cycle_count
        q_y = np.interp(mod_t, times, pelvis_ty_raw)
        q_tr = np.interp(mod_t, times, pelvis_tilt_raw)
        q_lh = np.interp(mod_t, times, hip_flex_l_raw)
        q_lk = np.interp(mod_t, times, knee_ang_l_raw)
        q_la = np.interp(mod_t, times, ankle_ang_l_raw)
        q_rh = np.interp(mod_t, times, hip_flex_r_raw)
        q_rk = np.interp(mod_t, times, knee_ang_r_raw)
        q_ra = np.interp(mod_t, times, ankle_ang_r_raw)
        
        q = np.stack([q_x, q_y, q_tr, q_lh, q_lk, q_la, q_rh, q_rk, q_ra], axis=-1)
        
        dt = target_t[1] - target_t[0]
        u = np.zeros_like(q)
        u[1:-1] = (q[2:] - q[:-2]) / (2.0 * dt)
        u[0] = (q[1] - q[0]) / dt
        u[-1] = (q[-1] - q[-2]) / dt
        
        q_tensor = torch.tensor(q, dtype=torch.float32, device=device)
        u_tensor = torch.tensor(u, dtype=torch.float32, device=device)
        return q_tensor, u_tensor
        
    except Exception as e:
        print(f"Error parsing reference motion: {e}")
        # Fallback mathematically
        t = torch.linspace(0, num_samples * 0.1, num_samples, device=device)
        q_x = 1.0 * t
        q_y = 0.90 + 0.03 * torch.cos(4.0 * torch.pi * t)
        q_tr = torch.zeros_like(t)
        q_lh = 0.3 * torch.sin(2.0 * torch.pi * t)
        q_lk = -0.4 * torch.sin(2.0 * torch.pi * t + 0.5)
        q_la = 0.1 * torch.cos(2.0 * torch.pi * t)
        q_rh = -q_lh
        q_rk = -q_lk
        q_ra = -q_la
        q = torch.stack([q_x, q_y, q_tr, q_lh, q_lk, q_la, q_rh, q_rk, q_ra], dim=-1)
        u = torch.zeros_like(q)
        u[1:-1] = (q[2:] - q[:-2]) / 0.2
        return q, u

# -----------------------------------------------------------------------------
# 3. Differentiable Step Integrator
# -----------------------------------------------------------------------------
def differentiable_step(q_curr, u_curr, q_ref_target, params, mtg_controller, dt_sim=0.002, dt_control=0.1):
    """Integrates biped skeleton dynamics while preserving torch computation graph."""
    num_steps = int(round(dt_control / dt_sim))
    tau_prev = torch.zeros((q_curr.shape[0], 6), dtype=torch.float32, device=q_curr.device)
    
    q_t = q_curr
    u_t = u_curr
    tau_t = tau_prev
    
    # Pre-unpack parameters for speed and differentiability
    # params structure: [m1..m4, l1..l6, c1..c4, I1..I4, g, h_g, kc, cc, mu, vs]
    g = params[0, 18]
    h_g = params[0, 19]
    kc = params[0, 20]
    cc = params[0, 21]
    mu = params[0, 22]
    vs = params[0, 23]
    
    # Foot segment limits
    l_heel = params[0, 8]
    l_toe = params[0, 9]
    
    for _ in range(num_steps):
        q_j = q_t[:, 3:9]
        u_j = u_t[:, 3:9]
        tau_active, tau_j = mtg_controller.compute_torques(q_j, u_j, q_ref_target, tau_t, dt_sim)
        dq, du = skeleton_dynamics(q_t, u_t, tau_j, params)
        u_t = u_t + dt_sim * du
        q_t = q_t + dt_sim * u_t
        tau_t = tau_active
        
    return q_t, u_t, tau_t

# -----------------------------------------------------------------------------
# 4. Trajectory Rollout & Training
# -----------------------------------------------------------------------------
def train_diffphys(num_epochs=300, batch_size=64, rollout_steps=40):
    q_ref, u_ref = load_reference_motion(num_samples=10000)
    num_samples = q_ref.shape[0]
    
    policy = DifferentiablePolicy(obs_dim=17, action_dim=6).to(device)
    optimizer = optim.Adam(policy.parameters(), lr=1e-3)
    mtg_controller = MuscleTorqueGenerator(device=device)
    
    # Environment parameters (aligned with Biped_Simulation.py test)
    single_params = torch.tensor([
        45.0, 10.0, 5.0, 1.5,   # Masses
        0.8, 0.45, 0.45, 0.25, 0.05, 0.2, # Lengths
        0.4, 0.2, 0.2, 0.08,    # COMs
        1.6, 0.1, 0.05, 0.01,   # Inertias
        9.81, 0.90,             # g, h_g
        1e6, 0.60, 0.8, 0.04    # kc, cc, mu, vs (cc=0.60 realistic Hunt-Crossley damping, vs=0.04 smooth stiction)
    ], dtype=torch.float32, device=device)
    params = single_params.unsqueeze(0).repeat(batch_size, 1)
    
    print("\nStarting Differentiable Physics Policy Search...")
    print(f"Rollout length: {rollout_steps} control cycles ({rollout_steps * 0.1:.1f}s) | Batch size: {batch_size}")
    
    for epoch in range(num_epochs):
        # 1. Reference State Initialization (RSI)
        # Sample random phases from the reference gait
        start_indices = torch.randint(0, num_samples - rollout_steps, (batch_size,), device=device)
        
        q_curr = q_ref[start_indices].clone()
        u_curr = u_ref[start_indices].clone()
        
        # Make states require gradients (except for the initial positions/velocities from reference)
        q_curr = q_curr.detach()
        u_curr = u_curr.detach()
        
        total_loss = 0.0
        survival_steps = torch.zeros(batch_size, device=device)
        active_mask = torch.ones(batch_size, dtype=torch.bool, device=device)
        
        # Track simulated positions for gradient flow
        for t in range(rollout_steps):
            # Compute observations
            obs = torch.cat([q_curr[:, 1:], u_curr], dim=-1)
            
            # Predict joint angle targets (residual offset added to current reference joints)
            ref_idx = start_indices + t
            q_ref_current_joints = q_ref[ref_idx, 3:]
            q_ref_target = q_ref_current_joints + policy(obs)
            # Clamp to physical joint limits
            q_ref_target = torch.clamp(q_ref_target, min=policy.limits_low, max=policy.limits_high)
            
            # Integrate physics step (differentiable)
            q_next, u_next, _ = differentiable_step(
                q_curr, u_curr, q_ref_target, params, mtg_controller, dt_sim=0.002, dt_control=0.1
            )
            
            # Clean states from NaNs to prevent gradient corruption during backward pass
            q_next = torch.where(torch.isnan(q_next), torch.zeros_like(q_next), q_next)
            u_next = torch.where(torch.isnan(u_next), torch.zeros_like(u_next), u_next)
            
            # Check for falls/failures (non-vectorized logic for termination tracking)
            torso_angle = q_next[:, 2]
            hip_height = q_next[:, 1]
            vel_x = u_next[:, 0]
            
            fell = (torch.abs(torso_angle) > 0.8) | (hip_height < 0.65) | (torch.isnan(q_next).any(dim=-1))
            
            # Accumulate survival steps
            survival_steps += active_mask.float()
            active_mask = active_mask & (~fell)
            
            # If all environments fell, break rollout early
            if not active_mask.any():
                break
                
            # Get corresponding reference targets
            ref_idx = start_indices + t + 1
            q_ref_target_state = q_ref[ref_idx]
            u_ref_target_state = u_ref[ref_idx]
            
            # Compute step losses (only for environments that haven't fallen yet)
            pose_err = torch.mean((q_next[:, 3:] - q_ref_target_state[:, 3:])**2, dim=-1)
            vel_err = torch.mean((u_next[:, 3:] - u_ref_target_state[:, 3:])**2, dim=-1)
            upright_err = q_next[:, 2]**2
            height_err = (q_next[:, 1] - q_ref_target_state[:, 1])**2
            speed_err = (u_next[:, 0] - 1.0)**2
            
            # Sum individual losses scaled by the active mask (using torch.where to prevent NaNs from propagating)
            step_loss = (1.5 * pose_err + 0.1 * vel_err + 3.0 * upright_err + 2.0 * height_err + 2.0 * speed_err)
            step_loss_masked = torch.mean(torch.where(active_mask, step_loss, torch.zeros_like(step_loss)))
            
            total_loss += step_loss_masked
            
            # Update states for next step
            q_curr = q_next
            u_curr = u_next
            
        # 2. Backpropagation
        optimizer.zero_grad()
        if total_loss > 0:
            total_loss.backward()
            # Check for NaN gradients before updating (skip corrupt steps)
            has_nan_grad = any(
                p.grad is not None and torch.isnan(p.grad).any()
                for p in policy.net.parameters()
            )
            if has_nan_grad:
                optimizer.zero_grad()  # discard corrupt gradients entirely
            else:
                # Gradient clipping to prevent large but finite gradient spikes
                torch.nn.utils.clip_grad_norm_(policy.net.parameters(), 1.0)
                optimizer.step()
            
        # Log progress
        mean_survival = survival_steps.mean().item()
        mean_loss = total_loss.item() if isinstance(total_loss, torch.Tensor) else total_loss
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch {epoch+1:3d} | Loss: {mean_loss:7.4f} | Mean Survival: {mean_survival:5.1f} steps")
            
        # Save checkpoints
        if (epoch + 1) % 50 == 0:
            os.makedirs("runs/biped_diffphys/checkpoints", exist_ok=True)
            # Save in a dict structure that matches our animate script expectation
            checkpoint = {
                "policy": policy.state_dict(),
                "state_preprocessor": None # DiffPhys doesn't require a state preprocessor due to direct gradients
            }
            torch.save(checkpoint, f"runs/biped_diffphys/checkpoints/agent_{epoch+1}.pt")
            
    # Save final agent
    os.makedirs("runs/biped_diffphys/checkpoints", exist_ok=True)
    checkpoint = {
        "policy": policy.state_dict(),
        "state_preprocessor": None
    }
    torch.save(checkpoint, "runs/biped_diffphys/checkpoints/agent_final.pt")
    print("\nTraining completed successfully! Final checkpoint saved to runs/biped_diffphys/checkpoints/agent_final.pt")

if __name__ == "__main__":
    train_diffphys()
