import os
import time
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from Biped_Actuators import create_actuator
from Biped_Dynamics import get_default_body_parameters, simulate_macro_step
from SymPy_Skeleton import skeleton_dynamics

# Define device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device for 10-DoF Differentiable Physics: {device}")

# -----------------------------------------------------------------------------
# 1. Reference Kinematics Loader with Static GRF Equilibrium Height Calibration
# -----------------------------------------------------------------------------
def load_reference_motion(num_samples=10000, device=device):
    local_path = "subject01_walk1_ik.mot"
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
    
    idx_start = np.where(np.abs(data[:, 0] - 0.40) < 1e-4)[0][0]
    idx_end = np.where(np.abs(data[:, 0] - 1.60) < 1e-4)[0][0]
    data = data[idx_start:idx_end+1]
    
    times = data[:, 0]
    conv = np.pi / 180.0
    
    pelvis_tx_raw = data[:, col_headers.index("pelvis_tx")]
    pelvis_tilt_raw = data[:, col_headers.index("pelvis_tilt")] * conv
    
    torso_des = 0.035 + 0.015 * np.cos(2.0 * np.pi * (times - times[0]) / (times[-1] - times[0]))
    q_lum_raw = torso_des - pelvis_tilt_raw
    
    q_lh_raw = data[:, col_headers.index("hip_flexion_l")] * conv
    q_lk_raw = data[:, col_headers.index("knee_angle_l")] * conv
    q_la_raw = data[:, col_headers.index("ankle_angle_l")] * conv
    
    q_rh_raw = data[:, col_headers.index("hip_flexion_r")] * conv
    q_rk_raw = data[:, col_headers.index("knee_angle_r")] * conv
    q_ra_raw = data[:, col_headers.index("ankle_angle_r")] * conv
    
    # Calibrate initial equilibrium pelvis height
    calib_params = get_default_body_parameters(batch_size=1, device=device)
    dummy_u = torch.zeros((1, 10), device=device)
    dummy_tau = torch.zeros((1, 7), device=device)
    
    calibrated_y = []
    for t in range(len(times)):
        y_low, y_high = 0.70, 1.15
        best_y = 0.88
        for _ in range(20):
            y_mid = 0.5 * (y_low + y_high)
            q_test = torch.tensor([[
                0.0, y_mid, pelvis_tilt_raw[t], q_lum_raw[t],
                q_lh_raw[t], q_lk_raw[t], q_la_raw[t],
                q_rh_raw[t], q_rk_raw[t], q_ra_raw[t]
            ]], dtype=torch.float32, device=device)
            dq, du = skeleton_dynamics(q_test, dummy_u, dummy_tau, calib_params)
            if du[0, 1].item() < 0.0:
                y_high = y_mid
            else:
                y_low = y_mid
                best_y = y_mid
        calibrated_y.append(best_y)

    q_x_raw = pelvis_tx_raw - pelvis_tx_raw[0]
    q_y_raw = np.array(calibrated_y)
    q_p_raw = pelvis_tilt_raw
    
    n_raw = len(times)
    stride_len = n_raw
    cycle_distance = q_x_raw[-1] - q_x_raw[0]
    cycle_duration = times[-1] - times[0]
    target_t = np.linspace(times[0], times[0] + cycle_duration * (num_samples / stride_len), num_samples)
    mod_t = times[0] + np.mod(target_t - times[0], cycle_duration)
    cycle_count = np.floor((target_t - times[0]) / cycle_duration)
    
    q_x = np.interp(mod_t, times, q_x_raw) + cycle_distance * cycle_count
    q_y = np.interp(mod_t, times, q_y_raw)
    q_p = np.interp(mod_t, times, q_p_raw)
    q_lum = np.interp(mod_t, times, q_lum_raw)
    q_lh = np.interp(mod_t, times, q_lh_raw)
    q_lk = np.interp(mod_t, times, q_lk_raw)
    q_la = np.interp(mod_t, times, q_la_raw)
    q_rh = np.interp(mod_t, times, q_rh_raw)
    q_rk = np.interp(mod_t, times, q_rk_raw)
    q_ra = np.interp(mod_t, times, q_ra_raw)
    
    q = np.stack([q_x, q_y, q_p, q_lum, q_lh, q_lk, q_la, q_rh, q_rk, q_ra], axis=-1)
    
    dt = target_t[1] - target_t[0]
    u = np.zeros_like(q)
    u[1:-1] = (q[2:] - q[:-2]) / (2.0 * dt)
    u[0] = (q[1] - q[0]) / dt
    u[-1] = (q[-1] - q[-2]) / dt
    
    q_tensor = torch.tensor(q, dtype=torch.float32, device=device)
    u_tensor = torch.tensor(u, dtype=torch.float32, device=device)
    obs_ref = torch.cat([q_tensor[:, 1:], u_tensor], dim=-1)
    return obs_ref, q_tensor, u_tensor

# -----------------------------------------------------------------------------
# 2. Differentiable Policy Network (19 observations -> 7 Residual Offsets)
# -----------------------------------------------------------------------------
class DifferentiablePolicy(nn.Module):
    def __init__(self, obs_dim=19, action_dim=7):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 128),
            nn.ELU(),
            nn.Linear(128, 128),
            nn.ELU(),
            nn.Linear(128, action_dim)
        )
        # Initialize output layer to zero for clean nominal tracking on step 0
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, obs):
        # Outputs small residual angle offset in [-0.3, 0.3] rad
        return 0.3 * torch.tanh(self.net(obs))

# -----------------------------------------------------------------------------
# 3. Differentiable Training Loop
# -----------------------------------------------------------------------------
def train(actuator_type="ideal_unlimited", num_epochs=300, batch_size=128, rollout_steps=60):
    save_dir = f"runs/biped_diffphys_{actuator_type}/checkpoints"
    os.makedirs(save_dir, exist_ok=True)
    
    # 1. Load Reference Motion & Initialize Actuator
    _, q_ref, u_ref = load_reference_motion(num_samples=10000, device=device)
    num_samples = q_ref.shape[0]
    
    actuator = create_actuator(actuator_type=actuator_type, device=device)
    params = get_default_body_parameters(batch_size=batch_size, device=device)
    
    policy = DifferentiablePolicy(obs_dim=19, action_dim=7).to(device)
    optimizer = optim.Adam(policy.parameters(), lr=2e-3)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-4)
    
    print("\n" + "=" * 80)
    print(f"  10-DoF Differentiable Physics Training  |  Actuator: {actuator_type.upper()}")
    print(f"  Batch size: {batch_size:,} envs  |  Total Epochs: {num_epochs}  |  Rollout Horizon: {rollout_steps} steps (1.2s)")
    print("=" * 80)
    print(f"  {'Epoch':>6}  {'Loss':>10}  {'Pose-Err':>10}  {'Pelvis-Tilt':>12}  {'Upright':>10}  {'Survival':>10}  {'Time':>7}")
    print("-" * 80)
    
    start_time = time.time()
    
    for epoch in range(num_epochs):
        t0 = time.time()
        curr_rollout = rollout_steps
        
        # 50% start at stride phase 0 (calibrated equilibrium), 50% across diverse gait phases (RSI)
        start_indices = torch.randint(0, num_samples - curr_rollout - 1, (batch_size,), device=device)
        start_indices[:int(batch_size * 0.5)] = 0
        
        q_curr = q_ref[start_indices].clone().detach()
        u_curr = u_ref[start_indices].clone().detach()
        q_curr[:, 0] = 0.0 # reset x coordinate
        
        # Initialize actuator state (muscle activations for MTG, None for ideal torques)
        actuator_state = torch.full((batch_size, 7), 0.01, device=device) if actuator_type == "mtg" else None
        
        total_loss = 0.0
        total_pose_err = 0.0
        total_pelvis_err = 0.0
        total_upright_err = 0.0
        survival_steps = torch.zeros(batch_size, device=device)
        active_mask = torch.ones(batch_size, dtype=torch.bool, device=device)
        
        tbptt_window = 10 # 10-step sliding window (0.20s) for stable BPTT gradient propagation
        window_loss = 0.0
        
        for t in range(curr_rollout):
            # Observation vector: [q_{1:10}, u_{0:10}]
            obs = torch.cat([q_curr[:, 1:], u_curr], dim=-1)
            
            # Policy predicts residual target joint angle adjustments delta_q
            ref_idx = start_indices + t
            q_target_ref = q_ref[ref_idx, 3:]
            delta_q = policy(obs)
            q_des = q_target_ref + delta_q
            
            # Differentiable macro-step simulation (Actuator internally converts q_des -> torques)
            q_next, u_next, actuator_state_next, tau_applied = simulate_macro_step(
                q_curr, u_curr, actuator_state, q_des, actuator, params,
                dt_sim=0.002, dt_control=0.02
            )
            
            # Diagnostics & failure checks
            pelvis_angle = q_next[:, 2]
            lumbar_angle = q_next[:, 3]
            torso_angle_world = pelvis_angle + lumbar_angle
            hip_height = q_next[:, 1]
            vel_x = u_next[:, 0]
            
            ref_next_idx = start_indices + t + 1
            q_ref_target_state = q_ref[ref_next_idx]
            u_ref_target_state = u_ref[ref_next_idx]
            torso_angle_ref = q_ref_target_state[:, 2] + q_ref_target_state[:, 3]
            
            # Locomotion balance & survival criteria
            fell = (torch.abs(torso_angle_world - torso_angle_ref) > 0.60) | (hip_height < 0.70) | \
                   (vel_x < -0.50) | (vel_x > 5.0) | torch.isnan(q_next).any(dim=-1)
            
            survival_steps += active_mask.float()
            active_mask = active_mask & (~fell)
            
            if not active_mask.any():
                break
                
            # Step Loss Terms
            pose_loss = torch.mean((q_next[:, 3:10] - q_ref_target_state[:, 3:10]) ** 2, dim=-1)
            vel_loss = torch.mean((u_next[:, 3:10] - u_ref_target_state[:, 3:10]) ** 2, dim=-1)
            pelvis_pitch_loss = (pelvis_angle - q_ref_target_state[:, 2]) ** 2
            upright_loss = (torso_angle_world - torso_angle_ref) ** 2
            torso_vel_loss = (u_next[:, 2] + u_next[:, 3]) ** 2
            height_loss = (hip_height - q_ref_target_state[:, 1]) ** 2
            
            # Differentiable Stability Barriers
            barrier_height = torch.relu(0.82 - hip_height) ** 2
            barrier_pitch = torch.relu(torch.abs(torso_angle_world - torso_angle_ref) - 0.20) ** 2
            
            speed_loss = (vel_x - 1.2) ** 2
            action_loss = torch.mean((tau_applied / 300.0) ** 2, dim=-1)
            
            raw_loss = (
                4.0 * pose_loss +
                0.2 * vel_loss +
                15.0 * pelvis_pitch_loss +
                25.0 * upright_loss +
                2.0 * torso_vel_loss +
                10.0 * height_loss +
                80.0 * barrier_height +
                50.0 * barrier_pitch +
                2.0 * speed_loss +
                0.01 * action_loss
            )
            
            step_loss = torch.clamp(raw_loss, max=40.0)
            num_active = torch.clamp(active_mask.sum(), min=1.0)
            active_step_loss = torch.where(active_mask, step_loss, torch.zeros_like(step_loss))
            step_loss_batch = torch.sum(active_step_loss) / num_active
            
            window_loss = window_loss + step_loss_batch
            total_loss = total_loss + step_loss_batch.item()
            total_pose_err += torch.nan_to_num(torch.sum(torch.where(active_mask, pose_loss, torch.zeros_like(pose_loss))) / num_active, nan=0.0).item()
            total_pelvis_err += torch.nan_to_num(torch.sum(torch.where(active_mask, pelvis_pitch_loss, torch.zeros_like(pelvis_pitch_loss))) / num_active, nan=0.0).item()
            total_upright_err += torch.nan_to_num(torch.sum(torch.where(active_mask, upright_loss, torch.zeros_like(upright_loss))) / num_active, nan=0.0).item()
            
            q_curr = torch.nan_to_num(q_next, nan=0.0)
            u_curr = torch.clamp(torch.nan_to_num(u_next, nan=0.0), -30.0, 30.0)
            if actuator_state_next is not None:
                actuator_state = torch.nan_to_num(actuator_state_next, nan=0.01)
                
            # Truncated BPTT Backward
            if (t + 1) % tbptt_window == 0 or (t + 1) == curr_rollout:
                if isinstance(window_loss, torch.Tensor) and window_loss.requires_grad and window_loss > 0 and not torch.isnan(window_loss):
                    optimizer.zero_grad()
                    window_loss.backward()
                    has_nan_grad = any(p.grad is not None and torch.isnan(p.grad).any() for p in policy.parameters())
                    if not has_nan_grad:
                        torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
                        optimizer.step()
                window_loss = 0.0
                q_curr = q_curr.detach()
                u_curr = u_curr.detach()
                if actuator_state is not None:
                    actuator_state = actuator_state.detach()
                    
        scheduler.step()
        
        # Log Progress
        mean_survival = survival_steps.mean().item()
        loss_val = total_loss.item() if isinstance(total_loss, torch.Tensor) else total_loss
        dt_epoch = time.time() - t0
        
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"  {epoch+1:6d}  {loss_val:10.4f}  {total_pose_err/(t+1):10.4f}  {total_pelvis_err/(t+1):12.4f}  {total_upright_err/(t+1):10.4f}  {mean_survival:10.1f}  {dt_epoch:6.2f}s", flush=True)
            
        if (epoch + 1) % 50 == 0 or (epoch + 1) == num_epochs:
            checkpoint = {
                "policy": policy.state_dict(),
                "actuator_type": actuator_type,
                "epoch": epoch + 1
            }
            torch.save(checkpoint, os.path.join(save_dir, f"agent_{epoch+1}.pt"))
            torch.save(checkpoint, os.path.join(save_dir, "agent_final.pt"))
            
    total_time = time.time() - start_time
    print("=" * 80)
    print(f"Training [{actuator_type}] Finished in {total_time/60:.2f} minutes!")
    print(f"Checkpoints saved to {save_dir}/agent_final.pt")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="10-DoF Differentiable Physics Biped Trainer")
    parser.add_argument("--actuator", type=str, default="ideal_unlimited",
                        choices=["ideal_unlimited", "ideal_limited", "mtg"],
                        help="Actuator model: ideal_unlimited (Tier 1), ideal_limited (Tier 2), or mtg (Tier 3)")
    parser.add_argument("--epochs", type=int, default=300, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size (parallel environments)")
    parser.add_argument("--rollout_steps", type=int, default=60, help="Rollout steps (horizon)")
    
    args = parser.parse_args()
    train(
        actuator_type=args.actuator,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        rollout_steps=args.rollout_steps
    )
