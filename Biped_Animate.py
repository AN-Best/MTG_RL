import os
import glob
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from Biped_Actuators import create_actuator
from Biped_Dynamics import get_default_body_parameters, simulate_macro_step
from Biped_Train_DiffPhys import DifferentiablePolicy, load_reference_motion

# Define device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device for Animation & Diagnostics: {device}")

def animate_biped(actuator_type="ideal_unlimited", checkpoint_path=None, output_gif=None, num_steps=300):
    _, q_ref_data, u_ref_data = load_reference_motion(num_samples=10000, device=device)
    
    # 1. Initialize Actuator & Body Parameters
    actuator = create_actuator(actuator_type=actuator_type, device=device)
    params = get_default_body_parameters(batch_size=1, device=device)
    
    # 2. Initialize Policy & Load Checkpoint
    policy = DifferentiablePolicy(obs_dim=19, action_dim=7).to(device)
    
    if checkpoint_path is None:
        default_dir = f"runs/biped_diffphys_{actuator_type}/checkpoints"
        pts = glob.glob(os.path.join(default_dir, "*.pt"))
        if pts:
            checkpoint_path = max(pts, key=os.path.getmtime)
        else:
            checkpoint_path = os.path.join(default_dir, "agent_final.pt")
            
    print(f"Loading checkpoint from: {checkpoint_path}")
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        if isinstance(checkpoint, dict) and "policy" in checkpoint:
            policy.load_state_dict(checkpoint["policy"])
        else:
            policy.load_state_dict(checkpoint)
        print("Successfully loaded trained policy!")
    else:
        print(f"Warning: Checkpoint not found at {checkpoint_path}. Running with nominal unperturbed baseline!")
        
    policy.eval()
    
    # 3. Initialize at Calibrated Stride Start
    q = q_ref_data[0:1].clone().detach()
    u = u_ref_data[0:1].clone().detach()
    actuator_state = torch.full((1, 7), 0.01, device=device) if actuator_type == "mtg" else None
    
    print(f"\n--- Running evaluation rollout ({actuator_type}) for {num_steps} steps ({(num_steps*0.02):.1f}s) ---")
    q_history = [q[0].cpu().numpy().copy()]
    
    for step in range(num_steps):
        obs = torch.cat([q[:, 1:], u], dim=-1)
        with torch.no_grad():
            delta_q = policy(obs)
            
        phase_idx = step % q_ref_data.shape[0]
        q_target = q_ref_data[phase_idx:phase_idx+1, 3:]
        q_des = q_target + delta_q
        
        q_next, u_next, actuator_state_next, _ = simulate_macro_step(
            q, u, actuator_state, q_des, actuator, params,
            dt_sim=0.002, dt_control=0.02
        )
        q, u, actuator_state = q_next, u_next, actuator_state_next
        q_history.append(q[0].detach().cpu().numpy().copy())
        
        pelvis_angle = q[0, 2].item()
        lumbar_angle = q[0, 3].item()
        torso_angle = pelvis_angle + lumbar_angle
        hip_height = q[0, 1].item()
        vel_x = u[0, 0].item()
        
        if hip_height < 0.50 or abs(torso_angle) > 1.05 or abs(vel_x) > 10.0 or torch.isnan(q).any():
            print(f"Rollout ended at step {step} due to balance termination condition.")
            break
            
    q_history = np.array(q_history)
    print(f"Recorded {len(q_history)} steps of 10-DoF joint coordinate trajectories.")
    
    # 4. Render 2D Biomechanical Stick-Figure Animation
    l_pelvis = 0.10
    l_torso = 0.4725
    l_thigh = 0.45
    l_leg = 0.45
    l_heel = 0.07
    l_toe = 0.17

    def forward_kinematics(q_row):
        qx, qy, q_p, q_lum, q_lh, q_lk, q_la, q_rh, q_rk, q_ra = q_row
        
        r_pelvis = np.array([qx, qy])
        r_torso = r_pelvis + np.array([-l_torso * np.sin(q_p + q_lum), l_torso * np.cos(q_p + q_lum)])
        
        # Left leg
        r_hip_l = r_pelvis
        r_knee_l = r_hip_l + np.array([l_thigh * np.sin(q_p + q_lh), -l_thigh * np.cos(q_p + q_lh)])
        r_ankle_l = r_knee_l + np.array([l_leg * np.sin(q_p + q_lh + q_lk), -l_leg * np.cos(q_p + q_lh + q_lk)])
        phi_foot_l = q_p + q_lh + q_lk + q_la
        r_heel_l = r_ankle_l + np.array([-l_heel * np.cos(phi_foot_l), -l_heel * np.sin(phi_foot_l)])
        r_toe_l = r_ankle_l + np.array([l_toe * np.cos(phi_foot_l), l_toe * np.sin(phi_foot_l)])
        
        # Right leg
        r_hip_r = r_pelvis
        r_knee_r = r_hip_r + np.array([l_thigh * np.sin(q_p + q_rh), -l_thigh * np.cos(q_p + q_rh)])
        r_ankle_r = r_knee_r + np.array([l_leg * np.sin(q_p + q_rh + q_rk), -l_leg * np.cos(q_p + q_rh + q_rk)])
        phi_foot_r = q_p + q_rh + q_rk + q_ra
        r_heel_r = r_ankle_r + np.array([-l_heel * np.cos(phi_foot_r), -l_heel * np.sin(phi_foot_r)])
        r_toe_r = r_ankle_r + np.array([l_toe * np.cos(phi_foot_r), l_toe * np.sin(phi_foot_r)])
        
        return {
            "torso": (r_pelvis, r_torso),
            "leg_l": (r_hip_l, r_knee_l, r_ankle_l, r_heel_l, r_toe_l),
            "leg_r": (r_hip_r, r_knee_r, r_ankle_r, r_heel_r, r_toe_r)
        }

    fig, ax = plt.subplots(figsize=(10, 6), dpi=100)
    ax.set_aspect("equal")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.set_title(f"10-DoF Musculoskeletal Biped Gait  |  Actuator: {actuator_type.upper()}", fontsize=13, fontweight="bold")
    ax.set_xlabel("Horizontal Distance (m)")
    ax.set_ylabel("Height (m)")
    
    ground_line = ax.axhline(0.0, color="black", lw=2)
    line_torso, = ax.plot([], [], "o-", color="#1f77b4", lw=5, label="Torso / Pelvis")
    line_leg_l, = ax.plot([], [], "o-", color="#ff7f0e", lw=3.5, label="Left Leg (Swing/Stance)")
    line_foot_l, = ax.plot([], [], "o-", color="#d62728", lw=3.5)
    line_leg_r, = ax.plot([], [], "o-", color="#2ca02c", lw=3.5, label="Right Leg (Swing/Stance)")
    line_foot_r, = ax.plot([], [], "o-", color="#9467bd", lw=3.5)
    
    ax.legend(loc="upper right")
    
    def init():
        return line_torso, line_leg_l, line_foot_l, line_leg_r, line_foot_r

    def update(frame):
        fk = forward_kinematics(q_history[frame])
        
        # Torso
        p1, p2 = fk["torso"]
        line_torso.set_data([p1[0], p2[0]], [p1[1], p2[1]])
        
        # Left Leg
        hip_l, knee_l, ankle_l, heel_l, toe_l = fk["leg_l"]
        line_leg_l.set_data([hip_l[0], knee_l[0], ankle_l[0]], [hip_l[1], knee_l[1], ankle_l[1]])
        line_foot_l.set_data([heel_l[0], ankle_l[0], toe_l[0]], [heel_l[1], ankle_l[1], toe_l[1]])
        
        # Right Leg
        hip_r, knee_r, ankle_r, heel_r, toe_r = fk["leg_r"]
        line_leg_r.set_data([hip_r[0], knee_r[0], ankle_r[0]], [hip_r[1], knee_r[1], ankle_r[1]])
        line_foot_r.set_data([heel_r[0], ankle_r[0], toe_r[0]], [heel_r[1], ankle_r[1], toe_r[1]])
        
        # Camera tracking horizontal COM
        com_x = q_history[frame, 0]
        ax.set_xlim(com_x - 1.2, com_x + 1.8)
        ax.set_ylim(-0.1, 1.8)
        
        return line_torso, line_leg_l, line_foot_l, line_leg_r, line_foot_r

    if output_gif is None:
        output_gif = f"biped_{actuator_type}_gait.gif"
        
    print(f"Saving animation to {output_gif}...")
    ani = animation.FuncAnimation(fig, update, frames=len(q_history), init_func=init, blit=False, interval=20)
    ani.save(output_gif, writer="pillow", fps=50)
    print(f"Animation saved successfully to {output_gif}!")
    plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Animate 10-DoF Biped Gait")
    parser.add_argument("--actuator", type=str, default="ideal_unlimited",
                        choices=["ideal_unlimited", "ideal_limited", "mtg"],
                        help="Actuator model: ideal_unlimited, ideal_limited, or mtg")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to policy checkpoint")
    parser.add_argument("--output", type=str, default=None, help="Output gif file path")
    parser.add_argument("--steps", type=int, default=300, help="Number of steps to evaluate")
    
    args = parser.parse_args()
    animate_biped(
        actuator_type=args.actuator,
        checkpoint_path=args.checkpoint,
        output_gif=args.output,
        num_steps=args.steps
    )
