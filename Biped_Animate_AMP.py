# pyrefly: ignore [missing-import]
import os
import glob
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from Biped_Gym_Env import BipedAMPEnv
from skrl.envs.wrappers.torch import wrap_env
from Biped_Train_AMP import Policy

# Define device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Initialize environment
print("Initializing environment...")
env = BipedAMPEnv(num_envs=1, device=device)
env = wrap_env(env)

# Initialize policy
print("Loading trained policy...")
policy = Policy(env.observation_space, env.action_space, device)
policy.to(device)

# Load the saved state dict from agent checkpoint
try:
    checkpoint = torch.load("amp_biped_policy.pt", map_location=device)
    if isinstance(checkpoint, dict) and "policy" in checkpoint:
        policy.load_state_dict(checkpoint["policy"])
    else:
        policy.load_state_dict(checkpoint)
    print("Successfully loaded trained policy from amp_biped_policy.pt!")
except Exception as e:
    print(f"Could not load amp_biped_policy.pt directly: {e}")
    print("Attempting to load from checkpoints folder...")
    
    checkpoint_path = "runs/biped_amp/checkpoints/best_agent.pt"
    if not os.path.exists(checkpoint_path):
        pts = glob.glob("runs/biped_amp/checkpoints/*.pt")
        if pts:
            checkpoint_path = max(pts, key=os.path.getmtime)
            print(f"best_agent.pt not found. Falling back to the latest checkpoint: {checkpoint_path}")
        else:
            raise FileNotFoundError("No checkpoint files found in runs/biped_amp/checkpoints/")
            
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if isinstance(checkpoint, dict) and "policy" in checkpoint:
        policy.load_state_dict(checkpoint["policy"])
    else:
        policy.load_state_dict(checkpoint)
    print(f"Successfully loaded trained policy from {checkpoint_path}!")

policy.eval()

# Reset environment and run rollout to record trajectory
print("\n--- Running evaluation rollout to record positions ---")
q_history = []
obs, info = env.reset()
q_history.append(env.unwrapped.q[0].cpu().numpy().copy() if env.unwrapped.q.ndim > 1 else env.unwrapped.q.cpu().numpy().copy())

for step in range(200):  # Record 200 control cycles (20 seconds)
    with torch.no_grad():
        inputs = {"states": obs}
        actions, _ = policy.act(inputs, role="policy")
        
    obs, reward, terminated, truncated, info = env.step(actions)
    q_history.append(env.unwrapped.q[0].cpu().numpy().copy() if env.unwrapped.q.ndim > 1 else env.unwrapped.q.cpu().numpy().copy())
    
    if terminated[0] or truncated[0]:
        print(f"Rollout ended at step {step} due to termination/truncation.")
        break

q_history = np.array(q_history)
print(f"Recorded {len(q_history)} steps of joint coordinate trajectories.")

# Biped kinematic segment parameters
l_torso = 0.8
l_thigh = 0.45
l_leg = 0.45
l_heel = 0.05
l_toe = 0.2
h_g = 0.90

# Set up matplotlib figure and axes
fig, ax = plt.subplots(figsize=(6, 6))
ax.set_xlim(-1.0, 1.0)
ax.set_ylim(-0.2, 1.8)
ax.set_aspect('equal')
ax.grid(True)
ax.axhline(0, color='black', lw=2)  # Ground line
ax.set_title("Biped AMP Walk Animation")
ax.set_xlabel("X (m)")
ax.set_ylabel("Y (m)")

# Create drawing elements
line_torso, = ax.plot([], [], 'k-', lw=5, label='Torso')
line_left_leg, = ax.plot([], [], 'r-o', lw=3, ms=8, label='Left Leg')
line_right_leg, = ax.plot([], [], 'b-o', lw=3, ms=8, label='Right Leg')
line_left_foot, = ax.plot([], [], 'r-', lw=3)
line_right_foot, = ax.plot([], [], 'b-', lw=3)

ax.legend()

# Animation update function
def animate(i):
    q = q_history[i]
    q_x, q_y, q_tr, q_lh, q_lk, q_la, q_rh, q_rk, q_ra = q
    
    # HJC (hip joint center) is dynamic
    x_hip, y_hip = q_x, q_y
    
    # Torso top position
    x_torso = x_hip - l_torso * np.sin(q_tr)
    y_torso = y_hip + l_torso * np.cos(q_tr)
    
    # Left Knee
    theta_lh = q_tr + q_lh
    x_knee_l = x_hip + l_thigh * np.sin(theta_lh)
    y_knee_l = y_hip - l_thigh * np.cos(theta_lh)
    
    # Left Ankle
    theta_lk = theta_lh + q_lk
    x_ankle_l = x_knee_l + l_leg * np.sin(theta_lk)
    y_ankle_l = y_knee_l - l_leg * np.cos(theta_lk)
    
    # Left Foot
    theta_la = theta_lk + q_la
    x_heel_l = x_ankle_l - l_heel * np.cos(theta_la)
    y_heel_l = y_ankle_l - l_heel * np.sin(theta_la)
    x_toe_l = x_ankle_l + l_toe * np.cos(theta_la)
    y_toe_l = y_ankle_l + l_toe * np.sin(theta_la)
    
    # Right Knee
    theta_rh = q_tr + q_rh
    x_knee_r = x_hip + l_thigh * np.sin(theta_rh)
    y_knee_r = y_hip - l_thigh * np.cos(theta_rh)
    
    # Right Ankle
    theta_rk = theta_rh + q_rk
    x_ankle_r = x_knee_r + l_leg * np.sin(theta_rk)
    y_ankle_r = y_knee_r - l_leg * np.cos(theta_rk)
    
    # Right Foot
    theta_ra = theta_rk + q_ra
    x_heel_r = x_ankle_r - l_heel * np.cos(theta_ra)
    y_heel_r = y_ankle_r - l_heel * np.sin(theta_ra)
    x_toe_r = x_ankle_r + l_toe * np.cos(theta_ra)
    y_toe_r = y_ankle_r + l_toe * np.sin(theta_ra)
    
    # Update line coordinates
    line_torso.set_data([x_hip, x_torso], [y_hip, y_torso])
    line_left_leg.set_data([x_hip, x_knee_l, x_ankle_l], [y_hip, y_knee_l, y_ankle_l])
    line_right_leg.set_data([x_hip, x_knee_r, x_ankle_r], [y_hip, y_knee_r, y_ankle_r])
    line_left_foot.set_data([x_heel_l, x_toe_l], [y_heel_l, y_toe_l])
    line_right_foot.set_data([x_heel_r, x_toe_r], [y_heel_r, y_toe_r])
    
    # Track hip joint center horizontally
    ax.set_xlim(x_hip - 1.0, x_hip + 1.0)
    
    return line_torso, line_left_leg, line_right_leg, line_left_foot, line_right_foot

# Create animation
ani = animation.FuncAnimation(fig, animate, frames=len(q_history), interval=100, blit=True)

gif_path = "biped_gait.gif"
print(f"\nSaving walking animation to {gif_path} (this may take a few seconds)...")
ani.save(gif_path, writer='pillow', fps=10)
print("Animation saved successfully!")
