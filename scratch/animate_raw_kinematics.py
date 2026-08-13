import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

local_path = "subject01_walk1_ik.mot"

with open(local_path, "r") as f:
    lines = f.readlines()

header_end = 0
for i, line in enumerate(lines):
    if "endheader" in line:
        header_end = i + 1
        break

data_lines = [l.strip().split() for l in lines[header_end:] if l.strip() and not l.startswith('time')]
data = np.array(data_lines, dtype=np.float32)

times = data[:, 0]
idx_start = np.where(np.abs(times - 0.40) < 1e-3)[0][0]
idx_end = np.where(np.abs(times - 1.60) < 1e-3)[0][0]
cropped_data = data[idx_start:idx_end+1]

# Retarget pelvis height down by 12.2cm
pelvis_tilt = np.radians(cropped_data[:, 1])
pelvis_tx = cropped_data[:, 4] - cropped_data[0, 4]  # start at x=0
pelvis_ty = cropped_data[:, 5] - 0.1220

hip_flex_r = np.radians(cropped_data[:, 7])
knee_ang_r = np.radians(cropped_data[:, 10])
ankle_ang_r = np.radians(cropped_data[:, 11])

hip_flex_l = np.radians(cropped_data[:, 14])
knee_ang_l = np.radians(cropped_data[:, 17])
ankle_ang_l = np.radians(cropped_data[:, 18])

# Segment lengths
l_torso = 0.8
l_thigh = 0.45
l_leg = 0.45
l_heel = 0.05
l_toe = 0.2

num_frames = len(cropped_data)

fig, ax = plt.subplots(figsize=(6, 6))
ax.set_xlim(-0.8, 1.2)
ax.set_ylim(-0.2, 1.8)
ax.set_aspect('equal')
ax.grid(True)
ax.axhline(0, color='black', lw=2, label='Ground (Y=0)')
ax.set_title("Retargeted MOT Reference Motion")
ax.set_xlabel("X (m)")
ax.set_ylabel("Y (m)")

line_torso, = ax.plot([], [], 'k-', lw=5, label='Torso')
line_left_leg, = ax.plot([], [], 'r-o', lw=3, ms=8, label='Left Leg')
line_right_leg, = ax.plot([], [], 'b-o', lw=3, ms=8, label='Right Leg')
line_left_foot, = ax.plot([], [], 'r-', lw=3)
line_right_foot, = ax.plot([], [], 'b-', lw=3)
ax.legend(loc='upper right')

def animate(i):
    x_hip = pelvis_tx[i]
    y_hip = pelvis_ty[i]
    q_tr = pelvis_tilt[i]
    
    # Torso top
    x_torso = x_hip - l_torso * np.sin(q_tr)
    y_torso = y_hip + l_torso * np.cos(q_tr)
    
    # Left Leg
    theta_lh = q_tr + hip_flex_l[i]
    x_knee_l = x_hip + l_thigh * np.sin(theta_lh)
    y_knee_l = y_hip - l_thigh * np.cos(theta_lh)
    
    theta_lk = theta_lh + knee_ang_l[i]
    x_ankle_l = x_knee_l + l_leg * np.sin(theta_lk)
    y_ankle_l = y_knee_l - l_leg * np.cos(theta_lk)
    
    theta_la = theta_lk + ankle_ang_l[i]
    x_heel_l = x_ankle_l - l_heel * np.cos(theta_la)
    y_heel_l = y_ankle_l - l_heel * np.sin(theta_la)
    x_toe_l = x_ankle_l + l_toe * np.cos(theta_la)
    y_toe_l = y_ankle_l + l_toe * np.sin(theta_la)
    
    # Right Leg
    theta_rh = q_tr + hip_flex_r[i]
    x_knee_r = x_hip + l_thigh * np.sin(theta_rh)
    y_knee_r = y_hip - l_thigh * np.cos(theta_rh)
    
    theta_rk = theta_rh + knee_ang_r[i]
    x_ankle_r = x_knee_r + l_leg * np.sin(theta_rk)
    y_ankle_r = y_knee_r - l_leg * np.cos(theta_rk)
    
    theta_ra = theta_rk + ankle_ang_r[i]
    x_heel_r = x_ankle_r - l_heel * np.cos(theta_ra)
    y_heel_r = y_ankle_r - l_heel * np.sin(theta_ra)
    x_toe_r = x_ankle_r + l_toe * np.cos(theta_ra)
    y_toe_r = y_ankle_r + l_toe * np.sin(theta_ra)
    
    line_torso.set_data([x_hip, x_torso], [y_hip, y_torso])
    line_left_leg.set_data([x_hip, x_knee_l, x_ankle_l], [y_hip, y_knee_l, y_ankle_l])
    line_right_leg.set_data([x_hip, x_knee_r, x_ankle_r], [y_hip, y_knee_r, y_ankle_r])
    line_left_foot.set_data([x_heel_l, x_toe_l], [y_heel_l, y_toe_l])
    line_right_foot.set_data([x_heel_r, x_toe_r], [y_heel_r, y_toe_r])
    
    ax.set_xlim(x_hip - 1.0, x_hip + 1.0)
    return line_torso, line_left_leg, line_right_leg, line_left_foot, line_right_foot

ani = animation.FuncAnimation(fig, animate, frames=num_frames, interval=50, blit=True)

os.makedirs("scratch", exist_ok=True)
gif_path = "scratch/retargeted_kinematics.gif"
print(f"Saving retargeted motion animation to {gif_path}...")
ani.save(gif_path, writer='pillow', fps=20)
print("Animation saved successfully!")
