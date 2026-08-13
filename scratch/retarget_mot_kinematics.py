import os
import numpy as np
import matplotlib.pyplot as plt

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

pelvis_tilt = np.radians(cropped_data[:, 1])
pelvis_tx = cropped_data[:, 4]
pelvis_ty = cropped_data[:, 5]

hip_flex_r = np.radians(cropped_data[:, 7])
knee_ang_r = np.radians(cropped_data[:, 10])
ankle_ang_r = np.radians(cropped_data[:, 11])

hip_flex_l = np.radians(cropped_data[:, 14])
knee_ang_l = np.radians(cropped_data[:, 17])
ankle_ang_l = np.radians(cropped_data[:, 18])

# Segment parameters
l_thigh = 0.45
l_leg = 0.45
l_heel = 0.05
l_toe = 0.2

# Function to compute lowest foot Y given a pelvis_ty offset
def get_foot_y(ty_offset):
    ty = pelvis_ty - ty_offset
    
    # Left Leg
    theta_lh = pelvis_tilt + hip_flex_l
    y_knee_l = ty - l_thigh * np.cos(theta_lh)
    theta_lk = theta_lh + knee_ang_l
    y_ankle_l = y_knee_l - l_leg * np.cos(theta_lk)
    theta_la = theta_lk + ankle_ang_l
    y_heel_l = y_ankle_l - l_heel * np.sin(theta_la)
    y_toe_l = y_ankle_l + l_toe * np.sin(theta_la)

    # Right Leg
    theta_rh = pelvis_tilt + hip_flex_r
    y_knee_r = ty - l_thigh * np.cos(theta_rh)
    theta_rk = theta_rh + knee_ang_r
    y_ankle_r = y_knee_r - l_leg * np.cos(theta_rk)
    theta_ra = theta_rk + ankle_ang_r
    y_heel_r = y_ankle_r - l_heel * np.sin(theta_ra)
    y_toe_r = y_ankle_r + l_toe * np.sin(theta_ra)

    min_l = np.minimum(y_heel_l, y_toe_l)
    min_r = np.minimum(y_heel_r, y_toe_r)
    return y_heel_l, y_toe_l, y_heel_r, y_toe_r, np.minimum(min_l, min_r)

# Find optimal offset so lowest point during stance penetrates ground by ~0.003m (3mm) for stable contact
raw_min_y = np.min(get_foot_y(0.0)[-1])
target_offset = raw_min_y + 0.003  # Allow 3mm initial spring engagement

yh_l, yt_l, yh_r, yt_r, min_y_retargeted = get_foot_y(target_offset)

print("=== RETARGETED MOT KINEMATICS ANALYSIS ===")
print(f"Optimal Pelvis Vertical Offset: -{target_offset:.4f} m (-{target_offset*100:.2f} cm)")
print(f"Retargeted Minimum Foot Y:     {np.min(min_y_retargeted):.4f} m ({np.min(min_y_retargeted)*100:.2f} cm)")
print(f"Retargeted Pelvis Y Range:     {np.min(pelvis_ty - target_offset):.4f} m to {np.max(pelvis_ty - target_offset):.4f} m")

# Plot comparison
t_rel = cropped_data[:, 0] - cropped_data[0, 0]
plt.figure(figsize=(10, 6))
plt.plot(t_rel, yh_l, 'r--', label='Retargeted Left Heel Y')
plt.plot(t_rel, yt_l, 'r-', label='Retargeted Left Toe Y')
plt.plot(t_rel, yh_r, 'b--', label='Retargeted Right Heel Y')
plt.plot(t_rel, yt_r, 'b-', label='Retargeted Right Toe Y')
plt.axhline(0.0, color='black', linewidth=2, label='Ground Surface (Y=0)')
plt.fill_between(t_rel, min_y_retargeted, 0, where=(min_y_retargeted < 0), color='green', alpha=0.3, label='Contact Spring Engagement (0 to 3mm)')

plt.title('Retargeted Reference Motion Kinematics (Ground-Aligned)')
plt.xlabel('Gait Cycle Time (seconds)')
plt.ylabel('Vertical Height (meters)')
plt.grid(True)
plt.legend(loc='upper right')
plt.tight_layout()

out_fig = "scratch/retargeted_mot_foot_kinematics.png"
plt.savefig(out_fig, dpi=150)
print(f"Plot saved to {out_fig}")
