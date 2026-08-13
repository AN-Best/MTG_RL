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

# Column mapping from OpenSim Gait2392 header:
# 0: time
# 1: pelvis_tilt
# 4: pelvis_tx
# 5: pelvis_ty
# 7: hip_flexion_r, 10: knee_angle_r, 11: ankle_angle_r
# 14: hip_flexion_l, 17: knee_angle_l, 18: ankle_angle_l

pelvis_tilt = np.radians(cropped_data[:, 1])
pelvis_tx = cropped_data[:, 4]
pelvis_ty = cropped_data[:, 5]

hip_flex_r = np.radians(cropped_data[:, 7])
knee_ang_r = np.radians(cropped_data[:, 10])
ankle_ang_r = np.radians(cropped_data[:, 11])

hip_flex_l = np.radians(cropped_data[:, 14])
knee_ang_l = np.radians(cropped_data[:, 17])
ankle_ang_l = np.radians(cropped_data[:, 18])

# Forward Kinematics segment parameters
l_thigh = 0.45
l_leg = 0.45
l_heel = 0.05
l_toe = 0.2

# Left Leg Forward Kinematics
theta_lh = pelvis_tilt + hip_flex_l
y_knee_l = pelvis_ty - l_thigh * np.cos(theta_lh)

theta_lk = theta_lh + knee_ang_l
y_ankle_l = y_knee_l - l_leg * np.cos(theta_lk)

theta_la = theta_lk + ankle_ang_l
y_heel_l = y_ankle_l - l_heel * np.sin(theta_la)
y_toe_l = y_ankle_l + l_toe * np.sin(theta_la)

# Right Leg Forward Kinematics
theta_rh = pelvis_tilt + hip_flex_r
y_knee_r = pelvis_ty - l_thigh * np.cos(theta_rh)

theta_rk = theta_rh + knee_ang_r
y_ankle_r = y_knee_r - l_leg * np.cos(theta_rk)

theta_ra = theta_rk + ankle_ang_r
y_heel_r = y_ankle_r - l_heel * np.sin(theta_ra)
y_toe_r = y_ankle_r + l_toe * np.sin(theta_ra)

min_l = np.minimum(y_heel_l, y_toe_l)
min_r = np.minimum(y_heel_r, y_toe_r)
overall_min_foot_y = np.minimum(min_l, min_r)

print("=== RAW MOT KINEMATIC FOOT HEIGHT ANALYSIS ===")
print(f"Min Foot Y (Left):  {np.min(min_l):.4f} m ({np.min(min_l)*100:.2f} cm)")
print(f"Min Foot Y (Right): {np.min(min_r):.4f} m ({np.min(min_r)*100:.2f} cm)")
print(f"Overall Minimum Foot Y: {np.min(overall_min_foot_y):.4f} m ({np.min(overall_min_foot_y)*100:.2f} cm)")
print(f"Max Penetration below Y=0: {abs(min(0.0, np.min(overall_min_foot_y)))*100:.2f} cm")
print(f"Pelvis Height (Y) Range: {np.min(pelvis_ty):.4f} m to {np.max(pelvis_ty):.4f} m")

# Plotting
t_rel = cropped_data[:, 0] - cropped_data[0, 0]
plt.figure(figsize=(10, 6))
plt.plot(t_rel, y_heel_l, 'r--', label='Left Heel Y')
plt.plot(t_rel, y_toe_l, 'r-', label='Left Toe Y')
plt.plot(t_rel, y_heel_r, 'b--', label='Right Heel Y')
plt.plot(t_rel, y_toe_r, 'b-', label='Right Toe Y')
plt.axhline(0.0, color='black', linewidth=2, label='Ground Surface (Y=0)')
plt.fill_between(t_rel, overall_min_foot_y, 0, where=(overall_min_foot_y < 0), color='red', alpha=0.3, label='Ground Penetration Zone')

plt.title('Raw Kinematics from subject01_walk1_ik.mot (Foot Heights vs Ground Y=0)')
plt.xlabel('Gait Cycle Time (seconds)')
plt.ylabel('Vertical Height (meters)')
plt.grid(True)
plt.legend(loc='upper right')
plt.tight_layout()

os.makedirs("scratch", exist_ok=True)
out_fig = "scratch/raw_mot_foot_kinematics.png"
plt.savefig(out_fig, dpi=150)
print(f"\nPlot saved to {out_fig}")
