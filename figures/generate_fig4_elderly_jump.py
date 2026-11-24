"""
Generate Figure 4 (Results — Elderly MTG Personalization & Maximum Vertical Jump):
  - Panel A: 5 MuJoCo keyframe snapshots of the 3-model side-by-side jump
             (Left/Cyan: Ideal Unlimited PD, Center/Orange-Green: Young MTG, Right/Purple: Elderly MTG)
  - Panel B: 3 Elderly MTG physiological scaling curves (no passive torque lines):
             Knee Extension Torque-Angle, Ankle & Hip Torque-Angle, Knee Torque-Velocity
  - Panel C: Isometric strength scaling bar chart (left) & 500 Hz Pelvis Height z(t) trajectory (right)
             (no jump-height bar chart).

Outputs:
  - figures/fig4_elderly_jump.png (300 DPI)
  - figures/fig4_elderly_jump.pdf (vector + high-res embedded frames)
"""

import os
import sys

sys.path.insert(0, os.getcwd())
import mujoco
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from run_all_experiments import MaxJumpEnv, device, make_jump_policy


def main():
    os.makedirs("figures", exist_ok=True)

    policy1 = make_jump_policy(obs_dim=64, device=device)
    policy1.load_state_dict(torch.load("checkpoints/exp4_jump_mode1.pt", map_location=device))
    policy2 = make_jump_policy(obs_dim=64, device=device)
    policy2.load_state_dict(torch.load("checkpoints/exp4_jump_mode2.pt", map_location=device))
    policy3 = make_jump_policy(obs_dim=64, device=device)
    policy3.load_state_dict(torch.load("checkpoints/exp4_jump_mode2_elderly.pt", map_location=device))

    max_steps = 60
    keyframe_steps = [0, 12, 20, 28, 35]  # 0.00s, 0.24s, 0.40s, 0.56s, 0.70s

    env1 = MaxJumpEnv(mode="mode1", max_steps=max_steps)
    env2 = MaxJumpEnv(mode="mode2", mtg_phase="hard", max_steps=max_steps)
    env3 = MaxJumpEnv(mode="mode2_elderly", mtg_phase="hard", max_steps=max_steps)

    obs1 = env1.reset(start_frame=0)
    obs2 = env2.reset(start_frame=0)
    obs3 = env3.reset(start_frame=0)

    overlay_model = mujoco.MjModel.from_xml_path("biped_overlay_jump.xml")
    overlay_data = mujoco.MjData(overlay_model)
    renderer = mujoco.Renderer(overlay_model, height=460, width=540)
    camera = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(camera)
    camera.distance = 3.25
    camera.elevation = -8.0
    camera.azimuth = 90.0

    t_500, z1_500, z2_500, z3_500 = [], [], [], []
    key_imgs = {}

    done1 = done2 = done3 = False
    q1_apex_pose = q2_apex_pose = q3_apex_pose = None
    substep_idx = 0

    for step in range(max_steps):
        if step in keyframe_steps:
            q1_vis = (q1_apex_pose if q1_apex_pose is not None else env1.env.data.qpos[:10]).copy()
            q2_vis = (q2_apex_pose if q2_apex_pose is not None else env2.env.data.qpos[:10]).copy()
            q3_vis = (q3_apex_pose if q3_apex_pose is not None else env3.env.data.qpos[:10]).copy()
            q1_vis[0] -= 0.65
            q3_vis[0] += 0.65
            overlay_data.qpos[:10] = q1_vis
            overlay_data.qpos[10:20] = q2_vis
            overlay_data.qpos[20:30] = q3_vis
            mujoco.mj_forward(overlay_model, overlay_data)
            camera.lookat[:] = [0.0, 0.0, 0.96]
            renderer.update_scene(overlay_data, camera)
            key_imgs[step] = renderer.render().copy()

        with torch.no_grad():
            a1 = policy1.get_action(torch.tensor(obs1, dtype=torch.float32, device=device).unsqueeze(0), deterministic=True)[0].squeeze(0).cpu().numpy()
            a2 = policy2.get_action(torch.tensor(obs2, dtype=torch.float32, device=device).unsqueeze(0), deterministic=True)[0].squeeze(0).cpu().numpy()
            a3 = policy3.get_action(torch.tensor(obs3, dtype=torch.float32, device=device).unsqueeze(0), deterministic=True)[0].squeeze(0).cpu().numpy()

        prev_f1 = env1.filtered_action.copy()
        f1 = 0.25 * env1.filtered_action + 0.75 * a1
        if not done1:
            env1.filtered_action = f1.copy()

        prev_f2 = env2.filtered_action.copy()
        f2 = 0.25 * env2.filtered_action + 0.75 * a2
        if not done2:
            env2.filtered_action = f2.copy()

        prev_f3 = env3.filtered_action.copy()
        f3 = 0.25 * env3.filtered_action + 0.75 * a3
        if not done3:
            env3.filtered_action = f3.copy()

        for sub_step in range(env1.decimation):
            alpha = (sub_step + 1) / float(env1.decimation)

            if not done1:
                act_k1 = (1.0 - alpha) * prev_f1 + alpha * f1
                q_j1 = env1.env.data.qpos[3:10].astype(np.float32)
                q_des1 = env1._compute_safe_q_des(env1.q_nom + act_k1, q_j1)
                env1.env.step_simulation(q_des1, dt_sim=0.002, u_des=None)
                z1_k = float(env1.env.data.qpos[1])
                if z1_k > env1.max_z:
                    env1.max_z = z1_k
            else:
                z1_k = float(env1.max_z)

            if not done2:
                act_k2 = (1.0 - alpha) * prev_f2 + alpha * f2
                q_j2 = env2.env.data.qpos[3:10].astype(np.float32)
                q_des2 = env2._compute_safe_q_des(env2.q_nom + act_k2, q_j2)
                env2.env.step_simulation(q_des2, dt_sim=0.002, u_des=None)
                z2_k = float(env2.env.data.qpos[1])
                if z2_k > env2.max_z:
                    env2.max_z = z2_k
            else:
                z2_k = float(env2.max_z)

            if not done3:
                act_k3 = (1.0 - alpha) * prev_f3 + alpha * f3
                q_j3 = env3.env.data.qpos[3:10].astype(np.float32)
                q_des3 = env3._compute_safe_q_des(env3.q_nom + act_k3, q_j3)
                env3.env.step_simulation(q_des3, dt_sim=0.002, u_des=None)
                z3_k = float(env3.env.data.qpos[1])
                if z3_k > env3.max_z:
                    env3.max_z = z3_k
            else:
                z3_k = float(env3.max_z)

            t_500.append(substep_idx * 0.002)
            z1_500.append(z1_k)
            z2_500.append(z2_k)
            z3_500.append(z3_k)
            substep_idx += 1

        for env_i, act_i, done_flag_idx in [(env1, a1, 1), (env2, a2, 2), (env3, a3, 3)]:
            if (done_flag_idx == 1 and done1) or (done_flag_idx == 2 and done2) or (done_flag_idx == 3 and done3):
                continue
            env_i.current_frame += env_i.decimation
            env_i.step_count += 1
            env_i.prev_action = act_i.copy()
            fl, fr, _ = env_i._get_contacts_and_foot_z()
            if fl < 1.0 and fr < 1.0 and env_i.env.data.qpos[1] > env_i.z_stand + 0.02:
                env_i.has_taken_off = True
            if env_i.has_taken_off and env_i.env.data.qvel[1] <= 0.0:
                env_i.apex_reached = True
                env_i.post_apex_steps += 1
            q_i = env_i.env.data.qpos
            is_fell = (q_i[1] < 0.62 or abs(q_i[2]) > 0.55 or abs(q_i[0]) > 0.75)
            is_apex_done = (env_i.apex_reached and env_i.post_apex_steps >= 2)
            if done_flag_idx == 1:
                obs1 = env1.get_obs()
                if is_fell or is_apex_done:
                    if is_apex_done:
                        q1_apex_pose = q_i[:10].copy()
                    done1 = True
            elif done_flag_idx == 2:
                obs2 = env2.get_obs()
                if is_fell or is_apex_done:
                    if is_apex_done:
                        q2_apex_pose = q_i[:10].copy()
                    done2 = True
            else:
                obs3 = env3.get_obs()
                if is_fell or is_apex_done:
                    if is_apex_done:
                        q3_apex_pose = q_i[:10].copy()
                    done3 = True

        if done1 and done2 and done3 and step >= max(keyframe_steps):
            break

    # Capture any remaining keyframe if all done early
    for st in keyframe_steps:
        if st not in key_imgs:
            q1_vis = (q1_apex_pose if q1_apex_pose is not None else env1.env.data.qpos[:10]).copy()
            q2_vis = (q2_apex_pose if q2_apex_pose is not None else env2.env.data.qpos[:10]).copy()
            q3_vis = (q3_apex_pose if q3_apex_pose is not None else env3.env.data.qpos[:10]).copy()
            q1_vis[0] -= 0.65
            q3_vis[0] += 0.65
            overlay_data.qpos[:10] = q1_vis
            overlay_data.qpos[10:20] = q2_vis
            overlay_data.qpos[20:30] = q3_vis
            mujoco.mj_forward(overlay_model, overlay_data)
            camera.lookat[:] = [0.0, 0.0, 0.96]
            renderer.update_scene(overlay_data, camera)
            key_imgs[st] = renderer.render().copy()

    t_arr = np.array(t_500)
    act_y = env2.env.actuator
    act_e = env3.env.actuator

    # -------------------------------------------------------------------------
    # Compose Publication Figure 4
    # -------------------------------------------------------------------------
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "mathtext.fontset": "dejavusans",
        "axes.spines.top": False,
        "axes.spines.right": False,
    })

    fig = plt.figure(figsize=(15.0, 11.4), dpi=300)
    outer = gridspec.GridSpec(3, 1, height_ratios=[1.02, 1.08, 1.08], hspace=0.36)

    # Panel A: 5 Jump Keyframes (3-model side-by-side)
    gs_a = gridspec.GridSpecFromSubplotSpec(1, 5, subplot_spec=outer[0], wspace=0.06)
    phase_labels = ["Quiet Standing", "Countermovement", "Takeoff", "MTG Apex", "Ideal Apex"]
    for idx, st in enumerate(keyframe_steps):
        ax_img = fig.add_subplot(gs_a[0, idx])
        ax_img.imshow(key_imgs[st])
        ax_img.set_xticks([])
        ax_img.set_yticks([])
        for sp in ax_img.spines.values():
            sp.set_visible(True)
            sp.set_edgecolor("#CBD5E1")
            sp.set_linewidth(1.1)
        ax_img.set_title(f"$t = {st*0.02:.2f}$ s  ({phase_labels[idx]})", fontsize=9.2, fontweight="bold", color="#1E293B", pad=5)
        if idx == 0:
            ax_img.text(-0.08, 1.12, "A", transform=ax_img.transAxes, fontsize=14, fontweight="bold", color="#0F172A", va="bottom", ha="right")

    fig.text(
        0.50, 0.678,
        "Side-by-Side Vertical Jump Keyframes:  Left (Cyan) = Mode 1 (Ideal Unlimited PD)   |   Center (Orange/Green) = Mode 2 (Young MTG)   |   Right (Purple) = Mode 2 (Elderly MTG)",
        ha="center", va="center", fontsize=8.7, color="#334155", style="italic"
    )

    # Panel B: 1x2 Knee Extension Elderly MTG Scaling Curves (Torque-Angle & Torque-Velocity)
    gs_b = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=outer[1], wspace=0.18)

    # B1: Knee Extension Torque-Angle
    ax_b1 = fig.add_subplot(gs_b[0, 0])
    q_knee = np.linspace(-1.75, 0.0, 200)
    fL_k_y = act_y.T_max_pos_np[2] * np.exp(-((q_knee - act_y.q_opt_np[2]) / act_y.w_np[2]) ** 2)
    fL_k_e = act_e.T_max_pos_np[2] * np.exp(-((q_knee - act_e.q_opt_np[2]) / act_e.w_np[2]) ** 2)
    ax_b1.plot(np.degrees(q_knee), fL_k_y, color="#DC2626", lw=2.2, label=f"Young Knee Ext ($T_{{\\max}}={act_y.T_max_pos_np[2]:.0f}$ N·m)")
    ax_b1.plot(np.degrees(q_knee), fL_k_e, color="#7E22CE", lw=2.2, label=f"Elderly Knee Ext ($s^s=0.659,\\;{act_e.T_max_pos_np[2]:.0f}$ N·m)")
    ax_b1.fill_between(np.degrees(q_knee), fL_k_e, fL_k_y, color="#C084FC", alpha=0.16)
    ax_b1.set_title("Knee Extension Torque–Angle ($-34.1\\%$ Isometric Capacity)", fontsize=9.8, fontweight="bold", color="#1E293B", pad=5)
    ax_b1.set_xlabel("Knee Angle (deg)", fontsize=9.0)
    ax_b1.set_ylabel("Isometric Torque Capacity (N·m)", fontsize=9.0)
    ax_b1.set_ylim(120, 560)
    ax_b1.grid(True, alpha=0.28)
    ax_b1.legend(loc="lower center", fontsize=8.2, frameon=True, facecolor="#FFFFFF", edgecolor="#CBD5E1")
    ax_b1.tick_params(labelsize=8.3)
    ax_b1.text(-0.11, 1.08, "B", transform=ax_b1.transAxes, fontsize=14, fontweight="bold", color="#0F172A", va="bottom", ha="right")

    # B2: Knee Extension Torque-Velocity
    ax_b2 = fig.add_subplot(gs_b[0, 1])
    u_grid = np.linspace(-15.0, 18.0, 250)
    nu_y = np.clip(u_grid / act_y.u_max_pos_np[2], -1.0, 1.0)
    fV_y = np.where(nu_y >= 0.0, np.clip((1.0 - nu_y) / (1.0 + 4.0 * nu_y + 1e-6), 0.0, 1.0), np.clip(1.0 - 1.4 * nu_y / (1.0 - 4.0 * nu_y + 1e-6), 1.0, 1.6))
    nu_e = np.clip(u_grid / act_e.u_max_pos_np[2], -1.0, 1.0)
    fV_e = np.where(nu_e >= 0.0, np.clip((1.0 - nu_e) / (1.0 + 4.0 * nu_e + 1e-6), 0.0, 1.0), np.clip(1.0 - 1.4 * nu_e / (1.0 - 4.0 * nu_e + 1e-6), 1.0, 1.6))
    ax_b2.plot(u_grid, act_y.T_max_pos_np[2] * fV_y, color="#DC2626", lw=2.2, label=f"Young Knee Ext ($u_{{\\max}}={act_y.u_max_pos_np[2]:.1f}$ rad/s)")
    ax_b2.plot(u_grid, act_e.T_max_pos_np[2] * fV_e, color="#7E22CE", lw=2.2, label=f"Elderly Knee Ext ($s^V=0.88,\\;{act_e.u_max_pos_np[2]:.1f}$ rad/s)")
    ax_b2.fill_between(u_grid, act_e.T_max_pos_np[2] * fV_e, act_y.T_max_pos_np[2] * fV_y, color="#C084FC", alpha=0.16)
    ax_b2.axvline(0.0, color="#64748B", ls=":", lw=1.0)
    ax_b2.set_title("Knee Extension Torque–Velocity (Type-II Fiber Atrophy)", fontsize=9.8, fontweight="bold", color="#1E293B", pad=5)
    ax_b2.set_xlabel("Joint Angular Velocity $\\dot{q}$ (rad/s, +Concentric)", fontsize=9.0)
    ax_b2.set_ylabel("Dynamic Torque $T_{\\max}\\,f_V(\\dot{q})$ (N·m)", fontsize=9.0)
    ax_b2.grid(True, alpha=0.28)
    ax_b2.legend(loc="upper right", fontsize=8.2, frameon=True, facecolor="#FFFFFF", edgecolor="#CBD5E1")
    ax_b2.tick_params(labelsize=8.3)

    # Panel C: 1x2 — Isometric Scaling Bar Chart (Left) & Vertical Jump Trajectory z(t) (Right)
    gs_c = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=outer[2], wspace=0.18)

    # C1: Isometric Strength Scaling Bar Chart
    ax_c1 = fig.add_subplot(gs_c[0, 0])
    j_labels = ["Lumbar\nExt", "Hip\nExt", "Knee\nExt", "Ankle\nPlantar", "Hip\nFlex", "Knee\nFlex"]
    y_vals = [act_y.T_max_pos_np[0], act_y.T_max_pos_np[1], act_y.T_max_pos_np[2], act_y.T_max_neg_np[3], act_y.T_max_neg_np[1], act_y.T_max_neg_np[2]]
    e_vals = [act_e.T_max_pos_np[0], act_e.T_max_pos_np[1], act_e.T_max_pos_np[2], act_e.T_max_neg_np[3], act_e.T_max_neg_np[1], act_e.T_max_neg_np[2]]
    x_pos = np.arange(len(j_labels))
    w_bar = 0.36
    b1 = ax_c1.bar(x_pos - w_bar / 2, y_vals, w_bar, color="#EF4444", alpha=0.88, edgecolor="#991B1B", lw=0.7, label="Young Adult MTG (20–29 yrs)")
    b2 = ax_c1.bar(x_pos + w_bar / 2, e_vals, w_bar, color="#7E22CE", alpha=0.88, edgecolor="#581C87", lw=0.7, label="Elderly Personalized MTG (70–86 yrs)")
    for b_y, b_e in zip(b1, b2):
        pct = (b_e.get_height() - b_y.get_height()) / b_y.get_height() * 100.0
        ax_c1.text(b_e.get_x() + b_e.get_width() / 2, b_e.get_height() + 9, f"{pct:.0f}%", ha="center", va="bottom", fontsize=8.3, fontweight="bold", color="#581C87")
    ax_c1.set_xticks(x_pos)
    ax_c1.set_xticklabels(j_labels, fontsize=8.8)
    ax_c1.set_ylim(0, 565)
    ax_c1.set_ylabel("Peak Isometric Torque $T_{\\max}$ (N·m)", fontsize=9.0)
    ax_c1.set_title("Isometric Strength Scaling ($s^s$) Across Lower-Limb & Trunk Joints", fontsize=9.8, fontweight="bold", color="#1E293B", pad=5)
    ax_c1.grid(True, axis="y", alpha=0.28)
    ax_c1.legend(loc="upper right", fontsize=8.2, frameon=True, facecolor="#FFFFFF", edgecolor="#CBD5E1")
    ax_c1.tick_params(labelsize=8.3)
    ax_c1.text(-0.11, 1.08, "C", transform=ax_c1.transAxes, fontsize=14, fontweight="bold", color="#0F172A", va="bottom", ha="right")

    # C2: 500 Hz Vertical Jump Pelvis Height Trajectory z(t)
    ax_c2 = fig.add_subplot(gs_c[0, 1])
    dz1_cm = (env1.max_z - env1.z_stand) * 100.0
    dz2_cm = (env2.max_z - env2.z_stand) * 100.0
    dz3_cm = (env3.max_z - env3.z_stand) * 100.0
    ax_c2.plot(t_arr, z1_500, color="#1D4ED8", ls="--", lw=1.9, label=f"Mode 1: Ideal Unlimited PD (Peak {env1.max_z:.2f} m, +{dz1_cm:.1f} cm)")
    ax_c2.plot(t_arr, z2_500, color="#DC2626", ls="-", lw=2.2, label=f"Mode 2: Young Adult MTG (Peak {env2.max_z:.2f} m, +{dz2_cm:.1f} cm)")
    ax_c2.plot(t_arr, z3_500, color="#7E22CE", ls="-", lw=2.2, label=f"Mode 2: Elderly Personalized MTG (Peak {env3.max_z:.2f} m, +{dz3_cm:.1f} cm)")
    ax_c2.axhline(env1.z_stand, color="#15803D", ls=":", lw=1.4, label=f"Quiet Standing ({env1.z_stand:.2f} m)")
    for st in keyframe_steps:
        ax_c2.axvline(st * 0.02, color="#CBD5E1", lw=0.8, ls="-", zorder=0)
    ax_c2.set_title("500 Hz Vertical Jump Pelvis Trajectory $z(t)$ (Squat $\\rightarrow$ Takeoff $\\rightarrow$ Apex)", fontsize=9.8, fontweight="bold", color="#1E293B", pad=5)
    ax_c2.set_xlabel("Time (s)", fontsize=9.0)
    ax_c2.set_ylabel("Pelvis Height $z$ (m)", fontsize=9.0)
    ax_c2.set_xlim(0.0, t_arr[-1])
    ax_c2.set_ylim(0.60, 1.38)
    ax_c2.grid(True, alpha=0.28)
    ax_c2.legend(loc="upper left", fontsize=8.0, frameon=True, facecolor="#FFFFFF", edgecolor="#CBD5E1")
    ax_c2.tick_params(labelsize=8.3)
    ax_c2.text(-0.11, 1.08, "D", transform=ax_c2.transAxes, fontsize=14, fontweight="bold", color="#0F172A", va="bottom", ha="right")

    fig.suptitle(
        "Figure 4: Elderly MTG Personalization & Maximum Vertical Jump — Keyframes, Scaling Curves & 500 Hz Pelvis Rise",
        fontsize=12.0, fontweight="bold", color="#0F172A", y=0.980
    )

    plt.subplots_adjust(left=0.06, right=0.98, top=0.925, bottom=0.055)
    fig.savefig("figures/fig4_elderly_jump.png", dpi=300, bbox_inches="tight")
    fig.savefig("figures/fig4_elderly_jump.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"[DONE] Saved figures/fig4_elderly_jump.{{png,pdf}} | Jump dz: Ideal=+{dz1_cm:.1f}cm, Young=+{dz2_cm:.1f}cm, Elderly=+{dz3_cm:.1f}cm")


if __name__ == "__main__":
    main()
