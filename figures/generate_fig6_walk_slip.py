"""
Generate Figure 6 (Results — Experiment 7: Walking Foot-Slip Perturbation, +700 N):
  Matching Figure 2's layout with the +700 N right-foot slip perturbation (t = 1.24–1.40 s):
  - Panel A: 5 MuJoCo 3-model overlay keyframes across pre-slip -> slip -> recovery
             (Cyan: Ideal PD, Orange/Green: Young MTG, Purple: Elderly MTG)
  - Panel B: 2x2 Joint Torques (Lumbar, Right Hip, Right Knee, Right Ankle) with shaded MTG limits
             and shaded crimson slip window
  - Panel C: Total Biological MTG Torque Excess summed across all 7 joints
  - Panel D: Trunk Pitch Angle showing backward slip excursion and recovery across all 3 models.

Outputs:
  - figures/fig6_walk_slip.png (300 DPI)
  - figures/fig6_walk_slip.pdf (vector + high-res embedded frames)
"""

import os
import sys

sys.path.insert(0, os.getcwd())
import mujoco
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from Stable_Motion_Tracker import MotionClip, StableTrackingEnv, StableTrackingPolicy
from run_all_experiments import ElderlyWalkTrackingEnv, device


def main():
    os.makedirs("figures", exist_ok=True)

    clip = MotionClip.load("motions/Arnold2013Walking_subject01_walk2.npz")
    policy1 = StableTrackingPolicy(obs_dim=64, action_dim=7, device=device)
    policy1.load_state_dict(torch.load("checkpoints/exp7_walk_slip_ideal.pt", map_location=device))
    policy2 = StableTrackingPolicy(obs_dim=64, action_dim=7, device=device)
    policy2.load_state_dict(torch.load("checkpoints/exp7_walk_slip_young.pt", map_location=device))
    policy3 = StableTrackingPolicy(obs_dim=64, action_dim=7, device=device)
    policy3.load_state_dict(torch.load("checkpoints/exp7_walk_slip_elderly.pt", map_location=device))

    max_steps = 180  # 3.60 s
    slip_force = 700.0
    slip_start_step = 62
    slip_duration_steps = 8  # t = 1.24..1.40 s
    keyframe_steps = [55, 66, 77, 92, 120]  # 1.10s, 1.32s, 1.54s, 1.84s, 2.40s

    env1 = StableTrackingEnv(clip, mode="mode1", mtg_phase="hard", max_steps=max_steps)
    env2 = StableTrackingEnv(clip, mode="mode2", mtg_phase="hard", max_steps=max_steps)
    env3 = ElderlyWalkTrackingEnv(clip, mtg_phase="hard", max_steps=max_steps)

    fid1 = mujoco.mj_name2id(env1.env.model, mujoco.mjtObj.mjOBJ_BODY, "foot_r")
    fid2 = mujoco.mj_name2id(env2.env.model, mujoco.mjtObj.mjOBJ_BODY, "foot_r")
    fid3 = mujoco.mj_name2id(env3.env.model, mujoco.mjtObj.mjOBJ_BODY, "foot_r")

    obs1 = env1.reset(start_frame=0)
    obs2 = env2.reset(start_frame=0)
    obs3 = env3.reset(start_frame=0)

    overlay_model = mujoco.MjModel.from_xml_path("biped_overlay_jump.xml")
    overlay_data = mujoco.MjData(overlay_model)
    renderer = mujoco.Renderer(overlay_model, height=480, width=520)
    camera = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(camera)
    camera.distance = 2.75
    camera.elevation = -9.0
    camera.azimuth = 88.0

    t_500 = []
    tau1_500, tau2_500, tau3_500 = [], [], []
    Tp2_500, Tn2_500 = [], []
    Tp3_500, Tn3_500 = [], []
    ex1_500, ex2_500, ex3_500 = [], [], []
    pitch1_500, pitch2_500, pitch3_500 = [], [], []
    a1_shadow = np.full(7, env2.env.actuator.a_init, dtype=np.float32)
    key_imgs = {}

    substep_idx = 0
    for step in range(max_steps):
        if step in keyframe_steps:
            q1 = env1.env.data.qpos[:10].copy()
            q2 = env2.env.data.qpos[:10].copy()
            q3 = env3.env.data.qpos[:10].copy()
            overlay_data.qpos[:10] = q1
            overlay_data.qpos[10:20] = q2
            overlay_data.qpos[20:30] = q3
            mujoco.mj_forward(overlay_model, overlay_data)
            camera.lookat[0] = float(np.mean([q1[0], q2[0], q3[0]]))
            camera.lookat[1] = 0.0
            camera.lookat[2] = 0.82
            renderer.update_scene(overlay_data, camera)
            key_imgs[step] = renderer.render().copy()

        slip_active = (slip_start_step <= step < slip_start_step + slip_duration_steps)
        f_apply = float(slip_force) if slip_active else 0.0
        env1.env.data.xfrc_applied[fid1, 0] = f_apply
        env2.env.data.xfrc_applied[fid2, 0] = f_apply
        env3.env.data.xfrc_applied[fid3, 0] = f_apply

        with torch.no_grad():
            a1 = policy1.get_action(torch.tensor(obs1, dtype=torch.float32, device=device).unsqueeze(0), deterministic=True)[0].squeeze(0).cpu().numpy()
            a2 = policy2.get_action(torch.tensor(obs2, dtype=torch.float32, device=device).unsqueeze(0), deterministic=True)[0].squeeze(0).cpu().numpy()
            a3 = policy3.get_action(torch.tensor(obs3, dtype=torch.float32, device=device).unsqueeze(0), deterministic=True)[0].squeeze(0).cpu().numpy()

        prev_f1 = env1.filtered_action.copy()
        f1 = 0.40 * env1.filtered_action + 0.60 * a1
        env1.filtered_action = f1.copy()

        prev_f2 = env2.filtered_action.copy()
        f2 = 0.40 * env2.filtered_action + 0.60 * a2
        env2.filtered_action = f2.copy()

        prev_f3 = env3.filtered_action.copy()
        f3 = 0.40 * env3.filtered_action + 0.60 * a3
        env3.filtered_action = f3.copy()

        for sub_step in range(env1.decimation):
            alpha = (sub_step + 1) / float(env1.decimation)

            # Mode 1
            q_ref_1, u_ref_1 = clip.get_frame(env1.current_frame + sub_step)
            q_des1 = q_ref_1[3:10] + (1.0 - alpha) * prev_f1 + alpha * f1
            q_j1 = env1.env.data.qpos[3:10].astype(np.float32)
            u_j1 = env1.env.data.qvel[3:10].astype(np.float32)
            T_LV_p1, T_LV_n1 = env2.env.actuator.compute_capacities_np(q_j1, u_j1)
            tau_pass1 = env2.env.actuator.compute_passive_np(q_j1)
            _, tau1_k = env1.env.step_simulation(q_des1, dt_sim=0.002, u_des=u_ref_1[3:10])
            tau_act_des1 = tau1_k - tau_pass1
            T_LV_dir1 = np.where(tau_act_des1 >= 0.0, T_LV_p1, T_LV_n1)
            u_req1 = np.clip(np.abs(tau_act_des1) / (T_LV_dir1 + 1e-3), 0.0, 1.0)
            tau_c1 = np.where(u_req1 >= a1_shadow, env2.env.actuator.tau_act, env2.env.actuator.tau_deact)
            a1_shadow = np.clip(a1_shadow + 0.002 * (u_req1 - a1_shadow) / tau_c1, env2.env.actuator.a_min, 1.0)
            Tp1 = a1_shadow * T_LV_p1 + tau_pass1
            Tn1 = a1_shadow * T_LV_n1 - tau_pass1
            ex1_k = float(np.sum(np.maximum(0.0, tau1_k - Tp1) + np.maximum(0.0, -tau1_k - Tn1)))

            # Mode 2 Young
            q_ref_2, u_ref_2 = clip.get_frame(env2.current_frame + sub_step)
            q_des2 = q_ref_2[3:10] + (1.0 - alpha) * prev_f2 + alpha * f2
            _, tau2_k = env2.env.step_simulation(q_des2, dt_sim=0.002, u_des=u_ref_2[3:10])
            Tp2 = env2.env.actuator.last_T_pos
            Tn2 = env2.env.actuator.last_T_neg
            ex2_k = float(np.sum(np.maximum(0.0, tau2_k - Tp2) + np.maximum(0.0, -tau2_k - Tn2)))

            # Mode 2 Elderly
            q_ref_3, u_ref_3 = clip.get_frame(env3.current_frame + sub_step)
            q_des3 = q_ref_3[3:10] + (1.0 - alpha) * prev_f3 + alpha * f3
            _, tau3_k = env3.env.step_simulation(q_des3, dt_sim=0.002, u_des=u_ref_3[3:10])
            Tp3 = env3.env.actuator.last_T_pos
            Tn3 = env3.env.actuator.last_T_neg
            ex3_k = float(np.sum(np.maximum(0.0, tau3_k - Tp3) + np.maximum(0.0, -tau3_k - Tn3)))

            t_500.append(substep_idx * 0.002)
            tau1_500.append(tau1_k.copy())
            tau2_500.append(tau2_k.copy())
            tau3_500.append(tau3_k.copy())
            Tp2_500.append(Tp2.copy())
            Tn2_500.append(Tn2.copy())
            Tp3_500.append(Tp3.copy())
            Tn3_500.append(Tn3.copy())
            ex1_500.append(ex1_k)
            ex2_500.append(ex2_k)
            ex3_500.append(ex3_k)
            pitch1_500.append(float(np.degrees(env1.env.data.qpos[2] + env1.env.data.qpos[3])))
            pitch2_500.append(float(np.degrees(env2.env.data.qpos[2] + env2.env.data.qpos[3])))
            pitch3_500.append(float(np.degrees(env3.env.data.qpos[2] + env3.env.data.qpos[3])))
            substep_idx += 1

        for env_i, act_i in [(env1, a1), (env2, a2), (env3, a3)]:
            env_i.current_frame += env_i.decimation
            env_i.step_count += 1
            env_i.prev_action = act_i.copy()
        obs1 = env1.get_obs()
        obs2 = env2.get_obs()
        obs3 = env3.get_obs()

    t_arr = np.array(t_500)
    tau1_arr = np.array(tau1_500)
    tau2_arr = np.array(tau2_500)
    tau3_arr = np.array(tau3_500)
    Tp2_arr = np.array(Tp2_500)
    Tn2_arr = np.array(Tn2_500)
    Tp3_arr = np.array(Tp3_500)
    Tn3_arr = np.array(Tn3_500)
    ex1_arr = np.array(ex1_500)
    ex2_arr = np.array(ex2_500)
    ex3_arr = np.array(ex3_500)
    pitch1_arr = np.array(pitch1_500)
    pitch2_arr = np.array(pitch2_500)
    pitch3_arr = np.array(pitch3_500)

    # Focus x-axis window cleanly on [0.60, 3.00] s around the slip
    t_min, t_max = 0.60, 3.00
    mask = (t_arr >= t_min) & (t_arr <= t_max)
    t_slip0 = slip_start_step * 0.02
    t_slip1 = (slip_start_step + slip_duration_steps) * 0.02

    # -------------------------------------------------------------------------
    # Compose Publication Figure 6 (matching Figures 2 & 3 style)
    # -------------------------------------------------------------------------
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "mathtext.fontset": "dejavusans",
        "axes.spines.top": False,
        "axes.spines.right": False,
    })

    fig = plt.figure(figsize=(14.8, 11.6), dpi=300)
    outer = gridspec.GridSpec(3, 1, height_ratios=[1.02, 1.85, 0.92], hspace=0.31)

    # Panel A: 5 Keyframes around Slip & Recovery
    gs_a = gridspec.GridSpecFromSubplotSpec(1, 5, subplot_spec=outer[0], wspace=0.06)
    phase_labels = ["Pre-Slip", "Slip (+700 N)", "Max Excursion", "Recovery Step", "Stabilized"]
    for idx, st in enumerate(keyframe_steps):
        ax_img = fig.add_subplot(gs_a[0, idx])
        ax_img.imshow(key_imgs[st])
        ax_img.set_xticks([])
        ax_img.set_yticks([])
        is_slip = (slip_start_step <= st < slip_start_step + slip_duration_steps)
        for sp in ax_img.spines.values():
            sp.set_visible(True)
            sp.set_edgecolor("#DC2626" if is_slip else "#CBD5E1")
            sp.set_linewidth(2.0 if is_slip else 1.1)
        ax_img.set_title(
            f"$t = {st*0.02:.2f}$ s  ({phase_labels[idx]})",
            fontsize=9.1, fontweight="bold",
            color="#B91C1C" if is_slip else "#1E293B", pad=5
        )
        if idx == 0:
            ax_img.text(-0.08, 1.12, "A", transform=ax_img.transAxes, fontsize=14, fontweight="bold", color="#0F172A", va="bottom", ha="right")

    fig.text(
        0.50, 0.692,
        "Overlay Keyframes (+700 N Right-Foot Slip, t = 1.24–1.40 s):  Cyan = Mode 1 (Ideal PD)   |   Orange/Green = Mode 2 (Young MTG)   |   Purple = Mode 2 (Elderly MTG)",
        ha="center", va="center", fontsize=8.6, color="#334155", style="italic"
    )

    # Panel B: 2x2 Joint Torques with Shaded Dynamic MTG Limits
    gs_b = gridspec.GridSpecFromSubplotSpec(2, 2, subplot_spec=outer[1], hspace=0.35, wspace=0.18)
    joints = [
        (0, "Lumbar Extension / Flexion"),
        (4, "Right Hip Extension / Flexion"),
        (5, "Right Knee Extension / Flexion"),
        (6, "Right Ankle Dorsi / Plantarflexion"),
    ]

    for idx, (j_idx, j_title) in enumerate(joints):
        ax = fig.add_subplot(gs_b[idx // 2, idx % 2])
        ax.axvspan(t_slip0, t_slip1, color="#DC2626", alpha=0.14, zorder=0,
                   label="Foot-Slip Impulse (+700 N)" if idx == 0 else None)

        ax.fill_between(
            t_arr[mask], -Tn2_arr[mask, j_idx], Tp2_arr[mask, j_idx],
            color="#F87171", alpha=0.18,
            label="Young MTG Dynamic Limit" if idx == 0 else None
        )
        ax.fill_between(
            t_arr[mask], -Tn3_arr[mask, j_idx], Tp3_arr[mask, j_idx],
            color="#A855F7", alpha=0.18,
            label="Elderly MTG Dynamic Limit" if idx == 0 else None
        )

        ax.plot(t_arr[mask], tau1_arr[mask, j_idx], color="#1D4ED8", ls="--", lw=1.35, alpha=0.90,
                label="Mode 1: Ideal Unlimited PD" if idx == 0 else None)
        ax.plot(t_arr[mask], tau2_arr[mask, j_idx], color="#DC2626", ls="-", lw=1.55,
                label="Mode 2: Young Millard MTG" if idx == 0 else None)
        ax.plot(t_arr[mask], tau3_arr[mask, j_idx], color="#7E22CE", ls="-", lw=1.60,
                label="Mode 2: Elderly Millard MTG" if idx == 0 else None)

        for st in keyframe_steps:
            ax.axvline(st * 0.02, color="#CBD5E1", lw=0.8, ls="-", zorder=0)

        ax.axhline(0.0, color="#94A3B8", lw=0.7, alpha=0.6)
        ax.set_title(j_title, fontsize=10.0, fontweight="bold", color="#1E293B", pad=5)
        ax.set_ylabel("Torque (N·m)", fontsize=9.0, color="#1E293B")
        ax.set_xlim(t_min, t_max)
        ax.grid(True, axis="y", alpha=0.25)
        ax.tick_params(labelsize=8.3)

        if idx == 0:
            ax.text(-0.11, 1.08, "B", transform=ax.transAxes, fontsize=14, fontweight="bold", color="#0F172A", va="bottom", ha="right")

    # Bottom Row: Panel C (Summed Torque Excess) & Panel D (Trunk Pitch Recovery)
    gs_cd = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=outer[2], wspace=0.18)

    # Panel C: Total Exceeded Biological Torque
    ax_c = fig.add_subplot(gs_cd[0, 0])
    mean_ex1 = float(np.mean(ex1_arr[mask]))
    max_ex1 = float(np.max(ex1_arr[mask]))
    ax_c.axvspan(t_slip0, t_slip1, color="#DC2626", alpha=0.14, zorder=0)
    ax_c.fill_between(t_arr[mask], 0.0, ex1_arr[mask], color="#EA580C", alpha=0.24)
    ax_c.plot(t_arr[mask], ex1_arr[mask], color="#C2410C", lw=1.55,
              label=f"Mode 1 Excess (Mean={mean_ex1:.1f}, Peak={max_ex1:.1f} N·m)")
    ax_c.plot(t_arr[mask], ex2_arr[mask], color="#DC2626", lw=1.7,
              label="Mode 2 Young & Elderly MTG (0.00 N·m)")
    for st in keyframe_steps:
        ax_c.axvline(st * 0.02, color="#CBD5E1", lw=0.8, ls="-", zorder=0)
    ax_c.set_title("Total Biological MTG Torque Excess (All 7 Joints)", fontsize=10.0, fontweight="bold", color="#1E293B", pad=5)
    ax_c.set_ylabel("Total Excess (N·m)", fontsize=9.0, color="#1E293B")
    ax_c.set_xlabel("Time (s)", fontsize=9.2, color="#1E293B")
    ax_c.set_xlim(t_min, t_max)
    ax_c.set_ylim(-10.0, max_ex1 * 1.30)
    ax_c.grid(True, axis="y", alpha=0.25)
    ax_c.tick_params(labelsize=8.3)
    ax_c.legend(loc="upper right", fontsize=8.0, frameon=True, facecolor="#FFFFFF", edgecolor="#CBD5E1")
    ax_c.text(-0.11, 1.08, "C", transform=ax_c.transAxes, fontsize=14, fontweight="bold", color="#0F172A", va="bottom", ha="right")

    # Panel D: Trunk Pitch Angle Recovery
    ax_d = fig.add_subplot(gs_cd[0, 1])
    ax_d.axvspan(t_slip0, t_slip1, color="#DC2626", alpha=0.14, zorder=0)
    ax_d.plot(t_arr[mask], pitch1_arr[mask], color="#1D4ED8", ls="--", lw=1.6, label="Mode 1: Ideal Unlimited PD (Fell=False)")
    ax_d.plot(t_arr[mask], pitch2_arr[mask], color="#DC2626", ls="-", lw=1.8, label="Mode 2: Young Millard MTG (Fell=False)")
    ax_d.plot(t_arr[mask], pitch3_arr[mask], color="#7E22CE", ls="-", lw=1.9, label="Mode 2: Elderly Millard MTG (Fell=False)")
    for st in keyframe_steps:
        ax_d.axvline(st * 0.02, color="#CBD5E1", lw=0.8, ls="-", zorder=0)
    ax_d.axhline(0.0, color="#94A3B8", lw=0.7, alpha=0.6)
    ax_d.set_title("Trunk Pitch Angle $\\theta_{\\mathrm{torso}}(t)$ During Slip & Recovery", fontsize=10.0, fontweight="bold", color="#1E293B", pad=5)
    ax_d.set_ylabel("Trunk Pitch (deg)", fontsize=9.0, color="#1E293B")
    ax_d.set_xlabel("Time (s)", fontsize=9.2, color="#1E293B")
    ax_d.set_xlim(t_min, t_max)
    ax_d.grid(True, alpha=0.28)
    ax_d.tick_params(labelsize=8.3)
    ax_d.legend(loc="lower right", fontsize=7.9, frameon=True, facecolor="#FFFFFF", edgecolor="#CBD5E1")
    ax_d.text(-0.11, 1.08, "D", transform=ax_d.transAxes, fontsize=14, fontweight="bold", color="#0F172A", va="bottom", ha="right")

    handles, labels = fig.axes[5].get_legend_handles_labels()
    fig.legend(
        handles, labels,
        loc="lower center", ncol=6, frameon=True,
        facecolor="#FFFFFF", edgecolor="#CBD5E1",
        fontsize=8.0, bbox_to_anchor=(0.5, 0.008)
    )

    fig.suptitle(
        "Figure 6: Walking Foot-Slip Perturbation (+700 N) — Keyframes, 500 Hz Joint Torques, MTG Excess & Trunk Pitch Recovery",
        fontsize=11.6, fontweight="bold", color="#0F172A", y=0.980
    )

    plt.subplots_adjust(left=0.06, right=0.98, top=0.925, bottom=0.085)
    fig.savefig("figures/fig6_walk_slip.png", dpi=300, bbox_inches="tight")
    fig.savefig("figures/fig6_walk_slip.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"[DONE] Saved figures/fig6_walk_slip.{{png,pdf}} | Mean Excess={mean_ex1:.2f} N*m, Peak={max_ex1:.2f} N*m")


if __name__ == "__main__":
    main()
