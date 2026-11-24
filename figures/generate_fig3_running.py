"""
Generate Figure 3 (Results — Experiment 3: Fast Running Imitation at 5.0 m/s):
  - Panel A: 5 keyframe snapshots across one full running stride (Mode 1 vs Mode 2 overlay)
  - Panel B: 500 Hz joint torques (Lumbar, Right Hip, Right Knee, Right Ankle) with
             shaded dynamic MTG torque envelopes (T_LV and active+passive bounds).
  - Panel C: Total exceeded biological torque summed across all 7 actuated joints.

Outputs:
  - figures/fig3_running.png (300 DPI)
  - figures/fig3_running.pdf (vector + high-res embedded frames)
"""

import os
import sys

sys.path.insert(0, os.getcwd())
import mujoco
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from run_all_experiments import (
    SprintTrackingEnv,
    device,
    load_calibrated_run50_clip,
    make_sprint_policy,
)


def main():
    os.makedirs("figures", exist_ok=True)

    clip = load_calibrated_run50_clip("motions/Hamner2013_Run50_cycle.npz")
    policy1 = make_sprint_policy(obs_dim=64, device=device, warm_start_path="checkpoints/exp3_run50_mode1.pt")
    policy2 = make_sprint_policy(obs_dim=64, device=device, warm_start_path="checkpoints/exp3_run50_mode2.pt")

    max_steps = 100  # 2.00 s (~3 full 5.0 m/s sprint strides)
    keyframe_steps = [0, 8, 16, 24, 32]  # 0.00s, 0.16s, 0.32s, 0.48s, 0.64s

    env1 = SprintTrackingEnv(clip, mode="mode1", max_steps=max_steps, enable_perturbations=False)
    env2 = SprintTrackingEnv(clip, mode="mode2", mtg_phase="hard", max_steps=max_steps, enable_perturbations=False)
    obs1 = env1.reset(start_frame=0)
    obs2 = env2.reset(start_frame=0)

    overlay_model = mujoco.MjModel.from_xml_path("biped_overlay.xml")
    overlay_data = mujoco.MjData(overlay_model)
    renderer = mujoco.Renderer(overlay_model, height=480, width=520)
    camera = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(camera)
    camera.distance = 2.75
    camera.elevation = -8.0
    camera.azimuth = 88.0

    t_500 = []
    tau1_500 = []
    tau2_500 = []
    Tp2_500, Tn2_500 = [], []
    T_LV_p2_500, T_LV_n2_500 = [], []
    ex1_500, ex2_500 = [], []
    vx1_500, vx2_500, vxref_500 = [], [], []
    a1_shadow = np.full(7, env2.env.actuator.a_init, dtype=np.float32)
    key_imgs = {}

    substep_idx = 0
    for step in range(max_steps):
        if step in keyframe_steps:
            q1 = env1.env.data.qpos[:10].copy()
            q2 = env2.env.data.qpos[:10].copy()
            overlay_data.qpos[:10] = q1
            overlay_data.qpos[10:20] = q2
            mujoco.mj_forward(overlay_model, overlay_data)
            camera.lookat[0] = 0.5 * (q1[0] + q2[0])
            camera.lookat[1] = 0.0
            camera.lookat[2] = 0.82
            renderer.update_scene(overlay_data, camera)
            key_imgs[step] = renderer.render().copy()

        with torch.no_grad():
            a1 = policy1.get_action(torch.tensor(obs1, dtype=torch.float32, device=device).unsqueeze(0), deterministic=True)[0].squeeze(0).cpu().numpy()
            a2 = policy2.get_action(torch.tensor(obs2, dtype=torch.float32, device=device).unsqueeze(0), deterministic=True)[0].squeeze(0).cpu().numpy()

        prev_f1 = env1.filtered_action.copy()
        f1 = 0.40 * env1.filtered_action + 0.60 * a1
        env1.filtered_action = f1.copy()

        prev_f2 = env2.filtered_action.copy()
        f2 = 0.40 * env2.filtered_action + 0.60 * a2
        env2.filtered_action = f2.copy()

        for sub_step in range(env1.decimation):
            alpha = (sub_step + 1) / float(env1.decimation)

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

            q_ref_2, u_ref_2 = clip.get_frame(env2.current_frame + sub_step)
            q_des2 = q_ref_2[3:10] + (1.0 - alpha) * prev_f2 + alpha * f2
            _, tau2_k = env2.env.step_simulation(q_des2, dt_sim=0.002, u_des=u_ref_2[3:10])
            Tp2 = env2.env.actuator.last_T_pos
            Tn2 = env2.env.actuator.last_T_neg
            ex2_k = float(np.sum(np.maximum(0.0, tau2_k - Tp2) + np.maximum(0.0, -tau2_k - Tn2)))

            vx1_k = float(env1.env.data.qvel[0])
            vx2_k = float(env2.env.data.qvel[0])
            vxref_k = float(u_ref_1[0])

            t_500.append(substep_idx * 0.002)
            tau1_500.append(tau1_k.copy())
            tau2_500.append(tau2_k.copy())
            Tp2_500.append(Tp2.copy())
            Tn2_500.append(Tn2.copy())
            T_LV_p2_500.append(env2.env.actuator.last_T_LV_pos.copy())
            T_LV_n2_500.append(env2.env.actuator.last_T_LV_neg.copy())
            ex1_500.append(ex1_k)
            ex2_500.append(ex2_k)
            vx1_500.append(vx1_k)
            vx2_500.append(vx2_k)
            vxref_500.append(vxref_k)
            substep_idx += 1

        env1.current_frame += env1.decimation
        env1.step_count += 1
        env1.prev_action = a1.copy()
        obs1 = env1.get_obs()

        env2.current_frame += env2.decimation
        env2.step_count += 1
        env2.prev_action = a2.copy()
        obs2 = env2.get_obs()

    t_arr = np.array(t_500)
    tau1_arr = np.array(tau1_500)
    tau2_arr = np.array(tau2_500)
    Tp2_arr = np.array(Tp2_500)
    Tn2_arr = np.array(Tn2_500)
    T_LV_p2_arr = np.array(T_LV_p2_500)
    T_LV_n2_arr = np.array(T_LV_n2_500)
    ex1_arr = np.array(ex1_500)
    ex2_arr = np.array(ex2_500)
    vx1_arr = np.array(vx1_500)
    vx2_arr = np.array(vx2_500)
    vxref_arr = np.array(vxref_500)

    # -------------------------------------------------------------------------
    # Compose Publication Figure 3 (matching Figure 2 layout & style)
    # -------------------------------------------------------------------------
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "mathtext.fontset": "dejavusans",
        "axes.spines.top": False,
        "axes.spines.right": False,
    })

    fig = plt.figure(figsize=(14.5, 11.4), dpi=300)
    outer = gridspec.GridSpec(3, 1, height_ratios=[1.05, 1.85, 0.88], hspace=0.30)

    # Panel A: 5 Running Stride Keyframes
    gs_top = gridspec.GridSpecFromSubplotSpec(1, 5, subplot_spec=outer[0], wspace=0.06)
    phase_labels = ["Touchdown", "Mid-Stance", "Flight", "Opp. Stance", "Next Stride"]
    for idx, st in enumerate(keyframe_steps):
        ax_img = fig.add_subplot(gs_top[0, idx])
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
        0.50, 0.690,
        "Overlay Keyframes (5.0 m/s Fast Running):  Blue/Cyan = Mode 1 (Ideal Unlimited PD)   |   Orange/Green = Mode 2 (500 Hz MTG Clamped)",
        ha="center", va="center", fontsize=8.8, color="#334155", style="italic"
    )

    # Panel B: 2x2 Joint Torques with Shaded Dynamic MTG Limits
    gs_bot = gridspec.GridSpecFromSubplotSpec(2, 2, subplot_spec=outer[1], hspace=0.34, wspace=0.18)
    joints = [
        (0, "Lumbar Extension / Flexion"),
        (4, "Right Hip Extension / Flexion"),
        (5, "Right Knee Extension / Flexion"),
        (6, "Right Ankle Dorsi / Plantarflexion"),
    ]

    for idx, (j_idx, j_title) in enumerate(joints):
        ax = fig.add_subplot(gs_bot[idx // 2, idx % 2])

        ax.fill_between(
            t_arr, -T_LV_n2_arr[:, j_idx], T_LV_p2_arr[:, j_idx],
            color="#94A3B8", alpha=0.16,
            label="Full-Activation MTG Envelope ($\\pm T_{\\mathrm{LV}}$)" if idx == 0 else None
        )
        ax.plot(t_arr, T_LV_p2_arr[:, j_idx], color="#64748B", ls=":", lw=0.9, alpha=0.8)
        ax.plot(t_arr, -T_LV_n2_arr[:, j_idx], color="#64748B", ls=":", lw=0.9, alpha=0.8)

        ax.fill_between(
            t_arr, -Tn2_arr[:, j_idx], Tp2_arr[:, j_idx],
            color="#F87171", alpha=0.22,
            label="Instantaneous MTG Limit ($a(t)\\,T_{\\mathrm{LV}} + \\tau_{\\mathrm{pass}}$)" if idx == 0 else None
        )
        ax.plot(t_arr, Tp2_arr[:, j_idx], color="#DC2626", ls="--", lw=0.9, alpha=0.75)
        ax.plot(t_arr, -Tn2_arr[:, j_idx], color="#DC2626", ls="--", lw=0.9, alpha=0.75)

        ax.plot(
            t_arr, tau1_arr[:, j_idx], color="#1D4ED8", ls="--", lw=1.45, alpha=0.90,
            label="Mode 1: Ideal Unlimited PD ($\\tau_{\\mathrm{PD}}$)" if idx == 0 else None
        )
        ax.plot(
            t_arr, tau2_arr[:, j_idx], color="#B91C1C", ls="-", lw=1.65,
            label="Mode 2: MTG Clamped ($\\tau_{\\mathrm{total}}$)" if idx == 0 else None
        )

        for st in keyframe_steps:
            ax.axvline(st * 0.02, color="#CBD5E1", lw=0.8, ls="-", zorder=0)

        ax.axhline(0.0, color="#94A3B8", lw=0.7, alpha=0.6)
        ax.set_title(j_title, fontsize=10.0, fontweight="bold", color="#1E293B", pad=5)
        ax.set_ylabel("Torque (N·m)", fontsize=9.0, color="#1E293B")
        ax.set_xlim(0.0, t_arr[-1])
        ax.grid(True, axis="y", alpha=0.25)
        ax.tick_params(labelsize=8.3)

        if idx == 0:
            ax.text(-0.11, 1.08, "B", transform=ax.transAxes, fontsize=14, fontweight="bold", color="#0F172A", va="bottom", ha="right")

    # Bottom Row: Panel C (Summed Torque Excess) & Panel D (Forward CoM Speed)
    gs_cd = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=outer[2], wspace=0.18)

    # Panel C: Total Exceeded Biological Torque Summed Across All 7 Joints
    ax_ex = fig.add_subplot(gs_cd[0, 0])
    mean_ex1 = float(np.mean(ex1_arr))
    max_ex1 = float(np.max(ex1_arr))
    ax_ex.fill_between(t_arr, 0.0, ex1_arr, color="#EA580C", alpha=0.24)
    ax_ex.plot(
        t_arr, ex1_arr, color="#C2410C", lw=1.55,
        label=f"Mode 1 Excess (Mean={mean_ex1:.1f}, Peak={max_ex1:.1f} N·m)"
    )
    ax_ex.plot(
        t_arr, ex2_arr, color="#B91C1C", lw=1.7, ls="-",
        label="Mode 2 MTG Clamped (0.00 N·m)"
    )
    for st in keyframe_steps:
        ax_ex.axvline(st * 0.02, color="#CBD5E1", lw=0.8, ls="-", zorder=0)
    ax_ex.set_title("Total Biological MTG Torque Excess (All 7 Joints)", fontsize=10.0, fontweight="bold", color="#1E293B", pad=5)
    ax_ex.set_ylabel("Total Excess (N·m)", fontsize=9.0, color="#1E293B")
    ax_ex.set_xlabel("Time (s)", fontsize=9.2, color="#1E293B")
    ax_ex.set_xlim(0.0, t_arr[-1])
    ax_ex.set_ylim(-15.0, max_ex1 * 1.32)
    ax_ex.grid(True, axis="y", alpha=0.25)
    ax_ex.tick_params(labelsize=8.3)
    ax_ex.legend(loc="upper right", fontsize=8.0, frameon=True, facecolor="#FFFFFF", edgecolor="#CBD5E1")
    ax_ex.text(-0.11, 1.08, "C", transform=ax_ex.transAxes, fontsize=14, fontweight="bold", color="#0F172A", va="bottom", ha="right")

    # Panel D: Pelvis Root Forward Speed vx(t)
    ax_vx = fig.add_subplot(gs_cd[0, 1])
    mean_vx1 = float(np.mean(vx1_arr))
    mean_vx2 = float(np.mean(vx2_arr))
    mean_vxref = float(np.mean(vxref_arr))
    ax_vx.axhline(mean_vxref, color="#15803D", ls=":", lw=1.4, label=f"Reference Target ({mean_vxref:.2f} m/s)")
    ax_vx.plot(t_arr, vx1_arr, color="#1D4ED8", ls="--", lw=1.55, label=f"Mode 1: Ideal PD (Mean={mean_vx1:.2f} m/s)")
    ax_vx.plot(t_arr, vx2_arr, color="#B91C1C", ls="-", lw=1.75, label=f"Mode 2: MTG Clamped (Mean={mean_vx2:.2f} m/s)")
    ax_vx.fill_between(t_arr, vx2_arr, vx1_arr, where=(vx1_arr > vx2_arr), color="#94A3B8", alpha=0.18)
    for st in keyframe_steps:
        ax_vx.axvline(st * 0.02, color="#CBD5E1", lw=0.8, ls="-", zorder=0)
    ax_vx.set_title("Pelvis Root Forward Speed $v_x(t)$", fontsize=10.0, fontweight="bold", color="#1E293B", pad=5)
    ax_vx.set_ylabel("Pelvis Speed $v_x$ (m/s)", fontsize=9.0, color="#1E293B")
    ax_vx.set_xlabel("Time (s)", fontsize=9.2, color="#1E293B")
    ax_vx.set_xlim(0.0, t_arr[-1])
    ymin_v = min(float(np.min(vx2_arr)), float(np.min(vx1_arr))) - 0.35
    ymax_v = max(float(np.max(vx1_arr)), mean_vxref) + 0.55
    ax_vx.set_ylim(ymin_v, ymax_v)
    ax_vx.grid(True, axis="y", alpha=0.25)
    ax_vx.tick_params(labelsize=8.3)
    ax_vx.legend(loc="lower left", fontsize=8.0, frameon=True, facecolor="#FFFFFF", edgecolor="#CBD5E1")
    ax_vx.text(-0.11, 1.08, "D", transform=ax_vx.transAxes, fontsize=14, fontweight="bold", color="#0F172A", va="bottom", ha="right")

    handles, labels = fig.axes[5].get_legend_handles_labels()
    fig.legend(
        handles, labels,
        loc="lower center", ncol=4, frameon=True,
        facecolor="#FFFFFF", edgecolor="#CBD5E1",
        fontsize=8.6, bbox_to_anchor=(0.5, 0.008)
    )

    fig.suptitle(
        "Figure 3: Fast Running Imitation (5.0 m/s) — Stride Keyframes, 500 Hz Joint Torques, MTG Excess & Pelvis Speed",
        fontsize=12.0, fontweight="bold", color="#0F172A", y=0.980
    )

    plt.subplots_adjust(left=0.06, right=0.98, top=0.925, bottom=0.085)
    fig.savefig("figures/fig3_running.png", dpi=300, bbox_inches="tight")
    fig.savefig("figures/fig3_running.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"[DONE] Saved figures/fig3_running.{{png,pdf}} | Mean Excess={mean_ex1:.2f} N*m | Mean Vx: M1={mean_vx1:.2f}, M2={mean_vx2:.2f}, Ref={mean_vxref:.2f} m/s")


if __name__ == "__main__":
    main()
