"""
Generate Figure 5 (Results — Experiment 6: Un-Time-Bounded 5-Times Sit-to-Stand):
  - Panel A: 5 MuJoCo 3-model side-by-side 5xSTS keyframes (backless chair)
             (Left/Cyan: Mode 1 Ideal PD, Center/Orange-Green: Mode 2 Young MTG, Right/Purple: Mode 2 Elderly MTG)
  - Panel B: 2x2 Joint Torques (Lumbar, Right Hip, Right Knee, Right Ankle) with shaded MTG limits
  - Panel C: Whole-Body CoM Height z(t) across all 5 repetitions (reference CoM lines removed).

Outputs:
  - figures/fig5_5xsts.png (300 DPI)
  - figures/fig5_5xsts.pdf (vector + high-res embedded frames)
"""

import os
import sys

sys.path.insert(0, os.getcwd())
import mujoco
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from Stable_Motion_Tracker import MotionClip
from run_all_experiments import (
    FiveTimesSTSCoMEnv,
    device,
    load_sts_policy_checkpoint,
    make_sts_policy,
)


def main():
    os.makedirs("figures", exist_ok=True)

    clip = MotionClip.load("motions/CeTI_5xSTS_subK1.npz")
    policy1 = make_sts_policy(obs_dim=64, device=device)
    assert load_sts_policy_checkpoint(policy1, "checkpoints/exp6_5xsts_com_mode1.pt", device=device)
    policy2 = make_sts_policy(obs_dim=64, device=device)
    assert load_sts_policy_checkpoint(policy2, "checkpoints/exp6_5xsts_com_mode2_young.pt", device=device)
    policy3 = make_sts_policy(obs_dim=64, device=device)
    assert load_sts_policy_checkpoint(policy3, "checkpoints/exp6_5xsts_com_mode2_elderly.pt", device=device)

    max_steps = 950  # 19.00 s (captures all 5 reps of Ideal, Young, and Elderly)
    keyframe_steps = [0, 70, 150, 380, 710]  # 0.00s, 1.40s, 3.00s, 7.60s, 14.20s

    env1 = FiveTimesSTSCoMEnv(clip, mode="mode1", mtg_phase="hard", max_steps=max_steps, is_train=False)
    env2 = FiveTimesSTSCoMEnv(clip, mode="mode2", mtg_phase="hard", max_steps=max_steps, is_train=False)
    env3 = FiveTimesSTSCoMEnv(clip, mode="mode2_elderly", mtg_phase="hard", max_steps=max_steps, is_train=False)

    obs1 = env1.reset()
    obs2 = env2.reset()
    obs3 = env3.reset()

    overlay_model = mujoco.MjModel.from_xml_path("biped_overlay_sts.xml")
    overlay_data = mujoco.MjData(overlay_model)
    renderer = mujoco.Renderer(overlay_model, height=460, width=560)
    camera = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(camera)
    camera.distance = 3.20
    camera.elevation = -10.0
    camera.azimuth = 90.0

    t_list = []
    comz1, comz2, comz3 = [], [], []
    tau1_list, tau2_list, tau3_list = [], [], []
    Tp2_list, Tn2_list = [], []
    Tp3_list, Tn3_list = [], []
    key_imgs = {}

    done1 = done2 = done3 = False

    for step in range(max_steps):
        if step in keyframe_steps:
            q1_vis = env1.env.data.qpos[:10].copy()
            q2_vis = env2.env.data.qpos[:10].copy()
            q3_vis = env3.env.data.qpos[:10].copy()
            q1_vis[0] -= 1.05
            q3_vis[0] += 1.05
            overlay_data.qpos[:10] = q1_vis
            overlay_data.qpos[10:20] = q2_vis
            overlay_data.qpos[20:30] = q3_vis
            mujoco.mj_forward(overlay_model, overlay_data)
            camera.lookat[:] = [-0.10, 0.0, 0.72]
            renderer.update_scene(overlay_data, camera)
            key_imgs[step] = renderer.render().copy()

        with torch.no_grad():
            a1 = policy1.get_action(obs1, deterministic=True)[0].cpu().numpy() if not done1 else np.zeros(7)
            a2 = policy2.get_action(obs2, deterministic=True)[0].cpu().numpy() if not done2 else np.zeros(7)
            a3 = policy3.get_action(obs3, deterministic=True)[0].cpu().numpy() if not done3 else np.zeros(7)

        if not done1:
            obs1, _, d1, inf1 = env1.step(a1)
            done1 = d1
        if not done2:
            obs2, _, d2, inf2 = env2.step(a2)
            done2 = d2
        if not done3:
            obs3, _, d3, inf3 = env3.step(a3)
            done3 = d3

        t_list.append((step + 1) * 0.02)
        comz1.append(float(env1._get_com()[2]))
        comz2.append(float(env2._get_com()[2]))
        comz3.append(float(env3._get_com()[2]))

        tau1_list.append(env1.env.data.ctrl[:7].copy())
        tau2_list.append(env2.env.data.ctrl[:7].copy())
        tau3_list.append(env3.env.data.ctrl[:7].copy())
        Tp2_list.append(env2.env.actuator.last_T_pos.copy())
        Tn2_list.append(env2.env.actuator.last_T_neg.copy())
        Tp3_list.append(env3.env.actuator.last_T_pos.copy())
        Tn3_list.append(env3.env.actuator.last_T_neg.copy())

    t_arr = np.array(t_list)
    tau1_arr = np.array(tau1_list)
    tau2_arr = np.array(tau2_list)
    tau3_arr = np.array(tau3_list)
    Tp2_arr = np.array(Tp2_list)
    Tn2_arr = np.array(Tn2_list)
    Tp3_arr = np.array(Tp3_list)
    Tn3_arr = np.array(Tn3_list)

    # -------------------------------------------------------------------------
    # Compose Publication Figure 5 (matching Figures 2 & 3 style)
    # -------------------------------------------------------------------------
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "mathtext.fontset": "dejavusans",
        "axes.spines.top": False,
        "axes.spines.right": False,
    })

    fig = plt.figure(figsize=(14.8, 11.6), dpi=300)
    outer = gridspec.GridSpec(3, 1, height_ratios=[1.02, 1.85, 0.96], hspace=0.31)

    # Panel A: 5 Side-by-Side 5xSTS Keyframes
    gs_a = gridspec.GridSpecFromSubplotSpec(1, 5, subplot_spec=outer[0], wspace=0.06)
    phase_labels = ["Seated Start", "Rep 1 Ascent", "Rep 1 Sit-Down", "Mid-Test Lag", "Young 5/5 Done"]
    for idx, st in enumerate(keyframe_steps):
        ax_img = fig.add_subplot(gs_a[0, idx])
        ax_img.imshow(key_imgs[st])
        ax_img.set_xticks([])
        ax_img.set_yticks([])
        for sp in ax_img.spines.values():
            sp.set_visible(True)
            sp.set_edgecolor("#CBD5E1")
            sp.set_linewidth(1.1)
        ax_img.set_title(f"$t = {st*0.02:.2f}$ s", fontsize=9.5, fontweight="bold", color="#1E293B", pad=5)
        if idx == 0:
            ax_img.text(-0.08, 1.12, "A", transform=ax_img.transAxes, fontsize=14, fontweight="bold", color="#0F172A", va="bottom", ha="right")

    fig.text(
        0.50, 0.692,
        "Side-by-Side 5xSTS Keyframes (Backless Chair):  Left (Cyan) = Mode 1 (Ideal PD)   |   Center (Orange/Green) = Mode 2 (Young MTG)   |   Right (Purple) = Mode 2 (Elderly MTG)",
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

        # Shaded Young MTG instantaneous envelope
        ax.fill_between(
            t_arr, -Tn2_arr[:, j_idx], Tp2_arr[:, j_idx],
            color="#F87171", alpha=0.18,
            label="Young MTG Dynamic Limit" if idx == 0 else None
        )
        # Shaded Elderly MTG instantaneous envelope
        ax.fill_between(
            t_arr, -Tn3_arr[:, j_idx], Tp3_arr[:, j_idx],
            color="#A855F7", alpha=0.18,
            label="Elderly MTG Dynamic Limit" if idx == 0 else None
        )

        # Torques
        ax.plot(
            t_arr, tau1_arr[:, j_idx], color="#1D4ED8", ls="--", lw=1.25, alpha=0.85,
            label="Mode 1: Ideal Unlimited PD" if idx == 0 else None
        )
        ax.plot(
            t_arr, tau2_arr[:, j_idx], color="#DC2626", ls="-", lw=1.45,
            label="Mode 2: Young Millard MTG" if idx == 0 else None
        )
        ax.plot(
            t_arr, tau3_arr[:, j_idx], color="#7E22CE", ls="-", lw=1.50,
            label="Mode 2: Elderly Millard MTG" if idx == 0 else None
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

    # Panel C: Whole-Body CoM Height z(t) (NO reference CoM height lines, NO zero-shot line)
    ax_c = fig.add_subplot(outer[2])
    ax_c.plot(t_arr, comz1, color="#1D4ED8", ls="--", lw=1.8, label=f"Mode 1: Ideal Unlimited PD ({env1.reps_completed}/5 reps)")
    ax_c.plot(t_arr, comz2, color="#DC2626", ls="-", lw=2.0, label=f"Mode 2: Young Millard MTG ({env2.reps_completed}/5 reps)")
    ax_c.plot(t_arr, comz3, color="#7E22CE", ls="-", lw=2.1, label=f"Mode 2: Elderly Millard MTG ({env3.reps_completed}/5 reps)")

    for st in keyframe_steps:
        ax_c.axvline(st * 0.02, color="#CBD5E1", lw=0.8, ls="-", zorder=0)

    ax_c.set_title("Un-Time-Bounded 5xSTS Whole-Body Center-of-Mass Height $z(t)$", fontsize=10.0, fontweight="bold", color="#1E293B", pad=5)
    ax_c.set_ylabel("CoM Height $z$ (m)", fontsize=9.0, color="#1E293B")
    ax_c.set_xlabel("Time (s)", fontsize=9.2, color="#1E293B")
    ax_c.set_xlim(0.0, t_arr[-1])
    ax_c.set_ylim(0.62, 0.94)
    ax_c.grid(True, alpha=0.28)
    ax_c.tick_params(labelsize=8.3)
    ax_c.legend(loc="lower right", ncol=3, fontsize=8.2, frameon=True, facecolor="#FFFFFF", edgecolor="#CBD5E1")
    ax_c.text(-0.052, 1.08, "C", transform=ax_c.transAxes, fontsize=14, fontweight="bold", color="#0F172A", va="bottom", ha="right")

    # Shared figure-level legend for Panel B at bottom
    handles, labels = fig.axes[5].get_legend_handles_labels()
    fig.legend(
        handles, labels,
        loc="lower center", ncol=5, frameon=True,
        facecolor="#FFFFFF", edgecolor="#CBD5E1",
        fontsize=8.3, bbox_to_anchor=(0.5, 0.008)
    )

    fig.suptitle(
        "Figure 5: Un-Time-Bounded 5-Times Sit-to-Stand (5xSTS) — Keyframes, Joint Torques with MTG Limits & CoM Height $z(t)$",
        fontsize=11.8, fontweight="bold", color="#0F172A", y=0.980
    )

    plt.subplots_adjust(left=0.06, right=0.98, top=0.925, bottom=0.085)
    fig.savefig("figures/fig5_5xsts.png", dpi=300, bbox_inches="tight")
    fig.savefig("figures/fig5_5xsts.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"[DONE] Saved figures/fig5_5xsts.{{png,pdf}} | Reps: M1={env1.reps_completed}/5, Young={env2.reps_completed}/5, Elderly={env3.reps_completed}/5")


if __name__ == "__main__":
    main()
