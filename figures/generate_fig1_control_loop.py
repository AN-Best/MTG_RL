"""
Generate Figure 1 (Methods): Hierarchical 50 Hz RL + 500 Hz PD & Hill-Type MTG
Clamped Control Loop Architecture.

Outputs:
  - figures/fig1_control_loop.png (300 DPI)
  - figures/fig1_control_loop.pdf (vector)
  - figures/fig1_control_loop.svg (vector)
"""

import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle


def draw_box(ax, x, y, w, h, title, body="", fc="#FFFFFF", ec="#2C3E50",
             lw=1.6, title_color="#1A252F", body_color="#2C3E50",
             title_fs=9.5, body_fs=8.0, radius=0.14, zorder=4):
    box = FancyBboxPatch(
        (x, y), w, h,
        boxstyle=f"round,pad=0.02,rounding_size={radius}",
        facecolor=fc, edgecolor=ec, linewidth=lw, zorder=zorder
    )
    ax.add_patch(box)
    if body:
        ax.text(
            x + w / 2.0, y + h * 0.72, title,
            ha="center", va="center", fontsize=title_fs, fontweight="bold",
            color=title_color, zorder=zorder + 1
        )
        ax.text(
            x + w / 2.0, y + h * 0.30, body,
            ha="center", va="center", fontsize=body_fs,
            color=body_color, linespacing=1.28, zorder=zorder + 1
        )
    else:
        ax.text(
            x + w / 2.0, y + h / 2.0, title,
            ha="center", va="center", fontsize=title_fs, fontweight="bold",
            color=title_color, linespacing=1.25, zorder=zorder + 1
        )
    return box


def draw_sum(ax, cx, cy, r=0.22, sym="+", fc="#FFFFFF", ec="#2C3E50", lw=1.6, zorder=6):
    c = Circle((cx, cy), r, facecolor=fc, edgecolor=ec, linewidth=lw, zorder=zorder)
    ax.add_patch(c)
    ax.text(cx, cy, sym, ha="center", va="center", fontsize=11, fontweight="bold",
            color=ec, zorder=zorder + 1)
    return c


def draw_arrow(ax, p1, p2, color="#2C3E50", lw=1.6, ls="-", rad=0.0, zorder=5,
               mutation_scale=12):
    arr = FancyArrowPatch(
        p1, p2,
        arrowstyle="-|>",
        mutation_scale=mutation_scale,
        color=color,
        linewidth=lw,
        linestyle=ls,
        connectionstyle=f"arc3,rad={rad}",
        zorder=zorder,
    )
    ax.add_patch(arr)
    return arr


def draw_ortho_arrow(ax, pts, color="#2C3E50", lw=1.6, ls="-", zorder=5, mutation_scale=12):
    """Draw a clean multi-segment orthogonal polyline ending with an arrowhead."""
    xs = [p[0] for p in pts[:-1]]
    ys = [p[1] for p in pts[:-1]]
    if len(pts) > 2:
        ax.plot(xs, ys, color=color, lw=lw, ls=ls, solid_capstyle="round",
                solid_joinstyle="round", zorder=zorder)
    draw_arrow(ax, pts[-2], pts[-1], color=color, lw=lw, ls=ls,
               zorder=zorder, mutation_scale=mutation_scale)


def main():
    os.makedirs("figures", exist_ok=True)

    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "mathtext.fontset": "dejavusans",
        "axes.edgecolor": "none",
    })

    fig, ax = plt.subplots(figsize=(15.5, 7.8), dpi=300)
    ax.set_xlim(0.0, 16.0)
    ax.set_ylim(0.0, 8.2)
    ax.axis("off")

    # -------------------------------------------------------------------------
    # Outer Rate-Domain Containers
    # -------------------------------------------------------------------------
    # 50 Hz RL Domain (Left)
    rl_bg = FancyBboxPatch(
        (0.25, 2.20), 4.75, 5.45,
        boxstyle="round,pad=0.04,rounding_size=0.22",
        facecolor="#F2F6FA", edgecolor="#3B6EA5", linewidth=1.6, linestyle="--", zorder=1
    )
    ax.add_patch(rl_bg)
    ax.text(
        0.45, 7.42, "50 Hz High-Level RL Control Loop  ($\\Delta t = 20$ ms)",
        ha="left", va="center", fontsize=9.8, fontweight="bold", color="#234E7A", zorder=2
    )

    # 500 Hz Physics & Actuation Domain (Right)
    phy_bg = FancyBboxPatch(
        (5.30, 1.35), 10.45, 6.30,
        boxstyle="round,pad=0.04,rounding_size=0.22",
        facecolor="#FDF9F3", edgecolor="#B86B27", linewidth=1.6, linestyle="-", zorder=1
    )
    ax.add_patch(phy_bg)
    ax.text(
        5.50, 7.42, "500 Hz Low-Level PD & Biological MTG Physics Loop  ($\\Delta t_{\\mathrm{sim}} = 2$ ms, $10\\times$ decimation)",
        ha="left", va="center", fontsize=9.8, fontweight="bold", color="#8A4B12", zorder=2
    )

    # Mode 2 MTG Clamping Sub-Container (inside 500 Hz domain)
    mtg_bg = FancyBboxPatch(
        (7.72, 1.65), 5.26, 4.20,
        boxstyle="round,pad=0.03,rounding_size=0.18",
        facecolor="#FFF3F0", edgecolor="#C0392B", linewidth=1.4, linestyle="-", zorder=2
    )
    ax.add_patch(mtg_bg)
    ax.text(
        7.90, 5.64, "Mode 2: Hill-Type MTG Clamping",
        ha="left", va="center", fontsize=9.0, fontweight="bold", color="#96281B", zorder=3
    )

    # -------------------------------------------------------------------------
    # Blocks: 50 Hz RL Domain
    # -------------------------------------------------------------------------
    draw_box(
        ax, 0.50, 4.45, 2.05, 1.35,
        "PPO Policy  $\\pi_\\theta$",
        "MLP Actor-Critic\n$o_t \\in \\mathbb{R}^{64} \\rightarrow a_t \\in \\mathbb{R}^7$",
        fc="#FFFFFF", ec="#2B5B84", title_color="#1B3B5A"
    )

    draw_box(
        ax, 2.95, 4.45, 1.85, 1.35,
        "Action Filter &\nSub-Step Interp.",
        "$\\tilde{a}_t = \\alpha\\tilde{a}_{t-1} + (1{-}\\alpha)a_t$\n(10 sub-steps)",
        fc="#FFFFFF", ec="#2B5B84", title_color="#1B3B5A", body_fs=7.4
    )

    draw_box(
        ax, 1.40, 6.15, 2.55, 0.92,
        "Reference Trajectory",
        "MoCap / CoM-Gated Phase\n$(q_{\\mathrm{ref}},\\; \\dot{q}_{\\mathrm{ref}})$",
        fc="#EBF3FB", ec="#3B6EA5", title_color="#1B3B5A", body_fs=7.8
    )

    # Summing junction for q_des = q_ref + a_k
    sum_q = (4.65, 6.61)
    draw_sum(ax, sum_q[0], sum_q[1], r=0.20, sym="+", ec="#2B5B84")

    # -------------------------------------------------------------------------
    # Blocks: 500 Hz PD & Mode 1 Bypass
    # -------------------------------------------------------------------------
    draw_box(
        ax, 5.55, 4.45, 1.85, 1.35,
        "PD Joint Controller",
        "$\\tau_{\\mathrm{PD}} = K_p(q_{\\mathrm{des}} - q)$\n$+\\; K_d(\\dot{q}_{\\mathrm{des}} - \\dot{q})$",
        fc="#FFFFFF", ec="#8A4B12", title_color="#5C310A", body_fs=7.9
    )

    # Mode 1 bypass badge (top of 500 Hz box)
    draw_box(
        ax, 8.65, 6.12, 3.55, 0.92,
        "Mode 1: Ideal Unlimited PD (Bypass)",
        "$\\tau_{\\mathrm{total}} = \\tau_{\\mathrm{PD}}$  (no biological torque limits)",
        fc="#EAF4FC", ec="#2980B9", lw=1.4, title_color="#1A5276", body_fs=7.8
    )

    # -------------------------------------------------------------------------
    # Blocks: Mode 2 MTG Clamping (inside mtg_bg)
    # -------------------------------------------------------------------------
    # Summing junction 1: tau_act_des = tau_PD - tau_pass
    sum_sub = (8.18, 4.92)
    draw_sum(ax, sum_sub[0], sum_sub[1], r=0.19, sym="$\\Sigma$", ec="#C0392B")
    ax.text(7.90, 5.04, "+", ha="center", va="center", fontsize=8.5, fontweight="bold", color="#C0392B", zorder=7)
    ax.text(8.30, 4.62, "–", ha="center", va="center", fontsize=9.5, fontweight="bold", color="#C0392B", zorder=7)

    # Hard Clamp Block
    draw_box(
        ax, 9.75, 4.42, 2.20, 1.00,
        "Instantaneous Hard Clamp",
        "$\\tau_{\\mathrm{act}} = \\mathrm{clip}(\\tau_{\\mathrm{act}}^{\\mathrm{des}},\\, -T_{\\mathrm{act}}^-,\\, +T_{\\mathrm{act}}^+)$",
        fc="#FFFFFF", ec="#C0392B", lw=1.8, title_color="#96281B", body_fs=7.5
    )

    # Summing junction 2: tau_total = tau_act + tau_pass
    sum_add = (12.55, 4.92)
    draw_sum(ax, sum_add[0], sum_add[1], r=0.19, sym="+", ec="#C0392B")

    # Passive Joint Torque Block
    draw_box(
        ax, 7.92, 1.88, 2.26, 0.98,
        "Passive Tissue Torque",
        "$\\tau_{\\mathrm{pass}}(q) = k_1[q_1{-}q]_+^2 - k_2[q{-}q_2]_+^2$",
        fc="#FFFFFF", ec="#A93226", lw=1.3, title_color="#78281F", body_fs=7.3
    )

    # Force-Length-Velocity Envelope Block
    draw_box(
        ax, 10.42, 1.88, 2.38, 0.98,
        "MTG Capacity  $T_{\\mathrm{LV}}^{\\pm}(q,\\dot{q})$",
        "$T_{\\max}^{\\pm}\\, f_L(q)\\, f_V^{\\pm}(\\dot{q})$",
        fc="#F9EBEA", ec="#7D3C98", lw=1.4, title_color="#5B2C6F", body_fs=8.0
    )

    # 1st-Order Activation Dynamics Block
    draw_box(
        ax, 8.70, 3.08, 3.78, 0.88,
        "1st-Order Activation Dynamics  $a(t)$",
        "$u_{\\mathrm{req}} = \\mathrm{clip}(|\\tau_{\\mathrm{act}}^{\\mathrm{des}}| / T_{\\mathrm{LV}}^{\\mathrm{dir}}, 0, 1),\\quad \\dot{a} = (u_{\\mathrm{req}} - a)/\\tau_c$  ($20/40$ ms)",
        fc="#FFFFFF", ec="#C0392B", lw=1.3, title_color="#78281F", body_fs=7.2
    )

    # -------------------------------------------------------------------------
    # Block: MuJoCo 10-DoF Biped Dynamics
    # -------------------------------------------------------------------------
    draw_box(
        ax, 13.45, 4.10, 2.05, 1.75,
        "MuJoCo 10-DoF\nBiped Dynamics",
        "7 Actuated Joints\n+ 3-DoF Planar Root\nLCP Ground/Chair\n& Slip Force $F_{\\mathrm{slip}}$",
        fc="#FFFFFF", ec="#1E8449", lw=1.8, title_color="#145A32", body_fs=7.6
    )

    # -------------------------------------------------------------------------
    # Signal Arrows: Forward Path
    # -------------------------------------------------------------------------
    # Policy -> Filter
    draw_arrow(ax, (2.55, 5.12), (2.95, 5.12), color="#2B5B84")
    ax.text(2.75, 5.30, "$a_t$", ha="center", va="bottom", fontsize=8.5, color="#1B3B5A", fontweight="bold")

    # Reference -> sum_q
    draw_arrow(ax, (3.95, 6.61), (4.45, 6.61), color="#2B5B84")
    ax.text(4.20, 6.76, "$q_{\\mathrm{ref}}$", ha="center", va="bottom", fontsize=8.2, color="#1B3B5A")

    # Filter -> sum_q
    draw_ortho_arrow(ax, [(3.88, 5.80), (3.88, 6.02), (4.65, 6.02), (4.65, 6.41)], color="#2B5B84")
    ax.text(4.27, 6.08, "$\\tilde{a}_k$", ha="center", va="bottom", fontsize=8.2, color="#1B3B5A")

    # sum_q -> PD Controller
    draw_ortho_arrow(ax, [(4.85, 6.61), (6.47, 6.61), (6.47, 5.80)], color="#2B5B84")
    ax.text(5.75, 6.75, "$q_{\\mathrm{des}} = q_{\\mathrm{ref}} + \\tilde{a}_k$",
            ha="center", va="bottom", fontsize=8.2, color="#1B3B5A", fontweight="bold")

    # PD Controller -> Mode 2 (sum_sub)
    draw_arrow(ax, (7.40, 4.92), (7.99, 4.92), color="#8A4B12", lw=1.8)
    ax.text(7.56, 4.62, "$\\tau_{\\mathrm{PD}}$", ha="center", va="top", fontsize=8.4,
            color="#5C310A", fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.10", fc="#FDF9F3", ec="none", alpha=0.9))

    # PD Controller -> Mode 1 Bypass (dashed blue)
    draw_ortho_arrow(ax, [(7.56, 4.92), (7.56, 6.58), (8.65, 6.58)], color="#2980B9", lw=1.5, ls="--")

    # Mode 1 Bypass -> MuJoCo (dashed blue)
    draw_ortho_arrow(ax, [(12.20, 6.58), (14.48, 6.58), (14.48, 5.85)], color="#2980B9", lw=1.5, ls="--")
    ax.text(13.35, 6.72, "$\\tau_{\\mathrm{total}} = \\tau_{\\mathrm{PD}}$ (Mode 1)",
            ha="center", va="bottom", fontsize=7.8, color="#1A5276")

    # sum_sub -> Hard Clamp
    draw_arrow(ax, (8.37, 4.92), (9.75, 4.92), color="#C0392B", lw=1.7)
    ax.text(9.06, 5.06, "$\\tau_{\\mathrm{act}}^{\\mathrm{des}}$", ha="center", va="bottom",
            fontsize=8.5, color="#96281B", fontweight="bold")

    # Tap tau_act_des down into Activation Dynamics
    draw_ortho_arrow(ax, [(9.06, 4.92), (9.06, 3.96)], color="#C0392B", lw=1.3)

    # T_LV -> Activation Dynamics
    draw_ortho_arrow(ax, [(11.61, 2.86), (11.61, 3.08)], color="#7D3C98", lw=1.4)

    # Activation Dynamics -> Hard Clamp (T_act = a * T_LV)
    draw_arrow(ax, (10.85, 3.96), (10.85, 4.42), color="#C0392B", lw=1.5)
    ax.text(11.72, 4.19, "$T_{\\mathrm{act}}^{\\pm} = a\\,T_{\\mathrm{LV}}^{\\pm}$",
            ha="center", va="center", fontsize=7.7, color="#96281B", fontweight="bold")

    # Hard Clamp -> sum_add
    draw_arrow(ax, (11.95, 4.92), (12.36, 4.92), color="#C0392B", lw=1.7)
    ax.text(12.16, 5.09, "$\\tau_{\\mathrm{act}}$", ha="center", va="bottom",
            fontsize=8.2, color="#96281B", fontweight="bold")

    # Passive Torque -> sum_sub (subtract) and sum_add (add)
    draw_ortho_arrow(ax, [(8.18, 2.86), (8.18, 4.73)], color="#A93226", lw=1.3)
    ax.text(8.10, 3.72, "$\\tau_{\\mathrm{pass}}$", ha="right", va="center", fontsize=7.9, color="#78281F")

    draw_ortho_arrow(ax, [(10.18, 2.36), (10.30, 2.36), (10.30, 1.75), (12.88, 1.75), (12.88, 4.48), (12.55, 4.48), (12.55, 4.73)],
                     color="#A93226", lw=1.3, ls="-")

    # sum_add -> MuJoCo
    draw_arrow(ax, (12.74, 4.92), (13.45, 4.92), color="#1E8449", lw=1.9)
    ax.text(13.21, 5.09, "$\\tau_{\\mathrm{total}}$", ha="center", va="bottom",
            fontsize=8.4, color="#145A32", fontweight="bold")

    # -------------------------------------------------------------------------
    # Feedback Loops (Bottom)
    # -------------------------------------------------------------------------
    # 500 Hz Internal State Feedback (q, qdot)
    fb_y_500 = 1.00
    ax.plot([14.48, 14.48, 6.47], [4.10, fb_y_500, fb_y_500], color="#1E8449", lw=1.6, zorder=4)
    draw_arrow(ax, (6.47, fb_y_500), (6.47, 4.45), color="#1E8449", lw=1.6)
    draw_arrow(ax, (9.05, fb_y_500), (9.05, 1.90), color="#1E8449", lw=1.4)
    draw_arrow(ax, (11.68, fb_y_500), (11.68, 1.90), color="#1E8449", lw=1.4)
    ax.text(
        10.50, fb_y_500 - 0.14,
        "500 Hz Proprioceptive Feedback: Joint Angles $q$ & Velocities $\\dot{q}$",
        ha="center", va="top", fontsize=8.3, color="#145A32", fontweight="bold"
    )

    # 50 Hz RL Observation Feedback (o_t)
    fb_y_50 = 0.42
    draw_ortho_arrow(
        ax,
        [(14.85, 4.10), (14.85, fb_y_50), (1.52, fb_y_50), (1.52, 4.45)],
        color="#2B5B84", lw=1.6, ls="--"
    )
    ax.text(
        8.20, fb_y_50 - 0.12,
        "50 Hz RL Observation $o_t = [q,\\, \\dot{q},\\, e_q,\\, e_{\\dot{q}},\\, \\mathrm{CoM},\\, \\mathrm{End\\,Effectors},\\, F_{\\mathrm{contact}},\\, \\tilde{a}_{t-1},\\, \\phi]$",
        ha="center", va="top", fontsize=8.3, color="#1B3B5A", fontweight="bold"
    )

    # Title Banner
    ax.text(
        8.0, 7.95,
        "Figure 1: Hierarchical RL–PD Control Architecture with Instantaneous 500 Hz Hill-Type MTG Clamping",
        ha="center", va="center", fontsize=11.5, fontweight="bold", color="#1A252F"
    )

    plt.tight_layout(pad=0.3)
    fig.savefig("figures/fig1_control_loop.png", dpi=300, bbox_inches="tight")
    fig.savefig("figures/fig1_control_loop.pdf", bbox_inches="tight")
    fig.savefig("figures/fig1_control_loop.svg", bbox_inches="tight")
    plt.close(fig)
    print("[DONE] Saved figures/fig1_control_loop.{png,pdf,svg}")


if __name__ == "__main__":
    main()
