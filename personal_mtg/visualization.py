"""
Visualization module for MTG personalized characteristic curves,
activation trajectories, and clinical gait comparison (matching Millard 2019 Figures 5 & B.7).
"""

import numpy as np
import matplotlib.pyplot as plt
import os
from typing import Dict, Any
from .torque_muscle import MillardTorqueMuscle


def plot_mtg_personalization_results(
    muscle: MillardTorqueMuscle,
    fit_result: Dict[str, Any],
    time: np.ndarray,
    theta: np.ndarray,
    omega: np.ndarray,
    tau_exp: np.ndarray,
    output_filepath: str,
    joint_name: str = "Joint"
):
    os.makedirs(os.path.dirname(os.path.abspath(output_filepath)), exist_ok=True)

    p_init = fit_result['params_initial']
    p_fit = fit_result['params_fitted']

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(f"Personalized MTG Fit: {joint_name} ({muscle.name})\n"
                 f"s^A={fit_result['s_A']:.2f}, s^V={fit_result['s_V']:.2f}, "
                 f"Delta^P={fit_result['delta_P']:.2f} deg, lambda^P={fit_result['lambda_P']:.2f}, s^s={fit_result['s_s']:.2f}",
                 fontsize=14, fontweight='bold')

    th_min, th_max = min(np.min(theta) - 15, -90), max(np.max(theta) + 15, 60)
    theta_domain = np.linspace(th_min, th_max, 250)
    om_max = max(np.max(np.abs(omega)) * 1.3, muscle.omega_max)
    omega_domain = np.linspace(-om_max, om_max, 250)

    ax_ta_init = axes[0, 0]
    tau_act_init = p_init[4] * muscle.tau_o * muscle.calc_tA(theta_domain, p_init[0])
    tau_pas_init = p_init[4] * muscle.tau_o * muscle.calc_tPE(theta_domain, p_init[2], p_init[3])
    tau_tot_init = tau_act_init + tau_pas_init

    ax_ta_init.plot(theta_domain, tau_tot_init, 'b-', label='Total Max Capacity (a=1)', lw=2)
    ax_ta_init.plot(theta_domain, tau_pas_init, 'g--', label='Passive (a=0)', lw=1.5)
    ax_ta_init.plot(theta_domain, tau_act_init, 'c:', label='Active Isometric', lw=1.2)
    ax_ta_init.scatter(theta, tau_exp, color='red', alpha=0.6, s=15, label='Demanded (Gait)', zorder=5)
    ax_ta_init.set_title("Default Torque-Angle Envelope")
    ax_ta_init.set_xlabel("Joint Angle (deg)")
    ax_ta_init.set_ylabel("Torque (Nm)")
    ax_ta_init.grid(True, alpha=0.3)
    ax_ta_init.legend(loc='best', fontsize=8)

    ax_tv_init = axes[0, 1]
    tv_curve_init = p_init[4] * muscle.tau_o * muscle.calc_tV(omega_domain, p_init[1])
    ax_tv_init.plot(omega_domain, tv_curve_init, 'b-', lw=2, label='Torque-Velocity Curve')
    tau_A_init = tau_exp - p_init[4] * muscle.tau_o * muscle.calc_tPE(theta, p_init[2], p_init[3])
    ax_tv_init.scatter(omega * muscle.direction, tau_A_init, color='red', alpha=0.6, s=15, label='Demanded Active', zorder=5)
    ax_tv_init.axvline(0, color='gray', linestyle=':')
    ax_tv_init.set_title("Default Torque-Velocity Curve")
    ax_tv_init.set_xlabel("Shortening Velocity (deg/s)")
    ax_tv_init.set_ylabel("Active Torque (Nm)")
    ax_tv_init.grid(True, alpha=0.3)
    ax_tv_init.legend(loc='best', fontsize=8)

    ax_act = axes[0, 2]
    ax_act.fill_between(time, 0.0, 1.0, color='green', alpha=0.08, label='Feasible Zone [0, 1]')
    ax_act.plot(time, fit_result['initial_activation'], 'r--', label='Default (Unfitted: Infeasible)', lw=1.5)
    act_fitted_clean = np.clip(fit_result['fitted_activation'], 0.0, 1.0)
    ax_act.plot(time, act_fitted_clean, 'g-', label='Fitted (Personalized: Feasible)', lw=2)
    ax_act.axhline(1.0, color='black', linestyle=':', lw=1.2, label='Max Bound (a=1.0)')
    ax_act.axhline(0.0, color='black', linestyle=':', lw=1.2, label='Min Bound (a=0.0)')
    ax_act.set_title("Muscle Activation Feasibility")
    ax_act.set_xlabel("Time (s)")
    ax_act.set_ylabel("Activation a(t)")
    ax_act.set_ylim(-0.2, max(1.5, float(np.max(fit_result['initial_activation'])) + 0.2))
    ax_act.grid(True, alpha=0.3)
    ax_act.legend(loc='best', fontsize=8)

    ax_ta_fit = axes[1, 0]
    tau_act_fit = p_fit[4] * muscle.tau_o * muscle.calc_tA(theta_domain, p_fit[0])
    tau_pas_fit = p_fit[4] * muscle.tau_o * muscle.calc_tPE(theta_domain, p_fit[2], p_fit[3])
    tau_tot_fit = tau_act_fit + tau_pas_fit

    ax_ta_fit.plot(theta_domain, tau_tot_fit, 'g-', label='Fitted Total Max (a=1)', lw=2)
    ax_ta_fit.plot(theta_domain, tau_pas_fit, 'm--', label='Fitted Passive (a=0)', lw=1.5)
    ax_ta_fit.plot(theta_domain, tau_act_fit, 'c:', label='Fitted Active Isometric', lw=1.2)
    ax_ta_fit.scatter(theta, tau_exp, color='red', alpha=0.6, s=15, label='Demanded (Gait)', zorder=5)
    ax_ta_fit.set_title("Fitted Torque-Angle Envelope")
    ax_ta_fit.set_xlabel("Joint Angle (deg)")
    ax_ta_fit.set_ylabel("Torque (Nm)")
    ax_ta_fit.grid(True, alpha=0.3)
    ax_ta_fit.legend(loc='best', fontsize=8)

    ax_tv_fit = axes[1, 1]
    tv_curve_fit = p_fit[4] * muscle.tau_o * muscle.calc_tV(omega_domain, p_fit[1])
    ax_tv_fit.plot(omega_domain, tv_curve_fit, 'g-', lw=2, label='Fitted Torque-Velocity')
    tau_A_fit = tau_exp - p_fit[4] * muscle.tau_o * muscle.calc_tPE(theta, p_fit[2], p_fit[3])
    ax_tv_fit.scatter(omega * muscle.direction, tau_A_fit, color='red', alpha=0.6, s=15, label='Demanded Active', zorder=5)
    ax_tv_fit.axvline(0, color='gray', linestyle=':')
    ax_tv_fit.set_title("Fitted Torque-Velocity Curve")
    ax_tv_fit.set_xlabel("Shortening Velocity (deg/s)")
    ax_tv_fit.set_ylabel("Active Torque (Nm)")
    ax_tv_fit.grid(True, alpha=0.3)
    ax_tv_fit.legend(loc='best', fontsize=8)

    ax_bar = axes[1, 2]
    param_names = [r'$s^A$', r'$s^V$', r'$\Delta^P (deg)$', r'$\lambda^P$', r'$s^s$']
    x_pos = np.arange(len(param_names))
    width = 0.35
    ax_bar.bar(x_pos - width/2, p_init, width, label='Initial', color='steelblue')
    ax_bar.bar(x_pos + width/2, p_fit, width, label='Personalized', color='forestgreen')
    ax_bar.set_xticks(x_pos)
    ax_bar.set_xticklabels(param_names)
    ax_bar.set_title("MTG Scaling Parameters")
    ax_bar.set_ylabel("Value")
    ax_bar.grid(True, alpha=0.3, axis='y')
    ax_bar.legend(loc='best')

    plt.tight_layout()
    plt.savefig(output_filepath, dpi=200)
    plt.close()
    print(f"Saved diagnostic plot: {output_filepath}")

