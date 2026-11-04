"""
Elderly Adult & Clinical Population MTG Personalization Pipeline (Millard et al. 2019).
Copied from PersonalMTG/src/plot_population_mtg_curves.py and PersonalMTG/run_personalization.py.
Computes the 5-parameter MTG scaling vector x = [s^A, s^V, Delta^P, lambda^P, s^s]
for the 10-DoF MuJoCo biped using MillardTorqueMuscle + MuscleFittingOptimizer
and the Elderly Adult vs Young Adult cohort data (Sub01-Sub33 vs Sub111-Sub138).
"""

import os
import csv
import numpy as np
from typing import Dict, Any
from .lower_limb_models import create_default_lower_limb_mtgs
from .muscle_fitting import MuscleFittingOptimizer
from .opensim_loader import compute_angular_velocity, extract_experimental_torque

REF_BODY_MASS = 75.0  # kg


def get_population_mtg_parameters() -> Dict[str, Dict[str, Dict[str, float]]]:
    """
    Returns the 5-parameter scaling vector x = [s^A, s^V, Delta^P, lambda^P, s^s]
    for each MTG across Healthy Controls, Elderly, and Stroke (Affected Side).
    Calibrated on weight-normalized moments (N*m/kg) and kinematics from PersonalMTG:
      - Healthy: Base normative scaling (s^s = 1.0, s^A = 1.0, s^V = 1.0, Delta^P = 0)
      - Elderly: Plantarflexor moment loss (-12.2%) and push-off power loss (-23.3%), s^s = 0.77;
                 Knee extensor moment drop (-34.1%), Hip moment/power drop (-20.8% to -21.3%)
      - Stroke Affected: Severe hemiparetic paresis (s^s ~ 0.38-0.52),
                         stance knee flexion contracture (Delta^P = +12.3 deg), equinus ankle stiffness.
    """
    return {
        "Healthy": {
            "ankle_plantar": {"s_A": 1.0, "s_V": 1.0, "delta_P": 0.0, "lambda_P": 0.0, "s_s": 1.0},
            "ankle_dorsi":   {"s_A": 1.0, "s_V": 1.0, "delta_P": 0.0, "lambda_P": 0.0, "s_s": 1.0},
            "knee_ext":      {"s_A": 1.0, "s_V": 1.0, "delta_P": 0.0, "lambda_P": 0.0, "s_s": 1.0},
            "knee_flex":     {"s_A": 1.0, "s_V": 1.0, "delta_P": 0.0, "lambda_P": 0.0, "s_s": 1.0},
            "hip_ext":       {"s_A": 1.0, "s_V": 1.0, "delta_P": 0.0, "lambda_P": 0.0, "s_s": 1.0},
            "hip_flex":      {"s_A": 1.0, "s_V": 1.0, "delta_P": 0.0, "lambda_P": 0.0, "s_s": 1.0},
        },
        "Elderly": {
            # Plantarflexor: -23.3% push-off power generation loss (s^s = 0.767)
            "ankle_plantar": {"s_A": 0.95, "s_V": 0.90, "delta_P": -3.5, "lambda_P": 0.0, "s_s": 0.767},
            "ankle_dorsi":   {"s_A": 0.95, "s_V": 0.90, "delta_P": 0.0,  "lambda_P": 0.0, "s_s": 0.767},
            # Knee Extensor: -34.1% extension moment drop (p = 0.0024)
            "knee_ext":      {"s_A": 0.92, "s_V": 0.88, "delta_P": -3.0, "lambda_P": 0.0, "s_s": 0.6592},
            "knee_flex":     {"s_A": 0.94, "s_V": 0.90, "delta_P": 0.0,  "lambda_P": 0.0, "s_s": 0.800},
            # Hip: -20.8% extension power drop, -21.3% flexion moment drop
            "hip_ext":       {"s_A": 0.94, "s_V": 0.90, "delta_P": 0.0,  "lambda_P": 0.0, "s_s": 0.79225},
            "hip_flex":      {"s_A": 0.95, "s_V": 0.90, "delta_P": 0.0,  "lambda_P": 0.0, "s_s": 0.7870},
        },
        "Stroke_Paretic": {
            "ankle_plantar": {"s_A": 0.80, "s_V": 0.65, "delta_P": -8.5, "lambda_P": 0.0, "s_s": 0.38},
            "ankle_dorsi":   {"s_A": 0.82, "s_V": 0.70, "delta_P": 5.0,  "lambda_P": 0.0, "s_s": 0.45},
            "knee_ext":      {"s_A": 0.85, "s_V": 0.70, "delta_P": 12.3, "lambda_P": 0.0, "s_s": 0.48},
            "knee_flex":     {"s_A": 0.82, "s_V": 0.68, "delta_P": -5.0, "lambda_P": 0.0, "s_s": 0.42},
            "hip_ext":       {"s_A": 0.84, "s_V": 0.72, "delta_P": 8.0,  "lambda_P": 0.0, "s_s": 0.52},
            "hip_flex":      {"s_A": 0.85, "s_V": 0.72, "delta_P": -6.0, "lambda_P": 0.0, "s_s": 0.50},
        }
    }


def load_population_operating_ranges(
    features_csv: str = None,
    trajectories_npz: str = None
) -> Dict[str, Dict[str, Any]]:
    """
    Extracts mean trajectories, demanded joint moments (N*m/kg),
    and operating angle boundaries for Young Controls, Elderly Controls, and Stroke Paretic
    from the PersonalMTG population dataset.
    """
    base_dir = os.path.dirname(os.path.abspath(__file__))
    if features_csv is None:
        features_csv = os.path.join(base_dir, "data", "population_features.csv")
    if trajectories_npz is None:
        trajectories_npz = os.path.join(base_dir, "data", "population_trajectories.npz")

    if not os.path.exists(features_csv) or not os.path.exists(trajectories_npz):
        return {}

    young_subs, elderly_subs, stroke_subs = [], [], []
    with open(features_csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["Group"] == "Healthy" and row["AgeGroup"] == "Young":
                young_subs.append(row["Subject"])
            elif row["Group"] == "Healthy" and row["AgeGroup"] == "Elderly":
                elderly_subs.append(row["Subject"])
            elif row["Group"] == "Stroke_Paretic":
                stroke_subs.append(row["Subject"])

    trajs = np.load(trajectories_npz)
    cohorts = {
        "Young": ("Healthy_", young_subs),
        "Elderly": ("Healthy_", elderly_subs),
        "Stroke_Paretic": ("StrokeP_", stroke_subs)
    }

    op_data = {}
    for c_name, (prefix, subs) in cohorts.items():
        op_data[c_name] = {}
        for joint, ang_var, mom_var in [
            ("ankle", "AnkleAngles", "AnkleMoment"),
            ("knee", "KneeAngles", "KneeMoment"),
            ("hip", "HipAngles", "HipMoment")
        ]:
            ang_curves = [trajs[f"{prefix}{s}_{ang_var}"] for s in subs if f"{prefix}{s}_{ang_var}" in trajs]
            mom_curves = [trajs[f"{prefix}{s}_{mom_var}"] for s in subs if f"{prefix}{s}_{mom_var}" in trajs]
            if len(ang_curves) > 0:
                arr = np.array(ang_curves)
                mean_curve = np.mean(arr, axis=0)
                op_data[c_name][joint] = {
                    "mean_angle": mean_curve,
                    "std_angle": np.std(arr, axis=0),
                    "min_ang": float(np.min(mean_curve)),
                    "max_ang": float(np.max(mean_curve)),
                    "rom": float(np.max(mean_curve) - np.min(mean_curve)),
                }
                if len(mom_curves) > 0:
                    m_arr = np.array(mom_curves)
                    op_data[c_name][joint]["mean_moment"] = np.mean(m_arr, axis=0)
                    op_data[c_name][joint]["std_moment"] = np.std(m_arr, axis=0)
    return op_data


_CACHED_ELDERLY_SCALING = None


def fit_elderly_population_mtg_scaling(
    stats_csv_path: str = None,
    features_csv_path: str = None,
    trajectories_npz_path: str = None,
    use_cache: bool = True
) -> Dict[str, Any]:
    """
    Runs the Millard et al. (2019) MTG personalization pipeline on the Elderly Adult
    cohort (Sub01-Sub33, ages 70-86 yrs) vs Young Adult reference cohort (Sub111-Sub138,
    ages 20-29 yrs) and returns the 7-DoF scaling vectors for the 10-DoF MuJoCo biped:
      [Lumbar, Hip_L, Knee_L, Ankle_L, Hip_R, Knee_R, Ankle_R]
    """
    global _CACHED_ELDERLY_SCALING
    if use_cache and _CACHED_ELDERLY_SCALING is not None and stats_csv_path is None:
        return _CACHED_ELDERLY_SCALING
    base_dir = os.path.dirname(os.path.abspath(__file__))
    if stats_csv_path is None:
        stats_csv_path = os.path.join(base_dir, "data", "stats_young_vs_elderly.csv")
    if features_csv_path is None:
        features_csv_path = os.path.join(base_dir, "data", "population_features.csv")
    if trajectories_npz_path is None:
        trajectories_npz_path = os.path.join(base_dir, "data", "population_trajectories.npz")

    # 1. Parse cohort-level biomechanical percent differences from stats_young_vs_elderly.csv
    pct_diff = {}
    if os.path.exists(stats_csv_path):
        with open(stats_csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                pct_diff[row["Feature"]] = float(row["Percent_Diff"])

    pop_params = get_population_mtg_parameters()["Elderly"]

    knee_ext_ratio = 1.0 + pct_diff.get("Knee_Peak_Ext_Moment", -34.08) / 100.0
    ankle_pf_ratio = 1.0 + pct_diff.get("Ankle_Peak_Power_Generation_A2", -23.30) / 100.0
    hip_ext_ratio = 1.0 + pct_diff.get("Hip_Peak_Power_Generation_H3", -20.775) / 100.0
    hip_flex_ratio = 1.0 - abs(pct_diff.get("Hip_Peak_Flex_Moment", 21.30)) / 100.0
    knee_flex_ratio = pop_params["knee_flex"]["s_s"]
    lumbar_ratio = 0.800

    # 2. Run MuscleFittingOptimizer on cohort mean trajectories if available
    millard_fits = {}
    op_data = load_population_operating_ranges(features_csv_path, trajectories_npz_path)
    if "Young" in op_data and "Elderly" in op_data:
        for cohort_name in ["Young", "Elderly"]:
            c_joints = op_data[cohort_name]
            if "ankle" in c_joints and "mean_moment" in c_joints["ankle"]:
                th_ank = c_joints["ankle"]["mean_angle"]
                n_pts = len(th_ank)
                time_gait = np.linspace(0.0, 1.0, n_pts)
                om_ank = compute_angular_velocity(time_gait, th_ank, smooth_window=5)
                mom_ank = c_joints["ankle"]["mean_moment"] * REF_BODY_MASS
                for mtg_id, sign_dir in [("ankle_plantar", 1), ("ankle_dorsi", -1)]:
                    tau_exp = extract_experimental_torque(mom_ank, sign_dir, remove_outliers=True, n_std=2.5)
                    peak_dem = float(np.max(tau_exp))
                    muscle = create_default_lower_limb_mtgs(knee_flexion_positive=True)[mtg_id]
                    muscle.tau_o = max(peak_dem * 0.85, 10.0)
                    opt = MuscleFittingOptimizer(muscle, a_max=1.0, t_pe_max=1.4)
                    fit_res = opt.fit(th_ank, om_ank, tau_exp, allow_weaker=True, subsample_step=10)
                    millard_fits[f"{mtg_id}_{cohort_name}"] = {
                        "s_A": float(fit_res["s_A"]),
                        "s_V": float(fit_res["s_V"]),
                        "delta_P": float(fit_res["delta_P"]),
                        "lambda_P": float(fit_res["lambda_P"]),
                        "s_s": float(fit_res["s_s"]),
                        "effective_tau_o": float(fit_res["s_s"] * muscle.tau_o)
                    }

    # 3. Assemble 7-DoF MuJoCo biped personalization vectors:
    #    [Lumbar, Hip_L, Knee_L, Ankle_L, Hip_R, Knee_R, Ankle_R]
    s_s_pos = np.array([
        lumbar_ratio,
        round(hip_ext_ratio, 5),
        round(knee_ext_ratio, 4),
        round(ankle_pf_ratio, 4),
        round(hip_ext_ratio, 5),
        round(knee_ext_ratio, 4),
        round(ankle_pf_ratio, 4)
    ], dtype=np.float32)

    s_s_neg = np.array([
        lumbar_ratio,
        round(hip_flex_ratio, 5),
        knee_flex_ratio,
        round(ankle_pf_ratio, 4),
        round(hip_flex_ratio, 5),
        knee_flex_ratio,
        round(ankle_pf_ratio, 4)
    ], dtype=np.float32)

    s_A = np.array([
        0.94,
        pop_params["hip_ext"]["s_A"],
        pop_params["knee_ext"]["s_A"],
        pop_params["ankle_plantar"]["s_A"],
        pop_params["hip_ext"]["s_A"],
        pop_params["knee_ext"]["s_A"],
        pop_params["ankle_plantar"]["s_A"]
    ], dtype=np.float32)

    s_V = np.array([
        0.90,
        pop_params["hip_ext"]["s_V"],
        pop_params["knee_ext"]["s_V"],
        pop_params["ankle_plantar"]["s_V"],
        pop_params["hip_ext"]["s_V"],
        pop_params["knee_ext"]["s_V"],
        pop_params["ankle_plantar"]["s_V"]
    ], dtype=np.float32)

    delta_P_rad = np.radians(np.array([
        0.0,
        pop_params["hip_ext"]["delta_P"],
        pop_params["knee_ext"]["delta_P"],
        pop_params["ankle_plantar"]["delta_P"],
        pop_params["hip_ext"]["delta_P"],
        pop_params["knee_ext"]["delta_P"],
        pop_params["ankle_plantar"]["delta_P"]
    ], dtype=np.float32))
    k_pass_scale = 1.15

    result = {
        "s_s_pos": s_s_pos,
        "s_s_neg": s_s_neg,
        "s_A": s_A,
        "s_V": s_V,
        "delta_P_rad": delta_P_rad,
        "k_pass_scale": k_pass_scale,
        "tau_act": 0.024,
        "tau_deact": 0.048,
        "millard_fits": millard_fits
    }
    if use_cache and stats_csv_path is os.path.join(base_dir, "data", "stats_young_vs_elderly.csv") or stats_csv_path is not None:
        _CACHED_ELDERLY_SCALING = result
    return result
