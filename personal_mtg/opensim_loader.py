"""
OpenSim file loader and kinematics/dynamics preprocessing for clinical gait data.
Supports .mot (motion/kinematics) and .sto (storage/moments) files.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
import os


def load_opensim_file(filepath: str) -> Tuple[np.ndarray, List[str], Dict[str, np.ndarray]]:
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")

    metadata = {}
    headers = []
    data_rows = []

    with open(filepath, 'r') as f:
        in_header = True
        for line in f:
            stripped = line.strip()
            if not stripped:
                continue

            if in_header:
                if stripped.lower() == 'endheader':
                    in_header = False
                    continue
                if '=' in stripped:
                    key, val = stripped.split('=', 1)
                    metadata[key.strip()] = val.strip()
                continue

            if not headers:
                headers = stripped.split('\t') if '\t' in stripped else stripped.split()
                headers = [h.strip() for h in headers if h.strip()]
                continue

            row_vals = stripped.split('\t') if '\t' in stripped else stripped.split()
            data_rows.append([float(v) for v in row_vals])

    data_arr = np.array(data_rows, dtype=np.float64)
    data_dict = {}
    for i, h in enumerate(headers):
        data_dict[h] = data_arr[:, i]

    time = data_dict.get('time', np.arange(len(data_rows), dtype=np.float64))
    return time, headers, data_dict


def compute_angular_velocity(time: np.ndarray, angles: np.ndarray, smooth_window: int = 5) -> np.ndarray:
    omega = np.gradient(angles, time)
    if smooth_window > 1 and len(omega) > smooth_window:
        kernel = np.ones(smooth_window) / smooth_window
        omega = np.convolve(omega, kernel, mode='same')
    return omega


def extract_experimental_torque(
    net_joint_moment: np.ndarray,
    direction: int,
    remove_outliers: bool = False,
    n_std: float = 3.0,
    outlier_action: str = "interpolate"
) -> np.ndarray:
    sign_condition = (net_joint_moment * direction) > 0
    tau_exp = np.where(sign_condition, net_joint_moment * direction, 0.0)
    if remove_outliers:
        tau_exp = filter_torque_outliers(tau_exp, n_std=n_std, action=outlier_action)
    return tau_exp


def filter_torque_outliers(
    torque: np.ndarray,
    n_std: float = 3.0,
    action: str = "interpolate",
    window_size: Optional[int] = None
) -> np.ndarray:
    filtered = np.copy(torque)
    n_pts = len(filtered)
    active_mask = filtered > 1e-4

    if not np.any(active_mask):
        return filtered

    is_outlier = np.zeros(n_pts, dtype=bool)
    if n_pts > 20:
        p95 = np.percentile(filtered[min(10, n_pts):], 95)
        for i in range(min(5, n_pts)):
            if filtered[i] > max(1.5 * p95, 10.0):
                is_outlier[i] = True

    if window_size is not None and window_size > 3:
        half_w = window_size // 2
        for i in range(n_pts):
            idx_start = max(0, i - half_w)
            idx_end = min(n_pts, i + half_w + 1)
            local_pts = filtered[idx_start:idx_end]
            local_active = local_pts[local_pts > 1e-4]
            if len(local_active) > 2:
                loc_med = np.median(local_active)
                loc_std = np.std(local_active)
                cutoff = loc_med + n_std * loc_std
                if filtered[i] > cutoff:
                    is_outlier[i] = True
    else:
        active_vals = filtered[active_mask]
        mean_val = np.mean(active_vals)
        std_val = np.std(active_vals)
        cutoff_threshold = mean_val + n_std * std_val
        is_outlier = is_outlier | (filtered > cutoff_threshold)

    if np.any(is_outlier):
        outlier_indices = np.where(is_outlier)[0]
        valid_indices = np.where(~is_outlier)[0]
        if action == "interpolate" and len(valid_indices) > 1:
            filtered[outlier_indices] = np.interp(outlier_indices, valid_indices, filtered[valid_indices])
        elif action == "clip":
            cutoff = cutoff_threshold if 'cutoff_threshold' in locals() else np.percentile(filtered[valid_indices], 99)
            filtered[outlier_indices] = cutoff
        elif action == "zero":
            filtered[outlier_indices] = 0.0

    return filtered

