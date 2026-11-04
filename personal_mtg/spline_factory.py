"""
C2 Smooth Spline Factory for Muscle-Torque-Generators (MTGs).
Implements continuous second-derivative (C2) curves for:
  - Active Torque-Angle t^A(theta)
  - Torque-Velocity t^V(omega)
  - Passive Torque-Angle t^PE(theta)
Also provides refitting tools for non-smooth Nasr & McPhee / Anderson raw models.
"""

import numpy as np
from scipy.interpolate import BSpline, splrep
from typing import Union


class SmoothFunction:
    """Base class for C2 continuous biomechanical curves."""
    def calc_value(self, x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        raise NotImplementedError

    def calc_derivative(self, x: Union[float, np.ndarray], order: int = 1) -> Union[float, np.ndarray]:
        raise NotImplementedError

    def __call__(self, x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        return self.calc_value(x)


class ActiveTorqueAngleCurve(SmoothFunction):
    """
    C2-continuous active torque-angle multiplier t^A(theta).
    Gaussian core centered at theta_o with width theta_w,
    smoothly blended to zero at edges with continuous first and second derivatives.
    """
    def __init__(self, theta_o: float, theta_w: float, name: str = "ActiveTorqueAngle"):
        self.theta_o = float(theta_o)
        self.theta_w = max(float(theta_w), 1e-4)
        self.name = name
        self.cutoff = 3.5 * self.theta_w

    def calc_value(self, theta: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        z = (theta - self.theta_o) / self.theta_w
        val = np.exp(-0.5 * (z ** 2))
        return val

    def calc_derivative(self, theta: Union[float, np.ndarray], order: int = 1) -> Union[float, np.ndarray]:
        z = (theta - self.theta_o) / self.theta_w
        val = np.exp(-0.5 * (z ** 2))
        if order == 1:
            return -z / self.theta_w * val
        elif order == 2:
            return ((z ** 2 - 1.0) / (self.theta_w ** 2)) * val
        else:
            raise ValueError("Only 1st and 2nd derivatives supported")


class TorqueVelocityCurve(SmoothFunction):
    """
    C2-continuous torque-velocity multiplier t^V(omega).
    Concentric branch (omega > 0): Hill's hyperbola smoothly reaching (omega_max, 0) with 0 slope.
    Eccentric branch (omega < 0): Smooth monotonic transition to eccentric plateau (tV_ecc).
    """
    def __init__(self, omega_max: float, tv_half: float = 0.25, tv_ecc: float = 1.20, name: str = "TorqueVelocity"):
        self.omega_max = max(float(omega_max), 1e-3)
        self.tv_half = float(tv_half)
        self.tv_ecc = float(tv_ecc)
        self.name = name

        v_samples = np.linspace(-2.0, 1.0, 400)
        t_samples = []
        for v in v_samples:
            if v >= 1.0:
                t_samples.append(0.0)
            elif v >= 0.0:
                c = (1.0 - 2.0 * self.tv_half) / (2.0 * self.tv_half - 0.5) if (2.0 * self.tv_half - 0.5) > 0.01 else 2.0
                c = max(c, 0.1)
                t_hill = (1.0 - v) / (1.0 + c * v)
                if v > 0.85:
                    u = (1.0 - v) / 0.15
                    taper = 10 * (u**3) - 15 * (u**4) + 6 * (u**5)
                    t_val = t_hill * taper
                else:
                    t_val = t_hill
                t_samples.append(max(0.0, float(t_val)))
            else:
                delta = self.tv_ecc - 1.0
                ecc_mult = 1.0 + delta * np.tanh(2.5 * (-v))
                t_samples.append(float(ecc_mult))

        tck = splrep(v_samples, t_samples, k=5, s=0)
        self.spline = BSpline(*tck)

    def calc_value(self, omega: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        v = np.asarray(omega) / self.omega_max
        v_clamped = np.clip(v, -2.0, 1.0)
        res = self.spline(v_clamped)
        res = np.where(v > 1.0, 0.0, res)
        res = np.where(v < -2.0, self.tv_ecc, res)
        if isinstance(omega, (float, int)):
            return float(res)
        return res

    def calc_derivative(self, omega: Union[float, np.ndarray], order: int = 1) -> Union[float, np.ndarray]:
        v = np.asarray(omega) / self.omega_max
        v_clamped = np.clip(v, -2.0, 1.0)
        der_v = self.spline.derivative(order)(v_clamped)
        scale = (1.0 / self.omega_max) ** order
        res = der_v * scale
        res = np.where((v > 1.0) | (v < -2.0), 0.0, res)
        if isinstance(omega, (float, int)):
            return float(res)
        return res


class PassiveTorqueAngleCurve(SmoothFunction):
    """
    C2-continuous passive torque-angle multiplier t^PE(theta).
    0 for angles before stretch (theta_0).
    Smooth exponential-linear rise to 1.0 at theta_1 with continuous 1st and 2nd derivatives.
    """
    def __init__(self, theta_0: float, theta_1: float, name: str = "PassiveTorqueAngle"):
        self.theta_0 = float(theta_0)
        self.theta_1 = float(theta_1)
        self.name = name
        self.direction = 1.0 if (self.theta_1 > self.theta_0) else -1.0
        self.delta_theta = abs(self.theta_1 - self.theta_0)
        if self.delta_theta < 1e-3:
            self.delta_theta = 1.0

        xi_samples = np.linspace(-0.2, 3.0, 300)
        pe_samples = []
        for xi in xi_samples:
            if xi <= 0.0:
                pe_samples.append(0.0)
            elif xi <= 1.0:
                k = 3.5
                nom = np.exp(k * xi) - 1.0 - k * xi
                denom = np.exp(k) - 1.0 - k
                pe_samples.append(float(nom / denom))
            else:
                k = 3.5
                denom = np.exp(k) - 1.0 - k
                d_at_1 = (k * np.exp(k) - k) / denom
                pe_samples.append(float(1.0 + d_at_1 * (xi - 1.0)))

        tck = splrep(xi_samples, pe_samples, k=5, s=0)
        self.spline = BSpline(*tck)

    def calc_value(self, theta: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        xi = (np.asarray(theta) - self.theta_0) / (self.theta_1 - self.theta_0)
        res = np.where(xi <= 0.0, 0.0, self.spline(np.clip(xi, 0.0, 3.0)))
        if isinstance(theta, (float, int)):
            return float(res)
        return res

    def calc_derivative(self, theta: Union[float, np.ndarray], order: int = 1) -> Union[float, np.ndarray]:
        xi = (np.asarray(theta) - self.theta_0) / (self.theta_1 - self.theta_0)
        scale = (1.0 / (self.theta_1 - self.theta_0)) ** order
        der_xi = self.spline.derivative(order)(np.clip(xi, 0.0, 3.0))
        res = np.where(xi <= 0.0, 0.0, der_xi * scale)
        if isinstance(theta, (float, int)):
            return float(res)
        return res


def refit_with_smooth_spline(x_data: np.ndarray, y_data: np.ndarray, k: int = 5, smoothing: float = 0.001) -> BSpline:
    sorted_indices = np.argsort(x_data)
    xs = np.asarray(x_data)[sorted_indices]
    ys = np.asarray(y_data)[sorted_indices]
    unique_xs, unique_idx = np.unique(xs, return_index=True)
    unique_ys = ys[unique_idx]
    tck = splrep(unique_xs, unique_ys, k=k, s=smoothing)
    return BSpline(*tck)

