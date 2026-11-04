"""
Millard2016 Muscle-Torque-Generator (MTG) model implementation.
Computes active, passive, and total joint torque with 5-parameter scaling
vector x = [s^A, s^V, Delta^P, lambda^P, s^s].
"""

import numpy as np
from typing import Union, Optional
from .spline_factory import ActiveTorqueAngleCurve, TorqueVelocityCurve, PassiveTorqueAngleCurve

EPSILON = 2.220446049250313e-16
DEFAULT_LAMBDA = EPSILON ** 0.25


class MillardTorqueMuscle:
    """
    Reduced Muscle Model / Muscle-Torque-Generator (MTG)
    Reference: Millard et al. (2019) J. Biomech 89:11-20
    """
    def __init__(
        self,
        name: str,
        tau_o: float,
        theta_o: float,
        theta_w: float,
        omega_max: float,
        tv_half: float,
        theta_0_pe: float,
        theta_1_pe: float,
        direction: int = 1,
        concentric_sign: Optional[int] = None,
        beta_pe: float = 0.1,
        lambda_a: float = DEFAULT_LAMBDA,
        lambda_v: float = DEFAULT_LAMBDA
    ):
        self.name = name
        self.tau_o = float(tau_o)
        self.theta_o = float(theta_o)
        self.theta_w = float(theta_w)
        self.omega_max = float(omega_max)
        self.tv_half = float(tv_half)
        self.direction = int(direction)
        self.concentric_sign = int(concentric_sign) if concentric_sign is not None else self.direction
        self.beta_pe = float(beta_pe)
        self.lambda_a = float(lambda_a)
        self.lambda_v = float(lambda_v)

        self.curve_ta = ActiveTorqueAngleCurve(self.theta_o, self.theta_w, name=f"{name}_tA")
        self.curve_tv = TorqueVelocityCurve(self.omega_max, tv_half=tv_half, name=f"{name}_tV")
        self.curve_tpe = PassiveTorqueAngleCurve(theta_0_pe, theta_1_pe, name=f"{name}_tPE")

    def calc_tA(self, theta: Union[float, np.ndarray], s_a: float) -> Union[float, np.ndarray]:
        """Transformed active torque-angle multiplier (Eq. 4)."""
        theta_eff = (np.asarray(theta) - self.theta_o) / s_a + self.theta_o
        base_val = self.curve_ta.calc_value(theta_eff)
        val = self.lambda_a + (1.0 - self.lambda_a) * base_val
        return val

    def calc_tV(self, omega: Union[float, np.ndarray], s_v: float) -> Union[float, np.ndarray]:
        """Transformed torque-velocity multiplier (Eq. 5)."""
        omega_eff = (np.asarray(omega) * self.concentric_sign) / s_v
        base_val = self.curve_tv.calc_value(omega_eff)
        val = self.lambda_v + (1.0 - self.lambda_v) * base_val
        return val

    def calc_tPE(self, theta: Union[float, np.ndarray], delta_p: float, lambda_p: float) -> Union[float, np.ndarray]:
        """Transformed passive torque-angle multiplier (Eq. 6)."""
        theta_eff = np.asarray(theta) - delta_p
        base_val = self.curve_tpe.calc_value(theta_eff)
        val = (1.0 - lambda_p) * base_val
        return val

    def calc_torque(
        self,
        activation: Union[float, np.ndarray],
        theta: Union[float, np.ndarray],
        omega: Union[float, np.ndarray],
        params: Union[np.ndarray, list, tuple] = (1.0, 1.0, 0.0, 0.0, 1.0)
    ) -> Union[float, np.ndarray]:
        """
        Total muscle torque output (Eq. 3).
        params = [s^A, s^V, Delta^P, lambda^P, s^s]
        """
        s_a, s_v, delta_p, lambda_p, s_s = params

        ta = self.calc_tA(theta, s_a)
        tv = self.calc_tV(omega, s_v)
        tpe = self.calc_tPE(theta, delta_p, lambda_p)

        omega_norm = (np.asarray(omega) * self.concentric_sign) / (s_v * self.omega_max)
        damping = 1.0 - self.beta_pe * omega_norm
        damping = np.clip(damping, 0.5, 1.5)

        total_normalized = activation * ta * tv + tpe * damping
        torque = s_s * self.tau_o * total_normalized
        return torque

    def calc_activation(
        self,
        demanded_torque: Union[float, np.ndarray],
        theta: Union[float, np.ndarray],
        omega: Union[float, np.ndarray],
        params: Union[np.ndarray, list, tuple] = (1.0, 1.0, 0.0, 0.0, 1.0)
    ) -> Union[float, np.ndarray]:
        """
        Inverts Eq. (3) to compute required activation given demanded torque.
        """
        s_a, s_v, delta_p, lambda_p, s_s = params

        ta = self.calc_tA(theta, s_a)
        tv = self.calc_tV(omega, s_v)
        tpe = self.calc_tPE(theta, delta_p, lambda_p)

        omega_norm = (np.asarray(omega) * self.concentric_sign) / (s_v * self.omega_max)
        damping = np.clip(1.0 - self.beta_pe * omega_norm, 0.5, 1.5)

        tau_passive = s_s * self.tau_o * tpe * damping
        tau_active = demanded_torque - tau_passive
        denom = s_s * self.tau_o * ta * tv
        denom = np.maximum(denom, 1e-6)

        activation = tau_active / denom
        return activation

