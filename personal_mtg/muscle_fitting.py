"""
Optimization engine for personalizing MTG strength and flexibility curves.
Formulates and solves the quadratic programming / constrained optimization problem
from Millard et al. (2019) Section 2.2.
"""

import numpy as np
from scipy.optimize import minimize, OptimizeResult
from typing import Dict, Any, Optional
from .torque_muscle import MillardTorqueMuscle


class MuscleFittingOptimizer:
    """
    Solves for x = [s^A, s^V, Delta^P, lambda^P, s^s]
    min (x - x_bar)^T W (x - x_bar)
    subject to:
      1. tau^M(a_max, theta, omega, x) >= tau_exp (Active capacity)
      2. tau_exp >= tau^M(0, theta, omega, x)       (Passive limit: min activation >= 0)
      3. t_pe_max >= t^PE(theta; Delta^P, lambda^P) (Passive multiplier limit <= 1.4)
    """
    def __init__(
        self,
        muscle: MillardTorqueMuscle,
        weights: Optional[np.ndarray] = None,
        a_max: float = 1.0,
        t_pe_max: float = 1.4
    ):
        self.muscle = muscle
        self.a_max = float(a_max)
        self.t_pe_max = float(t_pe_max)
        self.x_default = np.array([1.0, 1.0, 0.0, 0.0, 1.0], dtype=np.float64)

        if weights is None:
            self.W = np.diag([1.0, 1.0, 0.01, 1.0, 1.0])
        else:
            self.W = np.diag(weights) if len(weights.shape) == 1 else weights

    def fit(
        self,
        theta: np.ndarray,
        omega: np.ndarray,
        tau_exp: np.ndarray,
        allow_weaker: bool = False,
        subsample_step: int = 1
    ) -> Dict[str, Any]:
        if subsample_step > 1:
            th = theta[::subsample_step]
            om = omega[::subsample_step]
            te = tau_exp[::subsample_step]
        else:
            th = theta
            om = omega
            te = tau_exp

        initial_act = self.muscle.calc_activation(te, th, om, self.x_default)
        initial_feasible = bool(np.all((initial_act >= -1e-3) & (initial_act <= self.a_max + 1e-3)))

        def objective(x):
            diff = x - self.x_default
            return float(diff.T @ self.W @ diff)

        def objective_jac(x):
            return 2.0 * (self.W @ (x - self.x_default))

        def constr_active(x):
            tau_max_cap = self.muscle.calc_torque(self.a_max, th, om, x)
            return tau_max_cap - te

        def constr_passive_min(x):
            tau_min_passive = self.muscle.calc_torque(0.0, th, om, x)
            return te - tau_min_passive

        def constr_passive_max(x):
            delta_p, lambda_p = x[2], x[3]
            tpe = self.muscle.calc_tPE(th, delta_p, lambda_p)
            return self.t_pe_max - tpe

        constraints = [
            {'type': 'ineq', 'fun': constr_active},
            {'type': 'ineq', 'fun': constr_passive_min},
            {'type': 'ineq', 'fun': constr_passive_max}
        ]

        s_s_min = 0.1 if allow_weaker else 1.0
        s_a_min = 0.5 if allow_weaker else 1.0
        s_v_min = 0.5 if allow_weaker else 1.0

        bounds = [
            (s_a_min, 3.0),     # s^A
            (s_v_min, 3.0),     # s^V
            (-60.0, 60.0),      # Delta^P (deg)
            (0.0, 0.99),        # lambda^P
            (s_s_min, 5.0)      # s^s
        ]

        x0 = np.copy(self.x_default)
        if not allow_weaker and np.any(te > self.muscle.tau_o):
            x0[4] = float(np.max(te) / self.muscle.tau_o * 1.05)

        res: OptimizeResult = minimize(
            fun=objective,
            x0=x0,
            jac=objective_jac,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': 300, 'ftol': 1e-6, 'disp': False}
        )

        fitted_params = res.x
        final_act = self.muscle.calc_activation(te, th, om, fitted_params)
        final_feasible = bool(np.all((final_act >= -1e-2) & (final_act <= self.a_max + 1e-2)))

        return {
            'success': res.success,
            'message': res.message,
            'initial_feasible': initial_feasible,
            'final_feasible': final_feasible,
            'params_initial': self.x_default.copy(),
            'params_fitted': fitted_params,
            's_A': fitted_params[0],
            's_V': fitted_params[1],
            'delta_P': fitted_params[2],
            'lambda_P': fitted_params[3],
            's_s': fitted_params[4],
            'initial_activation': initial_act,
            'fitted_activation': final_act,
            'objective_value': res.fun
        }

