"""
PersonalMTG package for Millard et al. (2019) 5-parameter Muscle-Torque-Generator (MTG)
personalization and Elderly Adult cohort scaling.
Copied from PersonalMTG/src and integrated with the 10-DoF MuJoCo RL biped.
"""

from .spline_factory import (
    SmoothFunction,
    ActiveTorqueAngleCurve,
    TorqueVelocityCurve,
    PassiveTorqueAngleCurve,
    refit_with_smooth_spline
)
from .torque_muscle import MillardTorqueMuscle
from .lower_limb_models import create_default_lower_limb_mtgs
from .muscle_fitting import MuscleFittingOptimizer
from .opensim_loader import (
    load_opensim_file,
    compute_angular_velocity,
    extract_experimental_torque,
    filter_torque_outliers
)
from .visualization import plot_mtg_personalization_results
from .elderly_personalizer import fit_elderly_population_mtg_scaling

