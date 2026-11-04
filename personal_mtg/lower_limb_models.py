"""
Standard lower-limb Muscle-Torque-Generator (MTG) definitions for Hip, Knee, and Ankle.
Parameters extracted from Table A.2 of Millard et al. (2019) and Anderson et al. (2007).
"""

from typing import Dict
from .torque_muscle import MillardTorqueMuscle


def create_default_lower_limb_mtgs(knee_flexion_positive: bool = True) -> Dict[str, MillardTorqueMuscle]:
    """
    Creates the default 6 MTGs for the lower extremity:
      - hip_ext, hip_flex, knee_ext, knee_flex, ankle_plantar, ankle_dorsi
    """
    mtgs = {}

    mtgs["hip_ext"] = MillardTorqueMuscle(
        name="Hip_Extensor",
        tau_o=176.0,
        theta_o=61.0,
        theta_w=61.0,
        omega_max=517.0,
        tv_half=0.16,
        theta_0_pe=45.0,
        theta_1_pe=90.0,
        direction=-1,
        concentric_sign=-1
    )

    mtgs["hip_flex"] = MillardTorqueMuscle(
        name="Hip_Flexor",
        tau_o=157.0,
        theta_o=-42.0,
        theta_w=70.0,
        omega_max=526.0,
        tv_half=0.25,
        theta_0_pe=-5.0,
        theta_1_pe=-30.0,
        direction=1,
        concentric_sign=1
    )

    if knee_flexion_positive:
        mtgs["knee_ext"] = MillardTorqueMuscle(
            name="Knee_Extensor",
            tau_o=286.0,
            theta_o=54.0,
            theta_w=29.0,
            omega_max=1101.0,
            tv_half=0.12,
            theta_0_pe=115.0,
            theta_1_pe=155.0,
            direction=1,
            concentric_sign=-1
        )
        mtgs["knee_flex"] = MillardTorqueMuscle(
            name="Knee_Flexor",
            tau_o=99.0,
            theta_o=59.0,
            theta_w=64.0,
            omega_max=953.0,
            tv_half=0.20,
            theta_0_pe=0.0,
            theta_1_pe=-15.0,
            direction=-1,
            concentric_sign=1
        )
    else:
        mtgs["knee_ext"] = MillardTorqueMuscle(
            name="Knee_Extensor",
            tau_o=286.0,
            theta_o=-54.0,
            theta_w=29.0,
            omega_max=1101.0,
            tv_half=0.12,
            theta_0_pe=-115.0,
            theta_1_pe=-155.0,
            direction=1,
            concentric_sign=1
        )
        mtgs["knee_flex"] = MillardTorqueMuscle(
            name="Knee_Flexor",
            tau_o=99.0,
            theta_o=-59.0,
            theta_w=64.0,
            omega_max=953.0,
            tv_half=0.20,
            theta_0_pe=0.0,
            theta_1_pe=15.0,
            direction=-1,
            concentric_sign=-1
        )

    mtgs["ankle_plantar"] = MillardTorqueMuscle(
        name="Ankle_Plantarflexor",
        tau_o=128.0,
        theta_o=23.0,
        theta_w=38.0,
        omega_max=674.0,
        tv_half=0.41,
        theta_0_pe=2.0,
        theta_1_pe=45.0,
        direction=-1,
        concentric_sign=-1
    )

    mtgs["ankle_dorsi"] = MillardTorqueMuscle(
        name="Ankle_Dorsiflexor",
        tau_o=44.0,
        theta_o=-11.0,
        theta_w=35.0,
        omega_max=990.0,
        tv_half=0.11,
        theta_0_pe=-30.0,
        theta_1_pe=-60.0,
        direction=1,
        concentric_sign=1
    )

    return mtgs

