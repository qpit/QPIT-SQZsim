"""Shared physical and mathematical constants used across simulation modules."""

from __future__ import annotations

import numpy as np


C_M_PER_S = 299792458.0  # Speed of light in vacuum [m/s]

# Mathematical constants
PI = np.pi  # Archimedes' constant pi [-]
TWO_PI = 2.0 * PI  # Two pi [-]

# Electromagnetic constants
EPSILON_0_F_PER_M = 8.8541878128e-12  # Vacuum permittivity [F/m]
MU_0_H_PER_M = 1.25663706212e-6  # Vacuum permeability [H/m]
Z0_OHM = 376.730313668  # Vacuum impedance [ohm]

# Quantum constants
H_J_S = 6.62607015e-34  # Planck constant [J s]
HBAR_J_S = 1.054571817e-34  # Reduced Planck constant [J s]

# Fundamental constants
K_B_J_PER_K = 1.380649e-23  # Boltzmann constant [J/K]
E_CHARGE_C = 1.602176634e-19  # Elementary charge [C]

# Common aliases
EPS0 = EPSILON_0_F_PER_M
H = H_J_S
HBAR = HBAR_J_S

# Unit conversions
NM_TO_M = 1e-9
UM_TO_M = 1e-6
MM_TO_M = 1e-3
DEG_TO_RAD = PI / 180.0


__all__ = [
    "C_M_PER_S",
    "PI",
    "TWO_PI",
    "EPSILON_0_F_PER_M",
    "MU_0_H_PER_M",
    "Z0_OHM",
    "H_J_S",
    "HBAR_J_S",
    "K_B_J_PER_K",
    "E_CHARGE_C",
    "EPS0",
    "H",
    "HBAR",
    "NM_TO_M",
    "UM_TO_M",
    "MM_TO_M",
    "DEG_TO_RAD",
]
