"""Linearized quadrature-basis Langevin scaffold for OPO noise calculations.

The present implementation keeps a compact 2x2 state-space model in a
quadrature basis. It remains intentionally minimal, but the two axes are now
treated explicitly as orthogonal quadratures whose mixing is controlled by the
cavity detuning.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

try:
    from ..common.constants import TWO_PI
except ImportError:
    from common.constants import TWO_PI

from .opo_model import OPOModelResult


@dataclass(frozen=True)
class OPOLangevinModel:
    """Container for a linearized 2x2 quadrature-basis Langevin model."""

    quadrature_labels: tuple[str, ...]
    drift_matrix: np.ndarray
    input_matrix: np.ndarray
    noise_coupling_matrix: np.ndarray
    notes: tuple[str, ...]


def build_langevin_model(model: OPOModelResult) -> OPOLangevinModel:
    """Construct a minimal quadrature-basis Langevin model for a degenerate below-threshold OPO."""
    linewidth_hz = max(model.cavity_kappa_total_Hz, 0.0)
    linewidth_rad_s = TWO_PI * linewidth_hz
    detuning_rad_s = TWO_PI * model.cavity_detuning_Hz
    sigma = model.pump_parameter

    # The diagonal entries represent quadrature damping modified by the
    # below-threshold pump parameter. The off-diagonal entries represent
    # quadrature mixing/rotation induced by the cavity detuning.
    drift_matrix = np.array(
        [
            [-(linewidth_rad_s) * (1.0 - sigma), detuning_rad_s],
            [-detuning_rad_s, -(linewidth_rad_s) * (1.0 + sigma)],
        ],
        dtype=float,
    )

    identity = np.eye(2, dtype=float)
    return OPOLangevinModel(
        quadrature_labels=("X", "P"),
        drift_matrix=drift_matrix,
        input_matrix=identity.copy(),
        noise_coupling_matrix=identity,
        notes=(
            "Minimal 2x2 quadrature-basis Langevin model.",
            "The X/P axes represent orthogonal quadrature candidates for squeezing and anti-squeezing.",
            "Cavity detuning rotates and mixes the quadratures through the drift-matrix off-diagonal terms.",
        ),
    )


__all__ = [
    "OPOLangevinModel",
    "build_langevin_model",
]
