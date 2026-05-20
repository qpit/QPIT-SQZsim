"""Mode-matching and focusing helpers for crystal simulations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from common.constants import PI
from .crystal_boyd_kleinman import boyd_kleinman_efficiency, compute_focusing_parameter


@dataclass(frozen=True)
class ModeMatchingContext:
    """Cavity-derived context needed for crystal mode-matching metrics."""

    geometry: str | None
    crystal_length_m: float | None
    wavelength_m: float | None
    n_crystal: float | None
    waist_crystal_m: float | None
    q_sagittal: complex | None
    q_tangential: complex | None
    cavity_data: dict[str, Any]


@dataclass(frozen=True)
class ModeMatchingResult:
    """Derived focusing and simple overlap summary."""

    waist_crystal_m: float
    rayleigh_range_m: float
    confocal_parameter_m: float
    focusing_parameter: float
    focusing_parameter_xi: float
    boyd_kleinman_factor: float
    effective_nonlinear_overlap: float


def rayleigh_range_in_medium(w0_m: float, wavelength_m: float, n: float) -> float:
    """Return Rayleigh range in medium ``zR = π n w0^2 / λ``."""
    return float(PI * n * w0_m * w0_m / wavelength_m)


def focusing_parameter(crystal_length_m: float, zR_m: float) -> float:
    """Return normalized focusing parameter ``ξ = L/(2 zR)``."""
    return float(crystal_length_m / (2.0 * zR_m))


def build_mode_matching_context_from_cavity_output(cavity_data: dict) -> ModeMatchingContext:
    """Build a mode-matching context from cavity simulation JSON output."""
    inputs = cavity_data.get("inputs", {})
    results = cavity_data.get("results", {})

    q_s = results.get("q_sagittal")
    q_t = results.get("q_tangential")

    q_s_complex = None
    if isinstance(q_s, dict) and "real" in q_s and "imag" in q_s:
        q_s_complex = complex(float(q_s["real"]), float(q_s["imag"]))

    q_t_complex = None
    if isinstance(q_t, dict) and "real" in q_t and "imag" in q_t:
        q_t_complex = complex(float(q_t["real"]), float(q_t["imag"]))

    waist_um = results.get("beam_waist_crystal_um")
    return ModeMatchingContext(
        geometry=cavity_data.get("geometry"),
        crystal_length_m=_opt_float(inputs.get("crystal_length_m")),
        wavelength_m=_opt_float(inputs.get("wavelength_m")),
        n_crystal=_opt_float(inputs.get("n_crystal")),
        waist_crystal_m=None if waist_um is None else float(waist_um) * 1e-6,
        q_sagittal=q_s_complex,
        q_tangential=q_t_complex,
        cavity_data=cavity_data,
    )


def estimate_mode_matching_quantities(
    waist_crystal_m: float,
    crystal_length_m: float,
    wavelength_m: float,
    n_crystal: float,
    delta_k_rad_per_m: float = 0.0,
) -> ModeMatchingResult:
    """Compute core focusing quantities from waist and crystal parameters."""
    zR = rayleigh_range_in_medium(waist_crystal_m, wavelength_m, n_crystal)
    confocal = 2.0 * zR
    xi = compute_focusing_parameter(crystal_length_m, zR)
    bk = boyd_kleinman_efficiency(
        rayleigh_range_m=zR,
        crystal_length_m=crystal_length_m,
        delta_k=delta_k_rad_per_m,
    )
    overlap = float(np.sqrt(max(bk, 0.0)))
    return ModeMatchingResult(
        waist_crystal_m=float(waist_crystal_m),
        rayleigh_range_m=float(zR),
        confocal_parameter_m=float(confocal),
        focusing_parameter=float(xi),
        focusing_parameter_xi=float(xi),
        boyd_kleinman_factor=float(bk),
        effective_nonlinear_overlap=float(overlap),
    )


def _opt_float(x: Any) -> float | None:
    if x is None:
        return None
    return float(x)


__all__ = [
    "ModeMatchingContext",
    "ModeMatchingResult",
    "rayleigh_range_in_medium",
    "focusing_parameter",
    "build_mode_matching_context_from_cavity_output",
    "estimate_mode_matching_quantities",
]
