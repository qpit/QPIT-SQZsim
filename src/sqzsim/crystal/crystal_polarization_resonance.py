"""Crystal-side cavity resonance diagnostics for polarization-resolved fields."""

from __future__ import annotations

from typing import Any, Callable

import numpy as np

from common.constants import PI, TWO_PI

DEFAULT_RESONANCE_TOLERANCE_RAD = 1.0e-2


def _wrap_phase_to_pi(phase_rad: float) -> float:
    """Wrap a phase angle into the interval [-pi, pi]."""
    return float((phase_rad + PI) % TWO_PI - PI)


def _resolve_roundtrip_lengths(cavity_data: dict[str, Any]) -> tuple[float, float]:
    """Return geometric and crystal round-trip lengths using cavity exports."""
    results = cavity_data.get("results", {})
    inputs = cavity_data.get("inputs", {})

    geometric_roundtrip_length_m = results.get("cavity_length_m")
    if geometric_roundtrip_length_m is None:
        raise ValueError("Cavity output missing results.cavity_length_m")

    crystal_roundtrip_length_m = results.get("optical_crystal_length_m", inputs.get("crystal_length_m"))
    if crystal_roundtrip_length_m is None:
        raise ValueError("Cavity output missing crystal round-trip length information")

    return float(geometric_roundtrip_length_m), float(crystal_roundtrip_length_m)


def compute_polarization_resonance_diagnostic(
    cavity_data: dict[str, Any],
    temperature_K: float,
    signal_axis: str,
    idler_axis: str,
    wavelength_s_m: float,
    wavelength_i_m: float,
    n_s_of_lambda_T: Callable[[float, float], float],
    n_i_of_lambda_T: Callable[[float, float], float],
    resonance_tolerance_rad: float = DEFAULT_RESONANCE_TOLERANCE_RAD,
) -> dict[str, float | str | bool]:
    """Compute a compact polarization-resolved cavity resonance diagnostic.

    The cavity layer already exports the round-trip geometric length and the
    crystal round-trip path length. This helper reuses those quantities and only
    swaps in the polarization-dependent crystal refractive index on the crystal
    side; it does not rebuild any cavity model downstream.
    """
    geometric_roundtrip_length_m, crystal_roundtrip_length_m = _resolve_roundtrip_lengths(cavity_data)
    free_space_roundtrip_length_m = geometric_roundtrip_length_m - crystal_roundtrip_length_m

    n_signal = float(n_s_of_lambda_T(wavelength_s_m, temperature_K))
    n_idler = float(n_i_of_lambda_T(wavelength_i_m, temperature_K))

    signal_optical_roundtrip_length_m = free_space_roundtrip_length_m + n_signal * crystal_roundtrip_length_m
    idler_optical_roundtrip_length_m = free_space_roundtrip_length_m + n_idler * crystal_roundtrip_length_m

    phi_signal_rad = float((TWO_PI / wavelength_s_m) * signal_optical_roundtrip_length_m)
    phi_idler_rad = float((TWO_PI / wavelength_i_m) * idler_optical_roundtrip_length_m)
    delta_phi_rad = float(phi_signal_rad - phi_idler_rad)
    delta_phi_wrapped_rad = _wrap_phase_to_pi(delta_phi_rad)

    payload: dict[str, float | str | bool] = {
        "temperature_K": float(temperature_K),
        "signal_axis": str(signal_axis),
        "idler_axis": str(idler_axis),
        "signal_wavelength_m": float(wavelength_s_m),
        "idler_wavelength_m": float(wavelength_i_m),
        "n_signal": n_signal,
        "n_idler": n_idler,
        "geometric_roundtrip_length_m": float(geometric_roundtrip_length_m),
        "crystal_roundtrip_length_m": float(crystal_roundtrip_length_m),
        "free_space_roundtrip_length_m": float(free_space_roundtrip_length_m),
        "signal_optical_roundtrip_length_m": float(signal_optical_roundtrip_length_m),
        "idler_optical_roundtrip_length_m": float(idler_optical_roundtrip_length_m),
        "phi_signal_rad": phi_signal_rad,
        "phi_idler_rad": phi_idler_rad,
        "delta_phi_rad": delta_phi_rad,
        "delta_phi_wrapped_rad": delta_phi_wrapped_rad,
        "resonance_tolerance_rad": float(resonance_tolerance_rad),
        "is_double_resonant": bool(abs(delta_phi_wrapped_rad) < resonance_tolerance_rad),
    }

    c_m_per_s = cavity_data.get("inputs", {}).get(
        "c_m_per_s",
        cavity_data.get("constants", {}).get("c_m_per_s"),
    )
    if c_m_per_s is not None:
        c_value = float(c_m_per_s)
        fsr_signal_hz = c_value / signal_optical_roundtrip_length_m
        fsr_idler_hz = c_value / idler_optical_roundtrip_length_m
        payload["fsr_signal_Hz"] = float(fsr_signal_hz)
        payload["fsr_idler_Hz"] = float(fsr_idler_hz)
        payload["delta_fsr_Hz"] = float(fsr_signal_hz - fsr_idler_hz)

    return payload


__all__ = [
    "DEFAULT_RESONANCE_TOLERANCE_RAD",
    "compute_polarization_resonance_diagnostic",
]
