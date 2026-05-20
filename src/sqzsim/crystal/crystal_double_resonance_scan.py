"""Crystal-side 2D scans for polarization double-resonance diagnostics."""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Callable

import numpy as np

from .crystal_polarization_resonance import (
    DEFAULT_RESONANCE_TOLERANCE_RAD,
    compute_polarization_resonance_diagnostic,
)


def _resolve_crystal_roundtrip_scale(cavity_data: dict[str, Any]) -> float:
    """Infer how the cavity export maps physical crystal length to round-trip length."""
    inputs = cavity_data.get("inputs", {})
    results = cavity_data.get("results", {})
    crystal_length_m = inputs.get("crystal_length_m")
    crystal_roundtrip_length_m = results.get("optical_crystal_length_m")
    if crystal_length_m is not None and crystal_roundtrip_length_m is not None and float(crystal_length_m) > 0.0:
        return float(crystal_roundtrip_length_m) / float(crystal_length_m)
    if str(cavity_data.get("geometry", "")).lower() == "monolithic":
        return 2.0
    return 1.0


def _build_length_overridden_cavity_data(cavity_data: dict[str, Any], crystal_length_m: float) -> dict[str, Any]:
    """Return a cavity-data copy with only the round-trip length fields updated.

    This is a local crystal-side diagnostic override for scan purposes only; it
    reuses the exported cavity bookkeeping rather than rebuilding a cavity
    simulation at each scan point.
    """
    cavity_copy = deepcopy(cavity_data)
    inputs = cavity_copy.setdefault("inputs", {})
    results = cavity_copy.setdefault("results", {})

    geometric_roundtrip_length_m = float(results["cavity_length_m"])
    crystal_roundtrip_length_m = float(results["optical_crystal_length_m"])
    free_space_roundtrip_length_m = geometric_roundtrip_length_m - crystal_roundtrip_length_m
    crystal_roundtrip_scale = _resolve_crystal_roundtrip_scale(cavity_data)
    new_crystal_roundtrip_length_m = crystal_roundtrip_scale * float(crystal_length_m)

    inputs["crystal_length_m"] = float(crystal_length_m)
    results["optical_crystal_length_m"] = float(new_crystal_roundtrip_length_m)
    results["cavity_length_m"] = float(free_space_roundtrip_length_m + new_crystal_roundtrip_length_m)
    return cavity_copy


def compute_double_resonance_scan(
    cavity_data: dict[str, Any],
    signal_axis: str,
    idler_axis: str,
    wavelength_s_m: float,
    wavelength_i_m: float,
    n_s_of_lambda_T: Callable[[float, float], float],
    n_i_of_lambda_T: Callable[[float, float], float],
    temperature_min_K: float,
    temperature_max_K: float,
    n_temperature: int,
    crystal_length_min_m: float,
    crystal_length_max_m: float,
    n_crystal_length: int,
    resonance_tolerance_rad: float = DEFAULT_RESONANCE_TOLERANCE_RAD,
) -> dict[str, Any]:
    """Scan wrapped signal-idler resonance mismatch versus temperature and length."""
    if n_temperature < 2:
        raise ValueError("n_temperature must be at least 2")
    if n_crystal_length < 2:
        raise ValueError("n_crystal_length must be at least 2")
    if temperature_max_K <= temperature_min_K:
        raise ValueError("temperature_max_K must be greater than temperature_min_K")
    if crystal_length_max_m <= crystal_length_min_m:
        raise ValueError("crystal_length_max_m must be greater than crystal_length_min_m")

    temperature_grid_K = np.linspace(temperature_min_K, temperature_max_K, n_temperature, dtype=float)
    crystal_length_grid_m = np.linspace(crystal_length_min_m, crystal_length_max_m, n_crystal_length, dtype=float)

    delta_phi_wrapped_rad = np.empty((n_crystal_length, n_temperature), dtype=float)
    abs_delta_phi_wrapped_rad = np.empty_like(delta_phi_wrapped_rad)
    is_double_resonant = np.empty_like(delta_phi_wrapped_rad, dtype=bool)

    for i_length, crystal_length_m in enumerate(crystal_length_grid_m):
        cavity_data_for_length = _build_length_overridden_cavity_data(cavity_data, crystal_length_m)
        for i_temperature, temperature_K in enumerate(temperature_grid_K):
            diagnostic = compute_polarization_resonance_diagnostic(
                cavity_data=cavity_data_for_length,
                temperature_K=float(temperature_K),
                signal_axis=signal_axis,
                idler_axis=idler_axis,
                wavelength_s_m=wavelength_s_m,
                wavelength_i_m=wavelength_i_m,
                n_s_of_lambda_T=n_s_of_lambda_T,
                n_i_of_lambda_T=n_i_of_lambda_T,
                resonance_tolerance_rad=resonance_tolerance_rad,
            )
            delta_phi_wrapped_rad[i_length, i_temperature] = float(diagnostic["delta_phi_wrapped_rad"])
            abs_delta_phi_wrapped_rad[i_length, i_temperature] = abs(delta_phi_wrapped_rad[i_length, i_temperature])
            is_double_resonant[i_length, i_temperature] = bool(diagnostic["is_double_resonant"])

    best_index = np.unravel_index(int(np.argmin(abs_delta_phi_wrapped_rad)), abs_delta_phi_wrapped_rad.shape)
    best_length_index, best_temperature_index = best_index

    return {
        "temperature_grid_K": temperature_grid_K,
        "crystal_length_grid_m": crystal_length_grid_m,
        "delta_phi_wrapped_rad": delta_phi_wrapped_rad,
        "abs_delta_phi_wrapped_rad": abs_delta_phi_wrapped_rad,
        "is_double_resonant": is_double_resonant,
        "resonance_tolerance_rad": float(resonance_tolerance_rad),
        "best_temperature_K": float(temperature_grid_K[best_temperature_index]),
        "best_crystal_length_m": float(crystal_length_grid_m[best_length_index]),
        "best_delta_phi_wrapped_rad": float(delta_phi_wrapped_rad[best_index]),
        "best_abs_delta_phi_wrapped_rad": float(abs_delta_phi_wrapped_rad[best_index]),
        "best_is_double_resonant": bool(is_double_resonant[best_index]),
    }


__all__ = [
    "compute_double_resonance_scan",
]
