"""Phase-matching and QPM physics helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from common.constants import TWO_PI

def sinc(x: np.ndarray | float) -> np.ndarray | float:
    """Return normalized ``sin(x)/x`` with safe handling around zero."""
    x_arr = np.asarray(x)
    out = np.ones_like(x_arr, dtype=float)
    mask = np.abs(x_arr) > 0
    out[mask] = np.sin(x_arr[mask]) / x_arr[mask]
    return out if isinstance(x, np.ndarray) else float(out)


@dataclass(frozen=True)
class PhaseMatchingResult:
    """Per-temperature phase-matching output."""

    T_K: float
    n_p: float
    n_s: float
    n_i: float
    delta_k_rad_per_m: float
    delta_k_eff_rad_per_m: float
    pm_power: float
    Lambda_T_m: float


@dataclass(frozen=True)
class DesignPolingResult:
    """Design-stage QPM result derived from wavelengths and one temperature."""

    temperature_K: float
    delta_k_bulk_rad_per_m: float
    Lambda0_design_m: float


def k_of_n(wavelength_m: np.ndarray | float, n: np.ndarray | float) -> np.ndarray | float:
    """Return wave-vector magnitude ``k = 2πn/λ`` in rad/m."""
    wl = np.asarray(wavelength_m, dtype=float)
    nn = np.asarray(n, dtype=float)
    k = TWO_PI * nn / wl
    return k if isinstance(wavelength_m, np.ndarray) or isinstance(n, np.ndarray) else float(k)


def delta_k_three_wave(
    wavelength_p_m: float,
    wavelength_s_m: float,
    wavelength_i_m: float,
    n_p: float,
    n_s: float,
    n_i: float,
) -> float:
    """Return three-wave mismatch ``Δk = k_p - k_s - k_i`` in rad/m."""
    k_p = k_of_n(wavelength_p_m, n_p)
    k_s = k_of_n(wavelength_s_m, n_s)
    k_i = k_of_n(wavelength_i_m, n_i)
    return float(k_p - k_s - k_i)


def compute_design_poling_period(
    wavelength_p_m: float,
    wavelength_s_m: float,
    wavelength_i_m: float,
    temperature_K: float,
    n_p_of_lambda_T: Callable[[float, float], float],
    n_s_of_lambda_T: Callable[[float, float], float],
    n_i_of_lambda_T: Callable[[float, float], float],
    qpm_order_m: int,
) -> DesignPolingResult:
    """Return the poling period required for phase matching at one temperature."""
    n_p = float(n_p_of_lambda_T(wavelength_p_m, temperature_K))
    n_s = float(n_s_of_lambda_T(wavelength_s_m, temperature_K))
    n_i = float(n_i_of_lambda_T(wavelength_i_m, temperature_K))
    delta_k_bulk = delta_k_three_wave(wavelength_p_m, wavelength_s_m, wavelength_i_m, n_p, n_s, n_i)
    if not np.isfinite(delta_k_bulk) or np.isclose(delta_k_bulk, 0.0):
        raise ValueError("Cannot design poling period because the bulk phase mismatch is zero or non-finite.")
    return DesignPolingResult(
        temperature_K=float(temperature_K),
        delta_k_bulk_rad_per_m=float(delta_k_bulk),
        Lambda0_design_m=float(TWO_PI * qpm_order_m / delta_k_bulk),
    )


def qpm_grating_k(Lambda_m: float) -> float:
    """Return QPM grating vector ``K_g = 2π/Λ`` in rad/m."""
    return float(TWO_PI / Lambda_m)


def delta_k_qpm(delta_k_rad_per_m: float, Lambda_m: float, m: int = 1) -> float:
    """Return effective mismatch ``Δk_eff = Δk - m*K_g`` in rad/m."""
    return float(delta_k_rad_per_m - m * qpm_grating_k(Lambda_m))


def pm_amplitude_factor(delta_k_eff: float, crystal_length_m: float) -> float:
    """Return phase-matching amplitude factor ``sinc(Δk_eff*L/2)``."""
    return float(sinc(0.5 * delta_k_eff * crystal_length_m))


def pm_power_factor(delta_k_eff: float, crystal_length_m: float) -> float:
    """Return phase-matching power factor ``|sinc(Δk_eff*L/2)|^2``."""
    amp = pm_amplitude_factor(delta_k_eff, crystal_length_m)
    return float(amp * amp)


def expand_linear(L0: float, T_K: float, T0_K: float, alpha_perK: float) -> float:
    """Apply linear thermal expansion to ``L0``."""
    return float(L0 * (1.0 + alpha_perK * (T_K - T0_K)))


def poling_period_T(
    Lambda0_m: float,
    T_K: float,
    T0_K: float = 293.15,
    alpha_perK: float = 0.0,
) -> float:
    """Return temperature-adjusted poling period ``Λ(T)``."""
    return expand_linear(Lambda0_m, T_K, T0_K, alpha_perK)


def delta_k_eff_T(
    T_K: float,
    wavelength_p_m: float,
    wavelength_s_m: float,
    wavelength_i_m: float,
    n_p_of_T: Callable[[float], float],
    n_s_of_T: Callable[[float], float],
    n_i_of_T: Callable[[float], float],
    Lambda0_m: float,
    crystal_length_m: float,
    T0_K: float = 293.15,
    alpha_perK: float = 0.0,
    qpm_order_m: int = 1,
) -> PhaseMatchingResult:
    """Compute phase-matching quantities at one temperature point."""
    n_p = float(n_p_of_T(T_K))
    n_s = float(n_s_of_T(T_K))
    n_i = float(n_i_of_T(T_K))

    dk = delta_k_three_wave(wavelength_p_m, wavelength_s_m, wavelength_i_m, n_p, n_s, n_i)
    Lambda_T = poling_period_T(Lambda0_m, T_K, T0_K=T0_K, alpha_perK=alpha_perK)
    dk_eff = delta_k_qpm(dk, Lambda_T, m=qpm_order_m)
    pm_pow = pm_power_factor(dk_eff, crystal_length_m)

    return PhaseMatchingResult(
        T_K=float(T_K),
        n_p=n_p,
        n_s=n_s,
        n_i=n_i,
        delta_k_rad_per_m=float(dk),
        delta_k_eff_rad_per_m=float(dk_eff),
        pm_power=float(pm_pow),
        Lambda_T_m=float(Lambda_T),
    )


def scan_phase_matching_vs_temperature(
    T_min_K: float,
    T_max_K: float,
    n_T: int,
    wavelength_p_m: float,
    wavelength_s_m: float,
    wavelength_i_m: float,
    n_p_of_T: Callable[[float], float],
    n_s_of_T: Callable[[float], float],
    n_i_of_T: Callable[[float], float],
    Lambda0_m: float,
    crystal_length_m: float,
    T0_K: float = 293.15,
    alpha_perK: float = 0.0,
    qpm_order_m: int = 1,
) -> dict[str, np.ndarray]:
    """Scan phase-matching metrics over a temperature grid for a chosen QPM period."""
    Ts = np.linspace(float(T_min_K), float(T_max_K), int(n_T))

    n_p_vals = np.empty_like(Ts)
    n_s_vals = np.empty_like(Ts)
    n_i_vals = np.empty_like(Ts)
    dk_vals = np.empty_like(Ts)
    dk_eff_vals = np.empty_like(Ts)
    pm_vals = np.empty_like(Ts)
    Lambda_vals = np.empty_like(Ts)

    for i, T in enumerate(Ts):
        res = delta_k_eff_T(
            T,
            wavelength_p_m,
            wavelength_s_m,
            wavelength_i_m,
            n_p_of_T,
            n_s_of_T,
            n_i_of_T,
            Lambda0_m,
            crystal_length_m,
            T0_K=T0_K,
            alpha_perK=alpha_perK,
            qpm_order_m=qpm_order_m,
        )
        n_p_vals[i] = res.n_p
        n_s_vals[i] = res.n_s
        n_i_vals[i] = res.n_i
        dk_vals[i] = res.delta_k_rad_per_m
        dk_eff_vals[i] = res.delta_k_eff_rad_per_m
        pm_vals[i] = res.pm_power
        Lambda_vals[i] = res.Lambda_T_m

    i_best = int(np.argmax(pm_vals))
    i_reference = int(np.argmin(np.abs(dk_eff_vals)))

    return {
        "T_K": Ts,
        "n_p": n_p_vals,
        "n_s": n_s_vals,
        "n_i": n_i_vals,
        "delta_k_rad_per_m": dk_vals,
        "delta_k_eff_rad_per_m": dk_eff_vals,
        "pm_power": pm_vals,
        "Lambda_T_m": Lambda_vals,
        "T_best_K": np.array([Ts[i_best]]),
        "pm_power_best": np.array([pm_vals[i_best]]),
        "Lambda0_input_m": np.array([float(Lambda0_m)]),
        "Lambda0_effective_m": np.array([float(Lambda0_m)]),
        "T_reference_for_lambda_opt_K": np.array([Ts[i_reference]]),
    }


__all__ = [
    "PhaseMatchingResult",
    "DesignPolingResult",
    "sinc",
    "k_of_n",
    "delta_k_three_wave",
    "compute_design_poling_period",
    "qpm_grating_k",
    "delta_k_qpm",
    "pm_amplitude_factor",
    "pm_power_factor",
    "expand_linear",
    "poling_period_T",
    "delta_k_eff_T",
    "scan_phase_matching_vs_temperature",
]
