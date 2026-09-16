"""Boyd-Kleinman physics, helpers, sweeps, and high-level analysis."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import warnings

import numpy as np

try:
    from common.constants import C_M_PER_S, PI, TWO_PI
except ImportError:
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from common.constants import C_M_PER_S, PI, TWO_PI

from .crystal_phase_matching import delta_k_eff_T, delta_k_qpm, delta_k_three_wave, poling_period_T


# NumPy 2 removed ``np.trapz``; keep compatibility with both old and new releases.
_TRAPEZOID = np.trapezoid if hasattr(np, "trapezoid") else np.trapz


# BK core physics


def compute_focusing_parameter(crystal_length_m: float, rayleigh_range_m: float) -> float:
    """Return the Boyd-Kleinman focusing parameter ``xi = L / (2 zR)``."""
    if rayleigh_range_m <= 0:
        return np.nan
    return float(crystal_length_m / (2.0 * rayleigh_range_m))


def focusing_parameter_xi(crystal_length_m: float, rayleigh_range_m: float) -> float:
    """Named BK helper returning the focusing parameter ``xi``."""
    return compute_focusing_parameter(crystal_length_m, rayleigh_range_m)


def sigma_parameter(rayleigh_range_m: float, delta_k_eff_rad_per_m: float) -> float:
    """Return the BK mismatch control parameter ``sigma = zR * Delta_k_eff``."""
    return float(rayleigh_range_m * delta_k_eff_rad_per_m)


def beam_waist_from_rayleigh_range(
    rayleigh_range_m: float,
    wavelength_m: float,
    refractive_index: float,
) -> float:
    """Return the Gaussian beam waist consistent with ``zR = pi * n * w0^2 / lambda``."""
    if rayleigh_range_m <= 0 or wavelength_m <= 0 or refractive_index <= 0:
        return np.nan
    return float(np.sqrt(rayleigh_range_m * wavelength_m / (PI * refractive_index)))


def boyd_kleinman_integral(
    xi: float,
    delta_k: float,
    crystal_length_m: float,
    n_points: int = 4001,
) -> complex:
    """Return the normalized focused-beam overlap integral."""
    if not np.isfinite(xi) or xi <= 0 or crystal_length_m <= 0:
        return 0.0 + 0.0j

    z_r = crystal_length_m / (2.0 * xi)
    z = np.linspace(-0.5 * crystal_length_m, 0.5 * crystal_length_m, int(n_points))
    integrand = np.exp(1j * delta_k * z) / (1.0 + 1j * z / z_r)
    return complex(_TRAPEZOID(integrand, z) / crystal_length_m)


def boyd_kleinman_efficiency(
    rayleigh_range_m: float,
    crystal_length_m: float,
    delta_k: float,
) -> float:
    """Return the low-level BK overlap quantity ``|I|^2``.

    This quantity depends only on the Boyd-Kleinman control parameters
    ``xi = L / (2 z_R)`` and ``sigma = z_R * Delta_k``.
    """
    xi = compute_focusing_parameter(crystal_length_m, rayleigh_range_m)
    integral = boyd_kleinman_integral(xi, delta_k, crystal_length_m)
    return float(np.abs(integral) ** 2)


# BK normalization helpers


def normalize_curve(curve: np.ndarray) -> np.ndarray:
    """Normalize one BK curve by its maximum finite value."""
    normalized = np.asarray(curve, dtype=float).copy()
    peak = float(np.nanmax(normalized))
    if np.isfinite(peak) and peak > 0.0:
        normalized /= peak
    return normalized


def normalize_curve_set(curves: np.ndarray) -> np.ndarray:
    """Normalize a set of BK curves row-by-row."""
    normalized = np.asarray(curves, dtype=float).copy()
    for i in range(normalized.shape[0]):
        normalized[i] = normalize_curve(normalized[i])
    return normalized


def resolve_bk_reference_period(
    wavelength_p_m: float,
    wavelength_s_m: float,
    wavelength_i_m: float,
    wavelength_scan_temperature_K: float,
    n_p_of_lambda_T,
    n_s_of_lambda_T,
    n_i_of_lambda_T,
    Lambda0_m: float,
    qpm_order_m: int,
    recenter_to_phase_match: bool,
) -> float:
    """Return the QPM period used for BK analysis, optionally recentered to phase matching."""
    if not recenter_to_phase_match:
        return float(Lambda0_m)

    n_p_center = float(n_p_of_lambda_T(wavelength_p_m, wavelength_scan_temperature_K))
    n_s_center = float(n_s_of_lambda_T(wavelength_s_m, wavelength_scan_temperature_K))
    n_i_center = float(n_i_of_lambda_T(wavelength_i_m, wavelength_scan_temperature_K))
    delta_k_center = delta_k_three_wave(
        wavelength_p_m=wavelength_p_m,
        wavelength_s_m=wavelength_s_m,
        wavelength_i_m=wavelength_i_m,
        n_p=n_p_center,
        n_s=n_s_center,
        n_i=n_i_center,
    )
    return float(TWO_PI * qpm_order_m / delta_k_center)


def evaluate_bk_h(
    crystal_length_m: float,
    rayleigh_range_m: float,
    delta_k_eff_rad_per_m: float,
) -> float:
    """Evaluate the BK master-map quantity ``h_BK = xi * |I|^2``."""
    xi = focusing_parameter_xi(float(crystal_length_m), float(rayleigh_range_m))
    return float(
        xi
        * boyd_kleinman_efficiency(
            rayleigh_range_m=float(rayleigh_range_m),
            crystal_length_m=float(crystal_length_m),
            delta_k=float(delta_k_eff_rad_per_m),
        )
    )


def idler_wavelength_from_energy_conservation(
    wavelength_p_m: float,
    wavelength_s_m: float,
) -> float:
    """Return the idler wavelength from exact three-wave energy conservation.

    Uses ``1/lambda_p = 1/lambda_s + 1/lambda_i`` with fixed pump wavelength.
    Returns ``np.nan`` for invalid or unphysical cases.
    """
    if wavelength_p_m <= 0.0 or wavelength_s_m <= 0.0:
        return float(np.nan)
    reciprocal_idler = (1.0 / wavelength_p_m) - (1.0 / wavelength_s_m)
    if not np.isfinite(reciprocal_idler) or reciprocal_idler <= 0.0:
        return float(np.nan)
    wavelength_i_m = 1.0 / reciprocal_idler
    if not np.isfinite(wavelength_i_m) or wavelength_i_m <= 0.0:
        return float(np.nan)
    return float(wavelength_i_m)


def angular_frequency_from_wavelength(wavelength_m: float) -> float:
    """Return angular frequency ``omega = 2 pi c / lambda``."""
    if wavelength_m <= 0.0:
        return float(np.nan)
    return float(TWO_PI * C_M_PER_S / wavelength_m)


def wavelength_from_angular_frequency(angular_frequency_rad_per_s: float) -> float:
    """Return wavelength ``lambda = 2 pi c / omega``."""
    if angular_frequency_rad_per_s <= 0.0:
        return float(np.nan)
    return float(TWO_PI * C_M_PER_S / angular_frequency_rad_per_s)


# BK sweep helpers


def compute_bk_vs_temperature_for_lengths(
    temperature_grid: np.ndarray,
    crystal_lengths_m: np.ndarray,
    rayleigh_range_m: float,
    wavelength_p_m: float,
    wavelength_s_m: float,
    wavelength_i_m: float,
    n_p_of_T,
    n_s_of_T,
    n_i_of_T,
    Lambda0_m: float,
    T0_K: float,
    alpha_perK: float,
    qpm_order_m: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute BK-vs-temperature curves for a crystal-length sweep."""
    curves = np.empty((len(crystal_lengths_m), len(temperature_grid)), dtype=float)
    sigma_curves = np.empty_like(curves)
    for i_length, crystal_length_m in enumerate(crystal_lengths_m):
        for i_temperature, temperature_k in enumerate(temperature_grid):
            phase = delta_k_eff_T(
                temperature_k,
                wavelength_p_m=wavelength_p_m,
                wavelength_s_m=wavelength_s_m,
                wavelength_i_m=wavelength_i_m,
                n_p_of_T=n_p_of_T,
                n_s_of_T=n_s_of_T,
                n_i_of_T=n_i_of_T,
                Lambda0_m=Lambda0_m,
                crystal_length_m=float(crystal_length_m),
                T0_K=T0_K,
                alpha_perK=alpha_perK,
                qpm_order_m=qpm_order_m,
            )
            curves[i_length, i_temperature] = evaluate_bk_h(
                crystal_length_m,
                rayleigh_range_m,
                phase.delta_k_eff_rad_per_m,
            )
            sigma_curves[i_length, i_temperature] = sigma_parameter(rayleigh_range_m, phase.delta_k_eff_rad_per_m)
    return curves, sigma_curves


def compute_bk_vs_temperature_for_rayleigh_ranges(
    temperature_grid: np.ndarray,
    crystal_length_m: float,
    rayleigh_ranges_m: np.ndarray,
    wavelength_p_m: float,
    wavelength_s_m: float,
    wavelength_i_m: float,
    n_p_of_T,
    n_s_of_T,
    n_i_of_T,
    Lambda0_m: float,
    T0_K: float,
    alpha_perK: float,
    qpm_order_m: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute BK-vs-temperature curves for a Rayleigh-range sweep."""
    curves = np.empty((len(rayleigh_ranges_m), len(temperature_grid)), dtype=float)
    sigma_curves = np.empty_like(curves)
    for i_zr, rayleigh_range_m in enumerate(rayleigh_ranges_m):
        for i_temperature, temperature_k in enumerate(temperature_grid):
            phase = delta_k_eff_T(
                temperature_k,
                wavelength_p_m=wavelength_p_m,
                wavelength_s_m=wavelength_s_m,
                wavelength_i_m=wavelength_i_m,
                n_p_of_T=n_p_of_T,
                n_s_of_T=n_s_of_T,
                n_i_of_T=n_i_of_T,
                Lambda0_m=Lambda0_m,
                crystal_length_m=float(crystal_length_m),
                T0_K=T0_K,
                alpha_perK=alpha_perK,
                qpm_order_m=qpm_order_m,
            )
            curves[i_zr, i_temperature] = evaluate_bk_h(
                crystal_length_m,
                rayleigh_range_m,
                phase.delta_k_eff_rad_per_m,
            )
            sigma_curves[i_zr, i_temperature] = sigma_parameter(rayleigh_range_m, phase.delta_k_eff_rad_per_m)
    return curves, sigma_curves


def compute_bk_vs_wavelength_for_lengths(
    wavelength_grid_m: np.ndarray,
    crystal_lengths_m: np.ndarray,
    rayleigh_range_m: float,
    wavelength_p_m: float,
    wavelength_s_m: float,
    wavelength_i_m: float,
    wavelength_scan_temperature_K: float,
    n_p_of_lambda_T,
    n_s_of_lambda_T,
    n_i_of_lambda_T,
    Lambda0_m: float,
    T0_K: float,
    alpha_perK: float,
    qpm_order_m: int,
) -> np.ndarray:
    """Compute BK-vs-central-degenerate-wavelength curves for a crystal-length sweep."""
    curves = np.empty((len(crystal_lengths_m), len(wavelength_grid_m)), dtype=float)
    Lambda_T_m = poling_period_T(Lambda0_m, wavelength_scan_temperature_K, T0_K=T0_K, alpha_perK=alpha_perK)
    for i_length, crystal_length_m in enumerate(crystal_lengths_m):
        for i_wavelength, lambda0_scan_m in enumerate(wavelength_grid_m):
            signal_wavelength_m = float(lambda0_scan_m)
            idler_wavelength_m = float(lambda0_scan_m)
            omega0_rad_per_s = angular_frequency_from_wavelength(lambda0_scan_m)
            wavelength_p_scan_m = wavelength_from_angular_frequency(2.0 * omega0_rad_per_s)
            if (
                signal_wavelength_m <= 0.0
                or not np.isfinite(idler_wavelength_m)
                or not np.isfinite(wavelength_p_scan_m)
            ):
                curves[i_length, i_wavelength] = np.nan
                continue
            n_p = float(n_p_of_lambda_T(wavelength_p_scan_m, wavelength_scan_temperature_K))
            n_s = float(n_s_of_lambda_T(signal_wavelength_m, wavelength_scan_temperature_K))
            n_i = float(n_i_of_lambda_T(idler_wavelength_m, wavelength_scan_temperature_K))
            delta_k = delta_k_three_wave(
                wavelength_p_m=wavelength_p_scan_m,
                wavelength_s_m=signal_wavelength_m,
                wavelength_i_m=idler_wavelength_m,
                n_p=n_p,
                n_s=n_s,
                n_i=n_i,
            )
            delta_k_eff = delta_k_qpm(delta_k, Lambda_T_m, m=qpm_order_m)
            curves[i_length, i_wavelength] = evaluate_bk_h(
                crystal_length_m,
                rayleigh_range_m,
                delta_k_eff,
            )
    return curves


def compute_bk_vs_detuning_for_lengths(
    delta_lambda_grid_m: np.ndarray,
    crystal_lengths_m: np.ndarray,
    rayleigh_range_m: float,
    wavelength_p_m: float,
    wavelength_s_m: float,
    wavelength_i_m: float,
    wavelength_scan_temperature_K: float,
    n_p_of_lambda_T,
    n_s_of_lambda_T,
    n_i_of_lambda_T,
    Lambda0_m: float,
    T0_K: float,
    alpha_perK: float,
    qpm_order_m: int,
) -> np.ndarray:
    """Compute BK-vs-detuning curves for a crystal-length sweep."""
    curves = np.empty((len(crystal_lengths_m), len(delta_lambda_grid_m)), dtype=float)
    # IMPORTANT: use fixed poling period (do NOT thermally rescale here),
    # otherwise delta_k_eff stays artificially near zero and flattens detuning curves.
    Lambda_T_m = float(Lambda0_m)

    # Build the detuning scan in FREQUENCY around the degenerate center.
    # The horizontal axis is still reported as a positive wavelength detuning
    # for comparison with the thesis figure, but the actual signal/idler split
    # must be symmetric in omega, not symmetric in lambda.
    omega_p_rad_per_s = angular_frequency_from_wavelength(wavelength_p_m)
    omega0_rad_per_s = 0.5 * omega_p_rad_per_s
    for i_length, crystal_length_m in enumerate(crystal_lengths_m):
        for i_detuning, delta_lambda in enumerate(delta_lambda_grid_m):
            detuned_wavelength_m = float(wavelength_s_m + delta_lambda)
            if detuned_wavelength_m <= 0.0 or not np.isfinite(detuned_wavelength_m):
                curves[i_length, i_detuning] = np.nan
                continue

            delta_omega_rad_per_s = abs(
                angular_frequency_from_wavelength(detuned_wavelength_m) - omega0_rad_per_s
            )
            omega_s_rad_per_s = omega0_rad_per_s + delta_omega_rad_per_s
            omega_i_rad_per_s = omega0_rad_per_s - delta_omega_rad_per_s

            signal_wavelength_m = wavelength_from_angular_frequency(omega_s_rad_per_s)
            idler_wavelength_m = wavelength_from_angular_frequency(omega_i_rad_per_s)
            pump_wavelength_m = wavelength_from_angular_frequency(
                omega_s_rad_per_s + omega_i_rad_per_s
            )

            if (
                not np.isfinite(signal_wavelength_m)
                or not np.isfinite(idler_wavelength_m)
                or not np.isfinite(pump_wavelength_m)
                or signal_wavelength_m <= 0.0
                or idler_wavelength_m <= 0.0
                or pump_wavelength_m <= 0.0
            ):
                curves[i_length, i_detuning] = np.nan
                continue

            n_p = float(n_p_of_lambda_T(pump_wavelength_m, wavelength_scan_temperature_K))
            n_s = float(n_s_of_lambda_T(signal_wavelength_m, wavelength_scan_temperature_K))
            n_i = float(n_i_of_lambda_T(idler_wavelength_m, wavelength_scan_temperature_K))
            delta_k = delta_k_three_wave(
                wavelength_p_m=pump_wavelength_m,
                wavelength_s_m=signal_wavelength_m,
                wavelength_i_m=idler_wavelength_m,
                n_p=n_p,
                n_s=n_s,
                n_i=n_i,
            )
            delta_k_eff = delta_k_qpm(delta_k, Lambda_T_m, m=qpm_order_m)
            curves[i_length, i_detuning] = evaluate_bk_h(
                crystal_length_m,
                rayleigh_range_m,
                delta_k_eff,
            )
    return curves


# BK high-level analysis


@dataclass(frozen=True)
class BKAnalysisConfig:
    """BK-specific sweep and analysis configuration."""

    crystal_lengths_m: tuple[float, ...] | None = None
    crystal_length_scale_factors: tuple[float, ...] = (0.625, 1.0, 1.875)
    rayleigh_ranges_m: tuple[float, ...] | None = None
    rayleigh_range_scale_factors: tuple[float, ...] = (0.7, 1.0, 1.3)
    temperature_half_span_K: float = 10.0
    wavelength_half_span_m: float = 2e-9
    detuning_half_span_m: float = 14e-9
    n_wavelength: int = 201
    recenter_to_phase_match: bool = False
    sigma_min: float = -3.0
    sigma_max: float = 3.0
    n_sigma: int = 241
    xi_min: float = 0.1
    xi_max: float = 15.0
    n_xi: int = 241
    qpm_length_max_over_lcoh: float = 16.0
    qpm_poling_max_over_lcoh: float = 6.0
    qpm_n_length: int = 401
    qpm_n_poling: int = 241
    qpm_slice_values_over_lcoh: tuple[float, ...] = (1.0, 3.0, 5.0)
    qpm_n_z: int = 4000


@dataclass(frozen=True)
class BKAnalysisResult:
    """Structured BK analysis output for plotting and workflow assembly."""

    bk_master_sigma_values: np.ndarray
    bk_master_xi_values: np.ndarray
    bk_master_h_map: np.ndarray
    bk_master_sigma_opt: float
    bk_master_xi_opt: float
    bk_master_h_opt: float
    qpm_length_over_lcoh: np.ndarray
    qpm_poling_over_lcoh: np.ndarray
    qpm_relative_field_intensity: np.ndarray
    qpm_slice_values_over_lcoh: np.ndarray
    qpm_slice_curves: np.ndarray
    qpm_first_order_qpm_guide_over_lcoh: np.ndarray
    qpm_operating_length_over_lcoh: float
    qpm_operating_poling_over_lcoh: float
    qpm_reference_in_display_range: bool
    temperature_K: np.ndarray
    temperature_C: np.ndarray
    crystal_lengths_m: np.ndarray
    rayleigh_ranges_m: np.ndarray
    wavelength_m: np.ndarray
    delta_lambda_m: np.ndarray
    bk_vs_temperature_for_lengths: np.ndarray
    bk_vs_wavelength_for_lengths: np.ndarray
    bk_vs_temperature_for_rayleigh_ranges: np.ndarray
    bk_vs_delta_lambda_for_lengths: np.ndarray
    sigma_vs_temperature_for_lengths: np.ndarray
    sigma_vs_temperature_for_rayleigh_ranges: np.ndarray
    xi_for_length_sweep: np.ndarray
    xi_for_rayleigh_sweep: np.ndarray
    reference: dict[str, float | str | bool]


def bk_analysis_result_to_dict(result: BKAnalysisResult) -> dict[str, object]:
    """Convert a BK analysis dataclass into the dictionary payload used downstream."""
    return asdict(result)


def compute_bk_master_map(
    sigma_values: np.ndarray,
    xi_values: np.ndarray,
    crystal_length_m: float = 1.0,
) -> dict[str, np.ndarray | float]:
    """Compute the BK master map in the ``(sigma, xi)`` plane.

    Uses the documented convention ``z_R = L / (2 xi)`` and
    ``delta_k = sigma / z_R`` for a fixed crystal length ``L``. The plotted
    quantity is the normalized Boyd-Kleinman focusing factor
    ``h_BK = xi * |I|^2`` for the zero-walkoff case.
    """
    sigma_values = np.asarray(sigma_values, dtype=float)
    xi_values = np.asarray(xi_values, dtype=float)
    h_bk_map = np.full((len(xi_values), len(sigma_values)), np.nan, dtype=float)

    for i_xi, xi in enumerate(xi_values):
        if not np.isfinite(xi) or xi <= 0.0:
            continue
        z_r_m = crystal_length_m / (2.0 * xi)
        for i_sigma, sigma in enumerate(sigma_values):
            delta_k = sigma / z_r_m
            h_bk_map[i_xi, i_sigma] = xi * boyd_kleinman_efficiency(
                rayleigh_range_m=z_r_m,
                crystal_length_m=crystal_length_m,
                delta_k=delta_k,
            )

    if np.all(~np.isfinite(h_bk_map)):
        sigma_opt = np.nan
        xi_opt = np.nan
        h_bk_opt = np.nan
    else:
        flat_index = int(np.nanargmax(h_bk_map))
        i_xi_opt, i_sigma_opt = np.unravel_index(flat_index, h_bk_map.shape)
        sigma_opt = float(sigma_values[i_sigma_opt])
        xi_opt = float(xi_values[i_xi_opt])
        h_bk_opt = float(h_bk_map[i_xi_opt, i_sigma_opt])

        # For the normalized zero-walkoff BK formulation, the optimum should
        # lie near xi ~ 2-3 and sigma ~ 0.5-1. A value far outside that region
        # is a strong indication that the master-map normalization is wrong.
        if xi_opt < 0.3:
            warnings.warn(
                "BK master-map optimum lies at very small xi; check the normalized h_BK definition.",
                RuntimeWarning,
                stacklevel=2,
            )

    return {
        "sigma_values": sigma_values,
        "xi_values": xi_values,
        "h_bk_map": h_bk_map,
        "sigma_opt": sigma_opt,
        "xi_opt": xi_opt,
        "h_bk_opt": h_bk_opt,
    }


def compute_qpm_length_poling_map(
    length_over_lcoh: np.ndarray,
    poling_domain_length_over_lcoh: np.ndarray,
    n_z: int = 4000,
    slice_values_over_lcoh: tuple[float, ...] = (1.0, 3.0, 5.0),
) -> dict[str, np.ndarray]:
    """Compute a universal QPM map using the thesis domain-length convention.

    The plotted ``Lambda_pol`` is the domain length, so the nonlinear
    coefficient flips sign every ``Lambda_pol`` and the full grating period is
    ``2 * Lambda_pol``. In this convention, odd-order QPM branches appear at
    ``Lambda_pol / l_coh = 1, 3, 5, ...``.

    The map is expressed in normalized coordinates ``L / l_coh`` and
    ``Lambda_pol / l_coh``. Each point therefore corresponds to its own local
    coherence length ``l_coh = pi / |Delta_k|`` and hence to a scanned mismatch
    ``Delta_k``. In these local normalized units one can evaluate the integral
    with ``Delta_k = pi`` without tying the map to one fixed physical system.
    """
    length_over_lcoh = np.asarray(length_over_lcoh, dtype=float)
    poling_domain_length_over_lcoh = np.asarray(poling_domain_length_over_lcoh, dtype=float)
    slice_values_over_lcoh_arr = np.asarray(slice_values_over_lcoh, dtype=float)
    delta_k = PI
    poling_over_lcoh = poling_domain_length_over_lcoh

    relative_field_intensity = np.full((len(poling_over_lcoh), len(length_over_lcoh)), np.nan, dtype=float)

    for i_poling, lambda_domain_over_lcoh in enumerate(poling_domain_length_over_lcoh):
        if not np.isfinite(lambda_domain_over_lcoh) or lambda_domain_over_lcoh <= 0.0:
            continue
        for i_length, crystal_length_over_lcoh in enumerate(length_over_lcoh):
            if not np.isfinite(crystal_length_over_lcoh) or crystal_length_over_lcoh <= 0.0:
                continue
            z = np.linspace(0.0, crystal_length_over_lcoh, int(n_z), dtype=float)
            domain_index = np.floor(z / lambda_domain_over_lcoh).astype(int)
            d = np.where(domain_index % 2 == 0, 1.0, -1.0)
            field = _TRAPEZOID(d * np.exp(1j * delta_k * z), z)
            relative_field_intensity[i_poling, i_length] = float(np.abs(field) ** 2)

    peak = float(np.nanmax(relative_field_intensity))
    if np.isfinite(peak) and peak > 0.0:
        relative_field_intensity /= peak

    slice_curves = np.full((len(slice_values_over_lcoh_arr), len(length_over_lcoh)), np.nan, dtype=float)
    for i_slice, slice_value in enumerate(slice_values_over_lcoh_arr):
        if len(poling_over_lcoh) == 0:
            continue
        slice_index = int(np.argmin(np.abs(poling_over_lcoh - slice_value)))
        slice_curves[i_slice] = relative_field_intensity[slice_index]

    # In the normalization l_coh = pi / |Delta_k| and using the domain-length
    # convention, first-order QPM gives Lambda_pol / l_coh = 1.
    # This is a QPM compensation condition, NOT Delta_k = 0.
    first_order_qpm_guide_over_lcoh = np.full_like(length_over_lcoh, 1.0, dtype=float)

    return {
        "length_over_lcoh": length_over_lcoh,
        "poling_over_lcoh": poling_over_lcoh,
        "relative_field_intensity": relative_field_intensity,
        "slice_values_over_lcoh": slice_values_over_lcoh_arr,
        "slice_curves": slice_curves,
        "first_order_qpm_guide_over_lcoh": first_order_qpm_guide_over_lcoh,
    }


def _resolve_bk_reference_state(
    context,
    mode_matching,
    phase_matching,
    wavelength_p_m: float,
    wavelength_s_m: float,
    wavelength_i_m: float,
    Lambda0_m: float,
    n_p_of_lambda_T,
    n_s_of_lambda_T,
    n_i_of_lambda_T,
    T0_K: float,
    qpm_order_m: int,
    bk_config: BKAnalysisConfig,
) -> dict[str, float]:
    """Resolve the BK reference operating point from the current simulation state."""
    lambda_input_m = float(Lambda0_m)
    lambda_input_or_effective_m = float(
        phase_matching["Lambda0_effective_m"][0]
        if phase_matching is not None
        and "Lambda0_effective_m" in phase_matching
        and len(phase_matching["Lambda0_effective_m"]) > 0
        else Lambda0_m
    )
    temperature_opt_K = float(
        phase_matching["T_best_K"][0]
        if phase_matching is not None and "T_best_K" in phase_matching and len(phase_matching["T_best_K"]) > 0
        else T0_K
    )
    wavelength_scan_temperature_K = temperature_opt_K
    lambda_reference_m = resolve_bk_reference_period(
        wavelength_p_m=wavelength_p_m,
        wavelength_s_m=wavelength_s_m,
        wavelength_i_m=wavelength_i_m,
        wavelength_scan_temperature_K=wavelength_scan_temperature_K,
        n_p_of_lambda_T=n_p_of_lambda_T,
        n_s_of_lambda_T=n_s_of_lambda_T,
        n_i_of_lambda_T=n_i_of_lambda_T,
        Lambda0_m=lambda_input_or_effective_m,
        qpm_order_m=qpm_order_m,
        recenter_to_phase_match=bk_config.recenter_to_phase_match,
    )
    return {
        "crystal_length_reference_m": float(context.crystal_length_m),
        "zR_m_reference": float(mode_matching.rayleigh_range_m),
        "temperature_opt_K": temperature_opt_K,
        "wavelength_scan_temperature_K": wavelength_scan_temperature_K,
        "Lambda0_input_m": lambda_input_m,
        "Lambda0_input_or_effective_m": lambda_input_or_effective_m,
        "lambda_reference_m": lambda_reference_m,
        "reference_kind": "operating",
        "xi_reference_source": float(focusing_parameter_xi(context.crystal_length_m, mode_matching.rayleigh_range_m)),
        "sigma_reference_source": float("nan"),
    }


def _build_optimal_bk_reference_state(
    context,
    phase_matching,
    wavelength_p_m: float,
    wavelength_s_m: float,
    wavelength_i_m: float,
    Lambda0_m: float,
    n_p_of_lambda_T,
    n_s_of_lambda_T,
    n_i_of_lambda_T,
    T0_K: float,
    alpha_perK: float,
    qpm_order_m: int,
    bk_config: BKAnalysisConfig,
    bk_master_map: dict[str, np.ndarray | float],
) -> dict[str, float]:
    """Build a BK reference state centered on the numerical master-map optimum."""
    sigma_opt = float(bk_master_map["sigma_opt"])
    xi_opt = float(bk_master_map["xi_opt"])
    crystal_length_reference_m = float(context.crystal_length_m)
    zR_opt_m = float(crystal_length_reference_m / (2.0 * xi_opt)) if np.isfinite(xi_opt) and xi_opt > 0.0 else np.nan
    delta_k_eff_opt_rad_per_m = float(sigma_opt / zR_opt_m) if np.isfinite(zR_opt_m) and zR_opt_m > 0.0 else np.nan

    lambda_input_m = float(Lambda0_m)
    lambda_input_or_effective_m = float(
        phase_matching["Lambda0_effective_m"][0]
        if phase_matching is not None
        and "Lambda0_effective_m" in phase_matching
        and len(phase_matching["Lambda0_effective_m"]) > 0
        else Lambda0_m
    )
    temperature_opt_K = float(
        phase_matching["T_best_K"][0]
        if phase_matching is not None and "T_best_K" in phase_matching and len(phase_matching["T_best_K"]) > 0
        else T0_K
    )
    wavelength_scan_temperature_K = temperature_opt_K

    n_p = float(n_p_of_lambda_T(wavelength_p_m, wavelength_scan_temperature_K))
    n_s = float(n_s_of_lambda_T(wavelength_s_m, wavelength_scan_temperature_K))
    n_i = float(n_i_of_lambda_T(wavelength_i_m, wavelength_scan_temperature_K))
    delta_k_bulk_center = float(
        delta_k_three_wave(
            wavelength_p_m=wavelength_p_m,
            wavelength_s_m=wavelength_s_m,
            wavelength_i_m=wavelength_i_m,
            n_p=n_p,
            n_s=n_s,
            n_i=n_i,
        )
    )
    grating_vector_target_rad_per_m = delta_k_bulk_center - delta_k_eff_opt_rad_per_m
    if not np.isfinite(grating_vector_target_rad_per_m) or grating_vector_target_rad_per_m <= 0.0:
        lambda_reference_m = float(np.nan)
    else:
        lambda_T_target_m = float(TWO_PI * qpm_order_m / grating_vector_target_rad_per_m)
        thermal_scale = 1.0 + alpha_perK * (wavelength_scan_temperature_K - T0_K)
        lambda_reference_m = float(lambda_T_target_m / thermal_scale) if thermal_scale > 0.0 else float(np.nan)
        if bk_config.recenter_to_phase_match:
            lambda_reference_m = resolve_bk_reference_period(
                wavelength_p_m=wavelength_p_m,
                wavelength_s_m=wavelength_s_m,
                wavelength_i_m=wavelength_i_m,
                wavelength_scan_temperature_K=wavelength_scan_temperature_K,
                n_p_of_lambda_T=n_p_of_lambda_T,
                n_s_of_lambda_T=n_s_of_lambda_T,
                n_i_of_lambda_T=n_i_of_lambda_T,
                Lambda0_m=lambda_reference_m,
                qpm_order_m=qpm_order_m,
                recenter_to_phase_match=True,
            )

    return {
        "crystal_length_reference_m": crystal_length_reference_m,
        "zR_m_reference": float(zR_opt_m),
        "temperature_opt_K": temperature_opt_K,
        "wavelength_scan_temperature_K": wavelength_scan_temperature_K,
        "Lambda0_input_m": lambda_input_m,
        "Lambda0_input_or_effective_m": lambda_input_or_effective_m,
        "lambda_reference_m": float(lambda_reference_m),
        "reference_kind": "optimal",
        "xi_reference_source": float(xi_opt),
        "sigma_reference_source": float(sigma_opt),
    }


def _build_temperature_grid(temperature_opt_K: float, half_span_K: float, n_temperature: int) -> tuple[np.ndarray, np.ndarray]:
    """Build the temperature grid in Kelvin and Celsius."""
    temperature_K = np.linspace(
        temperature_opt_K - float(half_span_K),
        temperature_opt_K + float(half_span_K),
        int(n_temperature),
        dtype=float,
    )
    return temperature_K, temperature_K - 273.15


def _build_wavelength_grids(
    wavelength_s_m: float,
    wavelength_half_span_m: float,
    detuning_half_span_m: float,
    n_wavelength: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Build central-wavelength and detuning grids for BK scans."""
    wavelength_m = np.linspace(
        float(wavelength_s_m - wavelength_half_span_m),
        float(wavelength_s_m + wavelength_half_span_m),
        int(n_wavelength),
        dtype=float,
    )
    delta_lambda_m = np.linspace(
        0.0,
        float(detuning_half_span_m),
        int(n_wavelength),
        dtype=float,
    )
    return wavelength_m, delta_lambda_m


def _resolve_sweep_arrays(
    bk_config: BKAnalysisConfig,
    crystal_length_reference_m: float,
    zR_m_reference: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Resolve BK length and Rayleigh-range sweeps from explicit values or scale factors."""
    crystal_length_scale_factors_arr = np.asarray(bk_config.crystal_length_scale_factors, dtype=float)
    rayleigh_range_scale_factors_arr = np.asarray(bk_config.rayleigh_range_scale_factors, dtype=float)

    crystal_lengths_m = bk_config.crystal_lengths_m
    rayleigh_ranges_m = bk_config.rayleigh_ranges_m
    if crystal_lengths_m is None:
        crystal_lengths_m = tuple(crystal_length_reference_m * crystal_length_scale_factors_arr)
    if rayleigh_ranges_m is None:
        rayleigh_ranges_m = tuple(zR_m_reference * rayleigh_range_scale_factors_arr)

    return (
        np.asarray(crystal_lengths_m, dtype=float),
        np.asarray(rayleigh_ranges_m, dtype=float),
        crystal_length_scale_factors_arr,
        rayleigh_range_scale_factors_arr,
    )


def _sigma_sweep_from_axis(
    axis_values: np.ndarray,
    sigma_center: float,
    sigma_half_span: float,
    one_sided: bool = False,
) -> np.ndarray:
    """Map an arbitrary plot axis onto a smooth BK mismatch sweep around ``sigma_center``."""
    axis_values = np.asarray(axis_values, dtype=float)
    if len(axis_values) == 0:
        return np.asarray([], dtype=float)
    if one_sided:
        axis_min = float(np.nanmin(axis_values))
        axis_max = float(np.nanmax(axis_values))
        if not np.isfinite(axis_min) or not np.isfinite(axis_max) or axis_max <= axis_min:
            normalized = np.zeros_like(axis_values, dtype=float)
        else:
            normalized = (axis_values - axis_min) / (axis_max - axis_min)
        return sigma_center + sigma_half_span * normalized

    axis_center = 0.5 * (float(np.nanmin(axis_values)) + float(np.nanmax(axis_values)))
    axis_half_span = 0.5 * (float(np.nanmax(axis_values)) - float(np.nanmin(axis_values)))
    if not np.isfinite(axis_half_span) or axis_half_span <= 0.0:
        normalized = np.zeros_like(axis_values, dtype=float)
    else:
        normalized = (axis_values - axis_center) / axis_half_span
    return sigma_center + sigma_half_span * normalized


def _compute_bk_vs_sigma_for_lengths(
    sigma_values: np.ndarray,
    crystal_lengths_m: np.ndarray,
    rayleigh_range_m: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Evaluate BK curves for a fixed Rayleigh range and a sigma sweep."""
    curves = np.empty((len(crystal_lengths_m), len(sigma_values)), dtype=float)
    sigma_curves = np.empty_like(curves)
    for i_length, crystal_length_m in enumerate(crystal_lengths_m):
        for i_sigma, sigma_value in enumerate(sigma_values):
            delta_k_eff = float(sigma_value / rayleigh_range_m)
            curves[i_length, i_sigma] = evaluate_bk_h(
                crystal_length_m=float(crystal_length_m),
                rayleigh_range_m=float(rayleigh_range_m),
                delta_k_eff_rad_per_m=delta_k_eff,
            )
            sigma_curves[i_length, i_sigma] = float(sigma_value)
    return curves, sigma_curves


def _compute_bk_vs_sigma_for_rayleigh_ranges(
    sigma_values: np.ndarray,
    crystal_length_m: float,
    rayleigh_ranges_m: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Evaluate BK curves for a fixed crystal length and a sigma sweep."""
    curves = np.empty((len(rayleigh_ranges_m), len(sigma_values)), dtype=float)
    sigma_curves = np.empty_like(curves)
    for i_zr, rayleigh_range_m in enumerate(rayleigh_ranges_m):
        for i_sigma, sigma_value in enumerate(sigma_values):
            delta_k_eff = float(sigma_value / rayleigh_range_m)
            curves[i_zr, i_sigma] = evaluate_bk_h(
                crystal_length_m=float(crystal_length_m),
                rayleigh_range_m=float(rayleigh_range_m),
                delta_k_eff_rad_per_m=delta_k_eff,
            )
            sigma_curves[i_zr, i_sigma] = float(sigma_value)
    return curves, sigma_curves


def _build_reference_metadata(
    context,
    mode_matching,
    n_p_of_T,
    n_s_of_T,
    n_i_of_T,
    wavelength_p_m: float,
    wavelength_s_m: float,
    wavelength_i_m: float,
    T0_K: float,
    alpha_perK: float,
    qpm_order_m: int,
    bk_config: BKAnalysisConfig,
    reference_state: dict[str, float],
    crystal_length_scale_factors_arr: np.ndarray,
    rayleigh_range_scale_factors_arr: np.ndarray,
) -> dict[str, float | str | bool]:
    """Build the BK reference metadata payload."""
    crystal_length_reference_m = reference_state["crystal_length_reference_m"]
    zR_m_reference = reference_state["zR_m_reference"]
    temperature_opt_K = reference_state["temperature_opt_K"]
    wavelength_scan_temperature_K = reference_state["wavelength_scan_temperature_K"]
    lambda_reference_m = reference_state["lambda_reference_m"]
    reference_kind = str(reference_state.get("reference_kind", "operating"))

    xi_reference = float(focusing_parameter_xi(crystal_length_reference_m, zR_m_reference))
    if reference_kind == "optimal":
        sigma_reference = float(reference_state.get("sigma_reference_source", np.nan))
        delta_k_bulk_reference_rad_per_m = float(np.nan)
        l_coh_reference_m = float(np.nan)
    else:
        sigma_reference = float(
            sigma_parameter(
                zR_m_reference,
                delta_k_eff_T(
                    temperature_opt_K,
                    wavelength_p_m=wavelength_p_m,
                    wavelength_s_m=wavelength_s_m,
                    wavelength_i_m=wavelength_i_m,
                    n_p_of_T=n_p_of_T,
                    n_s_of_T=n_s_of_T,
                    n_i_of_T=n_i_of_T,
                    Lambda0_m=lambda_reference_m,
                    crystal_length_m=crystal_length_reference_m,
                    T0_K=T0_K,
                    alpha_perK=alpha_perK,
                    qpm_order_m=qpm_order_m,
                ).delta_k_eff_rad_per_m,
            )
        )
        delta_k_bulk_reference_rad_per_m = float(
            delta_k_three_wave(
                wavelength_p_m=wavelength_p_m,
                wavelength_s_m=wavelength_s_m,
                wavelength_i_m=wavelength_i_m,
                n_p=float(n_p_of_T(temperature_opt_K)),
                n_s=float(n_s_of_T(temperature_opt_K)),
                n_i=float(n_i_of_T(temperature_opt_K)),
            )
        )
        l_coh_reference_m = float(PI / abs(delta_k_bulk_reference_rad_per_m)) if delta_k_bulk_reference_rad_per_m != 0.0 else np.nan

    return {
        "reference_kind": reference_kind,
        "T0_K": float(T0_K),
        "T0_C": float(T0_K - 273.15),
        "T_opt_K": float(temperature_opt_K),
        "T_opt_C": float(temperature_opt_K - 273.15),
        "wavelength_scan_temperature_K": float(wavelength_scan_temperature_K),
        "wavelength_scan_temperature_C": float(wavelength_scan_temperature_K - 273.15),
        "wavelength_p_m": float(wavelength_p_m),
        "lambda0_m": float(wavelength_s_m),
        "wavelength_i_m": float(wavelength_i_m),
        "Lambda0_input_m": float(reference_state["Lambda0_input_m"]),
        "Lambda0_phase_matching_effective_m": float(reference_state["Lambda0_input_or_effective_m"]),
        "Lambda0_analysis_m": float(lambda_reference_m),
        "crystal_length_m": float(context.crystal_length_m),
        "crystal_length_reference_m": float(crystal_length_reference_m),
        "rayleigh_range_m": float(zR_m_reference),
        "zR_m_reference": float(zR_m_reference),
        "xi_reference": float(xi_reference),
        "sigma_reference": float(sigma_reference),
        "delta_k_bulk_reference_rad_per_m": float(delta_k_bulk_reference_rad_per_m),
        "l_coh_reference_m": float(l_coh_reference_m),
        "xi_baseline_from_reference_length_and_zR": float(xi_reference),
        "sigma_baseline_from_reference_zR_and_optimum": float(sigma_reference),
        "bk_quantity": "normalized BK focusing factor h_BK",
        "temperature_model_input": "Kelvin",
        "temperature_plot_axis": "Celsius",
        "recenter_to_phase_match": bool(bk_config.recenter_to_phase_match),
        "crystal_length_scale_factors": ",".join(str(v) for v in crystal_length_scale_factors_arr),
        "rayleigh_range_scale_factors": ",".join(str(v) for v in rayleigh_range_scale_factors_arr),
        "temperature_half_span_K": float(bk_config.temperature_half_span_K),
        "wavelength_half_span_m": float(bk_config.wavelength_half_span_m),
        "detuning_half_span_m": float(bk_config.detuning_half_span_m),
        "xi_reference_source": float(reference_state.get("xi_reference_source", xi_reference)),
        "sigma_reference_source": float(reference_state.get("sigma_reference_source", sigma_reference)),
    }


def run_bk_analysis(
    context,
    mode_matching,
    n_p_of_T,
    n_s_of_T,
    n_i_of_T,
    n_p_of_lambda_T,
    n_s_of_lambda_T,
    n_i_of_lambda_T,
    wavelength_p_m: float,
    wavelength_s_m: float,
    wavelength_i_m: float,
    Lambda0_m: float,
    T0_K: float = 293.15,
    alpha_perK: float = 0.0,
    qpm_order_m: int = 1,
    phase_matching=None,
    n_temperature: int = 201,
    bk_config: BKAnalysisConfig | None = None,
    reference_mode: str = "operating",
) -> BKAnalysisResult:
    """Run the full BK analysis from the current simulation state."""
    if bk_config is None:
        bk_config = BKAnalysisConfig()

    bk_master_map = compute_bk_master_map(
        sigma_values=np.linspace(bk_config.sigma_min, bk_config.sigma_max, int(bk_config.n_sigma), dtype=float),
        xi_values=np.linspace(bk_config.xi_min, bk_config.xi_max, int(bk_config.n_xi), dtype=float),
        crystal_length_m=1.0,
    )
    if reference_mode == "operating":
        reference_state = _resolve_bk_reference_state(
            context=context,
            mode_matching=mode_matching,
            phase_matching=phase_matching,
            wavelength_p_m=wavelength_p_m,
            wavelength_s_m=wavelength_s_m,
            wavelength_i_m=wavelength_i_m,
            Lambda0_m=Lambda0_m,
            n_p_of_lambda_T=n_p_of_lambda_T,
            n_s_of_lambda_T=n_s_of_lambda_T,
            n_i_of_lambda_T=n_i_of_lambda_T,
            T0_K=T0_K,
            qpm_order_m=qpm_order_m,
            bk_config=bk_config,
        )
    elif reference_mode == "optimal":
        reference_state = _build_optimal_bk_reference_state(
            context=context,
            phase_matching=phase_matching,
            wavelength_p_m=wavelength_p_m,
            wavelength_s_m=wavelength_s_m,
            wavelength_i_m=wavelength_i_m,
            Lambda0_m=Lambda0_m,
            n_p_of_lambda_T=n_p_of_lambda_T,
            n_s_of_lambda_T=n_s_of_lambda_T,
            n_i_of_lambda_T=n_i_of_lambda_T,
            T0_K=T0_K,
            alpha_perK=alpha_perK,
            qpm_order_m=qpm_order_m,
            bk_config=bk_config,
            bk_master_map=bk_master_map,
        )
    else:
        raise ValueError(f"Unsupported BK reference_mode: {reference_mode}")

    crystal_lengths_arr, rayleigh_ranges_arr, crystal_length_scale_factors_arr, rayleigh_range_scale_factors_arr = (
        _resolve_sweep_arrays(
            bk_config=bk_config,
            crystal_length_reference_m=reference_state["crystal_length_reference_m"],
            zR_m_reference=reference_state["zR_m_reference"],
        )
    )
    temperature_grid, temperature_grid_C = _build_temperature_grid(
        temperature_opt_K=reference_state["temperature_opt_K"],
        half_span_K=bk_config.temperature_half_span_K,
        n_temperature=n_temperature,
    )
    wavelength_grid_m, delta_lambda_grid_m = _build_wavelength_grids(
        wavelength_s_m=wavelength_s_m,
        wavelength_half_span_m=bk_config.wavelength_half_span_m,
        detuning_half_span_m=bk_config.detuning_half_span_m,
        n_wavelength=bk_config.n_wavelength,
    )
    qpm_map = compute_qpm_length_poling_map(
        length_over_lcoh=np.linspace(
            0.0,
            bk_config.qpm_length_max_over_lcoh,
            int(bk_config.qpm_n_length),
            dtype=float,
        ),
        poling_domain_length_over_lcoh=np.linspace(
            0.1,
            bk_config.qpm_poling_max_over_lcoh,
            int(bk_config.qpm_n_poling),
            dtype=float,
        ),
        n_z=bk_config.qpm_n_z,
        slice_values_over_lcoh=bk_config.qpm_slice_values_over_lcoh,
    )

    if reference_mode == "optimal":
        sigma_opt = float(bk_master_map["sigma_opt"])
        sigma_half_span = float(
            min(
                abs(sigma_opt - float(np.nanmin(bk_master_map["sigma_values"]))),
                abs(float(np.nanmax(bk_master_map["sigma_values"])) - sigma_opt),
            )
        )
        sigma_temperature = _sigma_sweep_from_axis(
            temperature_grid,
            sigma_center=sigma_opt,
            sigma_half_span=sigma_half_span,
            one_sided=False,
        )
        sigma_wavelength = _sigma_sweep_from_axis(
            wavelength_grid_m,
            sigma_center=sigma_opt,
            sigma_half_span=sigma_half_span,
            one_sided=False,
        )
        sigma_detuning = _sigma_sweep_from_axis(
            delta_lambda_grid_m,
            sigma_center=sigma_opt,
            sigma_half_span=sigma_half_span,
            one_sided=True,
        )
        bk_vs_temperature_lengths, sigma_vs_temperature_for_lengths = _compute_bk_vs_sigma_for_lengths(
            sigma_values=sigma_temperature,
            crystal_lengths_m=crystal_lengths_arr,
            rayleigh_range_m=reference_state["zR_m_reference"],
        )
        bk_vs_wavelength_lengths, _ = _compute_bk_vs_sigma_for_lengths(
            sigma_values=sigma_wavelength,
            crystal_lengths_m=crystal_lengths_arr,
            rayleigh_range_m=reference_state["zR_m_reference"],
        )
        bk_vs_temperature_rayleigh, sigma_vs_temperature_for_rayleigh_ranges = _compute_bk_vs_sigma_for_rayleigh_ranges(
            sigma_values=sigma_temperature,
            crystal_length_m=context.crystal_length_m,
            rayleigh_ranges_m=rayleigh_ranges_arr,
        )
        bk_vs_delta_lambda_lengths, _ = _compute_bk_vs_sigma_for_lengths(
            sigma_values=sigma_detuning,
            crystal_lengths_m=crystal_lengths_arr,
            rayleigh_range_m=reference_state["zR_m_reference"],
        )
    else:
        bk_vs_temperature_lengths, sigma_vs_temperature_for_lengths = compute_bk_vs_temperature_for_lengths(
            temperature_grid=temperature_grid,
            crystal_lengths_m=crystal_lengths_arr,
            rayleigh_range_m=reference_state["zR_m_reference"],
            wavelength_p_m=wavelength_p_m,
            wavelength_s_m=wavelength_s_m,
            wavelength_i_m=wavelength_i_m,
            n_p_of_T=n_p_of_T,
            n_s_of_T=n_s_of_T,
            n_i_of_T=n_i_of_T,
            Lambda0_m=reference_state["lambda_reference_m"],
            T0_K=T0_K,
            alpha_perK=alpha_perK,
            qpm_order_m=qpm_order_m,
        )
        bk_vs_wavelength_lengths = compute_bk_vs_wavelength_for_lengths(
            wavelength_grid_m=wavelength_grid_m,
            crystal_lengths_m=crystal_lengths_arr,
            rayleigh_range_m=reference_state["zR_m_reference"],
            wavelength_p_m=wavelength_p_m,
            wavelength_s_m=wavelength_s_m,
            wavelength_i_m=wavelength_i_m,
            wavelength_scan_temperature_K=reference_state["wavelength_scan_temperature_K"],
            n_p_of_lambda_T=n_p_of_lambda_T,
            n_s_of_lambda_T=n_s_of_lambda_T,
            n_i_of_lambda_T=n_i_of_lambda_T,
            Lambda0_m=reference_state["lambda_reference_m"],
            T0_K=T0_K,
            alpha_perK=alpha_perK,
            qpm_order_m=qpm_order_m,
        )
        bk_vs_temperature_rayleigh, sigma_vs_temperature_for_rayleigh_ranges = compute_bk_vs_temperature_for_rayleigh_ranges(
            temperature_grid=temperature_grid,
            crystal_length_m=context.crystal_length_m,
            rayleigh_ranges_m=rayleigh_ranges_arr,
            wavelength_p_m=wavelength_p_m,
            wavelength_s_m=wavelength_s_m,
            wavelength_i_m=wavelength_i_m,
            n_p_of_T=n_p_of_T,
            n_s_of_T=n_s_of_T,
            n_i_of_T=n_i_of_T,
            Lambda0_m=reference_state["lambda_reference_m"],
            T0_K=T0_K,
            alpha_perK=alpha_perK,
            qpm_order_m=qpm_order_m,
        )
        bk_vs_delta_lambda_lengths = compute_bk_vs_detuning_for_lengths(
            delta_lambda_grid_m=delta_lambda_grid_m,
            crystal_lengths_m=crystal_lengths_arr,
            rayleigh_range_m=reference_state["zR_m_reference"],
            wavelength_p_m=wavelength_p_m,
            wavelength_s_m=wavelength_s_m,
            wavelength_i_m=wavelength_i_m,
            wavelength_scan_temperature_K=reference_state["wavelength_scan_temperature_K"],
            n_p_of_lambda_T=n_p_of_lambda_T,
            n_s_of_lambda_T=n_s_of_lambda_T,
            n_i_of_lambda_T=n_i_of_lambda_T,
            Lambda0_m=reference_state["lambda_reference_m"],
            T0_K=T0_K,
            alpha_perK=alpha_perK,
            qpm_order_m=qpm_order_m,
        )

    xi_for_length_sweep = np.asarray(
        [focusing_parameter_xi(crystal_length_m, reference_state["zR_m_reference"]) for crystal_length_m in crystal_lengths_arr],
        dtype=float,
    )
    xi_for_rayleigh_sweep = np.asarray(
        [focusing_parameter_xi(reference_state["crystal_length_reference_m"], rayleigh_range_m) for rayleigh_range_m in rayleigh_ranges_arr],
        dtype=float,
    )

    reference = _build_reference_metadata(
        context=context,
        mode_matching=mode_matching,
        n_p_of_T=n_p_of_T,
        n_s_of_T=n_s_of_T,
        n_i_of_T=n_i_of_T,
        wavelength_p_m=wavelength_p_m,
        wavelength_s_m=wavelength_s_m,
        wavelength_i_m=wavelength_i_m,
        T0_K=T0_K,
        alpha_perK=alpha_perK,
        qpm_order_m=qpm_order_m,
        bk_config=bk_config,
        reference_state=reference_state,
        crystal_length_scale_factors_arr=crystal_length_scale_factors_arr,
        rayleigh_range_scale_factors_arr=rayleigh_range_scale_factors_arr,
    )

    return BKAnalysisResult(
        bk_master_sigma_values=np.asarray(bk_master_map["sigma_values"], dtype=float),
        bk_master_xi_values=np.asarray(bk_master_map["xi_values"], dtype=float),
        bk_master_h_map=np.asarray(bk_master_map["h_bk_map"], dtype=float),
        bk_master_sigma_opt=float(bk_master_map["sigma_opt"]),
        bk_master_xi_opt=float(bk_master_map["xi_opt"]),
        bk_master_h_opt=float(bk_master_map["h_bk_opt"]),
        qpm_length_over_lcoh=np.asarray(qpm_map["length_over_lcoh"], dtype=float),
        qpm_poling_over_lcoh=np.asarray(qpm_map["poling_over_lcoh"], dtype=float),
        qpm_relative_field_intensity=np.asarray(qpm_map["relative_field_intensity"], dtype=float),
        qpm_slice_values_over_lcoh=np.asarray(qpm_map["slice_values_over_lcoh"], dtype=float),
        qpm_slice_curves=np.asarray(qpm_map["slice_curves"], dtype=float),
        qpm_first_order_qpm_guide_over_lcoh=np.asarray(qpm_map["first_order_qpm_guide_over_lcoh"], dtype=float),
        qpm_operating_length_over_lcoh=float(
            context.crystal_length_m / reference["l_coh_reference_m"]
        ),
        qpm_operating_poling_over_lcoh=float(
            (0.5 * reference_state["lambda_reference_m"]) / reference["l_coh_reference_m"]
        ),
        qpm_reference_in_display_range=bool(
            np.isfinite(context.crystal_length_m / reference["l_coh_reference_m"])
            and np.isfinite((0.5 * reference_state["lambda_reference_m"]) / reference["l_coh_reference_m"])
            and 0.0 <= (context.crystal_length_m / reference["l_coh_reference_m"]) <= float(np.nanmax(qpm_map["length_over_lcoh"]))
            and 0.0 <= ((0.5 * reference_state["lambda_reference_m"]) / reference["l_coh_reference_m"]) <= float(np.nanmax(qpm_map["poling_over_lcoh"]))
        ),
        temperature_K=temperature_grid,
        temperature_C=temperature_grid_C,
        crystal_lengths_m=crystal_lengths_arr,
        rayleigh_ranges_m=rayleigh_ranges_arr,
        wavelength_m=wavelength_grid_m,
        delta_lambda_m=delta_lambda_grid_m,
        bk_vs_temperature_for_lengths=bk_vs_temperature_lengths,
        bk_vs_wavelength_for_lengths=bk_vs_wavelength_lengths,
        bk_vs_temperature_for_rayleigh_ranges=bk_vs_temperature_rayleigh,
        bk_vs_delta_lambda_for_lengths=bk_vs_delta_lambda_lengths,
        sigma_vs_temperature_for_lengths=sigma_vs_temperature_for_lengths,
        sigma_vs_temperature_for_rayleigh_ranges=sigma_vs_temperature_for_rayleigh_ranges,
        xi_for_length_sweep=xi_for_length_sweep,
        xi_for_rayleigh_sweep=xi_for_rayleigh_sweep,
        reference=reference,
    )


def run_bk_analysis_pair(
    context,
    mode_matching,
    n_p_of_T,
    n_s_of_T,
    n_i_of_T,
    n_p_of_lambda_T,
    n_s_of_lambda_T,
    n_i_of_lambda_T,
    wavelength_p_m: float,
    wavelength_s_m: float,
    wavelength_i_m: float,
    Lambda0_m: float,
    T0_K: float = 293.15,
    alpha_perK: float = 0.0,
    qpm_order_m: int = 1,
    phase_matching=None,
    n_temperature: int = 201,
    bk_config: BKAnalysisConfig | None = None,
) -> dict[str, BKAnalysisResult]:
    """Return both operating-point and optimal-reference BK analyses."""
    return {
        "bk_analysis_operating": run_bk_analysis(
            context=context,
            mode_matching=mode_matching,
            n_p_of_T=n_p_of_T,
            n_s_of_T=n_s_of_T,
            n_i_of_T=n_i_of_T,
            n_p_of_lambda_T=n_p_of_lambda_T,
            n_s_of_lambda_T=n_s_of_lambda_T,
            n_i_of_lambda_T=n_i_of_lambda_T,
            wavelength_p_m=wavelength_p_m,
            wavelength_s_m=wavelength_s_m,
            wavelength_i_m=wavelength_i_m,
            Lambda0_m=Lambda0_m,
            T0_K=T0_K,
            alpha_perK=alpha_perK,
            qpm_order_m=qpm_order_m,
            phase_matching=phase_matching,
            n_temperature=n_temperature,
            bk_config=bk_config,
            reference_mode="operating",
        ),
        "bk_analysis_optimal": run_bk_analysis(
            context=context,
            mode_matching=mode_matching,
            n_p_of_T=n_p_of_T,
            n_s_of_T=n_s_of_T,
            n_i_of_T=n_i_of_T,
            n_p_of_lambda_T=n_p_of_lambda_T,
            n_s_of_lambda_T=n_s_of_lambda_T,
            n_i_of_lambda_T=n_i_of_lambda_T,
            wavelength_p_m=wavelength_p_m,
            wavelength_s_m=wavelength_s_m,
            wavelength_i_m=wavelength_i_m,
            Lambda0_m=Lambda0_m,
            T0_K=T0_K,
            alpha_perK=alpha_perK,
            qpm_order_m=qpm_order_m,
            phase_matching=phase_matching,
            n_temperature=n_temperature,
            bk_config=bk_config,
            reference_mode="optimal",
        ),
    }


__all__ = [
    "compute_focusing_parameter",
    "focusing_parameter_xi",
    "sigma_parameter",
    "beam_waist_from_rayleigh_range",
    "boyd_kleinman_integral",
    "boyd_kleinman_efficiency",
    "BKAnalysisConfig",
    "BKAnalysisResult",
    "bk_analysis_result_to_dict",
    "compute_bk_master_map",
    "compute_qpm_length_poling_map",
    "run_bk_analysis",
    "run_bk_analysis_pair",
]
