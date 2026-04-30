"""Frequency-domain squeezing spectra built from the OPO Langevin scaffold.

This module defines the analysis-frequency grid and evaluates a compact
below-threshold degenerate OPO noise model from the existing linearized
Langevin state-space matrices.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np

try:
    from ..common.constants import TWO_PI
except ImportError:
    from common.constants import TWO_PI

from .opo_langevin import OPOLangevinModel
from .opo_model import OPOModelResult, OPOParameters


@dataclass(frozen=True)
class OPOSqueezingSpectrum:
    """Frequency-domain OPO squeezing output."""

    frequency_Hz: np.ndarray
    squeezing_spectrum: np.ndarray
    antisqueezing_spectrum: np.ndarray
    measured_quadrature_spectrum: np.ndarray
    shot_noise_reference: np.ndarray
    optimal_phase_rad: np.ndarray
    lo_phase_rad: float
    notes: tuple[str, ...]


def _compute_minimal_output_spectral_density(
    drift_matrix: np.ndarray,
    input_matrix: np.ndarray,
    noise_coupling_matrix: np.ndarray,
    linewidth_rad_s: float,
    omega_rad_s: float,
) -> np.ndarray:
    """Return a compact output spectral density for the current 2x2 model.

    This helper is intentionally minimal and only targets the present
    below-threshold 2x2 quadrature model. It should not be interpreted as a
    full general quantum input-output construction.
    """
    identity = np.eye(drift_matrix.shape[0], dtype=complex)
    response = np.linalg.inv(1j * float(omega_rad_s) * identity - drift_matrix.astype(complex))
    coupling_scale = np.sqrt(2.0 * float(linewidth_rad_s))
    effective_input = coupling_scale * input_matrix.astype(complex)
    effective_noise = coupling_scale * noise_coupling_matrix.astype(complex)
    output_transfer = identity - effective_input @ response @ effective_noise
    return output_transfer @ output_transfer.T.conj()


def build_analysis_frequency_grid(parameters: OPOParameters) -> np.ndarray:
    """Build the analysis-frequency axis used for spectrum calculations."""
    frequency_min_hz, frequency_max_hz = parameters.analysis_span_Hz
    if parameters.n_analysis_points < 2:
        raise ValueError("n_analysis_points must be at least 2")
    if frequency_min_hz < 0.0 or frequency_max_hz <= frequency_min_hz:
        raise ValueError("analysis_span_Hz must satisfy 0 <= frequency_min_hz < frequency_max_hz")
    return np.linspace(frequency_min_hz, frequency_max_hz, parameters.n_analysis_points, dtype=float)


def _real_symmetric_quadrature_spectrum(output_spectral_density: np.ndarray) -> np.ndarray:
    """Return the real symmetric quadrature spectral-density matrix."""
    spectrum = np.asarray(output_spectral_density, dtype=complex)
    real_spectrum = np.real(spectrum)
    return 0.5 * (real_spectrum + real_spectrum.T)


def _apply_vacuum_loss(spectrum: np.ndarray, efficiency: float) -> np.ndarray:
    """Mix a shot-noise-normalized spectrum with vacuum loss."""
    return 1.0 - efficiency + efficiency * spectrum


def _validate_ordered_principal_spectra(
    squeezing: np.ndarray,
    antisqueezing: np.ndarray,
    *,
    check_high_frequency_limit: bool,
    high_frequency_rtol: float = 0.2,
) -> None:
    """Validate non-negative, ordered, shot-noise-normalized principal spectra."""
    if np.any(~np.isfinite(squeezing)) or np.any(~np.isfinite(antisqueezing)):
        raise FloatingPointError("Non-finite principal squeezing spectrum value.")
    if np.any(squeezing < 0.0):
        raise FloatingPointError("Principal squeezing spectrum contains a negative linear value.")
    if np.any(antisqueezing < 0.0):
        raise FloatingPointError("Principal antisqueezing spectrum contains a negative linear value.")
    if np.any(squeezing > antisqueezing):
        raise FloatingPointError("Principal squeezing spectrum exceeds antisqueezing spectrum.")

    if check_high_frequency_limit and squeezing.size:
        high_squeezing = float(squeezing[-1])
        high_antisqueezing = float(antisqueezing[-1])
        if not (
            np.isclose(high_squeezing, 1.0, rtol=high_frequency_rtol, atol=0.0)
            and np.isclose(high_antisqueezing, 1.0, rtol=high_frequency_rtol, atol=0.0)
        ):
            warnings.warn(
                "Highest computed OPO analysis frequency is not yet close to the shot-noise asymptote "
                f"(squeezing={high_squeezing:.6g}, antisqueezing={high_antisqueezing:.6g}).",
                RuntimeWarning,
                stacklevel=2,
            )


def compute_squeezing_spectra(
    parameters: OPOParameters,
    model: OPOModelResult,
    langevin: OPOLangevinModel,
) -> OPOSqueezingSpectrum:
    """Return below-threshold quadrature spectra from the Langevin matrices."""

    frequency_hz = build_analysis_frequency_grid(parameters)
    shot_noise = np.ones_like(frequency_hz, dtype=float)

    sigma = float(model.pump_parameter)
    if sigma >= 1.0:
        raise ValueError(
            "compute_squeezing_spectra only supports the below-threshold model with sigma < 1. "
            f"Received sigma={sigma:.6f} for pump power {model.pump_power_W:.6e} W and "
            f"external threshold {model.effective_threshold_power_W:.6e} W."
        )

    eta_esc = float(np.clip(model.escape_efficiency, 0.0, 1.0))
    eta_det = float(np.clip(parameters.detection_efficiency, 0.0, 1.0))
    theta = float(parameters.lo_phase_rad)
    measurement_vector = np.array([np.cos(theta), np.sin(theta)], dtype=float)

    omega = TWO_PI * frequency_hz
    linewidth_rad_s = max(TWO_PI * max(float(model.cavity_kappa_total_Hz), 0.0), np.finfo(float).eps)

    drift_matrix = np.asarray(langevin.drift_matrix, dtype=float)
    input_matrix = np.asarray(langevin.input_matrix, dtype=float)
    noise_coupling_matrix = np.asarray(langevin.noise_coupling_matrix, dtype=float)

    squeezing_internal = np.empty_like(frequency_hz, dtype=float)
    antisqueezing_internal = np.empty_like(frequency_hz, dtype=float)
    measured_quadrature_output = np.empty_like(frequency_hz, dtype=float)
    optimal_phase_rad = np.empty_like(frequency_hz, dtype=float)
    failed_points = 0

    for i, omega_i in enumerate(omega):
        try:
            output_sd = _compute_minimal_output_spectral_density(
                drift_matrix,
                input_matrix,
                noise_coupling_matrix,
                linewidth_rad_s,
                omega_i,
            )
            symmetric_spectrum = _real_symmetric_quadrature_spectrum(output_sd)
            eigenvalues, eigenvectors = np.linalg.eigh(symmetric_spectrum)

            squeezing_value = float(eigenvalues[0])
            antisqueezing_value = float(eigenvalues[-1])
            numerical_floor = -100.0 * np.finfo(float).eps
            if squeezing_value < numerical_floor or antisqueezing_value < numerical_floor:
                raise FloatingPointError("Negative principal spectrum eigenvalue.")

            squeezing_internal[i] = max(squeezing_value, 0.0)
            antisqueezing_internal[i] = max(antisqueezing_value, 0.0)

            measured_value = float(measurement_vector @ symmetric_spectrum @ measurement_vector)
            if measured_value < numerical_floor:
                raise FloatingPointError("Negative measured quadrature spectrum value.")
            measured_quadrature_output[i] = max(measured_value, 0.0)

            optimal_phase_rad[i] = float(np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0]))

            if (
                not np.isfinite(squeezing_internal[i])
                or not np.isfinite(antisqueezing_internal[i])
                or not np.isfinite(measured_quadrature_output[i])
                or not np.isfinite(optimal_phase_rad[i])
            ):
                raise FloatingPointError("Non-finite spectrum value.")
        except (np.linalg.LinAlgError, ValueError, FloatingPointError):
            squeezing_internal[i] = 1.0
            antisqueezing_internal[i] = 1.0
            measured_quadrature_output[i] = 1.0
            optimal_phase_rad[i] = theta
            failed_points += 1

    squeezing_coupled = _apply_vacuum_loss(squeezing_internal, eta_esc)
    antisqueezing_coupled = _apply_vacuum_loss(antisqueezing_internal, eta_esc)
    measured_coupled = _apply_vacuum_loss(measured_quadrature_output, eta_esc)

    # Detection loss mixes the output with vacuum and pulls both spectra
    # toward the shot-noise reference.
    squeezing = _apply_vacuum_loss(squeezing_coupled, eta_det)
    antisqueezing = _apply_vacuum_loss(antisqueezing_coupled, eta_det)
    measured = _apply_vacuum_loss(measured_coupled, eta_det)

    max_frequency_is_asymptotic = bool(float(omega[-1]) >= 10.0 * linewidth_rad_s)
    _validate_ordered_principal_spectra(
        squeezing,
        antisqueezing,
        check_high_frequency_limit=max_frequency_is_asymptotic,
    )
    squeezing_db = 10.0 * np.log10(np.maximum(squeezing, np.finfo(float).eps))
    antisqueezing_db = 10.0 * np.log10(np.maximum(antisqueezing, np.finfo(float).eps))
    if np.any(squeezing_db > antisqueezing_db):
        warnings.warn(
            "Squeezing dB exceeds antisqueezing dB at one or more frequencies.",
            RuntimeWarning,
            stacklevel=2,
        )

    notes = [
        "Spectrum computed from a minimal output spectral-density construction in a 2x2 quadrature basis.",
        "For each frequency, squeezing is the minimum eigenvalue of the real symmetric quadrature spectral-density matrix.",
        "For each frequency, antisqueezing is the maximum eigenvalue of the same matrix.",
        f"Measured quadrature defined by homodyne LO phase theta = {theta:.6f} rad.",
        "optimal_phase_rad stores the phase of the minimum-noise quadrature at each analysis frequency.",
        "All spectra use one common shot-noise reference; principal spectra are not normalized independently.",
        f"Cavity detuning used in the quadrature response: {float(model.cavity_detuning_Hz):.6f} Hz.",
        "Escape efficiency and detection efficiency are included phenomenologically.",
    ]
    if not max_frequency_is_asymptotic:
        notes.append(
            "The highest requested analysis frequency is below 10 cavity linewidths, "
            "so the finite plotted endpoint is not treated as the asymptotic shot-noise limit."
        )
    if failed_points:
        notes.append(
            f"{failed_points} frequency point(s) fell back to shot noise after a response-matrix inversion failure."
        )

    return OPOSqueezingSpectrum(
        frequency_Hz=frequency_hz,
        squeezing_spectrum=np.asarray(squeezing, dtype=float),
        antisqueezing_spectrum=np.asarray(antisqueezing, dtype=float),
        measured_quadrature_spectrum=np.asarray(measured, dtype=float),
        shot_noise_reference=shot_noise,
        optimal_phase_rad=np.asarray(optimal_phase_rad, dtype=float),
        lo_phase_rad=theta,
        notes=tuple(notes),
    )


def spectrum_to_dict(spectrum: OPOSqueezingSpectrum) -> dict[str, list[float] | list[str]]:
    """Convert the squeezing spectrum dataclass into a JSON-friendly mapping."""
    return {
        "frequency_Hz": spectrum.frequency_Hz.tolist(),
        "squeezing_spectrum": spectrum.squeezing_spectrum.tolist(),
        "antisqueezing_spectrum": spectrum.antisqueezing_spectrum.tolist(),
        "measured_quadrature_spectrum": spectrum.measured_quadrature_spectrum.tolist(),
        "shot_noise_reference": spectrum.shot_noise_reference.tolist(),
        "optimal_phase_rad": spectrum.optimal_phase_rad.tolist(),
        "lo_phase_rad": float(spectrum.lo_phase_rad),
        "notes": list(spectrum.notes),
    }


__all__ = [
    "OPOSqueezingSpectrum",
    "build_analysis_frequency_grid",
    "compute_squeezing_spectra",
    "spectrum_to_dict",
]
