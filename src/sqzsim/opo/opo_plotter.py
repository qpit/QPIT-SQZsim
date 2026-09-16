"""Plotting helpers for OPO squeezing-spectrum and resonance diagnostics."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

try:
    from ..common.constants import C_M_PER_S, TWO_PI
except ImportError:
    from common.constants import C_M_PER_S, TWO_PI


def plot_opo_spectrum_summary(spectrum: dict[str, list[float]]):
    """Plot principal squeezing, antisqueezing, and shot-noise reference spectra."""
    frequency_hz = np.asarray(spectrum["frequency_Hz"], dtype=float)
    squeezing = np.maximum(np.asarray(spectrum["squeezing_spectrum"], dtype=float), np.finfo(float).eps)
    antisqueezing = np.maximum(np.asarray(spectrum["antisqueezing_spectrum"], dtype=float), np.finfo(float).eps)
    measured = np.maximum(
        np.asarray(spectrum.get("measured_quadrature_spectrum", spectrum["squeezing_spectrum"]), dtype=float),
        np.finfo(float).eps,
    )
    shot_noise = np.maximum(np.asarray(spectrum["shot_noise_reference"], dtype=float), np.finfo(float).eps)
    theta = float(spectrum.get("lo_phase_rad", 0.0))

    squeezing_db = 10.0 * np.log10(squeezing)
    antisqueezing_db = 10.0 * np.log10(antisqueezing)
    measured_db = 10.0 * np.log10(measured)
    shot_noise_db = 10.0 * np.log10(shot_noise)

    fig, ax = plt.subplots(figsize=(10.5, 5.6))
    ax.plot(frequency_hz, squeezing_db, lw=2.8, label="Squeezing (principal min)")
    ax.plot(frequency_hz, antisqueezing_db, lw=2.8, label="Antisqueezing (principal max)")
    ax.plot(frequency_hz, measured_db, "--", lw=2.4, label=fr"Measured LO ($\theta={theta:.2f}$ rad)")
    ax.plot(frequency_hz, shot_noise_db, "--", lw=1.6, color="#666666", label="Shot noise")
    ax.axhline(0.0, color="#666666", ls="--", lw=1.0, alpha=0.6)

    ax.set_xlabel("Analysis frequency [Hz]")
    ax.set_ylabel("Noise [dB]")
    ax.set_title("OPO squeezing spectrum", fontsize=15, fontweight="semibold", pad=10)

    ax.set_facecolor("white")
    ax.grid(True, color="#b0b0b0", alpha=0.5, linewidth=0.8)
    ax.tick_params(labelsize=10)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=True, facecolor="white", edgecolor="#cccccc", fontsize=10, loc="best")

    fig.subplots_adjust(left=0.10, right=0.98, bottom=0.14, top=0.90)
    return fig


def _lorentzian_sum(frequency_hz: np.ndarray, centers_hz: np.ndarray, gamma_hz: float) -> np.ndarray:
    """Return a summed Lorentzian resonance comb."""
    gamma = max(float(gamma_hz), np.finfo(float).eps)
    response = np.zeros_like(frequency_hz, dtype=float)
    for center_hz in np.asarray(centers_hz, dtype=float):
        response += 1.0 / (1.0 + ((frequency_hz - center_hz) / gamma) ** 2)
    max_value = np.max(response)
    if max_value > 0.0:
        response = response / max_value
    return response


def _style_opo_axes(ax) -> None:
    ax.set_facecolor("white")
    ax.grid(True, color="#b0b0b0", alpha=0.45, linewidth=0.8)
    ax.tick_params(labelsize=10)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def plot_opo_resonance_diagnostic(
    spectrum: dict[str, list[float]],
    model: dict[str, float | str | bool] | None = None,
    crystal_results: dict[str, object] | None = None,
):
    """Plot a diagnostic longitudinal-mode comb and broad gain envelope.

    This is a visualization aid for the current cavity/crystal operating point,
    not a full cavity transfer-function or quantum-noise calculation.
    """
    crystal_results = crystal_results or {}
    active_for_opo = crystal_results.get("active_for_opo", {})
    if not isinstance(active_for_opo, dict):
        active_for_opo = {}
    resonance = active_for_opo.get(
        "polarization_resonance",
        crystal_results.get(
            "active_polarization_resonance",
            crystal_results.get("polarization_resonance", {}),
        ),
    )
    if not resonance:
        raise ValueError("Crystal results missing polarization_resonance diagnostic")

    fsr_signal_hz = resonance.get("fsr_signal_Hz")
    if fsr_signal_hz is None and "signal_optical_roundtrip_length_m" in resonance:
        fsr_signal_hz = C_M_PER_S / float(resonance["signal_optical_roundtrip_length_m"])
    fsr_idler_hz = resonance.get("fsr_idler_Hz")
    if fsr_idler_hz is None and "idler_optical_roundtrip_length_m" in resonance:
        fsr_idler_hz = C_M_PER_S / float(resonance["idler_optical_roundtrip_length_m"])
    if fsr_signal_hz is None or fsr_idler_hz is None:
        raise KeyError("polarization_resonance is missing FSR data and optical round-trip lengths")
    fsr_signal_hz = float(fsr_signal_hz)
    fsr_idler_hz = float(fsr_idler_hz)
    delta_fsr_hz = float(resonance.get("delta_fsr_Hz", fsr_signal_hz - fsr_idler_hz))
    delta_phi_wrapped_rad = float(resonance.get("delta_phi_wrapped_rad", 0.0))
    is_double_resonant = bool(resonance.get("is_double_resonant", False))

    linewidth_hz = None
    if model is not None:
        linewidth_hz = (
            model.get("cavity_kappa_total_Hz")
            or model.get("cavity_linewidth_Hz")
            or model.get("kappa_total_Hz")
        )
    if linewidth_hz is None:
        frequency_axis_hz = np.asarray(spectrum["frequency_Hz"], dtype=float)
        linewidth_hz = max(float(np.ptp(frequency_axis_hz)) / 50.0, 1.0)
    linewidth_hz = max(float(linewidth_hz), 1.0)
    gamma_hz = max(0.5 * linewidth_hz, 1.0)

    mean_fsr_hz = 0.5 * (fsr_signal_hz + fsr_idler_hz)
    mode_count = 5
    mode_indices = np.arange(-mode_count, mode_count + 1, dtype=float)
    center_offset_hz = (delta_phi_wrapped_rad / TWO_PI) * mean_fsr_hz

    signal_centers_hz = mode_indices * fsr_signal_hz
    idler_centers_hz = center_offset_hz + mode_indices * fsr_idler_hz

    window_half_width_hz = max(
        mode_count * max(fsr_signal_hz, fsr_idler_hz) + abs(center_offset_hz) + 4.0 * linewidth_hz,
        4.0 * linewidth_hz,
    )
    frequency_hz = np.linspace(-window_half_width_hz, window_half_width_hz, 4000, dtype=float)

    signal_comb = _lorentzian_sum(frequency_hz, signal_centers_hz, gamma_hz)
    idler_comb = _lorentzian_sum(frequency_hz, idler_centers_hz, gamma_hz)

    envelope_width_hz = max(2.5 * mean_fsr_hz, 3.0 * linewidth_hz)
    envelope = np.exp(-0.5 * (frequency_hz / envelope_width_hz) ** 2)
    signal_weighted = signal_comb * envelope
    idler_weighted = idler_comb * envelope

    fig, axes = plt.subplots(2, 1, figsize=(11.0, 7.6), sharex=True, height_ratios=[1.0, 1.1])
    signal_color = "#1f77b4"
    idler_color = "#d95f02"
    envelope_color = "#2f2f2f"

    ax_top, ax_bottom = axes
    ax_top.plot(frequency_hz, signal_comb, color=signal_color, lw=2.1, label="Signal resonance comb")
    ax_top.plot(frequency_hz, idler_comb, color=idler_color, lw=2.1, label="Idler resonance comb")
    ax_top.set_ylabel("Relative response")
    ax_top.set_title("OPO longitudinal resonance diagnostic", fontsize=15, fontweight="semibold", pad=10)
    ax_top.legend(frameon=True, facecolor="white", edgecolor="#cccccc", fontsize=10, loc="upper right")

    ax_bottom.fill_between(frequency_hz, 0.0, envelope, color="#d9d9d9", alpha=0.35, zorder=0)
    ax_bottom.plot(frequency_hz, envelope, color=envelope_color, lw=2.4, label="Gain envelope")
    ax_bottom.plot(frequency_hz, signal_weighted, color=signal_color, lw=2.0, label="Signal resonances")
    ax_bottom.plot(frequency_hz, idler_weighted, color=idler_color, lw=2.0, label="Idler resonances")
    ax_bottom.set_xlabel("Frequency detuning [Hz]")
    ax_bottom.set_ylabel("Relative response")
    ax_bottom.legend(frameon=True, facecolor="white", edgecolor="#cccccc", fontsize=10, loc="upper right")

    annotation = "\n".join(
        [
            f"FSR signal: {fsr_signal_hz:.3e} Hz",
            f"FSR idler:  {fsr_idler_hz:.3e} Hz",
            f"ΔFSR:       {delta_fsr_hz:.3e} Hz",
            f"Δφ wrapped: {delta_phi_wrapped_rad:.3f} rad",
            f"Double resonant: {is_double_resonant}",
        ]
    )
    ax_bottom.text(
        0.02,
        0.97,
        annotation,
        transform=ax_bottom.transAxes,
        va="top",
        ha="left",
        fontsize=9.5,
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "edgecolor": "#cccccc", "alpha": 0.95},
    )

    for ax in axes:
        _style_opo_axes(ax)
        ax.ticklabel_format(axis="x", style="sci", scilimits=(0, 0))
        ax.set_ylim(bottom=0.0)

    fig.subplots_adjust(left=0.10, right=0.98, bottom=0.10, top=0.92, hspace=0.12)
    return fig

__all__ = [
    "plot_opo_spectrum_summary",
    "plot_opo_resonance_diagnostic",
]
