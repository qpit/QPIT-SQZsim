"""Plotting utilities for crystal phase-matching and resonance diagnostics."""

from __future__ import annotations

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np
from typing import Any

from .crystal_boyd_kleinman import boyd_kleinman_efficiency


def plot_phase_matching_temperature_scan(scan: dict[str, np.ndarray]):
    """Plot phase-matching power and mismatch versus temperature.

    This legacy figure remains available for direct visualization, but it does
    not participate in the BK analysis workflow.
    """
    T_K = np.asarray(scan["T_K"], dtype=float)
    pm = np.asarray(scan["pm_power"], dtype=float)
    dk = np.asarray(scan["delta_k_rad_per_m"], dtype=float)
    dk_eff = np.asarray(scan["delta_k_eff_rad_per_m"], dtype=float)

    fig, axes = plt.subplots(2, 1, figsize=(8, 7), sharex=True)
    axes[0].plot(T_K, pm, lw=2)
    axes[0].set_ylabel("PM power factor")
    axes[0].grid(alpha=0.3)

    axes[1].plot(T_K, dk, lw=1.8, label="Delta k")
    axes[1].plot(T_K, dk_eff, lw=1.8, label="Delta k eff")
    axes[1].set_xlabel("Temperature [K]")
    axes[1].set_ylabel("Mismatch [rad/m]")
    axes[1].grid(alpha=0.3)
    axes[1].legend()

    fig.tight_layout()
    return fig


def plot_mode_matching_summary(mode_result: dict[str, float]):
    """Plot a compact legacy bar chart for mode-matching quantities only."""
    labels = [
        "waist [um]",
        "zR [mm]",
        "confocal [mm]",
        "xi [-]",
    ]
    values = [
        float(mode_result["waist_crystal_m"]) * 1e6,
        float(mode_result["rayleigh_range_m"]) * 1e3,
        float(mode_result["confocal_parameter_m"]) * 1e3,
        float(mode_result.get("focusing_parameter_xi", mode_result["focusing_parameter"])),
    ]

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(labels, values)
    ax.set_title("Crystal mode-matching summary")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    return fig


def plot_boyd_kleinman_vs_focusing_parameter(
    xi_values: np.ndarray,
    crystal_length_m: float,
    delta_k_rad_per_m: float,
):
    """Plot BK efficiency factor versus focusing parameter.

    Deprecated legacy helper kept for compatibility with old notebooks.
    """
    xi_values = np.asarray(xi_values, dtype=float)
    eff = np.empty_like(xi_values)
    for i, xi in enumerate(xi_values):
        if xi <= 0:
            eff[i] = np.nan
            continue
        z_r = crystal_length_m / (2.0 * xi)
        eff[i] = boyd_kleinman_efficiency(
            rayleigh_range_m=z_r,
            crystal_length_m=crystal_length_m,
            delta_k=delta_k_rad_per_m,
        )

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(xi_values, eff, lw=2)
    ax.set_xlabel("Focusing parameter xi")
    ax.set_ylabel("Boyd-Kleinman efficiency")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig


def plot_boyd_kleinman_vs_delta_k(
    delta_k_values: np.ndarray,
    waist_m: float,
    rayleigh_range_m: float,
    crystal_length_m: float,
):
    """Plot BK efficiency factor versus phase mismatch Δk.

    Deprecated legacy helper kept for compatibility with old notebooks.
    The ``waist_m`` argument is ignored and kept only for API stability.
    """
    _ = waist_m
    delta_k_values = np.asarray(delta_k_values, dtype=float)
    eff = np.array(
        [
            boyd_kleinman_efficiency(
                rayleigh_range_m=rayleigh_range_m,
                crystal_length_m=crystal_length_m,
                delta_k=dk,
            )
            for dk in delta_k_values
        ],
        dtype=float,
    )

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(delta_k_values, eff, lw=2)
    ax.set_xlabel("Delta k [rad/m]")
    ax.set_ylabel("Boyd-Kleinman efficiency")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig


def plot_bk_master_map_sigma_xi(
    bk_data: dict,
    operating_point: dict | None = None,
    operating_point_label: str = "Operating point",
):
    """Plot the BK master map ``h_BK(sigma, xi)`` with its numerical optimum."""
    sigma_values = np.asarray(bk_data["bk_master_sigma_values"], dtype=float)
    xi_values = np.asarray(bk_data["bk_master_xi_values"], dtype=float)
    h_bk_map = np.asarray(bk_data["bk_master_h_map"], dtype=float)
    sigma_opt = float(bk_data["bk_master_sigma_opt"])
    xi_opt = float(bk_data["bk_master_xi_opt"])
    h_bk_opt = float(bk_data["bk_master_h_opt"])

    sigma_grid, xi_grid = np.meshgrid(sigma_values, xi_values)
    fig, ax = plt.subplots(figsize=(8.6, 6.6))
    contourf = ax.contourf(sigma_grid, xi_grid, h_bk_map, levels=18, cmap="viridis")
    ax.contour(sigma_grid, xi_grid, h_bk_map, levels=10, colors="black", linewidths=0.6, alpha=0.4)
    ax.scatter(
        sigma_opt,
        xi_opt,
        marker="x",
        color="black",
        s=100,
        label="Optimal BK point",
        zorder=6,
    )
    ax.set_xlabel(r"Phase mismatch factor: $\sigma$")
    ax.set_ylabel(r"Focusing strength factor: $\xi$")
    ax.set_title(r"Boyd-Kleinman master map $h_{\mathrm{BK}}(\sigma,\xi)$", pad=10)
    ax.set_facecolor("#f7fbf6")
    ax.grid(True, color="#bdd7c0", alpha=0.35, linewidth=0.7)

    sigma_reference_value: float | None = None
    xi_reference_value: float | None = None
    h_bk_operating: float | None = None
    if operating_point is not None:
        sigma_reference = operating_point.get("sigma_reference")
        xi_reference = operating_point.get("xi_reference")
        if sigma_reference is not None and xi_reference is not None:
            sigma_reference = float(sigma_reference)
            xi_reference = float(xi_reference)
            if np.isfinite(sigma_reference) and np.isfinite(xi_reference):
                sigma_reference_value = sigma_reference
                xi_reference_value = xi_reference
                i_sigma_ref = int(np.argmin(np.abs(sigma_values - sigma_reference)))
                i_xi_ref = int(np.argmin(np.abs(xi_values - xi_reference)))
                h_bk_operating = float(h_bk_map[i_xi_ref, i_sigma_ref])
                ax.scatter(
                    sigma_reference,
                    xi_reference,
                    color="red",
                    s=80,
                    label=operating_point_label,
                    zorder=6,
                )
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        unique_handles: list = []
        unique_labels: list[str] = []
        for handle, label in zip(handles, labels):
            if label not in unique_labels:
                unique_handles.append(handle)
                unique_labels.append(label)
        ax.legend(unique_handles, unique_labels, loc="upper right")

    text = (
        "Optimal BK point\n"
        rf"$\sigma_m = {sigma_opt:.3f}$" "\n"
        rf"$\xi_m = {xi_opt:.3f}$" "\n"
        rf"$h_{{\mathrm{{BK}}}}(\sigma_m,\xi_m) = {h_bk_opt:.3f}$"
    )
    if (
        sigma_reference_value is not None
        and xi_reference_value is not None
        and h_bk_operating is not None
        and np.isfinite(h_bk_operating)
    ):
        text += (
            "\n\n"
            f"{operating_point_label}\n"
            rf"$\sigma = {sigma_reference_value:.3f}$" "\n"
            rf"$\xi = {xi_reference_value:.3f}$" "\n"
            rf"$h_{{\mathrm{{BK}}}}(\sigma,\xi) = {h_bk_operating:.3f}$"
        )

    ax.text(
        0.02,
        0.02,
        text,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=10,
        bbox={"facecolor": "white", "edgecolor": "#7b8f7a", "alpha": 0.92, "boxstyle": "round,pad=0.3"},
    )

    cbar = fig.colorbar(contourf, ax=ax, pad=0.02)
    cbar.set_label(r"BK factor: $h_{\mathrm{BK}}(\sigma,\xi)$")
    fig.tight_layout()
    return fig


def plot_qpm_length_poling_map(bk_data: dict):
    """Plot the universal QPM length/poling map with optional operating-point overlay."""
    length_over_lcoh = np.asarray(bk_data["qpm_length_over_lcoh"], dtype=float)
    poling_over_lcoh = np.asarray(bk_data["qpm_poling_over_lcoh"], dtype=float)
    relative_field_intensity = np.asarray(bk_data["qpm_relative_field_intensity"], dtype=float)
    slice_values_over_lcoh = np.asarray(bk_data["qpm_slice_values_over_lcoh"], dtype=float)
    slice_curves = np.asarray(bk_data["qpm_slice_curves"], dtype=float)
    if "qpm_first_order_qpm_guide_over_lcoh" in bk_data:
        first_order_qpm_guide_over_lcoh = np.asarray(
            bk_data["qpm_first_order_qpm_guide_over_lcoh"],
            dtype=float,
        )
    else:
        first_order_qpm_guide_over_lcoh = np.asarray(
            bk_data["qpm_delta_k0_curve_over_lcoh"],
            dtype=float,
        )
    operating_length_over_lcoh = float(
        bk_data.get(
            "qpm_operating_length_over_lcoh",
            bk_data.get("reference", {}).get("crystal_length_reference_m", np.nan),
        )
    )
    operating_poling_over_lcoh = float(
        bk_data.get("qpm_operating_poling_over_lcoh", np.nan)
    )
    reference_in_display_range = bool(
        bk_data.get(
            "qpm_reference_in_display_range",
            np.isfinite(operating_length_over_lcoh)
            and np.isfinite(operating_poling_over_lcoh)
            and 0.0 <= operating_length_over_lcoh <= float(np.nanmax(length_over_lcoh))
            and 0.0 <= operating_poling_over_lcoh <= float(np.nanmax(poling_over_lcoh)),
        )
    )

    fig, (ax_top, ax_bottom) = plt.subplots(
        2,
        1,
        figsize=(9.2, 8.0),
        sharex=True,
        gridspec_kw={"height_ratios": (1.0, 1.35), "hspace": 0.08},
    )

    colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(slice_values_over_lcoh)))
    for i_slice, slice_value in enumerate(slice_values_over_lcoh):
        ax_top.plot(
            length_over_lcoh,
            slice_curves[i_slice],
            color=colors[i_slice],
            lw=2.7,
            label=rf"$\Lambda_{{pol}}/l_{{coh}} = {slice_value:.0f}$",
        )
    guide_curve = np.empty_like(length_over_lcoh)
    for i_length, poling_value in enumerate(first_order_qpm_guide_over_lcoh):
        poling_index = int(np.argmin(np.abs(poling_over_lcoh - poling_value)))
        guide_curve[i_length] = relative_field_intensity[poling_index, i_length]
    ax_top.plot(
        length_over_lcoh,
        guide_curve,
        color="black",
        lw=2.2,
        ls="--",
    )
    ax_top.set_ylabel("Relative field intensity")
    ax_top.set_title("QPM / poling-length map", pad=8)
    ax_top.grid(True, color="#cfcfcf", alpha=0.28, linewidth=0.7)
    ax_top.tick_params(axis="x", labelbottom=False)
    ax_top.legend(
        frameon=False,
        fontsize=9,
        loc="upper right",
    )

    intensity_for_plot = np.clip(relative_field_intensity, 1e-6, None)
    mesh = ax_bottom.pcolormesh(
        length_over_lcoh,
        poling_over_lcoh,
        intensity_for_plot,
        shading="auto",
        cmap="viridis",
        norm=LogNorm(vmin=1e-6, vmax=max(1.0, float(np.nanmax(intensity_for_plot)))),
    )
    x_min = float(np.nanmin(length_over_lcoh))
    x_max = float(np.nanmax(length_over_lcoh))
    if reference_in_display_range and np.isfinite(operating_length_over_lcoh) and np.isfinite(operating_poling_over_lcoh):
        ax_bottom.plot(
            operating_length_over_lcoh,
            operating_poling_over_lcoh,
            marker="o",
            ms=8,
            mfc="#d1495b",
            mec="white",
            mew=1.0,
            linestyle="None",
            label="Operating point",
        )
        ax_bottom.annotate(
            "Operating point",
            xy=(operating_length_over_lcoh, operating_poling_over_lcoh),
            xytext=(6, 6),
            textcoords="offset points",
            ha="left",
            va="bottom",
            fontsize=9,
            bbox={"facecolor": "white", "edgecolor": "#c0c0c0", "alpha": 0.9, "boxstyle": "round,pad=0.2"},
        )
    ax_bottom.set_xlabel(r"Crystal length: $l_{cry} / l_{coh}$")
    ax_bottom.set_ylabel(r"Poling period: $\Lambda_{pol} / l_{coh}$")
    ax_bottom.grid(True, color="#c7d9c8", alpha=0.24, linewidth=0.5)
    ax_bottom.set_xlim(x_min, x_max)
    if reference_in_display_range and np.isfinite(operating_length_over_lcoh) and np.isfinite(operating_poling_over_lcoh):
        ax_bottom.legend(frameon=False, fontsize=9, loc="upper right")

    cbar = fig.colorbar(mesh, ax=ax_bottom, pad=0.02)
    cbar.set_label("Relative field intensity")
    cbar.ax.tick_params(labelsize=10)
    fig.align_ylabels((ax_top, ax_bottom))
    fig.tight_layout(rect=(0.0, 0.0, 0.86, 1.0))
    return fig


def plot_boyd_kleinman_analysis(bk_data: dict, figure_title: str | None = None):
    """Plot a 2x2 publication-style Boyd-Kleinman analysis figure.

    All sweep definitions and BK values must be precomputed in the workflow
    layer and passed through ``bk_data``.
    """
    temperature_C = np.asarray(
        bk_data.get("temperature_C", np.asarray(bk_data["temperature_K"], dtype=float) - 273.15),
        dtype=float,
    )
    crystal_lengths_m = np.asarray(bk_data["crystal_lengths_m"], dtype=float)
    rayleigh_ranges_m = np.asarray(bk_data["rayleigh_ranges_m"], dtype=float)
    crystal_length_reference_m = float(
        bk_data["reference"].get("crystal_length_reference_m", crystal_lengths_m[0])
    )
    crystal_length_reference_mm = crystal_length_reference_m * 1e3
    wavelength_m = np.asarray(bk_data["wavelength_m"], dtype=float)
    delta_lambda_m = np.asarray(bk_data["delta_lambda_m"], dtype=float)

    bk_temp_lengths = np.asarray(bk_data["bk_vs_temperature_for_lengths"], dtype=float)
    bk_wave_lengths = np.asarray(bk_data["bk_vs_wavelength_for_lengths"], dtype=float)
    bk_temp_rayleigh = np.asarray(bk_data["bk_vs_temperature_for_rayleigh_ranges"], dtype=float)
    bk_detuning_lengths = np.asarray(bk_data["bk_vs_delta_lambda_for_lengths"], dtype=float)

    n_colors = max(len(crystal_lengths_m), len(rayleigh_ranges_m))
    if figure_title is not None and "Operating" in figure_title:
        colors = plt.cm.Blues(np.linspace(0.45, 0.9, n_colors))
    else:
        colors = plt.cm.Greens(np.linspace(0.45, 0.9, n_colors))
    fig, axes = plt.subplots(2, 2, figsize=(12.8, 8.6), sharey=True)
    axes = axes.ravel()

    for idx, crystal_length_m in enumerate(crystal_lengths_m):
        color = colors[idx]
        label = rf"$L = {crystal_length_m * 1e3:.0f}\,\mathrm{{mm}}$"
        axes[0].plot(temperature_C, bk_temp_lengths[idx], lw=2.8, color=color, label=label)
        axes[1].plot(wavelength_m * 1e9, bk_wave_lengths[idx], lw=2.8, color=color, label=label)
        axes[3].plot(delta_lambda_m * 1e9, bk_detuning_lengths[idx], lw=2.8, color=color, label=label)

    for idx, rayleigh_range_m in enumerate(rayleigh_ranges_m):
        color = colors[idx]
        label = rf"$z_R = {rayleigh_range_m * 1e3:.0f}\,\mathrm{{mm}}$"
        axes[2].plot(temperature_C, bk_temp_rayleigh[idx], lw=2.8, color=color, label=label)

    title_kwargs = {"fontsize": 13, "fontweight": "semibold", "pad": 10}

    axes[0].set_title(r"$\lambda = \lambda_0,\; z_R = z_{R,m}$", **title_kwargs)
    axes[0].set_xlabel("Temperature [°C]")
    axes[0].set_ylabel(r"BK-$h$ factor")

    axes[1].set_title(r"$z_R = z_{R,m},\; T = T_{\mathrm{opt}}$", **title_kwargs)
    axes[1].set_xlabel(r"Wavelength: $\lambda_0$ [nm]")

    axes[2].set_title(
        rf"$\lambda = \lambda_0,\; L_c = {crystal_length_reference_mm:.0f}\,\mathrm{{mm}}$",
        **title_kwargs,
    )
    axes[2].set_xlabel("Temperature [°C]")
    axes[2].set_ylabel(r"BK-$h$ factor")

    axes[3].set_title(r"$\lambda = \lambda_0 \pm \delta\lambda,\; z_R = z_{R,m},\; T = T_{\mathrm{opt}}$", **title_kwargs)
    axes[3].set_xlabel(r"Wavelength: $\delta\lambda$ [nm]")

    for ax in axes:
        ax.set_facecolor("white")
        ax.grid(True, color="#b0b0b0", alpha=0.5, linewidth=0.8)
        ax.tick_params(labelsize=10)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.legend(frameon=True, facecolor="white", edgecolor="#cccccc", fontsize=9, loc="best")

    y_max = float(
        np.nanmax(
            [
                np.nanmax(bk_temp_lengths),
                np.nanmax(bk_wave_lengths),
                np.nanmax(bk_temp_rayleigh),
                np.nanmax(bk_detuning_lengths),
            ]
        )
    )
    for ax in axes:
        ax.set_ylim(0.0, 1.05 * y_max if y_max > 0.0 else 1.0)

    if figure_title:
        fig.suptitle(figure_title, fontsize=14, fontweight="semibold", y=0.98)
        fig.subplots_adjust(left=0.08, right=0.98, bottom=0.09, top=0.89, wspace=0.16, hspace=0.35)
    else:
        fig.subplots_adjust(left=0.08, right=0.98, bottom=0.09, top=0.92, wspace=0.16, hspace=0.35)
    return fig


def plot_double_resonance_scan(scan_data: dict[str, Any]):
    """Plot the crystal-side double-resonance mismatch scan.

    This is a diagnostic heatmap of the wrapped signal-idler cavity phase
    mismatch, not a full cavity transfer calculation.
    """
    temperature_grid_K = np.asarray(scan_data["temperature_grid_K"], dtype=float)
    crystal_length_grid_m = np.asarray(scan_data["crystal_length_grid_m"], dtype=float)
    abs_delta_phi_wrapped_rad = np.asarray(scan_data["abs_delta_phi_wrapped_rad"], dtype=float)

    fig, ax = plt.subplots(figsize=(9.0, 6.3))
    image = ax.imshow(
        abs_delta_phi_wrapped_rad,
        origin="lower",
        aspect="auto",
        extent=[
            float(temperature_grid_K[0]),
            float(temperature_grid_K[-1]),
            float(crystal_length_grid_m[0] * 1e3),
            float(crystal_length_grid_m[-1] * 1e3),
        ],
        cmap="viridis",
    )

    best_temperature_K = float(scan_data["best_temperature_K"])
    best_crystal_length_mm = float(scan_data["best_crystal_length_m"]) * 1e3
    best_abs_delta_phi_wrapped_rad = float(scan_data["best_abs_delta_phi_wrapped_rad"])

    ax.scatter(
        best_temperature_K,
        best_crystal_length_mm,
        s=80,
        marker="o",
        facecolor="#d1495b",
        edgecolor="white",
        linewidth=1.0,
        zorder=4,
        label="Best scan point",
    )
    ax.annotate(
        (
            f"Best point\nT = {best_temperature_K:.2f} K\n"
            f"L = {best_crystal_length_mm:.3f} mm\n"
            f"|Δφ| = {best_abs_delta_phi_wrapped_rad:.3e} rad"
        ),
        xy=(best_temperature_K, best_crystal_length_mm),
        xytext=(10, 10),
        textcoords="offset points",
        ha="left",
        va="bottom",
        fontsize=9,
        bbox={"facecolor": "white", "edgecolor": "#cccccc", "alpha": 0.92, "boxstyle": "round,pad=0.25"},
    )

    ax.set_xlabel("Temperature [K]")
    ax.set_ylabel("Crystal length [mm]")
    ax.set_title(r"Double-resonance diagnostic: $|\Delta\phi_{\mathrm{wrapped}}|$", pad=10)
    ax.set_facecolor("white")
    ax.grid(True, color="#b0b0b0", alpha=0.35, linewidth=0.7)
    ax.tick_params(labelsize=10)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=True, facecolor="white", edgecolor="#cccccc", fontsize=9, loc="upper right")

    cbar = fig.colorbar(image, ax=ax, pad=0.02)
    cbar.set_label(r"$|\Delta\phi_{\mathrm{wrapped}}|$ [rad]")
    cbar.ax.tick_params(labelsize=10)
    fig.tight_layout()
    return fig


__all__ = [
    "plot_phase_matching_temperature_scan",
    "plot_mode_matching_summary",
    "plot_boyd_kleinman_vs_focusing_parameter",
    "plot_boyd_kleinman_vs_delta_k",
    "plot_bk_master_map_sigma_xi",
    "plot_qpm_length_poling_map",
    "plot_boyd_kleinman_analysis",
    "plot_double_resonance_scan",
]
