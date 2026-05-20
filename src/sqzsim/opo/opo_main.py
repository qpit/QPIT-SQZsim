# %%
"""
Main entry point for OPO simulation.

This script orchestrates the initial OPO workflow:
cavity/crystal outputs -> OPO model -> Langevin scaffold -> Langevin-based squeezing spectra -> export
"""

from __future__ import annotations

from pathlib import Path

# -------------------------------------------------------------------
# Main simulation script
# -------------------------------------------------------------------
# This file acts as the entry point for the OPO simulation.
# The detailed physics is implemented in the corresponding workflow
# and module files. This script is intentionally readable top-to-bottom
# and organized in notebook-style cells for interactive development.
# -------------------------------------------------------------------

# Support both package execution and direct interactive execution.
try:
    from .opo_plotter import plot_opo_resonance_diagnostic, plot_opo_spectrum_summary
    from .opo_workflow import (
        build_opo_simulation_output,
        build_opo_simulation_result,
        compute_opo_langevin,
        compute_opo_model,
        compute_opo_squeezing,
        load_opo_context,
        print_opo_summary,
        save_opo_outputs,
    )
except ImportError:
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from opo.opo_plotter import plot_opo_resonance_diagnostic, plot_opo_spectrum_summary
    from opo.opo_workflow import (
        build_opo_simulation_output,
        build_opo_simulation_result,
        compute_opo_langevin,
        compute_opo_model,
        compute_opo_squeezing,
        load_opo_context,
        print_opo_summary,
        save_opo_outputs,
    )


# %%
# Simulation parameters

GEOMETRY = "bowtie"  # Cavity geometry used to load upstream cavity/crystal results.

# Below-threshold degenerate OPO operating point.
PUMP_MODE = "fraction"        # "fraction" uses sigma or percent threshold; "absolute" uses pump power [W].
PUMP_PARAMETER_SIGMA = 0.6     # Pump parameter sigma = sqrt(P / P_threshold) [-].
PUMP_PERCENT_THRESHOLD = 60.0  # Optional pump amplitude percentage of threshold [%]; sigma takes priority.
PUMP_POWER_W = 0.2e-3          # Pump power coupled into the OPO [W], used only when PUMP_MODE = "absolute".
PUMP_RESONANCE_MODEL = "single_pass"  # "single_pass" for non-resonant pump, "resonant" for cavity-enhanced pump.
WAVELENGTH_P_M = 775e-9    # Pump wavelength [m].
WAVELENGTH_S_M = 1550e-9   # Signal wavelength [m].
WAVELENGTH_I_M = 1550e-9   # Idler wavelength [m].

# Homodyne / squeezing-spectrum analysis settings.
ANALYSIS_SIDEBAND_HZ = 5e6        # Representative analysis sideband frequency [Hz].
ANALYSIS_SPAN_HZ = (1e5, 20e6)    # Frequency span for the squeezing spectrum [Hz].
N_ANALYSIS_POINTS = 400           # Number of frequency samples in the spectrum.
DETECTION_EFFICIENCY = 0.86       # Total detection efficiency, including propagation and detector losses.
LO_PHASE_RAD = 1.0                # Homodyne local-oscillator phase [rad].


# %%
# Load upstream simulation outputs

context = load_opo_context(GEOMETRY)


# %%
# Build OPO model

OPO_CONFIG = {
    "pump_mode": PUMP_MODE,
    "pump_resonance_model": PUMP_RESONANCE_MODEL,
    "wavelength_p_m": WAVELENGTH_P_M,
    "wavelength_s_m": WAVELENGTH_S_M,
    "wavelength_i_m": WAVELENGTH_I_M,
    "analysis_sideband_Hz": ANALYSIS_SIDEBAND_HZ,
    "analysis_span_Hz": ANALYSIS_SPAN_HZ,
    "n_analysis_points": N_ANALYSIS_POINTS,
    "detection_efficiency": DETECTION_EFFICIENCY,
    "lo_phase_rad": LO_PHASE_RAD,
}
if PUMP_MODE == "fraction":
    OPO_CONFIG["pump_parameter_sigma"] = PUMP_PARAMETER_SIGMA
    OPO_CONFIG["pump_percent_threshold"] = PUMP_PERCENT_THRESHOLD
elif PUMP_MODE == "absolute":
    OPO_CONFIG["pump_power_W"] = PUMP_POWER_W
else:
    raise ValueError(f"Unsupported PUMP_MODE: {PUMP_MODE!r}")
parameters, model = compute_opo_model(context, OPO_CONFIG)


# %%
# Build Langevin scaffold

langevin = compute_opo_langevin(model)


# %%
# Compute Langevin-based squeezing spectrum

spectrum = compute_opo_squeezing(parameters, model, langevin)


# %%
# Build structured result

result = build_opo_simulation_result(
    context=context,
    parameters=parameters,
    model=model,
    langevin=langevin,
    spectrum=spectrum,
)


# %%
# Print summary

print_opo_summary(result)


# %%
# Build export payload

output = build_opo_simulation_output(result)


# %%
# Generate plots

fig_spectrum = plot_opo_spectrum_summary(output["results"]["spectrum"])
fig_resonance_diagnostic = plot_opo_resonance_diagnostic(
    output["results"]["spectrum"],
    model=output["results"]["model"],
    crystal_results=context.crystal_data.get("results", {}),
)


# %%
# Save outputs

outputs_info = save_opo_outputs(
    GEOMETRY,
    output,
    fig_spectrum,
    fig_resonance_diagnostic=fig_resonance_diagnostic,
)
print(f"Saved OPO output to: {outputs_info['opo_output_json']}")
print(f"Saved OPO spectrum plot to: {outputs_info['opo_squeezing_spectrum_png']}")
if "opo_resonance_diagnostic_png" in outputs_info:
    print(f"Saved OPO resonance diagnostic plot to: {outputs_info['opo_resonance_diagnostic_png']}")

# %%
