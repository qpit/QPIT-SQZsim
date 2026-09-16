# %%
"""Main entry point for crystal simulation."""

from __future__ import annotations

import inspect
from pathlib import Path

# Support both package execution and direct interactive execution.
try:
    from .crystal_plotter import (
        plot_bk_master_map_sigma_xi,
        plot_boyd_kleinman_analysis,
        plot_double_resonance_scan,
    )
    from .crystal_workflow import (
        build_crystal_simulation_result_from_blocks,
        build_crystal_simulation_output,
        compute_crystal_operating_point_block,
        compute_crystal_phase_matching_block,
        compute_crystal_physics_block,
        print_crystal_summary,
        save_crystal_outputs,
        setup_crystal_simulation,
    )
except ImportError:
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from crystal.crystal_plotter import (
        plot_bk_master_map_sigma_xi,
        plot_boyd_kleinman_analysis,
        plot_double_resonance_scan,
    )
    from crystal.crystal_workflow import (
        build_crystal_simulation_result_from_blocks,
        build_crystal_simulation_output,
        compute_crystal_operating_point_block,
        compute_crystal_phase_matching_block,
        compute_crystal_physics_block,
        print_crystal_summary,
        save_crystal_outputs,
        setup_crystal_simulation,
    )


# %%
# Simulation configuration

GEOMETRY = "bowtie"  # Cavity geometry used to load upstream cavity results.
CRYSTAL_MODEL = "Kato2002"  # Choices: "Kato2002", "Fan1987", "Konig2004"

WAVELENGTH_P_M = 775e-9   # Pump wavelength [m]
WAVELENGTH_S_M = 1550e-9  # Signal wavelength [m]
WAVELENGTH_I_M = 1550e-9  # Idler wavelength [m]

PHASE_MATCHING_MODE = "design"  # "design" or "analysis"
PHASE_MATCHING_TYPE = "type_II"  # allowed: "type_0", "type_I", "type_II"
DESIGN_TEMPERATURE_K = 317
ANALYSIS_LAMBDA0_M = 27.7e-6  # Poling period Lambda [m]
# 10.818391e-6 for analysis test

T_MIN_K = 300.0  # Minimum temperature of the phase-matching scan [K]
T_MAX_K = 340.0  # Maximum temperature of the phase-matching scan [K]
N_T = 401        # Number of temperature points in the scan

T0_K = 293.15         # Reference temperature [K] for thermo-optic / thermal-expansion models
ALPHA_PER_K = 6.7e-6  # Approximate crystal linear thermal expansion coefficient [1/K]
QPM_ORDER_M = 1       # Quasi-phase-matching order m (m = 1 is first-order QPM)

# Select crystal operating point used for downstream calculations.
# "phase_matching" maximizes nonlinear conversion; "double_resonance" enforces signal/idler resonance.
OPERATING_POINT_MODE = "phase_matching"


CONFIG = {
    "GEOMETRY": GEOMETRY,
    "CRYSTAL_MODEL": CRYSTAL_MODEL,
    "WAVELENGTH_P_M": WAVELENGTH_P_M,
    "WAVELENGTH_S_M": WAVELENGTH_S_M,
    "WAVELENGTH_I_M": WAVELENGTH_I_M,
    "PHASE_MATCHING_MODE": PHASE_MATCHING_MODE,
    "PHASE_MATCHING_TYPE": PHASE_MATCHING_TYPE,
    "DESIGN_TEMPERATURE_K": DESIGN_TEMPERATURE_K,
    "ANALYSIS_LAMBDA0_M": ANALYSIS_LAMBDA0_M,
    "T_MIN_K": T_MIN_K,
    "T_MAX_K": T_MAX_K,
    "N_T": N_T,
    "T0_K": T0_K,
    "ALPHA_PER_K": ALPHA_PER_K,
    "QPM_ORDER_M": QPM_ORDER_M,
    "OPERATING_POINT_MODE": OPERATING_POINT_MODE,
}


# %%
# Context setup
# Resolve configuration, material models, and the upstream cavity context.

setup = setup_crystal_simulation(CONFIG)
cfg = setup["cfg"]
context = setup["context"]
phase_config = setup["phase_config"]
d_eff_config = setup["d_eff_config"]
index_functions = setup["index_functions"]

# %%
# Phase matching
# Compute QPM poling period and the phase-matching temperature scan.

phase_matching = compute_crystal_phase_matching_block(
    cfg=cfg,
    context=context,
    phase_config=phase_config,
    d_eff_config=d_eff_config,
    index_functions=index_functions,
)

# %%
# Operating point selection
# Compare phase-matching and double-resonance candidates, then select the active operating point.

operating_point = compute_crystal_operating_point_block(
    cfg=cfg,
    context=context,
    phase_config=phase_config,
    d_eff_config=d_eff_config,
    index_functions=index_functions,
    Lambda0_m=phase_matching["Lambda0_m"],
    phase=phase_matching["phase"],
    phase_matching_resonance_diagnostic=phase_matching["phase_matching_resonance_diagnostic"],
)

# %%
# Nonlinear analysis
# Evaluate mode matching and Boyd-Kleinman nonlinear overlap at the active operating point.

nonlinear_analysis = compute_crystal_physics_block(
    cfg=cfg,
    context=context,
    index_functions=index_functions,
    phase_active=operating_point["phase_active"],
    active_operating_temperature_K=operating_point["active_operating_temperature_K"],
    Lambda0_m=phase_matching["Lambda0_m"],
)

# %%
# Build simulation result

result = build_crystal_simulation_result_from_blocks(
    setup=setup,
    phase_block=phase_matching,
    operating_block=operating_point,
    physics_block=nonlinear_analysis,
)
print_crystal_summary(result)
output = build_crystal_simulation_output(result)
outputs_info = save_crystal_outputs(GEOMETRY, output)
print(f"Saved crystal output JSON to: {outputs_info['crystal_output_json']}")


# %%
# Generate default plots

bk_analysis = result.bk_analysis or {}
bk_analysis_operating = bk_analysis.get("bk_analysis_operating", bk_analysis)

selected_operating_point_label = {
    "phase_matching": "Phase-matching operating point",
    "double_resonance": "Double-resonance operating point",
}.get(result.selected_operating_point_mode, "Operating point")

bk_master_operating_point = {
    "xi_reference": bk_analysis_operating["reference"]["xi_reference"],
    "sigma_reference": bk_analysis_operating["reference"]["sigma_reference"],
}
plot_signature = inspect.signature(plot_bk_master_map_sigma_xi)
if "operating_point_label" in plot_signature.parameters:
    fig_bk_master = plot_bk_master_map_sigma_xi(
        bk_analysis_operating,
        operating_point=bk_master_operating_point,
        operating_point_label=selected_operating_point_label,
    )
else:
    fig_bk_master = plot_bk_master_map_sigma_xi(
        bk_analysis_operating,
        operating_point=bk_master_operating_point,
    )
fig_bk = plot_boyd_kleinman_analysis(
    bk_analysis_operating,
    figure_title="BK Analysis Around Operating Point",
)
fig_double_resonance = (
    plot_double_resonance_scan(result.double_resonance_scan)
    if result.double_resonance_scan is not None
    else None
)


# %%
# Save outputs

outputs_info = save_crystal_outputs(
    GEOMETRY,
    output,
    fig_bk_master=fig_bk_master,
    fig_bk=fig_bk,
    fig_double_resonance_scan=fig_double_resonance,
)
print(f"Updated crystal output with plot metadata: {outputs_info['crystal_output_json']}")
print(f"Saved BK master map to: {outputs_info['boyd_kleinman_master_map_png']}")
print(f"Saved BK analysis plot to: {outputs_info['boyd_kleinman_analysis_png']}")
if "double_resonance_scan_png" in outputs_info:
    print(f"Saved double-resonance scan plot to: {outputs_info['double_resonance_scan_png']}")

# %%
