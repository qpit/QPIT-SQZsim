# %%
"""
Main entry point for cavity simulation.

This script orchestrates the cavity simulation workflow:
geometry -> stability -> mode -> derived quantities -> export
"""

# -------------------------------------------------------------------
# Main simulation script
# -------------------------------------------------------------------
# This file acts as the entry point for the cavity simulation.
# The detailed physics is implemented in the corresponding workflow
# and module files. This script orchestrates the workflow and exports
# results for the selected geometry.
# -------------------------------------------------------------------

import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
_SRC_ROOT = _HERE.parents[0]
for _path in (str(_HERE), str(_SRC_ROOT)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from common.constants import C_M_PER_S, DEG_TO_RAD
from cavity_plotter import CavityPlotter
from cavity_workflow import (
    build_cavity_context,
    build_cavity_simulation_output,
    build_cavity_simulation_result,
    build_geometry_estimators,
    compute_cavity_derived_quantities,
    compute_cavity_operating_point,
    print_derived_cavity_quantities,
    print_geometry_info,
    print_single_point_summary,
    save_cavity_outputs,
)

# %%
# Geometry selection

# Choose: "bowtie", "linear", "triangle", "hemilithic", or "monolithic"
GEOMETRY = "bowtie"

# %% Select parameters

# Crystal parameters
CRYSTAL_LENGTH_M = 16e-3     # Crystal length [m]
N_CRYSTAL = 1.78             # Crystal refractive index

ROC_1_M = 50e-3              # Mirror/surface 1 RoC [m]
ROC_2_M = 50e-3              # Mirror/surface 2 RoC [m]
WAVELENGTH_M = 1550e-9       # Signal/Idler wavelength [m]

# Resonant-field cavity losses parameters
R1_RESONANT = 0.999      # It is the input mirror/facet reflectivity and any transmission through it is treated as internal cavity loss
R2_RESONANT = 0.954      # It defines the output mirror

ALPHA_RESONANT_PER_M = 0.0     # Distributed losses along the cavity, such as absorption in the crystal or air. Here it is set to zero to isolate the effect of mirror coupling
L_PARASITIC_RT = 0.0           # Parasitic round-trip loss accounts for additional losses such as scattering or imperfect coatings. It is set to zero in this configuration to model an ideal cavity
DETUNING_HZ = 0.0              # The detuning is set to zero, meaning that the cavity is evaluated exactly at resonance, where the intracavity field is maximized


# Bowtie parameters
THETA_AOI_RAD = 6 * DEG_TO_RAD    # Aoi [deg]
SHORT_AXIS_SCAN_M = np.arange(56e-3, 71e-3, 0.01e-3)
LONG_AXIS_SCAN_M = np.arange(70e-3, 120e-3, 0.5e-3)
MESH_SHORT_AXIS_M, MESH_LONG_AXIS_M = np.meshgrid(SHORT_AXIS_SCAN_M, LONG_AXIS_SCAN_M)

# Linear parameters
L_CAV_M = 50e-3   # Cavity length (crystal included)

# Hemilithic parameters
L_AIR_M = 20e-3   # Air section length (Cavity length = CRYSTAL_LENGTH_M + L_AIR_M)

# Triangle parameters
TRIANGLE_WIDTH_SCAN_M = np.arange(max(CRYSTAL_LENGTH_M + 1e-3, 40e-3), 140e-3, 0.5e-3)
TRIANGLE_HEIGHT_SCAN_M = np.arange(10e-3, 80e-3, 0.5e-3)
MESH_TRIANGLE_WIDTH_M, MESH_TRIANGLE_HEIGHT_M = np.meshgrid(TRIANGLE_WIDTH_SCAN_M, TRIANGLE_HEIGHT_SCAN_M)

# %%
# Geometry info

print_geometry_info(GEOMETRY)

# %%
# Geometry-dependent estimators

PARAMETERS = {
    "crystal_length_m": CRYSTAL_LENGTH_M,
    "n_crystal": N_CRYSTAL,
    "roc_1_m": ROC_1_M,
    "roc_2_m": ROC_2_M,
    "wavelength_m": WAVELENGTH_M,
    "r1_resonant": R1_RESONANT,
    "r2_resonant": R2_RESONANT,
    "alpha_resonant_per_m": ALPHA_RESONANT_PER_M,
    "l_parasitic_rt": L_PARASITIC_RT,
    "detuning_hz": DETUNING_HZ,
    "theta_aoi_rad": THETA_AOI_RAD,
    "l_cav_m": L_CAV_M,
    "l_air_m": L_AIR_M,
    "mesh_short_axis_m": MESH_SHORT_AXIS_M,
    "mesh_long_axis_m": MESH_LONG_AXIS_M,
    "mesh_triangle_width_m": MESH_TRIANGLE_WIDTH_M,
    "mesh_triangle_height_m": MESH_TRIANGLE_HEIGHT_M,
}

estimators = build_geometry_estimators(GEOMETRY, PARAMETERS)
context = build_cavity_context(GEOMETRY, PARAMETERS, estimators=estimators)

# %%
# Stability and waist plots

plotter = CavityPlotter(GEOMETRY)
fig_stability = plotter.make_stability_plot(
    estimate_m_factor_s=estimators.estimate_m_factor_s,
    crystal_length=CRYSTAL_LENGTH_M,
    n_crystal=N_CRYSTAL,
    radius_of_curvature_1=ROC_1_M,
    radius_of_curvature_2=ROC_2_M,
    incidence_angle=THETA_AOI_RAD,
    mesh_x=estimators.mesh_x,
    mesh_y=estimators.mesh_y,
)

fig_waist = plotter.make_waist_plot(
    estimate_q_sagittal=estimators.estimate_q_sagittal,
    crystal_length=CRYSTAL_LENGTH_M,
    n_crystal=N_CRYSTAL,
    wavelength=WAVELENGTH_M,
    radius_of_curvature_1=ROC_1_M,
    radius_of_curvature_2=ROC_2_M,
    incidence_angle=THETA_AOI_RAD,
    mesh_x=estimators.mesh_x,
    mesh_y=estimators.mesh_y,
)

# %%
# Single-point evaluation

# Explicit single-point selections used for the detailed evaluation step.
SINGLE_POINT_PARAMETERS = {
    "ROC_1_M": 50e-3,
    "ROC_2_M": 50e-3,
    "BOWTIE_SHORT_AXIS_M": 68e-3,
    "BOWTIE_LONG_AXIS_M": 90e-3,
    "BOWTIE_THETA_AOI_RAD": 6 * DEG_TO_RAD,
    "LINEAR_CAVITY_LENGTH_M": 50e-3,
    "TRIANGLE_WIDTH_M": 80e-3,
    "TRIANGLE_HEIGHT_M": 30e-3,
    "HEMILITHIC_AIR_GAP_M": 20e-3,
    "MONOLITHIC_CRYSTAL_LENGTH_M": CRYSTAL_LENGTH_M,
}

single_point = compute_cavity_operating_point(context, SINGLE_POINT_PARAMETERS)
print_single_point_summary(GEOMETRY, single_point)

# %%
# Derived cavity figures

derived = compute_cavity_derived_quantities(
    context,
    single_point,
    c_m_per_s=C_M_PER_S,
    detuning_Hz=DETUNING_HZ,
    loss_model_parameters=PARAMETERS,
)
print_derived_cavity_quantities(derived)

# %%
# Export simulation output

cavity_result = build_cavity_simulation_result(context, single_point, derived)
simulation_output = build_cavity_simulation_output(cavity_result, c_m_per_s=C_M_PER_S)
saved_outputs = save_cavity_outputs(GEOMETRY, simulation_output, fig_stability, fig_waist)

print(f"Saved simulation output to: {saved_outputs['cavity_output_json']}")
print(f"Saved stability map to: {saved_outputs['stability_map_png']}")
print(f"Saved waist map to: {saved_outputs['waist_map_png']}")

# %%
