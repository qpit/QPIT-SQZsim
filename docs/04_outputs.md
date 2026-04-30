# 04 - Outputs

Each stage writes JSON under `results/<geometry>/<stage>/`. These files define the data interface between stages.

General structure:

```json
{
  "inputs": {},
  "results": {},
  "debug_data": {}
}
```

`debug_data` is present only when that stage writes diagnostics.

## Cavity JSON

Path:

```text
results/<geometry>/cavity/cavity_simulation_output.json
```

Top-level keys:

- `inputs`
- `results`
- `debug_data`

Important `inputs` fields include:

- `geometry`
- `crystal_length_m`
- `n_crystal`
- `RoC_1_m`, `RoC_2_m`
- `wavelength_m`
- resonant reflectivities and loss inputs
- `geometry_specific`

Important `results` fields include:

- `beam_waist_crystal_um`
- `cavity_length_m`
- `optical_crystal_length_m`
- `optical_roundtrip_length_m`
- `fsr_Hz`
- `roundtrip_propagation_length_m`
- `input_coupler_transmission`
- `output_coupling_transmission`
- `bulk_roundtrip_loss`
- `internal_roundtrip_loss`
- `kappa_ext_rad_s`, `kappa_ext_Hz`
- `kappa_loss_rad_s`, `kappa_loss_Hz`
- `kappa_total_rad_s`, `kappa_total_Hz`
- `escape_efficiency`
- `detuning_rad_s`
- `gouy_phase_sagittal_rad`, `gouy_phase_tangential_rad`

`debug_data` contains q-parameters, m-factor data, and resolved radii.

Downstream use:

- crystal uses beam waist, crystal length, wavelength, refractive index, and cavity data
- OPO uses kappa values, escape efficiency, detuning, and consistency checks

## Crystal JSON

Path:

```text
results/<geometry>/crystal/crystal_simulation_output.json
```

Standard top-level keys:

- `inputs`
- `results`

Current standard `inputs` contains:

- `d_eff_pm_per_V`

Current standard `results` contains:

- `active_for_opo`

`active_for_opo` contains the compact OPO handoff:

- `operating_point_mode`
- `temperature_K`
- `crystal_length_m`
- `refractive_indices`
  - `n_p`
  - `n_s`
  - `n_i`
- `mode_matching`
  - `waist_crystal_m`
  - `effective_nonlinear_overlap`
- `polarization_resonance`
  - `fsr_signal_Hz`
  - `fsr_idler_Hz`
  - `delta_fsr_Hz`
  - `delta_phi_wrapped_rad`
  - `is_double_resonant`
  - `signal_optical_roundtrip_length_m`
  - `idler_optical_roundtrip_length_m`
- `recommended_cavity_crystal_length_m`

Debug output can be enabled through the workflow builder. When enabled, it may include active phase matching, operating points, scans, double-resonance summaries or matrices, and Boyd-Kleinman diagnostics.

Downstream use:

- OPO reads `active_for_opo` as the selected crystal state
- OPO requires `refractive_indices`, `mode_matching`, `crystal_length_m`, and `d_eff_pm_per_V` for the physical threshold

## OPO JSON

Path:

```text
results/<geometry>/opo/opo_simulation_output.json
```

Top-level keys:

- `inputs`
- `results`
- `debug_data`

Important `inputs` fields include:

- `geometry`
- `cavity_output_path`
- `crystal_output_path`
- `pump_mode`
- `pump_power_W` or `pump_parameter_sigma` / `pump_percent_threshold`
- `pump_resonance_model`
- `signal_wavelength_m`, `idler_wavelength_m`, `pump_wavelength_m`
- analysis frequency and detection settings

`results.model` contains the resolved OPO operating point, including:

- `pump_mode`
- `pump_parameter`
- `pump_power_W`
- `effective_threshold_power_W`
- `threshold_external_pump_power_W`
- `threshold_intracavity_pump_photon_number`
- `threshold_nonlinear_coupling`
- `threshold_mode_area_m2`
- `threshold_overlap`
- `threshold_refractive_indices`
- `pump_resonance_model`
- `pump_buildup_factor`
- `pump_conversion_assumption`
- `d_eff_pm_per_V`
- `crystal_length_m`
- `cavity_kappa_ext_Hz`, `cavity_kappa_loss_Hz`, `cavity_kappa_total_Hz`
- `escape_efficiency`
- `below_threshold`

`results.spectrum` contains:

- `frequency_Hz`
- `squeezing_spectrum`: minimum-noise principal quadrature, linear units normalized to shot noise
- `antisqueezing_spectrum`: maximum-noise principal quadrature, linear units normalized to shot noise
- `measured_quadrature_spectrum`: fixed LO-phase quadrature, linear units normalized to shot noise
- `shot_noise_reference`: common shot-noise reference
- `optimal_phase_rad`
- `lo_phase_rad`
- `notes`

`debug_data.langevin` contains quadrature labels and the drift/input/noise matrices.
