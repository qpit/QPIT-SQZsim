# OPO Module

The OPO module consumes cavity and crystal outputs and computes the below-threshold squeezing spectra.

Entry point:

```text
src/opo/opo_main.py
```

Core implementation:

```text
src/opo/opo_model.py
src/opo/opo_langevin.py
src/opo/opo_squeezing.py
src/opo/opo_workflow.py
src/opo/opo_plotter.py
```

## What It Computes

The OPO stage computes:

- cavity/crystal consistency checks
- first-principles external threshold power
- resolved pump parameter and pump power
- below-threshold quadrature Langevin model
- squeezing, anti-squeezing, measured quadrature, and optimal phase spectra
- resonance diagnostic plot

## Inputs

From cavity JSON, the OPO stage uses:

- `kappa_ext`
- `kappa_loss`
- `kappa_total`
- `escape_efficiency`
- detuning
- cavity crystal length for consistency checks

From crystal JSON, it uses `results.active_for_opo`, especially:

- `crystal_length_m`
- `refractive_indices`
- `mode_matching.waist_crystal_m`
- `mode_matching.effective_nonlinear_overlap`
- polarization-resonance data for diagnostics

It also uses `inputs.d_eff_pm_per_V` from the crystal JSON.

## Pump Configuration

`opo_main.py` supports two pump input modes.

Fraction mode:

```python
PUMP_MODE = "fraction"
PUMP_PARAMETER_SIGMA = 0.1
```

Absolute mode:

```python
PUMP_MODE = "absolute"
PUMP_POWER_W = 0.2e-3
```

In fraction mode, `PUMP_PARAMETER_SIGMA` is prioritized over `PUMP_PERCENT_THRESHOLD` if both are present. The model always resolves and stores both final external pump power and pump parameter.

## Threshold Model

The threshold is computed from the loaded cavity and crystal state. It uses the nonlinear coupling, signal/idler decay rates, refractive indices, `d_eff`, crystal length, effective mode area, and nonlinear overlap.

The threshold condition is:

```text
g * sqrt(n_pump) = sqrt(kappa_s * kappa_i) / 2
```

The current default pump resonance model is:

```python
PUMP_RESONANCE_MODEL = "single_pass"
```

In `single_pass` mode, the threshold pump energy is converted to external pump power using the pump transit time through the crystal. The code does not apply accidental pump-cavity buildup in this mode.

A `resonant` pump branch exists, but it requires explicit pump linewidth and pump input-coupling efficiency.

## Langevin and Squeezing

The Langevin model is a 2x2 quadrature-basis model with X and P quadratures. It is valid only below threshold. At each analysis frequency, the squeezing calculation diagonalizes the real symmetric quadrature spectral-density matrix. `squeezing_spectrum` is the minimum principal eigenvalue, `antisqueezing_spectrum` is the maximum principal eigenvalue, and `measured_quadrature_spectrum` is the fixed LO-phase projection. All spectra use the common shot-noise reference, so spectra approach 0 dB at high analysis frequency.

The calculation includes escape efficiency and detection efficiency as loss channels.

## Outputs

The OPO stage writes:

```text
results/<geometry>/opo/opo_simulation_output.json
results/<geometry>/opo/opo_squeezing_spectrum.png
results/<geometry>/opo/opo_resonance_diagnostic.png
```

The JSON contains:

- `inputs`: resolved OPO configuration and upstream file paths
- `results.model`: threshold, pump, cavity, and crystal operating-point quantities
- `results.spectrum`: spectra arrays
- `debug_data.langevin`: Langevin matrices

See [../04_outputs.md](../04_outputs.md) for the JSON field list.

## Limitations

The current OPO model is:

- below threshold
- degenerate
- single-mode
- linearized

It does not model pump depletion, above-threshold dynamics, multimode behavior, or full non-degenerate dynamics.
