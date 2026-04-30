# Cavity Module

The cavity module defines the resonator geometry, Gaussian mode, and resonant-field loss model. Its output is consumed by both the crystal and OPO stages.

Entry point:

```text
src/cavity/cavity_main.py
```

Core implementation:

```text
src/cavity/cavity_abcd.py
src/cavity/cavity_analysis.py
src/cavity/cavity_workflow.py
src/cavity/cavity_plotter.py
```

## What It Computes

The cavity stage computes:

- geometry-dependent ABCD/q-parameter quantities
- stability and waist maps
- selected single-point cavity mode
- beam waist in the crystal
- geometric and optical round-trip lengths
- FSR
- resonant-field loss terms
- `kappa_ext`, `kappa_loss`, `kappa_total`
- escape efficiency
- Gouy phases

## Supported Geometries

Current geometry names:

- `bowtie`
- `linear`
- `triangle`
- `hemilithic`
- `monolithic`

The selected geometry is set with `GEOMETRY` in `cavity_main.py`.

## Geometry and Round-Trip Lengths

The code distinguishes geometric length from optical length.

- geometric lengths describe physical propagation distance
- optical lengths include refractive-index weighting inside the crystal
- `optical_roundtrip_length_m` sets the FSR

The exported `roundtrip_propagation_length_m` is used for distributed loss calculations.

## Loss and Decay Rates

The resonant-field loss model uses mirror/facet reflectivities, distributed loss, and parasitic round-trip loss. It exports:

- `kappa_ext`: useful output-coupling decay rate
- `kappa_loss`: internal loss decay rate
- `kappa_total`: total resonant-field decay rate
- `escape_efficiency`: useful output-coupling fraction

Values are exported in both rad/s and Hz where applicable.

## Beam Waist and Astigmatism

The cavity output includes `beam_waist_crystal_um`, used by the crystal layer for mode matching. Sagittal and tangential q-parameters are stored in `debug_data`. Gouy phases are exported in `results`.

## Downstream Use

The crystal layer uses the cavity JSON to obtain:

- crystal length
- signal/idler wavelength
- crystal refractive index
- beam waist in the crystal
- cavity round-trip data for resonance diagnostics

The OPO layer uses the cavity JSON for:

- decay rates
- escape efficiency
- detuning
- consistency checks against the selected crystal length

## Outputs

The cavity stage writes:

```text
results/<geometry>/cavity/cavity_simulation_output.json
results/<geometry>/cavity/stability_map.png
results/<geometry>/cavity/waist_map.png
```

See [../04_outputs.md](../04_outputs.md) for the JSON field list.
