# Crystal Module

The crystal module consumes the cavity output and selects the nonlinear operating point used by the OPO stage.

Entry point:

```text
src/crystal/crystal_main.py
```

Core implementation:

```text
src/crystal/crystal_materials.py
src/crystal/crystal_phase_matching.py
src/crystal/crystal_polarization_resonance.py
src/crystal/crystal_double_resonance_scan.py
src/crystal/crystal_mode_matching.py
src/crystal/crystal_boyd_kleinman.py
src/crystal/crystal_workflow.py
```

## What It Computes

The crystal stage computes:

- refractive-index functions from the selected material model
- phase-matching configuration for `type_0`, `type_I`, or `type_II`
- QPM poling period in design mode, or uses a fixed period in analysis mode
- phase-matching temperature scan
- phase-matching operating point
- double-resonance scan and operating point
- selected active operating point
- mode matching and Boyd-Kleinman nonlinear overlap
- compact `active_for_opo` handoff payload

## Configuration Modes

`PHASE_MATCHING_MODE` controls the QPM period:

- `design`: compute the QPM poling period at `DESIGN_TEMPERATURE_K`
- `analysis`: use `ANALYSIS_LAMBDA0_M` as the fixed poling period

`PHASE_MATCHING_TYPE` controls the interaction axes and effective nonlinearity. Current values are:

- `type_0`
- `type_I`
- `type_II`

`OPERATING_POINT_MODE` selects the active crystal state:

- `phase_matching`: use the maximum phase-matching power factor
- `double_resonance`: use the best signal/idler resonance match

## Phase Matching

The phase-matching scan evaluates refractive indices, effective QPM mismatch, and a sinc-squared power factor versus temperature. In design mode, the terminal summary prints the designed QPM poling period in micrometers and meters.

The phase-matching operating point and double-resonance operating point are separate candidates. They can have different temperatures and, for double resonance, a different crystal length.

## Double Resonance

The polarization-resonance diagnostic compares signal and idler optical round-trip phases. The double-resonance scan searches temperature and crystal length and reports the best wrapped phase mismatch.

The scan range for crystal length is centered around the selected cavity crystal length in the workflow. If the selected double-resonance length differs from the cavity length, rerun the cavity stage with the recommended length before running OPO.

## Boyd-Kleinman and Mode Matching

The crystal layer computes:

- beam waist in the crystal
- Rayleigh range
- focusing parameter `xi`
- normalized mismatch `sigma`
- Boyd-Kleinman factor
- effective nonlinear overlap

The OPO stage uses the compact overlap and waist from `active_for_opo.mode_matching`.

## `active_for_opo`

The standard crystal JSON is intentionally compact. The main downstream payload is:

```text
results.active_for_opo
```

It contains:

- selected operating-point mode
- active temperature
- active crystal length
- pump/signal/idler refractive indices
- mode waist and nonlinear overlap
- polarization-resonance diagnostic
- recommended cavity crystal length

This block is the OPO-facing crystal state. The OPO layer does not select a new crystal operating point.

## Outputs

The crystal stage writes:

```text
results/<geometry>/crystal/crystal_simulation_output.json
results/<geometry>/crystal/boyd_kleinman_master_map.png
results/<geometry>/crystal/boyd_kleinman_analysis.png
results/<geometry>/crystal/double_resonance_scan.png
```

The standard JSON contains only compact fields needed by OPO. Debug data can be added by calling the workflow output builder with debug options.

See [../04_outputs.md](../04_outputs.md) for the JSON field list.
