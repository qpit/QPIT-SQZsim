# 03 - Physics

This page summarizes the physical models used by the current implementation. It is intentionally concise; implementation details are in the module docs and source code.

## Cavity Model

The cavity layer uses ABCD matrices to evaluate the resonator mode for the selected geometry. It exports the beam waist in the crystal and round-trip quantities used downstream.

The optical round-trip length determines the free spectral range:

```text
FSR = c / L_optical_roundtrip
```

The resonant-field loss model separates useful output coupling from internal loss:

- `kappa_ext`: decay rate associated with output coupling
- `kappa_loss`: decay rate associated with internal/parasitic loss
- `kappa_total = kappa_ext + kappa_loss`

The escape efficiency is the useful output-coupling fraction of the total loss.

## Crystal Model

The crystal layer computes refractive indices using the selected material model and phase-matching type. Supported material model names are currently `Kato2002`, `Fan1987`, and `Konig2004`.

For three-wave mixing, the bulk mismatch is:

```text
Delta k = k_p - k_s - k_i
```

With quasi-phase matching:

```text
Delta k_eff = Delta k - m * 2*pi/Lambda
```

The phase-matching scan evaluates a sinc-squared conversion factor versus temperature.

`PHASE_MATCHING_MODE = "design"` computes the QPM poling period at `DESIGN_TEMPERATURE_K`. `PHASE_MATCHING_MODE = "analysis"` uses the supplied `ANALYSIS_LAMBDA0_M`.

The crystal layer also computes a polarization-resonance diagnostic for signal and idler. The double-resonance scan searches temperature and crystal length for small wrapped signal/idler phase mismatch.

## Boyd-Kleinman Quantities

The spatial nonlinear overlap is described by Boyd-Kleinman-style quantities:

- `xi = L / (2 z_R)`: focusing parameter
- `sigma = z_R * Delta k_eff`: normalized mismatch parameter
- Boyd-Kleinman factor: focusing/mismatch-dependent nonlinear overlap factor
- effective nonlinear overlap: compact overlap value exported to OPO

## OPO Threshold and Pump Parameter

The OPO model is below threshold, degenerate, and single-mode. It computes a physical threshold from the loaded cavity and crystal state.

The threshold condition is:

```text
g * sqrt(n_pump) = sqrt(kappa_s * kappa_i) / 2
```

The coupling `g` depends on `d_eff`, refractive indices, crystal length, mode area, pump/signal/idler frequencies, and nonlinear overlap.

The pump is currently configured as `PUMP_RESONANCE_MODEL = "single_pass"` in `opo_main.py`. In this mode, the required pump field is converted to external pump power using the pump transit time through the crystal, not a pump-cavity lifetime. A resonant pump branch exists in the model but requires explicit pump linewidth and input-coupling efficiency.

The pump operating point can be specified by sigma/fraction or by absolute external pump power:

```text
sigma = sqrt(P_pump / P_threshold)
```

## Squeezing Model

The OPO layer builds a 2x2 quadrature Langevin model. The pump parameter modifies the damping of the two quadratures, and detuning rotates/mixes the quadratures.

The spectrum calculation normalizes each quadrature to its high-frequency shot-noise asymptote. Therefore squeezing and anti-squeezing approach 0 dB at high analysis frequency.

## Assumptions and Limitations

- below-threshold only
- degenerate signal/idler model
- single mode
- no pump depletion
- no above-threshold dynamics
- no full non-degenerate or multimode OPO dynamics
- resonance diagnostic plots are interpretive aids, not full transfer-function calculations
