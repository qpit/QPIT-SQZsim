# 01 — Architecture

This section describes the structure of the simulator, the role of each layer, and how data flows through the pipeline.

The project is organized as a one-way pipeline:

cavity → crystal → OPO

Each layer owns a well-defined part of the model and exports the quantities needed by the next stage.

## Cavity Layer

The cavity layer defines the **optical system**.

It is responsible for:

- geometry
- Gaussian resonator mode properties
- resonant-field loss model
- derived quantities such as:
  - `kappa_ext_Hz`
  - `kappa_loss_Hz`
  - `kappa_total_Hz`
  - `escape_efficiency`

This is the **only layer that computes cavity losses and coupling rates**.

All downstream layers must use these values as-is.

## Crystal Layer

The crystal layer defines the **nonlinear interaction and operating point**.

It is responsible for:

- refractive-index and phase-matching calculations
- phase-matching and double-resonance scans
- selection of a single operating point
- mode-matching quantities
- Boyd–Kleinman summary
- polarization-resonance diagnostics

Its downstream interface is:

- `crystal.results.active_for_opo`

This block represents the **resolved crystal state** used by the OPO layer.

## OPO Layer

The OPO layer defines the **quantum model at the selected operating point**.

It consumes:

- cavity `results`
- crystal `results.active_for_opo`

It builds:

- a compact below-threshold operating-point model
- a minimal Langevin model
- squeezing and anti-squeezing spectra

**Important:**

The OPO layer is a pure consumer.

It must NOT:

- recompute cavity quantities
- reinterpret cavity geometry
- change the crystal operating point

All upstream decisions are taken as fixed inputs.

## Data Flow

The JSON outputs define the interface between layers.

- `cavity` exports geometry-dependent mode and loss quantities
- `crystal` loads the cavity output and exports a single resolved operating point
- `opo` loads both and uses them directly

This enforces a strict separation:

- geometry and losses → `cavity`
- nonlinear operating-point selection → `crystal`
- quantum noise and squeezing → `opo`

Each layer depends only on upstream outputs, never the other way around.

For execution details, see [02_workflow.md](02_workflow.md).  
For the JSON structure, see [04_outputs.md](04_outputs.md).