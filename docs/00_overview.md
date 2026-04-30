# 00 - Overview

QPIT-SQZsim is organized as a staged simulator for an OPO system:

```text
cavity -> crystal -> OPO
```

The stages are intentionally separated. The cavity stage defines the optical resonator, the crystal stage chooses the nonlinear operating point, and the OPO stage computes below-threshold squeezing from those upstream results.

## Documentation Map

- [01_architecture.md](01_architecture.md): module boundaries, responsibilities, and data flow.
- [02_workflow.md](02_workflow.md): how to run the pipeline and when to rerun stages.
- [03_physics.md](03_physics.md): physical models and assumptions.
- [04_outputs.md](04_outputs.md): JSON files and downstream interfaces.

Module-specific pages:

- [modules/cavity.md](modules/cavity.md)
- [modules/crystal.md](modules/crystal.md)
- [modules/opo.md](modules/opo.md)

## Main Idea

Each stage writes a JSON file. Downstream stages read these files instead of recomputing upstream decisions. This keeps the interfaces explicit:

- cavity defines geometry, mode, and losses
- crystal defines phase matching and selected nonlinear operating point
- OPO defines threshold, pump parameter, Langevin model, and spectra

## Current Scope

The simulator targets design and analysis of a below-threshold degenerate OPO. It is not currently a full multimode, above-threshold, or non-degenerate dynamics simulator.
