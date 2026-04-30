# QPIT-SQZsim

QPIT-SQZsim is a simulation framework for designing and analyzing optical parametric oscillators (OPOs), with a focus on cavity design, nonlinear crystal operation, and below-threshold squeezing performance.

The project is structured as a modular pipeline:

cavity → crystal → OPO

---

## What this project does

This framework allows you to:

- Design optical cavities and evaluate resonator properties and losses
- Compute phase matching and double-resonance conditions in nonlinear crystals
- Select physically consistent operating points
- Simulate below-threshold OPO behavior and squeezing spectra

---

## Simulation Pipeline

The simulation is performed in three sequential steps:

```
cavity
   ↓
crystal
   ↓
OPO
```

- **Cavity**: computes geometry, mode size, FSR, and loss rates (kappa, escape efficiency)
- **Crystal**: computes phase matching and double resonance, and selects the active operating point
- **OPO**: builds the operating-point model and computes squeezing spectra

Each stage consumes the output of the previous one.

---

## Installation

Clone the repository:

```bash
git clone <repository-url>
cd QPIT-SQZsim
```

Create a virtual environment (recommended):

```bash
python -m venv .venv
source .venv/bin/activate   # macOS/Linux
# .venv\Scripts\activate    # Windows
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Optional (editable install):

```bash
pip install -e .
```

---

## Quick Start

Run the full simulation pipeline:

```bash
python src/cavity/cavity_main.py
python src/crystal/crystal_main.py
python src/opo/opo_main.py
```

Each step depends on the outputs produced by the previous one.

---

## Project Structure

Main modules:

- `src/cavity` — cavity geometry, mode properties, resonant loss model, and derived cavity quantities
- `src/crystal` — phase matching, double resonance, operating-point selection, and crystal → OPO handoff
- `src/opo` — OPO operating-point model, Langevin equations, squeezing spectra, and diagnostic plots
- `src/common` — shared utilities, constants, and results-path helpers used across the pipeline

Representative files:

- `src/cavity/cavity_main.py` — cavity entry point and geometry/loss configuration
- `src/crystal/crystal_main.py` — crystal entry point and operating-point selection
- `src/opo/opo_main.py` — OPO entry point and final simulation stage
- `src/common/constants.py` — shared physical and numerical constants
- `src/common/results_paths.py` — common helpers for structured output paths

Documentation and outputs:

- `docs/` — high-level and module-specific documentation
- `results/` — generated JSON outputs and plots for each simulation stage

For detailed concepts and definitions, see the documentation in the `docs/` directory.

---

## Outputs

Each stage writes results to disk (JSON + plots). These outputs are used as inputs for the next stage in the pipeline.

---

## Documentation

High-level documentation:

- [Overview](docs/00_overview.md)
- [Architecture](docs/01_architecture.md)
- [Workflow](docs/02_workflow.md)
- [Physics](docs/03_physics.md)
- [Outputs](docs/04_outputs.md)

Module-specific documentation:

- [Cavity](docs/modules/cavity.md)
- [Crystal](docs/modules/crystal.md)
- [OPO](docs/modules/opo.md)

---

## Notes

- The current OPO model is:
  - below-threshold
  - degenerate
  - single-mode

- Full non-degenerate and multimode dynamics are not yet implemented.

---

## Future Work

The current implementation focuses on a below-threshold, degenerate, single-mode OPO model. Several extensions are planned to broaden the physical scope and modeling capabilities of the simulator.

### Planned Features

- **Non-degenerate OPO model**
  - Explicit signal and idler dynamics
  - Two-mode squeezing and frequency correlations

- **Pulsed (time-domain) squeezing**
  - Broadband / multimode description
  - Time-domain and frequency-resolved correlations beyond single-sideband analysis

- **Waveguide-based nonlinear interaction**
  - Reduced mode area and enhanced nonlinear coupling
  - Modified overlap, dispersion, and effective interaction length

---

### Design Approach

These features are intended to be integrated without modifying the high-level pipeline:

cavity → crystal → OPO

Instead of introducing new pipeline stages, they should be implemented as **configurable modes within the existing modules**:

- **Crystal module**
  - Extend to support different physical platforms (e.g. bulk vs waveguide)
  - Modify mode area, overlap, and interaction properties accordingly

- **OPO module**
  - Extend to support different dynamical regimes (e.g. continuous-wave vs pulsed)
  - Generalize the quantum model to include multi-mode and non-degenerate behavior

A possible internal structure is:

- `src/crystal/platforms/`
  - `bulk.py`
  - `waveguide.py`

- `src/opo/regimes/`
  - `cw.py`
  - `pulsed.py`


This approach preserves the modular structure of the project while enabling more advanced physical models to be added incrementally.


---

## Future Extensions: Optimization

A natural extension of the current framework is the introduction of parameter optimization capabilities.

Thanks to the modular pipeline:

cavity → crystal → OPO

it is possible to study how design parameters affect the final squeezing performance.

### Use Cases

This would allow systematic investigation of the impact of:

- Mirror coatings (input/output reflectivity and internal losses)
- Cavity geometry and beam waist
- Crystal length and phase-matching conditions
- Pump operating point

For example, varying mirror reflectivity directly affects:
- cavity losses (kappa)
- escape efficiency
- OPO threshold
- achievable squeezing

### Where Optimization Fits in the Pipeline

Optimization should not be implemented as a new stage, but as an external loop around the existing pipeline.

The idea is to repeatedly run:

cavity → crystal → OPO

while modifying selected input parameters (e.g. mirror coatings in the cavity stage).

Each iteration produces a new simulation output, from which a figure of merit (such as the minimum squeezing level) can be extracted.

In practice:

- **Cavity stage**  
  Parameters like mirror reflectivity and losses are varied here. This is typically the primary optimization entry point.

- **Crystal stage**  
  Can be optimized for phase matching or double resonance conditions (e.g. temperature, poling period, crystal length).

- **OPO stage**  
  Evaluates the final performance (squeezing spectrum), which serves as the optimization target.

This structure allows the simulator to be used not only for analysis, but also as a design tool for exploring and optimizing OPO configurations.