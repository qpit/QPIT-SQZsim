# 02 - Workflow

Run the stages in order:

```bash
python src/cavity/cavity_main.py
python src/crystal/crystal_main.py
python src/opo/opo_main.py
```

The selected `GEOMETRY` should be consistent across stages.

## 1. Cavity

`src/cavity/cavity_main.py` defines the geometry and resonant-field loss inputs. It performs geometry scans, evaluates a selected single point, computes derived cavity quantities, and writes:

```text
results/<geometry>/cavity/cavity_simulation_output.json
```

It also saves stability and waist plots.

## 2. Crystal

`src/crystal/crystal_main.py` loads the cavity JSON, resolves the material/index model, computes phase matching, evaluates double resonance, selects an operating point, evaluates nonlinear overlap, and writes:

```text
results/<geometry>/crystal/crystal_simulation_output.json
```

The standard crystal JSON is compact and focused on the OPO handoff.

## 3. OPO

`src/opo/opo_main.py` loads cavity and crystal JSON files, checks cavity/crystal consistency, computes the physical OPO threshold, resolves the pump operating point, builds the Langevin model, computes spectra, and writes:

```text
results/<geometry>/opo/opo_simulation_output.json
```

It also saves squeezing and resonance diagnostic plots.

## Rerun Rules

Rerun downstream stages whenever an upstream stage changes.

Rerun the cavity stage if the selected crystal operating point changes the crystal length. This can happen when `OPERATING_POINT_MODE = "double_resonance"`. The OPO stage checks this and stops if the cavity JSON and crystal JSON are inconsistent.

## Interactive Use

The `*_main.py` files are written as notebook-style scripts with `# %%` cells. They can be run as scripts or executed cell-by-cell in an IDE.
