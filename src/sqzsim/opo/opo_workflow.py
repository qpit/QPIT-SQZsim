"""High-level workflow assembly for OPO simulations.

The OPO layer consumes the exported cavity and crystal results for a selected
geometry, builds a minimal below-threshold degenerate OPO operating point,
prepares a Langevin-model scaffold, constructs Langevin-based squeezing spectra,
and saves the resulting outputs in ``results/<geometry>/opo/``.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any

import numpy as np

try:
    from common.results_paths import (
        ensure_geometry_results_subdirs,
        get_cavity_results_dir,
        get_crystal_results_dir,
        get_opo_results_dir,
    )
except ImportError:
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from common.results_paths import (
        ensure_geometry_results_subdirs,
        get_cavity_results_dir,
        get_crystal_results_dir,
        get_opo_results_dir,
    )

from .opo_langevin import OPOLangevinModel, build_langevin_model
from .opo_model import OPOModelResult, OPOParameters, build_opo_parameters, derive_opo_quantities
from .opo_squeezing import OPOSqueezingSpectrum, compute_squeezing_spectra, spectrum_to_dict


@dataclass(frozen=True)
class OPOContext:
    """Structured OPO input context loaded from cavity and crystal outputs."""

    geometry: str
    cavity_output_path: str
    crystal_output_path: str
    cavity_data: dict[str, Any]
    crystal_data: dict[str, Any]


@dataclass(frozen=True)
class OPOSimulationResult:
    """Combined OPO workflow output."""

    context: OPOContext
    parameters: OPOParameters
    model: OPOModelResult
    langevin: OPOLangevinModel
    spectrum: OPOSqueezingSpectrum


def _load_json(path: str | Path) -> dict[str, Any]:
    json_path = Path(path)
    if not json_path.exists():
        raise FileNotFoundError(f"Simulation output not found: {json_path}")
    with json_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_opo_context(
    geometry: str,
    cavity_output_path: str | Path | None = None,
    crystal_output_path: str | Path | None = None,
) -> OPOContext:
    """Load cavity and crystal outputs needed by the OPO workflow."""
    if cavity_output_path is None:
        cavity_output_path = get_cavity_results_dir(geometry) / "cavity_simulation_output.json"
    if crystal_output_path is None:
        crystal_output_path = get_crystal_results_dir(geometry) / "crystal_simulation_output.json"

    cavity_data = _load_json(cavity_output_path)
    crystal_data = _load_json(crystal_output_path)

    return OPOContext(
        geometry=str(cavity_data.get("geometry", cavity_data.get("inputs", {}).get("geometry", geometry))),
        cavity_output_path=str(Path(cavity_output_path)),
        crystal_output_path=str(Path(crystal_output_path)),
        cavity_data=cavity_data,
        crystal_data=crystal_data,
    )


def _validate_opo_context_consistency(context: OPOContext, tolerance_m: float = 1e-9) -> None:
    crystal_results = context.crystal_data.get("results", {})
    active_for_opo = crystal_results.get("active_for_opo", {})
    if not isinstance(active_for_opo, dict):
        return

    cavity_inputs = context.cavity_data.get("inputs", {})
    cavity_crystal_length_raw = cavity_inputs.get("crystal_length_m")
    active_crystal_length_raw = active_for_opo.get("crystal_length_m")
    recommended_cavity_length_raw = active_for_opo.get("recommended_cavity_crystal_length_m", active_crystal_length_raw)

    cavity_crystal_length_m = float(cavity_crystal_length_raw) if cavity_crystal_length_raw is not None else np.nan
    active_crystal_length_m = float(active_crystal_length_raw) if active_crystal_length_raw is not None else np.nan
    recommended_cavity_length_m = (
        float(recommended_cavity_length_raw)
        if recommended_cavity_length_raw is not None
        else active_crystal_length_m
    )

    if (
        np.isfinite(cavity_crystal_length_m)
        and np.isfinite(active_crystal_length_m)
        and abs(active_crystal_length_m - cavity_crystal_length_m) > float(tolerance_m)
    ):
        raise ValueError(
            "Selected crystal operating point is inconsistent with loaded cavity geometry.\n"
            f"Cavity JSON: {context.cavity_output_path}\n"
            f"Crystal JSON: {context.crystal_output_path}\n"
            f"Loaded cavity crystal length: {cavity_crystal_length_m:.12g} m\n"
            f"Selected crystal active length: {active_crystal_length_m:.12g} m\n"
            f"Recommended cavity crystal length: {recommended_cavity_length_m:.12g} m\n"
            "Rerun sequence:\n"
            "1. Update cavity_main.py to use the recommended cavity crystal length.\n"
            "2. Rerun cavity_main.py.\n"
            "3. Rerun crystal_main.py.\n"
            "4. Rerun opo_main.py."
        )


def compute_opo_model(context: OPOContext, config: dict[str, Any]) -> tuple[OPOParameters, OPOModelResult]:
    """Build OPO parameters and the minimal operating-point model."""
    parameters = build_opo_parameters(config)
    _validate_opo_context_consistency(context)
    model = derive_opo_quantities(parameters, context.cavity_data, context.crystal_data)
    return parameters, model


def compute_opo_langevin(model: OPOModelResult) -> OPOLangevinModel:
    """Build the Langevin scaffold for the current OPO model."""
    return build_langevin_model(model)


def compute_opo_squeezing(
    parameters: OPOParameters,
    model: OPOModelResult,
    langevin: OPOLangevinModel,
) -> OPOSqueezingSpectrum:
    """Build the Langevin-based squeezing-spectrum payload."""
    return compute_squeezing_spectra(parameters, model, langevin)


def build_opo_simulation_result(
    context: OPOContext,
    parameters: OPOParameters,
    model: OPOModelResult,
    langevin: OPOLangevinModel,
    spectrum: OPOSqueezingSpectrum,
) -> OPOSimulationResult:
    """Assemble the structured OPO workflow result."""
    return OPOSimulationResult(
        context=context,
        parameters=parameters,
        model=model,
        langevin=langevin,
        spectrum=spectrum,
    )


def print_opo_summary(result: OPOSimulationResult) -> None:
    """Print a concise OPO operating-point summary."""
    print("OPO simulation summary")
    print("----------------------")
    print(f"Geometry: {result.context.geometry}")
    print(f"Pump mode: {result.model.pump_mode}")
    print(f"Pump resonance model: {result.model.pump_resonance_model}")
    print(f"Pump parameter sigma: {result.model.pump_parameter:.6f}")
    print(f"Pump fraction P/P_thr: {result.model.pump_parameter**2:.6f}")
    print(f"External pump power: {result.model.pump_power_W:.6e} W")
    print(f"Threshold model: {result.model.threshold_model}")
    print(f"External threshold power: {result.model.threshold_external_pump_power_W:.6e} W")
    print(f"Below threshold: {result.model.below_threshold}")
    print(f"Pump conversion: {result.model.pump_conversion_assumption}")
    print(f"Pump buildup factor: {result.model.pump_buildup_factor:.6e}")
    print(f"d_eff source: {result.model.d_eff_source}")
    print(f"d_eff: {result.model.d_eff_pm_per_V:.6e} pm/V")
    print(f"Crystal length: {result.model.crystal_length_m * 1e3:.6f} mm")
    print(f"Mode area: {result.model.threshold_mode_area_m2:.6e} m^2")
    print(f"Nonlinear overlap: {result.model.threshold_overlap:.6f}")
    print(f"Nonlinear coupling: {result.model.threshold_nonlinear_coupling:.6e} rad/s")
    print(f"Cavity kappa_ext: {result.model.cavity_kappa_ext_Hz:.6f} Hz")
    print(f"Cavity kappa_loss: {result.model.cavity_kappa_loss_Hz:.6f} Hz")
    print(f"Cavity kappa_total: {result.model.cavity_kappa_total_Hz:.6f} Hz")
    print(f"Escape efficiency: {result.model.escape_efficiency:.6f}")
    print(f"Cavity detuning: {result.model.cavity_detuning_Hz:.6f} Hz")
    print(f"Crystal operating point mode: {result.model.crystal_operating_point_mode}")
    if np.isfinite(result.model.crystal_active_temperature_K):
        print(f"Crystal active temperature: {result.model.crystal_active_temperature_K:.3f} K")
    if np.isfinite(result.model.cavity_crystal_length_m):
        print(f"Cavity crystal length: {result.model.cavity_crystal_length_m * 1e3:.6f} mm")
    optimal_phase_low_frequency = float(result.spectrum.optimal_phase_rad[0])
    print(f"LO phase: {result.parameters.lo_phase_rad:.6f} rad")
    print(f"Optimal squeezing phase (low-frequency limit): {optimal_phase_low_frequency:.6f} rad")
    print(f"External detection efficiency: {result.parameters.detection_efficiency:.6f}")
    print(f"Analysis sideband: {result.parameters.analysis_sideband_Hz:.3f} Hz")


def _build_opo_inputs_payload(context: OPOContext, parameters: OPOParameters) -> dict[str, Any]:
    return {
        "geometry": context.geometry,
        "cavity_output_path": context.cavity_output_path,
        "crystal_output_path": context.crystal_output_path,
        **asdict(parameters),
    }


def _build_opo_results_payload(result: OPOSimulationResult) -> dict[str, Any]:
    model_payload = asdict(result.model)
    model_payload["notes"] = list(result.model.notes)
    model_payload["detection_efficiency"] = float(result.parameters.detection_efficiency)
    return {
        "model": model_payload,
        "spectrum": spectrum_to_dict(result.spectrum),
    }


def build_opo_simulation_output(result: OPOSimulationResult) -> dict[str, Any]:
    """Build JSON-serializable OPO simulation output."""
    output = {
        "inputs": _build_opo_inputs_payload(result.context, result.parameters),
        "results": _build_opo_results_payload(result),
    }
    output["debug_data"] = {
        "langevin": {
            "quadrature_labels": list(result.langevin.quadrature_labels),
            "drift_matrix": result.langevin.drift_matrix.tolist(),
            "input_matrix": result.langevin.input_matrix.tolist(),
            "noise_coupling_matrix": result.langevin.noise_coupling_matrix.tolist(),
            "notes": list(result.langevin.notes),
        }
    }
    return output


def save_opo_outputs(
    geometry: str,
    output: dict[str, Any],
    fig_spectrum,
    fig_resonance_diagnostic=None,
    results_root: str | Path | None = None,
) -> dict[str, str]:
    """Save OPO JSON and plots under ``results/<geometry>/opo/``."""
    ensure_geometry_results_subdirs(geometry, results_root=results_root)
    result_dir = get_opo_results_dir(geometry, results_root=results_root)
    project_root = Path(__file__).resolve().parents[2]

    json_path = result_dir / "opo_simulation_output.json"
    spectrum_path = result_dir / "opo_squeezing_spectrum.png"
    resonance_diagnostic_path = result_dir / "opo_resonance_diagnostic.png"

    def _repo_relative(path: Path) -> str:
        try:
            return str(path.resolve().relative_to(project_root))
        except ValueError:
            return str(path)

    outputs_info = {
        "result_dir": _repo_relative(result_dir),
        "opo_output_json": _repo_relative(json_path),
        "opo_squeezing_spectrum_png": _repo_relative(spectrum_path),
    }
    if fig_resonance_diagnostic is not None:
        outputs_info["opo_resonance_diagnostic_png"] = _repo_relative(resonance_diagnostic_path)

    with json_path.open("w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)

    if fig_spectrum is not None:
        fig_spectrum.savefig(spectrum_path, dpi=300, bbox_inches="tight")
    if fig_resonance_diagnostic is not None:
        fig_resonance_diagnostic.savefig(resonance_diagnostic_path, dpi=300, bbox_inches="tight")

    return outputs_info


__all__ = [
    "OPOContext",
    "OPOSimulationResult",
    "load_opo_context",
    "compute_opo_model",
    "compute_opo_langevin",
    "compute_opo_squeezing",
    "build_opo_simulation_result",
    "build_opo_simulation_output",
    "print_opo_summary",
    "save_opo_outputs",
]
