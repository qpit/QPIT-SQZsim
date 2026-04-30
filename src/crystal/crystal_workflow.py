"""High-level workflow assembly for crystal and mode-matching simulations."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any, Callable

import numpy as np

try:
    from common.results_paths import ensure_geometry_results_subdirs, get_cavity_results_dir, get_crystal_results_dir
except ImportError:
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from common.results_paths import ensure_geometry_results_subdirs, get_cavity_results_dir, get_crystal_results_dir

from .crystal_mode_matching import (
    ModeMatchingResult,
    build_mode_matching_context_from_cavity_output,
    estimate_mode_matching_quantities,
)
from .crystal_double_resonance_scan import compute_double_resonance_scan
from .crystal_materials import (
    build_refractive_index_model,
    get_axis_index_function,
    resolve_effective_nonlinearity,
    resolve_phase_matching_configuration,
)
from .crystal_polarization_resonance import compute_polarization_resonance_diagnostic
from .crystal_boyd_kleinman import (
    BKAnalysisConfig,
    bk_analysis_result_to_dict,
    run_bk_analysis_pair,
)
from .crystal_phase_matching import compute_design_poling_period as compute_design_poling_period_from_material_model
from .crystal_phase_matching import delta_k_eff_T, scan_phase_matching_vs_temperature

_BK_MAP_KEYS = frozenset(
    {
        "bk_master_sigma_values",
        "bk_master_xi_values",
        "bk_master_h_map",
    }
)
_CRYSTAL_LENGTH_MATCH_TOLERANCE_M = 1e-9
_INTERNAL_DEFAULTS: dict[str, Any] = {
    "cavity_output_path": None,
    "enable_double_resonance_scan": True,
    "double_resonance_t_min_K": 315.0,
    "double_resonance_t_max_K": 320.0,
    "double_resonance_n_T": 201,
    "double_resonance_l_min_m": 4.0e-3,
    "double_resonance_l_max_m": 4.2e-3,
    "double_resonance_n_L": 161,
}


@dataclass(frozen=True)
class CrystalContext:
    """Structured context loaded from cavity simulation outputs."""

    geometry: str | None
    cavity_output_path: str
    crystal_length_m: float
    wavelength_m: float
    n_crystal: float
    beam_waist_crystal_m: float
    cavity_data: dict[str, Any]


@dataclass(frozen=True)
class CrystalSimulationResult:
    """Combined crystal workflow output."""

    context: CrystalContext
    phase_matching: dict[str, Any]
    mode_matching: ModeMatchingResult
    selected_operating_phase_matching: dict[str, Any] | None = None
    phase_matching_operating_point: dict[str, Any] | None = None
    double_resonance_operating_point: dict[str, Any] | None = None
    selected_operating_point_mode: str | None = None
    selected_operating_point: dict[str, Any] | None = None
    polarization_resonance: dict[str, Any] | None = None
    active_polarization_resonance: dict[str, Any] | None = None
    double_resonance_scan: dict[str, Any] | None = None
    bk_analysis: dict[str, Any] | None = None
    simulation_inputs: dict[str, Any] | None = None
    qpm_order_m: int | None = None


def _load_cavity_output_data(path: str | Path) -> dict[str, Any]:
    cavity_output_path = Path(path)
    if not cavity_output_path.exists():
        raise FileNotFoundError(f"Cavity simulation output not found: {cavity_output_path}")
    with cavity_output_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_cavity_context_for_crystal(
    geometry: str,
    cavity_output_path: str | Path | None = None,
) -> CrystalContext:
    """Load cavity JSON output and build the crystal workflow context."""
    if cavity_output_path is None:
        cavity_output_path = get_cavity_results_dir(geometry) / "cavity_simulation_output.json"
    cavity_data = _load_cavity_output_data(cavity_output_path)

    inputs = cavity_data.get("inputs", {})
    results = cavity_data.get("results", {})
    waist_um = results.get("beam_waist_crystal_um")

    if waist_um is None:
        raise ValueError("Cavity output missing results.beam_waist_crystal_um")

    return CrystalContext(
        geometry=str(cavity_data.get("geometry", inputs.get("geometry", geometry))),
        cavity_output_path=str(Path(cavity_output_path)),
        crystal_length_m=float(inputs["crystal_length_m"]),
        wavelength_m=float(inputs["wavelength_m"]),
        n_crystal=float(inputs["n_crystal"]),
        beam_waist_crystal_m=float(waist_um) * 1e-6,
        cavity_data=cavity_data,
    )


def _phase_matching_scan_to_output(scan: dict[str, Any]) -> dict[str, Any]:
    return {key: (value.tolist() if isinstance(value, np.ndarray) else value) for key, value in scan.items()}


def _to_json_compatible(value: Any) -> Any:
    """Recursively convert NumPy-heavy workflow payloads into JSON-safe types."""
    if isinstance(value, dict):
        return {key: _to_json_compatible(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        items = [_to_json_compatible(item) for item in value]
        if len(items) == 1 and isinstance(items[0], (bool, int, float)) and not isinstance(items[0], str):
            return items[0]
        return items
    if isinstance(value, np.ndarray):
        if value.ndim == 0 or value.size == 1:
            return _to_json_compatible(value.reshape(-1)[0].item())
        return _to_json_compatible(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


def compute_crystal_phase_matching(
    context: CrystalContext,
    n_p_of_T: Callable[[float], float],
    n_s_of_T: Callable[[float], float],
    n_i_of_T: Callable[[float], float],
    wavelength_p_m: float,
    wavelength_s_m: float,
    wavelength_i_m: float,
    Lambda0_m: float,
    T_min_K: float,
    T_max_K: float,
    n_T: int,
    T0_K: float = 293.15,
    alpha_perK: float = 0.0,
    qpm_order_m: int = 1,
) -> dict[str, Any]:
    """Compute the phase-matching temperature scan from the cavity-derived context."""
    scan = scan_phase_matching_vs_temperature(
        T_min_K,
        T_max_K,
        n_T,
        wavelength_p_m,
        wavelength_s_m,
        wavelength_i_m,
        n_p_of_T,
        n_s_of_T,
        n_i_of_T,
        Lambda0_m,
        context.crystal_length_m,
        T0_K=T0_K,
        alpha_perK=alpha_perK,
        qpm_order_m=qpm_order_m,
    )
    return _phase_matching_scan_to_output(scan)


def compute_crystal_phase_matching_at_temperature(
    context: CrystalContext,
    n_p_of_T: Callable[[float], float],
    n_s_of_T: Callable[[float], float],
    n_i_of_T: Callable[[float], float],
    wavelength_p_m: float,
    wavelength_s_m: float,
    wavelength_i_m: float,
    Lambda0_m: float,
    temperature_K: float,
    T0_K: float = 293.15,
    alpha_perK: float = 0.0,
    qpm_order_m: int = 1,
) -> dict[str, Any]:
    """Compute phase-matching quantities at one explicit operating temperature."""
    result = delta_k_eff_T(
        temperature_K,
        wavelength_p_m,
        wavelength_s_m,
        wavelength_i_m,
        n_p_of_T,
        n_s_of_T,
        n_i_of_T,
        Lambda0_m,
        context.crystal_length_m,
        T0_K=T0_K,
        alpha_perK=alpha_perK,
        qpm_order_m=qpm_order_m,
    )
    return {
        "T_K": [result.T_K],
        "n_p": [result.n_p],
        "n_s": [result.n_s],
        "n_i": [result.n_i],
        "delta_k_rad_per_m": [result.delta_k_rad_per_m],
        "delta_k_eff_rad_per_m": [result.delta_k_eff_rad_per_m],
        "pm_power": [result.pm_power],
        "Lambda_T_m": [result.Lambda_T_m],
        "T_best_K": [result.T_K],
        "pm_power_best": [result.pm_power],
        "Lambda0_input_m": [float(Lambda0_m)],
        "Lambda0_effective_m": [float(Lambda0_m)],
        "T_reference_for_lambda_opt_K": [result.T_K],
    }


def compute_design_poling_period(
    wavelength_p_m: float,
    wavelength_s_m: float,
    wavelength_i_m: float,
    temperature_K: float,
    n_p_of_lambda_T: Callable[[float, float], float],
    n_s_of_lambda_T: Callable[[float, float], float],
    n_i_of_lambda_T: Callable[[float, float], float],
    qpm_order_m: int = 1,
) -> Any:
    """Compute the design poling period from wavelengths and one design temperature."""
    return compute_design_poling_period_from_material_model(
        wavelength_p_m=wavelength_p_m,
        wavelength_s_m=wavelength_s_m,
        wavelength_i_m=wavelength_i_m,
        temperature_K=temperature_K,
        n_p_of_lambda_T=n_p_of_lambda_T,
        n_s_of_lambda_T=n_s_of_lambda_T,
        n_i_of_lambda_T=n_i_of_lambda_T,
        qpm_order_m=qpm_order_m,
    )


def compute_crystal_mode_matching(
    context: CrystalContext,
    n_crystal: float | None = None,
    delta_k_rad_per_m: float = 0.0,
) -> ModeMatchingResult:
    """Compute focusing and mode-matching quantities from the cavity-derived beam parameters."""
    cavity_mode_matching_context = build_mode_matching_context_from_cavity_output(context.cavity_data)
    waist_crystal_m = cavity_mode_matching_context.waist_crystal_m or context.beam_waist_crystal_m
    medium_index = context.n_crystal if n_crystal is None else float(n_crystal)
    return estimate_mode_matching_quantities(
        waist_crystal_m=float(waist_crystal_m),
        crystal_length_m=context.crystal_length_m,
        wavelength_m=context.wavelength_m,
        n_crystal=medium_index,
        delta_k_rad_per_m=delta_k_rad_per_m,
    )


def compute_boyd_kleinman_analysis(
    context: CrystalContext,
    mode_matching: ModeMatchingResult,
    n_p_of_T: Callable[[float], float],
    n_s_of_T: Callable[[float], float],
    n_i_of_T: Callable[[float], float],
    n_p_of_lambda_T: Callable[[float, float], float],
    n_s_of_lambda_T: Callable[[float, float], float],
    n_i_of_lambda_T: Callable[[float, float], float],
    wavelength_p_m: float,
    wavelength_s_m: float,
    wavelength_i_m: float,
    Lambda0_m: float,
    n_T: int,
    T0_K: float = 293.15,
    alpha_perK: float = 0.0,
    qpm_order_m: int = 1,
    phase_matching: dict[str, Any] | None = None,
    bk_config: BKAnalysisConfig | None = None,
) -> dict[str, Any]:
    """Orchestrate BK analysis and expose the plotting-compatible dictionary payload.

    The workflow layer delegates all BK-specific reference construction,
    sweeps, normalization, defaults, and metadata assembly to
    ``crystal_boyd_kleinman``.
    """
    bk_results = run_bk_analysis_pair(
        context=context,
        mode_matching=mode_matching,
        n_p_of_T=n_p_of_T,
        n_s_of_T=n_s_of_T,
        n_i_of_T=n_i_of_T,
        n_p_of_lambda_T=n_p_of_lambda_T,
        n_s_of_lambda_T=n_s_of_lambda_T,
        n_i_of_lambda_T=n_i_of_lambda_T,
        wavelength_p_m=wavelength_p_m,
        wavelength_s_m=wavelength_s_m,
        wavelength_i_m=wavelength_i_m,
        Lambda0_m=Lambda0_m,
        T0_K=T0_K,
        alpha_perK=alpha_perK,
        qpm_order_m=qpm_order_m,
        phase_matching=phase_matching,
        n_temperature=n_T,
        bk_config=bk_config,
    )
    bk_operating = bk_analysis_result_to_dict(bk_results["bk_analysis_operating"])
    bk_optimal = bk_analysis_result_to_dict(bk_results["bk_analysis_optimal"])
    return {
        "reference": bk_operating.get("reference", {}),
        "bk_master_sigma_opt": bk_operating.get("bk_master_sigma_opt"),
        "bk_master_xi_opt": bk_operating.get("bk_master_xi_opt"),
        "bk_master_h_opt": bk_operating.get("bk_master_h_opt"),
        "bk_analysis_operating": bk_operating,
        "bk_analysis_optimal": bk_optimal,
    }


def build_crystal_simulation_result(
    context: CrystalContext,
    phase_matching: dict[str, Any],
    mode_matching: ModeMatchingResult,
    selected_operating_phase_matching: dict[str, Any] | None = None,
    phase_matching_operating_point: dict[str, Any] | None = None,
    double_resonance_operating_point: dict[str, Any] | None = None,
    selected_operating_point_mode: str | None = None,
    selected_operating_point: dict[str, Any] | None = None,
    polarization_resonance: dict[str, Any] | None = None,
    active_polarization_resonance: dict[str, Any] | None = None,
    double_resonance_scan: dict[str, Any] | None = None,
    bk_analysis: dict[str, Any] | None = None,
    simulation_inputs: dict[str, Any] | None = None,
    qpm_order_m: int | None = None,
) -> CrystalSimulationResult:
    """Build the structured crystal workflow result."""
    return CrystalSimulationResult(
        context=context,
        phase_matching=phase_matching,
        mode_matching=mode_matching,
        selected_operating_phase_matching=selected_operating_phase_matching,
        phase_matching_operating_point=phase_matching_operating_point,
        double_resonance_operating_point=double_resonance_operating_point,
        selected_operating_point_mode=selected_operating_point_mode,
        selected_operating_point=selected_operating_point,
        polarization_resonance=polarization_resonance,
        active_polarization_resonance=active_polarization_resonance,
        double_resonance_scan=double_resonance_scan,
        bk_analysis=bk_analysis,
        simulation_inputs=simulation_inputs,
        qpm_order_m=qpm_order_m,
    )


def build_phase_matching_operating_point(
    phase_matching: dict[str, Any],
    resonance_diagnostic: dict[str, Any] | None,
) -> dict[str, Any] | None:
    """Build the phase-matching operating point candidate."""
    if resonance_diagnostic is None:
        return None
    required_phase_keys = ("T_best_K", "pm_power_best")
    required_resonance_keys = (
        "signal_axis",
        "idler_axis",
        "n_signal",
        "n_idler",
        "delta_phi_wrapped_rad",
        "is_double_resonant",
    )
    missing_phase_keys = [key for key in required_phase_keys if key not in phase_matching]
    missing_resonance_keys = [key for key in required_resonance_keys if key not in resonance_diagnostic]
    if missing_phase_keys:
        raise ValueError(f"Cannot build phase-matching operating point; missing phase keys: {missing_phase_keys}")
    if missing_resonance_keys:
        raise ValueError(
            "Cannot build phase-matching operating point; "
            f"missing resonance keys: {missing_resonance_keys}"
        )

    return {
        "temperature_K": float(phase_matching["T_best_K"][0]),
        "phase_matching_power_factor": float(phase_matching["pm_power_best"][0]),
        "signal_axis": resonance_diagnostic["signal_axis"],
        "idler_axis": resonance_diagnostic["idler_axis"],
        "n_signal": float(resonance_diagnostic["n_signal"]),
        "n_idler": float(resonance_diagnostic["n_idler"]),
        "wrapped_phase_mismatch_rad": float(resonance_diagnostic["delta_phi_wrapped_rad"]),
        "is_double_resonant": bool(resonance_diagnostic["is_double_resonant"]),
    }


def build_double_resonance_operating_point(double_resonance_scan: dict[str, Any] | None) -> dict[str, Any] | None:
    """Build the double-resonance operating point candidate."""
    if double_resonance_scan is None:
        return None
    required_scan_keys = (
        "best_temperature_K",
        "best_crystal_length_m",
        "best_delta_phi_wrapped_rad",
        "best_abs_delta_phi_wrapped_rad",
        "best_is_double_resonant",
    )
    missing_scan_keys = [key for key in required_scan_keys if key not in double_resonance_scan]
    if missing_scan_keys:
        raise ValueError(
            "Cannot build double-resonance operating point; "
            f"missing scan keys: {missing_scan_keys}"
        )

    return {
        "temperature_K": float(double_resonance_scan["best_temperature_K"]),
        "crystal_length_m": float(double_resonance_scan["best_crystal_length_m"]),
        "wrapped_phase_mismatch_rad": float(double_resonance_scan["best_delta_phi_wrapped_rad"]),
        "abs_wrapped_phase_mismatch_rad": float(double_resonance_scan["best_abs_delta_phi_wrapped_rad"]),
        "is_double_resonant": bool(double_resonance_scan["best_is_double_resonant"]),
    }


def select_crystal_operating_point(
    mode: str,
    phase_matching_operating_point: dict[str, Any] | None,
    double_resonance_operating_point: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Return the active crystal operating point for one explicit selection mode."""
    normalized_mode = str(mode).strip().lower()
    candidates = {
        "phase_matching": phase_matching_operating_point,
        "double_resonance": double_resonance_operating_point,
    }
    if normalized_mode not in candidates:
        supported = ", ".join(sorted(candidates))
        raise ValueError(f"Unknown OPERATING_POINT_MODE: {mode}. Supported values: {supported}")

    selected_operating_point = candidates[normalized_mode]
    if selected_operating_point is None:
        available_modes = sorted(key for key, value in candidates.items() if value is not None)
        available = ", ".join(available_modes) if available_modes else "none"
        raise ValueError(
            f"Requested operating-point mode '{normalized_mode}' is unavailable for this run. "
            f"Available modes: {available}."
        )
    if "temperature_K" not in selected_operating_point:
        raise ValueError(
            f"Selected operating point '{normalized_mode}' is missing required key: temperature_K"
        )
    return selected_operating_point


def _get_config(config: dict[str, Any], key: str, default: Any = None) -> Any:
    """Read one config value, accepting the uppercase names used by entry scripts."""
    if key in config:
        return config[key]
    lower_key = key.lower()
    if lower_key in config:
        return config[lower_key]
    return default


def _merge_defaults(config: dict[str, Any]) -> dict[str, Any]:
    """Merge internal workflow defaults with caller-provided configuration."""
    return {**_INTERNAL_DEFAULTS, **config}


def _resolve_crystal_config(config: dict[str, Any]) -> dict[str, Any]:
    """Normalize entry-point config names after defaults have been merged."""
    cfg = config
    return {
        "geometry": str(_get_config(cfg, "GEOMETRY")),
        "crystal_model": str(_get_config(cfg, "CRYSTAL_MODEL", "Kato2002")),
        "wavelength_p_m": float(_get_config(cfg, "WAVELENGTH_P_M")),
        "wavelength_s_m": float(_get_config(cfg, "WAVELENGTH_S_M")),
        "wavelength_i_m": float(_get_config(cfg, "WAVELENGTH_I_M")),
        "phase_matching_mode": str(_get_config(cfg, "PHASE_MATCHING_MODE", "design")),
        "phase_matching_type": str(_get_config(cfg, "PHASE_MATCHING_TYPE", "type_II")),
        "design_temperature_K": float(_get_config(cfg, "DESIGN_TEMPERATURE_K")),
        "analysis_lambda0_m": float(_get_config(cfg, "ANALYSIS_LAMBDA0_M")),
        "T_min_K": float(_get_config(cfg, "T_MIN_K")),
        "T_max_K": float(_get_config(cfg, "T_MAX_K")),
        "n_T": int(_get_config(cfg, "N_T")),
        "T0_K": float(_get_config(cfg, "T0_K", 293.15)),
        "alpha_perK": float(_get_config(cfg, "ALPHA_PER_K", 0.0)),
        "qpm_order_m": int(_get_config(cfg, "QPM_ORDER_M", 1)),
        "operating_point_mode": str(_get_config(cfg, "OPERATING_POINT_MODE", "double_resonance")),
        "cavity_output_path": _get_config(cfg, "cavity_output_path"),
        "enable_double_resonance_scan": bool(_get_config(cfg, "enable_double_resonance_scan")),
        "double_resonance_t_min_K": float(_get_config(cfg, "double_resonance_t_min_K")),
        "double_resonance_t_max_K": float(_get_config(cfg, "double_resonance_t_max_K")),
        "double_resonance_n_T": int(_get_config(cfg, "double_resonance_n_T")),
        "double_resonance_l_min_m": float(_get_config(cfg, "double_resonance_l_min_m")),
        "double_resonance_l_max_m": float(_get_config(cfg, "double_resonance_l_max_m")),
        "double_resonance_n_L": int(_get_config(cfg, "double_resonance_n_L")),
    }


def _build_temperature_index_function(axis_model: Callable[[float, float], float], wavelength_m: float):
    """Bind one axis model to a fixed wavelength for a temperature scan."""

    def _index_of_T(T_K: float) -> float:
        return float(axis_model(wavelength_m, T_K))

    return _index_of_T


def _build_wavelength_temperature_index_function(axis_model: Callable[[float, float], float]):
    """Expose one axis model directly as ``n(lambda, T)``."""

    def _index_of_lambda_T(wavelength_m: float, T_K: float) -> float:
        return float(axis_model(wavelength_m, T_K))

    return _index_of_lambda_T


def _build_refractive_index_functions(
    crystal_model: str,
    phase_matching_type: str,
    wavelength_p_m: float,
    wavelength_s_m: float,
    wavelength_i_m: float,
) -> dict[str, Any]:
    refractive_index_model = build_refractive_index_model(crystal_model)
    phase_config = resolve_phase_matching_configuration(phase_matching_type)
    d_eff_config = resolve_effective_nonlinearity(crystal_model, phase_matching_type)

    pump_axis_model = get_axis_index_function(refractive_index_model, phase_config.pump_axis)
    signal_axis_model = get_axis_index_function(refractive_index_model, phase_config.signal_axis)
    idler_axis_model = get_axis_index_function(refractive_index_model, phase_config.idler_axis)

    return {
        "phase_config": phase_config,
        "d_eff_config": d_eff_config,
        "signal_axis_model": signal_axis_model,
        "n_p_of_T": _build_temperature_index_function(pump_axis_model, wavelength_p_m),
        "n_s_of_T": _build_temperature_index_function(signal_axis_model, wavelength_s_m),
        "n_i_of_T": _build_temperature_index_function(idler_axis_model, wavelength_i_m),
        "n_p_of_lambda_T": _build_wavelength_temperature_index_function(pump_axis_model),
        "n_s_of_lambda_T": _build_wavelength_temperature_index_function(signal_axis_model),
        "n_i_of_lambda_T": _build_wavelength_temperature_index_function(idler_axis_model),
    }


def _build_simulation_inputs_payload(
    *,
    crystal_model: str,
    mode_matching_n_crystal: float,
    phase_matching_mode: str,
    phase_config: Any,
    d_eff_config: Any,
    operating_point_mode: str,
    design_temperature_K: float,
    Lambda0_m: float,
    design_poling: Any | None,
) -> dict[str, Any]:
    payload = {
        "crystal_model": crystal_model,
        "n_crystal": mode_matching_n_crystal,
        "phase_matching_mode": phase_matching_mode,
        "phase_matching_type": phase_config.phase_matching_type,
        "pump_axis": phase_config.pump_axis,
        "signal_axis": phase_config.signal_axis,
        "idler_axis": phase_config.idler_axis,
        "d_eff_pm_per_V": d_eff_config.d_eff_pm_per_V,
        "d_eff_notes": list(d_eff_config.notes),
        "operating_point_mode": operating_point_mode,
        "design_temperature_K": design_temperature_K if str(phase_matching_mode).strip().lower() == "design" else None,
        "Lambda0_m": Lambda0_m,
        "store_bk_map": False,
    }
    if design_poling is not None:
        payload["delta_k_bulk_design_rad_per_m"] = design_poling.delta_k_bulk_rad_per_m
    return payload


def _setup_simulation(config: dict[str, Any]) -> dict[str, Any]:
    """Resolve config, material models, index functions, and cavity context."""
    resolved = _resolve_crystal_config(config)
    cfg = {**config, **resolved}
    index_functions = _build_refractive_index_functions(
        crystal_model=cfg["crystal_model"],
        phase_matching_type=cfg["phase_matching_type"],
        wavelength_p_m=cfg["wavelength_p_m"],
        wavelength_s_m=cfg["wavelength_s_m"],
        wavelength_i_m=cfg["wavelength_i_m"],
    )
    context = load_cavity_context_for_crystal(
        cfg["geometry"],
        cavity_output_path=cfg["cavity_output_path"],
    )
    return {
        "cfg": cfg,
        "context": context,
        "index_functions": index_functions,
        "phase_config": index_functions["phase_config"],
        "d_eff_config": index_functions["d_eff_config"],
    }


def _compute_phase_matching_block(
    *,
    cfg: dict[str, Any],
    context: CrystalContext,
    phase_config: Any,
    d_eff_config: Any,
    index_functions: dict[str, Any],
) -> dict[str, Any]:
    """Compute poling design, the phase scan, and the phase-scan resonance diagnostic."""
    n_p_of_T = index_functions["n_p_of_T"]
    n_s_of_T = index_functions["n_s_of_T"]
    n_i_of_T = index_functions["n_i_of_T"]
    n_p_of_lambda_T = index_functions["n_p_of_lambda_T"]
    n_s_of_lambda_T = index_functions["n_s_of_lambda_T"]
    n_i_of_lambda_T = index_functions["n_i_of_lambda_T"]

    phase_matching_mode = str(cfg["phase_matching_mode"]).strip().lower()
    if phase_matching_mode == "design":
        design_poling = compute_design_poling_period(
            wavelength_p_m=cfg["wavelength_p_m"],
            wavelength_s_m=cfg["wavelength_s_m"],
            wavelength_i_m=cfg["wavelength_i_m"],
            temperature_K=cfg["design_temperature_K"],
            n_p_of_lambda_T=n_p_of_lambda_T,
            n_s_of_lambda_T=n_s_of_lambda_T,
            n_i_of_lambda_T=n_i_of_lambda_T,
            qpm_order_m=cfg["qpm_order_m"],
        )
        Lambda0_m = float(design_poling.Lambda0_design_m)
    elif phase_matching_mode == "analysis":
        design_poling = None
        Lambda0_m = float(cfg["analysis_lambda0_m"])
    else:
        raise ValueError(f"Unknown PHASE_MATCHING_MODE: {cfg['phase_matching_mode']}")

    phase = compute_crystal_phase_matching(
        context,
        n_p_of_T=n_p_of_T,
        n_s_of_T=n_s_of_T,
        n_i_of_T=n_i_of_T,
        wavelength_p_m=cfg["wavelength_p_m"],
        wavelength_s_m=cfg["wavelength_s_m"],
        wavelength_i_m=cfg["wavelength_i_m"],
        Lambda0_m=Lambda0_m,
        T_min_K=cfg["T_min_K"],
        T_max_K=cfg["T_max_K"],
        n_T=cfg["n_T"],
        T0_K=cfg["T0_K"],
        alpha_perK=cfg["alpha_perK"],
        qpm_order_m=cfg["qpm_order_m"],
    )
    phase.update(
        {
            "phase_matching_type": phase_config.phase_matching_type,
            "pump_axis": phase_config.pump_axis,
            "signal_axis": phase_config.signal_axis,
            "idler_axis": phase_config.idler_axis,
            "d_eff_pm_per_V": d_eff_config.d_eff_pm_per_V,
            "d_eff_notes": list(d_eff_config.notes),
        }
    )

    phase_matching_temperature_K = float(phase["T_best_K"][0])
    phase_matching_resonance_diagnostic = compute_polarization_resonance_diagnostic(
        cavity_data=context.cavity_data,
        temperature_K=phase_matching_temperature_K,
        signal_axis=phase_config.signal_axis,
        idler_axis=phase_config.idler_axis,
        wavelength_s_m=cfg["wavelength_s_m"],
        wavelength_i_m=cfg["wavelength_i_m"],
        n_s_of_lambda_T=n_s_of_lambda_T,
        n_i_of_lambda_T=n_i_of_lambda_T,
    )
    return {
        "Lambda0_m": Lambda0_m,
        "design_poling": design_poling,
        "phase": phase,
        "phase_matching_resonance_diagnostic": phase_matching_resonance_diagnostic,
    }


def _compute_operating_point_block(
    *,
    cfg: dict[str, Any],
    context: CrystalContext,
    phase_config: Any,
    d_eff_config: Any,
    index_functions: dict[str, Any],
    Lambda0_m: float,
    phase: dict[str, Any],
    phase_matching_resonance_diagnostic: dict[str, Any],
) -> dict[str, Any]:
    """Compute double resonance, select the active operating point, and evaluate it."""
    n_p_of_T = index_functions["n_p_of_T"]
    n_s_of_T = index_functions["n_s_of_T"]
    n_i_of_T = index_functions["n_i_of_T"]
    n_s_of_lambda_T = index_functions["n_s_of_lambda_T"]
    n_i_of_lambda_T = index_functions["n_i_of_lambda_T"]

    double_resonance_scan = None
    if cfg["enable_double_resonance_scan"]:
        crystal_length_min_m = 0.8 * float(context.crystal_length_m)
        crystal_length_max_m = 1.2 * float(context.crystal_length_m)

        double_resonance_scan = compute_double_resonance_scan(
            cavity_data=context.cavity_data,
            signal_axis=phase_config.signal_axis,
            idler_axis=phase_config.idler_axis,
            wavelength_s_m=cfg["wavelength_s_m"],
            wavelength_i_m=cfg["wavelength_i_m"],
            n_s_of_lambda_T=n_s_of_lambda_T,
            n_i_of_lambda_T=n_i_of_lambda_T,
            temperature_min_K=cfg["double_resonance_t_min_K"],
            temperature_max_K=cfg["double_resonance_t_max_K"],
            n_temperature=cfg["double_resonance_n_T"],
            crystal_length_min_m=crystal_length_min_m,
            crystal_length_max_m=crystal_length_max_m,
            n_crystal_length=cfg["double_resonance_n_L"],
        )

    # 6. Operating-point selection
    phase_matching_operating_point = build_phase_matching_operating_point(
        phase_matching=phase,
        resonance_diagnostic=phase_matching_resonance_diagnostic,
    )
    double_resonance_operating_point = build_double_resonance_operating_point(double_resonance_scan)
    selected_operating_point = select_crystal_operating_point(
        cfg["operating_point_mode"],
        phase_matching_operating_point=phase_matching_operating_point,
        double_resonance_operating_point=double_resonance_operating_point,
    )
    active_operating_temperature_K = float(selected_operating_point["temperature_K"])

    active_polarization_resonance = compute_polarization_resonance_diagnostic(
        cavity_data=context.cavity_data,
        temperature_K=active_operating_temperature_K,
        signal_axis=phase_config.signal_axis,
        idler_axis=phase_config.idler_axis,
        wavelength_s_m=cfg["wavelength_s_m"],
        wavelength_i_m=cfg["wavelength_i_m"],
        n_s_of_lambda_T=n_s_of_lambda_T,
        n_i_of_lambda_T=n_i_of_lambda_T,
    )
    phase_active = compute_crystal_phase_matching_at_temperature(
        context,
        n_p_of_T=n_p_of_T,
        n_s_of_T=n_s_of_T,
        n_i_of_T=n_i_of_T,
        wavelength_p_m=cfg["wavelength_p_m"],
        wavelength_s_m=cfg["wavelength_s_m"],
        wavelength_i_m=cfg["wavelength_i_m"],
        Lambda0_m=Lambda0_m,
        temperature_K=active_operating_temperature_K,
        T0_K=cfg["T0_K"],
        alpha_perK=cfg["alpha_perK"],
        qpm_order_m=cfg["qpm_order_m"],
    )
    phase_active.update(
        {
            "phase_matching_type": phase_config.phase_matching_type,
            "pump_axis": phase_config.pump_axis,
            "signal_axis": phase_config.signal_axis,
            "idler_axis": phase_config.idler_axis,
            "d_eff_pm_per_V": d_eff_config.d_eff_pm_per_V,
            "d_eff_notes": list(d_eff_config.notes),
        }
    )
    return {
        "double_resonance_scan": double_resonance_scan,
        "phase_matching_operating_point": phase_matching_operating_point,
        "double_resonance_operating_point": double_resonance_operating_point,
        "selected_operating_point": selected_operating_point,
        "active_operating_temperature_K": active_operating_temperature_K,
        "active_polarization_resonance": active_polarization_resonance,
        "phase_active": phase_active,
    }


def _compute_physics_block(
    *,
    cfg: dict[str, Any],
    context: CrystalContext,
    index_functions: dict[str, Any],
    phase_active: dict[str, Any],
    active_operating_temperature_K: float,
    Lambda0_m: float,
) -> dict[str, Any]:
    """Compute mode matching and Boyd-Kleinman analysis at the active operating point."""
    n_p_of_T = index_functions["n_p_of_T"]
    n_s_of_T = index_functions["n_s_of_T"]
    n_i_of_T = index_functions["n_i_of_T"]
    n_p_of_lambda_T = index_functions["n_p_of_lambda_T"]
    n_s_of_lambda_T = index_functions["n_s_of_lambda_T"]
    n_i_of_lambda_T = index_functions["n_i_of_lambda_T"]

    mode_matching_n_crystal = float(
        index_functions["signal_axis_model"](
            cfg["wavelength_s_m"],
            active_operating_temperature_K,
        )
    )
    mode = compute_crystal_mode_matching(
        context,
        n_crystal=mode_matching_n_crystal,
    )
    bk_data = compute_boyd_kleinman_analysis(
        context=context,
        phase_matching=phase_active,
        mode_matching=mode,
        n_p_of_T=n_p_of_T,
        n_s_of_T=n_s_of_T,
        n_i_of_T=n_i_of_T,
        n_p_of_lambda_T=n_p_of_lambda_T,
        n_s_of_lambda_T=n_s_of_lambda_T,
        n_i_of_lambda_T=n_i_of_lambda_T,
        wavelength_p_m=cfg["wavelength_p_m"],
        wavelength_s_m=cfg["wavelength_s_m"],
        wavelength_i_m=cfg["wavelength_i_m"],
        Lambda0_m=Lambda0_m,
        n_T=cfg["n_T"],
        T0_K=cfg["T0_K"],
        alpha_perK=cfg["alpha_perK"],
        qpm_order_m=cfg["qpm_order_m"],
    )
    return {
        "mode": mode,
        "mode_matching_n_crystal": mode_matching_n_crystal,
        "bk_data": bk_data,
    }


def _build_result_from_blocks(
    *,
    setup: dict[str, Any],
    phase_block: dict[str, Any],
    operating_block: dict[str, Any],
    physics_block: dict[str, Any],
) -> CrystalSimulationResult:
    cfg = setup["cfg"]
    return build_crystal_simulation_result(
        context=setup["context"],
        phase_matching=phase_block["phase"],
        mode_matching=physics_block["mode"],
        selected_operating_phase_matching=operating_block["phase_active"],
        phase_matching_operating_point=operating_block["phase_matching_operating_point"],
        double_resonance_operating_point=operating_block["double_resonance_operating_point"],
        selected_operating_point_mode=cfg["operating_point_mode"],
        selected_operating_point=operating_block["selected_operating_point"],
        polarization_resonance=phase_block["phase_matching_resonance_diagnostic"],
        active_polarization_resonance=operating_block["active_polarization_resonance"],
        double_resonance_scan=operating_block["double_resonance_scan"],
        bk_analysis=physics_block["bk_data"],
        qpm_order_m=cfg["qpm_order_m"],
        simulation_inputs=_build_simulation_inputs_payload(
            crystal_model=cfg["crystal_model"],
            mode_matching_n_crystal=physics_block["mode_matching_n_crystal"],
            phase_matching_mode=cfg["phase_matching_mode"],
            phase_config=setup["phase_config"],
            d_eff_config=setup["d_eff_config"],
            operating_point_mode=cfg["operating_point_mode"],
            design_temperature_K=cfg["design_temperature_K"],
            Lambda0_m=phase_block["Lambda0_m"],
            design_poling=phase_block["design_poling"],
        ),
    )


def run_crystal_simulation(config: dict[str, Any]) -> CrystalSimulationResult:
    """Run the crystal workflow end-to-end from a small entry-point config."""
    cfg = _merge_defaults(config)
    setup = _setup_simulation(cfg)

    phase_block = _compute_phase_matching_block(
        cfg=setup["cfg"],
        context=setup["context"],
        phase_config=setup["phase_config"],
        d_eff_config=setup["d_eff_config"],
        index_functions=setup["index_functions"],
    )
    operating_block = _compute_operating_point_block(
        cfg=setup["cfg"],
        context=setup["context"],
        phase_config=setup["phase_config"],
        d_eff_config=setup["d_eff_config"],
        index_functions=setup["index_functions"],
        Lambda0_m=phase_block["Lambda0_m"],
        phase=phase_block["phase"],
        phase_matching_resonance_diagnostic=phase_block["phase_matching_resonance_diagnostic"],
    )
    physics_block = _compute_physics_block(
        cfg=setup["cfg"],
        context=setup["context"],
        index_functions=setup["index_functions"],
        phase_active=operating_block["phase_active"],
        active_operating_temperature_K=operating_block["active_operating_temperature_K"],
        Lambda0_m=phase_block["Lambda0_m"],
    )
    return _build_result_from_blocks(
        setup=setup,
        phase_block=phase_block,
        operating_block=operating_block,
        physics_block=physics_block,
    )


def setup_crystal_simulation(config: dict[str, Any]) -> dict[str, Any]:
    """Set up one crystal simulation, merging internal workflow defaults once."""
    return _setup_simulation(_merge_defaults(config))


compute_crystal_phase_matching_block = _compute_phase_matching_block
compute_crystal_operating_point_block = _compute_operating_point_block
compute_crystal_physics_block = _compute_physics_block
build_crystal_simulation_result_from_blocks = _build_result_from_blocks


def print_crystal_summary(result: CrystalSimulationResult) -> None:
    """Print concise phase-matching and mode-matching summary."""
    phase = result.phase_matching
    mode = result.mode_matching
    bk_analysis = result.bk_analysis
    t_best = float(phase["T_best_K"][0])
    pm_best = float(phase["pm_power_best"][0])

    print("Crystal simulation summary")
    print("-------------------------")
    print(f"Geometry: {result.context.geometry}")
    simulation_inputs = result.simulation_inputs or {}
    phase_matching_mode = simulation_inputs.get("phase_matching_mode")
    Lambda0_m = simulation_inputs.get("Lambda0_m")
    if phase_matching_mode is not None:
        print(f"Phase-matching mode: {phase_matching_mode}")
        if Lambda0_m is not None:
            Lambda0_m = float(Lambda0_m)
            Lambda0_um = Lambda0_m * 1e6
            if str(phase_matching_mode).strip().lower() == "design":
                print(f"Designed QPM poling period Lambda0: {Lambda0_um:.6f} um ({Lambda0_m:.6e} m)")
            else:
                print(f"Fixed input poling period Lambda0: {Lambda0_um:.6f} um ({Lambda0_m:.6e} m)")
        design_temperature_K = simulation_inputs.get("design_temperature_K")
        if design_temperature_K is not None:
            print(f"Design temperature: {float(design_temperature_K):.3f} K")
        if result.qpm_order_m is not None:
            print(f"QPM order: {int(result.qpm_order_m)}")
    phase_matching_type = phase.get("phase_matching_type")
    pump_axis = phase.get("pump_axis")
    signal_axis = phase.get("signal_axis")
    idler_axis = phase.get("idler_axis")
    if phase_matching_type is not None and pump_axis is not None and signal_axis is not None and idler_axis is not None:
        print(
            "Interaction type: "
            f"{phase_matching_type} ({pump_axis} -> {signal_axis} + {idler_axis})"
        )
    d_eff_pm_per_V = phase.get("d_eff_pm_per_V")
    if d_eff_pm_per_V is not None:
        print(f"Effective nonlinearity d_eff: {float(d_eff_pm_per_V):.6e} pm/V")
    print(f"Best phase-matching temperature: {t_best:.3f} K")
    print(f"Best phase-matching power factor: {pm_best:.6f}")
    print(f"Beam waist in crystal: {mode.waist_crystal_m*1e6:.3f} um")
    print(f"Rayleigh range in crystal: {mode.rayleigh_range_m*1e3:.3f} mm")
    print(f"Focusing parameter xi: {mode.focusing_parameter_xi:.6f}")
    print(f"Boyd-Kleinman factor: {mode.boyd_kleinman_factor:.6f}")
    print(f"Effective nonlinear overlap: {mode.effective_nonlinear_overlap:.6f}")

    phase_matching_operating_point = result.phase_matching_operating_point
    if phase_matching_operating_point is not None:
        print("Phase-matching operating point:")
        print(f"  temperature: {float(phase_matching_operating_point['temperature_K']):.3f} K")
        print(
            f"  phase-matching power factor: "
            f"{float(phase_matching_operating_point['phase_matching_power_factor']):.6f}"
        )
        print(
            f"  axes (signal/idler): {phase_matching_operating_point['signal_axis']} / "
            f"{phase_matching_operating_point['idler_axis']}"
        )
        print(
            f"  n(signal/idler): {float(phase_matching_operating_point['n_signal']):.6f} / "
            f"{float(phase_matching_operating_point['n_idler']):.6f}"
        )
        print(
            f"  wrapped phase mismatch: "
            f"{float(phase_matching_operating_point['wrapped_phase_mismatch_rad']):.6e} rad"
        )
        print(f"  double resonant: {bool(phase_matching_operating_point['is_double_resonant'])}")

    if result.double_resonance_operating_point is not None:
        double_resonance_operating_point = result.double_resonance_operating_point
        print("Double-resonance operating point:")
        print(f"  temperature: {float(double_resonance_operating_point['temperature_K']):.3f} K")
        print(f"  crystal length: {float(double_resonance_operating_point['crystal_length_m']) * 1e3:.3f} mm")
        print(
            f"  wrapped phase mismatch: "
            f"{float(double_resonance_operating_point['wrapped_phase_mismatch_rad']):.6e} rad"
        )
        print(f"  double resonant: {bool(double_resonance_operating_point['is_double_resonant'])}")

    if result.selected_operating_point_mode is not None:
        print(f"Selected operating point mode: {result.selected_operating_point_mode}")
    if result.selected_operating_point is not None:
        print(f"Active operating temperature: {float(result.selected_operating_point['temperature_K']):.3f} K")
        active_crystal_length_m = float(
            result.selected_operating_point.get("crystal_length_m", result.context.crystal_length_m)
        )
        if abs(active_crystal_length_m - float(result.context.crystal_length_m)) > _CRYSTAL_LENGTH_MATCH_TOLERANCE_M:
            print(
                "Selected crystal length differs from loaded cavity geometry; "
                f"rerun cavity with {active_crystal_length_m * 1e3:.6f} mm before OPO."
            )
    if result.active_polarization_resonance is not None:
        print("Active polarization-resonance diagnostic evaluated at selected operating temperature.")

    if bk_analysis is not None:
        reference = bk_analysis.get("reference", {})
        sigma_reference = reference.get("sigma_reference")
        xi_reference = reference.get("xi_reference")
        bk_master_sigma_opt = bk_analysis.get("bk_master_sigma_opt")
        bk_master_xi_opt = bk_analysis.get("bk_master_xi_opt")
        bk_master_h_opt = bk_analysis.get("bk_master_h_opt")

        if sigma_reference is not None and xi_reference is not None:
            print(f"BK reference point (sigma, xi): ({float(sigma_reference):.6f}, {float(xi_reference):.6f})")
        if bk_master_sigma_opt is not None and bk_master_xi_opt is not None:
            print(
                "BK master-map optimum (sigma, xi): "
                f"({float(bk_master_sigma_opt):.6f}, {float(bk_master_xi_opt):.6f})"
            )
        if bk_master_h_opt is not None:
            print(f"BK master-map optimum factor: {float(bk_master_h_opt):.6f}")


def _build_crystal_inputs_payload(context: CrystalContext) -> dict[str, Any]:
    return {
        "cavity_output_path": context.cavity_output_path,
        "crystal_length_m": context.crystal_length_m,
        "wavelength_m": context.wavelength_m,
        "n_crystal": context.n_crystal,
        "beam_waist_crystal_m": context.beam_waist_crystal_m,
    }


def _build_mode_matching_payload(mode_matching: ModeMatchingResult) -> dict[str, float]:
    return {
        "focusing_parameter_xi": float(mode_matching.focusing_parameter_xi),
        "boyd_kleinman_factor": float(mode_matching.boyd_kleinman_factor),
        "effective_nonlinear_overlap": float(mode_matching.effective_nonlinear_overlap),
    }


def _build_mode_matching_debug_payload(mode_matching: ModeMatchingResult) -> dict[str, float]:
    return asdict(mode_matching)


def _build_mode_matching_active_payload(mode_matching: ModeMatchingResult) -> dict[str, float]:
    return {
        "waist_crystal_m": float(mode_matching.waist_crystal_m),
        "rayleigh_range_m": float(mode_matching.rayleigh_range_m),
        "confocal_parameter_m": float(mode_matching.confocal_parameter_m),
        "focusing_parameter_xi": float(mode_matching.focusing_parameter_xi),
        "boyd_kleinman_factor": float(mode_matching.boyd_kleinman_factor),
        "effective_nonlinear_overlap": float(mode_matching.effective_nonlinear_overlap),
    }


def _strip_bk_map_payload(value: Any, store_bk_map: bool) -> Any:
    if isinstance(value, dict):
        return {
            key: _strip_bk_map_payload(item, store_bk_map)
            for key, item in value.items()
            if store_bk_map or key not in _BK_MAP_KEYS
        }
    if isinstance(value, list):
        return [_strip_bk_map_payload(item, store_bk_map) for item in value]
    if isinstance(value, tuple):
        return tuple(_strip_bk_map_payload(item, store_bk_map) for item in value)
    return value


def _build_bk_summary_payload(
    bk_analysis: dict[str, Any],
    *,
    store_bk_map: bool,
) -> dict[str, Any]:
    reference = bk_analysis.get("reference", {})
    payload = {
        "store_bk_map": bool(store_bk_map),
    }
    if reference:
        payload["reference"] = {
            key: reference[key]
            for key in (
                "reference_kind",
                "T_opt_K",
                "crystal_length_m",
                "xi_reference",
                "sigma_reference",
                "bk_reference_factor",
            )
            if key in reference
        }
    for key in ("bk_master_sigma_opt", "bk_master_xi_opt", "bk_master_h_opt"):
        if key in bk_analysis:
            payload[key] = bk_analysis[key]
    if store_bk_map:
        operating = bk_analysis.get("bk_analysis_operating", {})
        for key in _BK_MAP_KEYS:
            if key in operating:
                payload[key] = operating[key]
    return payload


def _build_bk_debug_payload(
    bk_analysis: dict[str, Any],
    *,
    store_bk_map: bool,
) -> dict[str, Any]:
    payload = _build_bk_summary_payload(bk_analysis, store_bk_map=store_bk_map)
    for key in ("bk_analysis_operating", "bk_analysis_optimal"):
        if key in bk_analysis:
            payload[key] = _strip_bk_map_payload(bk_analysis[key], store_bk_map=store_bk_map)
    return payload


def _build_active_for_opo_payload(
    result: CrystalSimulationResult,
    *,
    store_bk_map: bool,
) -> dict[str, Any] | None:
    if result.selected_operating_point_mode is None:
        return None

    cavity_crystal_length_m = float(result.context.crystal_length_m)
    selected_operating_point = result.selected_operating_point or {}
    active_crystal_length_m = float(selected_operating_point.get("crystal_length_m", cavity_crystal_length_m))
    active_for_opo = {
        "operating_point_mode": result.selected_operating_point_mode,
        "temperature_K": float(selected_operating_point.get("temperature_K", np.nan)),
        "crystal_length_m": active_crystal_length_m,
        "refractive_indices": (
            {
                "n_p": float(np.asarray(result.selected_operating_phase_matching["n_p"], dtype=float).ravel()[0]),
                "n_s": float(np.asarray(result.selected_operating_phase_matching["n_s"], dtype=float).ravel()[0]),
                "n_i": float(np.asarray(result.selected_operating_phase_matching["n_i"], dtype=float).ravel()[0]),
            }
            if result.selected_operating_phase_matching is not None
            and all(key in result.selected_operating_phase_matching for key in ("n_p", "n_s", "n_i"))
            else None
        ),
        "mode_matching": {
            "waist_crystal_m": float(result.mode_matching.waist_crystal_m),
            "effective_nonlinear_overlap": float(result.mode_matching.effective_nonlinear_overlap),
        },
        "polarization_resonance": (
            {
                key: result.active_polarization_resonance[key]
                for key in (
                    "fsr_signal_Hz",
                    "fsr_idler_Hz",
                    "delta_fsr_Hz",
                    "delta_phi_wrapped_rad",
                    "is_double_resonant",
                    "signal_optical_roundtrip_length_m",
                    "idler_optical_roundtrip_length_m",
                )
                if result.active_polarization_resonance is not None and key in result.active_polarization_resonance
            }
            if result.active_polarization_resonance is not None
            else None
        ),
        "recommended_cavity_crystal_length_m": active_crystal_length_m,
    }
    return {key: value for key, value in active_for_opo.items() if value is not None}


def _build_crystal_results_payload(
    result: CrystalSimulationResult,
    *,
    store_bk_map: bool,
) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    active_for_opo = _build_active_for_opo_payload(result, store_bk_map=store_bk_map)
    if active_for_opo is not None:
        payload["active_for_opo"] = active_for_opo
    return payload


def _build_double_resonance_scan_payload(
    scan: dict[str, Any],
    include_full_matrices: bool = False,
) -> dict[str, Any]:
    payload = {
        "temperature_grid_K": scan["temperature_grid_K"],
        "crystal_length_grid_m": scan["crystal_length_grid_m"],
        "resonance_tolerance_rad": float(scan["resonance_tolerance_rad"]),
        "best_temperature_K": float(scan["best_temperature_K"]),
        "best_crystal_length_m": float(scan["best_crystal_length_m"]),
        "best_delta_phi_wrapped_rad": float(scan["best_delta_phi_wrapped_rad"]),
        "best_abs_delta_phi_wrapped_rad": float(scan["best_abs_delta_phi_wrapped_rad"]),
        "best_is_double_resonant": bool(scan["best_is_double_resonant"]),
    }
    if include_full_matrices:
        payload["delta_phi_wrapped_rad"] = scan["delta_phi_wrapped_rad"]
        payload["abs_delta_phi_wrapped_rad"] = scan["abs_delta_phi_wrapped_rad"]
        payload["is_double_resonant"] = scan["is_double_resonant"]
    return payload


def _build_crystal_debug_payload(
    result: CrystalSimulationResult,
    *,
    save_full_double_resonance_scan: bool = False,
    store_bk_map: bool = False,
) -> dict[str, Any]:
    payload = {}
    payload["mode_matching"] = _build_mode_matching_debug_payload(result.mode_matching)
    if result.selected_operating_phase_matching is not None:
        payload["active_phase_matching"] = result.selected_operating_phase_matching
    operating_points = {}
    if result.phase_matching_operating_point is not None:
        operating_points["phase_matching_operating_point"] = result.phase_matching_operating_point
    if result.double_resonance_operating_point is not None:
        operating_points["double_resonance_operating_point"] = result.double_resonance_operating_point
    if operating_points:
        payload["operating_points"] = operating_points
    if result.polarization_resonance is not None:
        payload["phase_matching_polarization_resonance"] = result.polarization_resonance
    if result.phase_matching is not None:
        payload["phase_matching_scan"] = result.phase_matching
    if result.double_resonance_scan is not None:
        payload["double_resonance_scan"] = _build_double_resonance_scan_payload(
            result.double_resonance_scan,
            include_full_matrices=save_full_double_resonance_scan,
        )
    if result.bk_analysis is not None:
        payload["boyd_kleinman_analysis"] = _build_bk_debug_payload(
            result.bk_analysis,
            store_bk_map=store_bk_map,
        )
    return payload


def build_crystal_simulation_output(
    result: CrystalSimulationResult,
    save_full_double_resonance_scan: bool = False,
    debug: bool = False,
    store_bk_map: bool = False,
) -> dict[str, Any]:
    """Build JSON-serializable crystal simulation output."""
    simulation_inputs = result.simulation_inputs or {}
    inputs = {}
    if "d_eff_pm_per_V" in simulation_inputs:
        inputs["d_eff_pm_per_V"] = simulation_inputs["d_eff_pm_per_V"]
    output = {
        "inputs": inputs,
        "results": _build_crystal_results_payload(
            result,
            store_bk_map=store_bk_map,
        ),
    }
    if debug:
        debug_payload = _build_crystal_debug_payload(
            result,
            save_full_double_resonance_scan=save_full_double_resonance_scan,
            store_bk_map=store_bk_map,
        )
        if debug_payload:
            output["debug_data"] = debug_payload
    return output


def save_crystal_outputs(
    geometry: str,
    output: dict[str, Any],
    fig_bk_master=None,
    fig_qpm=None,
    fig_bk=None,
    results_root: str | Path | None = None,
    fig_bk_optimal=None,
    fig_double_resonance_scan=None,
) -> dict[str, str]:
    """Save crystal JSON and plots under ``results/<geometry>/crystal/``."""
    ensure_geometry_results_subdirs(geometry, results_root=results_root)
    result_dir = get_crystal_results_dir(geometry, results_root=results_root)
    project_root = Path(__file__).resolve().parents[2]

    json_path = result_dir / "crystal_simulation_output.json"
    bk_master_path = result_dir / "boyd_kleinman_master_map.png"
    qpm_path = result_dir / "qpm_length_poling_map.png"
    bk_path = result_dir / "boyd_kleinman_analysis.png"
    bk_optimal_path = result_dir / "boyd_kleinman_analysis_optimal.png"
    double_resonance_scan_path = result_dir / "double_resonance_scan.png"
    old_phase_path = result_dir / "phase_matching_scan.png"
    old_mode_path = result_dir / "mode_matching_summary.png"

    def _repo_relative(path: Path) -> str:
        try:
            return str(path.resolve().relative_to(project_root))
        except ValueError:
            return str(path)

    with json_path.open("w", encoding="utf-8") as f:
        json.dump(_to_json_compatible(output), f, indent=2)

    # Remove legacy plot files so the crystal results directory reflects the
    # current single-figure BK workflow after a fresh save.
    for legacy_path in (old_phase_path, old_mode_path):
        if legacy_path.exists():
            legacy_path.unlink()

    if fig_bk_master is not None:
        fig_bk_master.savefig(bk_master_path, dpi=300, bbox_inches="tight")
    if fig_qpm is not None:
        fig_qpm.savefig(qpm_path, dpi=300, bbox_inches="tight")
    if fig_bk is not None:
        fig_bk.savefig(bk_path, dpi=300, bbox_inches="tight")
    if fig_bk_optimal is not None:
        fig_bk_optimal.savefig(bk_optimal_path, dpi=300, bbox_inches="tight")
    if fig_double_resonance_scan is not None:
        fig_double_resonance_scan.savefig(double_resonance_scan_path, dpi=300, bbox_inches="tight")

    outputs = {
        "result_dir": _repo_relative(result_dir),
        "crystal_output_json": _repo_relative(json_path),
    }
    if fig_bk_master is not None:
        outputs["boyd_kleinman_master_map_png"] = _repo_relative(bk_master_path)
    if fig_qpm is not None:
        outputs["qpm_length_poling_map_png"] = _repo_relative(qpm_path)
    if fig_bk is not None:
        bk_relpath = _repo_relative(bk_path)
        outputs["boyd_kleinman_analysis_png"] = bk_relpath
        outputs["phase_matching_scan_png"] = bk_relpath
        outputs["mode_matching_summary_png"] = bk_relpath
    if fig_bk_optimal is not None:
        outputs["boyd_kleinman_analysis_optimal_png"] = _repo_relative(bk_optimal_path)
    if fig_double_resonance_scan is not None:
        outputs["double_resonance_scan_png"] = _repo_relative(double_resonance_scan_path)
    return outputs


__all__ = [
    "CrystalContext",
    "CrystalSimulationResult",
    "load_cavity_context_for_crystal",
    "compute_crystal_phase_matching",
    "compute_crystal_phase_matching_at_temperature",
    "compute_design_poling_period",
    "compute_crystal_mode_matching",
    "compute_boyd_kleinman_analysis",
    "build_phase_matching_operating_point",
    "build_double_resonance_operating_point",
    "select_crystal_operating_point",
    "setup_crystal_simulation",
    "compute_crystal_phase_matching_block",
    "compute_crystal_operating_point_block",
    "compute_crystal_physics_block",
    "build_crystal_simulation_result_from_blocks",
    "run_crystal_simulation",
    "build_crystal_simulation_result",
    "build_crystal_simulation_output",
    "print_crystal_summary",
    "save_crystal_outputs",
]
