"""OPO model definitions and first-principles threshold helpers.

This module builds the below-threshold OPO operating point from the simulated
cavity and crystal outputs. The oscillation threshold is computed directly from
physical cavity and nonlinear-optical parameters; no empirical threshold scale
is used.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

try:
    from ..common.constants import C_M_PER_S, EPSILON_0_F_PER_M, HBAR_J_S, PI, TWO_PI
except ImportError:
    from common.constants import C_M_PER_S, EPSILON_0_F_PER_M, HBAR_J_S, PI, TWO_PI

CRYSTAL_LENGTH_CONSISTENCY_TOLERANCE_M = 1e-9


@dataclass(frozen=True)
class OPOParameters:
    """User-facing OPO parameters for one workflow run."""

    pump_mode: str
    pump_power_W: float | None
    pump_parameter_sigma: float | None
    pump_percent_threshold: float | None
    pump_resonance_model: str
    pump_input_coupling_efficiency: float | None
    signal_wavelength_m: float
    idler_wavelength_m: float
    pump_wavelength_m: float
    analysis_sideband_Hz: float
    analysis_span_Hz: tuple[float, float]
    n_analysis_points: int
    detection_efficiency: float
    lo_phase_rad: float = 0.0


@dataclass(frozen=True)
class OPOModelResult:
    """Compact collection of derived OPO quantities used downstream."""

    pump_parameter: float
    effective_threshold_power_W: float
    threshold_external_pump_power_W: float
    threshold_intracavity_pump_photon_number: float
    threshold_intracavity_pump_energy_J: float
    threshold_intracavity_pump_power_scale_W: float
    pump_resonance_model: str
    pump_buildup_factor: float
    pump_conversion_assumption: str
    pump_mode: str
    pump_power_W: float
    threshold_model: str
    threshold_nonlinear_coupling: float
    threshold_mode_area_m2: float
    threshold_overlap: float
    threshold_refractive_indices: dict[str, float]
    threshold_signal_decay_rate_rad_s: float
    threshold_idler_decay_rate_rad_s: float
    threshold_pump_decay_rate_rad_s: float | None
    pump_interaction_time_s: float | None
    pump_input_coupling_efficiency: float | None
    d_eff_pm_per_V: float
    crystal_length_m: float
    cavity_crystal_length_m: float
    effective_mode_area_m2: float
    nonlinear_overlap: float
    effective_nonlinear_coupling: float
    cavity_kappa_ext_Hz: float
    cavity_kappa_loss_Hz: float
    cavity_kappa_total_Hz: float
    d_eff_source: str
    crystal_operating_point_mode: str
    crystal_active_temperature_K: float
    below_threshold: bool
    escape_efficiency: float
    cavity_detuning_Hz: float
    signal_wavelength_m: float
    idler_wavelength_m: float
    pump_wavelength_m: float
    notes: tuple[str, ...]


def _as_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _require_positive(value: Any, name: str) -> float:
    if value is None:
        raise ValueError(f"Missing required OPO threshold input: {name}.")
    numeric = float(value)
    if not np.isfinite(numeric) or numeric <= 0.0:
        raise ValueError(f"OPO threshold input {name} must be positive and finite; received {value!r}.")
    return numeric


def _first_scalar(value: Any, name: str) -> float:
    if isinstance(value, (list, tuple, np.ndarray)):
        array = np.asarray(value, dtype=float).ravel()
        if array.size == 0:
            raise ValueError(f"Missing required OPO threshold input: {name}.")
        return _require_positive(array[0], name)
    return _require_positive(value, name)


def _resolve_active_crystal_payload(
    crystal_results: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    active_for_opo = _as_dict(crystal_results.get("active_for_opo"))
    if active_for_opo:
        return (
            active_for_opo,
            _as_dict(active_for_opo.get("phase_matching")),
            _as_dict(active_for_opo.get("mode_matching")),
            _as_dict(active_for_opo.get("boyd_kleinman_analysis")),
            _as_dict(active_for_opo.get("polarization_resonance")),
        )

    selected_operating_point = _as_dict(crystal_results.get("selected_operating_point"))
    legacy_active = {
        "operating_point_mode": crystal_results.get("selected_operating_point_mode", "unknown"),
        "temperature_K": selected_operating_point.get("temperature_K"),
        "crystal_length_m": selected_operating_point.get("crystal_length_m"),
    }
    return (
        legacy_active,
        _as_dict(crystal_results.get("selected_operating_phase_matching", crystal_results.get("phase_matching", {}))),
        _as_dict(crystal_results.get("mode_matching")),
        _as_dict(crystal_results.get("boyd_kleinman_analysis")),
        _as_dict(crystal_results.get("polarization_resonance")),
    )


def _validate_crystal_length_consistency(
    cavity_inputs: dict[str, Any],
    active_for_opo: dict[str, Any],
    tolerance_m: float = CRYSTAL_LENGTH_CONSISTENCY_TOLERANCE_M,
) -> tuple[float, float]:
    cavity_crystal_length_raw = cavity_inputs.get("crystal_length_m")
    active_crystal_length_raw = active_for_opo.get("crystal_length_m")

    cavity_crystal_length_m = float(cavity_crystal_length_raw) if cavity_crystal_length_raw is not None else np.nan
    active_crystal_length_m = float(active_crystal_length_raw) if active_crystal_length_raw is not None else np.nan

    if (
        np.isfinite(cavity_crystal_length_m)
        and np.isfinite(active_crystal_length_m)
        and abs(active_crystal_length_m - cavity_crystal_length_m) > float(tolerance_m)
    ):
        raise ValueError(
            "Selected crystal operating point is inconsistent with loaded cavity geometry: "
            f"active crystal length {active_crystal_length_m:.12g} m vs cavity crystal length "
            f"{cavity_crystal_length_m:.12g} m. Rerun cavity_main with the selected crystal length before running OPO."
        )

    return cavity_crystal_length_m, active_crystal_length_m


def _resolve_refractive_indices(active_for_opo: dict[str, Any], phase_matching: dict[str, Any]) -> dict[str, float]:
    refractive_indices = _as_dict(active_for_opo.get("refractive_indices"))
    n_p_raw = refractive_indices.get("n_p", phase_matching.get("n_p"))
    n_s_raw = refractive_indices.get("n_s", phase_matching.get("n_s"))
    n_i_raw = refractive_indices.get("n_i", phase_matching.get("n_i"))
    try:
        return {
            "n_p": _first_scalar(n_p_raw, "crystal refractive index n_p"),
            "n_s": _first_scalar(n_s_raw, "crystal refractive index n_s"),
            "n_i": _first_scalar(n_i_raw, "crystal refractive index n_i"),
        }
    except ValueError as exc:
        raise ValueError(
            "Crystal output is missing refractive indices required for the physical OPO threshold. "
            "Rerun crystal_main.py for the same GEOMETRY so results.<geometry>.crystal."
            "crystal_simulation_output.json includes results.active_for_opo.refractive_indices."
        ) from exc


def _resolve_decay_rate_rad_s(cavity_results: dict[str, Any], key_hz: str, key_rad_s: str) -> float | None:
    if key_rad_s in cavity_results:
        return _require_positive(cavity_results[key_rad_s], key_rad_s)
    if key_hz in cavity_results:
        return TWO_PI * _require_positive(cavity_results[key_hz], key_hz)
    return None


def _resolve_pump_resonance_model(value: Any) -> str:
    model = str(value or "single_pass").strip().lower()
    if model not in {"single_pass", "resonant"}:
        raise ValueError(
            "pump_resonance_model must be 'single_pass' or 'resonant'; "
            f"received {value!r}."
        )
    return model


def _resolve_pump_mode(value: Any) -> str:
    mode = str(value or "absolute").strip().lower()
    if mode not in {"fraction", "absolute"}:
        raise ValueError(f"pump_mode must be 'fraction' or 'absolute'; received {value!r}.")
    return mode


def _resolve_pump_operating_point(parameters: OPOParameters, threshold_power_W: float) -> tuple[float, float]:
    threshold_power_W = _require_positive(threshold_power_W, "threshold_external_pump_power_W")
    if parameters.pump_mode == "fraction":
        if parameters.pump_power_W is not None:
            raise ValueError("pump_power_W must not be provided when pump_mode='fraction'.")
        if parameters.pump_parameter_sigma is not None:
            pump_parameter = _require_positive(parameters.pump_parameter_sigma, "pump_parameter_sigma")
        elif parameters.pump_percent_threshold is not None:
            pump_parameter = _require_positive(parameters.pump_percent_threshold, "pump_percent_threshold") / 100.0
        else:
            raise ValueError(
                "pump_mode='fraction' requires pump_parameter_sigma or pump_percent_threshold."
            )
        pump_power_W = pump_parameter**2 * threshold_power_W
    else:
        if parameters.pump_parameter_sigma is not None or parameters.pump_percent_threshold is not None:
            raise ValueError(
                "pump_parameter_sigma and pump_percent_threshold must not be provided when pump_mode='absolute'."
            )
        pump_power_W = _require_positive(parameters.pump_power_W, "pump_power_W")
        pump_parameter = float(np.sqrt(pump_power_W / threshold_power_W))

    return float(pump_power_W), float(pump_parameter)


def compute_physical_opo_threshold_power(
    *,
    kappa_total_rad_s: float,
    signal_decay_rate_rad_s: float | None,
    idler_decay_rate_rad_s: float | None,
    pump_decay_rate_rad_s: float | None,
    pump_resonance_model: str,
    pump_input_coupling_efficiency: float | None,
    pump_wavelength_m: float,
    signal_wavelength_m: float,
    idler_wavelength_m: float,
    d_eff_m_per_V: float,
    crystal_length_m: float,
    effective_mode_area_m2: float,
    nonlinear_overlap: float,
    n_p: float,
    n_s: float,
    n_i: float,
) -> dict[str, Any]:
    """Compute the OPO threshold from the three-wave quantum coupling rate.

    The signal/idler threshold condition is
    ``g * sqrt(n_pump) = sqrt(k_s k_i) / 2``. The resulting pump photon number
    is then converted to external pump power using either a single-pass transit
    time through the crystal or an explicitly specified resonant pump coupling.
    """
    kappa_total_rad_s = _require_positive(kappa_total_rad_s, "kappa_total_rad_s")
    kappa_s_rad_s = _require_positive(signal_decay_rate_rad_s or kappa_total_rad_s, "signal_decay_rate_rad_s")
    kappa_i_rad_s = _require_positive(idler_decay_rate_rad_s or kappa_total_rad_s, "idler_decay_rate_rad_s")
    pump_resonance_model = _resolve_pump_resonance_model(pump_resonance_model)

    pump_wavelength_m = _require_positive(pump_wavelength_m, "pump_wavelength_m")
    signal_wavelength_m = _require_positive(signal_wavelength_m, "signal_wavelength_m")
    idler_wavelength_m = _require_positive(idler_wavelength_m, "idler_wavelength_m")
    d_eff_m_per_V = _require_positive(d_eff_m_per_V, "d_eff_m_per_V")
    crystal_length_m = _require_positive(crystal_length_m, "crystal_length_m")
    effective_mode_area_m2 = _require_positive(effective_mode_area_m2, "effective_mode_area_m2")
    nonlinear_overlap = _require_positive(nonlinear_overlap, "nonlinear_overlap")
    n_p = _require_positive(n_p, "n_p")
    n_s = _require_positive(n_s, "n_s")
    n_i = _require_positive(n_i, "n_i")

    omega_p_rad_s = TWO_PI * C_M_PER_S / pump_wavelength_m
    omega_s_rad_s = TWO_PI * C_M_PER_S / signal_wavelength_m
    omega_i_rad_s = TWO_PI * C_M_PER_S / idler_wavelength_m

    nonlinear_coupling_rad_s = (
        d_eff_m_per_V
        * nonlinear_overlap
        * np.sqrt(
            HBAR_J_S
            * omega_p_rad_s
            * omega_s_rad_s
            * omega_i_rad_s
            / (2.0 * EPSILON_0_F_PER_M * n_p * n_s * n_i * effective_mode_area_m2 * crystal_length_m)
        )
    )
    nonlinear_coupling_rad_s = _require_positive(nonlinear_coupling_rad_s, "threshold_nonlinear_coupling")

    threshold_pump_photons = kappa_s_rad_s * kappa_i_rad_s / (4.0 * nonlinear_coupling_rad_s**2)
    threshold_pump_energy_J = HBAR_J_S * omega_p_rad_s * threshold_pump_photons
    pump_interaction_time_s = n_p * crystal_length_m / C_M_PER_S
    single_pass_equivalent_power_W = threshold_pump_energy_J / pump_interaction_time_s

    kappa_p_rad_s: float | None = None
    pump_input_efficiency: float | None = None
    if pump_resonance_model == "single_pass":
        if pump_decay_rate_rad_s is not None:
            raise ValueError(
                "single_pass pump threshold must not use a pump cavity linewidth. "
                "Remove kappa_pump_Hz/kappa_pump_rad_s or set pump_resonance_model='resonant'."
            )
        external_threshold_power_W = single_pass_equivalent_power_W
        pump_buildup_factor = 1.0
        pump_conversion_assumption = (
            "single_pass: external pump power is converted from the threshold pump energy "
            "using the pump transit time n_p * L / c through the crystal."
        )
    else:
        kappa_p_rad_s = _require_positive(pump_decay_rate_rad_s, "pump_decay_rate_rad_s")
        pump_input_efficiency = _require_positive(
            pump_input_coupling_efficiency,
            "pump_input_coupling_efficiency",
        )
        if pump_input_efficiency > 1.0:
            raise ValueError("pump_input_coupling_efficiency must be <= 1 for resonant pump threshold.")
        external_threshold_power_W = threshold_pump_energy_J * kappa_p_rad_s / pump_input_efficiency
        pump_buildup_factor = single_pass_equivalent_power_W / external_threshold_power_W
        pump_conversion_assumption = (
            "resonant: external pump power uses hbar * omega_p * kappa_p * n_pump / eta_pump_in."
        )

    external_threshold_power_W = _require_positive(
        external_threshold_power_W,
        "threshold_external_pump_power_W",
    )

    return {
        "effective_threshold_power_W": float(external_threshold_power_W),
        "threshold_external_pump_power_W": float(external_threshold_power_W),
        "threshold_intracavity_pump_photon_number": float(threshold_pump_photons),
        "threshold_intracavity_pump_energy_J": float(threshold_pump_energy_J),
        "threshold_intracavity_pump_power_scale_W": float(single_pass_equivalent_power_W),
        "threshold_nonlinear_coupling": float(nonlinear_coupling_rad_s),
        "threshold_signal_decay_rate_rad_s": float(kappa_s_rad_s),
        "threshold_idler_decay_rate_rad_s": float(kappa_i_rad_s),
        "threshold_pump_decay_rate_rad_s": float(kappa_p_rad_s) if kappa_p_rad_s is not None else None,
        "pump_interaction_time_s": float(pump_interaction_time_s) if pump_interaction_time_s is not None else None,
        "pump_input_coupling_efficiency": (
            float(pump_input_efficiency) if pump_input_efficiency is not None else None
        ),
        "pump_resonance_model": pump_resonance_model,
        "pump_buildup_factor": float(pump_buildup_factor),
        "pump_conversion_assumption": pump_conversion_assumption,
        "threshold_refractive_indices": {"n_p": float(n_p), "n_s": float(n_s), "n_i": float(n_i)},
    }


def build_opo_parameters(config: dict[str, Any]) -> OPOParameters:
    """Build a validated OPO parameter object from a plain configuration mapping."""
    pump_mode = _resolve_pump_mode(config.get("pump_mode", "absolute"))
    pump_power_W = (
        _require_positive(config.get("pump_power_W"), "pump_power_W")
        if config.get("pump_power_W") is not None
        else None
    )
    pump_parameter_sigma = (
        _require_positive(config.get("pump_parameter_sigma"), "pump_parameter_sigma")
        if config.get("pump_parameter_sigma") is not None
        else None
    )
    pump_percent_threshold = (
        _require_positive(config.get("pump_percent_threshold"), "pump_percent_threshold")
        if config.get("pump_percent_threshold") is not None
        else None
    )
    if pump_mode == "fraction" and pump_power_W is not None:
        raise ValueError("pump_power_W must not be provided when pump_mode='fraction'.")
    if pump_mode == "absolute" and pump_power_W is None:
        raise ValueError("pump_mode='absolute' requires pump_power_W.")
    if pump_mode == "absolute" and (pump_parameter_sigma is not None or pump_percent_threshold is not None):
        raise ValueError("Fractional pump inputs must not be provided when pump_mode='absolute'.")

    signal_wavelength_m = config.get("wavelength_s_m", config.get("signal_wavelength_m"))
    idler_wavelength_m = config.get("wavelength_i_m", signal_wavelength_m)
    pump_wavelength_m = config.get("wavelength_p_m", config.get("pump_wavelength_m"))
    if signal_wavelength_m is None:
        raise KeyError("Missing OPO signal wavelength: expected 'wavelength_s_m'.")
    if idler_wavelength_m is None:
        raise KeyError("Missing OPO idler wavelength: expected 'wavelength_i_m'.")
    if pump_wavelength_m is None:
        raise KeyError("Missing OPO pump wavelength: expected 'wavelength_p_m'.")

    return OPOParameters(
        pump_mode=pump_mode,
        pump_power_W=pump_power_W,
        pump_parameter_sigma=pump_parameter_sigma,
        pump_percent_threshold=pump_percent_threshold,
        pump_resonance_model=_resolve_pump_resonance_model(config.get("pump_resonance_model", "single_pass")),
        pump_input_coupling_efficiency=(
            _require_positive(config.get("pump_input_coupling_efficiency"), "pump_input_coupling_efficiency")
            if config.get("pump_input_coupling_efficiency") is not None
            else None
        ),
        signal_wavelength_m=_require_positive(signal_wavelength_m, "wavelength_s_m"),
        idler_wavelength_m=_require_positive(idler_wavelength_m, "wavelength_i_m"),
        pump_wavelength_m=_require_positive(pump_wavelength_m, "wavelength_p_m"),
        analysis_sideband_Hz=float(config["analysis_sideband_Hz"]),
        analysis_span_Hz=tuple(float(v) for v in config["analysis_span_Hz"]),
        n_analysis_points=int(config["n_analysis_points"]),
        detection_efficiency=float(config["detection_efficiency"]),
        lo_phase_rad=float(config.get("lo_phase_rad", 0.0)),
    )


def derive_opo_quantities(
    parameters: OPOParameters,
    cavity_data: dict[str, Any],
    crystal_data: dict[str, Any],
) -> OPOModelResult:
    """Build a first-principles below-threshold OPO operating point."""
    cavity_results = cavity_data.get("results", {})
    crystal_results = crystal_data.get("results", {})
    active_for_opo, phase_matching, mode_matching, _bk_analysis, _polarization_resonance = _resolve_active_crystal_payload(
        crystal_results
    )
    crystal_inputs = crystal_data.get("inputs", {})
    cavity_inputs = cavity_data.get("inputs", {})
    cavity_crystal_length_m, _active_crystal_length_m = _validate_crystal_length_consistency(
        cavity_inputs,
        active_for_opo,
    )
    selected_operating_point_mode = str(
        active_for_opo.get(
            "operating_point_mode",
            crystal_results.get("selected_operating_point_mode", "unknown"),
        )
    )
    selected_operating_point = _as_dict(crystal_results.get("selected_operating_point", {}))
    crystal_active_temperature_raw = active_for_opo.get("temperature_K", selected_operating_point.get("temperature_K"))
    crystal_active_temperature_K = (
        float(crystal_active_temperature_raw) if crystal_active_temperature_raw is not None else np.nan
    )

    nonlinear_overlap = _require_positive(
        mode_matching.get("effective_nonlinear_overlap"),
        "active_for_opo.mode_matching.effective_nonlinear_overlap",
    )
    d_eff_pm_per_V = _require_positive(crystal_inputs.get("d_eff_pm_per_V"), "crystal inputs.d_eff_pm_per_V")
    d_eff_m_per_V = d_eff_pm_per_V * 1e-12

    crystal_length_m = _require_positive(
        active_for_opo.get("crystal_length_m", selected_operating_point.get("crystal_length_m")),
        "active_for_opo.crystal_length_m",
    )
    waist_crystal_m = _require_positive(
        mode_matching.get("waist_crystal_m"),
        "active_for_opo.mode_matching.waist_crystal_m",
    )
    effective_mode_area_m2 = _require_positive(PI * waist_crystal_m**2, "effective_mode_area_m2")
    refractive_indices = _resolve_refractive_indices(active_for_opo, phase_matching)

    kappa_ext_rad_s = _resolve_decay_rate_rad_s(cavity_results, "kappa_ext_Hz", "kappa_ext_rad_s")
    kappa_loss_rad_s = _resolve_decay_rate_rad_s(cavity_results, "kappa_loss_Hz", "kappa_loss_rad_s")
    kappa_total_rad_s = _resolve_decay_rate_rad_s(cavity_results, "kappa_total_Hz", "kappa_total_rad_s")
    if kappa_total_rad_s is None:
        if kappa_ext_rad_s is None or kappa_loss_rad_s is None:
            raise ValueError("Missing required OPO threshold input: cavity kappa_total_Hz/kappa_total_rad_s.")
        kappa_total_rad_s = kappa_ext_rad_s + kappa_loss_rad_s

    signal_decay_rate_rad_s = _resolve_decay_rate_rad_s(cavity_results, "kappa_signal_Hz", "kappa_signal_rad_s")
    idler_decay_rate_rad_s = _resolve_decay_rate_rad_s(cavity_results, "kappa_idler_Hz", "kappa_idler_rad_s")
    pump_decay_rate_rad_s = _resolve_decay_rate_rad_s(cavity_results, "kappa_pump_Hz", "kappa_pump_rad_s")

    threshold = compute_physical_opo_threshold_power(
        kappa_total_rad_s=kappa_total_rad_s,
        signal_decay_rate_rad_s=signal_decay_rate_rad_s,
        idler_decay_rate_rad_s=idler_decay_rate_rad_s,
        pump_decay_rate_rad_s=pump_decay_rate_rad_s,
        pump_resonance_model=parameters.pump_resonance_model,
        pump_input_coupling_efficiency=parameters.pump_input_coupling_efficiency,
        pump_wavelength_m=parameters.pump_wavelength_m,
        signal_wavelength_m=parameters.signal_wavelength_m,
        idler_wavelength_m=parameters.idler_wavelength_m,
        d_eff_m_per_V=d_eff_m_per_V,
        crystal_length_m=crystal_length_m,
        effective_mode_area_m2=effective_mode_area_m2,
        nonlinear_overlap=nonlinear_overlap,
        n_p=refractive_indices["n_p"],
        n_s=refractive_indices["n_s"],
        n_i=refractive_indices["n_i"],
    )
    effective_threshold_power_W = float(threshold["effective_threshold_power_W"])
    pump_power_W, pump_parameter = _resolve_pump_operating_point(parameters, effective_threshold_power_W)

    kappa_ext_Hz = float(kappa_ext_rad_s / TWO_PI) if kappa_ext_rad_s is not None else np.nan
    kappa_loss_Hz = float(kappa_loss_rad_s / TWO_PI) if kappa_loss_rad_s is not None else np.nan
    kappa_total_Hz = float(kappa_total_rad_s / TWO_PI)
    detuning_hz = float(cavity_inputs.get("detuning_Hz", 0.0))
    escape_efficiency = float(cavity_results.get("escape_efficiency", 0.0))

    notes = [
        "First-principles OPO threshold model.",
        "Threshold is computed from the three-wave quantum coupling, cavity decay rates, d_eff, length, mode area, overlap, and refractive indices.",
        "Signal and idler use the simulated total resonant-field decay rate when separate signal/idler decay rates are not exported.",
        str(threshold["pump_conversion_assumption"]),
        "Pump parameter uses sigma = sqrt(P_pump_external / P_threshold_external).",
        f"Pump mode: {parameters.pump_mode}.",
        f"Crystal operating point mode loaded from crystal workflow: {selected_operating_point_mode}.",
    ]

    return OPOModelResult(
        pump_parameter=float(pump_parameter),
        effective_threshold_power_W=effective_threshold_power_W,
        threshold_external_pump_power_W=float(threshold["threshold_external_pump_power_W"]),
        threshold_intracavity_pump_photon_number=float(threshold["threshold_intracavity_pump_photon_number"]),
        threshold_intracavity_pump_energy_J=float(threshold["threshold_intracavity_pump_energy_J"]),
        threshold_intracavity_pump_power_scale_W=float(threshold["threshold_intracavity_pump_power_scale_W"]),
        pump_resonance_model=str(threshold["pump_resonance_model"]),
        pump_buildup_factor=float(threshold["pump_buildup_factor"]),
        pump_conversion_assumption=str(threshold["pump_conversion_assumption"]),
        pump_mode=parameters.pump_mode,
        pump_power_W=float(pump_power_W),
        threshold_model="physical_first_principles",
        threshold_nonlinear_coupling=float(threshold["threshold_nonlinear_coupling"]),
        threshold_mode_area_m2=float(effective_mode_area_m2),
        threshold_overlap=float(nonlinear_overlap),
        threshold_refractive_indices=dict(threshold["threshold_refractive_indices"]),
        threshold_signal_decay_rate_rad_s=float(threshold["threshold_signal_decay_rate_rad_s"]),
        threshold_idler_decay_rate_rad_s=float(threshold["threshold_idler_decay_rate_rad_s"]),
        threshold_pump_decay_rate_rad_s=(
            float(threshold["threshold_pump_decay_rate_rad_s"])
            if threshold["threshold_pump_decay_rate_rad_s"] is not None
            else None
        ),
        pump_interaction_time_s=(
            float(threshold["pump_interaction_time_s"])
            if threshold["pump_interaction_time_s"] is not None
            else None
        ),
        pump_input_coupling_efficiency=(
            float(threshold["pump_input_coupling_efficiency"])
            if threshold["pump_input_coupling_efficiency"] is not None
            else None
        ),
        d_eff_pm_per_V=float(d_eff_pm_per_V),
        crystal_length_m=float(crystal_length_m),
        cavity_crystal_length_m=float(cavity_crystal_length_m),
        effective_mode_area_m2=float(effective_mode_area_m2),
        nonlinear_overlap=float(nonlinear_overlap),
        effective_nonlinear_coupling=float(threshold["threshold_nonlinear_coupling"]),
        cavity_kappa_ext_Hz=kappa_ext_Hz,
        cavity_kappa_loss_Hz=kappa_loss_Hz,
        cavity_kappa_total_Hz=kappa_total_Hz,
        d_eff_source="crystal_input_d_eff_pm_per_V",
        crystal_operating_point_mode=str(selected_operating_point_mode),
        crystal_active_temperature_K=float(crystal_active_temperature_K),
        below_threshold=bool(pump_parameter < 1.0),
        escape_efficiency=escape_efficiency,
        cavity_detuning_Hz=detuning_hz,
        signal_wavelength_m=float(parameters.signal_wavelength_m),
        idler_wavelength_m=float(parameters.idler_wavelength_m),
        pump_wavelength_m=float(parameters.pump_wavelength_m),
        notes=tuple(notes),
    )


__all__ = [
    "OPOParameters",
    "OPOModelResult",
    "build_opo_parameters",
    "compute_physical_opo_threshold_power",
    "derive_opo_quantities",
]
