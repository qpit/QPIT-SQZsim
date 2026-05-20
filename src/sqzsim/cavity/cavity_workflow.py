"""High-level workflow assembly for cavity simulations."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import numpy as np

try:
    from common.constants import TWO_PI
    from common.results_paths import ensure_geometry_results_subdirs, get_cavity_results_dir
except ImportError:
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from common.constants import TWO_PI
    from common.results_paths import ensure_geometry_results_subdirs, get_cavity_results_dir

from cavity_analysis import (
    beam_waist_from_q,
    compute_decay_rates,
    fsr_from_roundtrip_length,
    gouy_phases_from_m_factor,
    make_m_factor_estimator,
    make_q_estimator,
    optical_roundtrip_length,
    resolve_resonant_loss_model,
)
from cavity_plotter import print_geometry_ascii


_GEOMETRY_ERROR = "GEOMETRY must be 'bowtie', 'linear', 'triangle', 'hemilithic', or 'monolithic'"


@dataclass(frozen=True)
class GeometryEstimators:
    """Container for geometry-dependent estimators and plotting meshes."""

    estimate_m_factor_s: Any
    estimate_m_factor_t: Any
    estimate_q_sagittal: Any
    estimate_q_tangential: Any
    mesh_x: np.ndarray | None
    mesh_y: np.ndarray | None


@dataclass(frozen=True)
class CavityContext:
    """Structured cavity inputs and geometry-dependent estimators."""

    geometry: str
    parameters: dict[str, Any]
    estimators: GeometryEstimators


@dataclass(frozen=True)
class CavityOperatingPoint:
    """Single cavity operating point evaluated for one geometry."""

    qs: complex
    qt: complex
    m_factor: dict[str, float]
    geometry_values: dict[str, float]
    cavity_length: float
    optical_crystal_length: float


@dataclass(frozen=True)
class CavitySimulationResult:
    """Combined cavity workflow output."""

    context: CavityContext
    operating_point: CavityOperatingPoint
    derived_quantities: dict[str, float]


def _validate_geometry(geometry: str) -> None:
    if geometry not in {"bowtie", "linear", "triangle", "hemilithic", "monolithic"}:
        raise ValueError(_GEOMETRY_ERROR)


def _validate_radius_value(value: Any, name: str) -> float:
    radius = float(value)
    if radius <= 0.0 or (not np.isfinite(radius) and not np.isinf(radius)):
        raise ValueError(f"{name} must be positive or np.inf. Received {value}.")
    return radius


def resolve_radius_pair(parameters: dict[str, Any], point_parameters: dict[str, Any] | None = None) -> tuple[float, float]:
    """Resolve new two-radius inputs with legacy single-radius fallback."""
    point_parameters = {} if point_parameters is None else point_parameters
    # legacy fallback – do not use in new configurations
    legacy_parameter = parameters.get("roc_m")
    legacy_point = point_parameters.get("ROC_M")
    parameter_1 = parameters.get("roc_1_m", legacy_parameter)
    parameter_2 = parameters.get("roc_2_m", legacy_parameter)

    radius_1 = point_parameters.get("ROC_1_M", legacy_point if legacy_point is not None else parameter_1)
    radius_2 = point_parameters.get("ROC_2_M", legacy_point if legacy_point is not None else parameter_2)
    return (
        _validate_radius_value(radius_1, "ROC_1_M/roc_1_m"),
        _validate_radius_value(radius_2, "ROC_2_M/roc_2_m"),
    )


def normalize_cavity_parameters(parameters: dict[str, Any]) -> dict[str, Any]:
    """Return parameters with explicit radius-pair keys populated."""
    normalized = dict(parameters)
    if not any(key in normalized for key in ("roc_m", "roc_1_m", "roc_2_m")):
        return normalized
    radius_1, radius_2 = resolve_radius_pair(normalized)
    normalized["roc_1_m"] = radius_1
    normalized["roc_2_m"] = radius_2
    return normalized


def print_geometry_info(geometry: str) -> None:
    """Print geometry parameter definitions and an ASCII sketch."""
    geometry_parameter_lines = {
        "bowtie": (
            "Bow-tie geometry parameters:",
            " (0) crystal length [mm]",
            " (1) crystal refractive index",
            " (2) short axis [mm]",
            " (3) long axis [mm]",
            " (4) mirror radius 1 [mm]",
            " (5) mirror radius 2 [mm]",
            " (6) AOI [deg]",
        ),
        "linear": (
            "Linear geometry parameters:",
            " (0) crystal length [mm]",
            " (1) crystal refractive index",
            " (2) cavity length [mm]",
            " (3) mirror radius 1 [mm]",
            " (4) mirror radius 2 [mm]",
        ),
        "triangle": (
            "Triangle geometry parameters:",
            " (0) crystal length [mm]",
            " (1) crystal refractive index",
            " (2) triangle width [mm]",
            " (3) triangle height [mm]",
            " (4) mirror radius 1 [mm]",
            " (5) mirror radius 2 [mm]",
        ),
        "hemilithic": (
            "Hemilithic geometry parameters:",
            " (0) crystal length [mm]",
            " (1) crystal refractive index",
            " (2) air gap [mm]",
            " (3) external mirror radius [mm]",
            " (4) crystal surface radius [mm]",
        ),
        "monolithic": (
            "Monolithic geometry parameters:",
            " (0) crystal length [mm]",
            " (1) crystal refractive index",
            " (2) facet 1 radius of curvature [mm]",
            " (3) facet 2 radius of curvature [mm]",
        ),
    }
    _validate_geometry(geometry)
    for line in geometry_parameter_lines[geometry]:
        print(line)
    print_geometry_ascii(geometry)


def build_geometry_estimators(geometry: str, parameters: dict[str, Any]) -> GeometryEstimators:
    """Build geometry-dependent m-factor/q estimators and plot meshes."""
    _validate_geometry(geometry)
    parameters = normalize_cavity_parameters(parameters)

    if geometry == "bowtie":
        return GeometryEstimators(
            estimate_m_factor_s=make_m_factor_estimator("bowtie", plane="sagittal"),
            estimate_m_factor_t=make_m_factor_estimator("bowtie", plane="tangential"),
            estimate_q_sagittal=make_q_estimator("bowtie", plane="sagittal"),
            estimate_q_tangential=make_q_estimator("bowtie", plane="tangential"),
            mesh_x=parameters["mesh_short_axis_m"],
            mesh_y=parameters["mesh_long_axis_m"],
        )

    if geometry == "triangle":
        return GeometryEstimators(
            estimate_m_factor_s=make_m_factor_estimator("triangle", plane="sagittal"),
            estimate_m_factor_t=make_m_factor_estimator("triangle", plane="tangential"),
            estimate_q_sagittal=make_q_estimator("triangle", plane="sagittal"),
            estimate_q_tangential=make_q_estimator("triangle", plane="tangential"),
            mesh_x=parameters["mesh_triangle_width_m"],
            mesh_y=parameters["mesh_triangle_height_m"],
        )

    if geometry == "monolithic":
        estimator = make_m_factor_estimator("monolithic")
        q_estimator = make_q_estimator("monolithic")
        return GeometryEstimators(
            estimate_m_factor_s=estimator,
            estimate_m_factor_t=estimator,
            estimate_q_sagittal=q_estimator,
            estimate_q_tangential=q_estimator,
            mesh_x=None,
            mesh_y=None,
        )

    m_estimator = make_m_factor_estimator(geometry)
    q_estimator = make_q_estimator(geometry)
    return GeometryEstimators(
        estimate_m_factor_s=m_estimator,
        estimate_m_factor_t=m_estimator,
        estimate_q_sagittal=q_estimator,
        estimate_q_tangential=q_estimator,
        mesh_x=None,
        mesh_y=None,
    )


def build_cavity_context(
    geometry: str,
    parameters: dict[str, Any],
    estimators: GeometryEstimators | None = None,
) -> CavityContext:
    """Build the structured cavity workflow context."""
    if estimators is None:
        estimators = build_geometry_estimators(geometry, parameters)
    parameters = normalize_cavity_parameters(parameters)
    return CavityContext(geometry=geometry, parameters=parameters, estimators=estimators)


def _evaluate_bowtie_operating_point(context: CavityContext, point_parameters: dict[str, Any]) -> CavityOperatingPoint:
    parameters = context.parameters
    estimators = context.estimators

    short_axis_val = point_parameters["BOWTIE_SHORT_AXIS_M"]
    long_axis_val = point_parameters["BOWTIE_LONG_AXIS_M"]
    theta = point_parameters.get("BOWTIE_THETA_AOI_RAD", parameters["theta_aoi_rad"])
    radius_1_val, radius_2_val = resolve_radius_pair(parameters, point_parameters)
    diagonal_val = float((long_axis_val + short_axis_val) / 2 * np.cos(theta))

    m_x = float(
        estimators.estimate_m_factor_s(
            long_axis_val,
            short_axis_val,
            theta,
            parameters["crystal_length_m"],
            radius_1_val,
            radius_2_val,
            parameters["n_crystal"],
        )
    )
    m_y = float(
        estimators.estimate_m_factor_t(
            long_axis_val,
            short_axis_val,
            theta,
            parameters["crystal_length_m"],
            radius_1_val,
            radius_2_val,
            parameters["n_crystal"],
        )
    )
    qs = complex(
        estimators.estimate_q_sagittal(
            long_axis_val,
            short_axis_val,
            theta,
            parameters["crystal_length_m"],
            radius_1_val,
            radius_2_val,
            parameters["n_crystal"],
        )
    )
    qt = complex(
        estimators.estimate_q_tangential(
            long_axis_val,
            short_axis_val,
            theta,
            parameters["crystal_length_m"],
            radius_1_val,
            radius_2_val,
            parameters["n_crystal"],
        )
    )

    return CavityOperatingPoint(
        qs=qs,
        qt=qt,
        m_factor={"sagittal": m_x, "tangential": m_y},
        geometry_values={
            "short_axis_m": float(short_axis_val),
            "long_axis_m": float(long_axis_val),
            "diagonal_m": diagonal_val,
            "theta_AOI_rad": float(theta),
            "theta_AOI_deg": float(np.degrees(theta)),
            "RoC_1_m": float(radius_1_val),
            "RoC_2_m": float(radius_2_val),
        },
        cavity_length=float(long_axis_val + short_axis_val + 2 * diagonal_val),
        optical_crystal_length=float(parameters["crystal_length_m"]),
    )


def _evaluate_linear_operating_point(context: CavityContext, point_parameters: dict[str, Any]) -> CavityOperatingPoint:
    parameters = context.parameters
    estimators = context.estimators

    cavity_length_val = point_parameters.get("LINEAR_CAVITY_LENGTH_M", parameters["l_cav_m"])
    radius_1_val, radius_2_val = resolve_radius_pair(parameters, point_parameters)
    m_val = float(
        estimators.estimate_m_factor_s(
            radius_1_val,
            radius_2_val,
            cavity_length_val,
            parameters["crystal_length_m"],
            parameters["n_crystal"],
        )
    )
    qs = complex(
        estimators.estimate_q_sagittal(
            radius_1_val,
            radius_2_val,
            cavity_length_val,
            parameters["crystal_length_m"],
            parameters["n_crystal"],
        )
    )

    return CavityOperatingPoint(
        qs=qs,
        qt=qs,
        m_factor={"sagittal": m_val, "tangential": m_val},
        geometry_values={
            "L_cav_m": float(cavity_length_val),
            "RoC_1_m": float(radius_1_val),
            "RoC_2_m": float(radius_2_val),
        },
        cavity_length=float(2 * cavity_length_val),
        optical_crystal_length=float(parameters["crystal_length_m"]),
    )


def _evaluate_triangle_operating_point(context: CavityContext, point_parameters: dict[str, Any]) -> CavityOperatingPoint:
    parameters = context.parameters
    estimators = context.estimators

    width_val = point_parameters["TRIANGLE_WIDTH_M"]
    height_val = point_parameters["TRIANGLE_HEIGHT_M"]
    radius_1_val, radius_2_val = resolve_radius_pair(parameters, point_parameters)
    diagonal_val = float(np.sqrt((width_val / 2) ** 2 + height_val**2))
    theta_half_val = float(np.arcsin(height_val / diagonal_val) / 2)

    m_x = float(
        estimators.estimate_m_factor_s(
            width_val,
            height_val,
            parameters["crystal_length_m"],
            radius_1_val,
            radius_2_val,
            parameters["n_crystal"],
        )
    )
    m_y = float(
        estimators.estimate_m_factor_t(
            width_val,
            height_val,
            parameters["crystal_length_m"],
            radius_1_val,
            radius_2_val,
            parameters["n_crystal"],
        )
    )
    qs = complex(
        estimators.estimate_q_sagittal(
            width_val,
            height_val,
            parameters["crystal_length_m"],
            radius_1_val,
            radius_2_val,
            parameters["n_crystal"],
        )
    )
    qt = complex(
        estimators.estimate_q_tangential(
            width_val,
            height_val,
            parameters["crystal_length_m"],
            radius_1_val,
            radius_2_val,
            parameters["n_crystal"],
        )
    )

    return CavityOperatingPoint(
        qs=qs,
        qt=qt,
        m_factor={"sagittal": m_x, "tangential": m_y},
        geometry_values={
            "triangle_width_m": float(width_val),
            "triangle_height_m": float(height_val),
            "triangle_diagonal_m": diagonal_val,
            "mirror_aoi_half_angle_rad": theta_half_val,
            "mirror_aoi_half_angle_deg": float(np.degrees(theta_half_val)),
            "RoC_1_m": float(radius_1_val),
            "RoC_2_m": float(radius_2_val),
        },
        cavity_length=float(width_val + 2 * diagonal_val),
        optical_crystal_length=float(parameters["crystal_length_m"]),
    )


def _evaluate_hemilithic_operating_point(
    context: CavityContext,
    point_parameters: dict[str, Any],
) -> CavityOperatingPoint:
    parameters = context.parameters
    estimators = context.estimators

    air_gap_val = point_parameters.get("HEMILITHIC_AIR_GAP_M", parameters["l_air_m"])
    radius_1_val, radius_2_val = resolve_radius_pair(parameters, point_parameters)
    m_val = float(
        estimators.estimate_m_factor_s(
            radius_1_val,
            radius_2_val,
            air_gap_val,
            parameters["crystal_length_m"],
            parameters["n_crystal"],
        )
    )
    qs = complex(
        estimators.estimate_q_sagittal(
            radius_1_val,
            radius_2_val,
            air_gap_val,
            parameters["crystal_length_m"],
            parameters["n_crystal"],
        )
    )

    return CavityOperatingPoint(
        qs=qs,
        qt=qs,
        m_factor={"sagittal": m_val, "tangential": m_val},
        geometry_values={
            "L_air_m": float(air_gap_val),
            "RoC_1_m": float(radius_1_val),
            "RoC_2_m": float(radius_2_val),
        },
        cavity_length=float(2 * (air_gap_val + parameters["crystal_length_m"])),
        optical_crystal_length=float(2 * parameters["crystal_length_m"]),
    )


def _evaluate_monolithic_operating_point(
    context: CavityContext,
    point_parameters: dict[str, Any],
) -> CavityOperatingPoint:
    parameters = context.parameters
    estimators = context.estimators

    crystal_length_val = point_parameters.get("MONOLITHIC_CRYSTAL_LENGTH_M", parameters["crystal_length_m"])
    radius_1_val, radius_2_val = resolve_radius_pair(parameters, point_parameters)
    m_val = float(
        estimators.estimate_m_factor_s(
            radius_1_val,
            radius_2_val,
            crystal_length_val,
            parameters["n_crystal"],
        )
    )
    qs = complex(
        estimators.estimate_q_sagittal(
            radius_1_val,
            radius_2_val,
            crystal_length_val,
            parameters["n_crystal"],
        )
    )

    return CavityOperatingPoint(
        qs=qs,
        qt=qs,
        m_factor={"sagittal": m_val, "tangential": m_val},
        geometry_values={
            "crystal_length_m": float(crystal_length_val),
            "RoC_1_m": float(radius_1_val),
            "RoC_2_m": float(radius_2_val),
        },
        cavity_length=float(2 * crystal_length_val),
        optical_crystal_length=float(2 * crystal_length_val),
    )


def compute_cavity_operating_point(context: CavityContext, point_parameters: dict[str, Any]) -> CavityOperatingPoint:
    """Evaluate one representative cavity operating point for the selected geometry."""
    geometry = context.geometry
    _validate_geometry(geometry)

    if geometry == "bowtie":
        return _evaluate_bowtie_operating_point(context, point_parameters)
    if geometry == "linear":
        return _evaluate_linear_operating_point(context, point_parameters)
    if geometry == "triangle":
        return _evaluate_triangle_operating_point(context, point_parameters)
    if geometry == "hemilithic":
        return _evaluate_hemilithic_operating_point(context, point_parameters)
    return _evaluate_monolithic_operating_point(context, point_parameters)


def evaluate_single_point(
    geometry: str,
    parameters: dict[str, Any],
    estimators: GeometryEstimators,
    point_parameters: dict[str, Any],
) -> dict[str, Any]:
    """Backward-compatible wrapper returning the evaluated point as a dictionary."""
    operating_point = compute_cavity_operating_point(
        build_cavity_context(geometry, parameters, estimators=estimators),
        point_parameters,
    )
    return _cavity_operating_point_to_dict(operating_point)


def compute_cavity_derived_quantities(
    context: CavityContext,
    operating_point: CavityOperatingPoint,
    c_m_per_s: float,
    detuning_Hz: float,
    loss_model_parameters: dict[str, Any] | None = None,
    T_ext: float | None = None,
    L_rt: float | None = None,
) -> dict[str, float]:
    """Compute derived cavity figures from one explicit resonant loss model."""
    beam_waist_crystal_um = (
        beam_waist_from_q(
            operating_point.qs,
            context.parameters["wavelength_m"],
            refractive_index=context.parameters["n_crystal"],
        )
        * 1e6
    )
    optical_roundtrip_length_m = optical_roundtrip_length(
        operating_point.cavity_length,
        operating_point.optical_crystal_length,
        context.parameters["n_crystal"],
    )
    effective_loss_parameters = dict(context.parameters if loss_model_parameters is None else loss_model_parameters)
    if T_ext is not None or L_rt is not None:
        if any(
            key in effective_loss_parameters
            for key in ("r1_resonant", "r2_resonant", "alpha_resonant_per_m", "l_parasitic_rt")
        ):
            raise ValueError("Do not mix explicit loss_model_parameters with legacy T_ext/L_rt inputs.")
        if T_ext is not None:
            effective_loss_parameters["t_ext"] = float(T_ext)
        if L_rt is not None:
            effective_loss_parameters["l_rt"] = float(L_rt)

    # ``alpha_resonant_per_m`` is defined as distributed loss in the resonant
    # crystal medium, so it should use the crystal round-trip propagation
    # length rather than the full cavity path.
    loss_model = resolve_resonant_loss_model(
        effective_loss_parameters,
        roundtrip_propagation_length_m=operating_point.optical_crystal_length,
    )
    decay = compute_decay_rates(
        optical_roundtrip_length_m,
        c_m_per_s,
        output_coupling_transmission=float(loss_model["output_coupling_transmission"]),
        internal_roundtrip_loss=float(loss_model["internal_roundtrip_loss"]),
    )
    gouy = gouy_phases_from_m_factor(context.geometry, operating_point.m_factor)

    return {
        "beam_waist_crystal_um": float(beam_waist_crystal_um),
        "cavity_length_m": float(operating_point.cavity_length),
        "optical_crystal_length_m": float(operating_point.optical_crystal_length),
        "optical_roundtrip_length_m": float(optical_roundtrip_length_m),
        "fsr_Hz": float(fsr_from_roundtrip_length(optical_roundtrip_length_m, c_m_per_s)),
        "loss_model_source": str(loss_model["loss_model_source"]),
        "roundtrip_propagation_length_m": float(loss_model["roundtrip_propagation_length_m"]),
        "reflectivity_input_resonant": float(loss_model["reflectivity_input_resonant"]),
        "reflectivity_output_resonant": float(loss_model["reflectivity_output_resonant"]),
        "input_coupler_transmission": float(loss_model["input_coupler_transmission"]),
        "output_coupling_transmission": float(loss_model["output_coupling_transmission"]),
        "alpha_resonant_per_m": float(loss_model["alpha_resonant_per_m"]),
        "bulk_roundtrip_loss": float(loss_model["bulk_roundtrip_loss"]),
        "parasitic_roundtrip_loss": float(loss_model["parasitic_roundtrip_loss"]),
        "internal_roundtrip_loss": float(loss_model["internal_roundtrip_loss"]),
        "kappa_ext_rad_s": decay["kappa_ext_rad_s"],
        "kappa_ext_Hz": decay["kappa_ext_Hz"],
        "kappa_loss_rad_s": decay["kappa_loss_rad_s"],
        "kappa_loss_Hz": decay["kappa_loss_Hz"],
        "kappa_int_rad_s": decay["kappa_int_rad_s"],
        "kappa_int_Hz": decay["kappa_int_Hz"],
        "kappa_total_rad_s": decay["kappa_total_rad_s"],
        "kappa_total_Hz": decay["kappa_total_Hz"],
        "escape_efficiency": decay["escape_efficiency"],
        "detuning_rad_s": float(TWO_PI * detuning_Hz),
        "gouy_phase_sagittal_rad": gouy["gouy_phase_sagittal_rad"],
        "gouy_phase_tangential_rad": gouy["gouy_phase_tangential_rad"],
    }


def compute_derived_cavity_quantities(
    geometry: str,
    qs: complex,
    qt: complex,
    m_factor_dict: dict[str, float],
    cavity_length: float,
    optical_crystal_length: float,
    wavelength: float,
    n_crystal: float,
    c_m_per_s: float,
    detuning_Hz: float,
    T_ext: float | None = None,
    L_rt: float | None = None,
    loss_model_parameters: dict[str, Any] | None = None,
) -> dict[str, float]:
    """Backward-compatible wrapper for derived cavity figures."""
    context = build_cavity_context(
        geometry,
        {
            "wavelength_m": wavelength,
            "n_crystal": n_crystal,
        },
        estimators=GeometryEstimators(None, None, None, None, None, None),
    )
    operating_point = CavityOperatingPoint(
        qs=qs,
        qt=qt,
        m_factor=m_factor_dict,
        geometry_values={},
        cavity_length=cavity_length,
        optical_crystal_length=optical_crystal_length,
    )
    return compute_cavity_derived_quantities(
        context,
        operating_point,
        c_m_per_s=c_m_per_s,
        detuning_Hz=detuning_Hz,
        T_ext=T_ext,
        L_rt=L_rt,
        loss_model_parameters=loss_model_parameters,
    )


def _print_instability_warning(m_factor: dict[str, float]) -> None:
    unstable_planes = [
        plane
        for plane, m_value in m_factor.items()
        if (not np.isfinite(m_value)) or abs(m_value) >= 1.0
    ]
    if unstable_planes:
        plane_list = ", ".join(unstable_planes)
        print(f"WARNING: Selected point is in an unstable region")


def _format_radius_mm(radius_m: float) -> str:
    if np.isinf(radius_m):
        return "inf"
    return f"{radius_m * 1e3:.3f} mm"


def print_single_point_summary(geometry: str, result: CavityOperatingPoint | dict[str, Any]) -> None:
    """Print a concise single-point summary for the selected geometry."""
    operating_point = _coerce_cavity_operating_point(result)
    qs = operating_point.qs
    qt = operating_point.qt
    m = operating_point.m_factor
    g = operating_point.geometry_values

    if geometry == "bowtie":
        print(
            f"Geometrical parameters: long axis = {g['long_axis_m']*1e3:.3f} mm, short axis = {g['short_axis_m']*1e3:.3f} mm, "
            f"diagonal = {g['diagonal_m']*1e3:.3f} mm, AOI = {g['theta_AOI_deg']:.3f} deg, "
            f"RoC1 = {_format_radius_mm(g['RoC_1_m'])}, RoC2 = {_format_radius_mm(g['RoC_2_m'])}"
        )
        print(f"m_sagittal = {m['sagittal']:.6f}, m_tangential = {m['tangential']:.6f}")
        _print_instability_warning(m)
        print(f"qs parameter in the crystal: {qs:.5f}")
        print(f"qt parameter in the crystal: {qt:.5f}")
        return

    if geometry == "linear":
        print(
            f"Linear cavity parameters: L_cav = {g['L_cav_m']*1e3:.3f} mm, "
            f"RoC1 = {_format_radius_mm(g['RoC_1_m'])}, RoC2 = {_format_radius_mm(g['RoC_2_m'])}"
        )
        print(f"m = {m['sagittal']:.6f} (stable if |m|<1)")
        _print_instability_warning(m)
        print(f"q parameter at crystal center: {qs:.5f}")
        return

    if geometry == "triangle":
        print(
            f"Triangle cavity parameters: width = {g['triangle_width_m']*1e3:.3f} mm, "
            f"height = {g['triangle_height_m']*1e3:.3f} mm, diagonal = {g['triangle_diagonal_m']*1e3:.3f} mm, "
            f"RoC1 = {_format_radius_mm(g['RoC_1_m'])}, RoC2 = {_format_radius_mm(g['RoC_2_m'])}"
        )
        print(f"m_sagittal = {m['sagittal']:.6f}, m_tangential = {m['tangential']:.6f}")
        _print_instability_warning(m)
        print(f"qs parameter in the crystal: {qs:.5f}")
        print(f"qt parameter in the crystal: {qt:.5f}")
        return

    if geometry == "hemilithic":
        print(
            f"Hemilithic cavity parameters: L_air = {g['L_air_m']*1e3:.3f} mm, "
            f"external RoC = {_format_radius_mm(g['RoC_1_m'])}, crystal RoC = {_format_radius_mm(g['RoC_2_m'])}"
        )
        print(f"m = {m['sagittal']:.6f} (stable if |m|<1)")
        _print_instability_warning(m)
        print(f"q parameter at crystal input face: {qs:.5f}")
        return

    if geometry == "monolithic":
        print(
            f"Monolithic cavity parameters: crystal length = {g['crystal_length_m']*1e3:.3f} mm, "
            f"facet1 RoC = {_format_radius_mm(g['RoC_1_m'])}, facet2 RoC = {_format_radius_mm(g['RoC_2_m'])}"
        )
        print(f"m = {m['sagittal']:.6f} (stable if |m|<1)")
        _print_instability_warning(m)
        print(f"q parameter at planar coated facet: {qs:.5f}")
        return

    raise ValueError(_GEOMETRY_ERROR)


def print_derived_cavity_quantities(results: dict[str, float]) -> None:
    """Print cavity-derived quantities using an aligned spec-style table."""

    def _print_specs(title: str, rows: list[tuple[str, str]]) -> None:
        print(f"\n{title}")
        print("-" * len(title))
        for label, value in rows:
            print(f"{label:<32}: {value}")

    detuning_hz = results["detuning_rad_s"] / TWO_PI
    _print_specs(
        "Derived cavity figures",
        [
            ("Beam waist in crystal", f"{results['beam_waist_crystal_um']:.3f} um"),
            ("Geometric cavity length", f"{results['cavity_length_m']:.6f} m"),
            ("Optical round-trip length", f"{results['optical_roundtrip_length_m']:.6f} m"),
            (
                "Reflectivity (input/output)",
                f"{results['reflectivity_input_resonant']:.6f} / {results['reflectivity_output_resonant']:.6f}",
            ),
            (
                "Transmission (input/output)",
                f"{results['input_coupler_transmission']:.6e} / {results['output_coupling_transmission']:.6e}",
            ),
            ("Distributed loss coeff.", f"{results['alpha_resonant_per_m']:.6e} 1/m"),
            ("Bulk round-trip loss", f"{results['bulk_roundtrip_loss']:.6e}"),
            ("Parasitic round-trip loss", f"{results['parasitic_roundtrip_loss']:.6e}"),
            ("Internal round-trip loss", f"{results['internal_roundtrip_loss']:.6e}"),
            ("FSR", f"{results['fsr_Hz']:.6f} Hz ({results['fsr_Hz']/1e6:.6f} MHz)"),
            (
                "kappa_ext",
                f"{results['kappa_ext_rad_s']:.3e} rad/s (kappa_ext/2pi = {results['kappa_ext_Hz']:.3e} Hz)",
            ),
            (
                "kappa_loss",
                f"{results['kappa_loss_rad_s']:.3e} rad/s (kappa_loss/2pi = {results['kappa_loss_Hz']:.3e} Hz)",
            ),
            (
                "kappa_total",
                f"{results['kappa_total_rad_s']:.3e} rad/s (kappa/2pi = {results['kappa_total_Hz']:.3e} Hz)",
            ),
            ("Escape efficiency", f"{results['escape_efficiency']:.4f}"),
            ("Detuning", f"{results['detuning_rad_s']:.3e} rad/s (Delta/2pi = {detuning_hz:.3e} Hz)"),
            ("Gouy phase sagittal", f"{results['gouy_phase_sagittal_rad']:.6f} rad"),
            ("Gouy phase tangential", f"{results['gouy_phase_tangential_rad']:.6f} rad"),
        ],
    )


def build_geometry_inputs_for_export(
    geometry: str,
    result: CavityOperatingPoint | dict[str, Any],
) -> tuple[dict[str, float], dict[str, float]]:
    """Build geometry-specific export fields for JSON output."""
    _validate_geometry(geometry)
    operating_point = _coerce_cavity_operating_point(result)
    return operating_point.geometry_values, operating_point.m_factor


def build_cavity_simulation_result(
    context: CavityContext,
    operating_point: CavityOperatingPoint,
    derived_quantities: dict[str, float],
) -> CavitySimulationResult:
    """Build the structured cavity workflow result."""
    return CavitySimulationResult(
        context=context,
        operating_point=operating_point,
        derived_quantities=derived_quantities,
    )


def build_cavity_simulation_output(
    result: CavitySimulationResult,
    c_m_per_s: float,
) -> dict[str, Any]:
    """Build JSON-serializable cavity simulation output."""
    parameters = result.context.parameters
    geometry_inputs, m_factor_export = build_geometry_inputs_for_export(result.context.geometry, result.operating_point)
    derived = result.derived_quantities

    output = {
        "inputs": {
            "geometry": result.context.geometry,
            "c_m_per_s": float(c_m_per_s),
            "crystal_length_m": float(parameters["crystal_length_m"]),
            "n_crystal": float(parameters["n_crystal"]),
            "RoC_1_m": float(result.operating_point.geometry_values.get("RoC_1_m", parameters["roc_1_m"])),
            "RoC_2_m": float(result.operating_point.geometry_values.get("RoC_2_m", parameters["roc_2_m"])),
            "wavelength_m": float(parameters["wavelength_m"]),
            "reflectivity_input_resonant": float(derived["reflectivity_input_resonant"]),
            "reflectivity_output_resonant": float(derived["reflectivity_output_resonant"]),
            "alpha_resonant_per_m": float(derived["alpha_resonant_per_m"]),
            "parasitic_roundtrip_loss": float(derived["parasitic_roundtrip_loss"]),
            "T_ext": float(derived["output_coupling_transmission"]),
            "L_rt": float(derived["internal_roundtrip_loss"]),
            "detuning_Hz": float(parameters["detuning_hz"]),
            "geometry_specific": geometry_inputs,
        },
        "results": {
            "beam_waist_crystal_um": float(derived["beam_waist_crystal_um"]),
            "cavity_length_m": float(derived["cavity_length_m"]),
            "optical_crystal_length_m": float(derived["optical_crystal_length_m"]),
            "optical_roundtrip_length_m": float(derived["optical_roundtrip_length_m"]),
            "fsr_Hz": float(derived["fsr_Hz"]),
            "loss_model_source": str(derived["loss_model_source"]),
            "roundtrip_propagation_length_m": float(derived["roundtrip_propagation_length_m"]),
            "reflectivity_input_resonant": float(derived["reflectivity_input_resonant"]),
            "reflectivity_output_resonant": float(derived["reflectivity_output_resonant"]),
            "input_coupler_transmission": float(derived["input_coupler_transmission"]),
            "output_coupling_transmission": float(derived["output_coupling_transmission"]),
            "bulk_roundtrip_loss": float(derived["bulk_roundtrip_loss"]),
            "parasitic_roundtrip_loss": float(derived["parasitic_roundtrip_loss"]),
            "internal_roundtrip_loss": float(derived["internal_roundtrip_loss"]),
            "kappa_ext_rad_s": float(derived["kappa_ext_rad_s"]),
            "kappa_ext_Hz": float(derived["kappa_ext_Hz"]),
            "kappa_loss_rad_s": float(derived["kappa_loss_rad_s"]),
            "kappa_loss_Hz": float(derived["kappa_loss_Hz"]),
            "kappa_int_rad_s": float(derived["kappa_int_rad_s"]),
            "kappa_int_Hz": float(derived["kappa_int_Hz"]),
            "kappa_total_rad_s": float(derived["kappa_total_rad_s"]),
            "kappa_total_Hz": float(derived["kappa_total_Hz"]),
            "escape_efficiency": float(derived["escape_efficiency"]),
            "detuning_rad_s": float(derived["detuning_rad_s"]),
            "gouy_phase_sagittal_rad": float(derived["gouy_phase_sagittal_rad"]),
            "gouy_phase_tangential_rad": float(derived["gouy_phase_tangential_rad"]),
        },
        "debug_data": {
            "q_sagittal": {
                "real": float(np.real(result.operating_point.qs)),
                "imag": float(np.imag(result.operating_point.qs)),
            },
            "q_tangential": {
                "real": float(np.real(result.operating_point.qt)),
                "imag": float(np.imag(result.operating_point.qt)),
            },
            "m_factor": m_factor_export,
            "radius_of_curvature": {
                "RoC_1_m": float(result.operating_point.geometry_values.get("RoC_1_m", parameters["roc_1_m"])),
                "RoC_2_m": float(result.operating_point.geometry_values.get("RoC_2_m", parameters["roc_2_m"])),
            },
        },
    }
    return output


def save_cavity_outputs(
    geometry: str,
    output: dict[str, Any],
    fig_stability,
    fig_waist,
    results_root: str | Path | None = None,
) -> dict[str, str]:
    """Save cavity JSON and plots under ``results/<geometry>/cavity/``."""
    ensure_geometry_results_subdirs(geometry, results_root=results_root)
    result_dir = get_cavity_results_dir(geometry, results_root=results_root)
    project_root = Path(__file__).resolve().parents[2]

    json_path = result_dir / "cavity_simulation_output.json"
    stability_path = result_dir / "stability_map.png"
    waist_path = result_dir / "waist_map.png"

    def _repo_relative(path: Path) -> str:
        try:
            return str(path.resolve().relative_to(project_root))
        except ValueError:
            return str(path)

    outputs_info = {
        "result_dir": _repo_relative(result_dir),
        "cavity_output_json": _repo_relative(json_path),
        "stability_map_png": _repo_relative(stability_path),
        "waist_map_png": _repo_relative(waist_path),
    }

    with json_path.open("w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)

    if fig_stability is not None:
        fig_stability.savefig(stability_path, dpi=300, bbox_inches="tight")
    if fig_waist is not None:
        fig_waist.savefig(waist_path, dpi=300, bbox_inches="tight")

    return outputs_info


def _coerce_cavity_operating_point(result: CavityOperatingPoint | dict[str, Any]) -> CavityOperatingPoint:
    if isinstance(result, CavityOperatingPoint):
        return result
    return CavityOperatingPoint(
        qs=result["qs"],
        qt=result["qt"],
        m_factor=result["m_factor"],
        geometry_values=result["geometry_values"],
        cavity_length=result["cavity_length"],
        optical_crystal_length=result["optical_crystal_length"],
    )


def _cavity_operating_point_to_dict(result: CavityOperatingPoint) -> dict[str, Any]:
    return {
        "qs": result.qs,
        "qt": result.qt,
        "m_factor": result.m_factor,
        "geometry_values": result.geometry_values,
        "cavity_length": result.cavity_length,
        "optical_crystal_length": result.optical_crystal_length,
    }


__all__ = [
    "GeometryEstimators",
    "CavityContext",
    "CavityOperatingPoint",
    "CavitySimulationResult",
    "print_geometry_info",
    "resolve_radius_pair",
    "normalize_cavity_parameters",
    "build_geometry_estimators",
    "build_cavity_context",
    "compute_cavity_operating_point",
    "evaluate_single_point",
    "compute_cavity_derived_quantities",
    "compute_derived_cavity_quantities",
    "print_single_point_summary",
    "print_derived_cavity_quantities",
    "build_geometry_inputs_for_export",
    "build_cavity_simulation_result",
    "build_cavity_simulation_output",
    "save_cavity_outputs",
]
