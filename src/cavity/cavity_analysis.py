"""Cavity physics analysis utilities built on ABCD matrices."""

from __future__ import annotations

import numpy as np
import sympy as sp

from common.constants import PI, TWO_PI
from cavity_abcd import CavityAbcdBuilder
from cavity_abcd import Abcd
from cavity_abcd import radius_to_curvature


def cavity_stability(matrix):
    """Return cavity stability m-factor = (A + D) / 2."""
    A, _, _, D = Abcd.parameters(matrix)
    return (A + D) / 2


def cavity_q_parameter(matrix):
    """Return stable cavity q parameter from round-trip ABCD matrix."""
    A, B, C, D = Abcd.parameters(matrix)
    return -1 * (D - A) / (2 * C) + sp.I * sp.sqrt(1 - ((D + A) / 2) ** 2) / sp.Abs(C)


def beam_waist_from_q(q_parameter, wavelength, refractive_index=1):
    """Compute beam waist radius from q parameter imaginary part."""
    q_im = np.imag(q_parameter)
    return np.sqrt(wavelength * q_im / (refractive_index * PI))


def optical_roundtrip_length(cavity_length_m, optical_crystal_length_m, n_crystal):
    """Return optical round-trip cavity length in meters."""
    return float(cavity_length_m + (n_crystal - 1.0) * optical_crystal_length_m)


def fsr_from_roundtrip_length(L_optical_m, c_m_per_s):
    """Return free spectral range in Hz from optical round-trip length."""
    return float(c_m_per_s / L_optical_m)


def distributed_roundtrip_loss(alpha_per_m: float, roundtrip_propagation_length_m: float) -> float:
    """Return distributed round-trip power loss from a crystal-medium attenuation coefficient."""
    alpha = max(float(alpha_per_m), 0.0)
    length_m = max(float(roundtrip_propagation_length_m), 0.0)
    return float(1.0 - np.exp(-alpha * length_m))


def resolve_resonant_loss_model(parameters: dict, roundtrip_propagation_length_m: float) -> dict[str, float | str]:
    """Resolve one physically explicit resonant-loss model from new or legacy inputs.

    Input convention:
    - ``r1_resonant``: non-output resonant mirror/facet reflectivity
    - ``r2_resonant``: output-coupler reflectivity
    - ``alpha_resonant_per_m``: distributed resonant power-loss coefficient
    - ``l_parasitic_rt``: extra round-trip internal power loss

    Alternative compact convention:
    - ``t_ext``: output coupling transmission
    - ``l_rt``: internal round-trip power loss
    """
    uses_new = any(
        key in parameters
        for key in ("r1_resonant", "r2_resonant", "alpha_resonant_per_m", "l_parasitic_rt")
    )
    uses_legacy = any(key in parameters for key in ("t_ext", "l_rt"))

    if uses_new and uses_legacy:
        raise ValueError(
            "Mixed cavity loss conventions are not allowed. Use either the reflectivity-based "
            "inputs (r1_resonant/r2_resonant/alpha_resonant_per_m/l_parasitic_rt) or the compact "
            "inputs (t_ext/l_rt), but not both."
        )

    if uses_new:
        if "r1_resonant" not in parameters or "r2_resonant" not in parameters:
            raise ValueError("Reflectivity-based cavity loss model requires both r1_resonant and r2_resonant.")
        reflectivity_input = float(parameters["r1_resonant"])
        reflectivity_output = float(parameters["r2_resonant"])
        alpha_resonant_per_m = float(parameters.get("alpha_resonant_per_m", 0.0))
        parasitic_roundtrip_loss = float(parameters.get("l_parasitic_rt", 0.0))
        source = "reflectivity_based"
    else:
        output_coupling_transmission = float(parameters.get("t_ext", 0.0))
        internal_roundtrip_loss = float(parameters.get("l_rt", 0.0))
        reflectivity_input = 1.0
        reflectivity_output = 1.0 - output_coupling_transmission
        alpha_resonant_per_m = 0.0
        parasitic_roundtrip_loss = internal_roundtrip_loss
        source = "legacy_transmission_loss"

    for label, value in (
        ("r1_resonant", reflectivity_input),
        ("r2_resonant", reflectivity_output),
    ):
        if not (0.0 <= value <= 1.0):
            raise ValueError(f"{label} must satisfy 0 <= R <= 1. Received {value}.")
    for label, value in (
        ("alpha_resonant_per_m", alpha_resonant_per_m),
        ("l_parasitic_rt", parasitic_roundtrip_loss),
    ):
        if value < 0.0:
            raise ValueError(f"{label} must be non-negative. Received {value}.")

    input_coupler_transmission = 1.0 - reflectivity_input
    output_coupling_transmission = 1.0 - reflectivity_output
    bulk_roundtrip_loss = distributed_roundtrip_loss(alpha_resonant_per_m, roundtrip_propagation_length_m)
    internal_roundtrip_loss = input_coupler_transmission + bulk_roundtrip_loss + parasitic_roundtrip_loss

    if output_coupling_transmission >= 1.0:
        raise ValueError("Output coupling transmission must be less than 1.")
    if internal_roundtrip_loss >= 1.0:
        raise ValueError(
            "Internal round-trip loss must be less than 1. Check r1_resonant, alpha_resonant_per_m, and l_parasitic_rt."
        )

    return {
        "loss_model_source": source,
        "roundtrip_propagation_length_m": float(roundtrip_propagation_length_m),
        "reflectivity_input_resonant": float(reflectivity_input),
        "reflectivity_output_resonant": float(reflectivity_output),
        "input_coupler_transmission": float(input_coupler_transmission),
        "output_coupling_transmission": float(output_coupling_transmission),
        "alpha_resonant_per_m": float(alpha_resonant_per_m),
        "bulk_roundtrip_loss": float(bulk_roundtrip_loss),
        "parasitic_roundtrip_loss": float(parasitic_roundtrip_loss),
        "internal_roundtrip_loss": float(internal_roundtrip_loss),
    }


def compute_decay_rates(
    L_optical_m: float,
    c_m_per_s: float,
    output_coupling_transmission: float,
    internal_roundtrip_loss: float,
):
    """Compute cavity decay rates.

    ``kappa_*_rad_s`` are angular decay rates in rad/s.
    ``kappa_*_Hz`` are the corresponding cycle-per-second rates.
    """
    roundtrip_frequency_hz = float(c_m_per_s / L_optical_m)
    kappa_ext_hz_like = 0.5 * roundtrip_frequency_hz * float(output_coupling_transmission)
    kappa_loss_hz_like = 0.5 * roundtrip_frequency_hz * float(internal_roundtrip_loss)
    kappa_ext = TWO_PI * kappa_ext_hz_like
    kappa_loss = TWO_PI * kappa_loss_hz_like
    kappa_total = kappa_ext + kappa_loss
    return {
        "kappa_ext_rad_s": float(kappa_ext),
        "kappa_ext_Hz": float(kappa_ext_hz_like),
        "kappa_loss_rad_s": float(kappa_loss),
        "kappa_loss_Hz": float(kappa_loss_hz_like),
        "kappa_int_rad_s": float(kappa_loss),
        "kappa_int_Hz": float(kappa_loss_hz_like),
        "kappa_total_rad_s": float(kappa_total),
        "kappa_total_Hz": float(kappa_ext_hz_like + kappa_loss_hz_like),
        "escape_efficiency": float(kappa_ext / kappa_total) if kappa_total != 0 else np.nan,
    }


def gouy_phases_from_m_factor(geometry, m_factor_dict):
    """Return sagittal/tangential Gouy phases in radians from m-factors."""
    psi_sagittal = np.arccos(m_factor_dict["sagittal"])
    if geometry in ("bowtie", "triangle"):
        psi_tangential = np.arccos(m_factor_dict["tangential"])
    else:
        psi_tangential = psi_sagittal
    return {
        "gouy_phase_sagittal_rad": float(psi_sagittal),
        "gouy_phase_tangential_rad": float(psi_tangential),
    }


def _radius_pair(radius_1, radius_2=None):
    """Return a backward-compatible radius pair."""
    return radius_1, radius_1 if radius_2 is None else radius_2


def _curvature_pair(radius_1, radius_2):
    """Convert a radius pair to curvatures for estimator internals."""
    return radius_to_curvature(radius_1), radius_to_curvature(radius_2)


def bowtie_m_factor(long_axis, short_axis, incidence_angle, crystal_length, radius_1, refractive_index, radius_2=None, plane="sagittal"):
    """Return bow-tie cavity m-factor for a selected plane."""
    radius_1, radius_2 = _radius_pair(radius_1, radius_2)
    matrix = CavityAbcdBuilder.bowtie_roundtrip(
        long_axis,
        short_axis,
        crystal_length,
        radius_1,
        radius_2,
        refractive_index,
        incidence_angle,
        plane=plane,
    )
    return cavity_stability(matrix)


def bowtie_q_parameter(long_axis, short_axis, incidence_angle, crystal_length, radius_1, refractive_index, radius_2=None, plane="sagittal"):
    """Return bow-tie cavity q parameter for a selected plane."""
    radius_1, radius_2 = _radius_pair(radius_1, radius_2)
    matrix = CavityAbcdBuilder.bowtie_roundtrip(
        long_axis,
        short_axis,
        crystal_length,
        radius_1,
        radius_2,
        refractive_index,
        incidence_angle,
        plane=plane,
    )
    return cavity_q_parameter(matrix)


def linear_m_factor(radius_1, cavity_length, crystal_length, refractive_index, radius_2=None):
    """Return m-factor for a linear cavity."""
    radius_1, radius_2 = _radius_pair(radius_1, radius_2)
    matrix = CavityAbcdBuilder.linear_roundtrip(
        cavity_length,
        crystal_length,
        radius_1,
        radius_2,
        refractive_index,
    )
    return cavity_stability(matrix)


def linear_q_parameter(radius_1, cavity_length, crystal_length, refractive_index, radius_2=None):
    """Return q parameter for a linear cavity."""
    radius_1, radius_2 = _radius_pair(radius_1, radius_2)
    matrix = CavityAbcdBuilder.linear_roundtrip(
        cavity_length,
        crystal_length,
        radius_1,
        radius_2,
        refractive_index,
    )
    return cavity_q_parameter(matrix)


def hemilithic_m_factor(radius_1, air_gap, crystal_length, refractive_index, radius_2=None):
    """Return m-factor for a hemilithic cavity."""
    radius_1, radius_2 = _radius_pair(radius_1, radius_2)
    matrix = CavityAbcdBuilder.hemilithic_roundtrip(
        air_gap,
        crystal_length,
        radius_1,
        radius_2,
        refractive_index,
    )
    return cavity_stability(matrix)


def hemilithic_q_parameter(radius_1, air_gap, crystal_length, refractive_index, radius_2=None):
    """Return q parameter for a hemilithic cavity."""
    radius_1, radius_2 = _radius_pair(radius_1, radius_2)
    matrix = CavityAbcdBuilder.hemilithic_roundtrip(
        air_gap,
        crystal_length,
        radius_1,
        radius_2,
        refractive_index,
    )
    return cavity_q_parameter(matrix)


def monolithic_m_factor(radius_1, crystal_length, refractive_index, radius_2=None):
    """Return m-factor for a monolithic crystal cavity."""
    radius_1, radius_2 = _radius_pair(radius_1, radius_2)
    matrix = CavityAbcdBuilder.monolithic_roundtrip(
        crystal_length,
        refractive_index,
        radius_1,
        radius_2,
    )
    return cavity_stability(matrix)


def monolithic_q_parameter(radius_1, crystal_length, refractive_index, radius_2=None):
    """Return q parameter for a monolithic crystal cavity."""
    radius_1, radius_2 = _radius_pair(radius_1, radius_2)
    matrix = CavityAbcdBuilder.monolithic_roundtrip(
        crystal_length,
        refractive_index,
        radius_1,
        radius_2,
    )
    return cavity_q_parameter(matrix)


def triangle_m_factor(width, height, crystal_length, radius_1, refractive_index, radius_2=None, plane="sagittal"):
    """Return m-factor for a triangular cavity."""
    radius_1, radius_2 = _radius_pair(radius_1, radius_2)
    matrix = CavityAbcdBuilder.triangle_roundtrip(
        width,
        height,
        crystal_length,
        radius_1,
        radius_2,
        refractive_index,
        plane=plane,
    )
    return cavity_stability(matrix)


def triangle_q_parameter(width, height, crystal_length, radius_1, refractive_index, radius_2=None, plane="sagittal"):
    """Return q parameter for a triangular cavity."""
    radius_1, radius_2 = _radius_pair(radius_1, radius_2)
    matrix = CavityAbcdBuilder.triangle_roundtrip(
        width,
        height,
        crystal_length,
        radius_1,
        radius_2,
        refractive_index,
        plane=plane,
    )
    return cavity_q_parameter(matrix)


def bowtie_m_factor_from_curvature(
    long_axis,
    short_axis,
    incidence_angle,
    crystal_length,
    curvature_1,
    curvature_2,
    refractive_index,
    plane="sagittal",
):
    """Return bow-tie m-factor using mirror curvatures."""
    matrix = CavityAbcdBuilder.bowtie_roundtrip_from_curvature(
        long_axis,
        short_axis,
        crystal_length,
        curvature_1,
        curvature_2,
        refractive_index,
        incidence_angle,
        plane=plane,
    )
    return cavity_stability(matrix)


def bowtie_q_parameter_from_curvature(
    long_axis,
    short_axis,
    incidence_angle,
    crystal_length,
    curvature_1,
    curvature_2,
    refractive_index,
    plane="sagittal",
):
    """Return bow-tie q parameter using mirror curvatures."""
    matrix = CavityAbcdBuilder.bowtie_roundtrip_from_curvature(
        long_axis,
        short_axis,
        crystal_length,
        curvature_1,
        curvature_2,
        refractive_index,
        incidence_angle,
        plane=plane,
    )
    return cavity_q_parameter(matrix)


def linear_m_factor_from_curvature(curvature_1, curvature_2, cavity_length, crystal_length, refractive_index):
    """Return linear-cavity m-factor using mirror curvatures."""
    matrix = CavityAbcdBuilder.linear_roundtrip_from_curvature(
        cavity_length,
        crystal_length,
        curvature_1,
        curvature_2,
        refractive_index,
    )
    return cavity_stability(matrix)


def linear_q_parameter_from_curvature(curvature_1, curvature_2, cavity_length, crystal_length, refractive_index):
    """Return linear-cavity q parameter using mirror curvatures."""
    matrix = CavityAbcdBuilder.linear_roundtrip_from_curvature(
        cavity_length,
        crystal_length,
        curvature_1,
        curvature_2,
        refractive_index,
    )
    return cavity_q_parameter(matrix)


def hemilithic_m_factor_from_curvature(curvature_1, curvature_2, air_gap, crystal_length, refractive_index):
    """Return hemilithic-cavity m-factor using curvatures."""
    matrix = CavityAbcdBuilder.hemilithic_roundtrip_from_curvature(
        air_gap,
        crystal_length,
        curvature_1,
        curvature_2,
        refractive_index,
    )
    return cavity_stability(matrix)


def hemilithic_q_parameter_from_curvature(curvature_1, curvature_2, air_gap, crystal_length, refractive_index):
    """Return hemilithic-cavity q parameter using curvatures."""
    matrix = CavityAbcdBuilder.hemilithic_roundtrip_from_curvature(
        air_gap,
        crystal_length,
        curvature_1,
        curvature_2,
        refractive_index,
    )
    return cavity_q_parameter(matrix)


def monolithic_m_factor_from_curvature(curvature_1, curvature_2, crystal_length, refractive_index):
    """Return monolithic-cavity m-factor using facet curvatures."""
    matrix = CavityAbcdBuilder.monolithic_roundtrip_from_curvature(
        crystal_length,
        refractive_index,
        curvature_1,
        curvature_2,
    )
    return cavity_stability(matrix)


def monolithic_q_parameter_from_curvature(curvature_1, curvature_2, crystal_length, refractive_index):
    """Return monolithic-cavity q parameter using facet curvatures."""
    matrix = CavityAbcdBuilder.monolithic_roundtrip_from_curvature(
        crystal_length,
        refractive_index,
        curvature_1,
        curvature_2,
    )
    return cavity_q_parameter(matrix)


def triangle_m_factor_from_curvature(width, height, crystal_length, curvature_1, curvature_2, refractive_index, plane="sagittal"):
    """Return triangular-cavity m-factor using mirror curvatures."""
    matrix = CavityAbcdBuilder.triangle_roundtrip_from_curvature(
        width,
        height,
        crystal_length,
        curvature_1,
        curvature_2,
        refractive_index,
        plane=plane,
    )
    return cavity_stability(matrix)


def triangle_q_parameter_from_curvature(width, height, crystal_length, curvature_1, curvature_2, refractive_index, plane="sagittal"):
    """Return triangular-cavity q parameter using mirror curvatures."""
    matrix = CavityAbcdBuilder.triangle_roundtrip_from_curvature(
        width,
        height,
        crystal_length,
        curvature_1,
        curvature_2,
        refractive_index,
        plane=plane,
    )
    return cavity_q_parameter(matrix)


def make_m_factor_estimator(geometry: str, plane: str = "sagittal"):
    """Build a NumPy-callable m-factor estimator for a geometry."""
    if geometry == "bowtie":
        long_axis, short_axis, incidence_angle, crystal_length, curvature_1, curvature_2, refractive_index = sp.symbols(
            "long_axis short_axis incidence_angle crystal_length curvature_1 curvature_2 refractive_index",
            nonnegative=True,
            real=True,
        )
        expr = bowtie_m_factor_from_curvature(
            long_axis,
            short_axis,
            incidence_angle,
            crystal_length,
            curvature_1,
            curvature_2,
            refractive_index,
            plane=plane,
        )
        estimator = sp.lambdify(
            (long_axis, short_axis, incidence_angle, crystal_length, curvature_1, curvature_2, refractive_index),
            expr,
            modules="numpy",
            cse=True,
        )
        return lambda long_axis_val, short_axis_val, incidence_angle_val, crystal_length_val, radius_1_val, radius_2_val, refractive_index_val: estimator(
            long_axis_val,
            short_axis_val,
            incidence_angle_val,
            crystal_length_val,
            *_curvature_pair(radius_1_val, radius_2_val),
            refractive_index_val,
        )

    if geometry == "linear":
        curvature_1, curvature_2, cavity_length, crystal_length, refractive_index = sp.symbols(
            "curvature_1 curvature_2 cavity_length crystal_length refractive_index", nonnegative=True, real=True
        )
        expr = linear_m_factor_from_curvature(curvature_1, curvature_2, cavity_length, crystal_length, refractive_index)
        estimator = sp.lambdify(
            (curvature_1, curvature_2, cavity_length, crystal_length, refractive_index),
            expr,
            modules="numpy",
            cse=True,
        )
        return lambda radius_1_val, radius_2_val, cavity_length_val, crystal_length_val, refractive_index_val: estimator(
            *_curvature_pair(radius_1_val, radius_2_val),
            cavity_length_val,
            crystal_length_val,
            refractive_index_val,
        )

    if geometry == "hemilithic":
        curvature_1, curvature_2, air_gap, crystal_length, refractive_index = sp.symbols(
            "curvature_1 curvature_2 air_gap crystal_length refractive_index", nonnegative=True, real=True
        )
        expr = hemilithic_m_factor_from_curvature(curvature_1, curvature_2, air_gap, crystal_length, refractive_index)
        estimator = sp.lambdify(
            (curvature_1, curvature_2, air_gap, crystal_length, refractive_index),
            expr,
            modules="numpy",
            cse=True,
        )
        return lambda radius_1_val, radius_2_val, air_gap_val, crystal_length_val, refractive_index_val: estimator(
            *_curvature_pair(radius_1_val, radius_2_val),
            air_gap_val,
            crystal_length_val,
            refractive_index_val,
        )

    if geometry == "monolithic":
        curvature_1, curvature_2, crystal_length, refractive_index = sp.symbols(
            "curvature_1 curvature_2 crystal_length refractive_index", nonnegative=True, real=True
        )
        expr = monolithic_m_factor_from_curvature(curvature_1, curvature_2, crystal_length, refractive_index)
        estimator = sp.lambdify(
            (curvature_1, curvature_2, crystal_length, refractive_index),
            expr,
            modules="numpy",
            cse=True,
        )
        return lambda radius_1_val, radius_2_val, crystal_length_val, refractive_index_val: estimator(
            *_curvature_pair(radius_1_val, radius_2_val),
            crystal_length_val,
            refractive_index_val,
        )

    if geometry == "triangle":
        width, height, crystal_length, curvature_1, curvature_2, refractive_index = sp.symbols(
            "width height crystal_length curvature_1 curvature_2 refractive_index", nonnegative=True, real=True
        )
        expr = triangle_m_factor_from_curvature(
            width,
            height,
            crystal_length,
            curvature_1,
            curvature_2,
            refractive_index,
            plane=plane,
        )
        estimator = sp.lambdify(
            (width, height, crystal_length, curvature_1, curvature_2, refractive_index),
            expr,
            modules="numpy",
            cse=True,
        )
        return lambda width_val, height_val, crystal_length_val, radius_1_val, radius_2_val, refractive_index_val: estimator(
            width_val,
            height_val,
            crystal_length_val,
            *_curvature_pair(radius_1_val, radius_2_val),
            refractive_index_val,
        )

    raise ValueError("geometry must be 'bowtie', 'linear', 'triangle', 'hemilithic', or 'monolithic'")


def make_q_estimator(geometry: str, plane: str = "sagittal"):
    """Build a NumPy-callable q-parameter estimator for a geometry."""
    if geometry == "bowtie":
        long_axis, short_axis, incidence_angle, crystal_length, curvature_1, curvature_2, refractive_index = sp.symbols(
            "long_axis short_axis incidence_angle crystal_length curvature_1 curvature_2 refractive_index",
            nonnegative=True,
            real=True,
        )
        expr = bowtie_q_parameter_from_curvature(
            long_axis,
            short_axis,
            incidence_angle,
            crystal_length,
            curvature_1,
            curvature_2,
            refractive_index,
            plane=plane,
        )
        estimator = sp.lambdify(
            (long_axis, short_axis, incidence_angle, crystal_length, curvature_1, curvature_2, refractive_index),
            expr,
            modules="numpy",
            cse=True,
        )
        return lambda long_axis_val, short_axis_val, incidence_angle_val, crystal_length_val, radius_1_val, radius_2_val, refractive_index_val: estimator(
            long_axis_val,
            short_axis_val,
            incidence_angle_val,
            crystal_length_val,
            *_curvature_pair(radius_1_val, radius_2_val),
            refractive_index_val,
        )

    if geometry == "linear":
        curvature_1, curvature_2, cavity_length, crystal_length, refractive_index = sp.symbols(
            "curvature_1 curvature_2 cavity_length crystal_length refractive_index", nonnegative=True, real=True
        )
        expr = linear_q_parameter_from_curvature(curvature_1, curvature_2, cavity_length, crystal_length, refractive_index)
        estimator = sp.lambdify(
            (curvature_1, curvature_2, cavity_length, crystal_length, refractive_index),
            expr,
            modules="numpy",
            cse=True,
        )
        return lambda radius_1_val, radius_2_val, cavity_length_val, crystal_length_val, refractive_index_val: estimator(
            *_curvature_pair(radius_1_val, radius_2_val),
            cavity_length_val,
            crystal_length_val,
            refractive_index_val,
        )

    if geometry == "hemilithic":
        curvature_1, curvature_2, air_gap, crystal_length, refractive_index = sp.symbols(
            "curvature_1 curvature_2 air_gap crystal_length refractive_index", nonnegative=True, real=True
        )
        expr = hemilithic_q_parameter_from_curvature(curvature_1, curvature_2, air_gap, crystal_length, refractive_index)
        estimator = sp.lambdify(
            (curvature_1, curvature_2, air_gap, crystal_length, refractive_index),
            expr,
            modules="numpy",
            cse=True,
        )
        return lambda radius_1_val, radius_2_val, air_gap_val, crystal_length_val, refractive_index_val: estimator(
            *_curvature_pair(radius_1_val, radius_2_val),
            air_gap_val,
            crystal_length_val,
            refractive_index_val,
        )

    if geometry == "monolithic":
        curvature_1, curvature_2, crystal_length, refractive_index = sp.symbols(
            "curvature_1 curvature_2 crystal_length refractive_index", nonnegative=True, real=True
        )
        expr = monolithic_q_parameter_from_curvature(curvature_1, curvature_2, crystal_length, refractive_index)
        estimator = sp.lambdify(
            (curvature_1, curvature_2, crystal_length, refractive_index),
            expr,
            modules="numpy",
            cse=True,
        )
        return lambda radius_1_val, radius_2_val, crystal_length_val, refractive_index_val: estimator(
            *_curvature_pair(radius_1_val, radius_2_val),
            crystal_length_val,
            refractive_index_val,
        )

    if geometry == "triangle":
        width, height, crystal_length, curvature_1, curvature_2, refractive_index = sp.symbols(
            "width height crystal_length curvature_1 curvature_2 refractive_index", nonnegative=True, real=True
        )
        expr = triangle_q_parameter_from_curvature(
            width,
            height,
            crystal_length,
            curvature_1,
            curvature_2,
            refractive_index,
            plane=plane,
        )
        estimator = sp.lambdify(
            (width, height, crystal_length, curvature_1, curvature_2, refractive_index),
            expr,
            modules="numpy",
            cse=True,
        )
        return lambda width_val, height_val, crystal_length_val, radius_1_val, radius_2_val, refractive_index_val: estimator(
            width_val,
            height_val,
            crystal_length_val,
            *_curvature_pair(radius_1_val, radius_2_val),
            refractive_index_val,
        )

    raise ValueError("geometry must be 'bowtie', 'linear', 'triangle', 'hemilithic', or 'monolithic'")


__all__ = [
    "cavity_stability",
    "cavity_q_parameter",
    "beam_waist_from_q",
    "radius_to_curvature",
    "optical_roundtrip_length",
    "fsr_from_roundtrip_length",
    "compute_decay_rates",
    "gouy_phases_from_m_factor",
    "bowtie_m_factor",
    "bowtie_q_parameter",
    "linear_m_factor",
    "linear_q_parameter",
    "hemilithic_m_factor",
    "hemilithic_q_parameter",
    "monolithic_m_factor",
    "monolithic_q_parameter",
    "triangle_m_factor",
    "triangle_q_parameter",
    "bowtie_m_factor_from_curvature",
    "bowtie_q_parameter_from_curvature",
    "linear_m_factor_from_curvature",
    "linear_q_parameter_from_curvature",
    "hemilithic_m_factor_from_curvature",
    "hemilithic_q_parameter_from_curvature",
    "monolithic_m_factor_from_curvature",
    "monolithic_q_parameter_from_curvature",
    "triangle_m_factor_from_curvature",
    "triangle_q_parameter_from_curvature",
    "make_m_factor_estimator",
    "make_q_estimator",
]
