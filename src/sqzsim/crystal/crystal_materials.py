"""Refractive-index and thermo-optic models for crystal simulations.

This module provides the available crystal refractive-index models, but does
not choose one internally. Model selection must be made externally by the
workflow or main script. The currently implemented model set is not fully
axis-distinct: the ``n_x`` branch reuses one shared reference fit across all
supported literature options, while the implemented model-specific differences
are in ``n_y`` and ``n_z``.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Callable, Iterator, Mapping, Optional

import numpy as np

ScalarOrArray = np.ndarray | float
AxisFunction = Callable[[ScalarOrArray, float], ScalarOrArray]
SUPPORTED_CRYSTAL_MODELS = ("Kato2002", "Fan1987", "Konig2004")


@dataclass(frozen=True)
class SellmeierCoefficients:
    """Generic Sellmeier coefficient container.

    Uses the common form:
    n^2(λ) = A + B/(λ^2 - C) + D/(λ^2 - E) + F*λ^2, with λ in micrometers.
    """

    A: float
    B: float
    C: float
    D: float = 0.0
    E: float = 0.0
    F: float = 0.0


@dataclass(frozen=True)
class ThermoOpticCoefficients:
    """Wavelength-dependent polynomial coefficients for quadratic thermo-optic terms."""

    a0: float
    a1: float
    a2: float
    a3: float


@dataclass(frozen=True)
class AxisModel:
    """One axis refractive-index model built from a base Sellmeier and thermo-optic terms."""

    base_index_um: Callable[[ScalarOrArray], ScalarOrArray]
    reference_temperature_C: float
    thermo_linear: ThermoOpticCoefficients | None = None
    thermo_quadratic: ThermoOpticCoefficients | None = None


@dataclass(frozen=True)
class LiteratureAxisModelSet:
    """Axis-model bundle for one literature source.

    The currently supported sources share a common ``n_x`` reference branch.
    Only ``n_y`` and ``n_z`` differ across the implemented literature fits.
    """

    n_x_of_T: AxisModel
    n_y_of_T: AxisModel
    n_z_of_T: AxisModel


@dataclass(frozen=True)
class RefractiveIndexModel(Mapping[str, object]):
    """Structured refractive-index model with backward-compatible mapping access."""

    model_name: str
    n_x_of_T: AxisFunction
    n_y_of_T: AxisFunction
    n_z_of_T: AxisFunction

    def __getitem__(self, key: str) -> object:
        try:
            return getattr(self, key)
        except AttributeError as exc:
            raise KeyError(key) from exc

    def __iter__(self) -> Iterator[str]:
        yield "model_name"
        yield "n_x_of_T"
        yield "n_y_of_T"
        yield "n_z_of_T"

    def __len__(self) -> int:
        return 4


@dataclass(frozen=True)
class PhaseMatchingConfiguration:
    """Resolved axis assignment for one nonlinear interaction type."""

    phase_matching_type: str
    pump_axis: str
    signal_axis: str
    idler_axis: str


@dataclass(frozen=True)
class EffectiveNonlinearityConfiguration:
    """Resolved effective nonlinearity for one crystal model and interaction type."""

    crystal_model: str
    phase_matching_type: str
    d_eff_pm_per_V: float
    notes: tuple[str, ...] = ()


def central_diff(f: Callable[[float], float], x: float, dx: float) -> float:
    """Return the central finite-difference derivative of ``f`` at ``x``."""
    return (f(x + dx) - f(x - dx)) / (2.0 * dx)


def n_sellmeier_um(lambda_um: ScalarOrArray, coeffs: SellmeierCoefficients) -> ScalarOrArray:
    """Evaluate refractive index from a standard Sellmeier model."""
    lam2 = np.asarray(lambda_um, dtype=float) ** 2
    n2 = (
        coeffs.A
        + coeffs.B / (lam2 - coeffs.C)
        + (coeffs.D / (lam2 - coeffs.E) if coeffs.D != 0.0 else 0.0)
        + coeffs.F * lam2
    )
    n = np.sqrt(n2)
    return n if isinstance(lambda_um, np.ndarray) else float(n)


def n_from_model(
    wavelength_m: ScalarOrArray,
    T_K: float,
    n_lambda: Callable[[ScalarOrArray], ScalarOrArray],
    dn_dT_perK: Optional[Callable[[float], float]] = None,
    T_ref_K: float = 293.15,
) -> ScalarOrArray:
    """Evaluate ``n(λ, T)`` from base ``n(λ)`` and an optional linear ``dn/dT`` term."""
    n0 = n_lambda(wavelength_m)
    if dn_dT_perK is None:
        return n0

    wl = np.asarray(wavelength_m, dtype=float)
    if wl.ndim == 0:
        dn = dn_dT_perK(float(wl))
        return n0 + dn * (T_K - T_ref_K)

    dn_vec = np.vectorize(lambda w: dn_dT_perK(float(w)), otypes=[float])(wl)
    return np.asarray(n0, dtype=float) + dn_vec * (T_K - T_ref_K)


def dn_dT_numeric(
    n_of_T: Callable[[float], float],
    T_K: float,
    dT_K: float = 1e-2,
) -> float:
    """Estimate ``dn/dT`` at temperature ``T_K`` via central finite difference."""
    return central_diff(n_of_T, T_K, dT_K)


def _to_lambda_um(wavelength_m: ScalarOrArray) -> ScalarOrArray:
    lam_um = np.asarray(wavelength_m, dtype=float) * 1e6
    return lam_um if isinstance(wavelength_m, np.ndarray) else float(lam_um)


def _sellmeier_reference_um(lambda_um: ScalarOrArray, coeffs: SellmeierCoefficients) -> ScalarOrArray:
    """Evaluate the five-coefficient Sellmeier form used for Kato2002."""
    return n_sellmeier_um(lambda_um, coeffs)


def _sellmeier_fan_konig_um(
    lambda_um: ScalarOrArray,
    b0: float,
    b1: float,
    b2: float,
    b3: float,
) -> ScalarOrArray:
    """Evaluate the reduced Sellmeier form used in Fan1987 and Konig2004."""
    lam = np.asarray(lambda_um, dtype=float)
    n = np.sqrt(b0 + b1 / (1.0 - b2 / lam**2) - b3 * lam**2)
    return n if isinstance(lambda_um, np.ndarray) else float(n)


def _sellmeier_fradkin_um(
    lambda_um: ScalarOrArray,
    b0: float,
    b1: float,
    b2: float,
    b3: float,
    b4: float,
    b5: float,
) -> ScalarOrArray:
    """Evaluate the six-coefficient Sellmeier form used by Fradkin1999."""
    lam = np.asarray(lambda_um, dtype=float)
    n = np.sqrt(b0 + b1 / (1.0 - b2 / lam**2) + b3 / (1.0 - b4 / lam**2) - b5 * lam**2)
    return n if isinstance(lambda_um, np.ndarray) else float(n)


def _deltan_coeff(lambda_um: ScalarOrArray, coeffs: ThermoOpticCoefficients) -> ScalarOrArray:
    """Evaluate the wavelength-dependent thermo-optic polynomial from the reference model."""
    lam = np.asarray(lambda_um, dtype=float)
    dn = coeffs.a0 + coeffs.a1 / lam + coeffs.a2 / lam**2 + coeffs.a3 / lam**3
    return dn if isinstance(lambda_um, np.ndarray) else float(dn)


def _nx_reference_um(lambda_um: ScalarOrArray) -> ScalarOrArray:
    """Shared ``n_x`` reference fit reused across all supported model options."""
    return _sellmeier_reference_um(
        lambda_um,
        SellmeierCoefficients(3.29100, 0.04140, 0.03978, 9.35522, 31.45571),
    )


def _ny_kato2002_um(lambda_um: ScalarOrArray) -> ScalarOrArray:
    return _sellmeier_reference_um(
        lambda_um,
        SellmeierCoefficients(3.45018, 0.04341, 0.04597, 16.98825, 39.43799),
    )


def _nz_kato2002_um(lambda_um: ScalarOrArray) -> ScalarOrArray:
    return _sellmeier_reference_um(
        lambda_um,
        SellmeierCoefficients(4.59423, 0.06206, 0.04763, 110.80672, 86.12171),
    )


def _ny_fan1987_um(lambda_um: ScalarOrArray) -> ScalarOrArray:
    return _sellmeier_fan_konig_um(lambda_um, 2.19229, 0.83547, 0.04970, 0.01621)


def _nz_fan1987_um(lambda_um: ScalarOrArray) -> ScalarOrArray:
    return _sellmeier_fan_konig_um(lambda_um, 2.25411, 1.06543, 0.05486, 0.02140)


def _ny_konig2004_um(lambda_um: ScalarOrArray) -> ScalarOrArray:
    return _sellmeier_fan_konig_um(lambda_um, 2.09930, 0.922683, 0.0467695, 0.0138408)


def _nz_konig2004_um(lambda_um: ScalarOrArray) -> ScalarOrArray:
    return _sellmeier_fradkin_um(
        lambda_um,
        2.12725,
        1.18431,
        0.0514852,
        0.66030,
        100.00507,
        9.68956e-3,
    )


def _evaluate_axis_model(
    wavelength_m: ScalarOrArray,
    T_K: float,
    axis_model: AxisModel,
) -> ScalarOrArray:
    """Evaluate one axis model at wavelength ``wavelength_m`` and temperature ``T_K``."""
    lambda_um = _to_lambda_um(wavelength_m)
    base_n = axis_model.base_index_um(lambda_um)

    if axis_model.thermo_linear is None and axis_model.thermo_quadratic is None:
        return base_n

    # The literature thermo-optic terms are referenced to a specific temperature
    # in degrees Celsius, so the workflow temperature in Kelvin is converted to
    # a Celsius offset relative to that reference point here.
    delta_T_C = float(T_K) - 273.15 - axis_model.reference_temperature_C
    n = np.asarray(base_n, dtype=float)

    if axis_model.thermo_linear is not None:
        n = n + np.asarray(_deltan_coeff(lambda_um, axis_model.thermo_linear), dtype=float) * delta_T_C
    if axis_model.thermo_quadratic is not None:
        n = n + np.asarray(_deltan_coeff(lambda_um, axis_model.thermo_quadratic), dtype=float) * delta_T_C**2

    return n if isinstance(wavelength_m, np.ndarray) else float(n)


_COMMON_NX_MODEL = AxisModel(
    # Kato2002 is the default baseline. Fan1987 and Konig2004 currently reuse
    # this shared ``n_x`` reference branch unchanged; only ``n_y`` and ``n_z``
    # differ across the implemented literature fits.
    base_index_um=_nx_reference_um,
    reference_temperature_C=25.0,
)

_THERMO_NY_LINEAR = ThermoOpticCoefficients(6.2897e-6, 6.3061e-6, -6.0629e-6, 2.6486e-6)
_THERMO_NY_QUADRATIC = ThermoOpticCoefficients(-0.14445e-8, 2.2244e-8, -3.5770e-8, 1.3470e-8)
_THERMO_NZ_LINEAR = ThermoOpticCoefficients(9.9587e-6, 9.9228e-6, -8.9603e-6, 4.1010e-6)
_THERMO_NZ_QUADRATIC = ThermoOpticCoefficients(-1.1882e-8, 10.459e-8, -9.8136e-8, 3.1481e-8)

_MODEL_SPECS: dict[str, LiteratureAxisModelSet] = {
    "Kato2002": LiteratureAxisModelSet(
        n_x_of_T=_COMMON_NX_MODEL,
        n_y_of_T=AxisModel(
            base_index_um=_ny_kato2002_um,
            reference_temperature_C=20.0,
            thermo_linear=_THERMO_NY_LINEAR,
            thermo_quadratic=_THERMO_NY_QUADRATIC,
        ),
        n_z_of_T=AxisModel(
            base_index_um=_nz_kato2002_um,
            reference_temperature_C=20.0,
            thermo_linear=_THERMO_NZ_LINEAR,
            thermo_quadratic=_THERMO_NZ_QUADRATIC,
        ),
    ),
    "Fan1987": LiteratureAxisModelSet(
        n_x_of_T=_COMMON_NX_MODEL,
        n_y_of_T=AxisModel(
            base_index_um=_ny_fan1987_um,
            reference_temperature_C=25.0,
            thermo_linear=_THERMO_NY_LINEAR,
            thermo_quadratic=_THERMO_NY_QUADRATIC,
        ),
        n_z_of_T=AxisModel(
            base_index_um=_nz_fan1987_um,
            reference_temperature_C=25.0,
            thermo_linear=_THERMO_NZ_LINEAR,
            thermo_quadratic=_THERMO_NZ_QUADRATIC,
        ),
    ),
    "Konig2004": LiteratureAxisModelSet(
        n_x_of_T=_COMMON_NX_MODEL,
        n_y_of_T=AxisModel(
            base_index_um=_ny_konig2004_um,
            reference_temperature_C=25.0,
            thermo_linear=_THERMO_NY_LINEAR,
            thermo_quadratic=_THERMO_NY_QUADRATIC,
        ),
        n_z_of_T=AxisModel(
            base_index_um=_nz_konig2004_um,
            reference_temperature_C=25.0,
            thermo_linear=_THERMO_NZ_LINEAR,
            thermo_quadratic=_THERMO_NZ_QUADRATIC,
        ),
    ),
}

_PHASE_MATCHING_CONFIGS: dict[str, PhaseMatchingConfiguration] = {
    "type_0": PhaseMatchingConfiguration(
        phase_matching_type="type_0",
        pump_axis="z",
        signal_axis="z",
        idler_axis="z",
    ),
    "type_i": PhaseMatchingConfiguration(
        phase_matching_type="type_I",
        pump_axis="z",
        signal_axis="y",
        idler_axis="y",
    ),
    # For degenerate operation the signal/idler labels are conventional; the
    # important distinction is that the two generated fields use different axes.
    "type_ii": PhaseMatchingConfiguration(
        phase_matching_type="type_II",
        pump_axis="z",
        signal_axis="y",
        idler_axis="z",
    ),
}

_COMMON_DEFF_NOTES = (
    "Effective nonlinearity for future OPO coupling integration.",
    "Current values are tied to the present axis-based interaction-type model.",
    "The current type_II value reflects the present model assumptions rather than a full general polarization treatment.",
)

_COMMON_DEFF_BY_TYPE_PM_PER_V: dict[str, float] = {
    "type_0": 16.9,
    "type_i": 3.64,
    "type_ii": 3.64,
}

_EFFECTIVE_NONLINEARITY_CONFIGS: dict[str, dict[str, float]] = {
    model_name: dict(_COMMON_DEFF_BY_TYPE_PM_PER_V) for model_name in SUPPORTED_CRYSTAL_MODELS
}


def supported_crystal_models() -> tuple[str, ...]:
    """Return the supported literature models."""
    return SUPPORTED_CRYSTAL_MODELS


def _validate_model_name(model_name: str) -> str:
    if model_name not in SUPPORTED_CRYSTAL_MODELS:
        supported = ", ".join(SUPPORTED_CRYSTAL_MODELS)
        raise ValueError(f"Unknown crystal model: {model_name}. Supported models: {supported}")
    return model_name


def _build_axis_function(axis_model: AxisModel) -> AxisFunction:
    return lambda wavelength_m, T_K: _evaluate_axis_model(wavelength_m, T_K, axis_model)


def get_axis_index_function(refractive_index_model: RefractiveIndexModel, axis: str) -> AxisFunction:
    """Return the refractive-index function for one crystal axis."""
    axis_key = str(axis).lower()
    try:
        return getattr(refractive_index_model, f"n_{axis_key}_of_T")
    except AttributeError as exc:
        raise ValueError(f"Unknown crystal axis: {axis}. Expected one of: x, y, z.") from exc


def resolve_phase_matching_configuration(phase_matching_type: str) -> PhaseMatchingConfiguration:
    """Resolve the pump/signal/idler axis assignment for one interaction type."""
    normalized_type = str(phase_matching_type).lower()
    try:
        return _PHASE_MATCHING_CONFIGS[normalized_type]
    except KeyError as exc:
        supported = ", ".join(_PHASE_MATCHING_CONFIGS)
        raise ValueError(
            f"Unknown phase_matching_type: {phase_matching_type}. Supported values: {supported}"
        ) from exc


def resolve_effective_nonlinearity(
    crystal_model: str,
    phase_matching_type: str,
) -> EffectiveNonlinearityConfiguration:
    """Resolve ``d_eff`` for one crystal model and nonlinear interaction type."""
    selected_model = _validate_model_name(crystal_model)
    normalized_type = str(phase_matching_type).lower()
    if normalized_type not in _PHASE_MATCHING_CONFIGS:
        supported = ", ".join(_PHASE_MATCHING_CONFIGS)
        raise ValueError(
            f"Unknown phase_matching_type: {phase_matching_type}. Supported values: {supported}"
        )

    try:
        d_eff_pm_per_V = _EFFECTIVE_NONLINEARITY_CONFIGS[selected_model][normalized_type]
    except KeyError as exc:
        raise ValueError(
            f"No d_eff configured for crystal_model={crystal_model} and phase_matching_type={phase_matching_type}."
        ) from exc

    return EffectiveNonlinearityConfiguration(
        crystal_model=selected_model,
        phase_matching_type=_PHASE_MATCHING_CONFIGS[normalized_type].phase_matching_type,
        d_eff_pm_per_V=float(d_eff_pm_per_V),
        notes=_COMMON_DEFF_NOTES,
    )


@lru_cache(maxsize=None)
def build_refractive_index_model(model_name: str) -> RefractiveIndexModel:
    """Build axis functions for one explicitly selected literature model."""

    selected_model = _validate_model_name(model_name)
    model_spec = _MODEL_SPECS[selected_model]
    return RefractiveIndexModel(
        model_name=selected_model,
        n_x_of_T=_build_axis_function(model_spec.n_x_of_T),
        n_y_of_T=_build_axis_function(model_spec.n_y_of_T),
        n_z_of_T=_build_axis_function(model_spec.n_z_of_T),
    )


def nx(
    lambda_m: ScalarOrArray,
    T_K: float,
    model: str,
) -> ScalarOrArray:
    """Return ``n_x(λ, T)`` for the selected literature model."""
    return build_refractive_index_model(model).n_x_of_T(lambda_m, T_K)


def ny(
    lambda_m: ScalarOrArray,
    T_K: float,
    model: str,
) -> ScalarOrArray:
    """Return ``n_y(λ, T)`` for the selected literature model."""
    return build_refractive_index_model(model).n_y_of_T(lambda_m, T_K)


def nz(
    lambda_m: ScalarOrArray,
    T_K: float,
    model: str,
) -> ScalarOrArray:
    """Return ``n_z(λ, T)`` for the selected literature model."""
    return build_refractive_index_model(model).n_z_of_T(lambda_m, T_K)


__all__ = [
    "SUPPORTED_CRYSTAL_MODELS",
    "SellmeierCoefficients",
    "ThermoOpticCoefficients",
    "AxisModel",
    "LiteratureAxisModelSet",
    "RefractiveIndexModel",
    "PhaseMatchingConfiguration",
    "EffectiveNonlinearityConfiguration",
    "central_diff",
    "n_sellmeier_um",
    "n_from_model",
    "dn_dT_numeric",
    "supported_crystal_models",
    "build_refractive_index_model",
    "get_axis_index_function",
    "resolve_phase_matching_configuration",
    "resolve_effective_nonlinearity",
    "nx",
    "ny",
    "nz",
]
