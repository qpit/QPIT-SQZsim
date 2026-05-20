"""Unified ABCD optics helpers and cavity-specific round-trip builders."""

from __future__ import annotations

import sympy as sp


# --- Generic ABCD optics utilities ---


class Abcd:
    """Static helpers for ABCD matrix elements and operations."""

    @staticmethod
    def propagation(length):
        """Free-space (or uniform-medium) propagation matrix."""
        return sp.Matrix([[1, length], [0, 1]])

    @staticmethod
    def planar_interface(n1, n2):
        """Planar dielectric interface matrix."""
        return sp.Matrix([[1, 0], [0, n1 / n2]])

    @staticmethod
    def curved_interface(n1, n2, radius):
        """Curved dielectric interface matrix."""
        validate_radius_of_curvature(radius, "radius")
        if is_infinite_radius(radius):
            return Abcd.planar_interface(n1, n2)
        return sp.Matrix([[1, 0], [(n1 - n2) / (n2 * radius), n1 / n2]])

    @staticmethod
    def thin_lens(focal_length):
        """Thin lens matrix."""
        return sp.Matrix([[1, 0], [-1 / focal_length, 1]])

    @staticmethod
    def mirror(radius_of_curvature, incidence_angle=0.0, plane: str = "tangential"):
        """Spherical mirror reflection matrix with optional astigmatism plane."""
        validate_plane(plane)
        validate_radius_of_curvature(radius_of_curvature, "radius_of_curvature")
        if is_infinite_radius(radius_of_curvature):
            return sp.eye(2)
        if plane == "tangential":
            radius_eff = radius_of_curvature * sp.cos(incidence_angle)
        else:
            radius_eff = radius_of_curvature / sp.cos(incidence_angle)
        return sp.Matrix([[1, 0], [-2 / radius_eff, 1]])

    @staticmethod
    def mirror_from_curvature(curvature, incidence_angle=0.0, plane: str = "tangential"):
        """Spherical mirror reflection matrix parameterized by curvature."""
        validate_plane(plane)
        if plane == "tangential":
            curvature_eff = curvature / sp.cos(incidence_angle)
        else:
            curvature_eff = curvature * sp.cos(incidence_angle)
        return sp.Matrix([[1, 0], [-2 * curvature_eff, 1]])

    @staticmethod
    def chain(*elements, simplify: bool = True):
        """Multiply a sequence of ABCD elements in order."""
        matrix = sp.eye(2)
        for element in elements:
            matrix = matrix @ element
        return sp.simplify(matrix) if simplify else matrix

    @staticmethod
    def parameters(matrix):
        """Extract (A, B, C, D) from a 2x2 ABCD matrix."""
        return matrix[0, 0], matrix[0, 1], matrix[1, 0], matrix[1, 1]


# --- Validation helpers ---


def validate_nonnegative(value, name: str) -> None:
    """Raise ValueError if a numeric value is negative."""
    if isinstance(value, sp.Basic):
        if value.is_number and float(sp.N(value)) < 0:
            raise ValueError(f"{name} must be >= 0")
    else:
        if float(value) < 0:
            raise ValueError(f"{name} must be >= 0")


def is_infinite_radius(value) -> bool:
    """Return True for symbolic or numeric infinite radii."""
    if isinstance(value, sp.Basic):
        return bool(value.is_infinite)
    return value == float("inf")


def validate_radius_of_curvature(value, name: str) -> None:
    """Validate radius as positive finite or infinite for a planar element."""
    if isinstance(value, sp.Basic):
        if value.is_infinite:
            return
        if value.is_number and float(sp.N(value)) <= 0:
            raise ValueError(f"{name} must be positive or np.inf")
        if value.is_positive is False:
            raise ValueError(f"{name} must be positive or np.inf")
        return
    numeric = float(value)
    if numeric <= 0 or numeric != numeric:
        raise ValueError(f"{name} must be positive or np.inf")


def radius_to_curvature(radius_m: float) -> float:
    """Return curvature in 1/m, mapping an infinite radius to zero."""
    validate_radius_of_curvature(radius_m, "radius_m")
    if is_infinite_radius(radius_m):
        return 0.0
    return 1.0 / float(radius_m)


def validate_plane(plane: str) -> None:
    """Validate sagittal/tangential plane selection."""
    if plane not in {"sagittal", "tangential"}:
        raise ValueError("plane must be 'sagittal' or 'tangential'")


# --- Cavity builders ---


def bowtie_diagonal(long_axis, short_axis, incidence_angle):
    """Return bow-tie diagonal segment length from long/short axes and AOI."""
    return (long_axis + short_axis) / 2 * sp.cos(incidence_angle)


class CavityAbcdBuilder:
    """Factory for round-trip cavity ABCD matrices across supported geometries."""

    @staticmethod
    def bowtie_roundtrip(
        long_axis,
        short_axis,
        crystal_length,
        radius_1,
        radius_2,
        refractive_index,
        incidence_angle,
        plane: str = "sagittal",
    ):
        """Round-trip matrix for a bow-tie cavity from crystal center.

        Reference plane is the crystal center. Radii use the reflection sign
        convention already used by :meth:`Abcd.mirror`; ``np.inf`` is planar.
        """
        validate_plane(plane)
        mirror = Abcd.mirror
        diagonal = bowtie_diagonal(long_axis, short_axis, incidence_angle)
        return Abcd.chain(
            Abcd.propagation(crystal_length / 2),
            Abcd.planar_interface(1, refractive_index),
            Abcd.propagation((short_axis - crystal_length) / 2),
            mirror(radius_1, incidence_angle, plane),
            Abcd.propagation(long_axis + 2 * diagonal),
            mirror(radius_2, incidence_angle, plane),
            Abcd.propagation((short_axis - crystal_length) / 2),
            Abcd.planar_interface(refractive_index, 1),
            Abcd.propagation(crystal_length / 2),
        )

    @staticmethod
    def bowtie_roundtrip_from_curvature(
        long_axis,
        short_axis,
        crystal_length,
        curvature_1,
        curvature_2,
        refractive_index,
        incidence_angle,
        plane: str = "sagittal",
    ):
        """Bow-tie round trip from crystal center using mirror curvatures."""
        validate_plane(plane)
        mirror = Abcd.mirror_from_curvature
        diagonal = bowtie_diagonal(long_axis, short_axis, incidence_angle)
        return Abcd.chain(
            Abcd.propagation(crystal_length / 2),
            Abcd.planar_interface(1, refractive_index),
            Abcd.propagation((short_axis - crystal_length) / 2),
            mirror(curvature_1, incidence_angle, plane),
            Abcd.propagation(long_axis + 2 * diagonal),
            mirror(curvature_2, incidence_angle, plane),
            Abcd.propagation((short_axis - crystal_length) / 2),
            Abcd.planar_interface(refractive_index, 1),
            Abcd.propagation(crystal_length / 2),
        )

    @staticmethod
    def linear_roundtrip(
        cavity_length,
        crystal_length,
        radius_1,
        radius_2,
        refractive_index,
    ):
        """Round-trip matrix for a linear cavity with centered crystal."""
        air_total = cavity_length - crystal_length
        validate_nonnegative(air_total, "cavity_length - crystal_length")
        air_half = air_total / 2
        return Abcd.chain(
            Abcd.propagation(air_half),
            Abcd.planar_interface(1, refractive_index),
            Abcd.propagation(crystal_length),
            Abcd.planar_interface(refractive_index, 1),
            Abcd.propagation(air_half),
            Abcd.mirror(radius_2, 0, "tangential"),
            Abcd.propagation(air_half),
            Abcd.planar_interface(1, refractive_index),
            Abcd.propagation(crystal_length),
            Abcd.planar_interface(refractive_index, 1),
            Abcd.propagation(air_half),
            Abcd.mirror(radius_1, 0, "tangential"),
        )

    @staticmethod
    def linear_roundtrip_from_curvature(
        cavity_length,
        crystal_length,
        curvature_1,
        curvature_2,
        refractive_index,
    ):
        """Round-trip matrix for a linear cavity using mirror curvatures."""
        air_total = cavity_length - crystal_length
        validate_nonnegative(air_total, "cavity_length - crystal_length")
        air_half = air_total / 2
        return Abcd.chain(
            Abcd.propagation(air_half),
            Abcd.planar_interface(1, refractive_index),
            Abcd.propagation(crystal_length),
            Abcd.planar_interface(refractive_index, 1),
            Abcd.propagation(air_half),
            Abcd.mirror_from_curvature(curvature_2, 0, "tangential"),
            Abcd.propagation(air_half),
            Abcd.planar_interface(1, refractive_index),
            Abcd.propagation(crystal_length),
            Abcd.planar_interface(refractive_index, 1),
            Abcd.propagation(air_half),
            Abcd.mirror_from_curvature(curvature_1, 0, "tangential"),
        )

    @staticmethod
    def hemilithic_roundtrip(
        air_gap,
        crystal_length,
        mirror_radius,
        crystal_surface_radius,
        refractive_index,
    ):
        """Round-trip matrix for a hemilithic cavity.

        Reference plane is the external mirror just after reflection. The first
        radius is the external mirror and the second is the crystal end facet.
        """
        validate_nonnegative(air_gap, "air_gap")
        return Abcd.chain(
            Abcd.propagation(air_gap),
            Abcd.planar_interface(1, refractive_index),
            Abcd.propagation(crystal_length),
            Abcd.mirror(crystal_surface_radius, 0, "tangential"),
            Abcd.propagation(crystal_length),
            Abcd.planar_interface(refractive_index, 1),
            Abcd.propagation(air_gap),
            Abcd.mirror(mirror_radius, 0, "tangential"),
        )

    @staticmethod
    def hemilithic_roundtrip_from_curvature(
        air_gap,
        crystal_length,
        mirror_curvature,
        crystal_surface_curvature,
        refractive_index,
    ):
        """Round-trip matrix for a hemilithic cavity using curvatures."""
        validate_nonnegative(air_gap, "air_gap")
        return Abcd.chain(
            Abcd.propagation(air_gap),
            Abcd.planar_interface(1, refractive_index),
            Abcd.propagation(crystal_length),
            Abcd.mirror_from_curvature(crystal_surface_curvature, 0, "tangential"),
            Abcd.propagation(crystal_length),
            Abcd.planar_interface(refractive_index, 1),
            Abcd.propagation(air_gap),
            Abcd.mirror_from_curvature(mirror_curvature, 0, "tangential"),
        )

    @staticmethod
    def monolithic_roundtrip(
        crystal_length,
        refractive_index,
        radius_1,
        radius_2,
    ):
        """Round-trip matrix for a monolithic two-facet crystal cavity.

        The reference plane is facet 1 on the crystal side just after
        reflection. ``radius_1`` and ``radius_2`` are the two coated crystal
        facets under the mirror reflection convention; ``np.inf`` is planar.
        """
        validate_nonnegative(crystal_length, "crystal_length")
        return Abcd.chain(
            Abcd.planar_interface(refractive_index, refractive_index),
            Abcd.propagation(crystal_length),
            Abcd.mirror(radius_2, 0, "tangential"),
            Abcd.propagation(crystal_length),
            Abcd.mirror(radius_1, 0, "tangential"),
        )

    @staticmethod
    def monolithic_roundtrip_from_curvature(
        crystal_length,
        refractive_index,
        curvature_1,
        curvature_2,
    ):
        """Round-trip matrix for a monolithic cavity using facet curvatures."""
        validate_nonnegative(crystal_length, "crystal_length")
        return Abcd.chain(
            Abcd.planar_interface(refractive_index, refractive_index),
            Abcd.propagation(crystal_length),
            Abcd.mirror_from_curvature(curvature_2, 0, "tangential"),
            Abcd.propagation(crystal_length),
            Abcd.mirror_from_curvature(curvature_1, 0, "tangential"),
        )

    @staticmethod
    def triangle_roundtrip(
        width,
        height,
        crystal_length,
        radius_1,
        radius_2,
        refractive_index,
        plane: str = "sagittal",
    ):
        """Round-trip matrix for a triangular cavity with crystal in base arm."""
        validate_plane(plane)
        diagonal = sp.sqrt((width / 2) ** 2 + height**2)
        fold_angle = sp.asin(height / diagonal)
        side_length = (width - crystal_length) / 2
        validate_nonnegative(side_length, "width - crystal_length")
        return Abcd.chain(
            Abcd.propagation(crystal_length / 2),
            Abcd.planar_interface(1, refractive_index),
            Abcd.propagation(side_length),
            Abcd.mirror(radius_1, fold_angle / 2, plane),
            Abcd.propagation(2 * diagonal),
            Abcd.mirror(radius_2, fold_angle / 2, plane),
            Abcd.propagation(side_length),
            Abcd.planar_interface(refractive_index, 1),
            Abcd.propagation(crystal_length / 2),
        )

    @staticmethod
    def triangle_roundtrip_from_curvature(
        width,
        height,
        crystal_length,
        curvature_1,
        curvature_2,
        refractive_index,
        plane: str = "sagittal",
    ):
        """Round-trip matrix for a triangular cavity using mirror curvatures."""
        validate_plane(plane)
        diagonal = sp.sqrt((width / 2) ** 2 + height**2)
        fold_angle = sp.asin(height / diagonal)
        side_length = (width - crystal_length) / 2
        validate_nonnegative(side_length, "width - crystal_length")
        return Abcd.chain(
            Abcd.propagation(crystal_length / 2),
            Abcd.planar_interface(1, refractive_index),
            Abcd.propagation(side_length),
            Abcd.mirror_from_curvature(curvature_1, fold_angle / 2, plane),
            Abcd.propagation(2 * diagonal),
            Abcd.mirror_from_curvature(curvature_2, fold_angle / 2, plane),
            Abcd.propagation(side_length),
            Abcd.planar_interface(refractive_index, 1),
            Abcd.propagation(crystal_length / 2),
        )

    @staticmethod
    def build(geometry: str, **kwargs):
        """Build a round-trip matrix for a named geometry."""
        if geometry == "bowtie":
            r1 = kwargs.get("radius_1", kwargs.get("radius_of_curvature"))
            r2 = kwargs.get("radius_2", kwargs.get("radius_of_curvature"))
            if r1 is None or r2 is None:
                raise ValueError("bowtie geometry requires radius_1/radius_2 or radius_of_curvature")
            return CavityAbcdBuilder.bowtie_roundtrip(
                kwargs["long_axis"],
                kwargs["short_axis"],
                kwargs["crystal_length"],
                r1,
                r2,
                kwargs["refractive_index"],
                kwargs["incidence_angle"],
                kwargs.get("plane", "sagittal"),
            )
        if geometry == "linear":
            r1 = kwargs.get("radius_1", kwargs.get("radius_of_curvature"))
            r2 = kwargs.get("radius_2", kwargs.get("radius_of_curvature"))
            if r1 is None or r2 is None:
                raise ValueError("linear geometry requires radius_1/radius_2 or radius_of_curvature")
            return CavityAbcdBuilder.linear_roundtrip(
                kwargs["cavity_length"],
                kwargs["crystal_length"],
                r1,
                r2,
                kwargs["refractive_index"],
            )
        if geometry == "hemilithic":
            r1 = kwargs.get("radius_1", kwargs.get("mirror_radius", kwargs.get("radius_of_curvature")))
            r2 = kwargs.get("radius_2", kwargs.get("crystal_surface_radius", kwargs.get("radius_of_curvature")))
            if r1 is None or r2 is None:
                raise ValueError("hemilithic geometry requires radius_1/radius_2 or radius_of_curvature")
            return CavityAbcdBuilder.hemilithic_roundtrip(
                kwargs["air_gap"],
                kwargs["crystal_length"],
                r1,
                r2,
                kwargs["refractive_index"],
            )
        if geometry == "monolithic":
            r1 = kwargs.get("radius_1", kwargs.get("radius_of_curvature"))
            r2 = kwargs.get("radius_2", kwargs.get("mirror_radius", kwargs.get("radius_of_curvature")))
            if r1 is None or r2 is None:
                raise ValueError("monolithic geometry requires radius_1/radius_2 or radius_of_curvature")
            return CavityAbcdBuilder.monolithic_roundtrip(
                kwargs["crystal_length"],
                kwargs["refractive_index"],
                r1,
                r2,
            )
        if geometry == "triangle":
            r1 = kwargs.get("radius_1", kwargs.get("radius_of_curvature"))
            r2 = kwargs.get("radius_2", kwargs.get("radius_of_curvature"))
            if r1 is None or r2 is None:
                raise ValueError("triangle geometry requires radius_1/radius_2 or radius_of_curvature")
            return CavityAbcdBuilder.triangle_roundtrip(
                kwargs["width"],
                kwargs["height"],
                kwargs["crystal_length"],
                r1,
                r2,
                kwargs["refractive_index"],
                kwargs.get("plane", "sagittal"),
            )
        raise ValueError("geometry must be 'bowtie', 'linear', 'triangle', 'hemilithic', or 'monolithic'")


__all__ = [
    "Abcd",
    "validate_nonnegative",
    "validate_radius_of_curvature",
    "radius_to_curvature",
    "validate_plane",
    "bowtie_diagonal",
    "CavityAbcdBuilder",
]
