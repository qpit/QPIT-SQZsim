"""Shared helpers for organizing simulation results on disk."""

from __future__ import annotations

from pathlib import Path


RESULT_SUBDIRS = ("cavity", "crystal", "opo")


def _default_results_root() -> Path:
    return Path(__file__).resolve().parents[2] / "results"


def get_results_dir(geometry: str, results_root: str | Path | None = None) -> Path:
    """Return the base results directory for a given geometry."""
    if results_root is None:
        results_root = _default_results_root()
    path = Path(results_root) / geometry
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_geometry_results_dir(geometry: str, results_root: str | Path | None = None) -> Path:
    """Backward-compatible alias for the geometry results directory."""
    return get_results_dir(geometry, results_root=results_root)


def ensure_geometry_results_subdirs(geometry: str, results_root: str | Path | None = None) -> Path:
    """Create the standard cavity/crystal/opo subdirectories for a geometry."""
    geometry_dir = get_results_dir(geometry, results_root=results_root)
    for subdir_name in RESULT_SUBDIRS:
        (geometry_dir / subdir_name).mkdir(parents=True, exist_ok=True)
    return geometry_dir


def get_cavity_results_dir(geometry: str, results_root: str | Path | None = None) -> Path:
    """Return the cavity results directory for a given geometry."""
    return get_geometry_results_subdir(geometry, "cavity", results_root=results_root)


def get_crystal_results_dir(geometry: str, results_root: str | Path | None = None) -> Path:
    """Return the crystal results directory for a given geometry."""
    return get_geometry_results_subdir(geometry, "crystal", results_root=results_root)


def get_opo_results_dir(geometry: str, results_root: str | Path | None = None) -> Path:
    """Return the OPO results directory for a given geometry."""
    return get_geometry_results_subdir(geometry, "opo", results_root=results_root)


def get_geometry_results_subdir(
    geometry: str,
    subdir_name: str,
    results_root: str | Path | None = None,
) -> Path:
    """Return one of the standard results subdirectories for a geometry."""
    geometry_dir = ensure_geometry_results_subdirs(geometry, results_root=results_root)
    return geometry_dir / subdir_name
