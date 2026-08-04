from __future__ import annotations

import math


def rectangular_channel_hydraulic_resistance(width_m, height_m, length_m, viscosity_pa_s):
    """Approximation for width >= height; preferable to circular-pipe Hagen–Poiseuille."""
    if min(width_m, height_m, length_m, viscosity_pa_s) <= 0:
        raise ValueError("Geometry and viscosity must be positive")
    w, h = max(width_m, height_m), min(width_m, height_m)
    correction = 1 - 0.630 * h / w
    return 12 * viscosity_pa_s * length_m / (w * h**3 * correction)


def percent_difference(simulated, experimental):
    if experimental == 0:
        return math.nan
    return 100 * abs(simulated - experimental) / abs(experimental)


def fdtd_manifest(parameters: dict) -> dict:
    required = {"solver", "mesh_nm", "boundary_conditions", "gold_model",
                "graphene_model", "geometry", "wavelength_range_nm"}
    missing = required - set(parameters)
    if missing:
        raise ValueError(f"FDTD manifest incomplete: {sorted(missing)}")
    return {"status": "parameters_validated_not_executed", **parameters}
