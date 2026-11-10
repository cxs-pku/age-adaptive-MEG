"""Sarvas spherical-conductor consistency check for radial OPM channels.

This standalone script reuses ``model.py`` for the source grid, source
orientations, sensor templates, geometry, and finite-aperture quadrature. It
compares the free-space Biot--Savart radial lead field with the Sarvas
spherical conductor field projected onto the same sensor normal. The script
has no random component and writes one CSV row per
age/layout/channel/aperture combination.

Run from any directory with:
    python sarvas_consistency.py

Optional:
    python sarvas_consistency.py --output ../outputs/sarvas_consistency.csv

The sensor centers are outside the spherical conductor and their axes are
radial. For a point sensor, the Sarvas radial component equals the free-space
Biot--Savart radial component. For finite cells, the same nine-point symmetric
planar-aperture rule used by the main model is applied to both fields.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys

import numpy as np


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import model as sim  # noqa: E402


REPRESENTATIVE_AGES = (12, 48, 120, 216)
CHANNEL_COUNTS = (25, 41, 61, 113)
APERTURE_DIAMETERS_MM = (0, 10, 25)
LAYOUTS = ("age-adaptive", "adult-helmet")


def sarvas_field(
    sensor_position_m: np.ndarray,
    source_positions_m: np.ndarray,
    source_moments_Am: np.ndarray,
) -> np.ndarray:
    """Return Sarvas B vectors [source, xyz] at one external point.

    Formula follows Sarvas (1987), as implemented in the spherical forward
    model of MNE-Python. The spherical-conductor center is the origin.
    """
    r = np.asarray(sensor_position_m, dtype=float)
    r0 = np.asarray(source_positions_m, dtype=float)
    q = np.asarray(source_moments_Am, dtype=float)
    R = float(np.linalg.norm(r))
    if R <= 0:
        raise ValueError("Sensor position must be outside the sphere center.")

    delta = r[None, :] - r0
    a = np.linalg.norm(delta, axis=1)
    if np.any(a <= 0):
        raise ValueError("Sensor integration point coincides with a source.")
    r_dot_r0 = r0 @ r
    ar = R * R - r_dot_r0
    F = a * (R * a + ar)
    grad_coeff_r = a * a / R + ar / a + 2.0 * (a + R)
    grad_coeff_r0 = a + 2.0 * R + ar / a
    grad_F = grad_coeff_r[:, None] * r[None, :] - grad_coeff_r0[:, None] * r0

    q_cross_r0 = np.cross(q, r0)
    q_cross_r0_dot_r = q_cross_r0 @ r
    b = sim.MU0_OVER_4PI * (
        F[:, None] * q_cross_r0
        - q_cross_r0_dot_r[:, None] * grad_F
    ) / (F[:, None] ** 2)
    return b


def sarvas_leadfield(age: int, layout: str, k: int, cell_diameter_mm: float) -> np.ndarray:
    """Build a Sarvas lead field using the model's source/sensor construction."""
    source_dirs, _, _ = sim.source_grid()
    geom = sim.geometry(age)
    source_positions = source_dirs * (geom.cortex_radius_mm * 1e-3)
    source_moments = sim.source_orientations(source_dirs) * sim.Q_AM
    sensor_dirs = sim.sensor_template_dirs(k)
    sensor_radius = sim.sensor_radius_mm(age, layout) * 1e-3
    leadfield = np.empty((len(sensor_dirs), len(source_dirs)), dtype=float)

    for j, normal in enumerate(sensor_dirs):
        center = normal * sensor_radius
        offsets = sim.aperture_offsets(normal, cell_diameter_mm)
        accumulated = np.zeros(len(source_dirs), dtype=float)
        for offset in offsets:
            field = sarvas_field(center + offset, source_positions, source_moments)
            accumulated += field @ normal
        leadfield[j] = accumulated / len(offsets) * sim.FT_PER_T
    return leadfield


def compare_case(age: int, layout: str, k: int, aperture_mm: int) -> dict[str, int | float | str]:
    dirs = sim.sensor_template_dirs(k)
    biot_savart = sim.compute_leadfield(age, layout, dirs, aperture_mm)
    sarvas = sarvas_leadfield(age, layout, k, aperture_mm)
    diff = sarvas - biot_savart
    base_scale = max(float(np.max(np.abs(biot_savart))), np.finfo(float).tiny)

    bs_metrics = sim.posterior_metrics(biot_savart, age)
    sarvas_metrics = sim.posterior_metrics(sarvas, age)
    return {
        "age_months": age,
        "layout": layout,
        "K": k,
        "cell_diameter_mm": aperture_mm,
        "source_depth_parameter_mm": sim.DEFAULT_SOURCE_DEPTH_MM,
        "realized_source_depth_mm": sim.DEFAULT_SOURCE_DEPTH_MM * (
            sim.HEAD_RADIUS_MM[age] - sim.CORTEX_OFFSET_MM[age]
        ) / (sim.HEAD_RADIUS_MM[216] - sim.CORTEX_OFFSET_MM[216]),
        "scalp_radius_mm": sim.HEAD_RADIUS_MM[age],
        "cortex_offset_mm": sim.CORTEX_OFFSET_MM[age],
        "cortex_radius_mm": sim.geometry(age).cortex_radius_mm,
        "sensor_radius_mm": sim.sensor_radius_mm(age, layout),
        "noise_ft": sim.NOISE_FT,
        "dipole_moment_nA_m": sim.Q_AM * 1e9,
        "leadfield_relative_frobenius": float(np.linalg.norm(diff) / np.linalg.norm(biot_savart)),
        "leadfield_max_abs_diff_fT": float(np.max(np.abs(diff))),
        "leadfield_max_entry_relative": float(np.max(np.abs(diff)) / base_scale),
        "median_error_abs_diff_mm": abs(sarvas_metrics["median_error"] - bs_metrics["median_error"]),
        "p90_error_abs_diff_mm": abs(sarvas_metrics["p90_error"] - bs_metrics["p90_error"]),
        "success20_abs_diff_percentage_points": abs(sarvas_metrics["success20"] - bs_metrics["success20"]),
        "median_snr_abs_diff": abs(sarvas_metrics["snr_median"] - bs_metrics["snr_median"]),
        "effective_rank_abs_diff": abs(sarvas_metrics["effective_rank"] - bs_metrics["effective_rank"]),
        "nearest_separation_abs_diff": abs(sarvas_metrics["nearest_sep"] - bs_metrics["nearest_sep"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    default_output = HERE.parent / "outputs" / "sarvas_consistency.csv"
    parser.add_argument("--output", type=Path, default=default_output)
    args = parser.parse_args()

    rows = [
        compare_case(age, layout, k, aperture_mm)
        for age in REPRESENTATIVE_AGES
        for layout in LAYOUTS
        for k in CHANNEL_COUNTS
        for aperture_mm in APERTURE_DIAMETERS_MM
    ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} deterministic Sarvas checks to {args.output}")
    for diameter in APERTURE_DIAMETERS_MM:
        subset = [r for r in rows if r["cell_diameter_mm"] == diameter]
        fro = [r["leadfield_relative_frobenius"] for r in subset]
        print(
            f"d={diameter:2d} mm: relative Frobenius difference "
            f"min/median/max = {min(fro):.3g}/{np.median(fro):.3g}/{max(fro):.3g}"
        )


if __name__ == "__main__":
    main()
