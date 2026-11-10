"""Fixed-direction tangential ECD sensitivity check for revision v2.

Runs four prescribed source-moment orientations at matched off-grid locations
and noise realizations. Results are written only to the v2 outputs directory.
"""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

import ecd_robustness_v2 as ecd
import model


SEED = 20261024
AGES = [12, 120, 216]
LAYOUTS = ["age-adaptive", "adult-helmet"]
K = 41
TRIALS = 50
ORIENTATIONS_DEG = [0, 45, 90, 135]
OUT = Path(__file__).resolve().parents[1] / "outputs" / "ecd_robustness_v2"


def main() -> None:
    candidate = ecd.candidate_directions()
    rows: list[dict] = []
    for age in AGES:
        for layout in LAYOUTS:
            sensor_dirs, sensor_pos = ecd.sensor_geometry(age, layout, K)
            candidate_lf = ecd.arbitrary_leadfield(candidate, sensor_dirs, sensor_pos, age)
            whitener = np.eye(K) / ecd.SIGMA_FT
            for trial in range(TRIALS):
                rng_truth = np.random.default_rng(np.random.SeedSequence([SEED, age, trial, 90]))
                rng_noise = np.random.default_rng(np.random.SeedSequence([SEED, age, trial, 91]))
                source_dir, source_rho, source_phi = ecd.sample_cap(rng_truth)
                e1, e2 = ecd.tangent_basis(source_dir)
                noise = ecd.SIGMA_FT * rng_noise.standard_normal(K)
                source_lf = ecd.arbitrary_single_source_leadfield(source_dir, sensor_dirs, sensor_pos, age)
                source_radius_mm = model.geometry(age).cortex_radius_mm
                for orientation_deg in ORIENTATIONS_DEG:
                    angle = np.deg2rad(orientation_deg)
                    q_coeff = ecd.Q_NAM * np.array([np.cos(angle), np.sin(angle)])
                    y = source_lf @ q_coeff + noise
                    best, coef, residual, fit_ok = ecd.fit_grid(y, candidate_lf, whitener)
                    est_dir = candidate[best]
                    err_mm = float(source_radius_mm * np.linalg.norm(source_dir - est_dir))
                    rows.append({
                        "age_months": age,
                        "layout": layout,
                        "K": K,
                        "trial": trial,
                        "orientation_deg": orientation_deg,
                        "source_rho": source_rho,
                        "source_phi": source_phi,
                        "error_mm": err_mm,
                        "fit_ok": int(fit_ok),
                    })

    trial_path = OUT / "orientation_sensitivity_trials.csv"
    with trial_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    summary: list[dict] = []
    for age in AGES:
        for layout in LAYOUTS:
            for orientation_deg in ORIENTATIONS_DEG:
                vals = np.asarray([
                    row["error_mm"] for row in rows
                    if row["age_months"] == age and row["layout"] == layout
                    and row["orientation_deg"] == orientation_deg
                ])
                summary.append({
                    "age_months": age,
                    "layout": layout,
                    "K": K,
                    "orientation_deg": orientation_deg,
                    "trials": len(vals),
                    "median_error_mm": float(np.median(vals)),
                    "p90_error_mm": float(np.percentile(vals, 90)),
                    "mean_error_mm": float(np.mean(vals)),
                })
    summary_path = OUT / "orientation_sensitivity_summary.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)

    print(f"Wrote {len(rows)} trials and {len(summary)} summaries to {OUT}")
    print(f"Fit failures: {sum(1 - row['fit_ok'] for row in rows)}")


if __name__ == "__main__":
    main()
