"""Off-grid ECD localization and finite robustness checks.

The forward model is imported from ``model.py`` so the source and sensor
geometry remain identical to the main model.  A continuous source is sampled
in the Lambert spherical cap, and localization uses a separate dense
candidate grid with a two-component tangential ECD fit.  The script records
every trial and a grouped summary.
"""
from __future__ import annotations

import csv
import math
from pathlib import Path
import time

import numpy as np

import model


SEED = 20261024
SIGMA_FT = 250.0
Q_NAM = 10.0
CAP_RHO = model.SOURCE_LAMBERT_RADIUS
CELL_MM = model.BASELINE_CELL_DIAMETER_MM
AGES = [12, 48, 120, 216]
KS = [25, 41, 61]
LAYOUTS = ["age-adaptive", "adult-helmet"]
MC_TRIALS = 100
ROBUST_TRIALS = 100
N_CANDIDATE_RINGS = 20


def normalize_rows(x: np.ndarray) -> np.ndarray:
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-15)


def tangent_basis(n: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Stable orthonormal basis for the tangent plane at a unit direction."""
    n = np.asarray(n, dtype=float)
    ref = np.array([0.0, 0.0, 1.0])
    if abs(float(np.dot(ref, n))) > 0.90:
        ref = np.array([0.0, 1.0, 0.0])
    e1 = ref - np.dot(ref, n) * n
    e1 /= max(np.linalg.norm(e1), 1e-15)
    e2 = np.cross(n, e1)
    e2 /= max(np.linalg.norm(e2), 1e-15)
    return e1, e2


def sample_cap(rng: np.random.Generator) -> tuple[np.ndarray, float, float]:
    """Uniform spherical-cap direction via the equal-area Lambert disk."""
    rho = CAP_RHO * math.sqrt(float(rng.uniform()))
    phi = float(rng.uniform(0.0, 2.0 * math.pi))
    xy = np.array([[rho * math.cos(phi), rho * math.sin(phi)]], dtype=float)
    return model.lambert_inverse(xy)[0], rho, phi


def candidate_directions() -> np.ndarray:
    """Independent dense grid inside the source cap (921 locations)."""
    pts = [(0.0, 0.0)]
    # Equal-area radial nodes and increasing azimuth counts avoid the six-ring
    # the 169-point dictionary; an explicit boundary ring covers cap-edge truths too.
    for ir in range(N_CANDIDATE_RINGS):
        rho = CAP_RHO * math.sqrt((ir + 0.5) / N_CANDIDATE_RINGS)
        n_az = 4 * (ir + 1)
        for ia in range(n_az):
            phi = 2.0 * math.pi * ia / n_az + (ir % 2) * math.pi / n_az
            pts.append((rho * math.cos(phi), rho * math.sin(phi)))
    for ia in range(4 * N_CANDIDATE_RINGS):
        phi = 2.0 * math.pi * ia / (4 * N_CANDIDATE_RINGS)
        pts.append((CAP_RHO * math.cos(phi), CAP_RHO * math.sin(phi)))
    return model.lambert_inverse(np.asarray(pts, dtype=float))


def rotate_directions(dirs: np.ndarray, rotvec: np.ndarray) -> np.ndarray:
    """Apply small independent rotation vectors to direction rows."""
    out = np.empty_like(dirs)
    for i, (n, w) in enumerate(zip(dirs, rotvec)):
        a = float(np.linalg.norm(w))
        if a < 1e-14:
            out[i] = n
            continue
        u = w / a
        out[i] = n * math.cos(a) + np.cross(u, n) * math.sin(a) + u * np.dot(u, n) * (1.0 - math.cos(a))
    return normalize_rows(out)


def arbitrary_leadfield(
    source_dirs: np.ndarray,
    sensor_dirs: np.ndarray,
    sensor_pos_m: np.ndarray,
    age: int,
    cell_mm: float = CELL_MM,
) -> np.ndarray:
    """Return K x N x 2 lead fields for unit tangential moments in nA m."""
    source_dirs = normalize_rows(np.asarray(source_dirs, dtype=float))
    sensor_dirs = normalize_rows(np.asarray(sensor_dirs, dtype=float))
    sensor_pos_m = np.asarray(sensor_pos_m, dtype=float)
    rs = source_dirs * (model.geometry(age).cortex_radius_mm * 1e-3)
    qbases = np.asarray([tangent_basis(n) for n in source_dirs])
    # Unit nA m moments; the Biot-Savart kernel converts Tesla to fT.
    qbases = qbases * 1e-9
    out = np.empty((len(sensor_dirs), len(source_dirs), 2), dtype=float)
    for j, n in enumerate(sensor_dirs):
        center = sensor_pos_m[j]
        offsets = model.aperture_offsets(n, cell_mm)
        accum = np.zeros((len(source_dirs), 2), dtype=float)
        for off in offsets:
            rr = center + off - rs
            dist = np.linalg.norm(rr, axis=1)
            for k in range(2):
                B = model.MU0_OVER_4PI * np.cross(qbases[:, k, :], rr) / np.maximum(dist[:, None] ** 3, 1e-30)
                accum[:, k] += B @ n
        out[j] = (accum / len(offsets)) * model.FT_PER_T
    return out


def arbitrary_single_source_leadfield(source_dir: np.ndarray, sensor_dirs: np.ndarray, sensor_pos_m: np.ndarray,
                                      age: int, cell_mm: float = CELL_MM) -> np.ndarray:
    return arbitrary_leadfield(np.asarray(source_dir, dtype=float).reshape(1, 3), sensor_dirs,
                               sensor_pos_m, age, cell_mm)[:, 0, :]


def sensor_geometry(age: int, layout: str, k: int) -> tuple[np.ndarray, np.ndarray]:
    dirs = model.sensor_template_dirs(k)
    radius = model.sensor_radius_mm(age, layout)
    pos = dirs * radius * 1e-3
    return dirs, pos


def golden_angle_dirs(k: int) -> np.ndarray:
    """Nested 25/41/61 family from a golden-angle sequence on the sensor cap."""
    pts = [(0.0, 0.0)]
    golden = math.pi * (3.0 - math.sqrt(5.0))

    def vdc_base2(index: int) -> float:
        value = 0.0
        denom = 1.0
        while index:
            index, bit = divmod(index, 2)
            denom *= 2.0
            value += bit / denom
        return value

    for i in range(1, k):
        rho = model.SENSOR_LAMBERT_RADIUS * math.sqrt(vdc_base2(i))
        phi = (i - 1) * golden
        pts.append((rho * math.cos(phi), rho * math.sin(phi)))
    return model.lambert_inverse(np.asarray(pts[:k], dtype=float))


def covariance(sensor_pos_m: np.ndarray, rho: float = 0.35, length_mm: float = 35.0) -> np.ndarray:
    d = np.linalg.norm(sensor_pos_m[:, None, :] - sensor_pos_m[None, :, :], axis=2) * 1e3
    corr = np.exp(-(d * d) / (2.0 * length_mm * length_mm))
    corr = (1.0 - rho) * np.eye(len(sensor_pos_m)) + rho * corr
    np.fill_diagonal(corr, 1.0)
    return SIGMA_FT * SIGMA_FT * corr


def white_transform(cov: np.ndarray) -> np.ndarray:
    # C = A A^T; W=A^{-1}, so W C W^T = I.
    return np.linalg.inv(np.linalg.cholesky(cov))


def fit_grid(y: np.ndarray, candidate_lf: np.ndarray, whitener: np.ndarray) -> tuple[int, np.ndarray, float, bool]:
    """Profiled two-component GLS fit across candidate positions."""
    # candidate_lf: K x N x 2
    lw = np.einsum("ij,jnk->ink", whitener, candidate_lf)
    yw = whitener @ y
    gram = np.einsum("ink,inl->nkl", lw, lw)
    det = gram[:, 0, 0] * gram[:, 1, 1] - gram[:, 0, 1] * gram[:, 1, 0]
    ok = det > 1e-14
    inv = np.zeros_like(gram)
    inv[:, 0, 0] = gram[:, 1, 1] / np.maximum(det, 1e-30)
    inv[:, 1, 1] = gram[:, 0, 0] / np.maximum(det, 1e-30)
    inv[:, 0, 1] = inv[:, 1, 0] = -gram[:, 0, 1] / np.maximum(det, 1e-30)
    b = np.einsum("ink,i->nk", lw, yw)
    coef = np.einsum("nkl,nl->nk", inv, b)
    residual = float(np.dot(yw, yw)) - np.einsum("nk,nk->n", b, coef)
    residual[~ok] = np.inf
    best = int(np.argmin(residual))
    return best, coef[best], float(max(residual[best], 0.0)), bool(ok[best])


def run_ecd_trials(
    age: int,
    k: int,
    layout: str,
    trials: int = MC_TRIALS,
    array_name: str = "D4",
    correlated: bool = False,
    mismatch: str = "none",
    rho: float = 0.35,
    length_mm: float = 35.0,
) -> list[dict]:
    if array_name == "D4":
        nominal_dirs, nominal_pos = sensor_geometry(age, layout, k)
    else:
        nominal_dirs = golden_angle_dirs(k)
        nominal_pos = nominal_dirs * model.sensor_radius_mm(age, layout) * 1e-3
    candidate = candidate_directions()
    candidate_lf = arbitrary_leadfield(candidate, nominal_dirs, nominal_pos, age)
    cov = covariance(nominal_pos, rho, length_mm) if correlated else SIGMA_FT * SIGMA_FT * np.eye(k)
    W = white_transform(cov)
    rows: list[dict] = []
    for trial in range(trials):
        # Purpose-specific streams make the same off-grid truths and standard
        # normal noise vectors pair across age-matched array/layout conditions.
        rng_truth = np.random.default_rng(np.random.SeedSequence([SEED, age, trial, 0]))
        rng_noise = np.random.default_rng(np.random.SeedSequence([SEED, age, trial, 1]))
        rng_pos = np.random.default_rng(np.random.SeedSequence([SEED, age, trial, 2]))
        rng_ori = np.random.default_rng(np.random.SeedSequence([SEED, age, trial, 3]))
        source_dir, source_rho, source_phi = sample_cap(rng_truth)
        e1, e2 = tangent_basis(source_dir)
        angle = float(rng_truth.uniform(0.0, 2.0 * math.pi))
        qcoeff = Q_NAM * np.array([math.cos(angle), math.sin(angle)])
        pos_true = nominal_pos.copy()
        dirs_true = nominal_dirs.copy()
        if mismatch in {"position_3mm", "combined_3mm_3deg"}:
            # 3 mm RMS vector displacement, fixed within each trial's array.
            pos_true += rng_pos.normal(0.0, 0.003 / math.sqrt(3.0), size=nominal_pos.shape)
        if mismatch in {"orientation_3deg", "combined_3mm_3deg"}:
            rotvec = np.zeros_like(nominal_dirs)
            for j, n in enumerate(nominal_dirs):
                a, b = tangent_basis(n)
                rotvec[j] = (rng_ori.normal(0.0, math.radians(3.0) / math.sqrt(2.0)) * a
                             + rng_ori.normal(0.0, math.radians(3.0) / math.sqrt(2.0)) * b)
            dirs_true = rotate_directions(nominal_dirs, rotvec)
        # Sensor positions and normals are independently perturbed in the true forward model.
        lf_true = arbitrary_single_source_leadfield(source_dir, dirs_true, pos_true, age)
        signal = lf_true @ qcoeff
        z = rng_noise.standard_normal(k)
        noise = np.linalg.cholesky(cov) @ z
        y = signal + noise
        best, coef, residual, fit_ok = fit_grid(y, candidate_lf, W)
        est_dir = candidate[best]
        angle_err = math.acos(float(np.clip(np.dot(source_dir, est_dir), -1.0, 1.0)))
        source_radius = model.geometry(age).cortex_radius_mm
        err_mm = float(2.0 * source_radius * math.sin(angle_err / 2.0))
        geo_err_mm = float(source_radius * angle_err)
        rows.append({
            "experiment": "ecd",
            "array": array_name,
            "age_months": age,
            "layout": layout,
            "K": k,
            "trial": trial,
            "seed": SEED,
            "noise_model": "correlated" if correlated else "iid",
            "correlation_rho": rho if correlated else 0.0,
            "correlation_length_mm": length_mm if correlated else 0.0,
            "mismatch": mismatch,
            "position_rms_mm": 3.0 if mismatch in {"position_3mm", "combined_3mm_3deg"} else 0.0,
            "orientation_rms_deg": 3.0 if mismatch in {"orientation_3deg", "combined_3mm_3deg"} else 0.0,
            "true_rho": source_rho,
            "true_phi_rad": source_phi,
            "true_orientation_rad": angle,
            "fit_candidate": best,
            "fit_residual_fT2": residual,
            "fit_ok": int(fit_ok),
            "candidate_edge_hit": int(best >= 1 + sum(4 * (i + 1) for i in range(N_CANDIDATE_RINGS))),
            "estimated_q1_nAm": float(coef[0]),
            "estimated_q2_nAm": float(coef[1]),
            "error_mm": err_mm,
            "geodesic_error_mm": geo_err_mm,
        })
    return rows


def summarize(rows: list[dict], experiment: str) -> list[dict]:
    keys = ["array", "age_months", "layout", "K", "noise_model", "mismatch", "correlation_rho", "correlation_length_mm"]
    out = []
    groups: dict[tuple, list[dict]] = {}
    for r in rows:
        key = tuple(r[k] for k in keys)
        groups.setdefault(key, []).append(r)
    for key, vals in groups.items():
        e = np.asarray([v["error_mm"] for v in vals], dtype=float)
        out.append({
            "experiment": experiment,
            **dict(zip(keys, key)),
            "n_trials": len(vals),
            "fit_failure_rate": float(1.0 - np.mean([v["fit_ok"] for v in vals])),
            "candidate_edge_hit_rate": float(np.mean([v["candidate_edge_hit"] for v in vals])),
            "median_error_mm": float(np.median(e)),
            "p90_error_mm": float(np.percentile(e, 90)),
            "median_geodesic_error_mm": float(np.median([v["geodesic_error_mm"] for v in vals])),
            "p90_geodesic_error_mm": float(np.percentile([v["geodesic_error_mm"] for v in vals], 90)),
            "mean_error_mm": float(np.mean(e)),
            "max_error_mm": float(np.max(e)),
        })
    return out


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    started = time.perf_counter()
    root = Path(__file__).resolve().parents[1]
    out = root / "outputs" / "ecd_robustness_v2"
    out.mkdir(parents=True, exist_ok=True)
    all_rows: list[dict] = []
    summaries: list[dict] = []
    # Paired baseline off-grid ECD trials across requested ages, channel counts, and layouts.
    for age in AGES:
        for k in KS:
            for layout in LAYOUTS:
                all_rows.extend(run_ecd_trials(age, k, layout))
        print(f"Completed baseline ECD age={age} months", flush=True)
    summaries.extend(summarize(all_rows, "ecd"))

    # Correlated noise with GLS whitening and independent geometric mismatch experiments.
    robust_rows: list[dict] = []
    robust_summary: list[dict] = []
    for age in [12, 120, 216]:
        for k in [41, 61]:
            for layout in LAYOUTS:
                robust_rows.extend(run_ecd_trials(age, k, layout, ROBUST_TRIALS, correlated=True))
                for mismatch in ["position_3mm", "orientation_3deg", "combined_3mm_3deg"]:
                    robust_rows.extend(run_ecd_trials(age, k, layout, ROBUST_TRIALS, mismatch=mismatch))
        print(f"Completed robustness ECD age={age} months", flush=True)
    robust_summary.extend(summarize([r for r in robust_rows if r["noise_model"] == "correlated"], "correlated_noise"))
    robust_summary.extend(summarize([r for r in robust_rows if r["mismatch"] != "none"], "registration_mismatch"))

    # Nested golden-angle family: matched-field metrics from the 169-point source dictionary.
    nested_rows: list[dict] = []
    for age in [12, 120, 216]:
        for k in KS:
            for layout in LAYOUTS:
                dirs = golden_angle_dirs(k)
                L = model.compute_leadfield(age, layout, dirs, CELL_MM, model.DEFAULT_SOURCE_DEPTH_MM)
                metrics = model.posterior_metrics(L, age)
                nested_rows.append({"array": "golden_nested", "age_months": age, "layout": layout, "K": k,
                                    **metrics})

    write_csv(out / "ecd_trials.csv", all_rows)
    write_csv(out / "ecd_summary.csv", summaries)
    write_csv(out / "robustness_trials.csv", robust_rows)
    write_csv(out / "robustness_summary.csv", robust_summary)
    write_csv(out / "nested_golden_angle_matched_field.csv", nested_rows)
    params = {
        "seed": SEED,
        "sigma_fT": SIGMA_FT,
        "q_nAm": Q_NAM,
        "cell_mm": CELL_MM,
        "ages_months": AGES,
        "K": KS,
        "layouts": LAYOUTS,
        "trials_per_ecd_configuration": MC_TRIALS,
        "trials_per_robust_configuration": ROBUST_TRIALS,
        "candidate_grid_points": len(candidate_directions()),
        "candidate_grid_construction": "center plus 20 equal-area radial rings with 4j azimuth samples per ring, plus 80-point cap boundary",
        "candidate_grid_rings": N_CANDIDATE_RINGS,
        "cap_rho_max": CAP_RHO,
        "nested_sensor_cap_rho_max": model.SENSOR_LAMBERT_RADIUS,
        "nested_sensor_sequence": "center plus van der Corput base-2 radii and golden-angle azimuth; K arrays are prefixes",
        "correlated_noise_rho": 0.35,
        "correlated_noise_length_mm": 35.0,
        "registration_position_rms_mm": 3.0,
        "registration_orientation_rms_deg": 3.0,
        "fit": "profiled two-component tangential ECD GLS residual over independent dense candidate grid",
        "noise_definition": "250 fT per-channel marginal RMS for each simulated sample; correlated covariance whitened with known Cholesky factor",
        "source_sampling": "uniform area in Lambert disk, equivalent to uniform spherical-cap area",
        "source_orientation": "uniform azimuth in the local tangential plane; dipole magnitude 10 nA m",
        "paired_random_streams": "SeedSequence([master_seed, age_months, trial, stream_id]); source, noise, position, and orientation streams are shared across comparable configurations",
    }
    params["elapsed_seconds"] = time.perf_counter() - started
    (out / "parameters.json").write_text(__import__("json").dumps(params, indent=2), encoding="utf-8")
    print(f"Wrote ECD and robustness outputs to {out}")


if __name__ == "__main__":
    main()
