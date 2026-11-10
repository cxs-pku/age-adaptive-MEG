"""Neonatal dHCP atlas source-pattern geometry sensitivity check.

This script uses the publicly released dHCP week-40 symmetric cortical atlas,
not subject-specific MRI or a BEM.  It keeps the Lambert sensor template
and evaluates the Biot--Savart point-sensor kernel for an age-0 and an
adult-radius shell around the atlas pial surface.  The output is therefore an
atlas geometry check rather than a pediatric localization validation.
"""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import nibabel as nib
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
SURF = OUT / "dhcpSym_template"
SEED = 20261026
N_SOURCES = 400
AGES = ((0, "age-adaptive", 58.4), (216, "adult-helmet", 91.5))
K_LEVELS = (25, 41, 61)
Q_AM = 10e-9
MU0_OVER_4PI = 1e-7
FT_PER_T = 1e15
HEAD0_CORTEX_MM = 54.4 - 6.5 - 9.0 * (54.4 - 6.5) / (86.5 - 10.4)
SOURCE_LAMBERT_RADIUS = 1.52
SENSOR_LAMBERT_RADIUS = 1.28


def normalize(x: np.ndarray) -> np.ndarray:
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-15)


def lambert_inverse(xy: np.ndarray) -> np.ndarray:
    x, y = xy[:, 0], xy[:, 1]
    rho2 = x * x + y * y
    z = 1.0 - 0.5 * rho2
    s = np.sqrt(np.maximum(0.0, 1.0 - 0.25 * rho2))
    return normalize(np.column_stack([x * s, y * s, z]))


def ring_xy(max_radius: float, ring_count: int, points_per_first_ring: int = 4) -> np.ndarray:
    pts = [(0.0, 0.0)]
    for j in range(1, ring_count + 1):
        r = max_radius * j / ring_count
        n = points_per_first_ring * j
        pts.extend((r * math.cos(2 * math.pi * t / n), r * math.sin(2 * math.pi * t / n)) for t in range(n))
    return np.asarray(pts, dtype=float)


def sensor_dirs(k: int) -> np.ndarray:
    # k=1+2m(m+1) for the high-symmetry family.
    m = next(m for m in range(1, 20) if 1 + 2 * m * (m + 1) == k)
    return lambert_inverse(ring_xy(SENSOR_LAMBERT_RADIUS, m))


def tangent_basis(n: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    n = n / max(np.linalg.norm(n), 1e-15)
    ref = np.array([0.0, 0.0, 1.0]) if abs(n[2]) < 0.9 else np.array([0.0, 1.0, 0.0])
    e1 = ref - n * float(ref @ n)
    e1 /= max(np.linalg.norm(e1), 1e-15)
    e2 = np.cross(n, e1)
    e2 /= max(np.linalg.norm(e2), 1e-15)
    return e1, e2


def atlas_vertices_and_normals() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    vertices, normals, triangles = [], [], []
    for hemi in ("left", "right"):
        pial = nib.load(SURF / f"week-40_hemi-{hemi}_space-dhcpSym_dens-32k_pial.surf.gii")
        v = np.asarray(pial.darrays[0].data, dtype=float)
        f = np.asarray(pial.darrays[1].data, dtype=np.int64)
        vertices.append(v)
        triangles.append(f)
    x = np.vstack(vertices)
    # Estimate robust source center from the complete pial atlas.
    center = np.mean(x, axis=0)
    # Vertex normals from pial triangles; orienting the sign is unnecessary for
    # random tangent moments, but we orient outward relative to the atlas center.
    nall = np.zeros_like(x)
    off = 0
    for v, f in zip(vertices, triangles):
        tri = v[f]
        fn = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
        # GIFTI triangle indices are local to each hemisphere; offset the
        # right-hemisphere indices into the concatenated vertex array.
        f_global = f + off
        for j in range(3):
            np.add.at(nall, f_global[:, j], fn)
        off += len(v)
    nall = normalize(nall)
    flip = np.sum(nall * (x - center), axis=1) < 0
    nall[flip] *= -1.0
    return x, nall, center


def selected_sources() -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    x, n, center = atlas_vertices_and_normals()
    radial = np.linalg.norm(x - center, axis=1)
    zdir = (x[:, 2] - center[2]) / np.maximum(radial, 1e-15)
    # Restrict the atlas source dictionary to the superior cap sampled by the
    # Lambert template (z >= z at the outer Lambert ring).
    # Match the full source-cap extent (rho=1.52), rather than the smaller
    # sensor footprint (rho=1.28).
    keep = zdir >= float(lambert_inverse(np.array([[SOURCE_LAMBERT_RADIUS, 0.0]]))[0, 2])
    pool = np.flatnonzero(keep)
    rng = np.random.default_rng(SEED)
    idx = np.sort(rng.choice(pool, size=N_SOURCES, replace=False))
    scale = HEAD0_CORTEX_MM / float(np.median(radial[idx]))
    pos = center + (x[idx] - center) * scale
    return pos, n[idx], center, {"atlas_vertices": int(len(x)), "cap_vertices": int(len(pool)), "median_radius_mm": float(np.median(radial[idx])), "scale": float(scale)}


def leadfield(pos: np.ndarray, normals: np.ndarray, shell_radius_mm: float, k: int, center: np.ndarray, seed: int = SEED + 1) -> np.ndarray:
    rng = np.random.default_rng(seed)
    q = np.empty_like(normals)
    for i, n in enumerate(normals):
        e1, e2 = tangent_basis(n)
        a = float(rng.uniform(0.0, 2.0 * math.pi))
        q[i] = math.cos(a) * e1 + math.sin(a) * e2
    out = np.empty((k, len(pos)), dtype=float)
    dirs = sensor_dirs(k)
    rs = pos * 1e-3
    q = q * Q_AM
    for j, d in enumerate(dirs):
        sensor = center + d * shell_radius_mm
        rr = sensor * 1e-3 - rs
        b = MU0_OVER_4PI * np.cross(q, rr) / np.maximum(np.linalg.norm(rr, axis=1)[:, None] ** 3, 1e-30)
        out[j] = (b @ d) * FT_PER_T
    return out


def metrics(matrix: np.ndarray) -> dict[str, float]:
    scale = np.sqrt(np.mean(matrix * matrix, axis=1))
    x = matrix / np.maximum(scale[:, None], np.percentile(scale[scale > 0], 5))
    s = np.linalg.svd(x, compute_uv=False)
    w = s / max(s.sum(), 1e-15)
    rank = float(np.exp(-np.sum(w * np.log(np.maximum(w, 1e-15)))))
    norm = np.linalg.norm(x, axis=0)
    cos = (x.T @ x) / np.maximum(norm[:, None] * norm[None, :], 1e-15)
    np.fill_diagonal(cos, -np.inf)
    dist = np.sqrt(np.maximum(2.0 - 2.0 * np.max(cos, axis=1), 0.0))
    return {"effective_rank": rank, "median_nearest_normalized_distance": float(np.median(dist)), "p10_nearest_normalized_distance": float(np.percentile(dist, 10))}


def main() -> None:
    pos, normals, center, atlas_meta = selected_sources()
    rows = []
    for age, layout, shell in AGES:
        for k in K_LEVELS:
            row = {"age_months": age, "layout": layout, "K": k, "shell_radius_mm": shell, "source_count": len(pos), "source_mode": "dHCP_GA40_pial_cap"}
            row.update(metrics(leadfield(pos, normals, shell, k, center)))
            rows.append(row)
    with (OUT / "dhcp_atlas_geometry_summary.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    meta = {"dataset": "dHCP symmetric week-40 cortical atlas", "source_url": "https://biomedic.doc.ic.ac.uk/brain-development/downloads/dhcpSym_template.zip", "seed": SEED, "source_count": N_SOURCES, "normalization": "per-channel RMS across selected atlas source patterns", "shells": {"age-adaptive": 58.4, "adult-helmet": 91.5}, "cortex_radius_mm": HEAD0_CORTEX_MM, **atlas_meta, "interpretation": "neonatal MRI-derived atlas geometry sensitivity check; no BEM, raw MEG, empirical noise, or localization accuracy"}
    with (OUT / "dhcp_atlas_geometry_parameters.json").open("w", encoding="utf-8") as f: json.dump(meta, f, indent=2)
    print("Wrote dHCP atlas geometry summary")


if __name__ == "__main__":
    main()
