"""Shared geometry and forward model for the pediatric OPM-MEG simulations."""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import math
import numpy as np
from numpy.typing import NDArray

K_LEVELS = [5, 13, 25, 41, 61, 85, 113]
LOBES = ["frontal", "left temporal", "right temporal", "central", "parietal", "occipital"]

HEAD_RADIUS_MM = {0: 54.4, 3: 63.7, 6: 68.1, 12: 72.4, 18: 74.5, 24: 76.0, 48: 79.2, 72: 79.6, 96: 80.9, 120: 82.3, 144: 83.7, 168: 84.9, 192: 86.0, 216: 86.5}
CORTEX_OFFSET_MM = {0: 6.5, 3: 7.6, 6: 8.2, 12: 8.7, 18: 8.9, 24: 9.1, 48: 9.5, 72: 9.6, 96: 9.7, 120: 9.9, 144: 10.1, 168: 10.2, 192: 10.3, 216: 10.4}
DEFAULT_SOURCE_DEPTH_MM = 9.0
OPM_STANDOFF_MM = 4.0
RIGID_STANDOFF_MM = 5.0
BASELINE_CELL_DIAMETER_MM = 10.0
NOISE_FT = 250.0
Q_AM = 10e-9
MU0_OVER_4PI = 1.0e-7
FT_PER_T = 1.0e15

# Lambert equal-area projection radii.  A single value is used for every K so
# the relative angular template is directly comparable across ages and layouts.
SOURCE_LAMBERT_RADIUS = 1.52
SENSOR_LAMBERT_RADIUS = 1.28
SOURCE_RING_COUNT = 6

@dataclass(frozen=True)
class Geometry:
    age: int
    scalp_radius_mm: float
    cortex_radius_mm: float


def geometry(age: int, adult_source_depth_mm: float = DEFAULT_SOURCE_DEPTH_MM) -> Geometry:
    scalp = HEAD_RADIUS_MM[int(age)]
    cortex_offset = CORTEX_OFFSET_MM[int(age)]
    real_source_depth_mm = adult_source_depth_mm * (scalp-cortex_offset)/(HEAD_RADIUS_MM[216]-CORTEX_OFFSET_MM[216])
    return Geometry(age=int(age), scalp_radius_mm=scalp, cortex_radius_mm=scalp - cortex_offset - real_source_depth_mm)

def normalize(v: NDArray[np.float64]) -> NDArray[np.float64]:
    v = np.asarray(v, dtype=float)
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / np.maximum(n, 1e-15)


def lambert_inverse(xy: NDArray[np.float64]) -> NDArray[np.float64]:
    xy = np.asarray(xy, dtype=float)
    x = xy[:, 0]
    y = xy[:, 1]
    rho2 = x * x + y * y
    z = 1.0 - 0.5 * rho2
    s = np.sqrt(np.maximum(0.0, 1.0 - 0.25 * rho2))
    dirs = np.column_stack([x * s, y * s, z])
    return normalize(dirs)


def ring_xy(max_radius: float, ring_count: int, points_per_first_ring: int = 4) -> NDArray[np.float64]:
    """Concentric ring grid with exact fourfold symmetry."""
    pts: list[tuple[float, float]] = [(0.0, 0.0)]
    for j in range(1, ring_count + 1):
        r = max_radius * j / ring_count
        n = points_per_first_ring * j
        for t in range(n):
            th = 2.0 * math.pi * t / n
            # Avoid tiny roundoff asymmetries on the symmetry axes.
            x = 0.0 if abs(math.cos(th)) < 1e-14 else r * math.cos(th)
            y = 0.0 if abs(math.sin(th)) < 1e-14 else r * math.sin(th)
            pts.append((float(x), float(y)))
    arr = np.asarray(pts, dtype=float)
    order = np.lexsort((arr[:, 0], arr[:, 1]))
    # Keep the center first for clarity in display and sensor construction.
    center = np.where(np.linalg.norm(arr, axis=1) < 1e-12)[0][0]
    rest = [i for i in order if i != center]
    return arr[[center] + rest]


def source_ring_xy() -> NDArray[np.float64]:
    # More source points than sensor points, with rings of 8,16,...,56 around a center.
    pts = [(0.0, 0.0)]
    for j in range(1, SOURCE_RING_COUNT + 1):
        r = SOURCE_LAMBERT_RADIUS * j / SOURCE_RING_COUNT
        n = 8 * j
        for t in range(n):
            th = 2.0 * math.pi * t / n
            x = 0.0 if abs(math.cos(th)) < 1e-14 else r * math.cos(th)
            y = 0.0 if abs(math.sin(th)) < 1e-14 else r * math.sin(th)
            pts.append((float(x), float(y)))
    return np.asarray(pts, dtype=float)


def classify_source_xy(x: float, y: float) -> str:
    # Coordinates are top-view Lambert projection: y>0 anterior/frontal, y<0 posterior.
    # The sectors partition one continuous whole-head grid rather than disconnected patches.
    r = math.hypot(x, y)
    if r < 0.26:
        return "central"
    if y >= 0.62:
        return "frontal"
    if y <= -0.88:
        return "occipital"
    if abs(x) <= 0.40 and -0.48 <= y <= 0.52:
        return "central"
    if x <= -0.56 and -0.62 <= y <= 0.48:
        return "left temporal"
    if x >= 0.56 and -0.62 <= y <= 0.48:
        return "right temporal"
    return "parietal"


@lru_cache(maxsize=None)
def source_grid() -> tuple[NDArray[np.float64], NDArray[np.float64], tuple[str, ...]]:
    xy = source_ring_xy()
    dirs = lambert_inverse(xy)
    labels = tuple(classify_source_xy(float(x), float(y)) for x, y in xy)
    counts = {l: labels.count(l) for l in LOBES}
    if any(v == 0 for v in counts.values()):
        raise RuntimeError(f"empty source sector: {counts}")
    return dirs, xy, labels


def ring_count_for_k(k: int) -> int:
    # K = 1 + 4(1+2+...+m) = 1 + 2m(m+1)
    if k not in K_LEVELS:
        raise ValueError(f"Unsupported high-symmetry channel count {k}")
    for m in range(1, 20):
        if 1 + 2 * m * (m + 1) == k:
            return m
    raise ValueError(k)


@lru_cache(maxsize=None)
def sensor_template_xy(k: int) -> NDArray[np.float64]:
    m = ring_count_for_k(k)
    return ring_xy(SENSOR_LAMBERT_RADIUS, m, points_per_first_ring=4)


@lru_cache(maxsize=None)
def sensor_template_dirs(k: int) -> NDArray[np.float64]:
    return lambert_inverse(sensor_template_xy(k))


def source_orientations(source_dirs: NDArray[np.float64]) -> NDArray[np.float64]:
    zaxis = np.array([0.0, 0.0, 1.0])
    xaxis = np.array([1.0, 0.0, 0.0])
    out = []
    for n in source_dirs:
        e1 = np.cross(zaxis, n)
        if np.linalg.norm(e1) < 0.08:
            e1 = np.cross(xaxis, n)
        e1 = e1 / np.linalg.norm(e1)
        e2 = np.cross(n, e1)
        e2 = e2 / np.linalg.norm(e2)
        # Mildly mixed tangential orientation to avoid a symmetry-induced zero in top-view maps.
        q = 0.84 * e1 + 0.16 * e2
        out.append(q / np.linalg.norm(q))
    return np.asarray(out, dtype=float)


def tangent_basis(n: NDArray[np.float64]) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    n = normalize(np.asarray(n, dtype=float).reshape(1, 3))[0]
    ref = np.array([0.0, 0.0, 1.0])
    if abs(float(np.dot(ref, n))) > 0.90:
        ref = np.array([0.0, 1.0, 0.0])
    e1 = ref - np.dot(ref, n) * n
    e1 = e1 / max(np.linalg.norm(e1), 1e-15)
    e2 = np.cross(n, e1)
    e2 = e2 / max(np.linalg.norm(e2), 1e-15)
    return e1, e2


def sensor_radius_mm(age: int, layout: str) -> float:
    if layout == "adult-helmet":
        return HEAD_RADIUS_MM[216] + RIGID_STANDOFF_MM
    return HEAD_RADIUS_MM[int(age)] + OPM_STANDOFF_MM


def aperture_offsets(direction: NDArray[np.float64], diameter_mm: float) -> NDArray[np.float64]:
    if diameter_mm <= 0:
        return np.zeros((1, 3), dtype=float)
    n = normalize(np.asarray(direction).reshape(1, 3))[0]
    e1, e2 = tangent_basis(n)
    a = 0.5 * diameter_mm * 1.0e-3
    offsets = [np.zeros(3)]
    for j in range(8):
        ph = 2.0 * math.pi * j / 8.0
        offsets.append(0.70 * a * (math.cos(ph) * e1 + math.sin(ph) * e2))
    return np.asarray(offsets, dtype=float)


def compute_leadfield(age: int, layout: str, sensor_dirs: NDArray[np.float64],
                      cell_diameter_mm: float = BASELINE_CELL_DIAMETER_MM,
                      adult_source_depth_mm: float = DEFAULT_SOURCE_DEPTH_MM) -> NDArray[np.float64]:
    sdirs, _, _ = source_grid()
    geom = geometry(age, adult_source_depth_mm)
    rs = sdirs * (geom.cortex_radius_mm * 1.0e-3)
    qvec = source_orientations(sdirs) * Q_AM
    rad = sensor_radius_mm(age, layout) * 1.0e-3
    out = np.empty((len(sensor_dirs), len(sdirs)), dtype=float)
    for j, n in enumerate(sensor_dirs):
        center = n * rad
        offs = aperture_offsets(n, cell_diameter_mm)
        accum = np.zeros(len(sdirs), dtype=float)
        for off in offs:
            rr = center + off - rs
            dist = np.linalg.norm(rr, axis=1)
            B = MU0_OVER_4PI * np.cross(qvec, rr) / np.maximum(dist[:, None] ** 3, 1e-30)
            accum += B @ n
        out[j, :] = (accum / len(offs)) * FT_PER_T
    return out


def source_positions_mm(age: int, adult_source_depth_mm: float = DEFAULT_SOURCE_DEPTH_MM) -> NDArray[np.float64]:
    sdirs, _, _ = source_grid()
    return sdirs * geometry(age, adult_source_depth_mm).cortex_radius_mm


def source_distance_matrix_mm(age: int, adult_source_depth_mm: float = DEFAULT_SOURCE_DEPTH_MM) -> NDArray[np.float64]:
    pos = source_positions_mm(age, adult_source_depth_mm)
    diff = pos[:, None, :] - pos[None, :, :]
    return np.linalg.norm(diff, axis=2)


def posterior_metrics(L: NDArray[np.float64], age: int, noise_ft: float = NOISE_FT,
                      adult_source_depth_mm: float = DEFAULT_SOURCE_DEPTH_MM) -> dict[str, float]:
    D = source_distance_matrix_mm(age, adult_source_depth_mm)
    G = L.T @ L
    n2 = np.diag(G)
    delta2 = np.maximum(n2[:, None] + n2[None, :] - 2.0 * G, 0.0) / (2.0 * noise_ft * noise_ft)
    logits = -delta2
    logits -= np.max(logits, axis=1, keepdims=True)
    P = np.exp(logits)
    P /= np.maximum(np.sum(P, axis=1, keepdims=True), 1e-300)
    exp_err = np.sum(P * D, axis=1)
    mass20 = np.sum(P * (D <= 20.0), axis=1)
    mass30 = np.sum(P * (D <= 30.0), axis=1)
    # Credible radii of the matched-field posterior, evaluated on the discrete source dictionary.
    r50 = []
    r90 = []
    for i in range(P.shape[0]):
        order = np.argsort(D[i])
        cdf = np.cumsum(P[i, order])
        r50.append(float(D[i, order[min(np.searchsorted(cdf, 0.50), len(order)-1)]]))
        r90.append(float(D[i, order[min(np.searchsorted(cdf, 0.90), len(order)-1)]]))
    r50 = np.asarray(r50, dtype=float)
    r90 = np.asarray(r90, dtype=float)
    col_norm = np.sqrt(np.maximum(n2, 0.0))
    # Effective rank from the nonzero singular values, evaluated through the
    # smaller sensor-space Gram matrix to avoid backend-specific SVD stalls.
    gram_rows = (L / noise_ft) @ (L / noise_ft).T
    evals = np.linalg.eigvalsh(0.5 * (gram_rows + gram_rows.T))
    sv = np.sqrt(np.maximum(evals, 1e-18))
    sv = sv[sv > 1e-12]
    if sv.size == 0:
        erank = 0.0
    else:
        psv = sv / np.sum(sv)
        erank = float(np.exp(-np.sum(psv * np.log(psv))))
    X = L / np.maximum(col_norm[None, :], 1e-15)
    C = np.clip(np.abs(X.T @ X), 0.0, 1.0)
    np.fill_diagonal(C, -np.inf)
    nearest_corr = float(np.median(np.max(C, axis=1)))
    Sep = np.sqrt(np.maximum(n2[:, None] + n2[None, :] - 2.0 * G, 0.0)) / max(noise_ft, 1e-15)
    np.fill_diagonal(Sep, np.inf)
    nearest_sep = float(np.median(np.min(Sep, axis=1)))
    return {
        "median_error": float(np.median(exp_err)),
        "p90_error": float(np.percentile(exp_err, 90)),
        "mean_error": float(np.mean(exp_err)),
        "success20": float(100.0 * np.mean(mass20)),
        "success30": float(100.0 * np.mean(mass30)),
        "median_r50": float(np.median(r50)),
        "median_r90": float(np.median(r90)),
        "snr_median": float(np.median(col_norm / noise_ft)),
        "effective_rank": erank,
        "nearest_corr": nearest_corr,
        "nearest_sep": nearest_sep,
        "resolution_length": float(np.median(exp_err)),
    }
