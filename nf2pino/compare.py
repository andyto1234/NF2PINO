"""Load NF2 extrapolation cubes and compare them to PINO exports on a common grid."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from scipy.interpolate import RegularGridInterpolator

from .metrics import field_quality_metrics


def _linear_map_to_range(values: np.ndarray, lo: float, hi: float) -> np.ndarray:
    """Map ``values`` affinely from its own min–max to ``[lo, hi]`` (per-axis domain alignment)."""
    arr = np.asarray(values, dtype=np.float64)
    mn, mx = float(arr.min()), float(arr.max())
    if mx <= mn:
        return np.full_like(arr, 0.5 * (lo + hi))
    s = (arr - mn) / (mx - mn)
    return lo + s * (hi - lo)


def load_nf2_b_cartesian(
    nf2_path: str | Path,
    *,
    device: str | torch.device = "cpu",
    progress: bool = True,
    compute_jacobian: bool = False,
) -> dict[str, Any]:
    """Evaluate **B** on the native Cartesian volume grid stored in ``extrapolation_result.nf2``."""
    from nf2.evaluation.output import CartesianOutput

    path = Path(nf2_path)
    if not path.is_file():
        raise FileNotFoundError(f"NF2 result not found: {path}")

    out_h = CartesianOutput(str(path), device=torch.device(device))
    cube = out_h.load_cube(progress=progress, compute_jacobian=compute_jacobian)
    b = np.asarray(cube["b"], dtype=np.float32)
    cr = cube["coords"]
    x_ds = np.asarray(cr[:, 0, 0, 0], dtype=np.float64)
    y_ds = np.asarray(cr[0, :, 0, 1], dtype=np.float64)
    z_ds = np.asarray(cr[0, 0, :, 2], dtype=np.float64)
    mmds = float(out_h.Mm_per_ds)
    return {
        "b": b,
        "x_m": x_ds * mmds,
        "y_m": y_ds * mmds,
        "z_m": z_ds * mmds,
        "cartesian_output": out_h,
        "cube_dict": cube,
    }


def evaluate_nf2_on_pino_axes(
    nf2_path: str | Path,
    *,
    x_m: np.ndarray,
    y_m: np.ndarray,
    z_m: np.ndarray,
    device: str | torch.device = "cpu",
    progress: bool = True,
    compute_jacobian: bool = False,
    align_domain: bool = True,
) -> tuple[np.ndarray, CartesianOutput]:
    """Evaluate NF2 **B** (Gauss) on ``meshgrid(x_m, y_m, z_m)`` (megameters).

    **Domain alignment (default):** RTMAG ``get_input`` builds PINO ``x_coords``/``y_coords``
    with a plate-scale whose **Mm span** often disagrees with the NF2 FITS ``coord_range``
    span. Converting PINO Mm with ``/ Mm_per_ds`` then places most query points **outside**
    the MLP training box (for example PINO x_ds ~ 3 vs NF2 ~ 0.77), giving tiny **B** and
    “blank” panels. When ``align_domain`` is true, each axis is **affinely rescaled** to
    NF2’s Mm bounding box before converting to ds (same **relative** cell index maps to
    the same **relative** position in the NF2 volume).

    Set ``align_domain=False`` only if you have verified PINO Mm axes already match NF2.
    """
    from nf2.evaluation.output import CartesianOutput

    path = Path(nf2_path)
    if not path.is_file():
        raise FileNotFoundError(f"NF2 result not found: {path}")

    out_h = CartesianOutput(str(path), device=torch.device(device))
    mmds = float(out_h.Mm_per_ds)
    xm = np.asarray(x_m, dtype=np.float64)
    ym = np.asarray(y_m, dtype=np.float64)
    zm = np.asarray(z_m, dtype=np.float64)
    if align_domain:
        xr, yr = out_h.coord_range[0], out_h.coord_range[1]
        x_lo, x_hi = float(xr[0] * mmds), float(xr[1] * mmds)
        y_lo, y_hi = float(yr[0] * mmds), float(yr[1] * mmds)
        z_lo, z_hi = 0.0, float(out_h.max_height)
        xm = _linear_map_to_range(xm, x_lo, x_hi)
        ym = _linear_map_to_range(ym, y_lo, y_hi)
        zm = _linear_map_to_range(zm, z_lo, z_hi)
    xd = xm / mmds
    yd = ym / mmds
    zd = zm / mmds
    coords = np.stack(np.meshgrid(xd, yd, zd, indexing="ij"), axis=-1).astype(np.float32)
    cube = out_h.load_coords(coords, progress=progress, compute_jacobian=compute_jacobian)
    b = np.asarray(cube["b"], dtype=np.float32)
    return b, out_h


def resample_vector_field_trilinear(
    b: np.ndarray,
    x_src: np.ndarray,
    y_src: np.ndarray,
    z_src: np.ndarray,
    x_dst: np.ndarray,
    y_dst: np.ndarray,
    z_dst: np.ndarray,
    *,
    fill_value: float = 0.0,
) -> np.ndarray:
    """Resample ``b`` (Gauss) from ``(x_src, y_src, z_src)`` onto destination 1-D grids (Mm).

    .. note::
        If **source** and **destination** ranges do not overlap (e.g. PINO Mm mesh vs NF2
        ``load_cube`` Mm mesh from different pixel scales), this returns **zeros** outside
        the source box when ``bounds_error=False``. Prefer :func:`evaluate_nf2_on_pino_axes`
        for PINO vs hybrid comparisons.
    """
    if b.shape[:3] != (len(x_src), len(y_src), len(z_src)):
        raise ValueError(f"b shape {b.shape[:3]} does not match axis lengths")

    pts = np.stack(np.meshgrid(x_dst, y_dst, z_dst, indexing="ij"), axis=-1).reshape(-1, 3)
    out = np.zeros((len(x_dst), len(y_dst), len(z_dst), 3), dtype=np.float32)
    for comp in range(3):
        fn = RegularGridInterpolator(
            (x_src, y_src, z_src),
            b[..., comp],
            bounds_error=False,
            fill_value=fill_value,
        )
        out[..., comp] = fn(pts).reshape(len(x_dst), len(y_dst), len(z_dst)).astype(np.float32)
    return out


def compare_b_on_same_grid(
    b_pino: np.ndarray,
    b_nf2: np.ndarray,
    spacing_m: tuple[float, float, float],
) -> dict[str, Any]:
    """Scalar / vector diagnostics for two fields on the **same** ``[nx,ny,nz,3]`` grid (Mm spacing)."""
    delta = b_nf2 - b_pino
    denom = np.sqrt(np.mean(np.square(b_pino))) + 1e-9
    return {
        "metrics_pino": field_quality_metrics(b_pino, spacing_m),
        "metrics_nf2": field_quality_metrics(b_nf2, spacing_m),
        "delta_l2_abs": float(np.sqrt(np.mean(np.square(delta)))),
        "delta_l2_rel": float(np.sqrt(np.mean(np.square(delta))) / denom),
        "max_abs_delta": float(np.max(np.abs(delta))),
        "mean_abs_delta_per_comp": [float(np.mean(np.abs(delta[..., c]))) for c in range(3)],
    }
