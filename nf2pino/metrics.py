"""Lightweight field metrics for PINO/NF2 comparisons."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


def current_density(b: np.ndarray, spacing: tuple[float, float, float] | None = None) -> np.ndarray:
    """Return `curl(B)` for a Cartesian `[nx, ny, nz, 3]` field."""
    spacing = spacing or (1.0, 1.0, 1.0)
    dfx_dx, dfx_dy, dfx_dz = np.gradient(b[..., 0], *spacing, edge_order=1)
    dfy_dx, dfy_dy, dfy_dz = np.gradient(b[..., 1], *spacing, edge_order=1)
    dfz_dx, dfz_dy, dfz_dz = np.gradient(b[..., 2], *spacing, edge_order=1)
    return np.stack([dfz_dy - dfy_dz, dfx_dz - dfz_dx, dfy_dx - dfx_dy], axis=-1)


def divergence(b: np.ndarray, spacing: tuple[float, float, float] | None = None) -> np.ndarray:
    """Return `div(B)` for a Cartesian `[nx, ny, nz, 3]` field."""
    spacing = spacing or (1.0, 1.0, 1.0)
    dbx_dx = np.gradient(b[..., 0], spacing[0], axis=0, edge_order=1)
    dby_dy = np.gradient(b[..., 1], spacing[1], axis=1, edge_order=1)
    dbz_dz = np.gradient(b[..., 2], spacing[2], axis=2, edge_order=1)
    return dbx_dx + dby_dy + dbz_dz


def field_quality_metrics(b: np.ndarray, spacing: tuple[float, float, float] | None = None) -> dict[str, float]:
    """Compute finite, divergence, force-free, and energy summaries."""
    b = np.asarray(b, dtype=np.float32)
    j = current_density(b, spacing)
    div = divergence(b, spacing)
    b_norm = np.linalg.norm(b, axis=-1)
    j_norm = np.linalg.norm(j, axis=-1)
    jxb_norm = np.linalg.norm(np.cross(j, b, axis=-1), axis=-1)
    sigma = jxb_norm / (b_norm * j_norm + 1e-7)
    weighted_sigma = float(np.nansum(sigma * j_norm) / (np.nansum(j_norm) + 1e-7))
    return {
        "finite": bool(np.isfinite(b).all()),
        "mean_abs_divergence": float(np.nanmean(np.abs(div))),
        "mean_normalized_divergence": float(np.nanmean(np.abs(div) / (b_norm + 1e-7))),
        "sigma_j_percent": weighted_sigma * 100.0,
        "theta_j_deg": float(np.rad2deg(np.arcsin(np.clip(weighted_sigma, -1.0, 1.0)))),
        "mean_energy_density": float(np.nanmean(np.sum(b**2, axis=-1) / (8 * np.pi))),
        "max_abs_b": float(np.nanmax(np.abs(b))),
    }


def write_metrics(path: str | Path, metrics: dict[str, Any]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metrics, indent=2))
    return path
