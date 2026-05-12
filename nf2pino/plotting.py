"""Notebook-friendly plotting helpers for hybrid PINO/NF2 runs."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np

from .metrics import current_density


def plot_field_slices(
    fields: dict[str, np.ndarray],
    *,
    z_index: int | None = None,
    component: int = 2,
    output_path: str | Path | None = None,
):
    """Plot a component slice for one or more fields.

    Supported shapes:
    - Vector field: `[nx, ny, nz, 3]` or `[3, nx, ny, nz]`
    - Scalar volume: `[nx, ny, nz]` (component ignored)
    """
    n = len(fields)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4), squeeze=False)
    for ax, (name, field) in zip(axes[0], fields.items()):
        data2d, z = _slice_field_for_plot(field, z_index=z_index, component=component)
        im = ax.imshow(data2d.T, origin="lower", cmap="RdBu_r")
        if z is None:
            ax.set_title(f"{name}")
        elif field.ndim == 3:
            ax.set_title(f"{name}: z={z}")
        else:
            ax.set_title(f"{name}: B{component} z={z}")
        fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()
    return _save_or_return(fig, output_path)


def plot_current_density_slices(
    fields: dict[str, np.ndarray],
    *,
    z_index: int | None = None,
    output_path: str | Path | None = None,
):
    """Plot `|curl(B)|` slices for one or more fields."""
    currents = {name: np.linalg.norm(current_density(field), axis=-1) for name, field in fields.items()}
    n = len(currents)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4), squeeze=False)
    for ax, (name, jnorm) in zip(axes[0], currents.items()):
        z = jnorm.shape[2] // 2 if z_index is None else z_index
        im = ax.imshow(jnorm[:, :, z].T, origin="lower", cmap="magma")
        ax.set_title(f"{name}: |J| z={z}")
        fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()
    return _save_or_return(fig, output_path)


def plot_convergence(
    histories: dict[str, Iterable[dict[str, float]]],
    *,
    output_path: str | Path | None = None,
):
    """Plot prefit or benchmark histories with heterogeneous loss schemas."""
    fig, ax = plt.subplots(figsize=(6, 4))
    for name, history in histories.items():
        rows = list(history)
        if not rows:
            continue
        key = _select_history_metric(rows)
        if key is None:
            continue
        steps = [row.get("step", row.get("global_step", i + 1)) for i, row in enumerate(rows)]
        values = [float(row.get(key, np.nan)) for row in rows]
        if not np.isfinite(values).any():
            continue
        label = name if key in {"loss", "total_loss", "mse"} else f"{name} ({key})"
        ax.plot(steps, values, label=label)
    ax.set_xlabel("step")
    ax.set_ylabel("loss")
    ax.set_yscale("log")
    if ax.lines:
        ax.legend()
    fig.tight_layout()
    return _save_or_return(fig, output_path)


def _select_history_metric(rows: list[dict[str, float]]) -> str | None:
    preferred_keys = [
        "loss",
        "total_loss",
        "mse",
        "boundary_loss",
        "force_free_loss",
        "divergence_loss",
    ]
    for key in preferred_keys:
        if any(key in row for row in rows):
            return key
    reserved = {"step", "epoch", "global_step", "learning_rate", "validation_boundary_diff_G"}
    for row in rows:
        for key, value in row.items():
            if key not in reserved and isinstance(value, (int, float, np.floating)):
                return key
    return None


def _save_or_return(fig, output_path):
    if output_path is None:
        return fig
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def _slice_field_for_plot(
    field: np.ndarray,
    *,
    z_index: int | None,
    component: int,
) -> tuple[np.ndarray, int | None]:
    """Return (data2d, z) suitable for imshow from a scalar or vector field."""
    arr = np.asarray(field)

    if arr.ndim == 2:
        return arr, None

    if arr.ndim == 3:
        z = arr.shape[2] // 2 if z_index is None else z_index
        return arr[:, :, z], z

    if arr.ndim == 4:
        if arr.shape[-1] == 3:
            z = arr.shape[2] // 2 if z_index is None else z_index
            return arr[:, :, z, component], z
        if arr.shape[0] == 3:
            z = arr.shape[3] // 2 if z_index is None else z_index
            return arr[component, :, :, z], z

    raise ValueError(
        "Unsupported field shape for plotting. Expected [nx, ny, nz, 3], "
        "[3, nx, ny, nz], or scalar [nx, ny, nz]. "
        f"Got shape={arr.shape} ndim={arr.ndim}."
    )
