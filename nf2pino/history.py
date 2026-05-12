"""Training-history capture for hybrid NF2 runs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from pytorch_lightning import Callback


class MetricsHistoryCallback(Callback):
    """Persist sampled convergence metrics at validation boundaries.

    NF2 already logs training losses and learning rate. This callback snapshots
    the latest scalar metrics whenever validation finishes, and appends a small
    JSON history that the hybrid notebook can load directly.
    """

    def __init__(self, output_path: str | Path):
        self.output_path = Path(output_path)
        self.history: list[dict[str, Any]] = []

    def on_validation_epoch_end(self, trainer, pl_module) -> None:
        row: dict[str, Any] = {
            "epoch": int(trainer.current_epoch),
            "global_step": int(trainer.global_step),
        }
        metrics = trainer.callback_metrics

        total_loss = _metric_to_float(metrics.get("train/loss"))
        if total_loss is not None:
            row["total_loss"] = total_loss

        boundary_loss = _find_metric(metrics, "boundary")
        if boundary_loss is not None:
            row["boundary_loss"] = boundary_loss

        force_free_loss = _find_metric(metrics, "force_free")
        if force_free_loss is not None:
            row["force_free_loss"] = force_free_loss

        divergence_loss = _find_metric(metrics, "divergence")
        if divergence_loss is not None:
            row["divergence_loss"] = divergence_loss

        learning_rate = _metric_to_float(metrics.get("Learning Rate"))
        if learning_rate is not None:
            row["learning_rate"] = learning_rate

        boundary_diff = _validation_boundary_diff(trainer, pl_module)
        if boundary_diff is not None:
            row["validation_boundary_diff_G"] = boundary_diff

        self.history.append(row)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.output_path.write_text(json.dumps(self.history, indent=2))


def _find_metric(metrics: dict[str, Any], token: str) -> float | None:
    for key, value in metrics.items():
        if isinstance(key, str) and key.startswith("train/") and token in key:
            return _metric_to_float(value)
    return None


def _metric_to_float(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        if value.numel() != 1:
            return None
        return float(value.detach().cpu().item())
    if isinstance(value, (float, int)):
        return float(value)
    return None


def _validation_boundary_diff(trainer, pl_module) -> float | None:
    outputs = getattr(pl_module, "validation_outputs", None) or {}
    datamodule = getattr(trainer, "datamodule", None)
    g_per_db = float(getattr(datamodule, "config", {}).get("G_per_dB", 1.0))

    for name, state in outputs.items():
        if not isinstance(name, str) or not name.startswith("validation_boundary_"):
            continue
        if "b" not in state or "b_true" not in state:
            continue
        b = state["b"]
        b_true = state["b_true"]
        if "transform" in state:
            b = torch.einsum("ijk,ik->ij", state["transform"], b)
        diff = torch.abs(b - b_true)
        return float(torch.nanmean(diff.pow(2).sum(-1).pow(0.5)).detach().cpu() * g_per_db)
    return None
